"""
Subtask 1 - Binary Insomnia Classification
Ensemble: Qwen (unsloth/Qwen3-4B-Instruct-2507) + Bio_ClinicalBERT
Span identification is handled in subtask2_span.py separately.

Train on ./train/  +  ./val/  (combined for final model after HP tuning on val)
Test  on ./test/



Outputs:
  test_predictions_subtask1_ensemble.json
"""

import os, json, csv, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModel, get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
import warnings; warnings.filterwarnings("ignore")

# -- Config --------------------------------------------------------------------
QWEN_NAME   = "unsloth/Qwen3-4B-Instruct-2507"                 ###### you change your specific LLM model 
BERT_NAME   = "emilyalsentzer/Bio_ClinicalBERT"                ###### you change your specific model, like baseline model used by organizers
MAX_LEN_Q   = 512       # Qwen window (tokens)
MAX_LEN_B   = 512      # ModernBERT window (tokens)
STRIDE_TOK  = 128       # token stride for chunk-based inference (ModernBERT)
BATCH_SIZE  = 4
GRAD_ACCUM  = 4
LR_QWEN     = 1e-5
LR_BERT     = 2e-5
EPOCHS      = 6
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL2ID    = {"no": 0, "yes": 1}
ID2LABEL    = {0: "no", 1: "yes"}
SAVE_QWEN   = "./ckpt_s1_qwen.pt"
SAVE_BERT   = "./ckpt_s1_bert.pt"

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# -- AMP helpers --------------------------------------------------------------
_USE_BF16   = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
_AMP_DTYPE  = torch.bfloat16 if _USE_BF16 else torch.float16
_USE_SCALER = DEVICE.type == "cuda" and not _USE_BF16


def amp_context():
    if DEVICE.type != "cuda":
        return torch.amp.autocast(device_type="cpu", enabled=False)
    return torch.amp.autocast(device_type="cuda", dtype=_AMP_DTYPE)


def make_scaler():
    return torch.cuda.amp.GradScaler() if _USE_SCALER else None


def do_backward(scaler, loss):
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def do_step(scaler, opt, model):
    if scaler is not None:
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

# -- I/O -----------------------------------------------------------------------
def load_data(corpus_csv, labels_json):
    with open(corpus_csv, newline="", encoding="utf-8") as f:
        corpus = {r["note_id"]: r["text"] for r in csv.DictReader(f)}
    with open(labels_json, encoding="utf-8") as f:
        labels = json.load(f)
    return [(nid, corpus[nid], LABEL2ID[d["Insomnia"]])
            for nid, d in labels.items() if nid in corpus]

def load_test(corpus_csv):
    with open(corpus_csv, newline="", encoding="utf-8") as f:
        return [(r["note_id"], r["text"]) for r in csv.DictReader(f)]

# -- Dataset (chunk-aware) -----------------------------------------------------
class NoteDataset(Dataset):
    """
    Each note is split into overlapping token-level chunks.
    Label of each chunk = note-level label (propagated).
    At inference, note prediction = sigmoid-max across chunks >= 0.5.
    """
    def __init__(self, samples, tokenizer, max_len, stride, is_test=False):
        self.items = []
        for s in samples:
            nid, text = (s[0], s[1]) if is_test else (s[0], s[1])
            label = -1 if is_test else s[2]
            enc = tokenizer(
                text,
                max_length=max_len, stride=stride,
                truncation=True, padding="max_length",
                return_overflowing_tokens=True, return_tensors="pt"
            )
            for i in range(enc["input_ids"].shape[0]):
                item = {
                    "input_ids":      enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "label":          torch.tensor(label, dtype=torch.long),
                    "note_id":        nid,
                }
                if "token_type_ids" in enc:
                    item["token_type_ids"] = enc["token_type_ids"][i]
                self.items.append(item)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

# -- Models --------------------------------------------------------------------
def build_classifier(model_name, hidden_size):
    """Shared encoder + 2-class head."""
    class _Clf(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder  = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.drop     = nn.Dropout(0.1)
            self.head     = nn.Linear(hidden_size, 2)
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kw = dict(input_ids=input_ids, attention_mask=attention_mask)
            if token_type_ids is not None:
                kw["token_type_ids"] = token_type_ids
            out    = self.encoder(**kw)
            hidden = out.last_hidden_state          # [B, L, H]
            m      = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
            return self.head(self.drop(pooled))     # [B, 2]
    return _Clf().to(DEVICE)

# -- Training helpers ----------------------------------------------------------
def train_epoch(model, loader, opt, sched, scaler, criterion):
    model.train(); total = 0; opt.zero_grad()
    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl  = batch["label"].to(DEVICE)
        kw = dict(input_ids=ids, attention_mask=mask)
        if "token_type_ids" in batch:
            kw["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
        with amp_context():
            logits = model(**kw)
            loss   = criterion(logits, lbl) / GRAD_ACCUM
        do_backward(scaler, loss)
        total += loss.item() * GRAD_ACCUM
        if (step+1) % GRAD_ACCUM == 0 or (step+1) == len(loader):
            do_step(scaler, opt, model)
            sched.step(); opt.zero_grad()
    return total / len(loader)

@torch.no_grad()
def get_note_probs(model, loader):
    """Returns dict note_id -> max prob[class=1] across chunks."""
    model.eval(); note_probs = {}
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        kw   = dict(input_ids=ids, attention_mask=mask)
        if "token_type_ids" in batch:
            kw["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
        probs = torch.softmax(model(**kw), -1)[:, 1].cpu().tolist()
        for nid, p in zip(batch["note_id"], probs):
            note_probs[nid] = max(note_probs.get(nid, 0.0), p)
    return note_probs

def evaluate_probs(note_probs, samples, threshold=0.5):
    label_map = {s[0]: s[2] for s in samples}
    ids = [nid for nid in note_probs if nid in label_map]
    gold  = [label_map[nid] for nid in ids]
    preds = [1 if note_probs[nid] >= threshold else 0 for nid in ids]
    return f1_score(gold, preds, pos_label=1, average="binary", zero_division=0), gold, preds

def make_loader(samples, tok, max_len, stride, shuffle=False, is_test=False):
    ds = NoteDataset(samples, tok, max_len, stride, is_test)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=2, pin_memory=True)

def make_optimizer_scheduler(model, lr, n_steps):
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, n_steps//10, n_steps)
    return opt, sched

# -- Train one model -----------------------------------------------------------
def train_model(model_name, hidden, lr, max_len, stride,
                tr_samples, vl_samples, tag):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    tr_dl = make_loader(tr_samples, tok, max_len, stride, shuffle=True)
    vl_dl = make_loader(vl_samples, tok, max_len, stride)

    model     = build_classifier(model_name, hidden)
    criterion = nn.CrossEntropyLoss()

    # Compute class weights for imbalance
    labels = [s[2] for s in tr_samples]
    n_pos  = sum(labels); n_neg = len(labels) - n_pos
    w      = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)

    n_steps   = (len(tr_dl) // GRAD_ACCUM + 1) * EPOCHS
    opt, sched = make_optimizer_scheduler(model, lr, n_steps)
    scaler     = make_scaler()

    best_f1, best_state = 0.0, None
    for ep in range(EPOCHS):
        loss = train_epoch(model, tr_dl, opt, sched, scaler, criterion)
        probs = get_note_probs(model, vl_dl)
        f1, _, _ = evaluate_probs(probs, vl_samples)
        print(f"  [{tag}] Ep {ep+1}/{EPOCHS} loss={loss:.4f} val-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1  = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"     best={best_f1:.4f}")

    print(f"  [{tag}] Best val-F1 = {best_f1:.4f}")
    return model, best_state, tok

# -- Ensemble inference --------------------------------------------------------
def ensemble_predict(models_toks, test_samples, alpha=0.5):
    """
    alpha = weight for first model (Qwen). (1-alpha) for ModernBERT.
    Returns dict note_id -> label string.
    """
    all_probs = []
    for (model, tok, max_len, stride) in models_toks:
        dl    = make_loader(test_samples, tok, max_len, stride, is_test=True)
        probs = get_note_probs(model, dl)
        all_probs.append(probs)
    nids = list(all_probs[0].keys())
    results = {}
    for nid in nids:
        p = alpha * all_probs[0].get(nid, 0.5) + (1 - alpha) * all_probs[1].get(nid, 0.5)
        results[nid] = ID2LABEL[1 if p >= 0.5 else 0]
    return results

def ensemble_val_f1(models_toks, vl_samples, alpha=0.5):
    all_probs = []
    for (model, tok, max_len, stride) in models_toks:
        dl    = make_loader(vl_samples, tok, max_len, stride)
        probs = get_note_probs(model, dl)
        all_probs.append(probs)
    nids      = list(all_probs[0].keys())
    label_map = {s[0]: s[2] for s in vl_samples}
    gold, preds = [], []
    for nid in nids:
        if nid not in label_map: continue
        p = alpha * all_probs[0].get(nid, 0.5) + (1-alpha) * all_probs[1].get(nid, 0.5)
        preds.append(1 if p >= 0.5 else 0)
        gold.append(label_map[nid])
    f1 = f1_score(gold, preds, pos_label=1, average="binary", zero_division=0)
    print(f"\nEnsemble val-F1 (α={alpha:.2f}): {f1:.4f}")
    print(classification_report(gold, preds, target_names=["no","yes"]))
    return f1

# -- Main ----------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")

    tr_samples = load_data("./train/sample_corpus.csv", "./train/subtask_1.json")
    vl_samples = load_data("./val/sample_corpus.csv",   "./val/subtask_1.json")
    te_samples = load_test("./test/sample_corpus.csv")
    all_train  = tr_samples + vl_samples

    print(f"Train={len(tr_samples)} Val={len(vl_samples)} Test={len(te_samples)}")

    # -- Phase 1: train on train-only, measure val F1 -------------------------
    print("\n=== Training Qwen ===")
    q_tok  = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True)
    if q_tok.pad_token is None: q_tok.pad_token = q_tok.eos_token
    q_hidden = AutoModel.from_pretrained(QWEN_NAME, trust_remote_code=True).config.hidden_size
    qwen_model, qwen_state, q_tok = train_model(
        QWEN_NAME, q_hidden, LR_QWEN, MAX_LEN_Q, MAX_LEN_Q // 2,
        tr_samples, vl_samples, "Qwen"
    )
    qwen_model.load_state_dict(qwen_state)
    torch.save(qwen_state, SAVE_QWEN)

    print("\n=== Training ModernBERT ===")
    b_hidden = AutoModel.from_pretrained(BERT_NAME).config.hidden_size
    bert_model, bert_state, b_tok = train_model(
        BERT_NAME, b_hidden, LR_BERT, MAX_LEN_B, STRIDE_TOK,
        tr_samples, vl_samples, "ModernBERT"
    )
    bert_model.load_state_dict(bert_state)
    torch.save(bert_state, SAVE_BERT)

    # -- Phase 2: tune ensemble alpha on val -----------------------------------
    print("\n=== Tuning ensemble alpha ===")
    best_alpha, best_ens_f1 = 0.5, 0.0
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        f1 = ensemble_val_f1(
            [(qwen_model, q_tok, MAX_LEN_Q, MAX_LEN_Q//2),
             (bert_model, b_tok, MAX_LEN_B, STRIDE_TOK)],
            vl_samples, alpha
        )
        if f1 > best_ens_f1:
            best_ens_f1 = f1; best_alpha = alpha
    print(f"Best alpha={best_alpha:.2f}  Ensemble val-F1={best_ens_f1:.4f}")

    # -- Phase 3: retrain both on combined data --------------------------------
    print("\n=== Retraining Qwen on combined data ===")
    qwen2, _, q_tok = train_model(
        QWEN_NAME, q_hidden, LR_QWEN, MAX_LEN_Q, MAX_LEN_Q//2,
        all_train, vl_samples, "Qwen-combined"
    )

    print("\n=== Retraining ModernBERT on combined data ===")
    bert2, _, b_tok = train_model(
        BERT_NAME, b_hidden, LR_BERT, MAX_LEN_B, STRIDE_TOK,
        all_train, vl_samples, "BERT-combined"
    )

    # -- Phase 4: test inference -----------------------------------------------
    print("\n=== Test inference ===")
    results = ensemble_predict(
        [(qwen2, q_tok, MAX_LEN_Q, MAX_LEN_Q//2),
         (bert2, b_tok, MAX_LEN_B, STRIDE_TOK)],
        te_samples, best_alpha
    )
    out = {nid: {"Insomnia": lbl} for nid, lbl in results.items()}
    with open("test_predictions_subtask1_ensemble_fixed.json", "w") as f:
        json.dump(out, f, indent=2)
    yes = sum(1 for v in results.values() if v == "yes")
    print(f"Saved -> test_predictions_subtask1_ensemble.json  (yes={yes}, no={len(results)-yes})")

if __name__ == "__main__":
    main()
