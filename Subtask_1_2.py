import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, set_seed
from sentence_transformers import SentenceTransformer
import spacy
from tqdm import tqdm
import warnings
import itertools
from collections import Counter
import copy

warnings.filterwarnings("ignore")

# -------------------------------
# Configuration
# -------------------------------
set_seed(42)
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"    ###### you can change this model to your model or baseline model used by the organizers
MAX_CHUNK_LEN = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 2
GRAD_ACCUM = 2
EPOCHS = 20
LR = 2e-5
TOP_K_SENTENCES = 30
SENTENCE_WINDOW = 1
TRAIN_DIR = Path("./train")
VAL_DIR = Path("./val")
TEST_DIR = Path("./test")

RULE_NAMES = ["Definition 1", "Definition 2", "Rule B", "Rule C"]

# BIO labels
label2id = {"O": 0}
id2label = {0: "O"}
for rule in RULE_NAMES:
    label2id[f"B-{rule}"] = len(label2id)
    id2label[len(id2label)] = f"B-{rule}"
    label2id[f"I-{rule}"] = len(label2id)
    id2label[len(id2label)] = f"I-{rule}"
NUM_LABELS = len(label2id)

# Sentence filtering
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
try:
    nlp = spacy.load("en_core_sci_sm")
except:
    nlp = spacy.load("en_core_web_sm")

RELEVANCE_PROTOTYPE = (
    "difficulty falling asleep trouble staying asleep waking early insomnia "
    "fatigue daytime sleepiness tired low energy mood irritable concentration "
    "zolpidem ambien eszopiclone lunesta temazepam restoril triazolam halcion "
    "trazodone quetiapine seroquel lorazepam ativan diazepam valium diphenhydramine benadryl"
)

def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def filter_relevant_sentences(text, top_k=TOP_K_SENTENCES, window=SENTENCE_WINDOW):
    sentences = split_sentences(text)
    if len(sentences) <= top_k * 3:
        return text
    sent_emb = sentence_model.encode(sentences, convert_to_tensor=True)
    proto_emb = sentence_model.encode(RELEVANCE_PROTOTYPE, convert_to_tensor=True)
    scores = torch.cosine_similarity(sent_emb, proto_emb.unsqueeze(0), dim=1)
    top_indices = torch.argsort(scores, descending=True)[:top_k].tolist()
    selected = set()
    for idx in top_indices:
        for w in range(-window, window+1):
            nb = idx + w
            if 0 <= nb < len(sentences):
                selected.add(nb)
    kept = [sentences[i] for i in sorted(selected)]
    return " ".join(kept)

# -------------------------------
# Data loading
# -------------------------------
def load_train_val(data_dir):
    corpus = pd.read_csv(data_dir / "sample_corpus.csv")
    with open(data_dir / "subtask_1.json") as f:
        sub1 = json.load(f)
    with open(data_dir / "subtask_2.json") as f:
        sub2 = json.load(f)
    records = []
    for _, row in corpus.iterrows():
        nid = str(row["note_id"])
        text = row["text"]
        filtered = filter_relevant_sentences(text)
        records.append({
            "note_id": nid,
            "text": filtered,
            "insomnia": 1 if sub1[nid]["Insomnia"] == "yes" else 0,
            "spans": sub2[nid]
        })
    return records

def load_test(data_dir):
    corpus = pd.read_csv(data_dir / "sample_corpus.csv")
    records = []
    for _, row in corpus.iterrows():
        nid = str(row["note_id"])
        text = row["text"]
        filtered = filter_relevant_sentences(text)
        records.append({
            "note_id": nid,
            "text": filtered,
        })
    return records

# -------------------------------
# Tokenizer and chunking
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def create_chunks(text, max_len=MAX_CHUNK_LEN, overlap=CHUNK_OVERLAP):
    tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]
    chunks = []
    stride = max_len - overlap
    for start in range(0, len(ids), stride):
        end = min(start + max_len, len(ids))
        chunk_ids = ids[start:end]
        chunk_offsets = offsets[start:end]
        chunks.append({
            "input_ids": chunk_ids,
            "offset_mapping": chunk_offsets,
            "global_start": chunk_offsets[0][0] if chunk_offsets else 0,
            "global_end": chunk_offsets[-1][1] if chunk_offsets else 0
        })
    return chunks

def chunk_labels(chunk, gold_spans):
    labels = [0] * len(chunk["input_ids"])
    offsets = chunk["offset_mapping"]
    token_rules = [[] for _ in range(len(offsets))]
    for rule_idx, rule in enumerate(RULE_NAMES):
        rule_data = gold_spans[rule]
        if rule_data["label"] != "yes":
            continue
        for span_str in rule_data["span"]:
            for part in span_str.split(";"):
                s_char, e_char = map(int, part.split())
                for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_end == 0:
                        continue
                    if max(tok_start, s_char) < min(tok_end, e_char):
                        token_rules[tok_idx].append(rule_idx)
    # Assign BIO
    for rule_idx in range(len(RULE_NAMES)):
        in_span = False
        start_tok = None
        for i in range(len(token_rules)):
            if rule_idx in token_rules[i]:
                if not in_span:
                    in_span = True
                    start_tok = i
            else:
                if in_span:
                    labels[start_tok] = label2id[f"B-{RULE_NAMES[rule_idx]}"]
                    for j in range(start_tok+1, i):
                        labels[j] = label2id[f"I-{RULE_NAMES[rule_idx]}"]
                    in_span = False
        if in_span:
            labels[start_tok] = label2id[f"B-{RULE_NAMES[rule_idx]}"]
            for j in range(start_tok+1, len(labels)):
                labels[j] = label2id[f"I-{RULE_NAMES[rule_idx]}"]
    return labels

def build_chunks_from_records(records):
    chunks = []
    for rec in records:
        chunks_doc = create_chunks(rec["text"])
        for ch in chunks_doc:
            lbls = chunk_labels(ch, rec["spans"])
            chunks.append({
                "input_ids": ch["input_ids"],
                "attention_mask": [1] * len(ch["input_ids"]),
                "token_labels": lbls,
                "insomnia_label": rec["insomnia"]
            })
    return chunks

# -------------------------------
# Dataset and collate
# -------------------------------
class ChunkDataset(TorchDataset):
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        return self.chunks[idx]

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    token_labels = [item["token_labels"] for item in batch]
    insomnia_labels = [item["insomnia_label"] for item in batch]
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = []
    padded_attention_mask = []
    padded_token_labels = []
    for ids, mask, lbls in zip(input_ids, attention_mask, token_labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [0] * pad_len)
        padded_attention_mask.append(mask + [0] * pad_len)
        padded_token_labels.append(lbls + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(padded_attention_mask),
        "token_labels": torch.tensor(padded_token_labels),
        "insomnia_labels": torch.tensor(insomnia_labels)
    }

# -------------------------------
# Model: base encoder + two heads
# -------------------------------
class DualHeadModel(nn.Module):
    def __init__(self, model_name, num_token_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.base_model.config.hidden_size
        self.token_head = nn.Linear(hidden_size, num_token_labels)
        self.seq_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        tok_logits = self.token_head(sequence_output)
        cls_output = sequence_output[:, 0, :]
        seq_logits = self.seq_head(cls_output)
        return tok_logits, seq_logits

# -------------------------------
# Training function with early stopping
# -------------------------------
def train_model(train_loader, val_loader, lr, epochs, batch_size, grad_accum, patience=3, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadModel(MODEL_NAME, NUM_LABELS).to(device)
    
    # Class weights for insomnia
    y_train = [c["insomnia_label"] for c in train_loader.dataset.chunks]
    if len(y_train) == 0:
        print("Warning: Training set is empty. Returning untrained model.")
        return model, 0.0
    class_weights = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
    insomnia_weight = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs // grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    seq_loss_fn = nn.CrossEntropyLoss(weight=insomnia_weight)
    
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        global_step = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_labels = batch["token_labels"].to(device)
            insomnia_labels = batch["insomnia_labels"].to(device)
            
            tok_logits, seq_logits = model(input_ids, attention_mask)
            loss_token = token_loss_fn(tok_logits.view(-1, NUM_LABELS), token_labels.view(-1))
            loss_seq = seq_loss_fn(seq_logits.view(-1, 2), insomnia_labels.view(-1))
            loss = loss_token + loss_seq
            loss = loss / grad_accum
            loss.backward()
            epoch_loss += loss.item() * grad_accum
            
            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            global_step += 1
            progress_bar.set_postfix(loss=loss.item() * grad_accum)
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished, avg loss: {avg_loss:.4f}")
        
        # Validation (skip if val_loader is empty)
        val_f1 = 0.0
        if len(val_loader) > 0:
            val_metrics = evaluate_model_chunk_level(model, val_loader, device)
            val_f1 = val_metrics['insomnia_f1']
            print(f"Validation Insomnia F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_val_f1

def evaluate_model_chunk_level(model, loader, device):
    """Chunk‑level insomnia F1 (used for early stopping)."""
    model.eval()
    y_true_ins, y_pred_ins = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        insomnia_labels = batch["insomnia_labels"].cpu().numpy()
        with torch.no_grad():
            _, seq_logits = model(input_ids, attention_mask)
        preds = torch.argmax(seq_logits, dim=-1).cpu().numpy()
        y_true_ins.extend(insomnia_labels)
        y_pred_ins.extend(preds)
    if len(y_true_ins) == 0:
        return {"insomnia_f1": 0.0}
    ins_f1 = f1_score(y_true_ins, y_pred_ins, average="binary")
    return {"insomnia_f1": ins_f1}

# -------------------------------
# Document‑level prediction with ensemble (FIXED)
# -------------------------------
def predict_document_ensemble(rec, models, device):
    """Predict insomnia and spans using ensemble voting."""
    chunks = create_chunks(rec["text"])
    if not chunks:
        return 0, {rule: [] for rule in RULE_NAMES}
    
    all_chunk_probs = []      # [model][chunk] → probability of insomnia class 1
    all_chunk_spans_per_model = []  # [model][chunk] → spans dict
    
    for model in models:
        model.eval()
        chunk_ins_probs = []
        chunk_spans_list = []
        for ch in chunks:
            ids = torch.tensor([ch["input_ids"]]).to(device)
            mask = torch.tensor([[1]*len(ch["input_ids"])]).to(device)
            with torch.no_grad():
                tok_logits, seq_logits = model(ids, mask)
            probs = torch.softmax(seq_logits, dim=-1).squeeze().cpu().numpy()
            # handle scalar probs
            if probs.ndim == 0:
                probs = np.array([probs, 1-probs]) if probs < 1 else np.array([0,1])
            chunk_ins_probs.append(probs[1])
            
            # Token predictions: ensure 1D array
            tok_preds = torch.argmax(tok_logits, dim=-1).squeeze().cpu().numpy()
            if tok_preds.ndim == 0:
                tok_preds = np.array([tok_preds])
            offsets = ch["offset_mapping"]
            spans = {rule: [] for rule in RULE_NAMES}
            for rule_idx, rule in enumerate(RULE_NAMES):
                b_id = label2id[f"B-{rule}"]
                i_id = label2id[f"I-{rule}"]
                in_span = False
                start_char = None
                for t, (ts, te) in enumerate(offsets):
                    if te == 0:
                        continue
                    if t >= len(tok_preds):
                        break
                    lab = tok_preds[t]
                    if lab == b_id:
                        if in_span:
                            spans[rule].append((start_char, offsets[t-1][1]))
                        in_span = True
                        start_char = ts
                    elif lab == i_id:
                        if not in_span:
                            in_span = True
                            start_char = ts
                    else:
                        if in_span:
                            spans[rule].append((start_char, offsets[t-1][1]))
                            in_span = False
                if in_span:
                    last_end = offsets[-1][1] if offsets and offsets[-1][1] != 0 else len(rec['text'])
                    spans[rule].append((start_char, last_end))
            chunk_spans_list.append(spans)
        all_chunk_probs.append(chunk_ins_probs)
        all_chunk_spans_per_model.append(chunk_spans_list)
    
    # Aggregate insomnia: average probabilities across models, then majority over chunks
    num_chunks = len(chunks)
    if num_chunks == 0:
        return 0, {rule: [] for rule in RULE_NAMES}
    avg_probs_per_chunk = [np.mean([all_chunk_probs[m][i] for m in range(len(models))]) for i in range(num_chunks)]
    doc_insomnia = 1 if np.mean(avg_probs_per_chunk) > 0.5 else 0
    
    # Aggregate spans: majority voting across models (threshold = half of models + 1)
    final_spans = {rule: [] for rule in RULE_NAMES}
    for rule in RULE_NAMES:
        for chunk_idx in range(num_chunks):
            span_counter = Counter()
            for m in range(len(models)):
                for span in all_chunk_spans_per_model[m][chunk_idx][rule]:
                    span_counter[span] += 1
            threshold = len(models) // 2 + 1
            for span, count in span_counter.items():
                if count >= threshold:
                    final_spans[rule].append(span)
        # Deduplicate and merge overlapping spans
        if final_spans[rule]:
            final_spans[rule] = list(set(final_spans[rule]))
            final_spans[rule].sort()
            merged = []
            cur_start, cur_end = final_spans[rule][0]
            for s, e in final_spans[rule][1:]:
                if s <= cur_end:
                    cur_end = max(cur_end, e)
                else:
                    merged.append((cur_start, cur_end))
                    cur_start, cur_end = s, e
            merged.append((cur_start, cur_end))
            final_spans[rule] = merged
    return doc_insomnia, final_spans

def evaluate_document_level_ensemble(models, records, device):
    """Full document‑level evaluation on validation records."""
    if len(records) == 0:
        print("Warning: No validation records, returning zero metrics.")
        return {k: 0.0 for k in ["insomnia_precision","insomnia_recall","insomnia_f1",
                                 "label_precision","label_recall","label_f1",
                                 "span_exact_precision","span_exact_recall","span_exact_f1",
                                 "span_partial_precision","span_partial_recall","span_partial_f1"]}
    
    y_true_ins, y_pred_ins = [], []
    all_gold_spans, all_pred_spans = [], []
    for rec in tqdm(records, desc="Evaluating validation"):
        gold_ins = rec["insomnia"]
        pred_ins, pred_spans = predict_document_ensemble(rec, models, device)
        y_true_ins.append(gold_ins)
        y_pred_ins.append(pred_ins)
        
        gold_spans_dict = {}
        for rule in RULE_NAMES:
            gold = rec["spans"][rule]
            if gold["label"] == "yes":
                spans = []
                for s in gold["span"]:
                    for part in s.split(";"):
                        spans.append(tuple(map(int, part.split())))
                gold_spans_dict[rule] = spans
            else:
                gold_spans_dict[rule] = []
        all_gold_spans.append(gold_spans_dict)
        all_pred_spans.append(pred_spans)
    
    # Insomnia metrics
    ins_prec = precision_score(y_true_ins, y_pred_ins, average="binary", zero_division=0)
    ins_rec = recall_score(y_true_ins, y_pred_ins, average="binary", zero_division=0)
    ins_f1 = f1_score(y_true_ins, y_pred_ins, average="binary")
    
    # Label-level (any span presence)
    y_true_label = []
    y_pred_label = []
    for gold, pred in zip(all_gold_spans, all_pred_spans):
        for rule in RULE_NAMES:
            y_true_label.append(1 if gold[rule] else 0)
            y_pred_label.append(1 if pred[rule] else 0)
    label_prec = precision_score(y_true_label, y_pred_label, average="micro", zero_division=0)
    label_rec = recall_score(y_true_label, y_pred_label, average="micro", zero_division=0)
    label_f1 = f1_score(y_true_label, y_pred_label, average="micro")
    
    # Span exact match
    tp_exact = fp_exact = fn_exact = 0
    for gold, pred in zip(all_gold_spans, all_pred_spans):
        for rule in RULE_NAMES:
            gold_set = set([f"{s} {e}" for s,e in gold[rule]])
            pred_set = set([f"{s} {e}" for s,e in pred[rule]])
            tp_exact += len(gold_set & pred_set)
            fp_exact += len(pred_set - gold_set)
            fn_exact += len(gold_set - pred_set)
    prec_exact = tp_exact / (tp_exact + fp_exact + 1e-8)
    rec_exact = tp_exact / (tp_exact + fn_exact + 1e-8)
    exact_f1 = 2 * prec_exact * rec_exact / (prec_exact + rec_exact + 1e-8)
    
    # Span partial match (overlap)
    total_correct = 0
    total_gold = 0
    for gold, pred in zip(all_gold_spans, all_pred_spans):
        for rule in RULE_NAMES:
            for (gs, ge) in gold[rule]:
                total_gold += 1
                for (ps, pe) in pred[rule]:
                    if max(gs, ps) < min(ge, pe):
                        total_correct += 1
                        break
    partial_rec = total_correct / (total_gold + 1e-8)
    total_pred = sum(len(pred[rule]) for pred in all_pred_spans for rule in RULE_NAMES)
    partial_prec = total_correct / (total_pred + 1e-8)
    partial_f1 = 2 * partial_prec * partial_rec / (partial_prec + partial_rec + 1e-8)
    
    return {
        "insomnia_precision": ins_prec, "insomnia_recall": ins_rec, "insomnia_f1": ins_f1,
        "label_precision": label_prec, "label_recall": label_rec, "label_f1": label_f1,
        "span_exact_precision": prec_exact, "span_exact_recall": rec_exact, "span_exact_f1": exact_f1,
        "span_partial_precision": partial_prec, "span_partial_recall": partial_rec, "span_partial_f1": partial_f1
    }

# -------------------------------
# Hyperparameter tuning (optional)
# -------------------------------
def hyperparameter_tuning(train_records, val_records):
    if len(val_records) == 0 or len(train_records) == 0:
        print("Validation or training set empty – using default hyperparameters.")
        return {"lr": 2e-5, "batch_size": 2, "epochs": 15}
    
    lr_options = [2e-5, 5e-5]
    batch_size_options = [2, 4]
    epochs_options = [15, 20]
    grad_accum = 2
    best_score = 0.0
    best_params = None
    
    train_chunks = build_chunks_from_records(train_records)
    val_chunks = build_chunks_from_records(val_records)
    if len(train_chunks) == 0 or len(val_chunks) == 0:
        print("No chunks generated – using default params.")
        return {"lr": 2e-5, "batch_size": 2, "epochs": 15}
    
    for lr, bs, epochs in itertools.product(lr_options, batch_size_options, epochs_options):
        print(f"\n--- Tuning: LR={lr}, BATCH_SIZE={bs}, EPOCHS={epochs} ---")
        train_dataset = ChunkDataset(train_chunks)
        val_dataset = ChunkDataset(val_chunks)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn)
        
        model, val_f1 = train_model(train_loader, val_loader, lr, epochs, bs, grad_accum, patience=3, seed=42)
        if val_f1 > best_score:
            best_score = val_f1
            best_params = {"lr": lr, "batch_size": bs, "epochs": epochs}
        del model
        torch.cuda.empty_cache()
    
    print(f"\nBest single model validation insomnia F1: {best_score:.4f} with params {best_params}")
    return best_params

# -------------------------------
# Main pipeline
# -------------------------------
def main():
    # Load data
    train_records = load_train_val(TRAIN_DIR)
    val_records = load_train_val(VAL_DIR)
    test_records = load_test(TEST_DIR)
    print(f"Train docs: {len(train_records)}, Val docs: {len(val_records)}, Test docs: {len(test_records)}")
    
    # Hyperparameter tuning
    if len(val_records) > 0 and len(train_records) > 0:
        best_params = hyperparameter_tuning(train_records, val_records)
    else:
        best_params = {"lr": 2e-5, "batch_size": 2, "epochs": 15}
        print("Insufficient data for tuning – using default hyperparameters.")
    
    # Combine train+val
    combined_records = train_records + val_records
    combined_chunks = build_chunks_from_records(combined_records)
    if len(combined_chunks) == 0:
        print("Error: No chunks generated from combined data. Check input files.")
        return
    combined_dataset = ChunkDataset(combined_chunks)
    combined_loader = DataLoader(combined_dataset, batch_size=best_params["batch_size"], shuffle=True, collate_fn=collate_fn)
    
    # Train ensemble
    n_ensemble = 3
    seeds = [42, 123, 999]
    ensemble_models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n=== Training ensemble of {n_ensemble} models on train+val ===")
    for i, seed in enumerate(seeds):
        print(f"\n--- Model {i+1}/{n_ensemble} with seed {seed} ---")
        dummy_loader = DataLoader(ChunkDataset([]), batch_size=1, shuffle=False, collate_fn=collate_fn)
        model, _ = train_model(combined_loader, dummy_loader,
                               best_params["lr"], best_params["epochs"],
                               best_params["batch_size"], GRAD_ACCUM,
                               patience=999, seed=seed)
        ensemble_models.append(model)
        torch.cuda.empty_cache()
    
    # Evaluate ensemble on validation set (if any)
    if len(val_records) > 0:
        print("\n=== Validation Set Evaluation (Ensemble) ===")
        val_metrics = evaluate_document_level_ensemble(ensemble_models, val_records, device)
        print(f"Insomnia - Precision: {val_metrics['insomnia_precision']:.4f}, Recall: {val_metrics['insomnia_recall']:.4f}, F1: {val_metrics['insomnia_f1']:.4f}")
        print(f"Label - Precision: {val_metrics['label_precision']:.4f}, Recall: {val_metrics['label_recall']:.4f}, F1: {val_metrics['label_f1']:.4f}")
        print(f"Span Exact - Precision: {val_metrics['span_exact_precision']:.4f}, Recall: {val_metrics['span_exact_recall']:.4f}, F1: {val_metrics['span_exact_f1']:.4f}")
        print(f"Span Partial - Precision: {val_metrics['span_partial_precision']:.4f}, Recall: {val_metrics['span_partial_recall']:.4f}, F1: {val_metrics['span_partial_f1']:.4f}")
    
    # Test inference
    print("\n=== Running inference on test set ===")
    test_pred_ins = []
    test_pred_spans = []
    for rec in tqdm(test_records, desc="Test Inference"):
        pred_ins, pred_spans = predict_document_ensemble(rec, ensemble_models, device)
        test_pred_ins.append(pred_ins)
        test_pred_spans.append(pred_spans)
    
    # Save predictions
    sub1_out = {rec["note_id"]: {"Insomnia": "yes" if p else "no"} 
                for rec, p in zip(test_records, test_pred_ins)}
    sub2_out = {}
    for rec, pred_spans in zip(test_records, test_pred_spans):
        sub2_out[rec["note_id"]] = {}
        for rule in RULE_NAMES:
            spans_str = [f"{s} {e}" for s,e in pred_spans[rule]]
            sub2_out[rec["note_id"]][rule] = {
                "label": "yes" if spans_str else "no",
                "span": spans_str,
                "text": []
            }
    with open("subtask_1_predictions_eff.json", "w") as f:
        json.dump(sub1_out, f, indent=2)
    with open("subtask_2_predictions_eff.json", "w") as f:
        json.dump(sub2_out, f, indent=2)
    print("\nTest predictions saved to subtask_1_predictions_eff.json and subtask_2_predictions_eff.json")

if __name__ == "__main__":
    main()




