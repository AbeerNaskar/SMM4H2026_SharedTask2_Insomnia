# SMM4H2026_SharedTask2_Insomnia

## Input Format and File Structure

```
/project_root/
├── train/           # Training split
│   ├── sample_corpus.csv
│   ├── subtask_1.json
│   └── subtask_2.json
└── val/             # Validation split
│   ├── sample_corpus.csv
│   ├── subtask_1.json
│   └── subtask_2.json
│── test/            # Test split
│    └── sample_corpus.csv
├── Subtask_1_2.py
├── Subtask_2.py
```

All the files provided by the shared task, and addition to that ```sample_corpus.csv``` has two columns ```note_id``` and ```text```, which is corresponding to the given ```note_id``` the corresponding MIMIC note in ```text``` column.




# Codes
Codes are end to end. Just place the data and code in the mentioned structure and run below codes. D experiment with different models and parameters.

## Subtask1 and Subtask2 (BERT encoder model) 
```python Subtask_1_2.py``` <br/>
This code generate two different output corresponding to both subtask, ```subtask_1_predictions_eff.json``` and ```subtask_2_predictions_eff.json```.

## Subtask1 (Ensemble) 
```python Subtask_1.py``` <br/>
This code generate one single output corresponding to subtask 1, ```test_predictions_subtask1_ensemble.json```.



# Results

We are getting our best output using ```test_predictions_subtask1_ensemble.json``` and ```subtask_2_predictions_eff.json```. Below is our score and mean/median score (mail from organizers). **Bold** and ★ means it surpass mean and median score.   

### Subtask 1 ours best

| Submission Date/Time | Filename | Precision | Recall | F-1 score |
|---|---|---:|---:|---:|
| 4/15/2026 2:48 PM | subtask_1.zip | 0.5455 | **0.9474** ★ | 0.6923 |

---

### Subtask 2 ours best

| Submission Date/Time | Filename | Label Classification | Exact Match | Partial Match |
|---|---|---:|---:|---:|
| 4/15/2026 3:48 PM | subtask_2.zip | **0.6444** ★ | **0.4472** ★ | **0.5093** ★ |

---

# Test Set Performance Summary (provided by organizers)

### Subtask 1 

| Statistic | Precision | Recall | F-1 score |
|---|---:|---:|---:|
| Mean | 0.7336 | 0.6935 | 0.6805 |
| Median | 0.8333 | 0.6842 | 0.7037 |

---

### Subtask 2 

| Statistic | Label Classification | Exact Match | Partial Match |
|---|---:|---:|---:|
| Mean | 0.5888 | 0.3129 | 0.4584 |
| Median | 0.6000 | 0.3586 | 0.4524 |
