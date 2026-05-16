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
```python Subtask_1_2.py``` \\
This code generate two different output corresponding to both subtask, ```subtask_1_predictions_eff.json``` and ```subtask_2_predictions_eff.json```.

## Subtask1 (Ensemble) 
```python Subtask_1.py``` \\
This code generate one single output corresponding to subtask 1, ```test_predictions_subtask1_ensemble.json```.


