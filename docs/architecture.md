# Architecture & Pipeline Overview

## Directory & Data Contract

1. **Source Code**: Tracked in Git, synchronized with Google Colab sessions.
2. **Research Data & Artifacts**: Kept strictly in Google Drive (`FetalBrain_XAI_Workspace`).
   - `00_Documentation/`: Audit and runbook specifications.
   - `01_Source_Code/`: Historical research scripts.
   - `02_Original_Dataset/`: Ultrasound zip archives.
   - `03_Manifests_And_Labels/`: Grouping mappings and split manifests.
   - `04_Current_Results/`: Synced evaluative metrics, confusion matrices, and test reports.
   - `05_Model_Backups/`: Checkpoint weights (.keras).
   - `08_Antigravity_Reports/`: Timestamped execution runs.

## Research Pipeline Stages

```
Raw Export (ZIP)
   │
   ▼
[01_dataset_audit.py] ──> pHash duplicate detection & patient leakage audit
   │
   ▼
[02_rebuild_dataset.py] ──> Leak-free case-level split (train-only augmentation)
   │
   ▼
[03_train.py] ───────────> 2-Phase Xception Transfer Learning (frozen head -> fine-tuning)
   │
   ├─> [04_evaluate.py] ──> Macro-F1 with TTA & per-class confusion matrices
   ├─> [05_diagnostics.py] > Confound & ventriculomegaly severity merge analysis
   ├─> [06_insights.py] ───> Pixel shortcut test & overfitting checks
   ├─> [07_build_grouped.py] -> Derived anomaly families & abnormal-only splits
   ├─> [09_compare_abnormal.py] -> McNemar paired significance test
   └─> [12_feature_probe.py] -> L1 sparse deep feature selection
```
