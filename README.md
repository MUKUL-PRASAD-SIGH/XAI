# Fetal Brain Ultrasound — XAI Research Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MUKUL-PRASAD-SIGH/XAI/blob/main/notebooks/fetal_brain_xai_colab.ipynb)

> **Research-grade explainability pipeline for fetal brain ultrasound classification.**  
> Source code lives here on GitHub. Datasets, trained models, and historical results live in Google Drive (`FetalBrain_XAI_Workspace`).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quick Start — Google Colab](#quick-start--google-colab)
4. [Local Development](#local-development)
5. [GitHub–Colab Workflow](#githubcolab-workflow)
6. [Data & Model Policy](#data--model-policy)
7. [Safety & Ethics](#safety--ethics)
8. [Contributing](#contributing)

---

## Project Overview

This project applies explainable AI (XAI) techniques to fetal brain ultrasound images to:
- Classify developmental anomalies with deep learning models
- Generate saliency maps (Grad-CAM, SHAP, LIME) to highlight diagnostically relevant regions
- Produce auditable, timestamped reports for clinical review
- Maintain reproducibility across Colab sessions without touching research data

---

## Repository Structure

```
XAI/
├── notebooks/
│   └── fetal_brain_xai_colab.ipynb   # Main Colab notebook
├── src/
│   └── pipeline/                     # Reusable Python modules
│       ├── __init__.py
│       ├── data_loader.py
│       ├── model_registry.py
│       ├── explainability.py
│       └── report_generator.py
├── scripts/
│   ├── inventory.py                  # Read-only dataset scan
│   ├── validate_labels.py            # Label map sanity checks
│   ├── detect_duplicates.py          # Hash-based duplicate detection
│   ├── split_dataset.py              # Train/val/test splitting
│   └── extract_metadata.py           # DICOM / NIfTI metadata
├── tests/
│   ├── test_data_loader.py
│   └── test_pipeline_safety.py
├── docs/
│   ├── architecture.md
│   ├── data_access.md
│   ├── operating_instructions.md
│   └── safety.md
├── reports/
│   └── outputs/                      # Timestamped run outputs (git-ignored)
├── data/                             # git-ignored; use Drive
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start — Google Colab

> **Datasets and models are in Google Drive, not in this repo.**

Click the badge at the top **or** open manually:

```
https://colab.research.google.com/github/MUKUL-PRASAD-SIGH/XAI/blob/main/notebooks/fetal_brain_xai_colab.ipynb
```

The notebook will guide you through:
1. Runtime + GPU check
2. Drive mount (requires your authorization in-browser)
3. Drive workspace validation (read-only — nothing is moved or deleted)
4. Fetching latest code from this repo
5. Installing dependencies for the current stage
6. Running the dataset inventory
7. Saving timestamped reports

---

## Local Development

```bash
# Clone
git clone https://github.com/MUKUL-PRASAD-SIGH/XAI.git
cd XAI

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

---

## GitHub–Colab Workflow

| Step | Who does it | What happens |
|------|-------------|--------------|
| Edit code | Antigravity / developer | Local changes in `src/`, `scripts/`, `notebooks/` |
| Commit & push | Developer | `git add . && git commit -m "..." && git push` |
| Open notebook | You | Click Colab badge or paste URL |
| Pull latest code | Notebook cell | `!git clone` or `!git pull` inside Colab runtime |
| Mount Drive | You | Authorize in-browser popup |
| Validate workspace | Notebook cell | Read-only scan — no files moved |
| Run experiment | You | Execute cells in sequence |
| Save results | Notebook cell | Timestamped outputs to Drive subfolder |

---

## Data & Model Policy

- **Raw data** (ultrasound images, DICOM, NIfTI) is **never committed** to this repo.
- **Trained model weights** (`.pt`, `.h5`, `.pth`) are **never committed** here.
- All data lives in `FetalBrain_XAI_Workspace` on Google Drive.
- Drive is mounted inside Colab only — Antigravity's local terminal has **no access** to Drive.
- New model backups require **explicit approval** before being written to Drive.

---

## Safety & Ethics

See [`docs/safety.md`](docs/safety.md) for:
- Patient data handling guidelines
- De-identification requirements
- Audit trail policy
- Responsible XAI disclosure

---

## Contributing

1. Branch from `main`: `git checkout -b feature/your-feature`
2. Follow the code style in `docs/architecture.md`
3. Add tests under `tests/`
4. Open a pull request with a clear description

---

*Last updated: October 2026*


---

## Cross-validated pipeline (current)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MUKUL-PRASAD-SIGH/XAI/blob/main/notebooks/run_full_cv_colab.ipynb)

One notebook, `notebooks/run_full_cv_colab.ipynb`, runs everything from the dataset zip in Drive:

| Step | Script | Purpose |
|---|---|---|
| audit | `01_dataset_audit.py` | duplicates, leakage, source families |
| classes | `02c_design_classes.py` + `configs/class_hierarchy.json` | merge diagnoses only as far as real case counts require |
| folds | `02b_make_cv_folds.py` | stratified, grouped 5-fold; own validation set per fold; leak checks |
| train | `03_train.py` | dropout, mixup, cutout, grayscale layer, AdamW, label smoothing, val-loss stopping |
| evaluate | `04_evaluate.py` | pooled CV, case-level bootstrap CIs, real majority baseline |
| probe | `13_cv_probe.py` | frozen-feature logistic regression on the same folds (overfitting check) |

Details: `docs/cross_validation.md`. Pre-CV scripts that were copied from Drive are kept in `legacy/`.
