"""
Sanity tests for the XAI pipeline scripts and directory structure.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_repo_structure():
    assert (REPO_ROOT / "scripts").is_dir()
    assert (REPO_ROOT / "notebooks").is_dir()
    assert (REPO_ROOT / "requirements.txt").is_file()
    assert (REPO_ROOT / "README.md").is_file()

def test_core_scripts_exist():
    expected_scripts = [
        "01_dataset_audit.py",
        "02_rebuild_dataset.py",
        "02b_make_cv_folds.py",
        "03_train.py",
        "04_evaluate.py",
        "05_diagnostics.py",
        "06_insights.py",
        "07_build_grouped.py",
        "08_compare_runs.py",
        "09_compare_abnormal.py",
        "12_feature_probe.py",
        "app.py",
    ]
    for s in expected_scripts:
        script_file = REPO_ROOT / "scripts" / s
        assert script_file.is_file(), f"Missing expected pipeline script: {s}"
