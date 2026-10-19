"""
validate_labels.py — Label map sanity checks (read-only).

Usage:
    python scripts/validate_labels.py --labels-file /path/to/labels.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REQUIRED_COLUMNS = {"filename", "label"}


def validate_labels(labels_file: str) -> dict:
    fpath = Path(labels_file)
    if not fpath.exists():
        raise FileNotFoundError(f"Labels file not found: {fpath}")

    df = pd.read_csv(fpath)
    missing = REQUIRED_COLUMNS - set(df.columns.str.lower())

    report = {
        "file": str(fpath),
        "total_rows": len(df),
        "columns": list(df.columns),
        "missing_required_columns": list(missing),
        "null_counts": df.isnull().sum().to_dict(),
        "label_distribution": df["label"].value_counts().to_dict() if "label" in df.columns else {},
        "duplicate_filenames": int(df["filename"].duplicated().sum()) if "filename" in df.columns else -1,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate label CSV file.")
    parser.add_argument("--labels-file", required=True, help="Path to labels CSV")
    args = parser.parse_args()

    import json
    result = validate_labels(args.labels_file)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
