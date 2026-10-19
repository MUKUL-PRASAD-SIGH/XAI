"""
detect_duplicates.py — Hash-based duplicate detection (read-only).

Usage:
    python scripts/detect_duplicates.py --workspace /path/to/FetalBrain_XAI_Workspace
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.data_loader import iter_images, file_hash
from src.pipeline.report_generator import save_json_report, make_run_id


def detect_duplicates(workspace: str, output_dir: str) -> dict:
    hash_map: dict[str, list[str]] = defaultdict(list)

    print(f"[detect_duplicates] Scanning: {workspace}")
    for fpath in iter_images(workspace):
        h = file_hash(fpath)
        hash_map[h].append(str(fpath))

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    run_id = make_run_id()
    report = {
        "run_id": run_id,
        "workspace": workspace,
        "total_files_scanned": sum(len(v) for v in hash_map.values()),
        "unique_files": len(hash_map),
        "duplicate_groups": len(duplicates),
        "duplicates": duplicates,
    }
    out_path = save_json_report(report, output_dir, f"{run_id}_duplicates.json")
    print(f"[detect_duplicates] Report saved: {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash-based duplicate file detection.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output-dir", default="reports/outputs")
    args = parser.parse_args()

    result = detect_duplicates(args.workspace, args.output_dir)
    print(f"Duplicate groups found: {result['duplicate_groups']}")


if __name__ == "__main__":
    main()
