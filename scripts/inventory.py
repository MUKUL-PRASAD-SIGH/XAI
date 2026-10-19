"""
inventory.py — Read-only dataset scan script.

Usage:
    python scripts/inventory.py --workspace /path/to/FetalBrain_XAI_Workspace

Outputs a JSON report to reports/outputs/<run_id>_inventory.json.
Never modifies, moves, renames, or deletes any file in the workspace.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.data_loader import validate_workspace, iter_images, file_hash
from src.pipeline.report_generator import save_json_report, make_run_id


def run_inventory(workspace: str, output_dir: str, sample_hashes: int = 20) -> dict:
    print(f"[inventory] Scanning: {workspace}")
    summary = validate_workspace(workspace)

    # Sample file hashes for integrity verification (read-only)
    hashes = {}
    for i, fpath in enumerate(iter_images(workspace)):
        if i >= sample_hashes:
            break
        try:
            hashes[str(fpath)] = file_hash(fpath)
        except Exception as e:
            hashes[str(fpath)] = f"ERROR: {e}"

    run_id = make_run_id()
    report = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "workspace": str(workspace),
        "summary": summary,
        "sample_hashes": hashes,
    }

    out_path = save_json_report(report, output_dir, f"{run_id}_inventory.json")
    print(f"[inventory] Report saved: {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only dataset inventory scan.")
    parser.add_argument("--workspace", required=True, help="Path to FetalBrain_XAI_Workspace")
    parser.add_argument("--output-dir", default="reports/outputs", help="Where to save the report")
    parser.add_argument("--sample-hashes", type=int, default=20, help="Files to hash-check")
    args = parser.parse_args()

    report = run_inventory(args.workspace, args.output_dir, args.sample_hashes)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
