"""Run a predeclared Xception ablation on existing folds; never overwrite a run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


EXPERIMENTS = {
    "reference": ["--mixup", "0.2", "--cutout", "0.25"],
    "basic": ["--mixup", "0", "--cutout", "0"],
    "mixup": ["--mixup", "0.2", "--cutout", "0"],
    "cutout": ["--mixup", "0", "--cutout", "0.25"],
    "basic-f1": ["--mixup", "0", "--cutout", "0", "--monitor", "macro_f1"],
    "anatomy": ["--mixup", "0.2", "--cutout", "0.25"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    ap.add_argument("--init-backbone", type=Path)
    ap.add_argument("--only-folds", nargs="+", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    scripts = Path(__file__).resolve().parent
    for name in ("cv_manifest.csv", "cv_config.json", "class_names.json"):
        if not (a.cv / name).is_file():
            ap.error(f"Missing {a.cv / name}; prepare the frozen CV dataset first")
    if a.out.exists():
        ap.error("Output already exists. Choose a new run folder to preserve results.")
    if a.experiment == "anatomy" and not (a.init_backbone and a.init_backbone.is_file()):
        ap.error("anatomy requires an existing --init-backbone weights file")
    if a.init_backbone and a.experiment != "anatomy":
        ap.error("--init-backbone is only permitted for the anatomy experiment")
    cmd = [sys.executable, "-u", str(scripts / "03_train.py"), "--cv", str(a.cv),
           "--model", "xception", "--out", str(a.out / "models"),
           "--mixed-precision", *EXPERIMENTS[a.experiment]]
    if a.only_folds:
        cmd += ["--only-folds", *map(str, a.only_folds)]
    if a.init_backbone:
        cmd += ["--init-backbone", str(a.init_backbone)]
    print("Training command:", cmd, flush=True)
    if a.dry_run:
        return
    a.out.mkdir(parents=True)
    provenance = {name: hashlib.sha256((a.cv / name).read_bytes()).hexdigest()
                  for name in ("cv_manifest.csv", "cv_config.json", "class_names.json")}
    (a.out / "experiment.json").write_text(json.dumps(
        {"experiment": a.experiment, "command": cmd, "dataset_sha256": provenance}, indent=2))
    # Stream to both console and disk, preserving the complete traceback.
    with (a.out / "training.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise SystemExit("Training failed; inspect training.log. Evaluation not run.")
    if a.only_folds:
        print("Diagnostic subset complete. No pooled evaluation for a partial run.")
        return
    subprocess.run([sys.executable, str(scripts / "04_evaluate.py"), "--cv", str(a.cv),
                    "--models", str(a.out / "models"), "--model", "xception",
                    "--out", str(a.out / "results"), "--tta"], check=True)


if __name__ == "__main__":
    main()
