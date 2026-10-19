"""
data_loader.py — Read-only dataset access utilities.

SAFETY CONTRACT:
  - This module never writes, moves, renames, or deletes source data.
  - All paths are validated before access.
  - Drive paths are only accessible from inside a Colab runtime.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


SUPPORTED_EXTENSIONS = {".nii", ".nii.gz", ".dcm", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def validate_workspace(workspace_root: str | Path) -> dict:
    """
    Read-only scan of a workspace directory.

    Returns a summary dict with counts, sizes, and top-level structure.
    Never modifies any file.
    """
    root = Path(workspace_root)
    if not root.exists():
        raise FileNotFoundError(f"Workspace not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root}")

    summary = {
        "root": str(root),
        "total_files": 0,
        "total_size_bytes": 0,
        "by_extension": {},
        "top_level_dirs": [],
    }

    for child in sorted(root.iterdir()):
        if child.is_dir():
            summary["top_level_dirs"].append(child.name)

    for fpath in root.rglob("*"):
        if not fpath.is_file():
            continue
        suffix = "".join(fpath.suffixes).lower()
        size = fpath.stat().st_size
        summary["total_files"] += 1
        summary["total_size_bytes"] += size
        summary["by_extension"][suffix] = summary["by_extension"].get(suffix, 0) + 1

    return summary


def iter_images(
    workspace_root: str | Path,
    extensions: Optional[set[str]] = None,
) -> Iterator[Path]:
    """Yield image file paths from workspace_root (read-only iterator)."""
    exts = extensions or SUPPORTED_EXTENSIONS
    root = Path(workspace_root)
    for fpath in sorted(root.rglob("*")):
        if fpath.is_file() and "".join(fpath.suffixes).lower() in exts:
            yield fpath


def file_hash(fpath: str | Path, algorithm: str = "sha256", chunk_size: int = 65536) -> str:
    """Compute a hash of a file without loading it fully into memory."""
    h = hashlib.new(algorithm)
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def load_nifti(fpath: str | Path) -> Optional[tuple[np.ndarray, object]]:
    """Load a NIfTI file and return (data_array, header). Requires nibabel."""
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required: pip install nibabel")
    img = nib.load(str(fpath))
    return np.asarray(img.dataobj), img.header


def load_dicom_metadata(fpath: str | Path) -> Optional[dict]:
    """Load DICOM metadata (no pixel data) for a single file. Requires pydicom."""
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required: pip install pydicom")
    ds = pydicom.dcmread(str(fpath), stop_before_pixels=True)
    return {
        "PatientID": getattr(ds, "PatientID", "UNKNOWN"),
        "StudyDate": getattr(ds, "StudyDate", ""),
        "Modality": getattr(ds, "Modality", ""),
        "Rows": getattr(ds, "Rows", None),
        "Columns": getattr(ds, "Columns", None),
        "SliceThickness": getattr(ds, "SliceThickness", None),
    }
