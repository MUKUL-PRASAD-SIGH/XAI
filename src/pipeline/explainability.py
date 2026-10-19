"""
explainability.py — XAI wrapper: Grad-CAM, SHAP, LIME.

All methods are purely inferential (read-only with respect to source data).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def compute_gradcam(
    model: "nn.Module",
    target_layers: list,
    input_tensor: "torch.Tensor",
    targets: Optional[list] = None,
) -> np.ndarray:
    """
    Return a Grad-CAM heatmap for a single input tensor.

    Args:
        model: Trained PyTorch model.
        target_layers: List of conv layers to hook.
        input_tensor: Shape (1, C, H, W).
        targets: Optional class targets; None uses argmax prediction.

    Returns:
        grayscale_cam: np.ndarray of shape (H, W), values in [0, 1].
    """
    if not HAS_GRADCAM:
        raise ImportError("Install pytorch-grad-cam: pip install grad-cam")
    if not HAS_TORCH:
        raise ImportError("Install torch: pip install torch")

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0]  # first item in batch


def overlay_cam(
    rgb_image: np.ndarray,
    grayscale_cam: np.ndarray,
    use_rgb: bool = True,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an RGB image."""
    if not HAS_GRADCAM:
        raise ImportError("Install pytorch-grad-cam: pip install grad-cam")
    rgb = rgb_image.astype(np.float32) / 255.0
    return show_cam_on_image(rgb, grayscale_cam, use_rgb=use_rgb)


def compute_shap_values(
    model_fn,
    background_data: np.ndarray,
    test_data: np.ndarray,
    n_samples: int = 100,
) -> np.ndarray:
    """
    Compute SHAP values using a DeepExplainer or KernelExplainer.

    Args:
        model_fn: Callable that returns predictions for an input array.
        background_data: Representative background dataset.
        test_data: Data points to explain.
        n_samples: Number of background samples for KernelExplainer.

    Returns:
        shap_values: np.ndarray matching test_data shape.
    """
    if not HAS_SHAP:
        raise ImportError("Install shap: pip install shap")
    explainer = shap.KernelExplainer(model_fn, background_data[:n_samples])
    return np.array(explainer.shap_values(test_data))
