"""
Visualization module for Retinal Vessel Segmentation.
Generates side-by-side comparison plots, probability heatmaps,
vessel overlays, and error classification maps.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless servers and Windows
import matplotlib.pyplot as plt
import numpy as np
import torch

def denormalize_image(
    tensor: Union[torch.Tensor, np.ndarray],
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Reverses ImageNet channel-wise normalization and converts tensor to (H, W, 3) uint8 RGB.
    """
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    else:
        img = tensor.copy()

    # If shape is (C, H, W), transpose to (H, W, C)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = (img * std) + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def create_error_map(pred_binary: np.ndarray, gt_binary: np.ndarray) -> np.ndarray:
    """
    Constructs an RGB diagnostic error map:
      - Green: True Positive (TP - correctly segmented vessel)
      - Red:   False Positive (FP - over-segmentation hallucination)
      - Blue:  False Negative (FN - missed vessel / under-segmentation)
      - Black: True Negative (TN - correct background)
    """
    h, w = pred_binary.shape[:2]
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    tp = (pred_binary == 1) & (gt_binary == 1)
    fp = (pred_binary == 1) & (gt_binary == 0)
    fn = (pred_binary == 0) & (gt_binary == 1)

    error_map[tp] = [0, 255, 0]    # Green for TP
    error_map[fp] = [255, 50, 50]  # Red for FP
    error_map[fn] = [50, 100, 255] # Blue for FN

    return error_map


def save_visual_comparison(
    image: Union[torch.Tensor, np.ndarray],
    gt_mask: Union[torch.Tensor, np.ndarray],
    pred_mask: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    sample_id: str = "sample",
    metrics: Optional[dict] = None,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    threshold: float = 0.5,
) -> Path:
    """
    Generates and saves a publication-quality side-by-side comparison figure:
      1. Original Fundus Image
      2. Ground Truth Mask
      3. Predicted Binary Mask (Thresholded)
      4. Prediction Probability Heatmap
      5. Diagnostic Error Overlay (TP=Green, FP=Red, FN=Blue)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Denormalize fundus image to uint8 RGB
    orig_rgb = denormalize_image(image, mean=mean, std=std)

    # Convert masks to 2D numpy arrays
    if isinstance(gt_mask, torch.Tensor):
        gt_arr = gt_mask.detach().cpu().squeeze().numpy()
    else:
        gt_arr = np.squeeze(gt_mask)

    if isinstance(pred_mask, torch.Tensor):
        pred_raw = pred_mask.detach().cpu().squeeze().numpy()
    else:
        pred_raw = np.squeeze(pred_mask)

    # If predictions are logits, convert to probabilities
    if pred_raw.min() < 0.0 or pred_raw.max() > 1.0:
        pred_probs = 1.0 / (1.0 + np.exp(-pred_raw))
    else:
        pred_probs = pred_raw

    pred_bin = (pred_probs > threshold).astype(np.uint8)
    gt_bin = (gt_arr > 0.5).astype(np.uint8)

    error_map = create_error_map(pred_bin, gt_bin)

    # Setup 1x5 multi-panel figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=150)
    fig.patch.set_facecolor("#1e1e1e")

    panels = [
        ("Original Fundus", orig_rgb, None),
        ("Ground Truth Mask", gt_bin, "gray"),
        (f"Prediction (Thresh={threshold})", pred_bin, "gray"),
        ("Probability Heatmap", pred_probs, "magma"),
        ("Error Map (G:TP, R:FP, B:FN)", error_map, None),
    ]

    for ax, (title, data, cmap) in zip(axes, panels):
        ax.set_facecolor("#1e1e1e")
        if cmap:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(data)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    # Overall title with metrics annotation
    title_str = f"FIVES Case: {sample_id}"
    if metrics:
        dice = metrics.get("dice", 0.0)
        iou = metrics.get("iou", 0.0)
        sens = metrics.get("sensitivity", 0.0)
        spec = metrics.get("specificity", 0.0)
        title_str += f"  |  Dice: {dice:.4f}  |  IoU: {iou:.4f}  |  Sensitivity: {sens:.4f}  |  Specificity: {spec:.4f}"

    fig.suptitle(title_str, color="white", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_prediction_triplet(
    image: Union[torch.Tensor, np.ndarray],
    gt_mask: Union[torch.Tensor, np.ndarray],
    pred_mask: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    sample_id: str = "sample",
    threshold: float = 0.5,
) -> Path:
    """
    Lightweight 3-panel side-by-side plot: Original, Ground Truth, Prediction.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    orig_rgb = denormalize_image(image)
    if isinstance(gt_mask, torch.Tensor):
        gt_bin = (gt_mask.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8)
    else:
        gt_bin = (np.squeeze(gt_mask) > 0.5).astype(np.uint8)

    if isinstance(pred_mask, torch.Tensor):
        pred_raw = pred_mask.detach().cpu().squeeze().numpy()
    else:
        pred_raw = np.squeeze(pred_mask)

    if pred_raw.min() < 0.0 or pred_raw.max() > 1.0:
        pred_probs = 1.0 / (1.0 + np.exp(-pred_raw))
    else:
        pred_probs = pred_raw
    pred_bin = (pred_probs > threshold).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    axes[0].imshow(orig_rgb)
    axes[0].set_title(f"Original: {sample_id}")
    axes[0].axis("off")

    axes[1].imshow(gt_bin, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_bin, cmap="gray")
    axes[2].set_title(f"Model Prediction (t={threshold})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return save_path
