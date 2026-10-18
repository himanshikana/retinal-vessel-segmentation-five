"""
Evaluation metrics for medical semantic segmentation.
Computes Dice Score (F1-Index), Mean IoU (Jaccard Index), Accuracy,
Sensitivity (Recall), Specificity, and Precision.
"""

from typing import Dict, Tuple
import numpy as np
import torch

class SegmentationMetrics:
    """
    Computes semantic segmentation metrics between predicted logits / probabilities
    and ground truth binary masks.
    """
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps

    def compute_confusion_matrix(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float, float, float]:
        """
        Calculates TP, FP, TN, FN counts for binary masks.
        """
        # Ensure binary {0, 1}
        preds_bin = (preds > self.threshold).float()
        targets_bin = (targets > 0.5).float()

        tp = (preds_bin * targets_bin).sum().item()
        fp = (preds_bin * (1.0 - targets_bin)).sum().item()
        fn = ((1.0 - preds_bin) * targets_bin).sum().item()
        tn = ((1.0 - preds_bin) * (1.0 - targets_bin)).sum().item()

        return tp, fp, tn, fn

    def compute_all(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Computes all segmentation metrics for a single prediction-target pair or batch.
        Args:
            preds: Tensor of probabilities or logits. If logits, applies sigmoid first.
            targets: Binary ground truth tensor.
        Returns:
            Dictionary of metrics: dice, iou, accuracy, sensitivity, specificity, precision.
        """
        if preds.min() < 0.0 or preds.max() > 1.0:
            preds = torch.sigmoid(preds)

        tp, fp, tn, fn = self.compute_confusion_matrix(preds, targets)

        dice = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)
        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        accuracy = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        sensitivity = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)

        return {
            "dice": dice,
            "iou": iou,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
        }


def calculate_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Convenience function that calculates mean metrics across a batch by evaluating
    each sample individually and averaging.
    """
    probs = torch.sigmoid(logits)
    batch_size = logits.size(0)
    calc = SegmentationMetrics(threshold=threshold, eps=eps)

    accumulated: Dict[str, float] = {
        "dice": 0.0, "iou": 0.0, "accuracy": 0.0,
        "sensitivity": 0.0, "specificity": 0.0, "precision": 0.0
    }

    for i in range(batch_size):
        sample_metrics = calc.compute_all(probs[i], targets[i])
        for k, v in sample_metrics.items():
            accumulated[k] += v

    return {k: v / batch_size for k, v in accumulated.items()}


class MetricTracker:
    """
    Accumulates metrics across batches to compute mean and standard deviation.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.records: Dict[str, list] = {
            "dice": [], "iou": [], "accuracy": [],
            "sensitivity": [], "specificity": [], "precision": []
        }

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k in self.records.keys():
            if k in metrics:
                self.records[k].append(metrics[k])

    def summary(self) -> Dict[str, Dict[str, float]]:
        res = {}
        for k, vals in self.records.items():
            if len(vals) > 0:
                arr = np.array(vals)
                res[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
            else:
                res[k] = {"mean": 0.0, "std": 0.0}
        return res
