"""
Evaluation module for Retinal Vessel Segmentation on the FIVES Test Dataset.
Calculates Mean Dice Score (F1-Index), Mean IoU (Jaccard Index), Accuracy,
Sensitivity (Recall), Specificity, and Precision with standard deviations.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import FIVESDataset
from model import UNet
from utils.metrics import SegmentationMetrics, MetricTracker


def evaluate_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    save_csv_path: Path = None,
    max_batches: int = None,
) -> dict:
    """
    Evaluates the model over the dataset and reports comprehensive segmentation metrics.

    Args:
        max_batches: If set, stops after this many batches (for quick smoke-testing).
    """
    model.eval()
    calc = SegmentationMetrics(threshold=threshold)
    tracker = MetricTracker()
    per_sample_records = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if len(batch) == 3:
                images, masks, sample_names = batch
            else:
                images, masks = batch
                sample_names = [f"sample_{i}" for i in range(images.size(0))]

            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu()
            masks = masks.cpu()

            for i in range(images.size(0)):
                metrics = calc.compute_all(probs[i], masks[i])
                tracker.update(metrics)

                record = {"sample": sample_names[i], **metrics}
                per_sample_records.append(record)

    summary = tracker.summary()

    if save_csv_path and per_sample_records:
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(per_sample_records)
        df.to_csv(save_csv_path, index=False)
        print(f"Per-sample metrics saved to: {save_csv_path.resolve()}")

    return summary


def run_evaluation(args):
    """Orchestrates model evaluation with clean reporting."""
    Config.create_dirs()
    device = torch.device(args.device if args.device else Config.DEVICE)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        if Config.LAST_MODEL_PATH.exists():
            print(f"Specified weights not found at {weights_path}, falling back to {Config.LAST_MODEL_PATH}")
            weights_path = Config.LAST_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"Model weights not found at {weights_path.resolve()}.\n"
                f"Please run 'python train.py' first to train the model and generate weights."
            )

    max_batches = args.max_batches if args.max_batches > 0 else None

    print("=" * 65)
    print("RETINAL VESSEL SEGMENTATION - TEST EVALUATION")
    print("=" * 65)
    print(f"Model Weights:      {weights_path.resolve()}")
    print(f"Test Image Dir:     {Path(args.image_dir).resolve()}")
    print(f"Decision Threshold: {args.threshold}")
    print(f"Device:             {device}")
    if max_batches:
        print(f"[Smoke Test] Limited to {max_batches} batch(es)")
    print("=" * 65)

    # 1. Dataset & DataLoader
    test_dataset = FIVESDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        img_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        is_train=False,
        return_filename=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Model Initialization & Weights Loading
    model = UNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.NUM_CLASSES,
        base_channels=64,
        dropout_rate=Config.MODEL_DROPOUT_RATE,
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch_info = checkpoint.get("epoch", "unknown")
        print(f"Loaded checkpoint trained up to epoch {epoch_info}")
    else:
        model.load_state_dict(checkpoint)

    # 3. Compute Metrics
    csv_out = Config.METRICS_DIR / "test_metrics_per_sample.csv" if (args.save_csv and not max_batches) else None
    results = evaluate_dataset(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=args.threshold,
        save_csv_path=csv_out,
        max_batches=max_batches,
    )

    # 4. Formatted Display
    print("\n" + "=" * 65)
    print(f"{'METRIC':<25} | {'MEAN':<12} | {'STD DEV':<12}")
    print("-" * 65)
    print(f"{'Dice Score (F1-Index)':<25} | {results['dice']['mean']:<12.4f} | {results['dice']['std']:<12.4f}")
    print(f"{'Mean IoU (Jaccard)':<25} | {results['iou']['mean']:<12.4f} | {results['iou']['std']:<12.4f}")
    print(f"{'Pixel Accuracy':<25} | {results['accuracy']['mean']:<12.4f} | {results['accuracy']['std']:<12.4f}")
    print(f"{'Sensitivity (Recall)':<25} | {results['sensitivity']['mean']:<12.4f} | {results['sensitivity']['std']:<12.4f}")
    print(f"{'Specificity':<25} | {results['specificity']['mean']:<12.4f} | {results['specificity']['std']:<12.4f}")
    print(f"{'Precision':<25} | {results['precision']['mean']:<12.4f} | {results['precision']['std']:<12.4f}")
    print("=" * 65)

    # 5. Save JSON summary (full run only)
    if not max_batches:
        summary_path = Config.METRICS_DIR / "evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "weights": str(weights_path),
                    "threshold": args.threshold,
                    "num_samples": len(test_dataset),
                    "metrics": results,
                },
                f,
                indent=4,
            )
        print(f"Summary JSON saved to: {summary_path.resolve()}\n")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Retinal Vessel Segmentation Model")
    parser.add_argument("--weights", type=str, default=str(Config.BEST_MODEL_PATH), help="Path to model weights")
    parser.add_argument("--image-dir", type=str, default=str(Config.TEST_IMG_DIR), help="Test image directory")
    parser.add_argument("--mask-dir", type=str, default=str(Config.TEST_MASK_DIR), help="Test mask directory")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--threshold", type=float, default=Config.DECISION_THRESHOLD, help="Binarization threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--save-csv", action="store_true", default=True, help="Save per-sample metrics CSV")
    parser.add_argument("--max-batches", type=int, default=-1,
                        help="Limit to N batches for quick smoke testing (default: -1 = full dataset)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
