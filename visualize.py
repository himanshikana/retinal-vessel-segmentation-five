"""
Inference and visualization script for Retinal Vessel Segmentation.
Generates publication-ready side-by-side comparison figures of the
Original Fundus Image, Ground Truth Mask, Model Prediction, Heatmap, and Error Map.
"""

import argparse
from pathlib import Path
import random
import torch

from config import Config
from dataset import FIVESDataset
from model import UNet
from utils.metrics import SegmentationMetrics
from utils.visualization import save_visual_comparison, plot_prediction_triplet


def run_visualizer(args):
    """Generates and saves visual comparison figures."""
    Config.create_dirs()
    device = torch.device(args.device if args.device else Config.DEVICE)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        if Config.LAST_MODEL_PATH.exists():
            print(f"Specified weights not found at {weights_path}, falling back to {Config.LAST_MODEL_PATH}")
            weights_path = Config.LAST_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"Model weights not found at {weights_path.resolve()}.\n"
                f"Please run 'python train.py' to produce a trained model checkpoint."
            )

    print("=" * 65)
    print("RETINAL VESSEL SEGMENTATION - VISUALIZATION ENGINE")
    print("=" * 65)
    print(f"Model Weights:      {weights_path.resolve()}")
    print(f"Output Directory:   {out_dir.resolve()}")
    print(f"Decision Threshold: {args.threshold}")
    print(f"Device:             {device}")
    print("=" * 65)

    # 1. Dataset
    dataset = FIVESDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        img_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        is_train=False,
        return_filename=True,
    )

    # 2. Model
    model = UNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.NUM_CLASSES,
        base_channels=64,
        dropout_rate=Config.MODEL_DROPOUT_RATE,
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. Select indices to visualize
    total_samples = len(dataset)
    indices = []

    if args.sample_ids:
        # Match requested sample stems
        name_to_idx = {dataset.sample_names[i]: i for i in range(total_samples)}
        for s_id in args.sample_ids:
            clean_id = Path(s_id).stem
            if clean_id in name_to_idx:
                indices.append(name_to_idx[clean_id])
            else:
                print(f"[Warning] Sample ID '{s_id}' not found in dataset. Skipping.")
    elif args.random:
        sample_k = min(args.num_samples, total_samples)
        indices = sorted(random.sample(range(total_samples), sample_k))
    else:
        sample_k = min(args.num_samples, total_samples)
        indices = list(range(sample_k))

    if not indices:
        print("No valid samples selected for visualization.")
        return

    print(f"\nGenerating {len(indices)} visualization figures...\n")
    metric_calc = SegmentationMetrics(threshold=args.threshold)

    saved_paths = []
    with torch.no_grad():
        for idx in indices:
            image, mask, sample_name = dataset[idx]
            img_tensor = image.unsqueeze(0).to(device)

            output = model(img_tensor)
            pred_probs = torch.sigmoid(output).squeeze(0).cpu()

            metrics = metric_calc.compute_all(pred_probs, mask)

            save_file = out_dir / f"comparison_{sample_name}.png"
            saved_fig = save_visual_comparison(
                image=image,
                gt_mask=mask,
                pred_mask=pred_probs,
                save_path=save_file,
                sample_id=sample_name,
                metrics=metrics,
                threshold=args.threshold,
            )
            saved_paths.append(saved_fig)
            print(f"  [+] Saved: {save_file.name} | Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}")

    print("\n" + "=" * 65)
    print(f"Successfully generated {len(saved_paths)} comparison figures.")
    print(f"Saved into directory: {out_dir.resolve()}")
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Retinal Vessel Predictions")
    parser.add_argument("--weights", type=str, default=str(Config.BEST_MODEL_PATH), help="Weights path")
    parser.add_argument("--image-dir", type=str, default=str(Config.TEST_IMG_DIR), help="Image directory")
    parser.add_argument("--mask-dir", type=str, default=str(Config.TEST_MASK_DIR), help="Mask directory")
    parser.add_argument("--output-dir", type=str, default=str(Config.VIS_DIR), help="Output visualization dir")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--sample-ids", nargs="*", default=None, help="Specific sample IDs to visualize")
    parser.add_argument("--random", action="store_true", default=False, help="Random sample selection")
    parser.add_argument("--threshold", type=float, default=Config.DECISION_THRESHOLD, help="Binary threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_visualizer(args)
