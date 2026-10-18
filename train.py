"""
Training pipeline for Retinal Vessel Segmentation on the FIVES dataset.
Features Compound Loss (BCE + Soft Dice), AdamW, ReduceLROnPlateau,
Mixed-Precision Training (AMP), and robust checkpointing.
"""

import argparse
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from dataset import get_dataloaders, FIVESDataset
from model import UNet
from utils.losses import CompoundLoss
from utils.metrics import calculate_batch_metrics, MetricTracker


def set_seed(seed: int = 42):
    """Ensures deterministic behavior across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: CompoundLoss,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    device_type: str,
    use_amp: bool,
    max_batches: int = None,
) -> dict:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice_loss = 0.0
    tracker = MetricTracker()
    total_samples = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(images)
            loss, bce, dice_l = criterion(outputs, masks, return_components=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        running_bce += bce.item() * batch_size
        running_dice_loss += dice_l.item() * batch_size
        total_samples += batch_size

        batch_metrics = calculate_batch_metrics(outputs.detach(), masks)
        tracker.update(batch_metrics)

    epoch_metrics = tracker.summary()
    return {
        "loss": running_loss / max(total_samples, 1),
        "bce": running_bce / max(total_samples, 1),
        "dice_loss": running_dice_loss / max(total_samples, 1),
        "dice": epoch_metrics["dice"]["mean"],
        "iou": epoch_metrics["iou"]["mean"],
    }


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: CompoundLoss,
    device: torch.device,
    device_type: str,
    use_amp: bool,
    max_batches: int = None,
) -> dict:
    """Evaluates model performance across validation set."""
    model.eval()
    running_loss = 0.0
    running_bce = 0.0
    running_dice_loss = 0.0
    tracker = MetricTracker()
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_size = images.size(0)

            with torch.autocast(device_type=device_type, enabled=use_amp):
                outputs = model(images)
                loss, bce, dice_l = criterion(outputs, masks, return_components=True)

            running_loss += loss.item() * batch_size
            running_bce += bce.item() * batch_size
            running_dice_loss += dice_l.item() * batch_size
            total_samples += batch_size

            batch_metrics = calculate_batch_metrics(outputs, masks)
            tracker.update(batch_metrics)

    epoch_metrics = tracker.summary()
    return {
        "loss": running_loss / max(total_samples, 1),
        "bce": running_bce / max(total_samples, 1),
        "dice_loss": running_dice_loss / max(total_samples, 1),
        "dice": epoch_metrics["dice"]["mean"],
        "iou": epoch_metrics["iou"]["mean"],
        "sensitivity": epoch_metrics["sensitivity"]["mean"],
        "specificity": epoch_metrics["specificity"]["mean"],
        "accuracy": epoch_metrics["accuracy"]["mean"],
    }


def train_pipeline(args):
    """Main training orchestration."""
    Config.create_dirs()
    set_seed(Config.SEED)

    device = Config.DEVICE
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = (device_type == "cuda" and args.amp)

    print("=" * 70)
    print("RETINAL VESSEL SEGMENTATION - TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device:           {device} (AMP: {'Enabled' if use_amp else 'Disabled'})")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Epochs:           {args.epochs}")
    print(f"Learning Rate:    {args.lr}")
    print(f"BCE Weight:       {args.bce_weight} | Dice Weight: {args.dice_weight}")
    print(f"Weights Dir:      {Config.WEIGHTS_DIR}")
    print("=" * 70)

    # 1. Initialize DataLoaders
    train_loader, val_loader = get_dataloaders(
        Config,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. Build Model
    model = UNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.NUM_CLASSES,
        base_channels=64,
        dropout_rate=Config.MODEL_DROPOUT_RATE
    ).to(device)

    # 3. Setup Loss, Optimizer, Scheduler, Scaler
    criterion = CompoundLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        smooth=Config.DICE_SMOOTH
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_dice = 0.0

    if args.resume and Path(args.resume).exists():
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_dice = checkpoint.get("best_dice", 0.0)
        else:
            model.load_state_dict(checkpoint)
        print(f"Resumed at epoch {start_epoch} with previous best Dice: {best_val_dice:.4f}")

    # 4. Training Loop
    total_start_time = time.time()
    max_batches = args.max_batches if args.max_batches > 0 else None

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch + 1:02d}/{args.epochs:02d}] | LR: {current_lr:.2e}")
        print("-" * 55)

        train_res = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            device_type=device_type,
            use_amp=use_amp,
            max_batches=max_batches,
        )

        val_res = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            device_type=device_type,
            use_amp=use_amp,
            max_batches=max_batches,
        )

        # Scheduler step based on validation Dice score
        scheduler.step(val_res["dice"])

        epoch_duration = time.time() - epoch_start
        print(
            f"  Train -> Loss: {train_res['loss']:.4f} (BCE: {train_res['bce']:.4f}, Dice: {train_res['dice_loss']:.4f}) | "
            f"Dice: {train_res['dice']:.4f} | IoU: {train_res['iou']:.4f}"
        )
        print(
            f"  Val   -> Loss: {val_res['loss']:.4f} | Dice: {val_res['dice']:.4f} | IoU: {val_res['iou']:.4f} | "
            f"Sens: {val_res['sensitivity']:.4f} | Spec: {val_res['specificity']:.4f} | [{epoch_duration:.1f}s]"
        )

        # Save latest checkpoint
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": best_val_dice,
            "config": Config.to_dict(),
        }
        torch.save(latest_ckpt, Config.LAST_MODEL_PATH)

        # Checkpointing best model
        if val_res["dice"] > best_val_dice:
            improvement = val_res["dice"] - best_val_dice
            best_val_dice = val_res["dice"]
            print(f"  >>> Best model saved! (Val Dice improved by +{improvement:.4f} to {best_val_dice:.4f})")

            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_val_dice,
                "val_iou": val_res["iou"],
                "val_sensitivity": val_res["sensitivity"],
                "val_specificity": val_res["specificity"],
                "config": Config.to_dict(),
            }
            torch.save(best_ckpt, Config.BEST_MODEL_PATH)

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE in {total_duration / 60:.2f} minutes")
    print(f"Best Validation Dice Score: {best_val_dice:.4f}")
    print(f"Saved Checkpoint: {Config.BEST_MODEL_PATH}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on FIVES Retinal Vessel Dataset")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=Config.WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--bce-weight", type=float, default=Config.BCE_WEIGHT, help="Weight for BCE loss")
    parser.add_argument("--dice-weight", type=float, default=Config.DICE_WEIGHT, help="Weight for Dice loss")
    parser.add_argument("--amp", action="store_true", default=Config.USE_AMP, help="Enable AMP")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--max-batches", type=int, default=-1, help="Limit batches per epoch (for quick testing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args)
