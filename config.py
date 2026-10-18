"""
Configuration module for Retinal Vessel Segmentation on FIVES Dataset.
Centralizes hyperparameters, directory paths, and runtime settings.
"""

import os
from pathlib import Path
import torch

class Config:
    # -------------------------------------------------------------------------
    # Base Directories & Relative Paths
    # -------------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent

    # Primary dataset path
    DATA_DIR = PROJECT_ROOT / "data" / "fives"

    # Backward-compatible path resolution: fallback to root if not in data/fives
    if not (DATA_DIR / "train").exists() and (PROJECT_ROOT / "train").exists():
        DATA_DIR = PROJECT_ROOT

    TRAIN_IMG_DIR = DATA_DIR / "train" / "Original"
    TRAIN_MASK_DIR = DATA_DIR / "train" / "Ground truth"
    TEST_IMG_DIR = DATA_DIR / "test" / "Original"
    TEST_MASK_DIR = DATA_DIR / "test" / "Ground truth"

    # Evaluation / Validation aliases
    VAL_IMG_DIR = TEST_IMG_DIR
    VAL_MASK_DIR = TEST_MASK_DIR

    # Output & Checkpoint Directories
    WEIGHTS_DIR = PROJECT_ROOT / "weights"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    VIS_DIR = OUTPUTS_DIR / "visualizations"
    METRICS_DIR = OUTPUTS_DIR / "metrics"

    BEST_MODEL_PATH = WEIGHTS_DIR / "unet_fives_best.pth"
    LAST_MODEL_PATH = WEIGHTS_DIR / "unet_fives_last.pth"

    # -------------------------------------------------------------------------
    # Image & Dataset Hyperparameters
    # -------------------------------------------------------------------------
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # ImageNet normalization statistics
    NORM_MEAN = (0.485, 0.456, 0.406)
    NORM_STD = (0.229, 0.224, 0.225)

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 4
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    SEED = 42

    # Compound Loss Weights (BCE + Soft Dice)
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    DICE_SMOOTH = 1e-6

    # Model Dropout Rate (must stay consistent between training and inference)
    MODEL_DROPOUT_RATE = 0.1

    # Learning Rate Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-7

    # Metric Threshold
    DECISION_THRESHOLD = 0.5

    # -------------------------------------------------------------------------
    # Hardware & Performance
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = torch.cuda.is_available()
    # On Windows, num_workers > 0 can cause multiprocessing overhead; default to 0 for maximum stability
    NUM_WORKERS = 0 if os.name == "nt" else 2
    PIN_MEMORY = torch.cuda.is_available()

    @classmethod
    def create_dirs(cls):
        """Ensures that all runtime output directories exist."""
        cls.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VIS_DIR.mkdir(parents=True, exist_ok=True)
        cls.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def to_dict(cls):
        """Returns all uppercase configuration parameters as a dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in cls.__dict__.items()
            if k.isupper() and not k.startswith("__")
        }

    @classmethod
    def display(cls):
        """Prints formatted configuration summary."""
        print("=" * 60)
        print("RETINAL VESSEL SEGMENTATION - CONFIGURATION")
        print("=" * 60)
        for k, v in cls.to_dict().items():
            print(f"  {k:<22} : {v}")
        print("=" * 60)


if __name__ == "__main__":
    Config.create_dirs()
    Config.display()
