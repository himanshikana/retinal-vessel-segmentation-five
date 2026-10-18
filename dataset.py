"""
PyTorch Dataset and DataLoaders for FIVES Retinal Vessel Segmentation.
Features strict filename stem matching and synchronized Albumentations transforms.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def get_transforms(
    img_size: Tuple[int, int] = (512, 512),
    is_train: bool = True,
    mean: Tuple[float, float, float] = Config.NORM_MEAN,
    std: Tuple[float, float, float] = Config.NORM_STD,
) -> A.Compose:
    """
    Constructs Albumentations composition pipeline.
    Augmentations are applied synchronously to both the retinal image and the vessel mask.
    """
    height, width = img_size

    if is_train:
        return A.Compose([
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(-0.05, 0.05),
                rotate=(-30, 30),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class FIVESDataset(Dataset):
    """
    FIVES Retinal Vessel Segmentation Dataset.
    Enforces strict 1-to-1 filename matching between fundus images and ground-truth vessel masks.
    """
    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        img_size: Tuple[int, int] = (512, 512),
        transform: Optional[A.Compose] = None,
        is_train: bool = False,
        return_filename: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.return_filename = return_filename

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir.resolve()}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir.resolve()}")

        self.transform = transform if transform is not None else get_transforms(img_size, is_train=is_train)

        # Build stem maps excluding non-image artifacts (e.g. Thumbs.db, hidden files)
        mask_map = {
            f.stem.lower(): f
            for f in self.mask_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS and not f.name.startswith(".")
        }

        image_files = [
            f for f in self.image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS and not f.name.startswith(".")
        ]

        self.image_paths: List[Path] = []
        self.mask_paths: List[Path] = []
        self.sample_names: List[str] = []

        # Strict filename-matching logic
        for img_path in sorted(image_files, key=lambda p: p.stem.lower()):
            stem_key = img_path.stem.lower()
            if stem_key in mask_map:
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_map[stem_key])
                self.sample_names.append(img_path.stem)

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No matching image-mask pairs found between:\n"
                f"  Images: {self.image_dir.resolve()}\n"
                f"  Masks:  {self.mask_dir.resolve()}"
            )

        print(f"[{'TRAIN' if is_train else 'EVAL'}] Loaded {len(self.image_paths)} matched fundus image-mask pairs from {self.image_dir.name}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        sample_name = self.sample_names[idx]

        # Read fundus image as RGB
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"Failed to read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read binary mask as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to read mask at: {mask_path}")
        # Binarize mask to float32 in {0.0, 1.0}
        mask = (mask > 127).astype(np.float32)

        # Apply synchronized Albumentations
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"]

        # Ensure mask is (1, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.float()

        if self.return_filename:
            return image_tensor, mask_tensor, sample_name
        return image_tensor, mask_tensor


def get_dataloaders(
    config: type = Config,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for training and evaluation.
    """
    bs = batch_size if batch_size is not None else config.BATCH_SIZE
    nw = num_workers if num_workers is not None else config.NUM_WORKERS

    train_dataset = FIVESDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        img_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        is_train=True,
    )

    test_dataset = FIVESDataset(
        image_dir=config.TEST_IMG_DIR,
        mask_dir=config.TEST_MASK_DIR,
        img_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=config.PIN_MEMORY,
        drop_last=True if len(train_dataset) > bs else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    Config.create_dirs()
    print("Testing FIVESDataset initialization and shape validation...")
    train_loader, test_loader = get_dataloaders(Config, batch_size=2)
    sample_imgs, sample_masks = next(iter(train_loader))
    print(f"Batch Image Shape: {sample_imgs.shape} | dtype: {sample_imgs.dtype}")
    print(f"Batch Mask Shape:  {sample_masks.shape} | dtype: {sample_masks.dtype}")
    print(f"Mask values range: [{sample_masks.min():.2f}, {sample_masks.max():.2f}]")
    print("FIVESDataset verification PASSED successfully.")
