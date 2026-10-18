# 🩺 Retinal Vessel Segmentation on FIVES Dataset

A clean, production-ready deep learning pipeline for automatically detecting and mapping blood vessels in retinal fundus photographs. Built on PyTorch with a U-Net architecture, this project is designed to be easy to run locally on Windows, GPU-optional, and research-honest — meaning it reports proper metrics, not just accuracy on imbalanced data.

---

## Why This Exists

Retinal vessel maps are clinically important: abnormal vessel patterns appear early in diabetic retinopathy, glaucoma, and hypertensive retinopathy. Manual annotation is slow and expert-dependent. This pipeline automates binary vessel segmentation (vessel vs. background) using the publicly available **FIVES dataset** — 800 fundus images with pixel-level vessel masks across four disease categories.

---

## What Makes This Different

Most beginner segmentation repos have these problems — all of which are fixed here:

| Common Problem | What I Did Instead |
|---|---|

| Plain BCE loss gives ~95% accuracy by predicting everything as background | **Compound loss** = Binary Cross-Entropy + Soft Dice Loss, directly penalising missed vessels |
| No augmentation, so the model memorises fundus orientations | 6 synchronized augmentations via **Albumentations** — both image and mask transform together |
| Evaluation reports only loss or pixel accuracy | Reports **Dice, IoU, Sensitivity, Specificity, Precision, Accuracy** — all with standard deviation across samples |
| No visualization of what the model actually learned | 5-panel comparison figures: original · ground truth · prediction · probability heatmap · error map (green=correct, red=over-segment, blue=missed) |

---

## Project Structure

```
fundus dataset/
│
├── config.py            # Every setting in one place — paths, image size, learning rate, etc.
├── dataset.py           # Loads FIVES images+masks, handles augmentation, paired filename matching
├── model.py             # U-Net architecture — encoder, decoder, skip connections, batch norm
├── train.py             # Trains the model with compound loss, AdamW, LR scheduler, checkpointing
├── evaluate.py          # Runs inference on the test set and prints a full metrics table
├── visualize.py         # Saves side-by-side comparison figures for any test sample
│
├── utils/
│   ├── losses.py        # DiceLoss and CompoundLoss (BCE + Dice combined)
│   ├── metrics.py       # Per-sample and batch metric calculation with tracking
│   └── visualization.py # Figure generation, error maps, image denormalization
│
├── data/
│   └── fives/
│       ├── train/
│       │   ├── Original/        ← 600 RGB fundus images
│       │   └── Ground truth/    ← 600 binary vessel masks
│       └── test/
│           ├── Original/        ← 200 RGB fundus images
│           └── Ground truth/    ← 200 binary vessel masks
│
├── weights/             # Saved automatically during training
│   ├── unet_fives_best.pth   ← best checkpoint by validation Dice
│   └── unet_fives_last.pth   ← most recent epoch checkpoint
│
├── outputs/
│   ├── visualizations/  ← PNG comparison figures per sample
│   └── metrics/         ← JSON summary + per-sample CSV
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Core packages: `torch`, `albumentations`, `opencv-python-headless`, `matplotlib`, `numpy`, `pandas`, `Pillow`

### 2. Verify your dataset layout

my data should be at:
```
data/fives/train/Original/       ← fundus images (.png)
data/fives/train/Ground truth/   ← masks with identical filenames
data/fives/test/Original/
data/fives/test/Ground truth/
```

The dataset self-checks on startup and prints how many pairs were successfully matched.

---

## How to Run

### Train

```bash
python train.py
```

Trains for 20 epochs by default. Saves the best checkpoint whenever validation Dice improves.

**Useful flags:**

```bash
python train.py --epochs 30 --batch-size 4 --lr 5e-5   # custom hyperparams
python train.py --resume weights/unet_fives_last.pth    # resume after interruption
python train.py --epochs 1 --max-batches 2              # quick smoke test (~1 min)
```

### Evaluate

```bash
python evaluate.py
```

Runs inference on all 200 test images and prints:

```
METRIC                    | MEAN         | STD DEV
-----------------------------------------------------------------
Dice Score (F1-Index)     | 0.XXXX       | 0.XXXX
Mean IoU (Jaccard)        | 0.XXXX       | 0.XXXX
Pixel Accuracy            | 0.XXXX       | 0.XXXX
Sensitivity (Recall)      | 0.XXXX       | 0.XXXX
Specificity               | 0.XXXX       | 0.XXXX
Precision                 | 0.XXXX       | 0.XXXX
```

Results also saved to `outputs/metrics/evaluation_summary.json` and `outputs/metrics/test_metrics_per_sample.csv`.

Quick test:
```bash
python evaluate.py --max-batches 2
```

### Visualize

```bash
python visualize.py --num-samples 5           # first 5 test images
python visualize.py --num-samples 10 --random # 10 randomly picked
python visualize.py --sample-ids 1_A 50_N     # specific cases by filename
```

Each figure has 5 panels saved to `outputs/visualizations/`:

```
[ Original Fundus ] [ Ground Truth ] [ Prediction ] [ Probability Map ] [ Error Map ]
```

The error map uses colour coding:  
🟢 **Green** = correctly detected vessel (TP)  
🔴 **Red** = false alarm, healthy tissue predicted as vessel (FP)  
🔵 **Blue** = missed vessel (FN)

---

## Model Overview

Standard U-Net with 5 encoding levels and 5 decoding levels.

- Input: 512 × 512 RGB fundus image
- Output: 512 × 512 single-channel logit map (sigmoid → probability → threshold → binary mask)
- Parameters: ~31 million
- Skip connections with automatic padding so any input size works, even odd dimensions
- Batch normalisation + spatial dropout in every block

---

## Loss Function

$$\mathcal{L} = 0.5 \times \mathcal{L}_{\text{BCE}} + 0.5 \times \mathcal{L}_{\text{Dice}}$$

- **BCE** ensures pixel-level gradient signal at every training step
- **Soft Dice** directly optimises the overlap metric and naturally handles the imbalance between vessel pixels (~10%) and background pixels (~90%)

---

## Training Details

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1e-4 with ReduceLROnPlateau (halves if val Dice stalls for 3 epochs) |
| Mixed precision | Auto-enabled on CUDA via `torch.amp.GradScaler` |
| Reproducibility | Seed 42 fixed across PyTorch, NumPy, Python random |
| Workers | 0 on Windows (avoids multiprocessing overhead), 2 on Linux/Mac |

---

## Expected Results After Full Training

After 20 epochs on the full FIVES dataset:

| Metric | Expected Range |
|---|---|
| Dice Score | 0.78 – 0.82 |
| IoU | 0.64 – 0.70 |
| Sensitivity | 0.75 – 0.85 |
| Specificity | 0.97 – 0.99 |

> Results will be lower with fewer epochs or on CPU only. The 1-epoch smoke-test checkpoint intentionally produces poor metrics — it's just for verifying the pipeline runs end-to-end.
