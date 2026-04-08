# Project 01 — MLP for MNIST

> **EEG Deep Learning Series | Warm-up #1**
> A clean PyTorch MLP baseline on MNIST — the foundation before we touch brain signals.

---

## Why This Exists

This is the first entry in a project series building toward deep learning–based EEG decoding. Before processing noisy, high-dimensional neural signals, this project establishes the core training loop, experiment tracking, and code structure that every later project will inherit.

---

## Architecture

A three-layer MLP with batch normalization and dropout at each hidden layer.

```
Input (784)  →  Linear → ReLU → BN → Dropout
             →  Linear → ReLU → BN → Dropout
             →  Linear → Softmax
                Output (10 classes)
```

| Layer       | In  | Out | Notes                    |
|-------------|-----|-----|--------------------------|
| Hidden 1    | 784 | 256 | ReLU + BN + Dropout(0.3) |
| Hidden 2    | 256 | 128 | ReLU + BN + Dropout(0.3) |
| Output      | 128 | 10  | CrossEntropyLoss         |

---

## Training Setup

| Hyperparameter | Value         |
|----------------|---------------|
| Optimizer      | Adam          |
| Learning rate  | 1e-3          |
| LR schedule    | StepLR (×0.5 every 5 epochs) |
| Epochs         | 10            |
| Batch size     | 32            |
| Dropout        | 0.3           |
| Grad clipping  | max norm 1.0  |

Loss: `CrossEntropyLoss`
Device: MPS (Apple Silicon) → CPU fallback

---

## Project Structure

```
project-01-mlp-mnist/
├── src/
│   ├── config.py      # Device detection + CONFIG dict
│   ├── data.py        # MNIST dataloaders (train / val)
│   └── model.py       # MLP class + instantiated model
├── train.py           # Full training loop with W&B logging
├── best_model.pt      # Saved on best val accuracy
└── README.md
```

---

## Experiment Tracking

Training is logged to [Weights & Biases](https://wandb.ai) under project `mnist`, run name `mlp_baseline`.

Logged metrics:
- `train/batch_loss` — every 100 steps
- `train/loss`, `train/accuracy` — per epoch
- `val/loss`, `val/accuracy` — per epoch
- `lr` — learning rate at each epoch

Model gradients are also watched (`log='gradients'`, every 100 steps).

---

## Key Implementation Notes

**Why `loss.item() * images.size(0)` when accumulating loss?**
`CrossEntropyLoss` returns the *mean* loss over a batch. To correctly compute the epoch-level average across batches of potentially different sizes, we undo the mean by multiplying back by batch size, then divide by total samples at the end.

**Why `x.view(x.size(0), -1)` in `forward`?**
MNIST images arrive as `(B, 1, 28, 28)`. The MLP expects a flat vector, so we reshape to `(B, 784)` while keeping the batch dimension intact.

**Why BN before Dropout?**
Batch normalization stabilizes activations; dropout then regularizes. Reversing the order can degrade BN's statistics since dropout creates artificial zeros.

---

---

## Requirements

```bash
pip install torch torchvision wandb
```

## Run

```bash
python train.py
```
