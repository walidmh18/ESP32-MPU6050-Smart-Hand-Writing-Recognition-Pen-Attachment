"""
SmartPen Training Script
────────────────────────
Trains the SmartPenModel (1D-CNN + LSTM) on 6-axis IMU data for
air-writing character recognition.

Usage:
    cd ML/
    python train.py

Requires:
    - dataset.py  (exports get_dataloaders)
    - model.py    (exports SmartPenModel)
    - ../datasets/<label>/dataset_<label>_<id>.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import SmartPenModel

# ──────────────────────── Hyperparameters ────────────────────────

BATCH_SIZE: int = 16
EPOCHS: int = 50
LEARNING_RATE: float = 0.001
NUM_CLASSES: int = 2
VAL_SPLIT: float = 0.2

DATASETS_DIR: str = "../datasets"
CHECKPOINT_DIR: str = "checkpoints"
BEST_MODEL_PATH: str = f"{CHECKPOINT_DIR}/best_model.pth"

# ──────────────────────── Device Setup ────────────────────────


def get_device() -> torch.device:
    """Select the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────── Training Step ────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Runs one full training epoch.

    Returns:
        Average training loss over all batches.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)  # (B, 6, 500)
        y_batch = y_batch.to(device)  # (B,)

        # Forward pass
        logits = model(x_batch)       # (B, NUM_CLASSES)
        loss = criterion(logits, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


# ──────────────────────── Validation Step ────────────────────────


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Runs one full validation pass.

    Returns:
        (average_val_loss, accuracy_percentage)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = (correct / max(total, 1)) * 100.0
    return avg_loss, accuracy


# ──────────────────────── Main ────────────────────────


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    device = get_device()

    print("\n" + "=" * 65)
    print("  SmartPen Training Pipeline")
    print("=" * 65)
    print(f"  Device:         {device}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Learning rate:  {LEARNING_RATE}")
    print(f"  Num classes:    {NUM_CLASSES}")
    print(f"  Val split:      {VAL_SPLIT}")
    print(f"  Datasets dir:   {DATASETS_DIR}")
    print("=" * 65)

    # ── Data ──
    train_loader, val_loader, scaler = get_dataloaders(
        root_dir=DATASETS_DIR,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
    )
    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # ── Model, Loss, Optimizer ──
    model = SmartPenModel(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:    {total_params:,}\n")

    # ── Checkpoint directory ──
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ──
    best_val_loss = float("inf")

    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val Acc':>8}  {'Status':>10}")
    print(f"  {'─' * 5}  {'─' * 11}  {'─' * 9}  {'─' * 8}  {'─' * 10}")

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Checkpointing: save if validation loss improved
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            status = "* saved"

        print(
            f"  {epoch:>5}  "
            f"{train_loss:>11.6f}  "
            f"{val_loss:>9.6f}  "
            f"{val_acc:>7.2f}%  "
            f"{status:>10}"
        )

    elapsed = time.time() - start_time

    print(f"\n  {'─' * 50}")
    print(f"  Training complete in {elapsed:.1f}s ({elapsed / EPOCHS:.2f}s/epoch)")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Best model saved to:  {BEST_MODEL_PATH}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
