"""
SmartPen IMU Dataset — PyTorch Data Pipeline
─────────────────────────────────────────────
Handles loading, preprocessing, and standardizing 6-axis IMU time-series
data for a 1D-CNN + LSTM hybrid character recognition model.

Data layout on disk:
    datasets/<label>/dataset_<label>_<id>.csv

Each CSV has columns: ax, ay, az, gx, gy, gz
Each file should contain exactly 500 rows (1.0 s @ 500 Hz).

Usage:
    from dataset import get_dataloaders

    train_loader, val_loader, scaler = get_dataloaders(
        root_dir="../datasets",
        batch_size=32,
        val_split=0.2,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

# ──────────────────────── Constants ────────────────────────

EXPECTED_TIMESTEPS: int = 500       # 500 samples @ 500 Hz = 1.0 second window
NUM_CHANNELS: int = 6               # ax, ay, az, gx, gy, gz
COLUMN_NAMES: list[str] = ["ax", "ay", "az", "gx", "gy", "gz"]
REPRODUCIBILITY_SEED: int = 42      # Fixed seed for train/val split

log = logging.getLogger(__name__)


# ──────────────────────── Dataset Class ────────────────────────

class SmartPenDataset(Dataset):
    """
    PyTorch Dataset for SmartPen 6-axis IMU recordings.

    Loads all CSV files from `root_dir/<label>/dataset_<label>_<id>.csv`,
    enforces a strict (500, 6) shape per sample, applies global Z-score
    normalization, and transposes to (6, 500) for Conv1d compatibility.

    Attributes:
        data (np.ndarray):    Normalized array of shape (N, 500, 6).
        labels (np.ndarray):  Integer class labels of shape (N,).
        label_map (dict):     Mapping from string label → integer index.
        scaler (StandardScaler): Fitted scaler (needed for live inference).
    """

    def __init__(
        self,
        root_dir: str | Path,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
    ) -> None:
        """
        Args:
            root_dir:   Path to the datasets/ directory containing class subfolders.
            scaler:     Pre-fitted StandardScaler to use (for inference). If None and
                        fit_scaler=True, a new scaler is fit on this data.
            fit_scaler: Whether to fit the scaler on this data. Set False when using
                        a pre-fitted scaler for test/inference data.
        """
        self.root_dir = Path(root_dir)
        self.scaler = scaler if scaler is not None else StandardScaler()

        # ── Step 1: Discover class folders and build label map ──
        self.label_map = self._build_label_map()
        if not self.label_map:
            raise FileNotFoundError(
                f"No class folders with CSV files found in {self.root_dir}. "
                f"Expected structure: {self.root_dir}/<label>/dataset_<label>_<id>.csv"
            )
        log.info(f"Label map: {self.label_map}")

        # ── Step 2: Load all CSV files into memory ──
        raw_samples, raw_labels = self._load_all_samples()
        log.info(
            f"Loaded {len(raw_samples)} samples across "
            f"{len(self.label_map)} classes from {self.root_dir}"
        )

        # ── Step 3: Stack into a single array of shape (N, 500, 6) ──
        self.data = np.stack(raw_samples, axis=0).astype(np.float32)  # (N, 500, 6)
        self.labels = np.array(raw_labels, dtype=np.int64)            # (N,)

        # ── Step 4: Global Z-score normalization ──
        n_samples = self.data.shape[0]
        # Flatten to (N*500, 6) so each channel gets its own mean/std
        flat = self.data.reshape(-1, NUM_CHANNELS)  # (N*500, 6)

        if fit_scaler:
            self.scaler.fit(flat)
            log.info(
                f"Scaler fitted — means: {self.scaler.mean_.round(1).tolist()}, "
                f"stds: {np.sqrt(self.scaler.var_).round(1).tolist()}"
            )

        flat_normalized = self.scaler.transform(flat)
        self.data = flat_normalized.reshape(n_samples, EXPECTED_TIMESTEPS, NUM_CHANNELS)
        self.data = self.data.astype(np.float32)

    # ──────────────── Dataset Protocol ────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Float tensor of shape (6, 500) — channels-first for Conv1d.
            y: Long tensor scalar — integer class label.
        """
        # Transpose from (500, 6) → (6, 500) for PyTorch Conv1d
        x = torch.from_numpy(self.data[idx]).T       # (6, 500)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    # ──────────────── Internal Methods ────────────────

    def _build_label_map(self) -> dict[str, int]:
        """
        Scans subdirectories of root_dir and assigns an integer index to each
        label that contains at least one CSV file. Sorted for deterministic
        ordering: digits first (ascending), then uppercase letters (ascending).
        """
        class_dirs: list[str] = []
        if not self.root_dir.exists():
            return {}

        for d in sorted(self.root_dir.iterdir()):
            if d.is_dir() and any(d.glob("dataset_*.csv")):
                class_dirs.append(d.name)

        # Sort: digits before letters, both ascending
        digits = sorted([c for c in class_dirs if c.isdigit()])
        letters = sorted([c for c in class_dirs if c.isalpha()])
        others = sorted([c for c in class_dirs if c not in digits and c not in letters])
        ordered = digits + letters + others

        return {label: idx for idx, label in enumerate(ordered)}

    def _load_all_samples(self) -> tuple[list[np.ndarray], list[int]]:
        """
        Reads every CSV from each class subfolder, standardizes length to
        exactly EXPECTED_TIMESTEPS (500) rows, and returns raw (unnormalized)
        arrays and their corresponding integer labels.
        """
        samples: list[np.ndarray] = []
        labels: list[int] = []

        for label_str, label_int in self.label_map.items():
            class_dir = self.root_dir / label_str
            csv_files = sorted(class_dir.glob("dataset_*.csv"))

            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path)

                    # Validate columns
                    if not all(col in df.columns for col in COLUMN_NAMES):
                        log.warning(
                            f"Skipping {csv_path.name}: missing required columns. "
                            f"Expected {COLUMN_NAMES}, got {df.columns.tolist()}"
                        )
                        continue

                    data = df[COLUMN_NAMES].to_numpy(dtype=np.float32)

                    # ── Length standardization: exactly 500 timesteps ──
                    data = self._standardize_length(data, csv_path.name)

                    samples.append(data)
                    labels.append(label_int)

                except Exception as e:
                    log.warning(f"Failed to load {csv_path}: {e}")

        return samples, labels

    @staticmethod
    def _standardize_length(data: np.ndarray, filename: str = "") -> np.ndarray:
        """
        Enforces shape (EXPECTED_TIMESTEPS, NUM_CHANNELS) = (500, 6).
        - Truncates if too long.
        - Zero-pads at the end if too short.
        """
        n_rows = data.shape[0]

        if n_rows == EXPECTED_TIMESTEPS:
            return data

        if n_rows > EXPECTED_TIMESTEPS:
            log.debug(f"{filename}: truncated {n_rows} → {EXPECTED_TIMESTEPS} rows")
            return data[:EXPECTED_TIMESTEPS]

        # Zero-pad at the end
        pad_rows = EXPECTED_TIMESTEPS - n_rows
        padding = np.zeros((pad_rows, NUM_CHANNELS), dtype=np.float32)
        log.debug(f"{filename}: zero-padded {n_rows} → {EXPECTED_TIMESTEPS} rows")
        return np.vstack([data, padding])

    # ──────────────── Utility Methods ────────────────

    def get_class_distribution(self) -> dict[str, int]:
        """Returns a dict mapping label string → sample count."""
        inv_map = {v: k for k, v in self.label_map.items()}
        unique, counts = np.unique(self.labels, return_counts=True)
        return {inv_map[int(u)]: int(c) for u, c in zip(unique, counts)}

    def __repr__(self) -> str:
        dist = self.get_class_distribution()
        dist_str = ", ".join(f"'{k}': {v}" for k, v in dist.items())
        return (
            f"SmartPenDataset("
            f"samples={len(self)}, "
            f"shape=({EXPECTED_TIMESTEPS}, {NUM_CHANNELS}), "
            f"classes={len(self.label_map)}, "
            f"distribution={{{dist_str}}})"
        )


# ──────────────────── DataLoader Helper ────────────────────

def get_dataloaders(
    root_dir: str | Path = "datasets",
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Creates train and validation DataLoaders with a reproducible random split.

    Args:
        root_dir:    Path to the datasets/ directory.
        batch_size:  Mini-batch size for both loaders.
        val_split:   Fraction of data reserved for validation (0.0 – 1.0).
        num_workers: Number of subprocess workers for data loading.
        pin_memory:  Pin memory for faster GPU transfer (set True if using CUDA).

    Returns:
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        scaler:       The fitted StandardScaler (save this for live inference).
    """
    # Load and normalize the entire dataset
    dataset = SmartPenDataset(root_dir=root_dir, fit_scaler=True)
    log.info(f"Dataset loaded: {dataset}")

    # Compute split sizes
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Dataset too small ({total} samples) for val_split={val_split}. "
            f"Need at least 2 samples. Train={train_size}, Val={val_size}."
        )

    # Reproducible split with a fixed seed
    generator = torch.Generator().manual_seed(REPRODUCIBILITY_SEED)
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    log.info(f"Split: {train_size} train / {val_size} val (seed={REPRODUCIBILITY_SEED})")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, dataset.scaler


# ──────────────────────── CLI Entrypoint ──────────────────────────

if __name__ == "__main__":
    """Quick self-test: loads the dataset, prints stats, and verifies tensor shapes."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve the datasets directory relative to this script's parent
    script_dir = Path(__file__).resolve().parent
    datasets_dir = script_dir.parent / "datasets"

    print("\n" + "=" * 65)
    print("  SmartPen Dataset Pipeline — Self-Test")
    print("=" * 65)
    print(f"  Datasets root: {datasets_dir}")

    try:
        train_loader, val_loader, scaler = get_dataloaders(
            root_dir=datasets_dir,
            batch_size=16,
            val_split=0.2,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)

    # Verify tensor shapes from a single batch
    x_batch, y_batch = next(iter(train_loader))
    print(f"\n  Train batches:  {len(train_loader)}")
    print(f"  Val batches:    {len(val_loader)}")
    print(f"  Batch X shape:  {x_batch.shape}   (batch, channels=6, timesteps=500)")
    print(f"  Batch Y shape:  {y_batch.shape}   (batch,)")
    print(f"  X dtype:        {x_batch.dtype}")
    print(f"  Y dtype:        {y_batch.dtype}")
    print(f"  X range:        [{x_batch.min():.3f}, {x_batch.max():.3f}]")
    print(f"  Labels in batch: {y_batch.tolist()}")

    # Scaler stats
    print(f"\n  Scaler means:   {scaler.mean_.round(1).tolist()}")
    print(f"  Scaler stds:    {np.sqrt(scaler.var_).round(1).tolist()}")

    # Class distribution
    dataset = train_loader.dataset.dataset  # type: ignore[attr-defined]
    dist = dataset.get_class_distribution()
    print(f"\n  Class distribution:")
    for label, count in dist.items():
        print(f"    '{label}': {count} samples")

    print("\n  [PASS] All checks passed.")
    print("=" * 65 + "\n")
