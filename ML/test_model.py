"""
SmartPen Test Script
────────────────────
Tests the trained model against existing CSV files from the dataset.
Runs inference on every sample and reports per-class and overall accuracy.

Usage:
    cd ML/
    python test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from inference import SmartPenPredictor

DATASETS_DIR = Path("../datasets")
COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("  SmartPen Model — Full Dataset Test")
    print("=" * 65)

    # Load model + scaler once
    predictor = SmartPenPredictor()

    total_correct = 0
    total_samples = 0

    # Iterate through each class folder
    for class_dir in sorted(DATASETS_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        csv_files = sorted(class_dir.glob("dataset_*.csv"))
        if not csv_files:
            continue

        true_label = class_dir.name
        class_correct = 0

        print(f"\n  Class '{true_label}' ({len(csv_files)} samples)")
        print(f"  {'File':<25} {'Predicted':>10} {'Confidence':>11} {'Result':>8}")
        print(f"  {'─' * 25} {'─' * 10} {'─' * 11} {'─' * 8}")

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            raw = df[COLUMNS].to_numpy(dtype=np.float32)

            result = predictor.predict(raw)
            pred = result["prediction"]
            conf = result["confidence"]
            is_correct = pred == true_label

            status = "OK" if is_correct else "MISS"
            class_correct += int(is_correct)
            total_correct += int(is_correct)
            total_samples += 1

            print(
                f"  {csv_path.name:<25} "
                f"{pred:>10} "
                f"{conf:>10.4f} "
                f"{'  ' + status:>8}"
            )

        class_acc = (class_correct / len(csv_files)) * 100
        print(f"  Class '{true_label}' accuracy: {class_correct}/{len(csv_files)} ({class_acc:.1f}%)")

    # Overall summary
    overall_acc = (total_correct / max(total_samples, 1)) * 100
    print(f"\n  {'─' * 50}")
    print(f"  Overall: {total_correct}/{total_samples} correct ({overall_acc:.1f}%)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
