"""
SmartPen Single File Predictor
──────────────────────────────
Usage:
    python predict.py ../datasets/0/dataset_0_1.csv
    python predict.py ../datasets/1/dataset_1_5.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from inference import SmartPenPredictor

COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # Load CSV
    df = pd.read_csv(csv_path)
    raw = df[COLUMNS].to_numpy(dtype=np.float32)

    # Predict
    predictor = SmartPenPredictor()
    result = predictor.predict(raw)

    print(f"\n  File:       {csv_path.name}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}\n")


if __name__ == "__main__":
    main()
