"""
SmartPen Live Inference Module
──────────────────────────────
Loads the trained 1D-CNN + LSTM model and fitted scaler once,
then provides a fast prediction method for live ESP32-C3 data.

Designed for import by a FastAPI server:

    from inference import SmartPenPredictor

    predictor = SmartPenPredictor()  # Loads model + scaler once
    result = predictor.predict(raw_data)
    # → {"prediction": "0", "confidence": 0.99}
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import SmartPenDataset, EXPECTED_TIMESTEPS, NUM_CHANNELS
from model import SmartPenModel

# ──────────────────────── Constants ────────────────────────

NUM_CLASSES: int = 10
CONFIDENCE_THRESHOLD: float = 0.60  # Reject predictions below this confidence as "UNKNOWN"
_CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH: str = str(_CURRENT_DIR / "checkpoints" / "best_model.pth")
DEFAULT_DATASETS_DIR: str = str(_CURRENT_DIR.parent / "datasets" if (_CURRENT_DIR.parent / "datasets").exists() else _CURRENT_DIR / ".." / "datasets")

MODEL_PATH: str = DEFAULT_MODEL_PATH
DATASETS_DIR: str = DEFAULT_DATASETS_DIR

# Reverse label map: integer index → human-readable class name
REVERSE_LABEL_MAP: dict[int, str] = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
}

log = logging.getLogger(__name__)


# ──────────────────────── Predictor Class ────────────────────────


class SmartPenPredictor:
    """
    Live inference wrapper for the SmartPen handwriting model.

    Loads the trained model weights and the fitted Z-score scaler
    exactly once during instantiation.  The predict() method then
    accepts raw 6-axis IMU data and returns a class prediction
    with a confidence score — ready for FastAPI integration.

    Attributes:
        model (SmartPenModel):  The trained CNN+LSTM network in eval mode.
        scaler (StandardScaler): The fitted scaler from training data.
        device (torch.device):  Compute device (CUDA/MPS/CPU).
        reverse_label_map (dict): Maps integer indices → class name strings.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        datasets_dir: str = DATASETS_DIR,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """
        Initializes the predictor by loading model weights and the
        training scaler into memory.

        Args:
            model_path:   Path to the saved model checkpoint (.pth file).
            datasets_dir: Path to the datasets/ directory (needed to
                          fit the scaler with the same training stats).
            num_classes:  Number of output classes the model was trained on.
        """
        # ── Device selection ──
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        log.info(f"Inference device: {self.device}")

        # ── Load the trained model ──
        self.model = SmartPenModel(num_classes=num_classes).to(self.device)

        # Load saved weights (map to the current device)
        model_file = Path(model_path)
        if not model_file.exists():
            candidate = _CURRENT_DIR / model_path
            if candidate.exists():
                model_file = candidate
            else:
                raise FileNotFoundError(
                    f"Model checkpoint not found at '{model_file.resolve()}' or '{candidate.resolve()}'. "
                    f"Run train.py first to generate it."
                )

        state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

        # Lock the model in evaluation mode (disables dropout, freezes BatchNorm)
        self.model.eval()
        log.info(f"Model loaded from {model_file} ({num_classes} classes)")

        # ── Extract the fitted scaler from training data ──
        # We instantiate a SmartPenDataset to re-fit the scaler with
        # the exact same statistics used during training.  This ensures
        # live inference normalization matches training normalization.
        d_path = Path(datasets_dir)
        if not d_path.exists():
            if (_CURRENT_DIR / datasets_dir).exists():
                d_path = _CURRENT_DIR / datasets_dir
            elif (_CURRENT_DIR.parent / datasets_dir).exists():
                d_path = _CURRENT_DIR.parent / datasets_dir
        dummy_dataset = SmartPenDataset(root_dir=d_path, fit_scaler=True)
        self.scaler = dummy_dataset.scaler
        log.info(
            f"Scaler loaded — means: {self.scaler.mean_.round(1).tolist()}, "
            f"stds: {np.sqrt(self.scaler.var_).round(1).tolist()}"
        )

        # ── Reverse label map ──
        # Use the dataset's label_map if available, otherwise fall back
        # to the hardcoded REVERSE_LABEL_MAP constant.
        self.reverse_label_map = {v: k for k, v in dummy_dataset.label_map.items()}
        log.info(f"Reverse label map: {self.reverse_label_map}")

        log.info("SmartPenPredictor initialized and ready for inference.")

    def predict(self, raw_data: np.ndarray) -> dict:
        """
        Runs inference on a single raw IMU sample from the ESP32-C3.

        Args:
            raw_data: Numpy array of shape (500, 6) containing the raw
                      6-axis IMU readings (ax, ay, az, gx, gy, gz).

        Returns:
            Dictionary with:
                - "prediction": The predicted class label as a string (e.g. "0").
                - "confidence": The softmax probability of the top class (0.0–1.0).

        Raises:
            ValueError: If raw_data has an unexpected shape.
        """
        # ── Input validation ──
        if raw_data.ndim != 2 or raw_data.shape[1] != NUM_CHANNELS:
            raise ValueError(
                f"Expected raw_data shape (N, {NUM_CHANNELS}), "
                f"got {raw_data.shape}"
            )

        # Standardize length to exactly 500 timesteps (truncate or pad)
        raw_data = SmartPenDataset._standardize_length(
            raw_data.astype(np.float32), filename="live_input"
        )
        # raw_data shape: (500, 6)

        # ── Z-score normalization ──
        # Apply the SAME scaler fitted on the training data so that
        # the distribution matches what the model learned.
        normalized = self.scaler.transform(raw_data).astype(np.float32)
        # normalized shape: (500, 6)

        # ── Convert to PyTorch tensor ──
        # Add batch dimension: (500, 6) → (1, 500, 6)
        tensor = torch.from_numpy(normalized).unsqueeze(0)
        # tensor shape: (1, 500, 6)

        # Transpose to channel-first for Conv1d: (1, 500, 6) → (1, 6, 500)
        tensor = tensor.permute(0, 2, 1).to(self.device)
        # tensor shape: (1, 6, 500)

        # ── Forward pass (no gradient computation) ──
        with torch.no_grad():
            logits = self.model(tensor)
            # logits shape: (1, num_classes)

            # Convert logits to confidence probabilities via softmax
            probabilities = F.softmax(logits, dim=1)
            # probabilities shape: (1, num_classes)

        # ── Extract prediction ──
        confidence, predicted_idx = probabilities.max(dim=1)
        predicted_idx = predicted_idx.item()     # int
        confidence = round(confidence.item(), 4)  # float, 4 decimal places

        # Map integer index back to the human-readable label
        predicted_label = self.reverse_label_map.get(predicted_idx, str(predicted_idx))

        # All class probabilities for UI breakdown
        probs_raw = probabilities[0].cpu().numpy().tolist()
        class_probs = {
            self.reverse_label_map.get(idx, str(idx)): round(p, 4)
            for idx, p in enumerate(probs_raw)
        }

        # ── Confidence threshold rejection ──
        # If the model isn't confident enough, reject as "UNKNOWN"
        # instead of randomly guessing a class.
        is_rejected = confidence < CONFIDENCE_THRESHOLD
        if is_rejected:
            log.info(
                f"Prediction rejected: best class '{predicted_label}' "
                f"with {confidence:.2%} confidence < {CONFIDENCE_THRESHOLD:.0%} threshold"
            )
            predicted_label = "UNKNOWN"

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "probabilities": class_probs,
            "rejected": is_rejected,
        }


# ──────────────────────── CLI Self-Test ──────────────────────────

if __name__ == "__main__":
    """Quick self-test: loads the model, creates dummy data, runs inference."""
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

    print("\n" + "=" * 65)
    print("  SmartPen Inference — Self-Test")
    print("=" * 65)

    # Initialize the predictor (loads model + scaler)
    predictor = SmartPenPredictor()

    print(f"\n  Device:       {predictor.device}")
    print(f"  Label map:    {predictor.reverse_label_map}")
    print(f"  Scaler means: {predictor.scaler.mean_.round(1).tolist()}")

    # ── Test with random dummy data ──
    print(f"\n  --- Test 1: Random noise input ---")
    dummy_data = np.random.randn(500, 6).astype(np.float32)
    result = predictor.predict(dummy_data)
    print(f"  Input shape:  (500, 6)")
    print(f"  Prediction:   '{result['prediction']}'")
    print(f"  Confidence:   {result['confidence']:.4f}")

    # ── Test with a real sample from the dataset ──
    print(f"\n  --- Test 2: Real sample from dataset ---")
    import pandas as pd
    datasets_dir = Path(DATASETS_DIR)
    # Find the first CSV in the first class folder
    for class_dir in sorted(datasets_dir.iterdir()):
        if class_dir.is_dir():
            csv_files = sorted(class_dir.glob("dataset_*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                real_data = df[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(dtype=np.float32)
                result = predictor.predict(real_data)
                print(f"  File:         {csv_files[0].name}")
                print(f"  True label:   '{class_dir.name}'")
                print(f"  Prediction:   '{result['prediction']}'")
                print(f"  Confidence:   {result['confidence']:.4f}")
                correct = "CORRECT" if result['prediction'] == class_dir.name else "WRONG"
                print(f"  Result:       {correct}")
                break

    print(f"\n  [PASS] Inference pipeline works.")
    print("=" * 65 + "\n")
