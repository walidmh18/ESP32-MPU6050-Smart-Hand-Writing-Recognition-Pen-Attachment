
"""
SmartPen Model — 1D-CNN + LSTM Hybrid Architecture
───────────────────────────────────────────────────
Classifies 6-axis IMU time-series (accelerometer + gyroscope) into
handwritten characters/digits captured at 500 Hz for 1.0 second.

Input:  (Batch, 6, 500)   — 6 channels, 500 timesteps
Output: (Batch, Num_Classes) — raw unnormalized logits

Architecture:
    CNN Feature Extractor  →  LSTM Temporal Modeler  →  FC Classifier

Usage:
    from model import SmartPenModel

    model = SmartPenModel(num_classes=5)
    logits = model(x)  # x: (B, 6, 500) → logits: (B, 5)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmartPenModel(nn.Module):
    """
    1D-CNN + LSTM hybrid for IMU-based handwriting recognition.

    CNN blocks extract local spatial-temporal features from the raw
    6-axis signal, aggressively downsampling the 500-step sequence.
    The LSTM then captures long-range temporal dependencies across
    the compressed feature map.  Only the final LSTM hidden state
    is used for classification, representing the completed stroke.

    Args:
        num_classes:    Number of output classes (e.g. 5 for '0','1','2','A','V').
        lstm_hidden:    Hidden size of the LSTM layers.
        lstm_layers:    Number of stacked LSTM layers.
        dropout:        Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # ──────────────────── CNN Feature Extractor ────────────────────
        # Three convolutional blocks progressively expand channels while
        # aggressively downsampling the time dimension:
        #   500 → 125 → 31 → 7  (timesteps)
        #   6   → 32  → 64 → 128 (channels)

        # Block 1: 6 → 32 channels, 500 → 125 timesteps
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 500 → 125
        )

        # Block 2: 32 → 64 channels, 125 → 31 timesteps
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 125 → 31
        )

        # Block 3: 64 → 128 channels, 31 → 7 timesteps
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 31 → 7
        )

        # ──────────────────── LSTM Temporal Modeler ────────────────────
        # Processes the CNN feature sequence to capture long-range
        # temporal dependencies across the compressed timesteps.
        self.lstm = nn.LSTM(
            input_size=128,           # Matches conv_block3 output channels
            hidden_size=lstm_hidden,  # 128
            num_layers=lstm_layers,   # 2 stacked layers
            batch_first=True,         # Expects (Batch, Seq, Features)
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ──────────────────── Classifier Head ────────────────────
        # Takes the final LSTM hidden state and maps it to class logits.
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full CNN → LSTM → Classifier pipeline.

        Args:
            x: Input tensor of shape (Batch, 6, 500).

        Returns:
            Raw unnormalized logits of shape (Batch, num_classes).
        """
        # ── CNN Feature Extraction ──
        # x shape: (B, 6, 500)
        x = self.conv_block1(x)
        # x shape: (B, 32, 125)

        x = self.conv_block2(x)
        # x shape: (B, 64, 31)

        x = self.conv_block3(x)
        # x shape: (B, 128, 7)

        # ── Dimensionality Bridge: CNN → LSTM ──
        # Conv1d outputs (Batch, Channels, Length)
        # LSTM expects   (Batch, Length, Channels)  with batch_first=True
        x = x.permute(0, 2, 1)
        # x shape: (B, 7, 128)

        # ── LSTM Temporal Modeling ──
        # lstm_out contains all hidden states: (B, 7, lstm_hidden)
        # We only need the final timestep's hidden state.
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (B, 7, 128)
        # h_n shape:      (num_layers, B, 128)

        # Extract the final hidden state from the last LSTM layer
        final_hidden = h_n[-1]
        # final_hidden shape: (B, 128)

        # ── Classification ──
        logits = self.classifier(final_hidden)
        # logits shape: (B, num_classes)

        return logits


# ──────────────────────── CLI Entrypoint ──────────────────────────

if __name__ == "__main__":
    """Quick self-test: builds the model, runs a dummy forward pass,
    and prints architecture details + tensor shape verification."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    NUM_CLASSES = 5  # Example: '0', '1', '2', 'A', 'V'
    BATCH_SIZE = 8

    print("\n" + "=" * 65)
    print("  SmartPen Model — Architecture Self-Test")
    print("=" * 65)

    # Build model
    model = SmartPenModel(num_classes=NUM_CLASSES)
    print(f"\n  Model architecture:\n")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Dummy forward pass
    dummy_input = torch.randn(BATCH_SIZE, 6, 500)
    print(f"\n  Input shape:  {dummy_input.shape}  (batch={BATCH_SIZE}, channels=6, timesteps=500)")

    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"  Output shape: {logits.shape}  (batch={BATCH_SIZE}, classes={NUM_CLASSES})")
    print(f"  Output dtype: {logits.dtype}")
    print(f"  Output range: [{logits.min():.4f}, {logits.max():.4f}]")

    # Shape assertions
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Expected ({BATCH_SIZE}, {NUM_CLASSES}), got {logits.shape}"
    )
    assert logits.dtype == torch.float32, f"Expected float32, got {logits.dtype}"

    print(f"\n  [PASS] All shape checks passed.")
    print("=" * 65 + "\n")
