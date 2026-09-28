"""
SmartPen Handwriting Recognition — Data Quality Verification & Waveform Comparison
──────────────────────────────────────────────────────────────────────────────────
Verifies data quality before training neural networks:
  1. Loads 500-sample (1.0s @ 500Hz) 6-axis IMU recordings (ax, ay, az, gx, gy, gz).
  2. Compares multiple samples of digit "0" against letter "A" (or "1" as fallback).
  3. Plots 6-axis waveforms (individual samples + mean ± std envelope).
  4. Plots rotational & acceleration magnitudes (movement energy profiles).
  5. Computes quantitative repeatability (intra-class) and separability (inter-class).
  6. Saves high-resolution figures to ML/plots/ and prints a data health report.

Usage:
  python ML/verify_data_quality.py
  python ML/verify_data_quality.py --class1 0 --class2 A
  python ML/verify_data_quality.py --class1 0 --class2 1 --num-samples 5
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────── Configuration ──────────────────────────
DATASET_DIR = Path("datasets")
OUTPUT_PLOT_DIR = Path("ML/plots")
SAMPLE_RATE_HZ = 500.0            # 500 Hz sampling rate
EXPECTED_SAMPLES = 500            # Exactly 500 samples per capture (1.0 sec)
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
CHANNEL_LABELS = {
    "ax": "Accel X (LSB)",
    "ay": "Accel Y (LSB)",
    "az": "Accel Z (LSB)",
    "gx": "Gyro X (LSB - bias)",
    "gy": "Gyro Y (LSB - bias)",
    "gz": "Gyro Z (LSB - bias)",
}

# ────────────────────────── Data Loading ──────────────────────────
def find_available_classes() -> list[str]:
    """Finds all class subdirectories in datasets/ that contain at least one CSV."""
    if not DATASET_DIR.exists():
        return []
    classes = []
    for d in sorted(DATASET_DIR.iterdir()):
        if d.is_dir() and list(d.glob("*.csv")):
            classes.append(d.name)
    return classes


def load_class_samples(label: str, max_samples: int = 10) -> tuple[list[np.ndarray], list[str]]:
    """
    Loads up to `max_samples` 500-sample CSV files for a given class label.
    Returns:
        samples: list of np.ndarray of shape (500, 6)
        filenames: list of file names loaded
    """
    class_dir = DATASET_DIR / label
    csv_files = sorted(class_dir.glob("dataset_*.csv")) if class_dir.exists() else []
    
    # Sort by numeric ID if matching dataset_<label>_<id>.csv
    def sort_key(p: Path):
        stem = p.stem
        parts = stem.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            return int(parts[-1])
        return 99999
    csv_files = sorted(csv_files, key=sort_key)

    samples = []
    filenames = []

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Ensure required columns exist
            if not all(col in df.columns for col in CHANNELS):
                continue
            data = df[CHANNELS].to_numpy(dtype=np.float32)
            if data.shape[0] == EXPECTED_SAMPLES:
                samples.append(data)
                filenames.append(f.name)
            elif data.shape[0] > 0:
                # If slightly off, truncate or pad to 500
                if data.shape[0] >= EXPECTED_SAMPLES:
                    samples.append(data[:EXPECTED_SAMPLES])
                else:
                    padded = np.pad(data, ((0, EXPECTED_SAMPLES - data.shape[0]), (0, 0)), mode="edge")
                    samples.append(padded)
                filenames.append(f"{f.name} (resized)")
            if len(samples) >= max_samples:
                break
        except Exception as e:
            print(f"  [Warning] Could not load {f.name}: {e}")

    # Fallback: check if legacy root file exists (e.g., dataset_letter_A.csv)
    if len(samples) == 0:
        legacy_file = Path(f"dataset_letter_{label}.csv")
        if legacy_file.exists():
            try:
                df = pd.read_csv(legacy_file)
                if all(col in df.columns for col in CHANNELS):
                    data = df[CHANNELS].to_numpy(dtype=np.float32)
                    num_windows = len(data) // EXPECTED_SAMPLES
                    for w in range(min(num_windows, max_samples)):
                        win_data = data[w * EXPECTED_SAMPLES : (w + 1) * EXPECTED_SAMPLES]
                        samples.append(win_data)
                        filenames.append(f"{legacy_file.name} [window {w+1}]")
            except Exception as e:
                print(f"  [Warning] Could not read legacy file {legacy_file}: {e}")

    return samples, filenames


# ────────────────────── Metrics & Health Check ──────────────────────
def analyze_data_quality(class1_name: str, class1_data: list[np.ndarray],
                         class2_name: str, class2_data: list[np.ndarray]) -> dict:
    """Computes repeatability and discriminability statistics."""
    report = {}

    def compute_intra_correlation(samples: list[np.ndarray]) -> float:
        if len(samples) < 2:
            return 0.0
        corrs = []
        for i in range(min(len(samples), 6)):
            for j in range(i + 1, min(len(samples), 6)):
                # Gyro magnitude or Gyro Z (yaw) correlation
                g1 = samples[i][:, 5]  # gz
                g2 = samples[j][:, 5]
                s1, s2 = np.std(g1), np.std(g2)
                if s1 > 1e-4 and s2 > 1e-4:
                    c = np.corrcoef(g1, g2)[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0

    def compute_inter_correlation(samples1: list[np.ndarray], samples2: list[np.ndarray]) -> float:
        if not samples1 or not samples2:
            return 0.0
        corrs = []
        for s1 in samples1[:4]:
            for s2 in samples2[:4]:
                g1 = s1[:, 5]
                g2 = s2[:, 5]
                s1_std, s2_std = np.std(g1), np.std(g2)
                if s1_std > 1e-4 and s2_std > 1e-4:
                    c = np.corrcoef(g1, g2)[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0

    report["c1_intra_corr"] = compute_intra_correlation(class1_data)
    report["c2_intra_corr"] = compute_intra_correlation(class2_data)
    report["inter_corr"] = compute_inter_correlation(class1_data, class2_data)

    # Check dynamic range (peak-to-peak)
    def avg_dynamic_range(samples: list[np.ndarray]) -> dict:
        if not samples:
            return {"accel_ptp": 0.0, "gyro_ptp": 0.0}
        accel_ptps = [np.ptp(s[:, :3]) for s in samples]
        gyro_ptps = [np.ptp(s[:, 3:]) for s in samples]
        return {
            "accel_ptp": float(np.mean(accel_ptps)),
            "gyro_ptp": float(np.mean(gyro_ptps))
        }

    report["c1_ptp"] = avg_dynamic_range(class1_data)
    report["c2_ptp"] = avg_dynamic_range(class2_data)

    return report


# ────────────────────────── Plotting ──────────────────────────
def plot_6axis_comparison(class1_name: str, class1_samples: list[np.ndarray],
                          class2_name: str, class2_samples: list[np.ndarray],
                          num_to_plot: int = 4, save_path: Path | None = None):
    """
    Plots a 6-row comparison grid (ax, ay, az, gx, gy, gz) showing individual sample
    trajectories and mean profiles for both classes side-by-side.
    """
    time_axis = np.linspace(0.0, EXPECTED_SAMPLES / SAMPLE_RATE_HZ, EXPECTED_SAMPLES)

    fig, axes = plt.subplots(6, 2, figsize=(16, 14), sharex=True)
    fig.patch.set_facecolor("#0f172a")  # Sleek dark background

    # Colors
    c1_line_color = "#38bdf8"       # Sky Blue for Class 1
    c1_mean_color = "#0284c7"
    c2_line_color = "#f43f5e"       # Rose / Red for Class 2
    c2_mean_color = "#be123c"
    grid_color = "#1e293b"
    text_color = "#f8fafc"

    fig.suptitle(
        f"SmartPen 6-Axis Waveform Verification: '{class1_name}' vs '{class2_name}'\n"
        f"Window: {EXPECTED_SAMPLES} samples @ {int(SAMPLE_RATE_HZ)} Hz (1.0 Second)",
        fontsize=16, fontweight="bold", color=text_color, y=0.98
    )

    c1_arr = np.array(class1_samples[:num_to_plot])  # (N, 500, 6)
    c2_arr = np.array(class2_samples[:num_to_plot])

    c1_mean = np.mean(c1_arr, axis=0) if len(c1_arr) > 0 else np.zeros((EXPECTED_SAMPLES, 6))
    c1_std = np.std(c1_arr, axis=0) if len(c1_arr) > 0 else np.zeros((EXPECTED_SAMPLES, 6))
    c2_mean = np.mean(c2_arr, axis=0) if len(c2_arr) > 0 else np.zeros((EXPECTED_SAMPLES, 6))
    c2_std = np.std(c2_arr, axis=0) if len(c2_arr) > 0 else np.zeros((EXPECTED_SAMPLES, 6))

    for ch_idx, col_name in enumerate(CHANNELS):
        # Column 0: Class 1
        ax_left = axes[ch_idx, 0]
        ax_left.set_facecolor("#1e293b")
        ax_left.grid(True, color="#334155", linestyle="--", alpha=0.6)

        # Plot individual samples
        for s_idx in range(len(c1_arr)):
            ax_left.plot(time_axis, c1_arr[s_idx, :, ch_idx], color=c1_line_color,
                         alpha=0.35, linewidth=1.2, label=f"Sample {s_idx+1}" if ch_idx == 0 else "")

        # Plot mean & std envelope
        ax_left.plot(time_axis, c1_mean[:, ch_idx], color=c1_line_color, linewidth=2.5,
                     label="Mean Pattern" if ch_idx == 0 else "")
        ax_left.fill_between(time_axis, c1_mean[:, ch_idx] - c1_std[:, ch_idx],
                             c1_mean[:, ch_idx] + c1_std[:, ch_idx], color=c1_line_color, alpha=0.15)

        ax_left.set_ylabel(CHANNEL_LABELS[col_name], color=text_color, fontsize=10, fontweight="bold")
        ax_left.tick_params(colors=text_color, labelsize=9)

        if ch_idx == 0:
            ax_left.set_title(f"Digit '{class1_name}' ({len(c1_arr)} samples overlaid)",
                              color=c1_line_color, fontsize=13, fontweight="bold", pad=8)
            ax_left.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155",
                           labelcolor=text_color, fontsize=8)

        # Column 1: Class 2
        ax_right = axes[ch_idx, 1]
        ax_right.set_facecolor("#1e293b")
        ax_right.grid(True, color="#334155", linestyle="--", alpha=0.6)

        for s_idx in range(len(c2_arr)):
            ax_right.plot(time_axis, c2_arr[s_idx, :, ch_idx], color=c2_line_color,
                          alpha=0.35, linewidth=1.2, label=f"Sample {s_idx+1}" if ch_idx == 0 else "")

        ax_right.plot(time_axis, c2_mean[:, ch_idx], color=c2_line_color, linewidth=2.5,
                      label="Mean Pattern" if ch_idx == 0 else "")
        ax_right.fill_between(time_axis, c2_mean[:, ch_idx] - c2_std[:, ch_idx],
                              c2_mean[:, ch_idx] + c2_std[:, ch_idx], color=c2_line_color, alpha=0.15)

        ax_right.tick_params(colors=text_color, labelsize=9)

        if ch_idx == 0:
            ax_right.set_title(f"Class '{class2_name}' ({len(c2_arr)} samples overlaid)",
                               color=c2_line_color, fontsize=13, fontweight="bold", pad=8)
            ax_right.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155",
                            labelcolor=text_color, fontsize=8)

    axes[-1, 0].set_xlabel("Time (seconds) [0.0s = Button Press + 500ms Settle]", color=text_color, fontsize=11)
    axes[-1, 1].set_xlabel("Time (seconds) [0.0s = Button Press + 500ms Settle]", color=text_color, fontsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"  [✓] 6-axis comparison plot saved to: {save_path}")
    return fig


def plot_motion_energy_profile(class1_name: str, class1_samples: list[np.ndarray],
                               class2_name: str, class2_samples: list[np.ndarray],
                               save_path: Path | None = None):
    """
    Plots the resultant magnitude for Accelerometer and Gyroscope:
      |a| = sqrt(ax^2 + ay^2 + az^2)
      |w| = sqrt(gx^2 + gy^2 + gz^2)
    This shows the overall kinetic signature / stroke intensity of the character.
    """
    time_axis = np.linspace(0.0, EXPECTED_SAMPLES / SAMPLE_RATE_HZ, EXPECTED_SAMPLES)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#0f172a")

    c1_color = "#38bdf8"
    c2_color = "#f43f5e"
    text_color = "#f8fafc"

    fig.suptitle(
        f"Kinetic Energy & Stroke Intensity Profiles: '{class1_name}' vs '{class2_name}'",
        fontsize=15, fontweight="bold", color=text_color
    )

    def calc_magnitudes(samples):
        accel_mags = [np.sqrt(s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) for s in samples]
        gyro_mags = [np.sqrt(s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2) for s in samples]
        return np.array(accel_mags), np.array(gyro_mags)

    c1_accel_m, c1_gyro_m = calc_magnitudes(class1_samples[:6])
    c2_accel_m, c2_gyro_m = calc_magnitudes(class2_samples[:6])

    # Plot Accel Mag
    for i in range(len(c1_accel_m)):
        axes[0, 0].plot(time_axis, c1_accel_m[i], color=c1_color, alpha=0.35, linewidth=1.2)
    axes[0, 0].plot(time_axis, np.mean(c1_accel_m, axis=0), color=c1_color, linewidth=2.5, label=f"Mean |a| ({class1_name})")
    axes[0, 0].set_title(f"Total Acceleration |a| — '{class1_name}'", color=c1_color, fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Linear Acceleration (LSB)", color=text_color)

    for i in range(len(c2_accel_m)):
        axes[0, 1].plot(time_axis, c2_accel_m[i], color=c2_color, alpha=0.35, linewidth=1.2)
    axes[0, 1].plot(time_axis, np.mean(c2_accel_m, axis=0), color=c2_color, linewidth=2.5, label=f"Mean |a| ({class2_name})")
    axes[0, 1].set_title(f"Total Acceleration |a| — '{class2_name}'", color=c2_color, fontsize=12, fontweight="bold")

    # Plot Gyro Mag
    for i in range(len(c1_gyro_m)):
        axes[1, 0].plot(time_axis, c1_gyro_m[i], color=c1_color, alpha=0.35, linewidth=1.2)
    axes[1, 0].plot(time_axis, np.mean(c1_gyro_m, axis=0), color=c1_color, linewidth=2.5, label=f"Mean |w| ({class1_name})")
    axes[1, 0].set_title(f"Angular Velocity |w| — '{class1_name}'", color=c1_color, fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Angular Velocity (LSB)", color=text_color)
    axes[1, 0].set_xlabel("Time (seconds)", color=text_color)

    for i in range(len(c2_gyro_m)):
        axes[1, 1].plot(time_axis, c2_gyro_m[i], color=c2_color, alpha=0.35, linewidth=1.2)
    axes[1, 1].plot(time_axis, np.mean(c2_gyro_m, axis=0), color=c2_color, linewidth=2.5, label=f"Mean |w| ({class2_name})")
    axes[1, 1].set_title(f"Angular Velocity |w| — '{class2_name}'", color=c2_color, fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Time (seconds)", color=text_color)

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.grid(True, color="#334155", linestyle="--", alpha=0.6)
        ax.tick_params(colors=text_color)
        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor=text_color, fontsize=9)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"  [✓] Magnitude energy plot saved to: {save_path}")
    return fig


# ────────────────────────── Main Execution ──────────────────────────
def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="SmartPen IMU Data Quality Verification")
    parser.add_argument("--class1", type=str, default="0", help="First class label (default: 0)")
    parser.add_argument("--class2", type=str, default="A", help="Second class label (default: A)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sample files to plot per class (default: 5)")
    parser.add_argument("--show", action="store_true", help="Display interactive plot window (GUI)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  SMARTPEN IMU DATA QUALITY VERIFICATION & WAVEFORM ANALYSIS")
    print("=" * 70)

    available_classes = find_available_classes()
    print(f"[INFO] Available dataset folders in {DATASET_DIR.resolve()}: {available_classes}")

    class1 = args.class1
    class2 = args.class2

    # Load Class 1 samples
    c1_samples, c1_files = load_class_samples(class1, max_samples=args.num_samples)
    print(f"\n[1/3] Loading Class '{class1}':")
    if c1_samples:
        print(f"  [OK] Found {len(c1_samples)} samples (e.g. {c1_files[:3]})")
    else:
        print(f"  [ERROR] No samples found for class '{class1}' in datasets/{class1}/")
        sys.exit(1)

    # Load Class 2 samples
    c2_samples, c2_files = load_class_samples(class2, max_samples=args.num_samples)
    print(f"\n[2/3] Loading Class '{class2}':")
    if not c2_samples:
        print(f"  [WARNING] No 500-sample files found in datasets/{class2}/ yet.")
        # Auto-fallback: check if another class like '1' has samples
        fallback = [c for c in available_classes if c != class1]
        if fallback:
            class2 = fallback[0]
            print(f"  [FALLBACK] Automatically using available class '{class2}' for comparison.")
            c2_samples, c2_files = load_class_samples(class2, max_samples=args.num_samples)
            print(f"  [OK] Loaded {len(c2_samples)} samples from '{class2}' (e.g. {c2_files[:3]})")
        else:
            print(f"  [ERROR] No other classes found to compare with '{class1}'.")
            sys.exit(1)
    else:
        print(f"  [OK] Found {len(c2_samples)} samples (e.g. {c2_files[:3]})")

    # Analyze Quality
    print(f"\n[3/3] Quantitative Data Health & Quality Report:")
    report = analyze_data_quality(class1, c1_samples, class2, c2_samples)
    print(f"  * Shape verification: Exactly {EXPECTED_SAMPLES} samples x 6 channels (ax, ay, az, gx, gy, gz)")
    print(f"  * Sampling frequency: {SAMPLE_RATE_HZ:.0f} Hz (Duration: {EXPECTED_SAMPLES/SAMPLE_RATE_HZ:.2f} seconds)")
    print(f"  * Dynamic Range (Peak-to-Peak):")
    print(f"      '{class1}': Accel PTP = {report['c1_ptp']['accel_ptp']:.0f} LSB | Gyro PTP = {report['c1_ptp']['gyro_ptp']:.0f} LSB")
    print(f"      '{class2}': Accel PTP = {report['c2_ptp']['accel_ptp']:.0f} LSB | Gyro PTP = {report['c2_ptp']['gyro_ptp']:.0f} LSB")
    print(f"  * Intra-class Repeatability (Mean Correlation within same class):")
    print(f"      '{class1}': {report['c1_intra_corr']:+.3f}")
    print(f"      '{class2}': {report['c2_intra_corr']:+.3f}")
    print(f"  * Inter-class Correlation ('{class1}' vs '{class2}'): {report['inter_corr']:+.3f}")

    # Verdict
    print("\n" + "-" * 70)
    if report["c1_ptp"]["gyro_ptp"] > 1000 and report["c2_ptp"]["gyro_ptp"] > 1000:
        print("  [PASS] VERDICT: Strong dynamic range detected. Sensors are actively capturing")
        print("         hand movement without signal clipping or dead flatlines.")
    else:
        print("  [WARNING] VERDICT: Low motion amplitude observed. Check pen movement or sensitivity.")
    print("-" * 70 + "\n")

    # Generate and save plots
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot1_path = OUTPUT_PLOT_DIR / f"waveform_comparison_{class1}_vs_{class2}.png"
    plot2_path = OUTPUT_PLOT_DIR / f"energy_magnitude_{class1}_vs_{class2}.png"

    print("Generating visualization plots...")
    plot_6axis_comparison(class1, c1_samples, class2, c2_samples, num_to_plot=args.num_samples, save_path=plot1_path)
    plot_motion_energy_profile(class1, c1_samples, class2, c2_samples, save_path=plot2_path)

    print("\nAll plots successfully generated!")
    print(f"  1. {plot1_path.resolve()}")
    print(f"  2. {plot2_path.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
