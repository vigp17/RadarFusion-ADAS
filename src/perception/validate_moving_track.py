"""
Radar-Camera Fusion ADAS Project
Phase 2F: Moving Lead-Vehicle Track Validation

CARLA Version: 0.9.16

Compares radar Track 1 against recorded
CARLA ground-truth motion.

Features:
    - Ground-truth CSV loading
    - Radar tracking CSV loading
    - Frame synchronization
    - Distance comparison
    - Radial velocity comparison
    - Startup outlier analysis
    - Distance-change validation
    - CSV summary export
    - Three validation plots

Limitations:
    - Radar measures reflecting surfaces.
    - Ground truth measures vehicle-center separation.
    - Radar velocity is line-of-sight velocity.
    - Track identity is not independently verified.
    - This is a stationary-ego scenario.
"""

from __future__ import annotations

import csv
import math

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_RUN = "moving_lead_20261106_151239_827681"

TRACKING_RUN = "tracking_20261106_151505_679849"

GROUND_TRUTH_CSV = (
    RESULTS_DIR
    / INPUT_RUN
    / "motion_summary.csv"
)

TRACKING_CSV = (
    RESULTS_DIR
    / INPUT_RUN
    / TRACKING_RUN
    / "radar_tracks.csv"
)

TRACK_ID = 1

# Exclude startup frames from steady-motion statistics.
STARTUP_FRAMES = 10

RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / INPUT_RUN
    / f"validation_{RUN_ID}"
)


# ==========================================================
# 2. DATA STRUCTURES
# ==========================================================

@dataclass(frozen=True)
class GroundTruthSample:
    frame: int
    timestamp_s: float
    ego_speed_mps: float
    lead_speed_mps: float
    center_distance_m: float


@dataclass(frozen=True)
class RadarTrackSample:
    frame: int
    timestamp_s: float
    x_m: float
    y_m: float
    radial_velocity_mps: float
    point_count: int


# ==========================================================
# 3. LOAD GROUND TRUTH
# ==========================================================

def load_ground_truth(
    csv_path: Path
) -> dict[int, GroundTruthSample]:

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth CSV not found: {csv_path}"
        )

    samples = {}

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "frame",
            "timestamp_s",
            "ego_speed_mps",
            "lead_speed_mps",
            "center_distance_m"
        }

        available = set(reader.fieldnames or [])

        if not required.issubset(available):
            raise ValueError(
                f"Missing ground-truth columns: "
                f"{sorted(required - available)}"
            )

        for row in reader:

            sample = GroundTruthSample(
                frame=int(row["frame"]),
                timestamp_s=float(row["timestamp_s"]),
                ego_speed_mps=float(row["ego_speed_mps"]),
                lead_speed_mps=float(row["lead_speed_mps"]),
                center_distance_m=float(
                    row["center_distance_m"]
                )
            )

            values = (
                sample.timestamp_s,
                sample.ego_speed_mps,
                sample.lead_speed_mps,
                sample.center_distance_m
            )

            if not all(math.isfinite(v) for v in values):
                continue

            samples[sample.frame] = sample

    return samples


# ==========================================================
# 4. LOAD RADAR TRACK
# ==========================================================

def load_radar_track(
    csv_path: Path,
    track_id: int
) -> dict[int, RadarTrackSample]:

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Tracking CSV not found: {csv_path}"
        )

    samples = {}

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "frame",
            "timestamp_s",
            "track_id",
            "x_m",
            "y_m",
            "mean_velocity_mps",
            "point_count"
        }

        available = set(reader.fieldnames or [])

        if not required.issubset(available):
            raise ValueError(
                f"Missing tracking columns: "
                f"{sorted(required - available)}"
            )

        for row in reader:

            if int(row["track_id"]) != track_id:
                continue

            sample = RadarTrackSample(
                frame=int(row["frame"]),
                timestamp_s=float(row["timestamp_s"]),
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                radial_velocity_mps=float(
                    row["mean_velocity_mps"]
                ),
                point_count=int(row["point_count"])
            )

            values = (
                sample.timestamp_s,
                sample.x_m,
                sample.y_m,
                sample.radial_velocity_mps
            )

            if not all(math.isfinite(v) for v in values):
                continue

            samples[sample.frame] = sample

    return samples


# ==========================================================
# 5. VALIDATION
# ==========================================================

def main() -> None:

    print("=" * 70)
    print("RADAR-CAMERA FUSION ADAS PROJECT")
    print("PHASE 2F: MOVING LEAD-VEHICLE TRACK VALIDATION")
    print("=" * 70)

    ground_truth = load_ground_truth(
        GROUND_TRUTH_CSV
    )

    radar_track = load_radar_track(
        TRACKING_CSV,
        TRACK_ID
    )

    common_frames = sorted(
        set(ground_truth) & set(radar_track)
    )

    if len(common_frames) < 2:
        raise RuntimeError(
            "Insufficient synchronized tracking samples."
        )

    # ------------------------------------------------------
    # Extract synchronized measurements
    # ------------------------------------------------------

    timestamps = np.array(
        [
            ground_truth[f].timestamp_s
            for f in common_frames
        ],
        dtype=float
    )

    time_s = timestamps - timestamps[0]

    gt_distance = np.array(
        [
            ground_truth[f].center_distance_m
            for f in common_frames
        ],
        dtype=float
    )

    radar_distance = np.array(
        [
            radar_track[f].x_m
            for f in common_frames
        ],
        dtype=float
    )

    radar_velocity = np.array(
        [
            radar_track[f].radial_velocity_mps
            for f in common_frames
        ],
        dtype=float
    )

    lead_speed = np.array(
        [
            ground_truth[f].lead_speed_mps
            for f in common_frames
        ],
        dtype=float
    )

    ego_speed = np.array(
        [
            ground_truth[f].ego_speed_mps
            for f in common_frames
        ],
        dtype=float
    )

    # ------------------------------------------------------
    # Validate timestamps
    # ------------------------------------------------------

    for frame in common_frames:

        gt_time = ground_truth[frame].timestamp_s
        radar_time = radar_track[frame].timestamp_s

        if abs(gt_time - radar_time) > 0.001:
            raise RuntimeError(
                f"Timestamp mismatch at frame {frame}"
            )

    if np.any(np.diff(timestamps) <= 0.0):
        raise RuntimeError(
            "Simulation timestamps are not increasing."
        )

    # ------------------------------------------------------
    # Ground-truth separation rate
    # ------------------------------------------------------

    gt_separation_rate = np.gradient(
        gt_distance,
        timestamps
    )

    # This is an approximate longitudinal relative speed.
    relative_speed = lead_speed - ego_speed

    # ------------------------------------------------------
    # Distance comparison
    # ------------------------------------------------------

    distance_difference = (
        gt_distance - radar_distance
    )

    # Compare changes relative to the first observation.
    # This removes the initial fixed range offset.
    gt_distance_change = (
        gt_distance - gt_distance[0]
    )

    radar_distance_change = (
        radar_distance - radar_distance[0]
    )

    change_error = (
        radar_distance_change - gt_distance_change
    )

    # ------------------------------------------------------
    # Startup exclusion
    # ------------------------------------------------------

    start_index = min(
        STARTUP_FRAMES,
        len(common_frames) - 1
    )

    steady_radar_velocity = (
        radar_velocity[start_index:]
    )

    steady_gt_rate = (
        gt_separation_rate[start_index:]
    )

    steady_change_error = (
        change_error[start_index:]
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    mean_distance_difference = float(
        np.mean(distance_difference)
    )

    distance_change_rmse = float(
        np.sqrt(
            np.mean(steady_change_error ** 2)
        )
    )

    radial_velocity_rmse = float(
        np.sqrt(
            np.mean(
                (
                    steady_radar_velocity
                    - steady_gt_rate
                ) ** 2
            )
        )
    )

    mean_radar_velocity = float(
        np.mean(steady_radar_velocity)
    )

    mean_gt_separation_rate = float(
        np.mean(steady_gt_rate)
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print(f"Track ID           : {TRACK_ID}")

    print(
        f"Synchronized frames       : "
        f"{len(common_frames)}"
    )

    print(
        f"Startup samples excluded  : "
        f"{start_index}"
    )

    print(
        f"Initial GT center distance: "
        f"{gt_distance[0]:.3f} m"
    )

    print(
        f"Final GT center distance  : "
        f"{gt_distance[-1]:.3f} m"
    )

    print(
        f"Initial radar distance    : "
        f"{radar_distance[0]:.3f} m"
    )

    print(
        f"Final radar distance      : "
        f"{radar_distance[-1]:.3f} m"
    )

    print(
        f"Mean center/radar offset  : "
        f"{mean_distance_difference:.3f} m"
    )

    print(
        f"Distance-change RMSE      : "
        f"{distance_change_rmse:.3f} m"
    )

    print(
        f"Mean radar radial velocity: "
        f"{mean_radar_velocity:.3f} m/s"
    )

    print(
        f"Mean GT separation rate   : "
        f"{mean_gt_separation_rate:.3f} m/s"
    )

    print(
        f"Radial velocity RMSE      : "
        f"{radial_velocity_rmse:.3f} m/s"
    )

    # ------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False
    )

    # ------------------------------------------------------
    # Plot 1: Distance comparison
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        time_s,
        gt_distance,
        label="Ground-Truth Center Distance"
    )

    ax.plot(
        time_s,
        radar_distance,
        label="Radar Track Forward Distance"
    )

    ax.set_title(
        "Moving Lead Vehicle: Distance Comparison"
    )

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Distance (m)")

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "distance_comparison.png",
        dpi=150
    )

    plt.close(fig)

    # ------------------------------------------------------
    # Plot 2: Velocity comparison
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        time_s,
        radar_velocity,
        label="Radar Radial Velocity"
    )

    ax.plot(
        time_s,
        gt_separation_rate,
        label="Ground-Truth Separation Rate"
    )

    ax.set_title(
        "Moving Lead Vehicle: Velocity Comparison"
    )

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Velocity (m/s)")

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "velocity_comparison.png",
        dpi=150
    )

    plt.close(fig)

    # ------------------------------------------------------
    # Plot 3: Distance-change discrepancy
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        time_s,
        change_error,
        label="Radar - GT Distance Change"
    )

    ax.axhline(
        0.0,
        linestyle="--"
    )

    ax.set_title(
        "Radar Relative-Distance Change Discrepancy"
    )

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Distance Difference (m)")

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "distance_change_error.png",
        dpi=150
    )

    plt.close(fig)

    # ------------------------------------------------------
    # Save synchronized validation data
    # ------------------------------------------------------

    output_csv = (
        OUTPUT_DIR
        / "validation_samples.csv"
    )

    with output_csv.open(
        "x",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "time_s",
            "gt_center_distance_m",
            "radar_forward_distance_m",
            "gt_separation_rate_mps",
            "radar_radial_velocity_mps",
            "distance_change_error_m"
        ])

        for i, frame in enumerate(common_frames):

            writer.writerow([
                frame,
                time_s[i],
                gt_distance[i],
                radar_distance[i],
                gt_separation_rate[i],
                radar_velocity[i],
                change_error[i]
            ])

    # ------------------------------------------------------
    # Save summary
    # ------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "validation_summary.csv"
    )

    with summary_path.open(
        "x",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "metric",
            "value"
        ])

        metrics = {
            "track_id": TRACK_ID,
            "synchronized_frames": len(common_frames),
            "startup_samples_excluded": start_index,
            "mean_center_radar_offset_m":
                mean_distance_difference,
            "distance_change_rmse_m":
                distance_change_rmse,
            "mean_radar_velocity_mps":
                mean_radar_velocity,
            "mean_gt_separation_rate_mps":
                mean_gt_separation_rate,
            "radial_velocity_rmse_mps":
                radial_velocity_rmse
        }

        for metric, value in metrics.items():
            writer.writerow([
                metric,
                value
            ])

    print("-" * 70)

    print(
        f"Validation results saved: {OUTPUT_DIR}"
    )

    print("=" * 70)
    print("PHASE 2F VALIDATION COMPLETE")
    print("=" * 70)

    print(
        "\nImportant: Distance-change and radial-velocity "
        "metrics are preliminary comparisons."
    )

    print(
        "Track identity, surface geometry, and radar "
        "line-of-sight effects still require validation."
    )


# ==========================================================
# 6. ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()