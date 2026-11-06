"""
Radar-Camera Fusion ADAS Project
Phase 2C: Lead-Vehicle Track Validation

Analyzes Track 1 from the stationary
lead-vehicle radar experiment.

Measures:
    - Track persistence
    - Mean forward distance
    - Mean lateral position
    - Mean radial velocity
    - Early versus late velocity
    - Minimum and maximum velocity
    - Forward-distance variation

This script does not require CARLA.
"""

from __future__ import annotations

import csv

from pathlib import Path
from statistics import mean


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

TRACK_CSV = Path(
    "results/"
    "lead_vehicle_20261106_145042/"
    "tracking_20261106_150200_937704/"
    "radar_tracks.csv"
)

TRACK_ID = "1"


# ==========================================================
# 2. CALCULATE TRACK STATISTICS
# ==========================================================

def summarize(
    label: str,
    rows: list[dict[str, str]]
) -> None:

    if not rows:
        print(f"{label}: No samples")
        return

    x_values = [
        float(row["x_m"])
        for row in rows
    ]

    y_values = [
        float(row["y_m"])
        for row in rows
    ]

    velocities = [
        float(row["mean_velocity_mps"])
        for row in rows
    ]

    print(f"\n{label}")
    print("-" * 60)

    print(f"Samples             : {len(rows)}")
    print(f"Mean forward X      : {mean(x_values):.3f} m")
    print(f"Mean lateral Y      : {mean(y_values):.3f} m")
    print(f"Mean radial velocity: {mean(velocities):.3f} m/s")

    print(
        f"Minimum velocity    : "
        f"{min(velocities):.3f} m/s"
    )

    print(
        f"Maximum velocity    : "
        f"{max(velocities):.3f} m/s"
    )

    print(
        f"Forward X range     : "
        f"{min(x_values):.3f} to "
        f"{max(x_values):.3f} m"
    )


# ==========================================================
# 3. MAIN ANALYSIS
# ==========================================================

def main() -> None:

    print("=" * 65)
    print("RADAR-CAMERA FUSION ADAS PROJECT")
    print("PHASE 2C: LEAD-VEHICLE TRACK VALIDATION")
    print("=" * 65)

    if not TRACK_CSV.is_file():
        raise FileNotFoundError(
            f"Tracking CSV not found: {TRACK_CSV.resolve()}"
        )

    with TRACK_CSV.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = [
            row
            for row in reader
            if row["track_id"] == TRACK_ID
        ]

    rows.sort(
        key=lambda row: int(row["frame"])
    )

    if not rows:
        raise ValueError(
            f"No observations found for Track {TRACK_ID}"
        )

    print(f"\nAnalyzing Track ID: {TRACK_ID}")
    print(f"Total observations: {len(rows)}")

    # Overall statistics
    summarize(
        "OVERALL TRACK STATISTICS",
        rows
    )

    # Initial measurements
    summarize(
        "FIRST 10 OBSERVATIONS",
        rows[:10]
    )

    # Later measurements
    summarize(
        "LAST 50 OBSERVATIONS",
        rows[-50:]
    )

    # Individual measurements
    print("\nFIRST AND LAST FIVE MEASUREMENTS")
    print("-" * 65)

    selected_rows = rows[:5] + rows[-5:]

    for row in selected_rows:

        frame = int(row["frame"])
        x = float(row["x_m"])
        velocity = float(row["mean_velocity_mps"])

        print(
            f"Frame={frame} | "
            f"X={x:.3f} m | "
            f"Radial velocity={velocity:+.3f} m/s"
        )

    print("=" * 65)
    print("TRACK VALIDATION ANALYSIS COMPLETE")
    print("=" * 65)

    print(
        "\nNote: Radar radial velocity is not "
        "the complete object velocity vector."
    )

    print(
        "Track 1 has not yet been conclusively "
        "associated with the lead vehicle."
    )


if __name__ == "__main__":
    main()