"""
Radar-Camera ADAS Project
Analyze radar track persistence and average measurements.
"""

import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean

CSV_PATH = Path(
    "results/moving_lead_20261106_151239_827681/"
    "tracking_20261106_151505_679849/"
    "radar_tracks.csv"
)


def main():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Tracking CSV not found: {CSV_PATH}"
        )

    tracks = defaultdict(list)

    with CSV_PATH.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            tracks[row["track_id"]].append(row)

    print("=" * 70)
    print("RADAR TRACK ANALYSIS")
    print("=" * 70)

    print(
        f"{'Track':<10}"
        f"{'Samples':<12}"
        f"{'Mean X (m)':<15}"
        f"{'Mean Y (m)':<15}"
        f"{'Mean Vel (m/s)':<18}"
    )

    print("-" * 70)

    sorted_tracks = sorted(
        tracks.items(),
        key=lambda item: len(item[1]),
        reverse=True
    )

    for track_id, rows in sorted_tracks[:10]:

        mean_x = mean(
            float(row["x_m"])
            for row in rows
        )

        mean_y = mean(
            float(row["y_m"])
            for row in rows
        )

        mean_velocity = mean(
            float(row["mean_velocity_mps"])
            for row in rows
        )

        print(
            f"{track_id:<10}"
            f"{len(rows):<12}"
            f"{mean_x:<15.2f}"
            f"{mean_y:<15.2f}"
            f"{mean_velocity:<18.2f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()