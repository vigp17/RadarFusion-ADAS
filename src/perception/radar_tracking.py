"""
Radar-Camera Fusion ADAS Project
Phase 2B: Radar Object Clustering and Tracking

CARLA Version: 0.9.16

Processes previously recorded front-radar measurements.

Features:
- Radar CSV loading
- Polar-to-Cartesian conversion
- Forward-region filtering
- Euclidean connected-component clustering
- Multi-frame nearest-neighbor tracking
- Persistent track IDs
- CSV export
- Tracking visualization

Limitations:
- Assumes stationary ego vehicle
- Radar clusters are not verified vehicles
- No ego-motion compensation
- No motion prediction
"""

from __future__ import annotations

import argparse
import csv
import math

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

MIN_X = 3.0
MAX_X = 100.0

MAX_Y = 8.0
MAX_Z = 3.0

CLUSTER_RADIUS = 1.5
MIN_POINTS = 2

ASSOCIATION_GATE = 4.0
MAX_MISSED = 5


# ==========================================================
# 2. DATA STRUCTURES
# ==========================================================

@dataclass(frozen=True)
class RadarPoint:
    frame: int
    timestamp_s: float
    x_m: float
    y_m: float
    z_m: float
    velocity_mps: float


@dataclass(frozen=True)
class Cluster:
    frame: int
    timestamp_s: float
    x_m: float
    y_m: float
    z_m: float
    velocity_mps: float
    point_count: int


@dataclass
class Track:
    track_id: int
    x_m: float
    y_m: float
    last_frame: int
    missed: int = 0


# ==========================================================
# 3. LOAD RADAR DATA
# ==========================================================

def load_points(
    path: Path
) -> dict[int, list[RadarPoint]]:

    if not path.is_file():
        raise FileNotFoundError(path)

    frames: dict[int, list[RadarPoint]] = {}

    with path.open(
        newline="",
        encoding="utf-8"
    ) as handle:

        reader = csv.DictReader(handle)

        required = {
            "frame",
            "timestamp_s",
            "depth_m",
            "velocity_mps",
            "azimuth_rad",
            "altitude_rad"
        }

        available = set(reader.fieldnames or [])

        if not required.issubset(available):
            missing = sorted(required - available)

            raise ValueError(
                f"Missing CSV columns: {missing}"
            )

        for row in reader:

            frame = int(row["frame"])

            depth = float(row["depth_m"])
            velocity = float(row["velocity_mps"])

            azimuth = float(row["azimuth_rad"])
            altitude = float(row["altitude_rad"])

            timestamp = float(row["timestamp_s"])

            values = (
                depth,
                velocity,
                azimuth,
                altitude,
                timestamp
            )

            if not all(
                math.isfinite(value)
                for value in values
            ):
                continue

            if depth <= 0:
                continue

            # CARLA radar sensor-local coordinates.
            # X: forward
            # Y: right
            # Z: up

            x = (
                depth
                * math.cos(altitude)
                * math.cos(azimuth)
            )

            y = (
                depth
                * math.cos(altitude)
                * math.sin(azimuth)
            )

            z = depth * math.sin(altitude)

            # Forward-region filtering

            if not MIN_X <= x <= MAX_X:
                continue

            if abs(y) > MAX_Y:
                continue

            if abs(z) > MAX_Z:
                continue

            point = RadarPoint(
                frame=frame,
                timestamp_s=timestamp,
                x_m=x,
                y_m=y,
                z_m=z,
                velocity_mps=velocity
            )

            frames.setdefault(
                frame,
                []
            ).append(point)

    return frames


# ==========================================================
# 4. EUCLIDEAN CLUSTERING
# ==========================================================

def cluster_points(
    points: list[RadarPoint]
) -> list[Cluster]:

    if not points:
        return []

    coordinates = np.array(
        [
            (point.x_m, point.y_m)
            for point in points
        ],
        dtype=float
    )

    visited = np.zeros(
        len(points),
        dtype=bool
    )

    clusters: list[Cluster] = []

    for start in range(len(points)):

        if visited[start]:
            continue

        visited[start] = True

        pending = [start]

        component: list[int] = []

        while pending:

            current = pending.pop()

            component.append(current)

            distances = np.linalg.norm(
                coordinates - coordinates[current],
                axis=1
            )

            neighbors = np.flatnonzero(
                (distances <= CLUSTER_RADIUS)
                & (~visited)
            )

            for neighbor in neighbors:

                index = int(neighbor)

                visited[index] = True

                pending.append(index)

        if len(component) < MIN_POINTS:
            continue

        group = [
            points[index]
            for index in component
        ]

        clusters.append(
            Cluster(
                frame=group[0].frame,
                timestamp_s=group[0].timestamp_s,

                x_m=float(
                    np.mean([
                        point.x_m
                        for point in group
                    ])
                ),

                y_m=float(
                    np.mean([
                        point.y_m
                        for point in group
                    ])
                ),

                z_m=float(
                    np.mean([
                        point.z_m
                        for point in group
                    ])
                ),

                velocity_mps=float(
                    np.mean([
                        point.velocity_mps
                        for point in group
                    ])
                ),

                point_count=len(group)
            )
        )

    return clusters


# ==========================================================
# 5. MULTI-FRAME TRACKING
# ==========================================================

class Tracker:

    def __init__(self) -> None:

        self.tracks: dict[int, Track] = {}

        self.next_id = 1

    def update(
        self,
        frame: int,
        clusters: list[Cluster]
    ) -> list[tuple[int, Cluster]]:

        options: list[
            tuple[float, int, int]
        ] = []

        # Generate candidate associations

        for track_id, track in self.tracks.items():

            for index, cluster in enumerate(clusters):

                distance = math.hypot(
                    track.x_m - cluster.x_m,
                    track.y_m - cluster.y_m
                )

                if distance <= ASSOCIATION_GATE:

                    options.append(
                        (
                            distance,
                            track_id,
                            index
                        )
                    )

        options.sort()

        used_tracks: set[int] = set()
        used_clusters: set[int] = set()

        matches: list[
            tuple[int, Cluster]
        ] = []

        # Associate existing tracks

        for _, track_id, index in options:

            if track_id in used_tracks:
                continue

            if index in used_clusters:
                continue

            cluster = clusters[index]

            track = self.tracks[track_id]

            track.x_m = cluster.x_m
            track.y_m = cluster.y_m

            track.last_frame = frame
            track.missed = 0

            matches.append(
                (track_id, cluster)
            )

            used_tracks.add(track_id)
            used_clusters.add(index)

        # Create tracks for unmatched clusters

        for index, cluster in enumerate(clusters):

            if index in used_clusters:
                continue

            track_id = self.next_id

            self.next_id += 1

            self.tracks[track_id] = Track(
                track_id=track_id,
                x_m=cluster.x_m,
                y_m=cluster.y_m,
                last_frame=frame
            )

            matches.append(
                (track_id, cluster)
            )

        # Age unmatched tracks

        for track_id in list(self.tracks):

            track = self.tracks[track_id]

            if track.last_frame != frame:

                track.missed += max(
                    1,
                    frame - max(
                        track.last_frame,
                        frame - 1
                    )
                )

            if track.missed > MAX_MISSED:

                del self.tracks[track_id]

        return matches


# ==========================================================
# 6. MAIN PROCESSING
# ==========================================================

def run(
    input_csv: Path
) -> Path:

    frames = load_points(input_csv)

    # Include frames without radar detections
    # using the lead-vehicle summary when available.

    summary_csv = input_csv.with_name(
        "lead_vehicle_summary.csv"
    )

    if summary_csv.exists():

        with summary_csv.open(
            newline="",
            encoding="utf-8"
        ) as handle:

            reader = csv.DictReader(handle)

            for row in reader:

                frame = int(row["frame"])

                frames.setdefault(
                    frame,
                    []
                )

    if not frames:

        raise ValueError(
            "No frames available in the supplied CSV files."
        )

    tracker = Tracker()

    rows: list[
        dict[str, Any]
    ] = []

    total_clusters = 0

    for frame in sorted(frames):

        clusters = cluster_points(
            frames[frame]
        )

        total_clusters += len(clusters)

        matches = tracker.update(
            frame,
            clusters
        )

        for track_id, cluster in matches:

            rows.append({
                "frame": frame,
                "timestamp_s": cluster.timestamp_s,
                "track_id": track_id,
                "x_m": cluster.x_m,
                "y_m": cluster.y_m,
                "z_m": cluster.z_m,
                "mean_velocity_mps": cluster.velocity_mps,
                "point_count": cluster.point_count
            })

    # Create a unique output directory

    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = (
        input_csv.parent
        / f"tracking_{run_id}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    output_csv = (
        output_dir
        / "radar_tracks.csv"
    )

    columns = [
        "frame",
        "timestamp_s",
        "track_id",
        "x_m",
        "y_m",
        "z_m",
        "mean_velocity_mps",
        "point_count"
    ]

    with output_csv.open(
        "x",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns
        )

        writer.writeheader()

        writer.writerows(rows)

    # ------------------------------------------------------
    # Generate tracking plot
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    track_ids = sorted({
        row["track_id"]
        for row in rows
    })

    for track_id in track_ids:

        selected = [
            row
            for row in rows
            if row["track_id"] == track_id
        ]

        x_values = [
            row["x_m"]
            for row in selected
        ]

        y_values = [
            row["y_m"]
            for row in selected
        ]

        ax.plot(
            x_values,
            y_values,
            ".",
            label=f"Track {track_id}"
        )

    ax.set_xlabel(
        "Forward distance (m)"
    )

    ax.set_ylabel(
        "Lateral distance (m, +right)"
    )

    ax.set_title(
        "CARLA Radar Candidate Tracks"
    )

    ax.grid(True)

    if 0 < len(track_ids) <= 12:
        ax.legend()

    fig.tight_layout()

    output_plot = (
        output_dir
        / "radar_tracks.png"
    )

    fig.savefig(
        output_plot,
        dpi=150
    )

    plt.close(fig)

    # ------------------------------------------------------
    # Print summary
    # ------------------------------------------------------

    print("=" * 65)
    print("PHASE 2B: RADAR CLUSTERING AND TRACKING")
    print("=" * 65)

    print(
        f"Processed frames: {len(frames)}"
    )

    print(
        f"Cluster observations: {total_clusters}"
    )

    print(
        f"Unique track IDs: {tracker.next_id - 1}"
    )

    print(
        f"Track observations: {len(rows)}"
    )

    print(f"Tracking CSV: {output_csv}")
    print(f"Tracking plot: {output_plot}")

    print("=" * 65)
    print("PROCESSING COMPLETE")
    print("=" * 65)

    return output_dir


# ==========================================================
# 7. ENTRY POINT
# ==========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Process stationary-ego CARLA "
            "front radar telemetry."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/"
            "lead_vehicle_20261106_145042/"
            "radar_detections.csv"
        )
    )

    args = parser.parse_args()

    run(args.input)


if __name__ == "__main__":
    main()