"""
Radar-Camera Fusion ADAS Project
Phase 3A: Adaptive Cruise Control

Offline longitudinal ACC simulation.

Features:
- Free-road cruise control
- Lead-vehicle following
- Constant-time-headway policy
- Lead-vehicle braking and acceleration
- Acceleration constraints
- CSV telemetry
- Performance plots

This is a simulation prototype, not a production ACC system.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================

DT = 0.05
DURATION_S = 40.0

OUTPUT_DIR = Path("results/acc_phase3a")


# ==========================================================
# VEHICLE STATE
# ==========================================================

@dataclass
class Vehicle:
    x: float
    v: float


# ==========================================================
# ACC CONFIGURATION
# ==========================================================

@dataclass(frozen=True)
class ACCConfig:
    cruise_speed: float = 15.0
    headway: float = 1.8
    standstill_gap: float = 5.0

    max_accel: float = 2.0
    max_brake: float = -4.0

    speed_gain: float = 0.6
    gap_gain: float = 0.15
    relative_speed_gain: float = 0.8


# ==========================================================
# ACC CONTROLLER
# ==========================================================

class ACCController:

    def __init__(self, config: ACCConfig):
        self.config = config

    def desired_gap(self, ego_speed: float) -> float:
        return (
            self.config.standstill_gap
            + self.config.headway * ego_speed
        )

    def acceleration(
        self,
        ego_speed: float,
        gap: float | None,
        lead_speed: float | None
    ) -> float:

        config = self.config

        cruise_acceleration = (
            config.speed_gain
            * (config.cruise_speed - ego_speed)
        )

        if gap is None or lead_speed is None:

            requested_acceleration = cruise_acceleration

        else:

            desired_gap = self.desired_gap(ego_speed)

            gap_error = gap - desired_gap

            relative_speed = lead_speed - ego_speed

            following_acceleration = (
                config.gap_gain * gap_error
                + config.relative_speed_gain * relative_speed
            )

            requested_acceleration = min(
                cruise_acceleration,
                following_acceleration
            )

        return max(
            config.max_brake,
            min(
                config.max_accel,
                requested_acceleration
            )
        )


# ==========================================================
# VEHICLE DYNAMICS
# ==========================================================

def advance(
    vehicle: Vehicle,
    acceleration: float
) -> Vehicle:

    next_speed = max(
        0.0,
        vehicle.v + acceleration * DT
    )

    next_position = (
        vehicle.x
        + 0.5 * (vehicle.v + next_speed) * DT
    )

    return Vehicle(
        x=next_position,
        v=next_speed
    )


# ==========================================================
# LEAD VEHICLE SCENARIO
# ==========================================================

def lead_accel(time_s: float) -> float:

    if time_s < 10.0:
        return 0.0

    if time_s < 15.0:
        return -1.0

    if time_s < 25.0:
        return 0.0

    if time_s < 30.0:
        return 1.0

    return 0.0


# ==========================================================
# SIMULATION
# ==========================================================

def simulate() -> list[dict[str, float]]:

    controller = ACCController(
        ACCConfig()
    )

    ego = Vehicle(
        x=0.0,
        v=10.0
    )

    lead = Vehicle(
        x=50.0,
        v=12.0
    )

    telemetry: list[dict[str, float]] = []

    steps = round(DURATION_S / DT)

    for step in range(steps):

        time_s = step * DT

        gap = lead.x - ego.x

        desired_gap = controller.desired_gap(
            ego.v
        )

        acceleration = controller.acceleration(
            ego_speed=ego.v,
            gap=gap,
            lead_speed=lead.v
        )

        telemetry.append({
            "time_s": time_s,
            "ego_position_m": ego.x,
            "lead_position_m": lead.x,
            "ego_speed_mps": ego.v,
            "lead_speed_mps": lead.v,
            "gap_m": gap,
            "desired_gap_m": desired_gap,
            "acceleration_mps2": acceleration
        })

        ego = advance(
            ego,
            acceleration
        )

        lead = advance(
            lead,
            lead_accel(time_s)
        )

        if lead.x <= ego.x:
            raise RuntimeError(
                "Vehicle-center overlap detected "
                f"at t={time_s + DT:.2f} s"
            )

    return telemetry


# ==========================================================
# SAVE TELEMETRY
# ==========================================================

def save(
    telemetry: list[dict[str, float]]
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = OUTPUT_DIR / "acc_telemetry.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(telemetry[0])
        )

        writer.writeheader()

        writer.writerows(telemetry)

    # ------------------------------------------------------
    # Prepare plotting data
    # ------------------------------------------------------

    time_s = [
        row["time_s"]
        for row in telemetry
    ]

    ego_speed = [
        row["ego_speed_mps"]
        for row in telemetry
    ]

    lead_speed = [
        row["lead_speed_mps"]
        for row in telemetry
    ]

    actual_gap = [
        row["gap_m"]
        for row in telemetry
    ]

    desired_gap = [
        row["desired_gap_m"]
        for row in telemetry
    ]

    acceleration = [
        row["acceleration_mps2"]
        for row in telemetry
    ]

    # ------------------------------------------------------
    # Create plots
    # ------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True
    )

    # Speed comparison

    axes[0].plot(
        time_s,
        ego_speed,
        label="Ego"
    )

    axes[0].plot(
        time_s,
        lead_speed,
        label="Lead"
    )

    axes[0].axhline(
        15.0,
        linestyle="--",
        label="Cruise Target"
    )

    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend()

    # Following distance

    axes[1].plot(
        time_s,
        actual_gap,
        label="Actual Center Gap"
    )

    axes[1].plot(
        time_s,
        desired_gap,
        label="Desired Gap"
    )

    axes[1].set_ylabel("Distance (m)")
    axes[1].legend()

    # Acceleration

    axes[2].plot(
        time_s,
        acceleration,
        label="ACC Acceleration"
    )

    axes[2].axhline(
        2.0,
        linestyle="--"
    )

    axes[2].axhline(
        -4.0,
        linestyle="--"
    )

    axes[2].set_ylabel(
        "Acceleration (m/s²)"
    )

    axes[2].set_xlabel(
        "Simulation Time (s)"
    )

    axes[2].legend()

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "acc_simulation.png",
        dpi=150
    )

    plt.close(fig)


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:

    print("=" * 65)
    print("RADAR-CAMERA FUSION ADAS PROJECT")
    print("PHASE 3A: ADAPTIVE CRUISE CONTROL")
    print("=" * 65)

    telemetry = simulate()

    save(telemetry)

    minimum_gap = min(
        row["gap_m"]
        for row in telemetry
    )

    minimum_acceleration = min(
        row["acceleration_mps2"]
        for row in telemetry
    )

    maximum_acceleration = max(
        row["acceleration_mps2"]
        for row in telemetry
    )

    final = telemetry[-1]

    print(
        f"Simulation steps: {len(telemetry)}"
    )

    print(
        f"Minimum center gap: "
        f"{minimum_gap:.2f} m"
    )

    print(
        f"Acceleration range: "
        f"{minimum_acceleration:.2f} to "
        f"{maximum_acceleration:.2f} m/s²"
    )

    print(
        f"Final ego speed: "
        f"{final['ego_speed_mps']:.2f} m/s"
    )

    print(
        f"Final lead speed: "
        f"{final['lead_speed_mps']:.2f} m/s"
    )

    print(
        f"Outputs: {OUTPUT_DIR}"
    )

    print("=" * 65)
    print("ACC SIMULATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()