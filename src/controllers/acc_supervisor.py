"""
Radar-Camera Fusion ADAS Project
Phase 3C: ACC Supervisor and Target-Loss Recovery

Offline simulation only.

Features:
- Cruise control
- Lead-vehicle following
- Target-loss handling
- Target reacquisition
- Emergency braking
- Six scenario tests
- CSV telemetry export

This is not a production safety system.
"""

from __future__ import annotations

import csv

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


# ==========================================================
# CONFIGURATION
# ==========================================================

DT = 0.05
DURATION_S = 35.0

OUTPUT_DIR = Path("results/acc_phase3c")


# ==========================================================
# ACC MODES
# ==========================================================

class Mode(str, Enum):

    CRUISE = "CRUISE"
    FOLLOW = "FOLLOW"
    TARGET_LOST = "TARGET_LOST"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"


# ==========================================================
# CONTROLLER CONFIGURATION
# ==========================================================

@dataclass(frozen=True)
class Config:

    cruise_speed: float = 15.0

    headway: float = 1.8
    standstill_gap: float = 5.0

    max_accel: float = 2.0
    max_decel: float = -4.0

    speed_gain: float = 0.6
    gap_gain: float = 0.15
    relative_speed_gain: float = 0.8

    lost_decel: float = -2.0

    emergency_ttc_s: float = 2.0
    emergency_gap_m: float = 6.0


# ==========================================================
# VEHICLE MODEL
# ==========================================================

@dataclass
class Vehicle:

    x: float
    v: float


def advance(
    vehicle: Vehicle,
    accel: float
) -> Vehicle:

    new_v = max(
        0.0,
        vehicle.v + accel * DT
    )

    new_x = (
        vehicle.x
        + (vehicle.v + new_v) * DT / 2
    )

    return Vehicle(
        x=new_x,
        v=new_v
    )


# ==========================================================
# ACC SUPERVISOR
# ==========================================================

class Supervisor:

    def __init__(
        self,
        cfg: Config
    ) -> None:

        self.cfg = cfg
        self.mode = Mode.TARGET_LOST

    def command(
        self,
        ego_v: float,
        gap: float | None,
        lead_v: float | None,
        road_clear: bool = False
    ) -> tuple[Mode, float]:

        c = self.cfg

        valid_target = (
            gap is not None
            and lead_v is not None
            and gap > 0
        )

        # --------------------------------------------------
        # Valid lead vehicle
        # --------------------------------------------------

        if valid_target:

            assert gap is not None
            assert lead_v is not None

            closing_speed = max(
                0.0,
                ego_v - lead_v
            )

            if closing_speed > 1e-9:

                ttc = gap / closing_speed

            else:

                ttc = float("inf")

            # Emergency condition

            if (
                gap <= c.emergency_gap_m
                or ttc < c.emergency_ttc_s
            ):

                self.mode = Mode.EMERGENCY_BRAKE

                return (
                    self.mode,
                    c.max_decel
                )

            # Normal following

            self.mode = Mode.FOLLOW

            desired_gap = (
                c.standstill_gap
                + c.headway * ego_v
            )

            gap_error = gap - desired_gap

            relative_speed = lead_v - ego_v

            following_accel = (
                c.gap_gain * gap_error
                + c.relative_speed_gain * relative_speed
            )

            cruise_accel = (
                c.speed_gain
                * (c.cruise_speed - ego_v)
            )

            requested_accel = min(
                following_accel,
                cruise_accel
            )

        # --------------------------------------------------
        # Confirmed clear road
        # --------------------------------------------------

        elif road_clear:

            self.mode = Mode.CRUISE

            requested_accel = (
                c.speed_gain
                * (c.cruise_speed - ego_v)
            )

        # --------------------------------------------------
        # Target lost / uncertain perception
        # --------------------------------------------------

        else:

            self.mode = Mode.TARGET_LOST

            if ego_v > 0.05:

                requested_accel = c.lost_decel

            else:

                requested_accel = 0.0

        # --------------------------------------------------
        # Apply acceleration limits
        # --------------------------------------------------

        acceleration = max(
            c.max_decel,
            min(
                c.max_accel,
                requested_accel
            )
        )

        return (
            self.mode,
            acceleration
        )


# ==========================================================
# SCENARIO DEFINITIONS
# ==========================================================

def scenario_conditions(
    name: str,
    t: float
) -> tuple[bool, bool, float]:

    """
    Returns:
        target_visible
        road_clear
        lead_acceleration
    """

    if name == "free_road":

        return (
            False,
            True,
            0.0
        )

    if name == "slower_lead":

        return (
            True,
            False,
            0.0
        )

    if name == "lead_braking":

        acceleration = (
            -2.0
            if 10.0 <= t < 13.0
            else 0.0
        )

        return (
            True,
            False,
            acceleration
        )

    if name == "target_loss":

        return (
            t < 15.0,
            False,
            0.0
        )

    if name == "reacquisition":

        visible = (
            t < 12.0
            or t >= 16.0
        )

        return (
            visible,
            False,
            0.0
        )

    if name == "emergency":

        acceleration = (
            -4.0
            if 3.0 <= t < 5.0
            else 0.0
        )

        return (
            True,
            False,
            acceleration
        )

    raise ValueError(
        f"Unknown scenario: {name}"
    )


# ==========================================================
# SIMULATION
# ==========================================================

def simulate(
    name: str,
    ego_v: float,
    lead_v: float,
    gap: float
) -> dict[str, object]:

    supervisor = Supervisor(
        Config()
    )

    ego = Vehicle(
        x=0.0,
        v=ego_v
    )

    lead = Vehicle(
        x=gap,
        v=lead_v
    )

    rows: list[
        dict[str, object]
    ] = []

    overlap = False

    min_gap = float("inf")

    for step in range(
        round(DURATION_S / DT)
    ):

        t = step * DT

        target_visible, road_clear, lead_a = (
            scenario_conditions(name, t)
        )

        actual_gap = lead.x - ego.x

        mode, accel = supervisor.command(
            ego_v=ego.v,
            gap=actual_gap if target_visible else None,
            lead_v=lead.v if target_visible else None,
            road_clear=road_clear
        )

        desired_gap = (
            5.0 + 1.8 * ego.v
        )

        rows.append({
            "time_s": round(t, 3),
            "mode": mode.value,
            "target_visible": target_visible,
            "road_clear": road_clear,
            "ego_speed_mps": ego.v,
            "lead_speed_mps": lead.v,
            "center_gap_m": actual_gap,
            "desired_gap_m": desired_gap,
            "command_accel_mps2": accel
        })

        min_gap = min(
            min_gap,
            actual_gap
        )

        # Update vehicle states

        ego = advance(
            ego,
            accel
        )

        lead = advance(
            lead,
            lead_a
        )

        if lead.x <= ego.x:

            overlap = True
            break

    # ------------------------------------------------------
    # Save telemetry
    # ------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / f"{name}.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------

    accelerations = [
        float(row["command_accel_mps2"])
        for row in rows
        if isinstance(row["command_accel_mps2"], (int, float))
    ]

    modes = sorted({
        str(row["mode"])
        for row in rows
    })

    return {
        "scenario": name,
        "steps": len(rows),
        "overlap": overlap,
        "min_center_gap_m": round(min_gap, 3),
        "min_accel_mps2": round(
            min(accelerations),
            3
        ),
        "max_accel_mps2": round(
            max(accelerations),
            3
        ),
        "modes": ",".join(modes),
        "final_ego_speed_mps": round(
            ego.v,
            3
        )
    }


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:

    print("=" * 65)
    print("RADAR-CAMERA FUSION ADAS PROJECT")
    print("PHASE 3C: ACC SUPERVISOR VALIDATION")
    print("=" * 65)

    scenarios = [
        (
            "free_road",
            0.0,
            15.0,
            500.0
        ),
        (
            "slower_lead",
            10.0,
            8.0,
            45.0
        ),
        (
            "lead_braking",
            12.0,
            12.0,
            55.0
        ),
        (
            "target_loss",
            10.0,
            12.0,
            70.0
        ),
        (
            "reacquisition",
            10.0,
            12.0,
            70.0
        ),
        (
            "emergency",
            15.0,
            7.0,
            22.0
        )
    ]

    results = [
        simulate(*scenario)
        for scenario in scenarios
    ]

    # ------------------------------------------------------
    # Save summary
    # ------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "scenario_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0])
        )

        writer.writeheader()
        writer.writerows(results)

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    for result in results:

        print("-" * 65)

        print(
            f"Scenario: {result['scenario']}"
        )

        print(
            f"Steps: {result['steps']}"
        )

        print(
            f"Vehicle-center overlap: {result['overlap']}"
        )

        print(
            f"Minimum center gap: "
            f"{result['min_center_gap_m']} m"
        )

        print(
            f"Acceleration range: "
            f"{result['min_accel_mps2']} to "
            f"{result['max_accel_mps2']} m/s²"
        )

        print(
            f"Modes: {result['modes']}"
        )

        print(
            f"Final ego speed: "
            f"{result['final_ego_speed_mps']} m/s"
        )

    print("=" * 65)
    print("PHASE 3C VALIDATION COMPLETE")
    print("=" * 65)

    print(
        f"Results saved to: {OUTPUT_DIR}"
    )

    if any(
        bool(result["overlap"])
        for result in results
    ):

        print(
            "WARNING: At least one scenario "
            "experienced vehicle-center overlap."
        )


if __name__ == "__main__":
    main()