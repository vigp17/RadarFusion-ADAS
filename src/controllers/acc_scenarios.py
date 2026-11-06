"""
Radar-Camera Fusion ADAS Project
Phase 3B: Offline ACC Scenario Validation

Scenarios:
1. Free-road cruising
2. Slower lead vehicle
3. Lead-vehicle braking
4. Target loss

Uses the existing Phase 3A ACC controller.

This is a simulation validation tool, not a
production automotive safety system.
"""

from __future__ import annotations

import csv

from dataclasses import dataclass
from pathlib import Path

from acc_controller import (
    ACCConfig,
    ACCController,
    Vehicle,
    advance,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

DT = 0.05
DURATION = 35.0

OUTPUT = Path("results/acc_phase3b")


# ==========================================================
# SCENARIO DEFINITION
# ==========================================================

@dataclass(frozen=True)
class Scenario:
    name: str
    ego_speed: float
    lead_speed: float
    gap: float


# ==========================================================
# LEAD VEHICLE BEHAVIOR
# ==========================================================

def lead_accel(name: str, t: float) -> float:

    if name == "lead_braking":

        if 10.0 <= t < 13.0:
            return -2.0

    return 0.0


# ==========================================================
# RUN SCENARIO
# ==========================================================

def run_scenario(
    scenario: Scenario
) -> dict[str, float | str | bool]:

    controller = ACCController(
        ACCConfig()
    )

    ego = Vehicle(
        x=0.0,
        v=scenario.ego_speed
    )

    lead = Vehicle(
        x=scenario.gap,
        v=scenario.lead_speed
    )

    records: list[dict[str, float | str]] = []

    min_gap = float("inf")
    max_gap_error = 0.0

    max_accel = float("-inf")
    min_accel = float("inf")

    overlap = False

    for step in range(round(DURATION / DT)):

        t = step * DT

        # Free-road scenario has no tracked lead.
        # Target-loss scenario removes the lead
        # measurement after 15 seconds.

        target_visible = (
            scenario.name != "free_road"
            and not (
                scenario.name == "target_loss"
                and t >= 15.0
            )
        )

        actual_gap = lead.x - ego.x

        desired = controller.desired_gap(
            ego.v
        )

        commanded = controller.acceleration(
            ego.v,
            actual_gap if target_visible else None,
            lead.v if target_visible else None
        )

        records.append({
            "scenario": scenario.name,
            "time_s": t,
            "ego_speed_mps": ego.v,
            "lead_speed_mps": lead.v,
            "center_gap_m": actual_gap,
            "desired_gap_m": desired,
            "acceleration_mps2": commanded,
            "target_visible": str(target_visible)
        })

        min_gap = min(
            min_gap,
            actual_gap
        )

        max_gap_error = max(
            max_gap_error,
            abs(actual_gap - desired)
        )

        max_accel = max(
            max_accel,
            commanded
        )

        min_accel = min(
            min_accel,
            commanded
        )

        # Update longitudinal vehicle states.

        ego = advance(
            ego,
            commanded
        )

        lead = advance(
            lead,
            lead_accel(scenario.name, t)
        )

        # Vehicle-center overlap check.
        # This is not a bumper-clearance calculation.

        if lead.x <= ego.x:
            overlap = True
            break

    # Save scenario telemetry.

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    path = OUTPUT / f"{scenario.name}.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0])
        )

        writer.writeheader()
        writer.writerows(records)

    limits_ok = (
        min_accel >= -4.0 - 1e-9
        and max_accel <= 2.0 + 1e-9
    )

    return {
        "scenario": scenario.name,
        "steps": len(records),
        "min_center_gap_m": round(min_gap, 3),
        "max_abs_gap_error_m": round(max_gap_error, 3),
        "min_accel_mps2": round(min_accel, 3),
        "max_accel_mps2": round(max_accel, 3),
        "overlap": overlap,
        "limits_ok": limits_ok,
        "final_ego_speed_mps": round(ego.v, 3)
    }


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:

    print("=" * 65)
    print("RADAR-CAMERA FUSION ADAS PROJECT")
    print("PHASE 3B: ACC SCENARIO VALIDATION")
    print("=" * 65)

    scenarios = [
        Scenario(
            "free_road",
            0.0,
            15.0,
            500.0
        ),

        Scenario(
            "slower_lead",
            10.0,
            8.0,
            45.0
        ),

        Scenario(
            "lead_braking",
            12.0,
            12.0,
            55.0
        ),

        Scenario(
            "target_loss",
            10.0,
            12.0,
            70.0
        )
    ]

    results = [
        run_scenario(scenario)
        for scenario in scenarios
    ]

    summary_path = OUTPUT / "scenario_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(results[0])
        )

        writer.writeheader()
        writer.writerows(results)

    for result in results:

        print("-" * 65)

        print(
            f"Scenario: {result['scenario']}"
        )

        print(
            f"Steps: {result['steps']}"
        )

        print(
            f"Minimum center gap: "
            f"{result['min_center_gap_m']} m"
        )

        print(
            f"Acceleration limits respected: "
            f"{result['limits_ok']}"
        )

        print(
            f"Vehicle-center overlap: "
            f"{result['overlap']}"
        )

        print(
            f"Final ego speed: "
            f"{result['final_ego_speed_mps']} m/s"
        )

    print("=" * 65)
    print("ACC SCENARIO VALIDATION COMPLETE")
    print("=" * 65)

    print(
        f"Results saved to: {OUTPUT}"
    )


if __name__ == "__main__":
    main()