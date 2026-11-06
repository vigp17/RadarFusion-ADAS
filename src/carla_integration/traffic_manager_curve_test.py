"""
Radar-Camera Fusion ADAS Project
Phase 2G: Full Curve-Following Validation

CARLA 0.9.16
Map: Town10HD_Opt

Ego vehicle:
    Tesla Model 3, stationary

Lead vehicle:
    Audi TT, controlled by CARLA Traffic Manager

Objectives:
    - Verify lead vehicle follows the road curve
    - Monitor vehicle heading and speed
    - Monitor road and lane IDs
    - Measure distance from lane center
    - Record complete vehicle trajectory
    - Check curve completion
    - Restore CARLA settings and clean up actors
"""

from __future__ import annotations

import csv
import math
import time

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import carla  # type: ignore[import-untyped]


# ==========================================================
# CONFIGURATION
# ==========================================================

HOST = "127.0.0.1"
PORT = 2000
TM_PORT = 8000

FIXED_DT = 0.05

TOTAL_FRAMES = 800

SPAWN_INDEX = 61

LEAD_AHEAD_M = 30.0

# Traffic Manager speed reduction percentage.
# This does not guarantee an exact target speed.
TM_SPEED_REDUCTION_PERCENT = 40.0

LANE_CENTER_WARNING_M = 1.5

# Approximate heading change used as an initial
# indicator of completing the major road curve.
CURVE_HEADING_THRESHOLD_DEG = 75.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

OUTPUT_DIR = (
    RESULTS_DIR / f"tm_full_curve_{RUN_ID}"
)

OUTPUT_CSV = (
    OUTPUT_DIR / "lead_trajectory.csv"
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def normalize_angle_deg(angle: float) -> float:
    """Normalize angle to [-180, 180)."""

    return (angle + 180.0) % 360.0 - 180.0


def spawn_vehicle(
    world: Any,
    blueprint: Any,
    transform: Any
) -> carla.Vehicle:

    actor = world.try_spawn_actor(
        blueprint,
        transform
    )

    if actor is None:
        raise RuntimeError(
            "Vehicle spawn failed. "
            "Verify that the spawn location is clear."
        )

    if not isinstance(actor, carla.Vehicle):
        actor.destroy()

        raise TypeError(
            "Spawned actor is not a CARLA Vehicle."
        )

    return cast(carla.Vehicle, actor)


def update_spectator_camera(
    world: Any,
    vehicle: carla.Vehicle
) -> None:

    transform = vehicle.get_transform()

    forward = transform.get_forward_vector()

    location = transform.location

    camera_location = carla.Location(
        x=location.x - 9.0 * forward.x,
        y=location.y - 9.0 * forward.y,
        z=location.z + 5.0
    )

    camera_rotation = carla.Rotation(
        pitch=-20.0,
        yaw=transform.rotation.yaw,
        roll=0.0
    )

    world.get_spectator().set_transform(
        carla.Transform(
            camera_location,
            camera_rotation
        )
    )


def calculate_lane_center_error(
    vehicle_transform: Any,
    waypoint: Any
) -> float:

    """
    Calculate signed lateral displacement from
    the nearest road waypoint center.

    This is a geometric approximation.
    """

    vehicle_location = vehicle_transform.location

    waypoint_transform = waypoint.transform

    reference_location = waypoint_transform.location

    yaw_rad = math.radians(
        waypoint_transform.rotation.yaw
    )

    dx = vehicle_location.x - reference_location.x
    dy = vehicle_location.y - reference_location.y

    lateral_error = (
        -dx * math.sin(yaw_rad)
        + dy * math.cos(yaw_rad)
    )

    return float(lateral_error)


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================

def main() -> None:

    print("=" * 70)
    print("RADAR-CAMERA FUSION ADAS PROJECT")
    print("PHASE 2G: FULL CURVE-FOLLOWING VALIDATION")
    print("=" * 70)

    client = carla.Client(
        HOST,
        PORT
    )

    client.set_timeout(30.0)

    world = client.get_world()

    carla_map = world.get_map()

    if "Town10HD_Opt" not in carla_map.name:
        raise RuntimeError(
            "Load Town10HD_Opt before running."
        )

    spawn_points = carla_map.get_spawn_points()

    if len(spawn_points) <= SPAWN_INDEX:
        raise RuntimeError(
            "Spawn point 61 is unavailable."
        )

    traffic_manager = client.get_trafficmanager(
        TM_PORT
    )

    original_settings = world.get_settings()

    ego_vehicle: carla.Vehicle | None = None
    lead_vehicle: carla.Vehicle | None = None

    settings_changed = False
    tm_sync_changed = False

    try:

        # --------------------------------------------------
        # Configure synchronous simulation
        # --------------------------------------------------

        settings = world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT

        world.apply_settings(settings)

        settings_changed = True

        traffic_manager.set_synchronous_mode(True)

        tm_sync_changed = True

        library = world.get_blueprint_library()

        # --------------------------------------------------
        # Spawn Tesla Model 3
        # --------------------------------------------------

        ego_vehicle = spawn_vehicle(
            world,
            library.find("vehicle.tesla.model3"),
            spawn_points[SPAWN_INDEX]
        )

        ego_vehicle.set_autopilot(False)

        ego_vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True
            )
        )

        print(
            f"Ego Tesla spawned: {ego_vehicle.id}"
        )

        # --------------------------------------------------
        # Find lead-vehicle location
        # --------------------------------------------------

        start_waypoint = carla_map.get_waypoint(
            spawn_points[SPAWN_INDEX].location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if start_waypoint is None:
            raise RuntimeError(
                "No driving waypoint at ego spawn."
            )

        next_waypoints = start_waypoint.next(
            LEAD_AHEAD_M
        )

        if len(next_waypoints) != 1:
            raise RuntimeError(
                "Ambiguous lead-vehicle spawn location."
            )

        lead_waypoint = next_waypoints[0]

        if (
            lead_waypoint.is_junction
            or lead_waypoint.lane_id != start_waypoint.lane_id
        ):
            raise RuntimeError(
                "Lead spawn is not on the expected lane."
            )

        lead_transform = lead_waypoint.transform

        lead_transform.location.z += 0.3

        # --------------------------------------------------
        # Spawn Audi TT
        # --------------------------------------------------

        lead_vehicle = spawn_vehicle(
            world,
            library.find("vehicle.audi.tt"),
            lead_transform
        )

        print(
            f"Lead Audi spawned: {lead_vehicle.id}"
        )

        # --------------------------------------------------
        # Configure Traffic Manager
        # --------------------------------------------------

        lead_vehicle.set_autopilot(
            True,
            TM_PORT
        )

        traffic_manager.auto_lane_change(
            lead_vehicle,
            False
        )

        traffic_manager.distance_to_leading_vehicle(
            lead_vehicle,
            5.0
        )

        traffic_manager.vehicle_percentage_speed_difference(
            lead_vehicle,
            TM_SPEED_REDUCTION_PERCENT
        )

        print(
            "Traffic Manager enabled for Audi."
        )

        # --------------------------------------------------
        # Spectator camera
        # --------------------------------------------------

        update_spectator_camera(
            world,
            ego_vehicle
        )

        # --------------------------------------------------
        # Prepare telemetry
        # --------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=False
        )

        initial_heading: float | None = None

        max_heading_change = 0.0
        max_lateral_error = 0.0

        lane_warning_count = 0
        non_driving_waypoint_count = 0

        observed_road_ids: set[int] = set()
        observed_lane_ids: set[int] = set()

        final_heading_change = 0.0
        final_speed = 0.0

        # --------------------------------------------------
        # Simulation
        # --------------------------------------------------

        with OUTPUT_CSV.open(
            "x",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "frame",
                "simulation_time_s",
                "lead_x_m",
                "lead_y_m",
                "lead_z_m",
                "lead_yaw_deg",
                "heading_change_deg",
                "lead_speed_mps",
                "ego_lead_distance_m",
                "road_id",
                "lane_id",
                "lane_center_error_m",
                "is_junction"
            ])

            for step in range(TOTAL_FRAMES):

                frame = world.tick()

                transform = lead_vehicle.get_transform()

                location = transform.location

                speed = lead_vehicle.get_velocity().length()

                yaw = float(
                    transform.rotation.yaw
                )

                if initial_heading is None:
                    initial_heading = yaw

                heading_change = normalize_angle_deg(
                    yaw - initial_heading
                )

                final_heading_change = heading_change
                final_speed = speed

                max_heading_change = max(
                    max_heading_change,
                    abs(heading_change)
                )

                distance = ego_vehicle.get_location().distance(
                    location
                )

                waypoint = carla_map.get_waypoint(
                    location,
                    project_to_road=False,
                    lane_type=carla.LaneType.Driving
                )

                road_id = ""
                lane_id = ""
                lane_error = float("nan")
                is_junction = ""

                if waypoint is not None:

                    road_id = waypoint.road_id
                    lane_id = waypoint.lane_id

                    observed_road_ids.add(
                        int(road_id)
                    )

                    observed_lane_ids.add(
                        int(lane_id)
                    )

                    is_junction = waypoint.is_junction

                    lane_error = calculate_lane_center_error(
                        transform,
                        waypoint
                    )

                    max_lateral_error = max(
                        max_lateral_error,
                        abs(lane_error)
                    )

                    if (
                        abs(lane_error)
                        > LANE_CENTER_WARNING_M
                    ):
                        lane_warning_count += 1

                else:
                    non_driving_waypoint_count += 1

                writer.writerow([
                    frame,
                    step * FIXED_DT,
                    location.x,
                    location.y,
                    location.z,
                    yaw,
                    heading_change,
                    speed,
                    distance,
                    road_id,
                    lane_id,
                    lane_error,
                    is_junction
                ])

                update_spectator_camera(
                    world,
                    ego_vehicle
                )

                if step % 20 == 0:

                    lane_text = (
                        f"{lane_error:+.2f} m"
                        if math.isfinite(lane_error)
                        else "Unavailable"
                    )

                    print(
                        f"Step {step:03d} | "
                        f"Speed={speed:.2f} m/s | "
                        f"Heading change={heading_change:+.1f} deg | "
                        f"Lane error={lane_text} | "
                        f"Road={road_id} | "
                        f"Lane={lane_id}"
                    )

        # --------------------------------------------------
        # Final summary
        # --------------------------------------------------

        heading_threshold_reached = (
            max_heading_change
            >= CURVE_HEADING_THRESHOLD_DEG
        )

        print("=" * 70)
        print("FULL CURVE TEST SUMMARY")
        print("=" * 70)

        print(
            f"Simulation duration     : "
            f"{TOTAL_FRAMES * FIXED_DT:.2f} s"
        )

        print(
            f"Maximum heading change  : "
            f"{max_heading_change:.2f} deg"
        )

        print(
            f"Final heading change    : "
            f"{final_heading_change:.2f} deg"
        )

        print(
            f"Final lead speed        : "
            f"{final_speed:.2f} m/s"
        )

        print(
            f"Maximum lane-center error: "
            f"{max_lateral_error:.3f} m"
        )

        print(
            f"Lane-center warnings    : "
            f"{lane_warning_count}"
        )

        print(
            f"No driving waypoint     : "
            f"{non_driving_waypoint_count} frames"
        )

        print(
            f"Observed road IDs       : "
            f"{sorted(observed_road_ids)}"
        )

        print(
            f"Observed lane IDs       : "
            f"{sorted(observed_lane_ids)}"
        )

        print(
            f"75-degree heading threshold reached: "
            f"{heading_threshold_reached}"
        )

        print(
            f"Trajectory saved        : {OUTPUT_CSV}"
        )

        print("=" * 70)

        if heading_threshold_reached:

            print(
                "The Audi achieved a substantial heading change."
            )

            print(
                "Review lane continuity and the trajectory "
                "before confirming full curve completion."
            )

        else:

            print(
                "The Audi has not yet demonstrated "
                "the required heading change."
            )

        time.sleep(5.0)

    finally:

        # --------------------------------------------------
        # Clean up lead vehicle
        # --------------------------------------------------

        if lead_vehicle is not None:

            try:

                if lead_vehicle.is_alive:

                    lead_vehicle.set_autopilot(
                        False,
                        TM_PORT
                    )

                    lead_vehicle.destroy()

            except Exception as exc:

                print(
                    f"Lead cleanup warning: {exc}"
                )

        # --------------------------------------------------
        # Clean up Tesla
        # --------------------------------------------------

        if ego_vehicle is not None:

            try:

                if ego_vehicle.is_alive:
                    ego_vehicle.destroy()

            except Exception as exc:

                print(
                    f"Ego cleanup warning: {exc}"
                )

        # --------------------------------------------------
        # Restore Traffic Manager
        # --------------------------------------------------

        if tm_sync_changed:

            try:

                traffic_manager.set_synchronous_mode(
                    False
                )

            except Exception as exc:

                print(
                    f"Traffic Manager restoration warning: {exc}"
                )

        # --------------------------------------------------
        # Restore CARLA world settings
        # --------------------------------------------------

        if settings_changed:

            try:

                world.apply_settings(
                    original_settings
                )

                print(
                    "Original CARLA settings restored."
                )

            except Exception as exc:

                print(
                    f"World restoration warning: {exc}"
                )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()