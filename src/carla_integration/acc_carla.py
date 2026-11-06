"""
Radar-Camera Fusion ADAS Project
Phase 3G: ACC Stop-and-Hold Validation

CARLA 0.9.16
Map: Town10HD_Opt

Ego:
    Tesla Model 3
    Stanley lateral control
    ACC longitudinal control
    Stop-and-hold supervisor

Lead:
    Audi TT
    CARLA Traffic Manager

Ground-truth route-relative target association.

Simulation only.
"""

from __future__ import annotations

import csv
import math
import sys

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import carla  # type: ignore[import-untyped]


# ==========================================================
# PROJECT IMPORTS
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from carla_integration.carla_adapter import CarlaControllerAdapter
from controllers.stanley_controller import StanleyController
from controllers.acc_supervisor import Config, Supervisor
from vehicle import Vehicle


# ==========================================================
# CONFIGURATION
# ==========================================================

HOST = "127.0.0.1"
PORT = 2000
TM_PORT = 8000

DT = 0.05
FRAMES = 800

SPAWN_INDEX = 61
LEAD_AHEAD = 35.0

MAX_CTE = 1.5
MAX_LEAD_ROUTE_ERROR = 2.0

MIN_CLEARANCE = 5.0

STOP_TARGET_GAP = 9.0
STOP_LEAD_SPEED = 0.3
HOLD_EGO_SPEED = 0.15

MAX_STEER = math.radians(30.0)
MAX_STEER_RATE = math.radians(20.0)

OUT = (
    ROOT
    / "results"
    / (
        "acc_stop_hold_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
)


# ==========================================================
# HELPERS
# ==========================================================

def clip(
    value: float,
    minimum: float,
    maximum: float
) -> float:

    return max(minimum, min(maximum, value))


def spawn(
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
            "Vehicle spawn failed. Check spawn clearance."
        )

    if not isinstance(actor, carla.Vehicle):
        actor.destroy()
        raise TypeError("Expected CARLA vehicle.")

    return cast(carla.Vehicle, actor)


# ==========================================================
# ROUTE GENERATION
# ==========================================================

def route_from(
    start: Any,
    distance: float = 180.0,
    spacing: float = 0.5
) -> list[Any]:

    route: list[Any] = [start]

    current = start

    for _ in range(round(distance / spacing)):

        options = [
            waypoint
            for waypoint in current.next(spacing)
            if (
                not waypoint.is_junction
                and waypoint.lane_id == start.lane_id
            )
        ]

        if len(options) != 1:
            break

        current = options[0]
        route.append(current)

    if len(route) < 100:
        raise RuntimeError(
            "Reference route is too short."
        )

    return route


# ==========================================================
# ROUTE-RELATIVE POSITION
# ==========================================================

def project_route(
    x: float,
    y: float,
    road_x: np.ndarray,
    road_y: np.ndarray,
    arc_length: np.ndarray
) -> tuple[float, float, int]:

    px = road_x[:-1]
    py = road_y[:-1]

    vx = np.diff(road_x)
    vy = np.diff(road_y)

    denominator = vx * vx + vy * vy

    projection = np.clip(
        (
            (x - px) * vx
            + (y - py) * vy
        ) / np.maximum(denominator, 1e-9),
        0.0,
        1.0
    )

    qx = px + projection * vx
    qy = py + projection * vy

    distance_squared = (
        (x - qx) ** 2
        + (y - qy) ** 2
    )

    index = int(np.argmin(distance_squared))

    segment_length = math.sqrt(
        float(denominator[index])
    )

    lateral_error = (
        float(vx[index]) * (y - float(qy[index]))
        - float(vy[index]) * (x - float(qx[index]))
    ) / max(segment_length, 1e-9)

    progress = float(
        arc_length[index]
        + projection[index] * segment_length
    )

    return progress, float(lateral_error), index


# ==========================================================
# SPECTATOR CAMERA
# ==========================================================

def spectator(
    world: Any,
    ego: carla.Vehicle
) -> None:

    transform = ego.get_transform()

    forward = transform.get_forward_vector()
    location = transform.location

    camera_location = carla.Location(
        x=location.x - 8.0 * forward.x,
        y=location.y - 8.0 * forward.y,
        z=location.z + 4.5
    )

    camera_rotation = carla.Rotation(
        pitch=-20.0,
        yaw=transform.rotation.yaw
    )

    world.get_spectator().set_transform(
        carla.Transform(
            camera_location,
            camera_rotation
        )
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:

    print("=" * 65)
    print("PHASE 3G: ACC STOP-AND-HOLD VALIDATION")
    print("=" * 65)

    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    world = client.get_world()
    carla_map = world.get_map()

    if "Town10HD_Opt" not in carla_map.name:
        raise RuntimeError(
            "Load Town10HD_Opt first."
        )

    spawn_points = carla_map.get_spawn_points()

    if len(spawn_points) <= SPAWN_INDEX:
        raise RuntimeError(
            "Spawn point 61 unavailable."
        )

    traffic_manager = client.get_trafficmanager(
        TM_PORT
    )

    original_settings = world.get_settings()

    ego: carla.Vehicle | None = None
    lead: carla.Vehicle | None = None
    collision: carla.Sensor | None = None

    collision_events: list[Any] = []

    settings_changed = False
    tm_changed = False

    termination_reason = "not_started"

    active_steps = 0
    hold_steps = 0

    try:

        # --------------------------------------------------
        # Synchronous simulation
        # --------------------------------------------------

        settings = world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT

        world.apply_settings(settings)
        settings_changed = True

        traffic_manager.set_synchronous_mode(True)
        tm_changed = True

        library = world.get_blueprint_library()

        # --------------------------------------------------
        # Reference route
        # --------------------------------------------------

        start = carla_map.get_waypoint(
            spawn_points[SPAWN_INDEX].location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if start is None:
            raise RuntimeError(
                "No starting driving waypoint."
            )

        route = route_from(start)

        print(
            f"Reference route: {len(route)} waypoints"
        )

        lead_waypoints = start.next(LEAD_AHEAD)

        if (
            len(lead_waypoints) != 1
            or lead_waypoints[0].is_junction
            or lead_waypoints[0].lane_id != start.lane_id
        ):
            raise RuntimeError(
                "Ambiguous lead spawn location."
            )

        # --------------------------------------------------
        # Spawn Tesla
        # --------------------------------------------------

        ego = spawn(
            world,
            library.find("vehicle.tesla.model3"),
            spawn_points[SPAWN_INDEX]
        )

        ego.set_autopilot(False)

        ego.apply_control(
            carla.VehicleControl(brake=1.0)
        )

        print(f"Ego Tesla spawned: {ego.id}")

        # --------------------------------------------------
        # Spawn Audi
        # --------------------------------------------------

        lead_transform = lead_waypoints[0].transform
        lead_transform.location.z += 0.3

        lead = spawn(
            world,
            library.find("vehicle.audi.tt"),
            lead_transform
        )

        lead.set_autopilot(True, TM_PORT)

        traffic_manager.auto_lane_change(
            lead,
            False
        )

        traffic_manager.vehicle_percentage_speed_difference(
            lead,
            40.0
        )

        print(f"Lead Audi spawned: {lead.id}")

        # --------------------------------------------------
        # Collision sensor
        # --------------------------------------------------

        collision_actor = world.spawn_actor(
            library.find("sensor.other.collision"),
            carla.Transform(),
            attach_to=ego
        )

        if not isinstance(collision_actor, carla.Sensor):
            collision_actor.destroy()
            raise TypeError(
                "Expected collision sensor."
            )

        collision = cast(carla.Sensor, collision_actor)

        collision.listen(
            lambda event: collision_events.append(event)
        )

        # --------------------------------------------------
        # Settle vehicles
        # --------------------------------------------------

        for _ in range(10):
            world.tick()

        # --------------------------------------------------
        # Initialize controllers
        # --------------------------------------------------

        adapter = CarlaControllerAdapter(ego)

        road_x, road_y, road_yaw = (
            adapter.convert_waypoints(route)
        )

        segment_lengths = np.hypot(
            np.diff(road_x),
            np.diff(road_y)
        )

        arc_length = np.concatenate([
            np.array([0.0]),
            np.cumsum(segment_lengths)
        ])

        stanley = StanleyController(
            stanley_gain=1.5,
            epsilon=1e-3
        )

        acc = Supervisor(
            Config(cruise_speed=8.0)
        )

        previous_steering = 0.0

        print(
            f"Tesla wheelbase: {adapter.wheelbase:.4f} m"
        )

        # --------------------------------------------------
        # Telemetry
        # --------------------------------------------------

        OUT.mkdir(
            parents=True,
            exist_ok=False
        )

        output_csv = OUT / "telemetry.csv"

        termination_reason = "duration_complete"

        with output_csv.open(
            "x",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "step",
                "frame",
                "ego_speed_mps",
                "lead_speed_mps",
                "route_gap_m",
                "approx_route_clearance_m",
                "lead_route_error_m",
                "ego_cte_m",
                "ego_progress_m",
                "lead_progress_m",
                "acc_mode",
                "accel_cmd_mps2",
                "throttle",
                "brake",
                "carla_steer",
                "collision_events",
                "stop_hold_state"
            ])

            # ----------------------------------------------
            # Simulation loop
            # ----------------------------------------------

            for step in range(FRAMES):

                frame = world.tick()

                if collision_events:
                    termination_reason = "collision_detected"
                    break

                # ------------------------------------------
                # Ego state
                # ------------------------------------------

                rear_x, rear_y, yaw = (
                    adapter.get_rear_axle_pose()
                )

                ego_speed = adapter.get_speed(
                    planar_only=True
                )

                vehicle_state = Vehicle(
                    x=rear_x,
                    y=rear_y,
                    yaw=yaw,
                    wheelbase=adapter.wheelbase
                )

                details = stanley.compute_with_details(
                    vehicle=vehicle_state,
                    road_x=road_x,
                    road_y=road_y,
                    velocity=ego_speed,
                    road_yaw=road_yaw
                )

                cte = float(
                    details["cross_track_error"]
                )

                requested_steering = float(
                    details["steering"]
                )

                # ------------------------------------------
                # Route-relative positions
                # ------------------------------------------

                ego_transform = ego.get_transform()
                lead_transform = lead.get_transform()

                ego_x = ego_transform.location.x
                ego_y = -ego_transform.location.y

                lead_x = lead_transform.location.x
                lead_y = -lead_transform.location.y

                ego_progress, _, _ = project_route(
                    ego_x,
                    ego_y,
                    road_x,
                    road_y,
                    arc_length
                )

                lead_progress, lead_route_error, lead_index = (
                    project_route(
                        lead_x,
                        lead_y,
                        road_x,
                        road_y,
                        arc_length
                    )
                )

                route_gap = lead_progress - ego_progress

                clearance = (
                    route_gap
                    - ego.bounding_box.extent.x
                    - lead.bounding_box.extent.x
                )

                # ------------------------------------------
                # Safety checks
                # ------------------------------------------

                values = (
                    ego_speed,
                    cte,
                    requested_steering,
                    route_gap,
                    lead_route_error,
                    clearance
                )

                if not all(
                    math.isfinite(value)
                    for value in values
                ):
                    termination_reason = "invalid_measurement"
                    break

                if abs(cte) > MAX_CTE:
                    termination_reason = "lateral_error_abort"
                    break

                if (
                    abs(lead_route_error) > MAX_LEAD_ROUTE_ERROR
                    or lead_progress >= float(arc_length[-1]) - 2.0
                ):
                    termination_reason = "lead_route_unverified"
                    break

                if ego_progress >= float(arc_length[-1]) - 5.0:
                    termination_reason = "ego_route_endpoint"
                    break

                if (
                    route_gap <= 0.0
                    or clearance <= MIN_CLEARANCE
                ):
                    termination_reason = "minimum_clearance_abort"
                    break

                # ------------------------------------------
                # Lead speed along route
                # ------------------------------------------

                lead_velocity = lead.get_velocity()

                tangent_x = float(
                    road_x[lead_index + 1]
                    - road_x[lead_index]
                )

                tangent_y = float(
                    road_y[lead_index + 1]
                    - road_y[lead_index]
                )

                tangent_norm = max(
                    math.hypot(tangent_x, tangent_y),
                    1e-9
                )

                lead_speed = (
                    lead_velocity.x * tangent_x
                    - lead_velocity.y * tangent_y
                ) / tangent_norm

                # ------------------------------------------
                # ACC supervisor
                # ------------------------------------------

                mode, acceleration = acc.command(
                    ego_v=ego_speed,
                    gap=clearance,
                    lead_v=lead_speed,
                    road_clear=False
                )

                # ------------------------------------------
                # Stop-and-hold supervisor
                # ------------------------------------------

                stop_state = "FOLLOW"

                if abs(lead_speed) <= STOP_LEAD_SPEED:

                    remaining = max(
                        0.0,
                        clearance - STOP_TARGET_GAP
                    )

                    target_speed = min(
                        1.5,
                        math.sqrt(
                            2.0 * 1.5 * remaining
                        )
                    )

                    if clearance <= STOP_TARGET_GAP + 0.5:
                        target_speed = 0.0

                    stop_acceleration = clip(
                        1.2 * (target_speed - ego_speed),
                        -3.0,
                        0.0
                    )

                    acceleration = min(
                        acceleration,
                        stop_acceleration
                    )

                    stop_state = "STOPPING"

                    if (
                        ego_speed <= HOLD_EGO_SPEED
                        and clearance <= STOP_TARGET_GAP + 1.0
                    ):

                        stop_state = "HOLD"

                        acceleration = -2.0

                        hold_steps += 1

                # ------------------------------------------
                # Stanley steering constraints
                # ------------------------------------------

                max_delta = MAX_STEER_RATE * DT

                angle_limited = clip(
                    requested_steering,
                    -MAX_STEER,
                    MAX_STEER
                )

                applied_steering = clip(
                    angle_limited,
                    previous_steering - max_delta,
                    previous_steering + max_delta
                )

                previous_steering = applied_steering

                # ------------------------------------------
                # Throttle and brake
                # ------------------------------------------

                if acceleration >= 0.0:

                    throttle = clip(
                        0.35 * acceleration,
                        0.0,
                        0.60
                    )

                    brake = 0.0

                else:

                    throttle = 0.0

                    brake = clip(
                        -0.20 * acceleration,
                        0.0,
                        0.80
                    )

                # ------------------------------------------
                # Stationary hold
                # ------------------------------------------

                if stop_state == "HOLD":
                    throttle = 0.0
                    brake = 1.0

                # ------------------------------------------
                # Apply vehicle control
                # ------------------------------------------

                control = adapter.to_vehicle_control(
                    controller_steering_angle=applied_steering,
                    throttle=throttle,
                    brake=brake
                )

                ego.apply_control(control)

                spectator(world, ego)

                active_steps += 1

                # ------------------------------------------
                # Telemetry
                # ------------------------------------------

                writer.writerow([
                    step,
                    frame,
                    ego_speed,
                    lead_speed,
                    route_gap,
                    clearance,
                    lead_route_error,
                    cte,
                    ego_progress,
                    lead_progress,
                    mode.value,
                    acceleration,
                    throttle,
                    brake,
                    control.steer,
                    len(collision_events),
                    stop_state
                ])

                if step % 20 == 0:

                    print(
                        f"Step {step:03d} | "
                        f"Ego={ego_speed:.2f} m/s | "
                        f"Lead={lead_speed:.2f} m/s | "
                        f"Clearance~{clearance:.2f} m | "
                        f"CTE={cte:+.3f} m | "
                        f"ACC={mode.value} | "
                        f"Stop={stop_state}"
                    )

        # --------------------------------------------------
        # Final results
        # --------------------------------------------------

        print("=" * 65)
        print("PHASE 3G RESULTS")
        print("=" * 65)

        print(
            f"Termination: {termination_reason}"
        )

        print(
            f"ACC commanded duration: "
            f"{active_steps * DT:.2f} / {FRAMES * DT:.2f} s"
        )

        print(
            f"Collision events: {len(collision_events)}"
        )

        print(
            f"Stop-hold duration: {hold_steps * DT:.2f} s"
        )

        print(
            f"Telemetry: {output_csv}"
        )

    finally:

        # --------------------------------------------------
        # Stop Tesla
        # --------------------------------------------------

        if ego is not None:
            try:
                if ego.is_alive:
                    ego.apply_control(
                        carla.VehicleControl(
                            brake=1.0,
                            hand_brake=True
                        )
                    )
            except Exception:
                pass

        # --------------------------------------------------
        # Destroy collision sensor
        # --------------------------------------------------

        if collision is not None:

            try:
                collision.stop()
            except Exception:
                pass

            try:
                if collision.is_alive:
                    collision.destroy()
            except Exception:
                pass

        # --------------------------------------------------
        # Destroy Audi
        # --------------------------------------------------

        if lead is not None:

            try:
                if lead.is_alive:
                    lead.set_autopilot(False, TM_PORT)
                    lead.destroy()
            except Exception:
                pass

        # --------------------------------------------------
        # Destroy Tesla
        # --------------------------------------------------

        if ego is not None:

            try:
                if ego.is_alive:
                    ego.destroy()
            except Exception:
                pass

        # --------------------------------------------------
        # Restore Traffic Manager
        # --------------------------------------------------

        if tm_changed:

            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception as exc:
                print(f"TM restore warning: {exc}")

        # --------------------------------------------------
        # Restore world settings
        # --------------------------------------------------

        if settings_changed:

            try:
                world.apply_settings(original_settings)
                print("Original CARLA settings restored")
            except Exception as exc:
                print(f"World restore warning: {exc}")


if __name__ == "__main__":
    main()