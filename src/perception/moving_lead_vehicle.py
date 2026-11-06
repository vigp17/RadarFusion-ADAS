"""
Phase 2D: Moving Lead-Vehicle Radar Validation

CARLA 0.9.16
Radar-Camera Fusion ADAS Project

Stationary Tesla Model 3 with a moving lead vehicle.

Features:
- Tesla Model 3 ego vehicle
- Audi TT lead vehicle
- Scripted lead-vehicle velocity
- Front radar
- Front RGB camera
- Synchronous sensor acquisition
- Spectator camera behind Tesla
- Radar CSV logging
- Ground-truth motion logging
- Camera image capture
- Actor cleanup
- CARLA settings restoration

This is a simulation-based perception experiment.
"""

from __future__ import annotations

import csv
import math
import queue
import time

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import carla  # type: ignore[import-untyped]


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

HOST = "127.0.0.1"
PORT = 2000

FIXED_DT = 0.05
TOTAL_FRAMES = 200

SPAWN_INDEX = 61

INITIAL_GAP_M = 30.0
LEAD_SPEED_MPS = 5.0

SENSOR_TIMEOUT_S = 15.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

RUN_DIR = RESULTS_DIR / f"moving_lead_{RUN_ID}"

RADAR_CSV = RUN_DIR / "radar_detections.csv"

SUMMARY_CSV = RUN_DIR / "motion_summary.csv"

IMAGE_DIR = RUN_DIR / "camera_images"


# ==========================================================
# 2. SENSOR FRAME SYNCHRONIZATION
# ==========================================================

def await_frame(
    sensor_queue: queue.Queue[Any],
    expected_frame: int
) -> Any:

    deadline = time.monotonic() + SENSOR_TIMEOUT_S

    while True:

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            raise TimeoutError(
                f"Missing sensor frame {expected_frame}"
            )

        try:
            measurement = sensor_queue.get(
                timeout=remaining
            )

        except queue.Empty as exc:
            raise TimeoutError(
                f"Missing sensor frame {expected_frame}"
            ) from exc

        if measurement.frame == expected_frame:
            return measurement

        if measurement.frame > expected_frame:
            raise RuntimeError(
                f"Expected frame {expected_frame}, "
                f"received {measurement.frame}"
            )


# ==========================================================
# 3. VEHICLE CONTROL
# ==========================================================

def hold_vehicle_stationary(
    vehicle: carla.Vehicle
) -> None:

    vehicle.set_autopilot(False)

    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True
        )
    )


# ==========================================================
# 4. SPECTATOR CAMERA
# ==========================================================

def update_spectator_camera(
    world: Any,
    vehicle: carla.Vehicle
) -> None:

    transform = vehicle.get_transform()

    forward = transform.get_forward_vector()

    location = transform.location

    camera_location = carla.Location(
        x=location.x - 8.0 * forward.x,
        y=location.y - 8.0 * forward.y,
        z=location.z + 4.5
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


# ==========================================================
# 5. VEHICLE SPAWNING
# ==========================================================

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
            "Check spawn-point clearance."
        )

    if not isinstance(actor, carla.Vehicle):

        actor.destroy()

        raise TypeError(
            "Spawned actor is not a CARLA Vehicle."
        )

    return cast(carla.Vehicle, actor)


# ==========================================================
# 6. MAIN EXPERIMENT
# ==========================================================

def main() -> None:

    print("=" * 65)
    print("RADAR-CAMERA FUSION ADAS PROJECT")
    print("PHASE 2D: MOVING LEAD-VEHICLE VALIDATION")
    print("=" * 65)

    client = carla.Client(HOST, PORT)

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

    original_settings = world.get_settings()

    actors: list[Any] = []

    sensors: list[carla.Sensor] = []

    settings_changed = False

    try:

        # --------------------------------------------------
        # Configure synchronous simulation
        # --------------------------------------------------

        settings = world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT

        world.apply_settings(settings)

        settings_changed = True

        library = world.get_blueprint_library()

        # --------------------------------------------------
        # Spawn ego Tesla
        # --------------------------------------------------

        ego_vehicle = spawn_vehicle(
            world,
            library.find("vehicle.tesla.model3"),
            spawn_points[SPAWN_INDEX]
        )

        actors.append(ego_vehicle)

        hold_vehicle_stationary(ego_vehicle)

        print(
            f"Ego Tesla spawned: {ego_vehicle.id}"
        )

        # --------------------------------------------------
        # Determine lead-vehicle spawn location
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
            INITIAL_GAP_M
        )

        if len(next_waypoints) != 1:
            raise RuntimeError(
                "Lead placement requires one "
                "unambiguous forward waypoint."
            )

        lead_waypoint = next_waypoints[0]

        if (
            lead_waypoint.lane_id != start_waypoint.lane_id
            or lead_waypoint.is_junction
        ):
            raise RuntimeError(
                "Lead waypoint is not a valid "
                "same-lane location."
            )

        lead_transform = lead_waypoint.transform

        lead_transform.location.z += 0.3

        # --------------------------------------------------
        # Spawn lead vehicle
        # --------------------------------------------------

        lead_vehicle = spawn_vehicle(
            world,
            library.find("vehicle.audi.tt"),
            lead_transform
        )

        actors.append(lead_vehicle)

        lead_vehicle.set_autopilot(False)

        lead_vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=0.0,
                hand_brake=False
            )
        )

        print(
            f"Lead Audi spawned: {lead_vehicle.id}"
        )

        # --------------------------------------------------
        # Spectator camera
        # --------------------------------------------------

        update_spectator_camera(
            world,
            ego_vehicle
        )

        # --------------------------------------------------
        # Configure front radar
        # --------------------------------------------------

        radar_bp = library.find(
            "sensor.other.radar"
        )

        radar_attributes = {
            "horizontal_fov": 30,
            "vertical_fov": 10,
            "range": 100,
            "sensor_tick": 0.0
        }

        for key, value in radar_attributes.items():
            radar_bp.set_attribute(
                key,
                str(value)
            )

        # --------------------------------------------------
        # Configure front RGB camera
        # --------------------------------------------------

        camera_bp = library.find(
            "sensor.camera.rgb"
        )

        camera_attributes = {
            "image_size_x": 1280,
            "image_size_y": 720,
            "fov": 90,
            "sensor_tick": 0.0
        }

        for key, value in camera_attributes.items():
            camera_bp.set_attribute(
                key,
                str(value)
            )

        radar_queue: queue.Queue[Any] = queue.Queue()
        camera_queue: queue.Queue[Any] = queue.Queue()

        sensor_configs = [
            (
                "front_radar",
                radar_bp,
                carla.Transform(
                    carla.Location(
                        x=2.0,
                        y=0.0,
                        z=0.6
                    )
                ),
                radar_queue
            ),
            (
                "front_camera",
                camera_bp,
                carla.Transform(
                    carla.Location(
                        x=1.2,
                        y=0.0,
                        z=1.5
                    )
                ),
                camera_queue
            )
        ]

        # --------------------------------------------------
        # Attach sensors
        # --------------------------------------------------

        for name, bp, transform, sensor_queue in sensor_configs:

            actor = world.spawn_actor(
                bp,
                transform,
                attach_to=ego_vehicle
            )

            if not isinstance(actor, carla.Sensor):

                actor.destroy()

                raise TypeError(
                    f"{name} is not a CARLA sensor."
                )

            sensor = cast(carla.Sensor, actor)

            sensors.append(sensor)

            sensor.listen(
                lambda data, q=sensor_queue: q.put(data)
            )

            print(
                f"Attached sensor: {name}"
            )

        # --------------------------------------------------
        # Prepare output directories
        # --------------------------------------------------

        RUN_DIR.mkdir(
            parents=True,
            exist_ok=False
        )

        IMAGE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------
        # Collect synchronized measurements
        # --------------------------------------------------

        with (
            RADAR_CSV.open(
                "x",
                newline="",
                encoding="utf-8"
            ) as radar_file,
            SUMMARY_CSV.open(
                "x",
                newline="",
                encoding="utf-8"
            ) as summary_file
        ):

            radar_writer = csv.writer(
                radar_file
            )

            summary_writer = csv.writer(
                summary_file
            )

            radar_writer.writerow([
                "frame",
                "timestamp_s",
                "depth_m",
                "velocity_mps",
                "azimuth_rad",
                "altitude_rad"
            ])

            summary_writer.writerow([
                "frame",
                "timestamp_s",
                "ego_speed_mps",
                "lead_speed_mps",
                "center_distance_m",
                "radar_candidate_depth_m",
                "radar_candidate_velocity_mps",
                "candidate_count"
            ])

            # ----------------------------------------------
            # Simulation loop
            # ----------------------------------------------

            for step in range(TOTAL_FRAMES):

                # Scripted lead-vehicle velocity.
                # This is not an ACC controller.

                direction = (
                    lead_vehicle
                    .get_transform()
                    .get_forward_vector()
                )

                lead_vehicle.set_target_velocity(
                    carla.Vector3D(
                        x=direction.x * LEAD_SPEED_MPS,
                        y=direction.y * LEAD_SPEED_MPS,
                        z=0.0
                    )
                )

                frame = world.tick()

                radar_data = await_frame(
                    radar_queue,
                    frame
                )

                camera_data = await_frame(
                    camera_queue,
                    frame
                )

                update_spectator_camera(
                    world,
                    ego_vehicle
                )

                # ------------------------------------------
                # Save radar detections
                # ------------------------------------------

                for detection in radar_data:

                    radar_writer.writerow([
                        frame,
                        radar_data.timestamp,
                        detection.depth,
                        detection.velocity,
                        detection.azimuth,
                        detection.altitude
                    ])

                # ------------------------------------------
                # Simple forward radar candidates
                # ------------------------------------------

                candidates = [
                    detection
                    for detection in radar_data
                    if (
                        3.0 < detection.depth < 100.0
                        and abs(detection.azimuth)
                        < math.radians(5.0)
                    )
                ]

                nearest = (
                    min(
                        candidates,
                        key=lambda detection: detection.depth
                    )
                    if candidates
                    else None
                )

                # ------------------------------------------
                # Ground-truth measurements
                # ------------------------------------------

                ego_speed = (
                    ego_vehicle.get_velocity().length()
                )

                lead_speed = (
                    lead_vehicle.get_velocity().length()
                )

                center_distance = (
                    ego_vehicle.get_location().distance(
                        lead_vehicle.get_location()
                    )
                )

                summary_writer.writerow([
                    frame,
                    radar_data.timestamp,
                    ego_speed,
                    lead_speed,
                    center_distance,
                    nearest.depth if nearest else "",
                    nearest.velocity if nearest else "",
                    len(candidates)
                ])

                # ------------------------------------------
                # Camera images and progress
                # ------------------------------------------

                if step % 10 == 0:

                    image_path = (
                        IMAGE_DIR
                        / f"frame_{frame:06d}.png"
                    )

                    camera_data.save_to_disk(
                        str(image_path)
                    )

                    radar_text = (
                        f"{nearest.depth:.2f} m"
                        if nearest
                        else "No candidate"
                    )

                    print(
                        f"Step {step:03d} | "
                        f"Center gap = {center_distance:.2f} m | "
                        f"Lead speed = {lead_speed:.2f} m/s | "
                        f"Nearest radar = {radar_text}"
                    )

        print("=" * 65)
        print("MOVING LEAD-VEHICLE TEST COMPLETE")
        print("=" * 65)

        print(f"Radar CSV: {RADAR_CSV}")
        print(f"Motion summary: {SUMMARY_CSV}")
        print(f"Camera images: {IMAGE_DIR}")

        print(
            "\nImportant: The nearest radar return "
            "is not a verified lead-vehicle track."
        )

        # Brief visual inspection.
        time.sleep(5.0)

    finally:

        # --------------------------------------------------
        # Stop sensors
        # --------------------------------------------------

        for sensor in sensors:

            try:
                sensor.stop()
            except Exception:
                pass

        # --------------------------------------------------
        # Destroy sensors
        # --------------------------------------------------

        for sensor in reversed(sensors):

            try:
                if sensor.is_alive:
                    sensor.destroy()
            except Exception:
                pass

        # --------------------------------------------------
        # Destroy vehicles
        # --------------------------------------------------

        for actor in reversed(actors):

            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass

        # --------------------------------------------------
        # Restore original CARLA settings
        # --------------------------------------------------

        if settings_changed:

            try:

                world.apply_settings(
                    original_settings
                )

                print(
                    "CARLA world settings restored."
                )

            except Exception as exc:

                print(
                    "WARNING: Settings restoration failed: "
                    f"{exc}"
                )


# ==========================================================
# 7. ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()