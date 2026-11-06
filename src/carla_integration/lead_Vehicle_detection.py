"""
Radar-Camera Fusion ADAS Project

Phase 2A:
Stationary Lead-Vehicle Detection

CARLA Version: 0.9.16
Map: Town10HD_Opt

Features:
    - Tesla Model 3 ego vehicle
    - Stationary lead vehicle
    - Front radar
    - Front RGB camera
    - Third-person spectator camera
    - Synchronous sensor collection
    - Radar detection filtering
    - Ground-truth distance comparison
    - CSV logging
    - Camera image capture
    - Actor cleanup
    - CARLA settings restoration

This is a simulation experiment, not a
production ADAS perception system.
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

EXPECTED_MAP = "Town10HD_Opt"

FIXED_DT = 0.05
TOTAL_FRAMES = 100

SPAWN_INDEX = 61

LEAD_DISTANCE_M = 30.0

RADAR_RANGE_M = 100.0

RADAR_HORIZONTAL_FOV = 30.0
RADAR_VERTICAL_FOV = 10.0

# Narrow forward angular gate.
TARGET_AZIMUTH_DEG = 5.0

SENSOR_TIMEOUT_S = 15.0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results"

RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

RUN_DIR = RESULTS_DIR / f"lead_vehicle_{RUN_ID}"

IMAGE_DIR = RUN_DIR / "camera_images"

RADAR_CSV = RUN_DIR / "radar_detections.csv"

SUMMARY_CSV = RUN_DIR / "lead_vehicle_summary.csv"


# ==========================================================
# 2. SENSOR CONFIGURATION
# ==========================================================

def create_radar_blueprint(
    library: Any
) -> Any:

    bp = library.find("sensor.other.radar")

    attributes = {
        "horizontal_fov": RADAR_HORIZONTAL_FOV,
        "vertical_fov": RADAR_VERTICAL_FOV,
        "range": RADAR_RANGE_M,
        "sensor_tick": 0.0,
    }

    for key, value in attributes.items():
        bp.set_attribute(key, str(value))

    return bp


def create_camera_blueprint(
    library: Any
) -> Any:

    bp = library.find("sensor.camera.rgb")

    attributes = {
        "image_size_x": CAMERA_WIDTH,
        "image_size_y": CAMERA_HEIGHT,
        "fov": 90,
        "sensor_tick": 0.0,
    }

    for key, value in attributes.items():
        bp.set_attribute(key, str(value))

    return bp


# ==========================================================
# 3. SPECTATOR CAMERA
# ==========================================================

def update_spectator_camera(
    world: Any,
    vehicle: carla.Vehicle
) -> None:

    transform = vehicle.get_transform()

    location = transform.location

    forward = transform.get_forward_vector()

    camera_location = carla.Location(
        x=location.x - 8.0 * forward.x,
        y=location.y - 8.0 * forward.y,
        z=location.z + 4.5,
    )

    dx = location.x - camera_location.x
    dy = location.y - camera_location.y
    dz = location.z - camera_location.z

    yaw = math.degrees(
        math.atan2(dy, dx)
    )

    pitch = math.degrees(
        math.atan2(
            dz,
            math.hypot(dx, dy)
        )
    )

    spectator = world.get_spectator()

    spectator.set_transform(
        carla.Transform(
            camera_location,
            carla.Rotation(
                pitch=pitch,
                yaw=yaw,
                roll=0.0,
            ),
        )
    )


# ==========================================================
# 4. SENSOR SYNCHRONIZATION
# ==========================================================

def await_frame(
    sensor_queue: queue.Queue[Any],
    expected_frame: int,
    timeout: float = SENSOR_TIMEOUT_S,
) -> Any:

    deadline = time.monotonic() + timeout

    while True:

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            raise TimeoutError(
                f"Sensor frame {expected_frame} timed out."
            )

        try:
            measurement = sensor_queue.get(
                timeout=remaining
            )

        except queue.Empty as exc:
            raise TimeoutError(
                f"Sensor frame {expected_frame} timed out."
            ) from exc

        if measurement.frame == expected_frame:
            return measurement

        if measurement.frame < expected_frame:
            continue

        raise RuntimeError(
            "Sensor synchronization mismatch: "
            f"expected {expected_frame}, "
            f"received {measurement.frame}"
        )


# ==========================================================
# 5. RADAR DETECTION PROCESSING
# ==========================================================

def filter_forward_detections(
    radar_measurement: Any
) -> list[Any]:
    """
    Select radar detections close to the
    forward centerline.

    This does not identify a particular vehicle.
    """

    selected: list[Any] = []

    max_azimuth = math.radians(
        TARGET_AZIMUTH_DEG
    )

    for detection in radar_measurement:

        if detection.depth <= 0:
            continue

        if detection.depth > RADAR_RANGE_M:
            continue

        if abs(detection.azimuth) > max_azimuth:
            continue

        selected.append(detection)

    return selected


def estimate_forward_target(
    detections: list[Any]
) -> tuple[float | None, float | None]:

    """
    Return the nearest gated radar detection.

    This is a simple forward-target candidate,
    not a verified lead-vehicle track.
    """

    if not detections:
        return None, None

    nearest = min(
        detections,
        key=lambda detection: detection.depth
    )

    return (
        float(nearest.depth),
        float(nearest.velocity),
    )


# ==========================================================
# 6. GROUND-TRUTH DISTANCE
# ==========================================================

def calculate_ground_truth_distance(
    ego_vehicle: carla.Vehicle,
    lead_vehicle: carla.Vehicle,
) -> float:

    """
    Center-to-center 3D distance.

    This is not bumper-to-bumper clearance.
    """

    ego_location = ego_vehicle.get_location()

    lead_location = lead_vehicle.get_location()

    return float(
        ego_location.distance(lead_location)
    )


# ==========================================================
# 7. STATIONARY VEHICLE CONTROL
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
            hand_brake=True,
        )
    )


# ==========================================================
# 8. MAIN EXPERIMENT
# ==========================================================

def main() -> None:

    print("=" * 65)
    print("RADAR-CAMERA FUSION ADAS PROJECT")
    print("PHASE 2A: LEAD VEHICLE DETECTION")
    print("=" * 65)

    client = carla.Client(HOST, PORT)

    client.set_timeout(30.0)

    world = client.get_world()

    map_name = world.get_map().name

    if EXPECTED_MAP not in map_name:
        raise RuntimeError(
            f"Expected {EXPECTED_MAP}, got {map_name}"
        )

    print(f"Connected Map: {map_name}")

    original_settings = world.get_settings()

    ego_vehicle: carla.Vehicle | None = None
    lead_vehicle: carla.Vehicle | None = None

    sensors: list[carla.Sensor] = []

    settings_changed = False

    radar_queue: queue.Queue[Any] = queue.Queue()
    camera_queue: queue.Queue[Any] = queue.Queue()

    try:

        # --------------------------------------------------
        # Configure synchronous mode
        # --------------------------------------------------

        settings = world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT

        world.apply_settings(settings)

        settings_changed = True

        library = world.get_blueprint_library()

        carla_map = world.get_map()

        spawn_points = carla_map.get_spawn_points()

        if len(spawn_points) <= SPAWN_INDEX:
            raise RuntimeError(
                "Spawn point 61 is unavailable."
            )

        # --------------------------------------------------
        # Spawn ego vehicle
        # --------------------------------------------------

        ego_bp = library.find(
            "vehicle.tesla.model3"
        )

        ego_transform = spawn_points[SPAWN_INDEX]

        actor = world.try_spawn_actor(
            ego_bp,
            ego_transform
        )

        if actor is None:
            raise RuntimeError(
                "Unable to spawn ego Tesla."
            )

        if not isinstance(actor, carla.Vehicle):
            actor.destroy()
            raise TypeError(
                "Ego actor is not a vehicle."
            )

        ego_vehicle = cast(carla.Vehicle, actor)

        hold_vehicle_stationary(ego_vehicle)

        print(
            f"Ego Tesla spawned: {ego_vehicle.id}"
        )

        # --------------------------------------------------
        # Find lead vehicle location
        # --------------------------------------------------

        start_waypoint = carla_map.get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if start_waypoint is None:
            raise RuntimeError(
                "No driving waypoint at ego spawn."
            )

        next_waypoints = start_waypoint.next(
            LEAD_DISTANCE_M
        )

        # Only use an unambiguous forward route.
        if len(next_waypoints) != 1:
            raise RuntimeError(
                "Lead placement requires exactly one "
                "forward waypoint 30 m ahead."
            )

        lead_waypoint = next_waypoints[0]

        if lead_waypoint.is_junction:
            raise RuntimeError(
                "Lead waypoint is inside a junction."
            )

        if lead_waypoint.lane_id != start_waypoint.lane_id:
            raise RuntimeError(
                "Lead waypoint is not in the ego lane."
            )

        lead_transform = lead_waypoint.transform

        # Raise the vehicle slightly above the road.
        lead_transform.location.z += 0.3

        # --------------------------------------------------
        # Spawn lead vehicle
        # --------------------------------------------------

        lead_bp = library.find(
            "vehicle.audi.tt"
        )

        lead_actor = world.try_spawn_actor(
            lead_bp,
            lead_transform
        )

        if lead_actor is None:
            raise RuntimeError(
                "Unable to spawn lead vehicle."
            )

        if not isinstance(
            lead_actor,
            carla.Vehicle
        ):
            lead_actor.destroy()
            raise TypeError(
                "Lead actor is not a vehicle."
            )

        lead_vehicle = cast(
            carla.Vehicle,
            lead_actor
        )

        hold_vehicle_stationary(lead_vehicle)

        print(
            f"Lead vehicle spawned: {lead_vehicle.id}"
        )

        initial_distance = (
            calculate_ground_truth_distance(
                ego_vehicle,
                lead_vehicle
            )
        )

        print(
            f"Initial center distance: "
            f"{initial_distance:.2f} m"
        )

        # --------------------------------------------------
        # Spectator camera
        # --------------------------------------------------

        update_spectator_camera(
            world,
            ego_vehicle
        )

        # --------------------------------------------------
        # Front radar
        # --------------------------------------------------

        radar_bp = create_radar_blueprint(
            library
        )

        radar_transform = carla.Transform(
            carla.Location(
                x=2.0,
                y=0.0,
                z=0.6,
            )
        )

        radar_actor = world.spawn_actor(
            radar_bp,
            radar_transform,
            attach_to=ego_vehicle,
        )

        if not isinstance(
            radar_actor,
            carla.Sensor
        ):
            radar_actor.destroy()
            raise TypeError(
                "Radar actor type invalid."
            )

        radar_sensor = cast(
            carla.Sensor,
            radar_actor
        )

        sensors.append(radar_sensor)

        radar_sensor.listen(
            lambda data: radar_queue.put(data)
        )

        print("Front radar attached.")

        # --------------------------------------------------
        # Front camera
        # --------------------------------------------------

        camera_bp = create_camera_blueprint(
            library
        )

        camera_transform = carla.Transform(
            carla.Location(
                x=1.2,
                y=0.0,
                z=1.5,
            )
        )

        camera_actor = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=ego_vehicle,
        )

        if not isinstance(
            camera_actor,
            carla.Sensor
        ):
            camera_actor.destroy()
            raise TypeError(
                "Camera actor type invalid."
            )

        camera_sensor = cast(
            carla.Sensor,
            camera_actor
        )

        sensors.append(camera_sensor)

        camera_sensor.listen(
            lambda data: camera_queue.put(data)
        )

        print("Front RGB camera attached.")

        # --------------------------------------------------
        # Create output directories
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
        # Collect measurements
        # --------------------------------------------------

        with (
            RADAR_CSV.open(
                "x",
                newline="",
                encoding="utf-8",
            ) as radar_file,
            SUMMARY_CSV.open(
                "x",
                newline="",
                encoding="utf-8",
            ) as summary_file,
        ):

            radar_writer = csv.writer(radar_file)
            summary_writer = csv.writer(summary_file)

            radar_writer.writerow([
                "frame",
                "timestamp_s",
                "depth_m",
                "velocity_mps",
                "azimuth_rad",
                "altitude_rad",
            ])

            summary_writer.writerow([
                "frame",
                "timestamp_s",
                "ground_truth_center_distance_m",
                "nearest_radar_depth_m",
                "nearest_radar_velocity_mps",
                "candidate_detection_count",
            ])

            for step in range(TOTAL_FRAMES):

                frame = world.tick()

                update_spectator_camera(
                    world,
                    ego_vehicle
                )

                radar_data = await_frame(
                    radar_queue,
                    frame
                )

                camera_data = await_frame(
                    camera_queue,
                    frame
                )

                # ------------------------------------------
                # Process radar detections
                # ------------------------------------------

                detections = (
                    filter_forward_detections(
                        radar_data
                    )
                )

                estimated_depth, estimated_velocity = (
                    estimate_forward_target(
                        detections
                    )
                )

                # ------------------------------------------
                # Ground truth
                # ------------------------------------------

                true_distance = (
                    calculate_ground_truth_distance(
                        ego_vehicle,
                        lead_vehicle
                    )
                )

                # ------------------------------------------
                # Save all radar detections
                # ------------------------------------------

                for detection in radar_data:

                    radar_writer.writerow([
                        frame,
                        radar_data.timestamp,
                        detection.depth,
                        detection.velocity,
                        detection.azimuth,
                        detection.altitude,
                    ])

                # ------------------------------------------
                # Save summary
                # ------------------------------------------

                summary_writer.writerow([
                    frame,
                    radar_data.timestamp,
                    true_distance,
                    estimated_depth,
                    estimated_velocity,
                    len(detections),
                ])

                # ------------------------------------------
                # Save camera images
                # ------------------------------------------

                if step % 10 == 0:

                    image_path = (
                        IMAGE_DIR
                        / f"frame_{frame:06d}.png"
                    )

                    camera_data.save_to_disk(
                        str(image_path)
                    )

                # ------------------------------------------
                # Print progress
                # ------------------------------------------

                if step % 10 == 0:

                    depth_text = (
                        f"{estimated_depth:.2f} m"
                        if estimated_depth is not None
                        else "No candidate"
                    )

                    print(
                        f"Step {step:03d} | "
                        f"Frame {frame} | "
                        f"Ground truth = {true_distance:.2f} m | "
                        f"Radar candidate = {depth_text}"
                    )

        print("=" * 65)
        print("PHASE 2A DATA COLLECTION COMPLETE")
        print("=" * 65)

        print(f"Radar CSV: {RADAR_CSV}")
        print(f"Summary CSV: {SUMMARY_CSV}")
        print(f"Camera images: {IMAGE_DIR}")

        print(
            "\nNote: Nearest radar detection is not "
            "necessarily the lead vehicle."
        )

        # Brief inspection pause.
        print(
            "\nTesla and lead vehicle remain visible "
            "for 10 seconds."
        )

        time.sleep(10.0)

    finally:

        # --------------------------------------------------
        # Stop sensor callbacks
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
        # Destroy lead vehicle
        # --------------------------------------------------

        if lead_vehicle is not None:

            try:
                if lead_vehicle.is_alive:
                    lead_vehicle.destroy()
            except Exception:
                pass

        # --------------------------------------------------
        # Destroy ego vehicle
        # --------------------------------------------------

        if ego_vehicle is not None:

            try:
                if ego_vehicle.is_alive:
                    ego_vehicle.destroy()
            except Exception:
                pass

        # --------------------------------------------------
        # Restore original settings
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
                    "WARNING: Failed to restore settings: "
                    f"{exc}"
                )


# ==========================================================
# 9. ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()