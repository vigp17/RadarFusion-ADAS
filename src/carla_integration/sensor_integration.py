"""
Radar-Camera Fusion ADAS Project
Phase 1D: Sensor Integration

CARLA Version: 0.9.16
Vehicle: Tesla Model 3
Map: Town10HD_Opt

Features:
    - Front radar
    - Front RGB camera
    - Rear-left radar
    - Rear-right radar
    - Synchronous sensor acquisition
    - Radar CSV logging
    - Camera image capture
    - Third-person spectator camera following Tesla
    - Safe actor cleanup
    - CARLA world settings restoration

Run only in a dedicated CARLA simulation session.
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
TOTAL_FRAMES = 100

SPAWN_INDEX = 61

SENSOR_TIMEOUT_S = 15.0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 90.0

RADAR_RANGE_M = 100.0

# Spectator camera configuration
SPECTATOR_DISTANCE_M = 8.0
SPECTATOR_HEIGHT_M = 4.5

# Pause for visual inspection before cleanup.
VISUAL_INSPECTION_SECONDS = 10.0

# Enable callback diagnostics when troubleshooting.
DEBUG_SENSOR_CALLBACKS = False

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_DIR = RESULTS_DIR / f"sensor_test_{RUN_ID}"

IMAGE_DIR = RUN_DIR / "camera_images"

RADAR_CSV = RUN_DIR / "radar_detections.csv"


# ==========================================================
# 2. RADAR BLUEPRINT
# ==========================================================

def radar_blueprint(
    library: Any,
    hfov: float,
    vfov: float
) -> Any:

    bp = library.find("sensor.other.radar")

    attributes = {
        "horizontal_fov": hfov,
        "vertical_fov": vfov,
        "range": RADAR_RANGE_M,
        "sensor_tick": 0.0
    }

    for key, value in attributes.items():
        bp.set_attribute(key, str(value))

    return bp


# ==========================================================
# 3. CAMERA BLUEPRINT
# ==========================================================

def camera_blueprint(
    library: Any
) -> Any:

    bp = library.find("sensor.camera.rgb")

    attributes = {
        "image_size_x": CAMERA_WIDTH,
        "image_size_y": CAMERA_HEIGHT,
        "fov": CAMERA_FOV,
        "sensor_tick": 0.0
    }

    for key, value in attributes.items():
        bp.set_attribute(key, str(value))

    return bp


# ==========================================================
# 4. SPECTATOR CAMERA
# ==========================================================

def update_spectator_camera(
    world: Any,
    vehicle: carla.Vehicle
) -> None:
    """
    Position CARLA's spectator camera behind and above
    the Tesla Model 3.

    This camera controls the simulator viewport.
    It is separate from the attached RGB sensor.
    """

    transform = vehicle.get_transform()

    vehicle_location = transform.location

    forward = transform.get_forward_vector()

    camera_x = (
        vehicle_location.x
        - SPECTATOR_DISTANCE_M * forward.x
    )

    camera_y = (
        vehicle_location.y
        - SPECTATOR_DISTANCE_M * forward.y
    )

    camera_z = (
        vehicle_location.z
        + SPECTATOR_HEIGHT_M
    )

    camera_location = carla.Location(
        x=camera_x,
        y=camera_y,
        z=camera_z
    )

    # Aim the camera toward the vehicle.
    dx = vehicle_location.x - camera_x
    dy = vehicle_location.y - camera_y
    dz = vehicle_location.z - camera_z

    horizontal_distance = math.hypot(dx, dy)

    camera_yaw = math.degrees(
        math.atan2(dy, dx)
    )

    camera_pitch = math.degrees(
        math.atan2(
            dz,
            horizontal_distance
        )
    )

    camera_rotation = carla.Rotation(
        pitch=camera_pitch,
        yaw=camera_yaw,
        roll=0.0
    )

    spectator = world.get_spectator()

    spectator.set_transform(
        carla.Transform(
            camera_location,
            camera_rotation
        )
    )


# ==========================================================
# 5. SENSOR FRAME SYNCHRONIZATION
# ==========================================================

def await_frame(
    sensor_queue: queue.Queue[Any],
    expected_frame: int,
    timeout: float = SENSOR_TIMEOUT_S
) -> Any:
    """
    Wait for a sensor measurement matching the expected
    CARLA simulation frame.

    Discard stale measurements.
    """

    deadline = time.monotonic() + timeout

    while True:

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for frame {expected_frame}"
            )

        try:
            measurement = sensor_queue.get(
                timeout=remaining
            )

        except queue.Empty as exc:
            raise TimeoutError(
                f"Timed out waiting for frame {expected_frame}"
            ) from exc

        received_frame = int(measurement.frame)

        if received_frame == expected_frame:
            return measurement

        if received_frame < expected_frame:
            continue

        raise RuntimeError(
            "Sensor synchronization mismatch: "
            f"Expected frame {expected_frame}, "
            f"received frame {received_frame}"
        )


# ==========================================================
# 6. SENSOR MOUNTING CONFIGURATION
# ==========================================================

def create_sensor_mounts() -> dict[str, tuple]:
    """
    Sensor mounting positions relative to the Tesla.

    CARLA vehicle-local coordinates:
        X: Forward
        Y: Right
        Z: Up
    """

    return {

        "front_radar": (
            "radar",

            carla.Transform(
                carla.Location(
                    x=2.0,
                    y=0.0,
                    z=0.6
                )
            ),

            30.0,
            10.0
        ),

        "front_camera": (
            "camera",

            carla.Transform(
                carla.Location(
                    x=1.2,
                    y=0.0,
                    z=1.5
                )
            ),

            0.0,
            0.0
        ),

        "rear_left_radar": (
            "radar",

            carla.Transform(
                carla.Location(
                    x=-2.0,
                    y=-0.8,
                    z=0.6
                ),

                carla.Rotation(
                    yaw=150.0
                )
            ),

            60.0,
            15.0
        ),

        "rear_right_radar": (
            "radar",

            carla.Transform(
                carla.Location(
                    x=-2.0,
                    y=0.8,
                    z=0.6
                ),

                carla.Rotation(
                    yaw=-150.0
                )
            ),

            60.0,
            15.0
        )
    }


# ==========================================================
# 7. SENSOR CALLBACK
# ==========================================================

def create_sensor_callback(
    sensor_name: str,
    measurement_queue: queue.Queue[Any]
):
    """
    Create an independent callback for each sensor.
    """

    def callback(data: Any) -> None:

        if DEBUG_SENSOR_CALLBACKS:

            print(
                f"[CALLBACK] {sensor_name} | "
                f"Frame={data.frame} | "
                f"Timestamp={data.timestamp:.3f}"
            )

        measurement_queue.put(data)

    return callback


# ==========================================================
# 8. MAIN SENSOR INTEGRATION
# ==========================================================

def main() -> None:

    print("=" * 65)
    print("RADAR-CAMERA FUSION ADAS PROJECT")
    print("PHASE 1D: SENSOR INTEGRATION")
    print("=" * 65)

    client = carla.Client(HOST, PORT)

    client.set_timeout(30.0)

    world = client.get_world()

    map_name = world.get_map().name

    if "Town10HD_Opt" not in map_name:
        raise RuntimeError(
            "Please load Town10HD_Opt before running."
        )

    print(f"Connected Map: {map_name}")

    spawn_points = world.get_map().get_spawn_points()

    if len(spawn_points) <= SPAWN_INDEX:
        raise RuntimeError(
            "Spawn point 61 is unavailable."
        )

    original_settings = world.get_settings()

    vehicle: carla.Vehicle | None = None

    sensors: list[carla.Sensor] = []

    sensor_queues: dict[
        str,
        queue.Queue[Any]
    ] = {}

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

        print(
            f"Synchronous Mode: Enabled | "
            f"dt = {FIXED_DT:.2f} s"
        )

        # --------------------------------------------------
        # Spawn Tesla Model 3
        # --------------------------------------------------

        library = world.get_blueprint_library()

        vehicle_bp = library.find(
            "vehicle.tesla.model3"
        )

        actor = world.try_spawn_actor(
            vehicle_bp,
            spawn_points[SPAWN_INDEX]
        )

        if actor is None:
            raise RuntimeError(
                "Tesla spawn failed. "
                "Verify spawn point 61 is clear."
            )

        if not isinstance(actor, carla.Vehicle):

            actor.destroy()

            raise TypeError(
                "Tesla blueprint did not produce a Vehicle."
            )

        vehicle = cast(carla.Vehicle, actor)

        vehicle.set_autopilot(False)

        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True
            )
        )

        print(
            f"Tesla Model 3 spawned | "
            f"Actor ID: {vehicle.id}"
        )

        # --------------------------------------------------
        # Move spectator camera behind Tesla
        # --------------------------------------------------

        update_spectator_camera(
            world,
            vehicle
        )

        print(
            "Spectator camera positioned behind Tesla."
        )

        # --------------------------------------------------
        # Attach sensors
        # --------------------------------------------------

        mounts = create_sensor_mounts()

        for name, config in mounts.items():

            sensor_type, transform, hfov, vfov = config

            if sensor_type == "camera":

                bp = camera_blueprint(library)

            else:

                bp = radar_blueprint(
                    library,
                    hfov,
                    vfov
                )

            spawned = world.spawn_actor(
                bp,
                transform,
                attach_to=vehicle
            )

            if not isinstance(spawned, carla.Sensor):

                spawned.destroy()

                raise TypeError(
                    f"{name} did not produce a Sensor."
                )

            sensor = cast(
                carla.Sensor,
                spawned
            )

            sensors.append(sensor)

            measurement_queue: queue.Queue[Any] = (
                queue.Queue()
            )

            sensor_queues[name] = measurement_queue

            sensor.listen(
                create_sensor_callback(
                    name,
                    measurement_queue
                )
            )

            print(
                f"Attached sensor: {name}"
            )

        # --------------------------------------------------
        # Prepare results directories
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
        # Collect synchronized sensor data
        # --------------------------------------------------

        with RADAR_CSV.open(
            "x",
            newline="",
            encoding="utf-8"
        ) as handle:

            writer = csv.writer(handle)

            writer.writerow([
                "frame",
                "timestamp_s",
                "sensor",
                "depth_m",
                "velocity_mps",
                "azimuth_rad",
                "altitude_rad"
            ])

            for step in range(TOTAL_FRAMES):

                # Advance CARLA by one frame.
                frame = world.tick()

                # Keep spectator camera following the Tesla.
                update_spectator_camera(
                    world,
                    vehicle
                )

                measurements: dict[
                    str,
                    Any
                ] = {}

                # ------------------------------------------
                # Wait for all four sensors
                # ------------------------------------------

                for name, sensor_queue in sensor_queues.items():

                    try:

                        measurements[name] = await_frame(
                            sensor_queue,
                            frame,
                            timeout=SENSOR_TIMEOUT_S
                        )

                    except TimeoutError as exc:

                        raise TimeoutError(
                            f"Sensor '{name}' failed to deliver "
                            f"CARLA frame {frame}"
                        ) from exc

                # ------------------------------------------
                # Save radar and camera measurements
                # ------------------------------------------

                for name, measurement in measurements.items():

                    if name == "front_camera":

                        if step % 10 == 0:

                            image_path = (
                                IMAGE_DIR
                                / f"frame_{frame:06d}.png"
                            )

                            measurement.save_to_disk(
                                str(image_path)
                            )

                    else:

                        for detection in measurement:

                            writer.writerow([
                                frame,
                                measurement.timestamp,
                                name,
                                detection.depth,
                                detection.velocity,
                                detection.azimuth,
                                detection.altitude
                            ])

                # ------------------------------------------
                # Progress reporting
                # ------------------------------------------

                if step % 10 == 0:

                    print(
                        f"Step {step:03d} | "
                        f"CARLA Frame {frame} | "
                        "All sensors synchronized"
                    )

        print("=" * 65)
        print("SENSOR INTEGRATION COMPLETE")
        print("=" * 65)

        print(f"Radar CSV saved: {RADAR_CSV}")

        print(f"Camera images saved: {IMAGE_DIR}")

        # --------------------------------------------------
        # Allow visual inspection before cleanup
        # --------------------------------------------------

        print(
            "\nTesla remains visible for "
            f"{VISUAL_INSPECTION_SECONDS:.0f} seconds."
        )

        print(
            "Switch to the CARLA window to inspect the Tesla."
        )

        time.sleep(VISUAL_INSPECTION_SECONDS)

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
        # Destroy Tesla
        # --------------------------------------------------

        if vehicle is not None:

            try:

                if vehicle.is_alive:
                    vehicle.destroy()

            except Exception:
                pass

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
                    "WARNING: Failed to restore "
                    f"CARLA settings: {exc}"
                )


# ==========================================================
# 9. ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()