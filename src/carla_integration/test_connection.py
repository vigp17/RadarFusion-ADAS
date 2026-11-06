"""
Phase 1: CARLA connection verification.

Radar-Camera Fusion ADAS Project
"""

import carla


def main():

    client = carla.Client(
        "127.0.0.1",
        2000
    )

    client.set_timeout(30.0)

    world = client.get_world()

    carla_map = world.get_map()

    blueprint_library = world.get_blueprint_library()

    print("=" * 60)
    print("RADAR-CAMERA ADAS PROJECT")
    print("CARLA CONNECTION TEST")
    print("=" * 60)

    print(f"Map: {carla_map.name}")

    print(
        f"Vehicle blueprints: "
        f"{len(blueprint_library.filter('vehicle.*'))}"
    )

    print(
        f"Radar blueprints: "
        f"{len(blueprint_library.filter('sensor.other.radar'))}"
    )

    print(
        f"RGB camera blueprints: "
        f"{len(blueprint_library.filter('sensor.camera.rgb'))}"
    )

    print("=" * 60)
    print("Connection successful!")


if __name__ == "__main__":
    main()