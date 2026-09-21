"""Create or refresh the dedicated Mavic 3 Enterprise swarm movement preset.

Run after building the editor module:

    UnrealEditor-Cmd.exe <project> -run=pythonscript -script=<this file> \
        -unattended -nop4 -nosound -nullrhi

The script owns only /Game/Simulation/Presets/DA_Mavic3E_NormalFlight. It does
not modify maps, actors, or the older example movement preset.
"""

import json
from pathlib import Path
import unreal


ASSET_NAME = "DA_Mavic3E_NormalFlight"
ASSET_DIRECTORY = "/Game/Simulation/Presets"
ASSET_PATH = ASSET_DIRECTORY + "/" + ASSET_NAME

# Unreal units are centimetres and seconds. Published DJI limits are converted
# from metres per second. Acceleration is g*tan(30 degrees); jerk is controller
# tuning because DJI does not publish an acceleration-jerk envelope.
EXPECTED = {
    "drone_radius_cm": 25.0,
    "max_speed_cm_per_second": 1500.0,
    "cruise_speed_cm_per_second": 900.0,
    "max_acceleration_cm_per_second_squared": 566.0,
    "max_turn_degrees_per_second": 200.0,
    "max_ascent_speed_cm_per_second": 600.0,
    "max_descent_speed_cm_per_second": 600.0,
    "max_tilt_degrees": 30.0,
    "max_jerk_cm_per_second_cubed": 1200.0,
    "response_seconds": 0.5,
    "arrival_radius_cm": 120.0,
}


def load_or_create():
    if unreal.EditorAssetLibrary.does_asset_exist(ASSET_PATH):
        asset = unreal.EditorAssetLibrary.load_asset(ASSET_PATH)
        if not isinstance(asset, unreal.IstanaSwarmMovementPreset):
            raise RuntimeError("Existing asset has the wrong class: " + ASSET_PATH)
        return asset, False

    factory = unreal.DataAssetFactory()
    factory.set_editor_property("data_asset_class", unreal.IstanaSwarmMovementPreset)
    asset = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        ASSET_NAME, ASSET_DIRECTORY, unreal.IstanaSwarmMovementPreset, factory)
    if asset is None:
        raise RuntimeError("Could not create " + ASSET_PATH)
    return asset, True


def run():
    preset, created = load_or_create()
    settings = unreal.IstanaSwarmSettings()
    for property_name, expected_value in EXPECTED.items():
        settings.set_editor_property(property_name, expected_value)
    settings.set_editor_property("wind_velocity_cm_per_second", unreal.Vector(0.0, 0.0, 0.0))
    preset.set_editor_property("settings", settings)

    if not unreal.EditorAssetLibrary.save_loaded_asset(preset, only_if_is_dirty=False):
        raise RuntimeError("Could not save " + ASSET_PATH)

    saved = preset.get_editor_property("settings")
    for property_name, expected_value in EXPECTED.items():
        actual_value = saved.get_editor_property(property_name)
        if abs(actual_value - expected_value) > 0.001:
            raise RuntimeError(
                f"Preset verification failed for {property_name}: "
                f"expected {expected_value}, got {actual_value}")

    receipt = {
        "status": "PASS",
        "asset": ASSET_PATH,
        "created": created,
        "settings": EXPECTED,
        "wind_velocity_cm_per_second": [0.0, 0.0, 0.0],
    }
    output = (Path(unreal.Paths.project_saved_dir()) / "Automation" / "Swarm"
              / "mavic-preset.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    unreal.log("MAVIC_SWARM_PRESET_PASS: " + ASSET_PATH)


run()
