"""Create a fictional swarm test arena and run an actor-adapter smoke check.

Run through UE 5.5.4 editor Python after building:
UnrealEditor-Cmd.exe <project> -run=pythonscript -script=<this file>
    -unattended -nop4 -nosound -nullrhi

Creates missing assets only. Existing demo maps are loaded, never overwritten.
Run in a separate commandlet process; this script changes the active editor level.
"""

import json
from pathlib import Path
import unreal


MAP = "/Game/Simulation/Maps/SyntheticSwarmArena"
PRESET = "/Game/Simulation/Examples/DA_SwarmMovementExample"
ACTORS = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
LEVELS = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)


def spawn(cls, name, position, rotation=unreal.Rotator()):
    actor = ACTORS.spawn_actor_from_class(cls, unreal.Vector(*position), rotation)
    if actor is None:
        raise RuntimeError("Could not spawn " + name)
    actor.set_actor_label(name)
    return actor


def run():
    if unreal.EditorAssetLibrary.does_asset_exist(PRESET):
        preset = unreal.EditorAssetLibrary.load_asset(PRESET)
    else:
        factory = unreal.DataAssetFactory()
        factory.set_editor_property("data_asset_class", unreal.IstanaSwarmMovementPreset)
        preset = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "DA_SwarmMovementExample", "/Game/Simulation/Examples", unreal.IstanaSwarmMovementPreset, factory)
        if preset is None:
            raise RuntimeError("Could not create movement preset")
        settings = unreal.IstanaSwarmSettings()
        preset.set_editor_property("settings", settings)
        if not unreal.EditorAssetLibrary.save_loaded_asset(preset):
            raise RuntimeError("Could not save movement preset")

    if not unreal.EditorAssetLibrary.does_asset_exist(MAP):
        if not LEVELS.new_level(MAP):
            raise RuntimeError("Could not create synthetic arena")
        floor = spawn(unreal.StaticMeshActor, "Synthetic arena floor", (0, 0, 0))
        floor.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Cube.Cube"))
        floor.set_actor_scale3d(unreal.Vector(80, 80, 1))
        sphere = spawn(unreal.StaticMeshActor, "Level collision obstacle", (0, 0, 700))
        sphere.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Sphere.Sphere"))
        sphere.set_actor_scale3d(unreal.Vector(5, 5, 5))
        spawn(unreal.DirectionalLight, "Sun", (0, 0, 2000), unreal.Rotator(pitch=-45, yaw=-30))
        spawn(unreal.SkyLight, "Ambient", (0, 0, 2000))
        spawn(unreal.SkyAtmosphere, "Sky", (0, 0, 0))
        spawn(unreal.PlayerStart, "Observer start", (-3600, -4200, 2800), unreal.Rotator(pitch=-25, yaw=50))
        manager = spawn(unreal.IstanaSwarmManager, "Swarm demo - select to configure", (0, 0, 0))
        group = unreal.IstanaSwarmConfig()
        group.set_editor_property("group_id", 0)
        group.set_editor_property("drone_count", 12)
        group.set_editor_property("spawn_origin_cm", unreal.Vector(-1400, -1400, 700))
        group.set_editor_property("spawn_radius_cm", 350.0)
        group.set_editor_property("movement_preset_id", "Boids")
        manager.set_editor_property("swarms", [group])
        manager.set_editor_property("movement_preset", preset)
        manager.set_editor_property("enable_demo_keyboard", True)
        manager.set_editor_property("demo_waypoints_cm", [
            unreal.Vector(-1400, -1400, 700), unreal.Vector(1400, -1400, 700),
            unreal.Vector(1400, 1400, 700), unreal.Vector(-1400, 1400, 700)])
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        world.get_world_settings().set_editor_property("default_game_mode", unreal.IstanaSwarmDemoGameMode)
        if not LEVELS.save_current_level():
            raise RuntimeError("Could not save synthetic arena")
    elif not LEVELS.load_level(MAP):
        raise RuntimeError("Could not load existing arena")

    # Tests use a transient manager, so preexisting user edits are not mutated.
    manager = ACTORS.spawn_actor_from_class(unreal.IstanaSwarmManager, unreal.Vector(0, 0, 700))
    manager.set_editor_property("use_world_collision", False)
    manager.set_editor_property("spawn_visuals", True)
    # UE Python folds a bool return + one output parameter into Optional[str].
    # Success returns the empty error string; failure returns None.
    error = manager.reset_simulation()
    if error != "":
        raise RuntimeError("Smoke manager reset failed: " + str(error))

    def visual_count():
        return sum(1 for actor in ACTORS.get_all_level_actors()
                   if isinstance(actor, unreal.IstanaDroneVisual) and actor.get_owner() == manager)

    assert visual_count() == 12, "Missing visual actors"

    def positions():
        return [(s.position_cm.x, s.position_cm.y, s.position_cm.z) for s in manager.get_drone_states()]

    initial = positions()
    command = unreal.IstanaSwarmCommand()
    command.set_editor_property("run_id", manager.get_run_id())
    command.set_editor_property("group_id", 0)
    command.set_editor_property("type", unreal.IstanaSwarmCommandType.FOLLOW_WAYPOINTS)
    command.set_editor_property("waypoints_cm", [unreal.Vector(1500, 0, 700)])
    error = manager.submit_command(command)
    if error != "":
        raise RuntimeError("Smoke route rejected: " + str(error))
    for _ in range(300):
        manager.advance_one_step()
    diagnostics = manager.get_diagnostics()
    assert diagnostics.executed_steps == 300
    assert len(manager.get_drone_states()) == 12
    assert diagnostics.speed_violation_steps == 0
    assert diagnostics.acceleration_violation_steps == 0
    error = manager.reset_simulation()
    if error != "":
        raise RuntimeError("Smoke reset failed: " + str(error))
    assert initial == positions(), "Reset failed to reproduce initial state"
    assert visual_count() == 12, "Reset leaked visual actors"
    # Destroy the unsaved smoke manager and its visuals; never save smoke state to the map.
    ACTORS.destroy_actor(manager)
    receipt = {
        "status": "PASS", "map": MAP, "movement_preset": PRESET,
        "actor_adapter_steps": 300, "drone_count": 12, "deterministic_reset": True,
        "speed_violation_steps": diagnostics.speed_violation_steps,
        "acceleration_violation_steps": diagnostics.acceleration_violation_steps,
        "overlap_pair_steps": diagnostics.overlap_pair_steps,
    }
    output = Path(unreal.Paths.project_saved_dir()) / "Automation" / "Swarm" / "actor-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    unreal.log("SWARM_ACTOR_SMOKE_PASS")


run()
