"""Create BP_SwarmObjective (a TargetPoint Blueprint) and test marker following.

Run in a separate UE editor Python commandlet after building. Loads the synthetic
arena but never saves it. Existing Blueprint assets are not replaced.
"""
import json
from pathlib import Path
import unreal


ASSET_PATH = "/Game/Simulation/Examples/BP_SwarmObjective"
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)


def run():
    if not unreal.EditorAssetLibrary.does_asset_exist(ASSET_PATH):
        factory = unreal.BlueprintFactory()
        factory.set_editor_property("parent_class", unreal.TargetPoint)
        blueprint = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "BP_SwarmObjective", "/Game/Simulation/Examples", unreal.Blueprint, factory)
        if blueprint is None:
            raise RuntimeError("Could not create objective Blueprint")
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        if not unreal.EditorAssetLibrary.save_loaded_asset(blueprint):
            raise RuntimeError("Could not save objective Blueprint")

    if not levels.load_level("/Game/Simulation/Maps/SyntheticSwarmArena"):
        raise RuntimeError("Synthetic test arena is missing")
    objective_class = unreal.EditorAssetLibrary.load_blueprint_class(ASSET_PATH)
    marker = actors.spawn_actor_from_class(objective_class, unreal.Vector(1000, 0, 700))
    manager = actors.spawn_actor_from_class(unreal.IstanaSwarmManager, unreal.Vector())
    manager.set_editor_property("use_world_collision", False)
    manager.set_editor_property("spawn_visuals", False)
    manager.set_editor_property("auto_advance", False)
    group = unreal.IstanaSwarmConfig()
    group.set_editor_property("group_id", 0)
    group.set_editor_property("spawn_origin_cm", unreal.Vector(0, 0, 700))
    group.set_editor_property("movement_preset_id", "Boids")
    manager.set_editor_property("swarms", [group])
    manager.set_editor_property("objective_target", marker)
    assert manager.reset_simulation() == ""
    for _ in range(300):
        manager.advance_one_step()
    position = manager.get_drone_states()[0].position_cm
    assert abs(position.x - 1000) < 5 and abs(position.y) < 5, "Did not reach initial marker"
    assert manager.get_group_statuses()[0].route_completed

    marker.set_actor_location(unreal.Vector(-1000, 500, 700), False, False)
    for _ in range(500):
        manager.advance_one_step()
    position = manager.get_drone_states()[0].position_cm
    assert abs(position.x + 1000) < 5 and abs(position.y - 500) < 5, "Did not follow moved marker"

    # Clearing the shared marker must stop the old route.
    manager.set_editor_property("objective_target", None)
    manager.advance_one_step()
    assert manager.get_group_statuses()[0].mode == unreal.IstanaSwarmCommandType.STOP
    assert "removed" in manager.get_editor_property("objective_status")

    # Existing plain Target Point actors are also supported, independent of their label.
    plain = actors.spawn_actor_from_class(unreal.TargetPoint, unreal.Vector(0, 0, 700))
    plain.set_actor_label("objective")
    manager.set_editor_property("objective_target", plain)
    manager.advance_one_step()
    assert manager.get_group_statuses()[0].mode == unreal.IstanaSwarmCommandType.FOLLOW_WAYPOINTS
    actors.destroy_actor(plain)
    manager.advance_one_step()
    assert manager.get_group_statuses()[0].mode == unreal.IstanaSwarmCommandType.STOP

    actors.destroy_actor(marker)
    actors.destroy_actor(manager)
    output = Path(unreal.Paths.project_saved_dir()) / "Automation" / "Swarm" / "objective-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "PASS", "blueprint": ASSET_PATH,
        "arrival": True, "moving_marker": True, "cleared_marker_stops": True,
        "plain_target_point": True, "removed_marker_stops": True,
        "map_saved": False,
    }, indent=2), encoding="utf-8")
    unreal.log("SWARM_OBJECTIVE_SMOKE_PASS")


run()
