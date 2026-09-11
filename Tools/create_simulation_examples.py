"""Run with UE 5.5.4 editor Python after building the shared contracts.

Creates missing example assets under /Game/Simulation/Examples without replacing
existing assets. The scenario deliberately has no map: choose a fictional map.
Also checks a Blueprint subclass of the no-op policy and writes a smoke receipt.

From the Unreal Python console:
    exec(open(unreal.Paths.project_dir() + 'Tools/create_simulation_examples.py').read())

Or UnrealEditor-Cmd.exe <project> -run=pythonscript -script=<this file>
    -unattended -nop4 -nosound -nullrhi
"""

import json
from pathlib import Path
import unreal


ROOT = "/Game/Simulation/Examples"
ASSET_TOOLS = unreal.AssetToolsHelpers.get_asset_tools()


def data_asset(name, asset_class):
    path = ROOT + "/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        asset = unreal.EditorAssetLibrary.load_asset(path)
        if not isinstance(asset, asset_class):
            raise RuntimeError("Existing asset has a different class: " + path)
        return asset, False
    factory = unreal.DataAssetFactory()
    factory.set_editor_property("data_asset_class", asset_class)
    asset = ASSET_TOOLS.create_asset(name, ROOT, asset_class, factory)
    if asset is None:
        raise RuntimeError("Failed to create " + path)
    return asset, True


def run():
    scenario, created = data_asset("DA_SyntheticContractExample", unreal.IstanaScenarioDataAsset)
    if created:
        scenario.set_editor_property("description", "Contract example. Assign a fictional map before use.")
        scenario.set_editor_property("config", unreal.IstanaSimulationValidation.make_example_scenario())
        if not unreal.EditorAssetLibrary.save_loaded_asset(scenario):
            raise RuntimeError("Could not save scenario preset")

    sensor, created = data_asset("DA_AbstractSensorExample", unreal.IstanaSensorPresetDataAsset)
    if created:
        sensor.set_editor_property("description", "Synthetic coverage model defaults; not a calibrated sensor.")
        if not unreal.EditorAssetLibrary.save_loaded_asset(sensor):
            raise RuntimeError("Could not save sensor preset")

    blueprint_path = ROOT + "/BP_NoOpPolicyExample"
    if not unreal.EditorAssetLibrary.does_asset_exist(blueprint_path):
        factory = unreal.BlueprintFactory()
        factory.set_editor_property("parent_class", unreal.IstanaNoOpPolicy)
        blueprint = ASSET_TOOLS.create_asset("BP_NoOpPolicyExample", ROOT, unreal.Blueprint, factory)
        if blueprint is None:
            raise RuntimeError("Could not create Blueprint policy example")
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        if not unreal.EditorAssetLibrary.save_loaded_asset(blueprint):
            raise RuntimeError("Could not save Blueprint policy example")

    policy_class = unreal.EditorAssetLibrary.load_blueprint_class(blueprint_path)
    policy = unreal.new_object(policy_class)
    context = unreal.IstanaPolicyContext()
    context.set_editor_property("run_id", unreal.GuidLibrary.new_guid())
    policy.reset_policy(context)
    batch = unreal.IstanaPolicyObservationBatch()
    batch.set_editor_property("run_id", context.get_editor_property("run_id"))
    batch.set_editor_property("decision_step", 7)
    policy.receive_observations(batch)
    actions = policy.produce_actions()
    if actions.get_editor_property("decision_step") != 7:
        raise RuntimeError("Blueprint subclass did not execute the policy contract")
    policy.reset_policy(context)
    if policy.produce_actions().get_editor_property("decision_step") != -1:
        raise RuntimeError("Blueprint subclass did not reset")

    runtime = scenario.create_runtime_config()
    old_seed = scenario.get_editor_property("config").get_editor_property("seed")
    runtime.set_editor_property("seed", old_seed ^ 1)
    if scenario.get_editor_property("config").get_editor_property("seed") != old_seed:
        raise RuntimeError("Runtime copy mutated the scenario preset")

    receipt = {
        "status": "PASS",
        "blueprint_subclass_policy_lifecycle": True,
        "python_preset_copy_isolation": True,
        "assets": [scenario.get_path_name(), sensor.get_path_name(), blueprint_path],
        "note": "Scenario requires an operator-selected fictional map. No simulation executed.",
    }
    output = Path(unreal.Paths.project_saved_dir()) / "Automation" / "Contracts" / "example-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    unreal.log("Simulation contract example smoke: PASS")


run()
