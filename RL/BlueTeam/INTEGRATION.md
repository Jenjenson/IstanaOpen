# Integration and live training

## What is runnable here

The Python dry-run, policy, checkpoint inspection, and Python tests are
self-contained. The live client speaks TRIAD's `triad.rl_training.v4` contract
over Unreal Remote Control. It is not an adapter for Istana Open's
`IIstanaPolicyInterface`, whose schema-1 actions are no-op only.

`NativeReference/` contains selected public, private, and editor source from
the TRIAD RL implementation. It is **not a complete Unreal plugin** and is not
compiled by Istana Open. It references TRIAD sensor actors, drone and swarm
components, protected-zone types, and Cesium georeferencing that are not
provided by this package. The original toy map and generated data asset are
also not included. Copying these sources into `Source/IstanaOpen` is not a
supported setup procedure.

| Component | Current state |
| --- | --- |
| Blue profile/position policy and CPU trainer | Included; runnable with the Python dry-run |
| TRIAD Remote Control client | Included; requires a compatible live TRIAD manager |
| TRIAD native manager/model/types | Selected reference source only |
| Istana Open Red Team Manager | Existing separate swarm implementation |
| Blue-to-Istana policy adapter, sensor runtime, rewards and episode coordinator | Integration work still required |

## Use an existing TRIAD host

The commands below assume you already have the original functioning TRIAD
project, its complete `TRIADSensorFusion` plugin, Cesium, and the
`/Game/TRIAD/RL/RL-Toy-OneDrone` map. They do not launch the Istana Open viewer.
Use the Python environment created in the [quickstart](README.md#try-it-without-unreal).

1. Build and open that TRIAD project with Unreal Engine 5.5. Start the editor
   with `-TRIADRLTraining -TRIADRLDebug`. To evaluate checkpoints, also include
   `-TRIADRLEvaluation`.
2. Make the host use this exact `DefaultTrainingConfig.json`, either through
   its normal `Content/TRIAD/RL/DefaultTrainingConfig.json` location or
   `-TRIADRLConfig="<absolute path to this package's JSON>"`. The raw JSON
   fingerprint must match the Python side.
3. Open the toy map, start the loopback Remote Control server with
   `WebControl.StartServer`, and press Play.
4. Print the active manager path in the Unreal editor Python console:

   ```python
   world = unreal.EditorLevelLibrary.get_pie_worlds(False)[0]
   manager = unreal.GameplayStatics.get_all_actors_of_class(
       world, unreal.TRIADAdversarialTrainingManager
   )[0]
   print(manager.get_path_name())
   ```

Use the printed path in the commands below. It can change after restarting
Play. One Python driver should own each manager. The client uses the local
endpoint `http://127.0.0.1:30010`.

## Train and resume

From the Istana Open repository root, this starts a new live run using the
published experiment's optimizer settings and fixed Red controls:

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\train_blue_placement.py `
  --object-path "<printed PIE manager path>" `
  --config .\RL\BlueTeam\DefaultTrainingConfig.json `
  --episodes 210 --batch-size 5 --seed 7301 `
  --hidden-size 32 --learning-rate 0.0003 --entropy-coefficient 0.01 `
  --checkpoint-every 50 --checkpoint-dir .\Saved\BlueRL\live-01
```

The default Red deployment is `[0,0,0,0,0]` and movement is `[0,-1,0]` in
normalized ENU controls. The JSON fixes the approach bearing, altitude,
population, and spawn radius. `--red-mode uniform` samples controls, but cannot
add variation along a dimension whose configured range is fixed.

To add 50 episodes to the published checkpoint:

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\train_blue_placement.py `
  --object-path "<printed PIE manager path>" `
  --config .\RL\BlueTeam\DefaultTrainingConfig.json `
  --resume .\RL\BlueTeam\Checkpoints\toy-210 `
  --episodes 50 --batch-size 5 --seed 7301 `
  --learning-rate 0.0003 --entropy-coefficient 0.01 `
  --checkpoint-dir .\Saved\BlueRL\continued
```

`--episodes` means episodes to add. Resume requires matching feature contract,
config fingerprint, seed, optimizer settings, and Red script. Checkpoints
contain integrity-hashed NumPy arrays loaded with pickle disabled. Changing
sensor settings or scenario ranges creates a different experiment; start a
new run for that configuration.

The published run was developed in multiple segments while the implementation
was being corrected. The command above reproduces its configuration and
training method with the final code; it does not promise an identical
historical sequence or checkpoint hash.

## Evaluate the published checkpoint

With the TRIAD host in evaluation mode, run:

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\evaluate_checkpoint.py `
  --checkpoint .\RL\BlueTeam\Checkpoints\toy-210 --stochastic `
  --object-path "<printed PIE manager path>" `
  --config .\RL\BlueTeam\DefaultTrainingConfig.json `
  --episodes 50 --seed 1500000000 --output .\Saved\BlueRL\trained-evaluation
```

For the initialized stochastic baseline, replace
`--checkpoint .\RL\BlueTeam\Checkpoints\toy-210` with
`--initial --policy-seed 7301`, retaining `--stochastic`, the scenario, and the
same episode seed range. Choose a different output directory. For deterministic
deployment, use the checkpoint and omit `--stochastic`.

Evaluation never runs the optimizer and verifies that the parameter hash
remains unchanged. The native state is checked against the supplied JSON;
the checkpoint records the training configuration's fingerprint. Evaluation
can accept a different feature-compatible configuration, so compare those
fingerprints when reproducing a result. The JSON fingerprint does not identify
the map, collision/LOS geometry, objective transform, or native binary.
Record those separately for new experiments.

## Connect to Istana Open

An implementation against the existing
[simulation contracts](../../Docs/SIMULATION_CONTRACTS.md) needs these changes:

1. Extend the versioned policy action contract to represent profile selection,
   continuous placement, and commit. Define normalized coordinates relative
   to an objective and translate them to Istana Open's centimetre units.
2. Implement sensor profiles, legal placement, analytical observations, and
   stable sensor identities against the shared sensor/detection types.
3. Add an episode coordinator that resets the
   [Red Team Manager](../../Source/IstanaOpen/Simulation/RedTeam/README.md),
   applies Blue's layout, advances fixed simulation steps, and owns rewards
   and termination. Blue must receive permitted sensor observations rather
   than unrestricted swarm ground truth.
4. Implement and test an environment adapter for that coordinator. Version the
   feature contract, train a new checkpoint, and evaluate on a separate
   curriculum of approach bearings, altitudes, emitters, and swarm sizes.

Until that work lands, the Python example and TRIAD experiment remain separate
from the packaged Istana Open application.
