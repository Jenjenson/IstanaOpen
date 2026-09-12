# Shared simulation contracts (schema 1)

These contracts support a fictional, synthetic sensor simulation. They do not
implement sensing, drone motion, placement optimization, or learning. The existing
architectural viewer remains the default game mode.

The separate [swarm implementation](SWARM_SIMULATION.md) now consumes these group
configs and produces drone states. Its typed local commands do not change the
shared schema-1 policy no-op. The implemented [Red Team Manager](../Source/IstanaOpen/Simulation/RedTeam/README.md)
generates world-space `FIstanaSwarmConfig` groups around a shared objective and uses
that solver through `AIstanaSwarmManager`. It is a scene spawner/controller, not an
implementation of the shared policy interface or the planned episode coordinator.
Use the swarm guide for movement configuration and partial-path objective following.

Manager-relative offsets are an editor setup convenience for `AIstanaSwarmManager`;
its explicit `InitializeSimulation` API still accepts world-space configs. Red Team
Manager generates world-space centers relative to the objective instead. Local
`FIstanaSwarmCommand.bAllowPartialPath` and group navigation status do not change the
shared schema, sensor observations or `FIstanaPolicyActionBatch`.

## Start here

All public types live in `Source/IstanaOpen/Simulation`. Include them with
`#include "Simulation/IstanaSimulationTypes.h"`. No external plugins or packages
are required. Build against Unreal Engine 5.5.4. The project explicitly enables its
bundled editor browser providers and `EngineAssetDefinitions`; retain these entries.

See the root [README contributor setup](../README.md#working-on-another-machine)
for Git LFS, engine association, and local engine path guidance.

## Type reference

All `F` types below are `USTRUCT(BlueprintType)` value types. `U` preset classes
are Data Assets. Blueprint displays friendly names with spaces and may omit the
C++ `F` prefix. Read-only metadata such as schema version is set by C++ defaults.

| Type | Fields / responsibility |
| --- | --- |
| `FIstanaScenarioConfig` | `SchemaVersion`, `ScenarioId`, `Seed`, `DurationSeconds`, `FixedStepSeconds`, `Map`, `SensorBudget`, `Swarms`, `Conditions`, `InitialSensors` |
| `FIstanaSwarmConfig` | `GroupId`, `DroneCount`, `SpawnOriginCm`, `SpawnRadiusCm`, `MovementPresetId`; registry resolution is future coordinator work |
| `FIstanaConditions` | `TimeOfDayHours`, `PrecipitationFraction`, `VisibilityFraction`, `WindVelocityCmPerSecond` |
| `FIstanaDroneState` | `DroneId`, `GroupId`, `PositionCm`, `VelocityCmPerSecond`, `bActive`; simulator-only truth |
| `FIstanaSensorConfig` | `SensorId`, `SensorType`, `Transform`, `Model`, `SampleEveryNSteps` |
| `FIstanaSensorModelParams` | `CoverageRadiusCm`, `DetectionProbabilityPerSample`, `FalsePositiveProbabilityPerSample`, `PositionNoiseStdDevCm`, `LatencySteps`, `bRequiresLineOfSight` |
| `FIstanaReportedObservation` | `bHasPosition`, `PositionCm`; measured/reported position, not guaranteed truth |
| `FIstanaDetectionEvent` | `SensorId`, `SequenceNumber`, `SampleStep`, `DeliveryStep`, `Observation`, `Confidence` |
| `FIstanaEpisodeMetrics` | `SensorSampleCount`, `EligibleObjectSampleCount`, `TrueDetectionCount`, `MissedDetectionCount`, `FalseObservationCount`, `bHasFirstDetection`, `FirstDetectionStep` |
| `FIstanaEpisodeResult` | `SchemaVersion`, `ScenarioId`, `RunId`, `Seed`, `SimulationVersion`, `SensorModelVersion`, `ExecutedSteps`, `FixedStepSeconds`, `Metrics`, `CompletionReason` |
| `FIstanaPolicyContext` | `RunId`, `Role`, `PolicySeed`, `FixedStepSeconds`; passed once on reset |
| `FIstanaPolicyObservationBatch` | `RunId`, `DecisionStep`, `Detections`; coordinator-filtered inputs |
| `FIstanaPolicyActionBatch` | `RunId`, `DecisionStep`; no-op only, no controller commands yet |
| `FIstanaValidationIssue` | `FieldPath`, `Severity`, `Message`; display in UI or log |
| `UIstanaScenarioDataAsset` | `Description`, `Config`; `CreateRuntimeConfig()` returns a copy |
| `UIstanaSensorPresetDataAsset` | `Description`, `SensorType`, `Model`, `SampleEveryNSteps`; `CreateSensorConfig()` assigns instance ID and placement |
| `IIstanaPolicyInterface` | `ResetPolicy`, `ReceiveObservations`, `ProduceActions` |
| `UIstanaNoOpPolicy` | Reference policy implementing the interface; no world mutations |

Enums: `EIstanaSensorType` currently supports `AbstractCoverage`;
`EIstanaPolicyRole` supports `Observer`, `Blue`, `Red`; completion reasons are
`NotCompleted`, `DurationReached`, `ScenarioCompleted`, `Cancelled`, `Failed`.
Validation severities are `Warning` and `Error` (only errors currently emitted).

| File | Responsibility |
| --- | --- |
| `IstanaSimulationTypes.h` | Blueprint-accessible value contracts and enums |
| `IstanaSimulationPresets.h` | Scenario and sensor Data Asset classes |
| `IstanaSimulationValidation.h` | Blueprint/C++ structural validation and an in-memory example |
| `IstanaPolicyInterface.h` | Synchronous policy interface and no-op reference implementation |
| `Tests/IstanaSimulationContractsTests.cpp` | Contract validation, preset isolation, policy lifecycle tests |

## Ownership across the five workstreams

| Owner | Reads | Produces / owns |
| --- | --- | --- |
| Simulation coordinator | Resolved scenario, proposed actions | Run ID, step clock, live sensor registry, episode lifecycle; validates and applies actions |
| Swarm simulation | `Swarms` and conditions | `FIstanaDroneState` ground truth, deterministic episode-local drone IDs |
| Sensors | Sensor configs, conditions, evaluator-accessible world state | `FIstanaDetectionEvent`; separate private truth associations |
| Operator UI | Preset assets and validation issues | A mutable scenario copy; never writes runtime changes back into the preset |
| Evaluation / policies | Explicitly permitted observation batches; evaluator separately reads truth | Action envelopes and final `FIstanaEpisodeResult` |

The coordinator owns runtime state. Policies propose changes rather than mutating
actors through this API. Review contract changes with affected owners before
changing fields, units, or semantics.

## Units, identity, time, and versions

- Positions are world-space centimetres, Z up; velocities are centimetres/second.
  Sensor transforms require unit scale. Model coverage does not depend on scale.
- Sensor, group, and drone IDs are nonnegative integers in separate namespaces.
  Each is unique within its namespace and episode. `INDEX_NONE` means unset.
  Allocate IDs in a stable order, retain them across snapshots, and do not reuse
  them during an episode. Do not use actor names or addresses as IDs.
- `RunId` is a new GUID per episode; it is identity metadata, not a randomness seed.
- Steps are zero-based integers. Timestamp seconds = step * fixed timestep.
  `ExecutedSteps` is a count, so a sample in a completed run has a step smaller
  than that count. At reset, the first observation batch may have step zero.
- Sample each sensor on steps divisible by `SampleEveryNSteps`, beginning at zero.
  Sampling interval seconds = `SampleEveryNSteps * FixedStepSeconds`.
- `SampleStep` is measurement time; `DeliveryStep` is availability time and cannot
  precede sampling. The coordinator must not deliver observations early.
- `SequenceNumber` is nonnegative and monotonically increases per sensor per run.
  Deduplication and sequence ordering require coordinator state and are not checked
  by the stateless event validator.
- The coordinator should stop after the first completed step that reaches the
  configured duration. Nonintegral duration/timestep ratios therefore round up;
  use executed steps when computing the actual duration.
- `SchemaVersion` identifies the data layout (currently 1). `SimulationVersion`
  and `SensorModelVersion` identify behavior and should record meaningful build
  or model revisions. Reject unsupported schemas rather than silently accepting them.
- A seed alone does not guarantee cross-platform reproducibility. Stable iteration,
  separate seeded random streams, and timestep scheduling remain coordinator duties.

## Presets and the operator panel

In the Content Browser, choose **Miscellaneous > Data Asset**, then select
`IstanaScenarioDataAsset` or `IstanaSensorPresetDataAsset`. Prefer separate assets
per scenario so teammates rarely edit the same binary file.

Ready-made examples are included in `Content/Simulation/Examples`:

- `DA_SyntheticContractExample`: scenario template with its map unassigned.
- `DA_AbstractSensorExample`: reusable abstract sensor defaults.
- `BP_NoOpPolicyExample`: Blueprint subclass of the native no-op policy.

Duplicate these assets for your work. `Tools/create_simulation_examples.py` creates
missing examples and smoke-tests the Blueprint subclass without overwriting existing
assets. The subclass inherits the native behavior; the smoke check does not test
a custom Blueprint event-graph override.

Scenario workflow:

1. Select a scenario Data Asset.
2. Call `Create Runtime Config` and store its returned struct on the coordinator
   or UI model. This is a value copy, including nested arrays and model parameters.
3. Edit that copy through the panel.
4. Call `Validate Scenario`; show issues using `FieldPath` and `Message`.
5. If valid, the future map loader resolves the map reference and the coordinator
   starts the run. Validation does not load or prove existence of referenced assets.

`Make Example Scenario` returns an in-memory example with three stationary objects
and one abstract sensor. Its map is intentionally unassigned: select a fictional
map before validation/launch. This is also usable as the source for a preset's Config.

`Create Sensor Config` on a sensor preset copies model defaults, assigns an ID,
and applies placement. Validate the returned instance before adding it to the
runtime registry. Changing the instance does not change the preset.

Assets are templates, not live state. C++ callers technically can mutate assets;
the copy API and tests enforce the intended usage convention, not access isolation.
Soft map references remain references in the copy. Package/cook configuration must
include the chosen maps when a simulation launcher is implemented.

## Observations and abstract models

### C++ configuration and event example

This is a usage fragment for a future coordinator; `ScenarioPreset` is a selected
`UIstanaScenarioDataAsset*`. The example does not implement sensor behavior.

```cpp
#include "Simulation/IstanaSimulationPresets.h"
#include "Simulation/IstanaSimulationValidation.h"

FIstanaScenarioConfig Runtime = ScenarioPreset->CreateRuntimeConfig();
TArray<FIstanaValidationIssue> Issues;
if (!UIstanaSimulationValidation::ValidateScenario(Runtime, Issues))
{
    for (const FIstanaValidationIssue& Issue : Issues)
    {
        UE_LOG(LogTemp, Warning, TEXT("%s: %s"), *Issue.FieldPath, *Issue.Message);
    }
    return; // Do not start an invalid scenario.
}

FIstanaDetectionEvent Event;
Event.SensorId = 0; // Must exist in the coordinator's current sensor registry.
Event.SequenceNumber = 0;
Event.SampleStep = 10;
Event.DeliveryStep = 12;
Event.Observation.bHasPosition = true;
Event.Observation.PositionCm = FVector(100, 200, 300);
Event.Confidence = 0.8;
const TArray<int32> KnownSensorIds = {0};
const bool bValid = UIstanaSimulationValidation::ValidateDetection(
    Event, KnownSensorIds, Issues);
// Deliver only when the coordinator reaches DeliveryStep and permits this observation.
```

In Blueprint, use **Make Istana Detection Event** and **Make Istana Reported
Observation**, then **Validate Detection** with the current sensor IDs. Use
**Break** nodes or **Set Members in Struct** to inspect/update values. Always store
the result of a copy-returning function before editing nested arrays in the UI.

`FIstanaDroneState` is ground truth. It is deliberately absent from policy
observation batches. A detection contains only a sensor ID, sample/delivery steps,
reported position and confidence. It has no true drone ID or false-positive label.
Keep evaluator-only associations in a separate store, never in policy payloads.

Schema 1 supports reported-position detections only. Set `bHasPosition = true` and
supply finite coordinates; `(0,0,0)` is a valid position. The availability flag is
explicit so additional measurement types can be introduced without interpreting
zero as missing. Confidence zero is valid and is not a missing-value marker.

The model parameters specify:

- Radius in centimetres and optional line-of-sight gating.
- Detection probability per eligible object per sample.
- Probability of one false observation per sensor sample (at most one).
- Position noise standard deviation in centimetres and delivery latency in steps.

These are synthetic parameters, not measured sensor specifications. No noise
distribution, eligibility rule, condition-to-probability mapping, or sensing
implementation is supplied yet. The sensor owner must document/version these
choices. Visual weather effects do not automatically change sensor probabilities.

Conditions specify hour in `[0,24)`, precipitation and visibility fractions in
`[0,1]`, and a wind vector. They are inputs shared by renderers and model consumers,
not an implicit physics model.

## Policy lifecycle

The interface supports C++ and Blueprint implementations. C++ callers must use
the generated `Execute_` wrappers so Blueprint overrides are honored:

```cpp
#include "Simulation/IstanaPolicyInterface.h"
#include "Simulation/IstanaSimulationValidation.h"

// Keep Policy in a UPROPERTY on its owner across frames to prevent garbage collection.
UIstanaNoOpPolicy* Policy = NewObject<UIstanaNoOpPolicy>(Coordinator);
FIstanaPolicyContext Context;
Context.RunId = FGuid::NewGuid();
IIstanaPolicyInterface::Execute_ResetPolicy(Policy, Context);

FIstanaPolicyObservationBatch Batch;
Batch.RunId = Context.RunId;
Batch.DecisionStep = 0;
// The coordinator appends permitted, available detections only.
IIstanaPolicyInterface::Execute_ReceiveObservations(Policy, Batch);
const FIstanaPolicyActionBatch Actions =
    IIstanaPolicyInterface::Execute_ProduceActions(Policy);
TArray<FIstanaValidationIssue> Issues;
const bool bAccepted = UIstanaSimulationValidation::ValidateActionBatch(
    Actions, Context, Batch.DecisionStep, Issues);
```

In Blueprint, construct an `IstanaNoOpPolicy` object or a Blueprint subclass,
retain it in a variable, and call the same interface functions. Reset once per run,
then receive observations and produce actions once per decision step.

The reference policy ignores other runs and nonincreasing steps. Before any valid
observation, it returns step `-1`, which action validation rejects. Reset clears
that step. The coordinator must additionally prevent applying a batch twice.

**Schema 1 action batches are no-op envelopes only.** Role-specific commands,
capabilities, action budgets, and policy-specific observation payloads must be added
with their controller implementations and validation. Do not add an arbitrary string
command or a raw world-state field to bypass that work. All roles currently have the
same no-op capability. There is no external Python or RL integration yet.

An interface is not a security boundary: arbitrary in-process Blueprint/C++ can
inspect the world. Observation restrictions depend on disciplined implementations
and coordinator construction of payloads.

## Validation and results

Each validator clears and replaces its output issue array. Currently all issues
are errors, and the return value is true exactly when the array is empty. Warning
severity is reserved for future nonblocking diagnostics. Editor clamps are hints;
runtime validation is authoritative.

- `ValidateScenario`: schema, IDs, budget, finite values, conditions, timing, map path.
- `ValidateSensor`: instance/model values and transform.
- `ValidateDroneStates`: unique IDs, group references, finite positions/velocities.
  Allows partial snapshots; spawn-count enforcement belongs to the coordinator.
- `ValidateDetection`: known sensor, timing, probability and required measurement.
  Pass the current sensor registry IDs, not just the initial scenario placements.
- `ValidateEpisodeResult`: final identity/version metadata, counts and first detection.
- `ValidateActionBatch`: active run, current step, supported context role and timing.

The metrics count true detections and misses per eligible object/sample pair:
`TrueDetectionCount + MissedDetectionCount == EligibleObjectSampleCount`.
False observations use sensor samples as their denominator. First detection means
first true detection at sample time, not delivery time. A no-detection result sets
`bHasFirstDetection = false`; its step field is ignored. A final result must have a
completion reason other than `NotCompleted`; cancelled runs can have zero steps.

These validators do not verify that metrics match world history. The evaluator is
responsible for truth association and consistent accumulation. JSON export and
schema migration are follow-on work; exports should include the resolved scenario
snapshot, not only mutable asset references.

## Build and test

Close the editor before rebuilding reflected C++ types, then run from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor
```

In the editor's Session Frontend Automation tab, run `Istana.Simulation.Contracts`.
Alternatively use the installed editor command-line executable:

```powershell
& 'C:\Program Files\Epic Games\UE_5.5\Engine\Binaries\Win64\UnrealEditor-Cmd.exe' `
  "$PWD\IstanaOpen.uproject" -unattended -nop4 -nosound -nullrhi `
  '-ExecCmds=Automation RunTests Istana.Simulation.Contracts' `
  '-TestExit=Automation Test Queue Empty' '-ReportExportPath=Saved/Automation/Contracts'
```

Tests cover malformed configs/events/results, identity references, nested preset
copy isolation, reflected native interface calls, and stale/reset policy behavior.
Blueprint subclass smoke verification is available through
`Tools/create_simulation_examples.py` after an editor build; see its header.

## Compatibility review (11 September 2026)

- Restored the shared engine association to `5.5` after Unreal wrote a locally
  registered installation GUID. All developers should use patch version 5.5.4.
- Both editor and standalone Win64 Development targets built on the development
  machine. The standalone game target was checked after the editor plugin fixes.
- All four contract automation tests passed with the final plugin configuration.
  The example Blueprint subclass lifecycle and preset copy smoke check also passed.
- The three new example assets match the repository's existing Git LFS rules.
- No developer-specific paths occur in the new runtime code or example generator.
  The module include path uses `ModuleDirectory`, resolved by Unreal per checkout.
- The original viewer implementation, original map/assets, and Config files are
  unchanged by the shared-contract work. Browser helpers are editor-only bundled
  plugins; runtime module dependencies are unchanged.

This is local build and contract evidence, not a clean-machine certification.
Packaging/cooking was not rerun for this change, and macOS/Linux builds were not
tested. A colleague should pull LFS content, build locally, and run the contract
tests before beginning integration. The local compiler was MSVC 14.44; Unreal
warned that 14.38.33130 is preferred, but the builds completed successfully.
