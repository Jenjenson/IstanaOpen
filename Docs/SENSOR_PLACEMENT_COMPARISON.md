# Native sensor-placement comparisons

Open **Compare placements**, choose **Temporal RL · policy A, B or C**, and select
one of three **Perimeter approach** episodes. Both layouts appear side by side.
Play, pause, step or scrub the shared timeline to inspect exactly the same drone
flight on both maps. There are no manual placement or sensor configuration controls.

Use **Fixed placement** to select either:

- **Matched common sense · 3 sensors**: the original 60-drone paired evaluation.
- **Five directional sensors · measured workbench**: five surface-mounted,
  limited-FOV thermal cameras spread across representative coverage bearings.

The five-sensor choice is also measured native evidence, not a layout-only
mock-up. It replays each archived RL placement unchanged in the same seeded
random-bearing episode as the fixed placement. Both sides share the five drones, trajectories,
speeds, sensor capabilities, sites, budget and seed. The archived RL checkpoint
was trained under the older three-sensor contract and has **not** been retrained
as a five-sensor policy, so use this view to inspect the placement and warning
times rather than to claim a fair five-sensor RL benchmark.

The recorded mean warning times are:

| Episode | Archived policy A | Archived policy B | Archived policy C | Five sensors |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.53 s | 0.00 s | 0.00 s | 9.29 s |
| 2 | 1.50 s | 0.00 s | 0.84 s | 21.28 s |
| 3 | 0.00 s | 0.00 s | 0.00 s | 10.28 s |

Missed drones contribute zero warning, as in the rest of the project.

The nine comparisons are recordings of real runs in the repository's native
Unreal simulator, using the current merged sensor model. Playback needs no
running Unreal process. The separate **Recorded replay** mode retains all 18
historical synthetic cases with their original evidence unchanged.

## Start the console

Use the existing setup and launcher from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\setup_blue_python.ps1
powershell -ExecutionPolicy Bypass -File .\Tools\start_simulation_console.ps1
```

Alternatively, with Python 3.11+:

```sh
python -m pip install -e './RL/BlueTeam/Python[test,media]'
python RL/BlueTeam/Python/simulation_console.py
```

Open <http://127.0.0.1:9048/> and select **Compare placements**.

## Which algorithms and sensors are compared

Policy A, B and C are the existing published temporal RL checkpoints from training
runs 406, 407 and 408 respectively. The numbers remain in checkpoint metadata for
reproducibility. The console uses neutral names instead of implying a ranking.

The capture calls the same `istana_live.make_plan` used by **Live Unreal**. These
policies learned sensor type and site selection before the directional update.
The current public-forecast adapter supplies yaw and pitch; these checkpoints did
not learn orientation and have not been retrained on the new thermal model.
This comparison measures their present transfer behavior, not an intrinsic limit
of reinforcement learning or the performance of a newly trained directional policy.

The native catalogue includes the Teledyne FLIR Boson+ 640 18 mm thermal profile:
24-degree horizontal FOV, calculated vertical FOV, orientation, pixels-on-target
probability, weather effects, and native world-static line-of-sight. The other
current profiles (including the inexpensive EO sensors chosen by the baseline)
remain legacy radial simulation models. Their costs and probabilities are
simulator assumptions, not procurement prices or validated hardware performance.

Red uses the existing seeded radial placement helper and current native swarm
movement toward the objective. Each episode has **5 groups of 12 drones**. The
repository also contains a separate learned Red placement trainer, but the normal
Live evaluation does not load it; this is not self-play or a learned Red comparison.

## The fixed baseline

The common-sense rule favors new coverage per unit cost and discounts coverage
already provided by another sensor:

`marginal_coverage / actual_cost * (1 - 0.5 * overlap / candidate_coverage)`

Equal scores prefer greater separation, then stable catalogue/site/orientation
order. It stops at the budget or sensor limit, or when no useful legal coverage
remains. This approximates a sensible inexpensive deployment; it is not a human
participant study and does not claim optimality.

The rule uses only the public forecast, sensor catalogue, weather and legal sites.
It does not inspect hidden drone paths, score candidate layouts against the actual
episode, train, or optimize predicted temporal return. Directional sensor choices
use the shared joint type/site/yaw/pitch feature builder, so a 500 m thermal range
is not treated as an omnidirectional circle. Public planning assumes clear LOS;
actual scoring uses Unreal's world geometry for profiles requiring LOS.

All nine comparisons use exactly the same baseline: **three EO sensors, cost 2.1**.
Both sides receive **budget 3** and the native **three-sensor cap**, along with the
same catalogue, supported sites, spacing and deployment limits. A larger budget
would not by itself remove the native sensor-count cap. The rule is also available
as **Common sense · non-RL** in Live Unreal.

## Reading the metrics

- **Detected / confirmed / timely confirmed:** numbers of the 60 adversaries.
  Timely means confirmation at least 4 seconds before entry into the protected zone.
- **First detection:** earliest detection by any sensor on that side, or no detection.
- **Mean detection / confirmation time:** compares only the same adversary IDs
  detected or confirmed by both layouts. The shared target count is displayed.
  Missing events are never converted into zero-second detections.
- **Mean warning:** the project's existing per-drone
  `max(0, protected-zone entry - first detection)` in seconds, averaged over every
  drone, with missed drones contributing zero. This is not silently changed to
  confirmation-based warning.
- **Sensors / cost:** actual accepted deployments and native catalogue units.

Differences are **RL minus common sense**. Negative timing differences mean RL
was earlier; positive warning differences mean RL provided more advance notice.
When neither side shares any detected or confirmed targets, paired timing is
unavailable. Counts and warning still expose misses, so an early detection on one
rare target cannot be mistaken for good overall performance. No interception is
simulated.

The shipped episodes show the baseline detecting all 60 drones while the transferred
RL policies detect between 0 and 6. All predefined cases are retained. The current
model mismatch described above is material to interpreting that result; three
illustrative scenarios do not establish general performance.

## Paired evidence and reproducibility

`RL/BlueTeam/Results/native-placement-comparison/` contains:

- `bundle.json.gz`: all nine pairs, layouts, trajectories, public contexts, native
  target timing evidence, and measured final metrics.
- `manifest.json`: bundle SHA-256, size, case count and protocol.
- `protocol.json`: preset episode seeds, policy checkpoint hashes, source hashes,
  native module hash, launch configuration and evaluation limitations.

`RL/BlueTeam/Results/workbench-placement-comparison/` contains the corresponding
bundle, manifest and protocol for the measured five-directional-sensor option.
Its protocol records that the archived layouts were replayed without retraining.

The three episode seeds were selected before outcomes were known. All three
published RL policies and all three cases are retained. The baseline layout is
checked to remain unchanged. Native drone trajectories and sample times must
match exactly within every pair before the capture is published or served.
Native sensing uses indexed draws keyed by episode seed, drone ID, site ID,
sensing look and salt; sensor order and request batching do not shift the stream.

The initial capture used the compiled Training workbench checkout with optional
training configuration unused. Its sensing and movement implementations match
the merged sensor model; the protocol records actual native source and binary
hashes separately from the Python checkout. The stored data is new native
comparison evidence, not a replacement for historical published benchmarks.

To regenerate, build the current editor and launch the ordinary native scene on
an isolated loopback port (do not use the warning/demo approach overrides, which
change the frozen checkpoints' required temporal configuration):

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_live.ps1 -Port 8767
# Then, from RL/BlueTeam/Python, use new output directories:
python build_native_comparison.py --port 8767 --native-project D:\Path\To\IstanaOpen --output D:\Path\To\new-comparison --raw-output D:\Path\To\new-raw-captures
```

The capture refuses to overwrite output directories. It runs all 12 native
simulations (three policies plus the fixed baseline for each of three episodes).
Only after exact pairing checks pass does it write the portable bundle. No new
policy is trained or selected by this command. Serve a replacement bundle only
after reviewing the results and updating its manifest together.

To regenerate the five-sensor workbench evidence, launch the explicit workbench
contract and write to new output directories for review:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_live.ps1 -DelayedDetectionDemo -TrainingWorkbench
python .\RL\BlueTeam\Python\build_workbench_comparison.py --port 8765 --output D:\Path\To\new-workbench-comparison --raw-output D:\Path\To\new-workbench-raw
```

## Verification

From `RL/BlueTeam/Python`:

```sh
python -m pytest tests/test_native_comparison.py tests/test_native_comparison_capture.py tests/test_console_comparison.py tests/test_directional_common_sense.py tests/test_common_sense.py tests/test_simulation_console.py tests/test_istana_live.py -q
```

Coverage includes fixed baseline legality and orientation, public-only planning,
paired target timing, missing detections, exact native trajectories, source
artifact integrity, invalid/manual API inputs and preservation of Live state.
Browser verification covers policy and episode changes, synchronized playback,
result interpretation, responsive layout and historical replay selection.
