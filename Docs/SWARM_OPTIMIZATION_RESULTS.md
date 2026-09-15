# Swarm optimization results

Validated on 15 September 2026 with UE 5.5.4, Win64 Development.
See the [original plan](SWARM_OPTIMIZATION_AND_RL_PLAN.md) and
[agent integration guide](RED_TEAM_AGENT.md).

## Implementation

- A* skips collision queries for edges that cannot improve a discovered node or add a
  node after the unchanged search budget is exhausted. Search order, goal selection,
  per-drone routes, smoothing and retry intervals are retained.
- Exact start/radius overlap results are cached only inside one synchronous search.
  Later searches and final movement sweeps query live collision again. The original
  complex-capable SweepSingleByChannel query is retained.
- Cached group membership/indexes, reusable state buffers and affected-route transaction
  staging reduce repeated work. Whole-objective command updates remain atomic.
- An exact spatial broad phase is used for populations of at least 192 drones. Candidates
  are sorted into original state order before arithmetic. Small/dense populations use
  direct scans, which benchmarked faster there. There are no neighbor caps.
- Existing visual actors and ownership are retained. Moving visuals update position and
  rotation together; stopped visuals retain rotation. Debug labels/colors are cached and
  each debug pass uses one status snapshot. Visuals are current before step delegates fire.
- AgentPlacement provides explicit centers, atomic validation/commit, a separate versioned
  policy contract, deterministic member spawning, isolated observations, fixed-step episodes,
  idempotent requests, optional evaluator hook, local provider, runtime socket bridge and
  external Python client. SeededLayout remains the default.
- CPU trace scopes, CSV timings, work counters and reproducible benchmark tools are included.

No arena boundary or predefined obstacle was reintroduced. Motion limits, boid weights,
formation offsets, collision fidelity, partial approaches and default presentation options
were not reduced to obtain better timings.

## Verified behavior

Both Editor and Game Development targets built successfully. The final
**Istana.Simulation suite passed 17 tests with 0 failures**, retaining all original
13 tests. Reports are under Saved/Automation/SwarmFinalRegression.

Exact reference replay compares every step's states, statuses, physical diagnostics and
internal route fingerprints. It covers 12, 36, 60, 96, 120, 128 and 256 drones, three seeds,
open/floor scenes, retargeting and collision removal: 42 cases of 220 steps each. Both
implementations use the same 300-node limit in this lightweight matrix. Actual-map tests
use the saved level's movement settings, including its full navigation budget.

MSVC fast-math initially introduced a one-bit velocity difference between equivalent loop
layouts. Both solver and reference now use controlled precise floating-point evaluation.
Equality is exact under that common build mode; bitwise equivalence with the previously
compiled fast-math binary or other platforms is not claimed. Tolerances were not widened.

New placement tests include blocked solid-volume spawning, preservation of the previous
run on failure, fresh collision validation, stale/duplicate actions, exact centers, local
provider submission, reset isolation and exact stepping. ExternalPythonRoundTrip launches
a separate Python process against a real loopback bridge and tests malformed versions,
reset/place/step, duplicate-step prevention, reconnects, cancellation and idle timeout.
Existing tests retain ground/embedded objectives, detours, moving obstacles, shared-target
arrival/retargeting, visual population and cleanup.

## Actual-level CPU results

The benchmark loads Istana without saving it and compares the reference and optimized
implementations against the same collision geometry and spawn layout. It uses NullRHI,
so these results measure CPU simulation/collision, not rendered FPS. Each case executes
240 fixed steps after initial path commands. Startup planning is timed separately.

Completed real-level coverage: **12 cases**, spanning 12, 36, 60 and 96 drones with seeds
7, 41 and 12345. These were collected during navigation optimization; the final small
membership/presentation changes were subsequently covered by exact regression replay.
The longer 120/128/256 actual-map cases were interrupted and remain unmeasured. The
lightweight 42-case solver matrix does include 128/256 drones. Do not infer their real-map
performance from the lightweight results.

For five swarms / 60 drones, across the three seeds:

| Metric | Reference | Optimized |
| --- | ---: | ---: |
| Median initial path-command time | 67.50 s | 15.91 s |
| Median of per-case step p95 | 2.356 ms | 2.353 ms |
| Worst sampled step | 3382 ms | 809 ms |

Initial planning fell approximately **76%**. Ordinary step time is similar because live
collision queries dominate this map. The seed-12345 case still has optimized p99 around
511 ms due to retries. **The no-spike performance target is not met**: initial planning and
some retries still visibly stall. Low p95 values must not be presented as eliminating lag.
Per-case percentile medians are not pooled-frame percentiles.

Raw snapshots are in [Benchmarks](Benchmarks/); regenerate local results under
Saved/Benchmarks and summarize them with Tools/summarize_swarm_benchmarks.py.
Graphics profiling is a separate check. The launcher completed offscreen 120-frame
smoke captures with zero and five swarms, including real rendering. Another editor was
open, so these are launcher/presentation checks, not clean GPU/FPS comparisons. No new
GPU/FPS improvement is claimed here.

## Retained limits and conditional decisions

Planning remains synchronous. The project lacks an authoritative collision snapshot or
whole-world logical-step barrier. Splitting live queries over frames could change paths
when geometry moves, violating compatibility. The agent clock does not silently skip
requested steps, but must wait for each synchronous operation to complete. Full-world
training also needs a coordinator to advance dynamic scenery/physics consistently.

An instanced visual backend and worker-thread physics queries are not enabled. Existing
actor-reference, ownership and per-step event behavior is preserved. Sensors, learned
policies, training rewards and a whole-world episode coordinator remain separate work.

Coverage does not imply exhaustive validation of streamed geometry, every imported mesh,
or long-duration interactive operation. The implementation is ready for review and use,
with the remaining planning stalls and profiling coverage explicitly recorded above.

## Reproduce

```powershell
./Tools/build.ps1 -Target All
python Tools/summarize_swarm_benchmarks.py
```

Run the unattended test command in the [swarm guide](SWARM_SIMULATION.md#build-and-verify).
For the expensive real-level benchmark, select
`Automation RunTests Istana.Performance.Swarm.LevelCollision` instead. Add
`-SwarmBenchmarkResume` to continue completed cases only with the same map/settings;
otherwise it starts a fresh CSV. Each completed case is saved independently.

For separate offscreen graphics captures:

```powershell
./Tools/profile_swarm.ps1 -Swarms 0
./Tools/profile_swarm.ps1 -Swarms 5
./Tools/profile_swarm.ps1 -Swarms 5 -NoDebug -NoVisuals
```

Close other running editor/game instances for meaningful timing. The launcher starts CSV
capture after RHI initialization to avoid UE 5.5's early-capture assertion. It saves a
separate log and CSV under Saved/Logs and Saved/Profiling/CSV, and never saves map assets.
These short captures include warmup: use longer runs and separate startup from steady
frames before judging FPS. IstanaSwarm CSV timings and Istana_* Insights scopes attribute
planning, collision, solver, presentation and debug costs.
