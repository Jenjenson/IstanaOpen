# Swarm optimization and red-team RL placement plan

Implementation is tracked in [results and limitations](SWARM_OPTIMIZATION_RESULTS.md).
The [agent guide](RED_TEAM_AGENT.md) documents the delivered placement API and bridge.
The text below retains the original design and acceptance targets; it is not a claim
that every performance target or optional architecture was achieved.

Status: proposed implementation plan, 14 September 2026. Source inspected at commit
`12870cc` (`feat(red-team): implement configurable multi-swarm objective following`).
The working tree already contains a modified `Content/Maps/Istana.umap`; preserve it.
This review changes documentation only. It does not implement optimizations or an RL
adapter, and it does not claim to have profiled the reported lag.

## Goal and preservation requirements

Remove the lag reported at five or more swarms while preserving current simulation
functionality, numerical behavior, collision accuracy and swarm dynamics. Add a typed
interface through which an RL agent chooses the actual placement of each swarm.

Keep these invariants in the default optimized implementation:

- Fixed 0.05-second timestep by default; identical acceleration, speed, turn limits,
  steering equations, separation/alignment/cohesion weights and formation offsets.
- All neighbor calculations read the previous snapshot. Separation spans groups;
  alignment and cohesion remain group-local. Diagnostics use the committed next state.
- Seeded placement, ID ordering, command sequencing, transactional failure and reset.
- Current collision channel, complex-trace setting, radii, clearance, endpoint rules,
  final body sweeps, and static/dynamic/streamed level-collision behavior.
- Complete and partial routes, shared-objective retargeting, removal/Stop behavior,
  completion criteria and the existing simulation-step retry schedule.
- No artificial arena boundary, predefined obstacle list or population reduction.
- Current public Blueprint behavior, visual appearance and visual lifecycle by default.

The existing model has soft inter-drone avoidance and can overlap near a crowded
objective. Preserving it does not mean introducing a physical collision guarantee.
The existing 128 default / 256 hard population cap remains until separately justified.

## Findings from the implementation

| Priority | Current code | Cost and interpretation |
| --- | --- | --- |
| 1 | `FIstanaSwarmSimulation::FindPath`, `SubmitCommand`, `Step` | Each drone runs its own synchronous A* search, including on objective assignment and retries. Up to 12000 discovered nodes and 26 neighbor directions per expansion; smoothing also issues traces. Likely hitch source, especially for partial/blocked targets. |
| 1 | `AIstanaSwarmManager::InitializeSimulation` collision callback | Every segment query checks a start overlap and, if clear, performs a sweep. Complex collision defaults on. The same start/radius is checked repeatedly across a node's outgoing edges. |
| 1 | `Step` retry scheduling | Newly commanded routes share `ExecutedSteps + 20`; blocked/partial retries use common 20/100-step intervals. Many drones can search in the same update. There is no global planning-work budget. |
| 2 | `Step` neighbor loop and diagnostics loop | Neighbor evaluation is O(N^2); diagnostics visits all unordered pairs and repeatedly searches group arrays. Costs grow with total drone count, not just swarm count. |
| 2 | `DrawDiagnostics`, enabled by default | Draws text/arrows per drone and wire spheres/bounds per group every rendered frame. Group status and bounds rescan state; status is obtained twice. |
| 2 | `AIstanaDroneVisual`, `RefreshVisuals` | One actor with three mesh components per drone. All visuals refresh after every substep, potentially eight times in a catch-up frame. Rendering cost must be measured separately from solver cost. |
| 3 | `CommandAllGroups`, `SubmitCommand` | Copies the full solver for an atomic group update, then copies all navigation routes again for each group's command. Long paths magnify allocation/copy cost. |
| 3 | `RedTeamManager::ResetSimulation` | Up to 16 whole-layout attempts; initialization can make up to 1000 spawn attempts per drone. Mainly spawn/reset cost, not the main steady-state loop. |

At the default 12 drones per swarm, five swarms mean 60 drones: 3540 directed neighbor
visits plus 1770 diagnostic pairs per step, about 106200 combined visits per simulated
second at 20 Hz. Three swarms perform 1890 combined visits per step, so this portion
increases about 2.8 times from three to five swarms. These counts are derived from loops,
not measured timings. Actual user counts/settings still need recording.

A fully expanded 12000-node search can execute roughly 312000 neighbor-loop iterations
before goal checks and smoothing; many skip tracing. A synchronized batch of 60 searches
can therefore be much more expensive than the boid loops. There is no explicit
five-swarm threshold in the code. A rendering bottleneck or dense-mesh query cost could
also dominate; profiling must establish the attribution.

The current Population test runs 128 drones for 100 steps without a world-collision
callback, actors or debug rendering. Its successful timing does not establish real-map
performance. The 13 existing functional tests are a regression starting point, not a
performance baseline for the user's scene.

## Phase 1 — Measure and capture a reference

Add timing scopes and counters around spawning, objective-command staging, path search,
collision overlap/sweep, smoothing, neighbors, diagnostics, visual updates and debug draw.
Capture CPU game/render thread and GPU timings with Unreal's profiling tools, allocations,
query counts and search queue sizes. Separate first spawn, retarget, steady flight and
arrival/partial retries. Record actual group/member counts and effective preset settings.

Benchmark the affected map at 1, 3, 5, 8 and 10 swarms with 12 members, plus total counts
128 and 256 where configuration allows. At a fixed total count, compare one group versus
many groups to distinguish group overhead from population overhead. Also test open space,
a wall detour, floor/embedded objectives, an unreachable target, moving obstacles and
streamed geometry changes. Use at least three fixed seeds and equal simulated durations,
including multiple 100-step retry cycles.

For each case capture:

- Frame and subsystem p50/p95/p99/max, simulation-step time, search/retarget latency.
- Nodes expanded, overlap/sweep calls, collision-cache hit rates, neighbor candidate counts.
- CPU/GPU separation, allocations, draw calls, resident memory and pending-job high water.
- ExecutedSteps, simulated versus wall time, DroppedWallSeconds, arrivals, partial/blocked
  flags, positions/velocities, path lengths, overlap/spacing counts and emergency stops.

Compare debug on/off and visuals on/off as diagnostic experiments, restoring settings
afterward. Headless runs must retain world collision. Measure a no-swarm rendering baseline
at the same view/resolution so terrain/building rendering is not blamed on the solver.
Save replay inputs and reference outputs before optimizing, with map/collision revision,
compiler/build settings, seeds, commands, objective motion and effective configuration.
Do not close/edit the user's level or change its presets just to collect these traces.

Deliverable: benchmark runner, reference replay fixtures and a before/after report template.

## Phase 2 — Reduce duplicate work without changing decisions

Implement small, independently validated changes, prioritized by Phase 1 timings:

1. Cache group membership/index lookups after initialization. Keep each group's member
   order identical to the current state array; compute reusable group aggregates once per
   applicable snapshot. Do not reuse a pre-step centroid for post-step status.
2. Reuse state, navigation and scratch buffers; reserve bounded capacity. Avoid rebuilding
   heap/hash allocations for every path. Preserve queue tie-breaking, neighbor enumeration,
   discovered-node accounting and the path-relative grid origin exactly.
3. Stage only affected groups' route deltas and commit atomically after all required
   validation succeeds. Preserve strict multi-waypoint validation and partial-command
   acceptance. Failed commands must leave state, command sequence and run identity intact.
4. Cache identical overlap and directed segment results within a single stable search/query
   batch. Key by exact endpoints, radius and query configuration; do not quantize coordinates
   or assume reversed sweeps are interchangeable. A start-overlap cache can remove repeated
   identical overlap tests for the 26 edges leaving one node.
5. Cache group debug summaries and format unchanged labels only when necessary. Keep full
   debug available; any display-only refresh throttling must not throttle diagnostics.

Cross-frame collision caches require explicit invalidation on geometry movement, collision
setting changes, actor lifecycle and streaming. Until a reliable revision system exists,
keep caches local to an immutable query batch. Never reuse a cached clear result for the
final movement sweep across an uncertain geometry change. Do not remove overlap checks or
switch complex traces off as a shortcut.

Acceptance: identical reference states, paths, flags, metrics and command outcomes on the
same platform/build for equivalent collision snapshots, with lower measured query/allocation
cost. If exact equality changes, investigate operation ordering before allowing tolerance.

## Phase 3 — Exact spatial neighbor lookup

Build a sparse 3D spatial hash from the previous state snapshot for boid neighbor queries.
Return all candidates within the maximum applicable radius, then apply the existing exact
distance comparisons and equations. Sort candidates into original state/DroneId order
before accumulation so floating-point summation and steering remain reproducible.

Build/update a separate post-step index for diagnostics. Visit each potentially relevant
unordered pair once in stable order, using max(group A spacing, group B spacing, drone
diameter) to avoid omitting asymmetric-spacing and overlap counts. Maintain exactly the
same pair-step metrics; do not disable diagnostics for apparent speed gains.

Use sparse cells with sufficient coordinate range for negative and distant positions.
This index is an acceleration structure, not an arena or a cap on neighbors. Dense swarms
may still approach O(N^2), which is unavoidable when all drones really interact.

Acceptance: a brute-force reference comparison covering coincident points, cell boundaries,
multiple groups, inactive drones, changed spacing and distant coordinates; unchanged seeded
trajectories and diagnostics. Parallel per-drone calculations are optional only after this
passes: no actor/world calls from workers, immutable inputs, disjoint output slots and
stable reductions. Move pure arithmetic, not live Unreal collision queries, to workers.

## Phase 4 — Navigation scaling with explicit equivalence gates

First optimize exact search work: reuse scratch storage, memoize repeat queries, reduce
avoidable path-copy/removal work and cache identical complete requests only when their
inputs and geometry revision truly match. Preserve per-drone formation destinations.
Different starts currently produce different grids; shared goals alone do not justify
sharing a route or occupancy answer.

If profiling still shows large search spikes, prototype resumable planner jobs with:

- Immutable start/goal/configuration, run ID, request generation and collision revision.
- Stable priority by request simulation step, group ID and drone ID.
- Deterministic expansion-count budgets, retained heap/search state and bounded memory.
- Cancellation on reset, changed target, destroyed owner or invalidated geometry.
- A strict commit barrier: route results become visible at the same logical simulation
  step as in the reference, before the relevant motion update.

Important tradeoff: spreading live collision searches over frames while letting drones or
obstacles advance changes route decisions and timing. Freezing only the swarm is insufficient
if world geometry keeps moving. Exact equivalence requires an authoritative collision
snapshot or synchronized world-step control. If neither is available, finish required work
synchronously and use the safe optimizations above; do not silently relax the requirement.

For deterministic training, process planning jobs at a simulation-step barrier and advance
only after all due results are ready. Wall-clock cost affects throughput, not episode time.
For interactive use, a separate best-effort scheduler that delays replans or continues old
paths would change behavior. It is outside the default preservation scope and must not be
presented as an equivalent optimization. A frame budget alone does not guarantee 60 FPS.

Do not substitute a leader-only path, shared flow field, coarse grid, lower search limit,
staggered retry steps or different avoidance weights in the compatibility path. Those may
be future experiments, but they alter current routes or behavior.

## Phase 5 — Presentation and headless throughput

Use profiling to decide whether rendering/actor overhead warrants changes. First batch
transform work and avoid redundant updates while preserving the observable contract:
`AdvanceOneStep` currently updates visuals before `OnSimulationStepped`. Callers may read
actor transforms in that event, so deferring all updates to the last frame substep is not
an unconditional drop-in change.

Prototype instanced body/arm rendering with stable DroneId-to-instance mapping and identical
mesh transforms, materials, visibility and heading/pitch rules. Keep the current actor path
as the compatibility default until actor ownership, selection, Blueprint references,
per-step transforms, reset and cleanup are supported and tested. Instancing must not change
solver state or collision queries. Do not claim zero actor overhead while retaining actors.

Training mode may suppress visuals and debug using existing switches; it must still load
and query the same level collision. Full diagnostics stay available in training outputs.

## RL placement interface — required design

### Ownership and scope

Add an explicit placement source: `SeededLayout` (current default) or `AgentPlacement`.
The agent decides each swarm's actual center. Merely changing Seed or Min/MaxSpawnRadius
is not sufficient. V1 decisions occur once during episode setup; existing movement,
formation generation and shared-objective following continue after placement. Mid-episode
spawning, relocating live drones, reward learning and flight-control policies are separate
features, not implicit meanings of ResetSimulation.

Retain the current seeded generator unchanged for ordinary users. In agent mode,
BeginPlay/reset enters AwaitingPlacement without invoking it. No automatic respawn or
random fallback is allowed to replace an agent decision. Refactor reset so it cannot
accidentally re-enable a competing spawn source.

### Proposed typed contract (new, not implemented)

Create `RedTeamPlacementTypes.h`, `RedTeamPlacementInterface.h/.cpp` and an adapter
owned by RedTeamManager, with Blueprint-accessible value types and C++ interfaces.
Keep the shared schema-1 `IIstanaPolicyInterface` and no-op `FIstanaPolicyActionBatch`
unchanged; they cannot carry placement actions today. Future general-policy integration
must use an explicit versioned translator rather than reinterpret those payloads.

| Type | Required contents |
| --- | --- |
| `FIstanaRedTeamPlacementContext` | SchemaVersion, RunId allocated before action, decision step/phase, objective ID/position/revision, world/collision revision, deterministic member seed, expected group IDs/counts, member counts, spread and spacing, configured placement constraints |
| `FIstanaRedTeamPlacementAction` | SchemaVersion, RunId, monotonic request/sequence ID, context revision, array of `{GroupId, SpawnCenterWorldCm}` |
| `FIstanaRedTeamPlacementResult` | Accepted/rejected/pending status, matching IDs, field-level rejection reasons, accepted centers/configurations, assigned drone IDs and actual seeded member positions for audit |
| `FIstanaRedTeamStepResult` | RunId, exact completed step, approved observations, group navigation status, diagnostics, terminated/truncated flags and reason; optional versioned reward output from a separate evaluator |

For V1, group count, drones per group and spread remain environment configuration; the
action supplies every group's center. Extend the action schema explicitly if the agent
should later choose allocation/counts. Positions are world-space centimeters with Z up.
An external model may use normalized objective-relative coordinates, but its adapter must
publish the invertible transform, bounds and frame revision and record decoded world centers.
No arbitrary global movement bounds are introduced by action normalization. Optional scenario
placement limits apply to setup only and are visible to the agent in the context.

Define provider calls equivalent to ResetPlacementPolicy(Context) and RequestPlacement(Context).
Blueprint/local providers may respond synchronously; external inference responds asynchronously
through the same validated action-submission API. Proposed manager/adapter entry points:

- `BeginPlacementEpisode(config, seed)` -> authoritative context, no spawn yet.
- `ValidatePlacement(action)` -> structured result with no mutation.
- `SubmitPlacement(action)` -> validated atomic commit or structured rejection.
- `AdvanceEpisode(numFixedSteps)` -> result after exactly those steps and due navigation work.
- `GetPlacementContext`, `GetEpisodeObservation`, `CancelEpisode` for lifecycle/control.

Implement `SubmitPlacement` through the existing initializer only after validation, passing
its preallocated RunId rather than letting reset generate another GUID. Keep validation and
commit on the game thread against the same revision; revalidate after asynchronous work.

### Validation and randomization rules

Validate schema/run/request/context/phase, finite coordinates, complete unique group IDs,
population, configured placement constraints, spawn-region separation and per-member collision/
spacing. Reuse existing geometric rules rather than creating different RL physics.
The objective itself may be obstructed; preserve current partial-objective acceptance.

Accepted swarm centers must equal the decoded agent action. Seeded member sampling within
each requested sphere is allowed and recorded, using the current initializer and its seed.
If members cannot fit after its bounded attempts, reject with a reason; do not shift centers,
reroll another layout, clip an action silently or invoke the 16-layout random retry loop.
Failed validation must not mutate the previous committed run, IDs, metrics or visuals.

Repeated delivery of the same request ID and identical payload returns its recorded result
without a second spawn. A reused ID with a different payload, stale episode or changed context
is rejected. The adapter owns request ordering, not packet arrival time. After acceptance,
placement is locked for that episode; a new setup decision requires an explicit episode reset.

### Episode and training lifecycle

Use Idle -> AwaitingPlacement -> Validating -> Running -> Completed/Cancelled. Rejection
returns to AwaitingPlacement with reasons; a configurable invalid-action budget can terminate
setup without random fallback. If an old run is retained during proposed replacement, it must
remain isolated and not advance as though it belongs to the new RunId.

Disable auto-initialization, auto-advance and demo keyboard reset in agent mode. After a valid
placement, explicitly retain shared-objective following; it controls navigation, not spawn
positions. The external runner requests exact fixed steps with a planning barrier and no
wall-clock accumulator or dropped simulation steps. Network/inference waits do not advance
episode time. Define termination by configured episode duration or actual route-completion
rules; partial approach alone is not exact arrival. Target removal/cancellation is explicit.

An RL-facing observation is a versioned value snapshot, not unrestricted actor/world access.
Expose configured objective/placement information and allowed own-team states; keep evaluator
truth and future sensor-only observations separate. No sensor model or task reward exists in
the current code. Provide an evaluator hook and document an optional navigation smoke-test
reward separately from the future training objective; do not invent a sensor/evasion reward.

Deliver a local scripted provider first, proving that supplied centers are honored. Then add
a transport-neutral asynchronous bridge and a Python reset/step client for an external agent
(optional Gym-style wrapper). Unreal editor Python alone is not a packaged runtime interface.
Keep inference off the game thread, marshal actions onto it, cancel stale responses, and make
transport failures return explicit errors rather than silently changing actions. Log schema,
seed, accepted/rejected actions, revisions and step results for deterministic replay.

## Verification and acceptance gates

1. Run the existing 13 functional tests and preserve their assertions. Add old-versus-new
   solver replay tests comparing every step, IDs, commands, paths, partial status and metrics.
   Seek exact same-platform equality for compatibility phases; cross-platform floating-point
   differences require documented tight tolerances, not broad statistical similarity.
2. Test both pre/post-state spatial indexes against brute force, including dense all-neighbor
   scenes. Verify numerical limits, collision decisions and pair-step counts remain unchanged.
3. Test collision-cache invalidation for moved, added, removed, streamed and collision-toggled
   geometry; ensure no cached clear route bypasses the final body sweep.
4. Exercise retarget bursts, unreachable/embedded objectives, repeated retries, reset during
   planning, stale-job rejection and all supported clock/visual modes. Use larger multi-swarm
   collision scenes, not only the current geometry-free Population test.
5. RL tests: explicit centers retained; seeded mode unchanged; invalid/stale/duplicate actions;
   blocked member spawns with no reroll; atomic failure; one commit per episode; exact stepping;
   disconnect/timeout; reset cancellation; deterministic replay and allowed observation fields.
6. Repeat Phase 1 benchmarks on the same machine, map, settings and seeds. Proposed initial
   performance goal: at five default swarms, swarm CPU p95 <= 4 ms/frame and p99 <= 8 ms/frame
   in the interactive benchmark, with no swarm-attributed >50 ms spikes after warmup and no
   dropped simulation time in the sustained baseline. These are targets to calibrate, not
   measured promises. Report startup/retarget/retry spikes separately, including any that
   cannot meet a frame budget under strict equivalence. For 128/256 drones report the scaling
   curve and memory, without claiming a guaranteed frame rate. Training reports fixed-step
   throughput with identical collision/behavior and zero silently skipped steps.

If the world alone exceeds the frame budget, report that separately. FPS improvements from
less debug rendering or omitted visuals do not prove the simulation itself became faster.
No performance target permits reducing collision fidelity or changing the requested behavior.

## Delivery sequence

| Change set | Files/area | Completion criterion |
| --- | --- | --- |
| 1. Baseline instrumentation | Swarm solver/manager, new benchmark runner and replay fixtures | Reproducible attribution of five-swarm lag |
| 2. Duplicate work and allocation removal | SwarmSimulation, SwarmManager collision/query staging | Exact reference parity; query/copy reduction measured |
| 3. Spatial indexing | New spatial index helper, solver and tests | Brute-force parity and improved sparse scaling |
| 4. RL placement contract and local provider | RedTeam placement types/interface, RedTeamManager | Agent centers honored; legacy seeded mode unchanged |
| 5. Optional resumable planning | Planner helper and clock adapter | Same logical-step decisions with valid collision snapshots; otherwise retain synchronous compatibility mode |
| 6. Presentation optimization | Drone visual backend, manager/debug | Visual/API compatibility and measured render benefit |
| 7. External RL bridge and final validation | Runtime adapter, Python client, integration tests/docs | Exact reset/step/action replay and full before/after report |

RL contract work depends on the baseline state/reset tests, but not on a particular rendering
backend. Start with measurement and exact duplicate-work reduction rather than changing
flocking parameters, disabling collision or adopting asynchronous planning indiscriminately.

## Inspected source

- [Swarm solver](../Source/IstanaOpen/Simulation/Swarm/IstanaSwarmSimulation.cpp)
- [Actor adapter and collision callback](../Source/IstanaOpen/Simulation/Swarm/IstanaSwarmManager.cpp)
- [Manager defaults and public API](../Source/IstanaOpen/Simulation/Swarm/IstanaSwarmManager.h)
- [Red-team placement generator](../Source/IstanaOpen/Simulation/RedTeam/RedTeamManager.cpp)
- [Shared no-op policy interface](../Source/IstanaOpen/Simulation/IstanaPolicyInterface.h)
- [Existing regression tests](../Source/IstanaOpen/Simulation/Tests/IstanaSwarmTests.cpp)
