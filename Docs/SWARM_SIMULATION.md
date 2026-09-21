# Drone swarm simulation

The project provides seeded group spawning, boid movement, 3D pathfinding against
level collision, drone visuals and a shared objective marker. It has no artificial
arena min/max boundary and no predefined obstacle list. Movement uses a deterministic,
Mavic 3 Enterprise-inspired flight envelope. It is still a constrained kinematic model:
it does not simulate individual motors, propellers, battery discharge, aerodynamic
downwash, turbulence, sensors, rewards or learning.

## Mavic 3 Enterprise flight envelope

New `FIstanaSwarmSettings` instances use the DJI Mavic 3 Enterprise normal-mode
published limits below. Unreal uses centimetres, so the source values are converted
from metres per second. These are limits, not a claim that every real aircraft reaches
them in every payload, altitude, temperature or wind condition. Source: [DJI Mavic 3
Enterprise specifications](https://enterprise.dji.com/mavic-3-enterprise/specs).

| Setting | Default | Basis |
| --- | ---: | --- |
| Drone Radius Cm | 25 cm | Conservative spherical collision proxy for the 347.5 x 283 mm unfolded airframe |
| Max Speed Cm Per Second | 1500 (15 m/s) | Published normal-mode horizontal speed |
| Cruise Speed Cm Per Second | 900 (9 m/s) | Speed used by DJI for its windless endurance measurement |
| Max Ascent / Descent Speed | 600 / 600 (6 m/s) | Published normal-mode limits |
| Max Turn Degrees Per Second | 200 degrees/s | Published maximum angular velocity |
| Max Tilt Degrees | 30 degrees | Published normal-mode attitude limit |
| Max Acceleration | 566 cm/s2 | Derived as `g * tan(30 degrees)`; not a published DJI acceleration value |
| Max Jerk | 1200 cm/s3 | Simulation controller tuning; DJI does not publish a jerk limit |

The controller separates horizontal airspeed from vertical climb/descent limits.
Horizontal acceleration is constrained by the configured tilt angle, and acceleration
commands are jerk-limited to remove instantaneous changes. `Wind Velocity Cm Per Second`
is a steady world-space wind vector. The controller attempts to preserve its requested
ground track by subtracting wind before enforcing the airspeed envelope; sufficiently
strong wind therefore causes drift. DJI publishes 12 m/s as the aircraft's maximum wind
resistance, which is a reference operating limit rather than a hard wind clamp here.

Pitch and bank shown by `AIstanaDroneVisual` are derived from the solver's horizontal
acceleration and capped at the same tilt setting. Yaw follows horizontal travel and is
rate-limited in the solver. Visual attitude and rotor animation do not feed back into
movement or observations.

Movement presets and settings serialized before this change can retain their old
per-property values. Reset those fields to defaults, update the preset, or create a new
preset to use the values above. This avoids silently rewriting existing Unreal assets.

The dedicated `/Game/Simulation/Presets/DA_Mavic3E_NormalFlight` data asset is
checked in. The repository also includes `Tools/create_mavic_swarm_preset.py` to
recreate or refresh that asset without changing the older example preset or any
map. Close Unreal, build the editor module, then run:

```powershell
$swarmEngineRoot = 'C:\Program Files\Epic Games\UE_5.5'
& "$swarmEngineRoot\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" `
  "$PWD\IstanaOpen.uproject" -run=pythonscript `
  "-script=$PWD\Tools\create_mavic_swarm_preset.py" `
  -unattended -nop4 -nosound -nullrhi
```

In the level, select `RedTeamManager` and set **Swarm > Setup > Movement Preset** to
`DA_Mavic3E_NormalFlight`. A preset overrides the manager's inline movement settings.

## One manually placed swarm manager

1. Place `IstanaSwarmManager` in free air at the desired spawn location.
2. Leave **Spawn Relative To Manager** enabled and set a group's **Spawn Origin Cm**
   to `(0,0,0)` to center its spawn sphere on the manager. Nonzero offsets rotate with
   the actor; actor scale does not change distances. Existing saved offsets are retained.
3. Configure the **Swarms** array: group IDs, drone counts and spawn radius. Multiple
   drones need a nonzero radius and enough room for minimum spacing.
4. Place a Target Point or `BP_SwarmObjective`, assign **Objective Target**, and keep
   **Follow Objective**, **Auto Initialize** and **Auto Advance** enabled.
5. Play. Every group follows that objective and holds around it, retaining individual
   formation offsets. The target can be anywhere in the loaded world, including on or inside geometry.

Moving a manager after spawning does not teleport drones; reset/restart applies its
new placement. The explicit `InitializeSimulation` API and submitted waypoints always
use world coordinates. A movement preset overrides the manager's inline movement settings.
Old serialized arena bounds and sphere-obstacle settings are ignored and no longer exposed.

## Multiple groups with Red Team Manager

Place **Red Team Manager** (`ARedTeamManager`) and assign its inherited **Objective
Target**. This actor generates its own groups; do not manually populate its inherited
Swarms array. Its own location does not determine spawning: spawn centers are generated
around the objective's position when spawning/resetting. See the
[Red Team Manager README](../Source/IstanaOpen/Simulation/RedTeam/README.md) for its
setup checklist, lifecycle and troubleshooting.

| Setting | Meaning | Default |
| --- | --- | --- |
| Number Of Swarms | Number of independently identified groups | 3 |
| Drones Per Swarm | Drones in each group | 12 |
| Min Spawn Radius Cm | Minimum horizontal distance from objective to group spawn center | 3000 |
| Max Spawn Radius Cm | Maximum horizontal distance from objective to group spawn center | 6000 |
| Swarm Spread Radius Cm | Radius of each group's drone spawn sphere | 400 |
| Spawn Height Offset Cm | Group-center height relative to objective | 0 |
| Max Spawn Layout Attempts | Bounded retries for obstructed/overlapping layouts | 16 |
| Seed (inherited) | Reproducible layout and drone placement | 12345 |

All distances are in centimeters: 3000 cm is 30 m. Set min=max for a fixed-distance
ring. One randomized angular sector per group distributes groups around the objective.
The radial limits apply to group centers; individual drones can extend beyond them by
the spread radius. Height is relative to the objective, not terrain-following: choose a
height and spread that fit free air. Layout generation rejects intersecting group spawn
regions and the solver rejects drone positions inside level collision or too close to
other drones. If attempts are exhausted, **Spawn Status** explains the failure and an
existing simulation is preserved.

All groups share the same target reference, movement settings, solver, clock and
cross-group separation. Each drone still gets a visible actor and a unique ID. Moving
the objective retargets all groups without respawning. Clearing/removing the objective
brakes all groups. **Spawn Swarms**, **Reset Simulation**, or **Restart Demo** regenerates
the layout around the objective's current location. The same seed and unchanged world
reproduce placement. **Spawned Groups** exposes accepted group IDs and spawn centers.

Total drones must fit the movement settings' **Max Drones** (default 128, hard cap 256).
Change the active movement preset's limit if necessary. Spawning is initial/reset setup;
there is no append-one-group operation during an episode. Removing the manager cleans
up its visuals.

## Collision and navigation

**Use World Collision** defaults on. Objects must have query collision enabled and
block **Obstacle Trace Channel** (Visibility by default). **Trace Complex Obstacles**
defaults on to query triangle collision, including scenery without simple hulls.
Collision-disabled meshes are ignored. The manager supplies the query callback to the
solver; there are no manually configured sphere obstacles or arena walls. Existing
visible obstacle meshes in a level still participate through their collision settings.

Every drone follows a 26-neighbor 3D A* route with visibility smoothing. Clear direct
routes skip search. Search coordinates are local to the route's starting point and
can extend in any direction; no fixed map boundary constrains movement or targets.

| Navigation setting | Default | Tradeoff |
| --- | --- | --- |
| Navigation Cell Size Cm | 200 | Smaller resolves narrower passages at higher cost |
| Navigation Clearance Cm | 80 | Extra clearance beyond drone radius along grid edges |
| Max Navigation Nodes | 12000 | Caps search work per path, not world dimensions |

Short endpoint connections may use body clearance when an endpoint is close to a
surface. Search failure means no path was found at the configured resolution/budget,
not proof that no geometric path exists. Search is synchronous; large populations and
complex blocked routes can cause hitches. Increasing node budgets increases work.

Objective following accepts any finite marker coordinates; collision at the marker or
at an individual formation destination does not reject the other groups. Each drone
tries a complete path first, then approaches a reachable point near the objective if
necessary. For obstructed endpoints, search may stop at a clear node within one cell
plus Arrival Radius; otherwise it uses the closest node explored within the search
budget. This is an approximate reachable approach, not a globally nearest-point guarantee.

The original objective stays assigned. `bHasPartialPath` identifies groups with an
incomplete approach; `bRouteCompleted` requires non-partial paths and every active
drone within Arrival Radius Cm of its formation destination. Newly submitted paths
become eligible for a retry after 20 steps; subsequent partial searches are throttled
to 100 steps (5 seconds at the default timestep) once their endpoint is approached.
Completely unavailable paths are retried every
20 steps. Removing an obstruction therefore allows an unchanged marker to be reached
without resetting or moving it. Other groups continue on their own paths.

Strict scripted waypoint commands retain full route validation by default. Callers can
set `bAllowPartialPath` to request the same approach behavior as objective following.
Malformed commands and nonfinite coordinates remain rejected. Collision sweeps remain
active during both modes. A moving obstacle can engulf a stationary drone; there is
no depenetration or motion prediction.

Navigation uses separate horizontal airspeed, climb/descent, acceleration, jerk, tilt
and yaw-rate limits, plus braking near path points. Separation repels neighbors, while alignment/cohesion consider group members. Drone-to-drone
avoidance is soft steering, not a rigid collision solver; crowding at a common objective
can cause overlaps. Arrival requires every active member within Arrival Radius Cm of
its own formation destination, not every drone at the exact marker or zero velocity.

## Runtime API and ownership

| Class/file | Role |
| --- | --- |
| `FIstanaSwarmSimulation` | States, deterministic movement, path planning, command validation |
| `AIstanaSwarmManager` | Actor adapter, clock, world collision queries, marker following, visuals |
| `AIstanaDroneVisual` | Mesh/orientation only; no independent simulation tick |
| `ARedTeamManager` | Generates multiple groups around the shared objective using the same adapter |
| `FIstanaSwarmSettings` / movement preset | Motion, boids, navigation and population settings |

Supported commands: Hold, FollowWaypoints, SetCruiseSpeed, SetSpacing and Stop.
Commands carry the active RunId, current DecisionStep, target GroupId and increasing
per-group SequenceNumber. Invalid commands do not consume a sequence or replace state.
Read `GetDroneStates`, `GetGroupStatuses` and `GetDiagnostics`; `OnSimulationStepped`
fires after state and visuals update. Reset replaces the whole run and generates a new
run ID. Run identity does not change the seeded positions.

For externally controlled stepping, disable both Auto Initialize and Auto Advance
before BeginPlay. Disable Follow Objective before supplying competing controller
commands. Call InitializeSimulation, SubmitCommand and AdvanceOneStep as needed.
Direct C++ solver callers may omit the collision callback for geometry-free tests.
The shared policy action contract remains no-op only; RedTeamManager is a scene spawner,
not a learned policy, reward system or episode coordinator.

The standalone clock defaults to 0.05 seconds per simulation step. Catch-up work is
limited by Max Steps Per Frame; discarded time is recorded in DroppedWallSeconds.
Space pauses, R restarts and V toggles debug when demo keyboard input is enabled.
Debug shows group bounds, IDs, velocity arrows, targets, blocked groups and status;
group debug boxes are visual only and never constrain movement.

## Build and verify

Close Unreal, then build with `Tools/build.ps1 -Target All` using UE 5.5.4.
Run the `Istana.Simulation` automation group. It covers shared contracts, spawn/reset,
commands, steering/collision, wall detours, population, real static-mesh collision,
unrestricted coordinates, RedTeam spawning/shared-target/cleanup behavior, and partial
approach to floor/embedded objectives with resumption after an obstruction moves. The
Mavic flight-envelope test additionally checks wind-relative horizontal speed,
ascent/descent speed, tilt-derived acceleration and jerk constraints. Historical test
results are recorded in [validation](VALIDATION.md).

```powershell
# Set this to your UE 5.5 installation.
$swarmEngineRoot = 'C:\Program Files\Epic Games\UE_5.5'
& "$swarmEngineRoot\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" `
  "$PWD\IstanaOpen.uproject" -unattended -nop4 -nosound -nullrhi `
  '-ExecCmds=Automation RunTests Istana.Simulation' `
  '-TestExit=Automation Test Queue Empty' '-ReportExportPath=Saved/Automation/SwarmTests'
```

`Tools/create_swarm_demo.py` and `Tools/create_swarm_objective.py` create missing
example assets and smoke-test transient managers without saving existing levels.
Their transient tests disable level collision; the native world test checks actual
static-mesh collision after physics-scene initialization. Reports are written under
Saved/Automation. Automated checks do not substitute for manual graphics testing.

The example map retains the name `Content/Simulation/Maps/SyntheticSwarmArena`; this
name does not imply a simulation boundary. The architectural viewer still starts at
`Content/Maps/Istana`. Neither map is automatically edited by these code changes.

## Optimization and agent placement

The solver removes discarded A* edge queries, caches exact overlap results only within
one synchronous path search, reuses state storage, and stages only affected navigation
routes. Large populations use an exact spatial broad phase in original accumulation order;
small populations retain the faster direct scan. Work counters and Istana_* CPU trace
scopes expose planning, collision, visuals and debug costs.

Red-team Agent Placement accepts externally selected world-space centers with an explicit
fixed-step episode clock. See the [agent guide](RED_TEAM_AGENT.md),
[implementation results](SWARM_OPTIMIZATION_RESULTS.md), and original
[plan](SWARM_OPTIMIZATION_AND_RL_PLAN.md). Navigation remains synchronous; equivalent
asynchronous planning requires a synchronized collision snapshot.
