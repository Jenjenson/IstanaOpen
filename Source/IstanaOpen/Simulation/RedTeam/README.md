# Red Team Manager

`ARedTeamManager` is the implemented red-team scene spawner and shared-objective
controller. It creates multiple visible drone swarms around one target and uses the
shared movement solver to navigate through level geometry.

## Set up in Unreal

1. Build the editor with UE 5.5.4 after pulling native changes, then reopen the project.
2. Place **Red Team Manager** in your level. Its own location is not the spawn anchor.
3. Place a **Target Point** or `BP_SwarmObjective` and assign it to **Objective Target**.
4. Configure **Red Team > Spawning** using the settings below. Leave **Auto Initialize**,
   **Auto Advance**, **Follow Objective**, and **Spawn Visuals** enabled for normal Play.
5. Optionally assign a movement preset. It overrides the inherited inline **Settings**.
6. Press Play. Read **Spawn Status**, **Objective Status**, and **Get Group Statuses**
   when diagnosing a swarm that cannot spawn or reach the exact objective.

The inherited manual **Swarms** array and **Spawn Relative To Manager** option are not
used by this spawner. Use `AIstanaSwarmManager` instead for manually placed spawn groups.
No Blueprint subclass is required, although the native actor can be extended in Blueprint.

## Configuration

| Setting | Default | Meaning |
| --- | ---: | --- |
| Number Of Swarms | 3 | Number of independently identified groups |
| Drones Per Swarm | 12 | Members of each group |
| Min Spawn Radius Cm | 3000 | Minimum horizontal objective-to-group-center distance |
| Max Spawn Radius Cm | 6000 | Maximum horizontal objective-to-group-center distance |
| Swarm Spread Radius Cm | 400 | Sphere radius used to place members around each center |
| Spawn Height Offset Cm | 0 | Center height relative to the objective's Z coordinate |
| Max Spawn Layout Attempts | 16 | Maximum layout retries |
| Seed (inherited) | 12345 | Repeatable group layout and member placement |

Distances use centimeters: the defaults create 36 drones in three groups centered
30–60 meters from the objective, each with a 4-meter spread radius. Min/max radii
apply to centers, not every member. Set min=max for a fixed-distance ring. Height is
relative to the marker, not terrain-following; a ground-level marker may require a
positive spawn height offset so the spread sphere fits in free air.

Total population must fit the active movement settings' **Max Drones** (default 128,
hard cap 256). Multiple drones need a positive spread radius and enough space for
minimum spacing. One randomized angular sector per group distributes centers around
the target. Overlapping spawn regions and collision-invalid drone positions are rejected;
layouts are retried up to the configured limit. Exhausted retries preserve an existing run.

## Objective following and collision

All groups share the same objective reference. Moving it by more than 1 cm retargets
all groups at the next simulation step without respawning. Clearing or removing it
brakes all groups. Spawning and navigation are separate: a successful Spawn Status
confirms placement, not that every drone can reach its final destination.

The marker may touch or lie inside level geometry. Objective following accepts finite
coordinates and enables partial paths for each drone. Drones try to route around
obstacles and approach reachable space near blocked destinations. A blocked member
does not reject the objective for all groups. The original target remains assigned.

`bHasPartialPath` reports an incomplete approach. `bNavigationBlocked` reports members
without a usable path. `bRouteCompleted` is true only when all active members have
non-partial paths and are within Arrival Radius Cm of their formation destinations;
it does not require zero velocity or all drones occupying the marker's exact position.
Unavailable paths retry after 20 simulation steps. Once a partial endpoint is approached,
subsequent partial searches are throttled to 100 steps (5 seconds at the default timestep).
Removing an obstruction can therefore resume movement without changing the marker.

There are no arena min/max boundaries or predefined obstacle lists. **Use World
Collision** defaults on, querying objects that block **Visibility** (or the selected
Obstacle Trace Channel). **Trace Complex Obstacles** defaults on for triangle collision.
Collision-disabled objects are ignored. Partial paths retain collision sweeps; they do
not allow drones to fly through walls. Navigation is an approximate, bounded search,
so reaching every arbitrary destination is not guaranteed.

## Ownership and lifecycle

| Implementation | Responsibility |
| --- | --- |
| [RedTeamManager.h](RedTeamManager.h) / [RedTeamManager.cpp](RedTeamManager.cpp) | Spawn settings, seeded layouts, placement retries and spawn status |
| [AIstanaSwarmManager](../Swarm/IstanaSwarmManager.h) | World collision queries, shared objective, fixed-step clock, commands and visual lifecycle |
| [FIstanaSwarmSimulation](../Swarm/IstanaSwarmSimulation.h) | Group/drone state, boid movement, per-drone pathfinding and diagnostics |
| `AIstanaDroneVisual` | Visible mesh and orientation; no independent movement policy |

One Red Team Manager owns all its groups through one solver and clock, allowing
cross-group separation from the same previous-state snapshot. It does not spawn one
manager per group. Drone IDs are unique within the run; group IDs start at zero.

**Spawn Swarms**, **Reset Simulation**, and **Restart Demo** replace the whole run around
the objective's current location. A successful reset creates a new run ID, clears
metrics and replaces visuals. The same seed, target and collision state reproduce
placement. Spawning enables Follow Objective; subsequent stepping issues the objective
commands. **Spawned Groups** exposes accepted world-space configurations. Removing the
manager destroys its visuals. There is no mid-run append-group or spawn-one-drone API.

For external stepping, disable Auto Initialize and Auto Advance before BeginPlay.
After spawning, disable Follow Objective if another controller will supply commands.
Use the inherited command and snapshot APIs described in the [swarm guide](../../../../Docs/SWARM_SIMULATION.md).
Strict scripted waypoints validate complete routes by default; `bAllowPartialPath`
opts into the approach behavior used automatically by objective following.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No drones spawn | Objective assignment, total population, spread/spacing, free spawn height and Spawn Status |
| Inline tuning has no effect | An assigned movement preset overrides Settings |
| Drones hover near a marker inside a mesh | Partial approach is expected; inspect group status and collision geometry |
| Drones ignore an obstacle | Query collision must be enabled and block the selected trace channel |
| Long pause while finding a route | Search is synchronous; review population, cell size, clearance and node budget |
| Groups crowd together at the target | Inter-drone separation is soft steering; it is not a hard collision guarantee |

## Validation and scope

The verified 12 September 2026 run passed all **13 `Istana.Simulation` tests**, including
red-team placement, visible population, deterministic reset, shared-target arrival,
retargeting without respawn, invalid-setup preservation, target removal and cleanup.
Related tests cover floor/embedded objectives, partial approach with collision retained,
resuming after an obstruction moves, and movement beyond the old arena bounds.
Editor and game Development builds passed. See [validation](../../../../Docs/VALIDATION.md#swarm-and-red-team-validation-12-september-2026)
and the [test source](../Tests/IstanaSwarmTests.cpp) for scope and reproduction.

This is a kinematic scene spawner and scripted objective follower. It does not implement
motor dynamics, sensors, a learned policy, training, rewards or an episode coordinator.
The shared `IIstanaPolicyInterface` and schema-1 no-op action payload are unchanged.
