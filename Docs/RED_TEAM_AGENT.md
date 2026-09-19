# Red-team placement agent

For the bounded learned initial-placement policy, training command and frozen
policy comparison, see [Red initial-placement RL](RED_TEAM_RL.md).

**Seeded Layout** remains the default and preserves the original annular generator.
**Agent Placement** waits for externally selected world-space swarm centers. Placement
happens once per episode; individual drones retain the existing solver's formation,
collision avoidance, shared-objective following and partial approaches. No learned
policy is embedded in the native interface itself. The companion
[Red initial-placement RL](RED_TEAM_RL.md) package supplies external Python
training, learned/baseline policies and live evaluation while keeping this
transport contract policy-agnostic.

## Setup

1. Place a `RedTeamManager`, assign its Objective Target and configure population,
   spread, radius/height constraints and movement settings.
2. Set **Placement Source = Agent Placement**. BeginPlay disables automatic advancement
   and demo-keyboard reset, allocates a context and waits for placement.
3. For external Python, place `RedTeamAgentBridge`, assign the manager and enable
   **Start On Begin Play**. The default endpoint is `127.0.0.1:8765`.
4. Run PIE or a development game with these actors, then run:

   ```powershell
   python Tools/red_team_client.py --port 8765 --steps 100
   ```

Replace the example `scripted_centers(context)` function with agent inference. Rejected
centers are never replaced by a random layout. Members are sampled within the supplied
centers' spread using the advertised seed and original initializer; placement results
include actual member positions and accepted group configs.

The opt-in bridge is part of the runtime module and does not require Editor Python.
Inference runs in the external process. Nonblocking socket polling and all world queries
run on Unreal's game thread. Disconnect or idle timeout cancels the episode; normal
inference delay never advances simulation time.

## Native and Blueprint API

| API | Behavior |
| --- | --- |
| `BeginPlacementEpisode(seed, context, error)` | Allocates run ID/context before requesting placement; retains but freezes the previous run |
| `GetPlacementContext()` | Objective identity/position, revisions, movement/collision settings, seed, fixed step and placement constraints |
| `ValidatePlacement(action, error)` | Checks all centers and actual member placement against collision without committing |
| `SubmitPlacement(action)` | Atomically initializes all groups or preserves the previous simulation |
| `AdvanceEpisode(steps, observation, error)` | 1-1000 exact fixed steps; stops early only on evaluator termination/truncation |
| `GetEpisodeObservation()` | Current episode's own-team state, group statuses and diagnostics |
| `CancelEpisode(reason)` | Freezes the episode and marks it truncated |
| `NotifyPlacementWorldChanged()` | Environment/streaming placement-context revision notification |
| `EvaluateEpisode(observation)` | Optional Blueprint/C++ reward and termination hook |

`IRedTeamPlacementPolicy` provides ResetPlacementPolicy and RequestPlacement events.
A provider may submit immediately or asynchronously later. `URedTeamScriptedPlacementPolicy`
provides a manager reference and explicit centers as an example. The shared schema-1
`IIstanaPolicyInterface` no-op payload is unchanged; placement has a separate contract.

Version 1 group IDs are `0 .. groupCount-1`. Population and spread are environment choices.
Centers must satisfy the advertised horizontal annulus and objective-relative height
(0.001 cm validation tolerance), plus the original spawn-region separation rule. Coordinates
are never clipped or moved. Validation is not a reservation: actual member collision is
checked again at commit.

The environment calls NotifyPlacementWorldChanged for geometry/streaming context changes.
Objective identity/position and collision-option changes are detected directly. Movement
settings are frozen in the episode context. Live collision is always checked at commit;
world revision does not enable persistent navigation collision caching.

Placement actions contain schemaVersion=1, runId, revision, a monotonically increasing
nonnegative requestId, and all `{groupId, centerWorldCm}` entries. Duplicate/missing IDs,
stale context, nonfinite values, invalid constraints and blocked member placement return
errors. Identical retries return the stored result without respawn or RNG consumption;
a changed payload with the same ID is rejected. The manager retains up to 256 placement
requests per episode. Reset invalidates prior run IDs. Old retained state is excluded from
the new episode's observations. There is no mid-flight placement action.

Automatic ticks, AdvanceOneStep and RestartDemo cannot bypass the agent episode clock.
Do not mix inherited low-level initialization/command APIs with the agent episode API;
those remain available to trusted coordinators for seeded-mode compatibility.

The evaluator runs after each fixed step. By default bHasReward=false and neither terminal
flag is set. Define scenario reward/time-limit/termination in the evaluator. Observations
contain own-team simulation state, not sensor observations or hidden opponent truth.
Dynamic scenery still uses the host world's clock: a full-environment coordinator must
advance that scenery consistently for reproducible dynamic-world training.

## Wire protocol

One UTF-8 JSON object per line, maximum 64 KiB request. Send requests serially with an
increasing integer `id` (within JSON's exact 53-bit range) and `op`. Replies contain id,
ok and a result or explicit error.

| op | Additional input | Output |
| --- | --- | --- |
| reset | seed (int32) | context |
| context | none | context |
| place | action: schemaVersion, runId, revision, requestId, centers | result |
| step | runId, expectedStep, steps | observation |
| observe | none | observation |
| cancel | none | acknowledgment |

A center is `{"groupId":0,"centerWorldCm":{"x":4000,"y":0,"z":500}}`.
Reflected fields use lower camel case; boolean fields retain their b prefix. Use GUID
strings returned in context. expectedStep rejects stale state transitions. Exact retries
of the latest wire request return the cached response; conflicting or older wire IDs
are rejected. Keep one request outstanding. Transport IDs are scoped to each connection and restart at zero after reconnecting.
Disconnect cancels the episode, requiring a fresh episode reset before stepping again.

Accepted/rejected dispatches, context, seed, results and observations are logged under
Saved/RedTeamReplay/*.jsonl. Reproduction also requires the same map, geometry changes,
settings, engine/compiler and evaluator. Malformed JSON is rejected before dispatch.
The Python client raises exceptions on rejection/disconnection and never silently changes
the action. It is a minimal synchronous request client, not a Gym environment or trainer.
A blocking path search must finish before the game thread can detect disconnection or
serve another request. Use small step batches; the default client requests one step.
With larger batches the observation contains the last step's optional evaluator reward;
accumulate rewards externally using single-step requests when needed.

## Validation

AgentPlacement tests cover explicit centers, atomic validation, retained-run isolation,
stale/duplicate actions, fixed steps and cancellation. ExternalPythonRoundTrip starts a
real loopback bridge in a collision-enabled native world and launches
Tools/test_red_team_runtime.py in a separate Python process to verify reset/place/step,
retry without double-stepping, stale-step rejection and cancellation. Existing seeded-mode
tests remain. Runtime integration and game builds are tested; no packaged deployment or
trained-agent performance is implied.
