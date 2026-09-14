# Adaptive sensor-input contract

The same `build_observation` function serves the randomized simulator and
`LiveObservationAdapter`. A provider translates its measurements and planning
constraints into `triad.sensor_input.v1`; the policy selects an approved sensor
and site. This is an **offline recommendation interface**, not a sensor driver,
device deployment command, validated real-world sensing model, or live control
integration.

Implementation: [adaptive_inputs.py](Python/triad_rl/adaptive_inputs.py),
[recommend_adaptive.py](Python/recommend_adaptive.py).

## Coordinates, directions and time

All positions share one provider-defined local origin at the protected objective.
Site/placement positions are `[east, north]` in metres. Track positions are
`[east, north, up]` in metres; velocity uses the same axes in metres/second.
Catalogue `height_m` is the sensor's up-coordinate in that frame: no terrain or
geodetic conversion is performed. Convert latitude/longitude and sensor-specific
frames before building the snapshot.

The eight forecast sectors run counterclockwise from east, not clockwise from
north:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Direction | E | NE | N | NW | W | SW | S | SE |
| Angle | 0° | 45° | 90° | 135° | 180° | 225° | 270° | 315° |

Snapshot/track timestamps, `max_track_age`, and `--now` use seconds on one
consistent clock: either Unix time or a common monotonic epoch. Do not mix them.
`--now` must be at least the snapshot timestamp and no greater than 10¹⁵. Without it, the command uses the
snapshot timestamp for reproducible offline replay; it does **not** check the
current wall clock. A current-input integration should supply its current time.

## Public snapshot

All values must be finite JSON. Unknown top-level fields are rejected. The
following table lists every accepted top-level field; omitted optional fields use
the defaults shown.

| Field | Required / default | Meaning and accepted limits |
|---|---|---|
| `schema` | Required | Exactly `triad.sensor_input.v1`. |
| `sites` | Required | 1–512 `[east,north]` pairs; each coordinate in ±10,000 m. These are provider-approved candidate locations. |
| `budget_total` | Required | Total planning budget, 0.001–100 units. |
| `budget_remaining` | Required | Uncommitted budget, 0–`budget_total`; same units as catalogue costs. |
| `max_sites` | Required | Integer 1–32, including existing placements. |
| `timestamp` | `0` | Snapshot time, 0–10¹⁵ seconds. |
| `source` | `"unspecified"` | Audit label identifying the provider; not a learned feature. |
| `episode_id` | Optional | Opaque audit/correlation value; not a learned feature. |
| `max_track_age` | `10` | Track freshness threshold, 0.001–3,600 seconds. |
| `placements` | `[]` | Existing committed/planned sensors; rows described below. |
| `min_separation` | `20` | Required distance from existing placements, 0–1,000 m. |
| `deployment_min_radius` | `30` | Minimum allowed distance from origin, 0–1,000 m. |
| `deployment_max_radius` | `150` | Maximum allowed distance, from the minimum radius through 2,000 m. |
| `available_sensor_ids` | All catalogue IDs | List of usable profiles; an empty list allows only STOP. Unknown IDs are rejected. |
| `blocked_sites` | `[]` | Zero-based integer indices into `sites` that cannot be selected. |
| `weather` | Defaults below | Public normalized environmental estimates. |
| `forecast` | Defaults below | Public threat priors, not future trajectory truth. |
| `tracks` | `[]` | Up to 256 observed/noisy track rows. |
| `fresh_track_fraction` | Computed | Normally omit. Optional upstream freshness fraction in [0,1], multiplied by the adapter's surviving-track fraction. |
| `done` | `false` | Use JSON boolean. A stopped plan permits no additional deployment actions. |

A placement row needs `sensor_id` from the catalogue and `position:[east,north]`.
Its `sensor_index` and `cost` are reconstructed from the catalogue, not trusted
from the input. The number of rows cannot exceed `max_sites`, and:

```text
sum(existing placement costs) + budget_remaining <= budget_total
```

This permits budget reserved elsewhere, but not overspending. The provider owns
the correctness of existing placements and real-world site approval. The action
mask validates **new** placements against radius, separation, site availability,
budget and remaining slots. Reusing an occupied coordinate is forbidden even
when `min_separation` is zero.

### Weather and forecast

Every weather field is in [0,1]; these are normalized model inputs, not literal
lux, rainfall or meteorological visibility units. Map actual readings explicitly
and calibrate before considering operational use.

| `weather` field | Default | Interpretation |
|---|---:|---|
| `visibility` | 1 | Optical visibility: low to clear. |
| `rain` | 0 | Rain severity: none to severe. |
| `illumination` | 1 | Scene illumination: dark to bright. |
| `humidity` | 0.4 | Relative normalized humidity. |
| `rf_noise` | 0.1 | RF noise/interference severity. |

| `forecast` field | Default | Limits / meaning |
|---|---|---|
| `approach_weights` | Eight `0.125` values | Exactly eight values in [0,1] with positive sum; normalized to probabilities. Sector order is shown above. |
| `altitude` | 45 | Estimated up-coordinate, 0–1,000 m. |
| `speed` | 12 | Estimated speed, 0.1–200 m/s. |
| `emitter_probability` | 0.5 | Probability of a relevant RF emission, [0,1]. |
| `swarm_size` | 1 | Estimated threat count, 1–64; fractional estimates are accepted. |
| `angular_uncertainty` | 0.4 | Angular spread in radians, 0–π; not a calibrated confidence interval. |

The feature builder uses forecast-sector probabilities and fresh noisy track
bearings to approximate plausible ingress coverage. It does not infer exact
future paths. Missing values revert to the listed priors; a provider should
supply its best honest estimates rather than treating defaults as measurements.

### Tracks and staleness

| Track field | Required / default | Limits / meaning |
|---|---|---|
| `id` | Required | Provider track identifier, converted to a string. |
| `position` | Required | Three coordinates, each in ±100,000 m. |
| `velocity` | `[0,0,0]` | Three components, each in ±500 m/s. |
| `confidence` | 0.5 | [0,1]. |
| `emitter_probability` | Forecast value | [0,1]. |
| `timestamp` | Snapshot timestamp | 0 through evaluation time; future-relative-to-evaluation tracks are rejected. |
| `confirmed` | `false` | Provider confirmation flag; use a JSON boolean. |

Tracks older than `max_track_age` at evaluation time are discarded. Empty or
entirely stale tracks produce a forecast-only observation, with zero freshness.
The policy receives aggregate confidence/freshness and candidate-to-track distance
information. Current coverage uses track bearings and confidence; it does not
extrapolate track velocities or treat `confirmed` as a simulated defence outcome.
Track velocity, per-track emission estimates and confirmation flags are retained
in the public state but are not separately scored by the current policy.

## Sensor catalogue

Omit `--catalogue` to use these synthetic profiles; every sensor has `height_m=4`:

| ID | Cost | Nominal slant ranges | Per-look strengths |
|---|---:|---|---|
| `rf` | 0.8 | RF 130 m | RF 0.88 |
| `radar` | 1.2 | Radar 100 m | Radar 0.86 |
| `eo` | 0.7 | EO 100 m | EO 0.94 |
| `thermal` | 1.0 | Thermal 115 m | Thermal 0.86 |
| `fused` | 2.0 | Radar 100 m + thermal 125 m | 0.86 for each modality |

A custom catalogue JSON file is a list of 1–32 capability rows:

```json
[
  {
    "id": "custom-radar-thermal",
    "label": "Custom fused profile",
    "cost": 1.8,
    "ranges": {"radar": 110, "thermal": 120},
    "strengths": {"radar": 0.86, "thermal": 0.82},
    "height_m": 4
  }
]
```

`id` must be unique and 1–128 characters; `label` defaults to the ID. `cost` is
0.001–100. Each `ranges` entry is 0–2,000 m, with at least one positive range.
Only `rf`, `radar`, `eo` and `thermal` are represented; missing modalities have
range zero. `strengths` values are [0,1], defaulting to 0.85 for a present modality
and zero otherwise. `height_m` defaults to 4 and accepts 0–200 m.

The scorer is independent of catalogue length, IDs and site count: capabilities,
geometry and public context determine its input. This is schema compatibility,
**not proof of performance** for unseen hardware, extreme ranges or new operating
conditions. A genuinely new modality requires a feature/schema change and new
training. The current synthetic model omits terrain, occlusion, detailed antenna
patterns, device latency and calibrated false-alarm behavior.

## Recommendations and replanning

The current action is one integer selecting a joint `(sensor, site)` option, or
STOP. Options are sensor-major, then site order, with STOP last. Resolve the
integer against the **same observation** that the policy scored; do not retain an
index across catalogue/site changes. Each recommendation includes `sensor_id`,
`sensor_index`, `site_index`, physical `position` and `stop`.

The provider can offer arbitrary approved site coordinates, not just the
training simulator's rotated rings. The policy chooses among these finite
locations. Unlike the original toy-210 model's continuous normalized East/North
head, this model does **not** invent a coordinate outside the offered site list.

For a local planning loop, call `LiveObservationAdapter.observe(payload, now=...)`,
then `policy.act(observation)`, resolve with `adapter.recommendation`, and call
`apply_placement(payload, action, catalogue, now=...)`. Rebuild the observation
after each choice so coverage, remaining budget and separation reflect the
planned layout. `apply_placement` copies state; it sends no physical commands.

For updated measurements, start from a new authoritative snapshot containing
the actual retained placements and current remaining budget. Do not relabel an
unexecuted plan as installed equipment. STOP ends that local plan; a fresh
snapshot may start another. The present training/evaluation covers sequential
pre-deployment planning, not autonomous mid-flight relocation or a validated
closed-loop controller.

From the repository root, using an installed Python environment and the supplied
[public-only example snapshot](Examples/public-snapshot.json):

```powershell
python RL/BlueTeam/Python/recommend_adaptive.py --checkpoint RL/BlueTeam/Checkpoints/adaptive-v1 --input RL/BlueTeam/Examples/public-snapshot.json --now 0 --output plan.json
```

The example's clock starts at zero. Replace its input path and `--now` with your
snapshot and evaluation time for other inputs. For a custom catalogue, append
`--catalogue custom-catalogue.json` and use
its IDs in `placements`/`available_sensor_ids`. The output schema is
`triad.deployment_recommendation.v1`; it includes decisions, new placements, final
planning state, the checkpoint weight fingerprint, evaluation time and
`physical_commands_sent:false`. No cloud service, socket or hardware driver is
invoked by this command. [sensor-catalogue.json](Examples/sensor-catalogue.json)
provides the default five profiles in the external catalogue format.

## Truth separation and metric meanings

Pass observable measurements, weather estimates, approved capabilities and
explicit priors—not the simulator's `scenario`, private `targets`, future paths,
emitter phases or future detection outcomes. Private truth appears only in
simulation scoring and evaluation replay. A live snapshot does not contain enough
information to declare future mission success or assign the simulator's reward.

Coverage is the synthetic expected per-look probability of detecting a target
over weighted **forecast corridor samples**, combined across sensors/modalities.
It is not measured geographic area or a guarantee of detection. Marginal
coverage measures a candidate's improvement over existing placements; sector
coverage uses the same eight directional sectors. Early-coverage features weight
farther corridor samples more heavily. Forecast samples currently use radii
55, 105, 155 and 205 m, so very different site scales need revalidation.

In simulator results, `detected_fraction`/`detection_rate` means at least one
sighting; `confirmed_fraction` means enough sightings in the confirmation window.
Default confirmation is two sightings within three one-second simulation ticks.
`early_detection` measures confirmation earliness relative to zone-arrival time,
with zero for never-confirmed threats. A target is labelled defended if confirmed
at least four seconds before arrival; episode `success` requires every target to
meet that threshold. This is a sensing-only proxy, not a weapon engagement model.

Replay `detected` means a hit at the displayed tick, `ever_detected` is cumulative,
and `tracked` records that confirmation has occurred. Confirmation does not decay:
track loss/reacquisition is not modeled. Detection-to-sensor attribution is
explanatory and selects the strongest contributing sensor per successful
modality. Reward is a simulator aggregate of defence, breaches, sightings,
confirmation, earliness, forecast coverage, cost and penalties—not a field metric.
