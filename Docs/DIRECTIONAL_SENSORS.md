# Directional sensor model

The live Unreal Blue Team now supports generic camera-like directional sensor
profiles. A placement is `profile + approved site + yaw + pitch`; the native
bridge accepts schema version 2 and validates the selected profile's advertised
orientation bins before it commits a layout.

## Boson+ 640 18 mm reference profile

### Manufacturer specifications

The first profile uses the Teledyne FLIR Boson+ 640, 18 mm / 24° HFOV variant.
The following fields are copied from the [official Boson+ product page](https://oem.flir.com/products/boson-plus/?model=22640A024):

| Field | Value |
| --- | --- |
| Detector | 640 × 512, uncooled LWIR |
| Pixel pitch | 12 µm |
| Horizontal FOV / focal length | 24° / 18 mm |
| IFOV | 0.667 mrad |
| Industrial NEdT | ≤20 mK |
| Frame rate | 60 Hz baseline; 30 Hz selectable |

These are camera specifications, not a manufacturer-rated drone detection
range. The simulator does not claim that this camera reliably detects a drone
at any particular distance.

### Calculated geometry

The product page does not publish VFOV for this exact row. The simulator derives
it from the rectilinear 640:512 detector geometry:

```text
VFOV = 2 atan(tan(24° / 2) × 512 / 640) = 19.301°
```

This value is serialized under `calculated_geometry`, separately from the
manufacturer fields.

### Simulation assumptions

The following values are modelling choices and are editable in
`FDirectionalSensorProfile`:

| Parameter | Initial value |
| --- | --- |
| Maximum evaluation distance | 500 m |
| Nominal target/background thermal contrast | 8 K |
| Contrast/noise multiplier | 8 |
| Pixels for 63% pixel response | 3 px |
| Atmospheric attenuation distance | 900 m |
| Maximum rain loss | 0.45 |
| Maximum humidity loss | 0.35 |
| FOV-edge exponent | 1.5 |
| Yaw bins | 0°, 45°, …, 315° |
| Pitch bins | 0°, 10°, 20° |
| Mount height | 4 m |

The 500 m value is only the simulator's evaluation boundary. It is not a FLIR
range specification. The target diameter comes from the active drone movement
configuration (`2 × DroneRadiusCm`), so different actor sizes change the result.

For a target diameter `s`, distance `d`, and IFOV in radians per pixel:

```text
angular diameter = 2 atan(s / (2d))
pixels on target = angular diameter / IFOV
```

For every sensing look, the model combines pixels on target, assumed contrast
relative to NEdT, smooth cosine distance taper, atmospheric attenuation,
visibility, rain, humidity, angular displacement from the optical axis, and a
binary line-of-sight gate. World-static Unreal line traces provide LOS against
terrain and building collision. The resulting probability is then sampled by
the existing deterministic indexed RNG and fed into the unchanged repeated-hit
confirmation window.

Hardware frame rate is descriptive metadata. The simulation deliberately uses
the existing 1 s sensing-look interval; it does not incorrectly treat 60 video
frames as 60 independent Bernoulli detections.

## RL and compatibility

`triad.directional_placement_features.v1` enumerates profile/site/yaw/pitch
choices plus STOP. `istana.warning_directional_reinforce.v2` learns a separate
logit for every legal joint option and is trained only from terminal native
warning-time rewards.

The checksum-pinned `adaptive_inputs.py` and `temporal_inputs.py` contracts are
unchanged. Historical checkpoints 406/407/408 learned profile and site only.
When one is deliberately used with a directional live catalogue, the wrapper
adds yaw/pitch using a labelled public-prior adapter; it never describes those
angles as learned by the frozen checkpoint. New warning-policy checkpoints
learn the full joint action directly.

## Visualization and evidence

With Blue debug coverage enabled, Unreal draws the selected sensor's 3D frustum,
centre ray, location, yaw/pitch, configured boundary, and target diagnostics.
The observation API also exposes presentation-only distance, in/out-of-FOV,
LOS/blocked, pixels-on-target and per-look probability. Terminal evaluation
evidence records first detection, first confirmation, target-zone arrival,
warning time, pixels and probability at first detection. These observer fields
never enter the policy's public input.

Run the focused checks from `RL/BlueTeam/Python`:

```powershell
..\.venv\Scripts\python.exe -m pytest tests\test_directional_inputs.py tests\test_warning_training.py tests\test_istana_live.py -q
```

The native automation test is
`Istana.Simulation.BlueTeam.DirectionalSensorModel`.
