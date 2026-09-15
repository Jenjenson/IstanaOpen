# Validation

The packaged-viewer measurements below were taken on 10 September 2026. A separate
swarm/red-team validation run from 12 September is recorded at the end of this document.
These are local results, not minimum hardware requirements or guarantees for other PCs.

## Tested build and machine

- Unreal Engine 5.5.4, Development configuration, Windows 64-bit.
- Windows 11, AMD Ryzen 7 5700X3D, 32 GB RAM, NVIDIA TITAN V with 12 GB VRAM.
- Packaged application copied to a separate release folder and launched from
  an unrelated working directory, without the Unreal Editor.
- Packaging completed with zero errors and 16 nonfatal optional-plugin or
  texture-codec warnings. The package was approximately 708.9 MB before
  adding these validation documents and evidence.

The relocated test copy was at
`D:\triad\IstanaOpen\Saved\Standalone Tests\Windows`, launched with `D:\` as
the working directory. Those are test-machine locations, not installation
requirements. A destination computer does not need either path.

## Packaged rendering measurements

All captures use a 1600 × 900 window and a 60 FPS cap. The first five seconds
are excluded as warmup. The reported average is measured frames divided by
measured time; it is not an uncapped GPU benchmark or a percentile frame-time
measurement. Initial loading is not timed; cold shader/driver-cache startup
behavior was not separately benchmarked.

| Render path | Quality and view | Measured time | Frames | Average FPS | Evidence |
| --- | --- | ---: | ---: | ---: | --- |
| DirectX 12 | Medium, palace | 33.018 s | 1,938 | 58.695 | [Capture](Evidence/packaged-medium.png), [raw report](Evidence/packaged-medium.json) |
| DirectX 12 | High, aerial | 28.029 s | 1,668 | 59.510 | [Capture](Evidence/packaged-aerial-high.png), [raw report](Evidence/packaged-aerial-high.json) |
| DirectX 11 | Medium, garden | 28.029 s | 1,667 | 59.474 | [Capture](Evidence/packaged-dx11.png), [raw report](Evidence/packaged-dx11.json) |

The completed captures report a standalone game world, possessed camera,
saved screenshot, the requested resolution, and 28 static mesh actors.
Raw reports preserve their original absolute screenshot paths. The matching
PNGs are included beside the reports in [Evidence](Evidence/README.md); those
capture paths are not runtime dependencies.

## Controls and interaction

The controls are implemented in the native application, and the native build
succeeded. The packaged tests verified camera possession and the palace,
aerial and garden presets selected by the automated capture launch options.

Manual keyboard and mouse interaction was not completed. The supported
desktop automation launch failed with `GetCursorPos: Access is denied (0x80070005)`
before launching a process for that interaction attempt. This
is an uncompleted test, not a keyboard-control pass. The three separate
packaged rendering runs above did launch and complete successfully.

WASD movement, mouse look, quality switching, tour toggling, photo mode,
pause/resume and quit have therefore not been confirmed through manual UI
interaction in this session.

## Source and saved scene checks

- All 60 scanned source-asset files match their recorded sizes and SHA-256
  provenance hashes.
- All 28 generated OBJ files pass finite-coordinate and vertex/UV/normal-index
  checks, with no degenerate triangles detected.
- The saved map reloads with 3,074 vegetation mesh instances across 2,904
  planting positions. The 170 Pachira positions each pair a trunk and crown.
- Imported OBJ bounds match the source centimetre coordinates and axis
  convention; the scene contains 28 static mesh actors.
- The source audit checks the publishable tree for nonempty credentials,
  unexpected enabled plugins, prohibited asset directories and restricted
  package references, and records file hashes. Build products, caches and
  release folders are excluded. See [submission audit](submission-audit.json).

Recheck the source with `python Tools/verify_release.py --source-only` and
`python Tools/audit_submission.py`. A source audit is a technical check;
asset terms and attribution are documented separately in
[LICENSING.md](LICENSING.md).

## Scope and limits

This is an artistic reconstruction of the publicly visible palace exterior.
The landscape, hill, garden layout and planting are authored approximations.
The city uses an August 2026 OpenStreetMap snapshot: 1,413 building footprints
and 926 road ways. Only 26 building heights use explicit mapped height tags;
631 are inferred from mapped floor counts, and 756 use authored plausible
heights. Generic facade treatments are authored. None of these are a survey
or an as-built digital twin.

Interiors, current restoration works, security arrangements and physical
sensor simulation are not modeled. The camera flies freely without building
collision. The materials and vegetation support visual exploration, rather
than a scientific simulation of lighting, botany or water.

Other computers, clean prerequisite installation, macOS and Linux have not
been tested. Different GPUs, drivers, memory capacities and display settings
can change startup behavior and performance. No minimum hardware claim is
made. DirectX 11 compatibility uses a simplified rendering
path and is not expected to match the DirectX 12 appearance.

## Swarm and red-team validation (12 September 2026)

Both Unreal Engine 5.5.4 Win64 Development targets (`IstanaOpenEditor` and `IstanaOpen`)
built successfully after the objective-following changes. The unattended
`Istana.Simulation` run completed with **13 successes, 0 failures, 0 skipped tests**
and no test warnings. This run used WindowsEditor with `-nullrhi`; it is not a rendering
benchmark and does not inherit the packaged-viewer GPU/FPS measurements above.

The suite covers four shared-contract tests, eight swarm tests and one red-team test:

- Seeded spawn/reset, group ordering, command validation, movement limits and population.
- Collision-aware detours, static-mesh collision and manager-relative placement.
- Spawn and arrival at coordinates outside the removed arena boundary.
- Multiple visible red-team groups around a shared objective, repeatable reset,
  shared-target arrival/retargeting, invalid-spawn preservation and visual cleanup.
- Floor and embedded objectives accepted for partial approach without disabling collision;
  automatic movement to the unchanged objective after an obstruction moves.

The report for this run is stored locally at
`Saved/Automation/ObjectiveApproach/index.json` (report timestamp
`2026.09.12-02.34.45`). Saved reports are generated/ignored files, not published repository
artifacts. Reproduce the suite using the [swarm guide](SWARM_SIMULATION.md#build-and-verify)
and inspect [the regression tests](../Source/IstanaOpen/Simulation/Tests/IstanaSwarmTests.cpp).

Earlier visual/reset and objective-marker Python smoke checks passed after the arena
and predefined-obstacle removal. They use transient managers and leave existing maps
unsaved. The later 13-test native run above validates the partial-path change. No new
packaged release, manual play session, large-map navigation benchmark, or physical
flight-fidelity validation is claimed by these results.

## Red-team optimization and agent API (15 September 2026)

Editor and Game Development builds passed. The final native suite passed **17 tests,
0 failures**, including unchanged original assertions, exact source-reference state/route
replay, explicit placement, blocked placement atomicity, provider integration, and a real
external Python client exercising reset/place/step/retry/reconnect/timeout.
The local report is Saved/Automation/SwarmFinalRegression/index.json.

See [optimization results](SWARM_OPTIMIZATION_RESULTS.md) for the 42-case lightweight
solver matrix, completed real-level cases, raw data, numerical comparison scope and
remaining synchronous planning spikes. See [agent setup](RED_TEAM_AGENT.md) for use.
These checks do not imply a trained policy, packaged deployment or GPU/FPS improvement.
