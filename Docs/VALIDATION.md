# Validation

Tested on 10 September 2026. These results describe one Windows computer;
they are not minimum hardware requirements or a guarantee for arbitrary PCs.

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
