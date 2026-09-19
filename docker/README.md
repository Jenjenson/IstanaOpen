# Running Istana Open in containers

Everything that *can* be made reproducible-anywhere is in one CPU-only image:
the Blue Team RL trainer and planner, the full 66-module test suite, the
published experiment evidence, and the standard-library project tools. It needs
no GPU, no network, no account and no API key.

It builds and runs natively on **linux/amd64 and linux/arm64**, so Apple Silicon
needs no emulation and no extra flags. See
[Architectures](#architectures-amd64-and-arm64).

Unreal Engine is the exception, and the reason is licensing rather than
packaging. It is handled separately and honestly in [Unreal](#unreal-engine-optional-and-entitlement-gated).

## What is containerised

| Component | Container status | Notes |
| --- | --- | --- |
| `triad_rl` trainer, planner, evaluators | **Fully containerised** | CPU-only NumPy/Gymnasium/PettingZoo |
| Test suite (66 modules, 1,682 checks) | **Fully containerised** | Includes byte-exact verification of archived artifacts; passes with `--network none` |
| Published evidence (`Results/`, `Checkpoints/`, `Examples/`) | **Baked into the image** | ~184 MB, so verification works offline |
| Offline HTML replays | **Fully containerised** | Rendered into `./runs` |
| Stdlib tools (geometry, benchmarks, release verify, red-team client) | **Fully containerised** | Some need the repo bind-mounted; see below |
| Unreal C++ build and automation tests | **Optional, entitlement-gated** | Epic base image; amd64 only; never verified on Linux |
| Packaged Unreal viewer (the interactive 3-D app) | **Not containerised** | Windows DX11/DX12 GPU application; use the [release ZIP](../README.md#download-and-run-on-windows) |

## Quick start

```bash
git clone https://github.com/Jenjenson/IstanaOpen.git
cd IstanaOpen

# No `git lfs pull` needed — the image excludes Content/ and SourceAssets/.
docker compose run --rm test      # full suite, offline
docker compose run --rm smoke     # train → evaluate → replay → plan
docker compose run --rm plan      # layout recommendation into ./runs
```

Or without compose:

```bash
docker build -t istana-rl .
docker run --rm --network none istana-rl test
docker run --rm -v "$PWD/runs:/workspace/runs" istana-rl smoke
```

On Windows PowerShell, replace `$PWD` with `${PWD}`.

## Commands

`docker run --rm istana-rl help` prints the full list. The common ones:

| Command | Does |
| --- | --- |
| `test [pytest args]` | Full suite; `-q` by default |
| `smoke` | End-to-end pipeline mirroring `.github/workflows/blue-team-rl.yml` |
| `versions` | Interpreter and resolved dependency versions |
| `recommend` | Temporal planner on the bundled public snapshot → `runs/temporal-plan.json` |
| `recommend-adaptive` | Same via the `adaptive-v1` checkpoint |
| `train`, `train-adaptive`, `train-robust`, `train-balanced`, `train-temporal` | Trainers |
| `evaluate`, `evaluate-adaptive` | Evaluators |
| `demo-adaptive`, `demo-balanced`, `demo-temporal` | Offline HTML replays |
| `benchmarks`, `geometry`, `verify`, `audit` | Project tools |
| `redteam-client` | Wire client for a host-side Unreal bridge |
| `python`, `bash`, `exec` | Escape hatches |

Trainer flags pass straight through:

```bash
docker run --rm -v "$PWD/runs:/workspace/runs" istana-rl \
  train-adaptive --episodes 200 --batch-size 8 --seed 42 --output /workspace/runs/adaptive
```

Write outputs under `/workspace/runs`; the rest of the filesystem is not a mount
and the container runs as an unprivileged user (`istana`, uid 1000).

Docker Desktop on macOS and Windows maps bind-mount ownership for you. On Linux,
a host uid other than 1000 cannot write into `./runs`, so declare your own ids:

```bash
export ISTANA_UID=$(id -u) ISTANA_GID=$(id -g)   # picked up by docker-compose.yml
docker compose run --rm smoke
```

With plain `docker run`, pass `--user "$(id -u):$(id -g)"`. The entrypoint
redirects `HOME` to `/tmp` when the uid has no passwd entry, so an arbitrary id
works.

## Architectures: amd64 and arm64

`python:3.11-slim-bookworm` publishes both architectures, and every pin in
[`constraints.txt`](constraints.txt) has a manylinux wheel for `x86_64` and
`aarch64`, so the same Dockerfile produces a native image on either. Nothing in
the quick start changes on an Apple Silicon Mac.

The build passes `--only-binary=:all:` deliberately. The image carries no
compiler, so if a wheel ever disappears for one architecture we want pip to name
the package rather than fail halfway through a source build.

Two caveats:

1. **The optional Unreal image is amd64-only.** Epic publishes no arm64 tag, and
   emulating a full engine build is not a practical path. Build the native
   modules on an x86-64 machine, or on Windows with `Tools/build.ps1`.
2. **The archived numbers were generated on x86-64.** The suite passes on arm64,
   but floating-point reduction order is not guaranteed identical across
   architectures, and this project's own evidence shows how small those
   differences can be while still changing a hash (see below). If you are
   checking artifact hashes rather than behaviour, pin the architecture:

   ```bash
   DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose run --rm test
   ```

   On an arm64 host that runs under emulation and is several times slower.
   For everyday use, run natively.

To build both architectures at once, for a registry:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t <registry>/istana-rl:1.0 --push .
```

## Reproducibility

Four things are pinned deliberately.

**Dependency versions.** [`constraints.txt`](constraints.txt) pins NumPy to
`2.4.6`, the version recorded by the published experiments
(`RL/BlueTeam/TEMPORAL_RESULTS.md`). To build against current upstream instead:

```bash
docker build --build-arg CONSTRAINTS=docker/constraints-current.txt -t istana-rl .
```

**Thread count.** The image sets `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.
This is not a performance choice — the recorded runs used single-threaded BLAS,
and letting the thread count follow the host's core count changes float
reduction order. Leave these alone when reproducing archived numbers.

**Hash seed.** `PYTHONHASHSEED=0`.

**Python minor version.** `python:3.11-slim-bookworm`, matching the declared
`requires-python` and the CI matrix. Override with `--build-arg PYTHON_VERSION=3.12`.

### What containerisation does *not* fix

Cross-platform float behaviour. `RL/BlueTeam/BALANCED_RESULTS.md` records 172
last-bit differences (max ~5.7e-14) between Windows and Linux across 47 of 60
snapshots, which changed hashes of unrounded JSON without changing a single
sensor choice or action count. Running Linux in a container on a Windows host
gives you the *Linux* numbers consistently — which is the point — but it does
not make Windows-generated and Linux-generated hashes identical. The frozen
verifiers already account for this with a declared `1e-12` tolerance on derived
numerics while keeping artifact identities exact.

## Tools that need the repository mounted

`Content/` and `SourceAssets/` are excluded from the image (1 GB of Git LFS
content that no Python test touches). Tools that read them need a bind mount and
a real `git lfs pull` on the host:

```bash
git lfs install && git lfs pull
docker run --rm -v "$PWD:/workspace" istana-rl verify --source-only
docker run --rm -v "$PWD:/workspace" istana-rl audit
docker run --rm -v "$PWD:/workspace" istana-rl geometry
```

Mounting the repo over `/workspace` shadows the baked-in copy, so the installed
`triad_rl` package and your working tree can diverge; re-run
`pip install --no-deps ./RL/BlueTeam/Python` inside the container if you have
edited the package itself.

## Talking to the Unreal red-team bridge

The bridge (`RedTeamAgentBridge`) runs inside Unreal on the **host**, during PIE
or a packaged development game, and binds loopback on port 8765. A container
cannot reach the host's loopback, so `Tools/red_team_client.py` now accepts
`--host` (default `127.0.0.1`, or `$ISTANA_REDTEAM_HOST`):

```bash
docker compose run --rm redteam-client
# equivalently:
docker run --rm --add-host host.docker.internal:host-gateway istana-rl \
  redteam-client --host host.docker.internal --port 8765 --steps 100
```

The bridge's own behaviour is unchanged: one request outstanding at a time,
idempotent retries, and disconnect cancels the episode. If the container's
connection drops, the episode is cancelled and needs a fresh `reset`.

## Unreal Engine: optional and entitlement-gated

```bash
git lfs install && git lfs pull                       # .uasset files must be real
docker login ghcr.io -u <github-user> -p <pat>        # Epic-linked account
docker compose --profile unreal run --rm unreal all
```

Three caveats, stated plainly:

1. **It is not "runs anywhere."** Epic's `ghcr.io/epicgames/unreal-engine` image
   is only pullable by accounts linked to an Epic Games account and admitted to
   the EpicGames GitHub organisation. It cannot be mirrored or redistributed.
   No packaging work can remove that constraint.
2. **It is amd64-only, and the Linux build is unverified.** Epic publishes no
   arm64 tag. Beyond that, this project has only ever been built and tested on
   Windows — `Docs/VALIDATION.md` records the 18 passing native tests from that
   host. Nobody has run a Linux build. Treat failures as new work.
3. **The project is mounted, not copied.** Unreal needs the LFS trees the Python
   image excludes, and its `Intermediate/`/`Binaries/` output is multi-gigabyte.
   Compose bind-mounts the repository at `/workspace` and builds at container
   start, keeping that output on the host.

The interactive viewer is out of scope entirely: it is a Windows DirectX
application, and containerising a GPU desktop app would add far more fragility
than it removes. Use the packaged release ZIP.

## Troubleshooting

**Build fails resolving `numpy==2.4.6`.** That pin matches the archived
experiments. Because the build uses `--only-binary=:all:`, pip will say plainly
that no wheel exists for your platform or Python minor version rather than
attempting a source build. Rebuild with
`--build-arg CONSTRAINTS=docker/constraints-current.txt` and accept last-bit
numeric drift from the archived artifacts.

**`entrypoint.sh: no such file or directory`.** CRLF line endings on the script.
`.gitattributes` forces `*.sh` to LF; re-clone or run
`git add --renormalize . && git checkout -- docker/`.

**Permission denied writing output.** Write under `/workspace/runs`, and on Linux
export `ISTANA_UID`/`ISTANA_GID` (or pass `--user "$(id -u):$(id -g)"`) so the
container writes as you. See [Commands](#commands).

**`exec format error`, or the wrong architecture after switching platforms.**
Compose reuses the `istana-rl:local` tag, so an image built for the other
architecture can linger. Force a rebuild with `docker compose build --no-cache`,
and confirm with `docker compose run --rm rl versions`, which prints the
architecture it is actually running on.

**Tests fail on published artifacts after you edited `RL/BlueTeam/Results/`.**
Expected — those tests verify archived bytes. Restore them with
`git checkout -- RL/BlueTeam/Results`.

**Image is larger than expected.** ~184 MB of that is the published evidence
tree, deliberately baked in so verification works offline. Building with
`RL/BlueTeam/Results/` added to `.dockerignore` shrinks the image but makes the
published-artifact tests fail.
