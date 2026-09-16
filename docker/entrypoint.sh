#!/usr/bin/env bash
# Task launcher for the Istana Open RL/tooling container.
#
# Every subcommand is a thin wrapper over a command already documented in the
# repository, so behaviour inside the container matches the documented host
# workflow. Outputs go to /workspace/runs, which is the intended bind mount.
set -euo pipefail

PY_DIR="/workspace/RL/BlueTeam/Python"
RUNS="${ISTANA_RUNS_DIR:-/workspace/runs}"

usage() {
  cat <<'EOF'
Istana Open container — Blue Team RL, planner and project tools.

USAGE
  docker run --rm istana-rl <command> [args...]

VERIFICATION
  test [pytest args]     Run the full test suite (includes byte-exact checks of
                         the published experiment artifacts). Default: -q
  smoke                  End-to-end pipeline: train, evaluate, replay, recommend.
                         Mirrors .github/workflows/blue-team-rl.yml. Writes runs/smoke.
  versions               Print interpreter and pinned dependency versions.

PLANNING (no Unreal, no network, no device commands)
  recommend [args]       Plan a layout from a public snapshot using the temporal
                         planner. With no args, uses the bundled example.
  recommend-adaptive [args]
                         Same, using the adaptive-v1 checkpoint.

TRAINING AND EVALUATION
  train [args]           train_blue_placement.py (add --dry-run for the toy env)
  train-adaptive [args]  python -m triad_rl.train_adaptive
  train-robust [args]    python -m triad_rl.train_robust
  train-balanced [args]  python -m triad_rl.train_balanced
  evaluate [args]        evaluate_checkpoint.py
  evaluate-adaptive [args]
                         evaluate_adaptive.py

REPLAYS
  demo-adaptive [args]   Render an offline HTML replay into runs/
  demo-balanced [args]
  demo-temporal [args]

PROJECT TOOLS
  geometry               Regenerate Istana + environment OBJ geometry (stdlib only)
  benchmarks             Summarise swarm benchmark CSVs
  redteam-client [args]  Wire client for a RedTeamAgentBridge running on the host.
                         Needs the bridge reachable; see docker/README.md.
  verify [args]          Tools/verify_release.py (needs SourceAssets/ mounted)
  audit                  Tools/audit_submission.py (needs the full tree mounted)

ESCAPE HATCHES
  python [args]          Run python inside RL/BlueTeam/Python
  bash | shell           Interactive shell
  exec <cmd> [args]      Run an arbitrary command
EOF
}

# Run from the Python workspace so relative paths such as ../Examples and
# ../Checkpoints resolve exactly as they do in the documented host commands.
in_py() { cd "$PY_DIR" && exec "$@"; }

command="${1:-help}"
shift || true

case "$command" in
  help|--help|-h)
    usage
    exit 0
    ;;
esac

# Output directory, for the commands that write. Tolerated if the mount is
# absent or read-only; the individual command still reports a real error.
mkdir -p "$RUNS" 2>/dev/null || true

case "$command" in

  versions)
    cd "$PY_DIR"
    exec python - <<'PY'
import platform
from importlib.metadata import version
print("python     ", platform.python_version(), platform.machine())
for name in ("numpy", "gymnasium", "pettingzoo", "pytest", "triad-rl"):
    try:
        print(f"{name:<11}", version(name))
    except Exception as error:
        print(f"{name:<11} unavailable ({error})")
PY
    ;;

  test)
    cd "$PY_DIR"
    if [ "$#" -eq 0 ]; then set -- -q; fi
    # Keep pytest's cache off the project tree so the suite also works when the
    # repository is bind-mounted read-only or owned by another uid.
    exec python -m pytest tests -o cache_dir=/tmp/pytest_cache "$@"
    ;;

  smoke)
    cd "$PY_DIR"
    out="$RUNS/smoke"
    mkdir -p "$out"
    echo "==> toy trainer"
    python train_blue_placement.py --dry-run --episodes 4 --batch-size 2 --seed 22 \
      --checkpoint-every 0 --checkpoint-dir "$out/checkpoints" \
      --metrics-path "$out/training_metrics.jsonl"
    echo "==> evaluate trained checkpoint"
    python evaluate_checkpoint.py --dry-run --episodes 3 --seed 1500000000 \
      --checkpoint "$out/checkpoints/episode_000004_final" --output "$out/evaluation"
    echo "==> evaluate initialized baseline"
    python evaluate_checkpoint.py --dry-run --initial --stochastic --policy-seed 22 \
      --episodes 3 --seed 1500000000 --output "$out/baseline"
    echo "==> adaptive trainer"
    python -m triad_rl.train_adaptive --episodes 8 --batch-size 4 --seed 22 \
      --validation-every 8 --validation-episodes 2 --output "$out/adaptive"
    echo "==> adaptive vs non-RL baselines"
    python evaluate_adaptive.py --checkpoint "$out/adaptive/best" \
      --episodes 3 --bootstrap-samples 100 --replays 1 \
      --output "$out/adaptive-comparison.json"
    echo "==> offline replay"
    python demo_adaptive.py --checkpoint "$out/adaptive/best" --episodes 2 \
      --output "$out/adaptive-demo.html"
    echo "==> external-snapshot recommendation"
    python recommend_adaptive.py --checkpoint ../Checkpoints/adaptive-v1 \
      --input ../Examples/public-snapshot.json \
      --catalogue ../Examples/sensor-catalogue.json --now 0 \
      --output "$out/adaptive-recommendation.json"
    echo
    echo "Smoke pipeline complete. Artifacts in ${out#/workspace/}"
    ;;

  recommend)
    cd "$PY_DIR"
    if [ "$#" -eq 0 ]; then
      set -- --checkpoint ../Results/temporal-v6-pilot/training/seed-406/last \
             --input ../Examples/public-snapshot.json \
             --catalogue ../Examples/sensor-catalogue.json \
             --config ../Examples/temporal-config.json --now 0 \
             --output "$RUNS/temporal-plan.json"
    fi
    exec python recommend_temporal.py "$@"
    ;;

  recommend-adaptive)
    cd "$PY_DIR"
    if [ "$#" -eq 0 ]; then
      set -- --checkpoint ../Checkpoints/adaptive-v1 \
             --input ../Examples/public-snapshot.json \
             --catalogue ../Examples/sensor-catalogue.json --now 0 \
             --output "$RUNS/adaptive-plan.json"
    fi
    exec python recommend_adaptive.py "$@"
    ;;

  train)             in_py python train_blue_placement.py "$@" ;;
  train-adaptive)    in_py python -m triad_rl.train_adaptive "$@" ;;
  train-robust)      in_py python -m triad_rl.train_robust "$@" ;;
  train-balanced)    in_py python -m triad_rl.train_balanced "$@" ;;
  train-temporal)    in_py python train_temporal.py "$@" ;;
  evaluate)          in_py python evaluate_checkpoint.py "$@" ;;
  evaluate-adaptive) in_py python evaluate_adaptive.py "$@" ;;
  demo-adaptive)     in_py python demo_adaptive.py "$@" ;;
  demo-balanced)     in_py python demo_balanced.py "$@" ;;
  demo-temporal)     in_py python demo_temporal.py "$@" ;;
  redteam-client)    cd /workspace && exec python Tools/red_team_client.py "$@" ;;
  benchmarks)        cd /workspace && exec python Tools/summarize_swarm_benchmarks.py "$@" ;;
  verify)            cd /workspace && exec python Tools/verify_release.py "$@" ;;
  audit)             cd /workspace && exec python Tools/audit_submission.py "$@" ;;

  geometry)
    cd /workspace
    echo "==> generate_istana.py"
    python Tools/generate_istana.py
    echo "==> generate_environment.py"
    python Tools/generate_environment.py
    echo "Geometry written under SourceAssets/ (mount the repo to keep the output)."
    ;;

  python)        in_py python "$@" ;;
  bash|shell|sh) cd /workspace && exec bash ;;
  exec)          cd /workspace && exec "$@" ;;

  *)
    echo "Unknown command: ${command}" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
