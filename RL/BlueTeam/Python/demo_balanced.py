"""Record balanced-policy decisions using the unchanged offline replay viewer.

The simulator's private truth is used only for scored replay frames and audit,
never for selecting an action. The browser does not run inference or control
devices. All replay cases are validation examples, not independent final tests.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from demo_robust import collect_replay, implementation_fingerprints as replay_sources
from evaluate_robust import checkpoint_files, validate_seed_range
from triad_rl.adaptive_evaluation import canonical_hash
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.balanced_policy import BalancedPolicy, POLICY_SCHEMA


RELEASE = "balanced-v3"
REPLAY_SCHEMA = "triad.balanced_replay.v1"
DEFAULT_SEED = 1_000_991_900_000_000


def export_balanced_html(replays, output, *, checkpoint, weights_sha256):
    """Reuse the frozen viewer, routing sensor labels through its collision layout."""
    if not replays or any(not replay["frames"] for replay in replays):
        raise ValueError("At least one replay with frames is required")
    template = Path(__file__).with_name("adaptive_demo.html").read_text(encoding="utf-8")
    original = '''text([x + 10, y - 9], compact ? `${index + 1}` : `${index + 1} ${c.label || p.sensor_id}`, {
              fill: "#d3eaff",
              "font-size": 12,
            });'''
    replacement = '''threatLabel([x, y], compact ? `${index + 1}` : `${index + 1} ${c.label || p.sensor_id}`, "#d3eaff");'''
    if template.count(original) != 1 or template.count("__REPLAY_DATA__") != 1:
        raise ValueError("Frozen viewer label/data insertion contract changed")
    template = template.replace(original, replacement)
    payload = json.dumps({"schema": "triad.adaptive_demo.v1", "checkpoint": checkpoint,
                          "weights_sha256": weights_sha256, "replays": replays},
                         allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REPLAY_DATA__", payload), encoding="utf-8")


def implementation_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {**replay_sources(), **{
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("demo_balanced.py", "triad_rl/balanced_policy.py")}}


def run_demo(*, checkpoint: Path, output: Path, seed: int = DEFAULT_SEED,
             episodes: int = 6, profile: str = "capability") -> list[dict]:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or not 1 <= episodes <= 50:
        raise ValueError("episodes must be an integer in [1, 50]")
    validate_seed_range(seed, episodes, "validation")
    if profile not in ("normal", "stress", "capability", "mixed"):
        raise ValueError("Unknown replay profile")
    checkpoint, output = Path(checkpoint), Path(output)
    sources = implementation_fingerprints()
    files = checkpoint_files(checkpoint)
    policy = BalancedPolicy.load(checkpoint, feature_names=FEATURE_NAMES)
    validate_seed_range(seed, episodes, "validation", policy.metadata)
    weights = policy.weights_fingerprint()
    rng = canonical_hash(policy.rng.bit_generator.state)
    replays = []
    for index in range(episodes):
        replay = collect_replay(policy, seed=seed + index, profile=profile)
        replay.update(schema=REPLAY_SCHEMA, release=RELEASE,
                      split=f"{RELEASE} / validation / {replay['profile']}")
        replay["audit"].update(
            schema="triad.balanced_replay_audit.v1", policy_schema=POLICY_SCHEMA,
            implementation_sha256_before=deepcopy(sources),
            implementation_sha256_after=deepcopy(sources),
            checkpoint_files_sha256_before=deepcopy(files),
            checkpoint_files_sha256_after=deepcopy(files),
            deterministic_rule="STOP probability >= 0.5, else best legal deployment")
        replays.append(replay)
    if (sources != implementation_fingerprints() or files != checkpoint_files(checkpoint)
            or weights != policy.weights_fingerprint()
            or rng != canonical_hash(policy.rng.bit_generator.state)):
        raise RuntimeError("Replay source/checkpoint/RNG drift; no output written")
    export_balanced_html(replays, output, checkpoint=f"{RELEASE} / {checkpoint.name}",
                         weights_sha256=weights)
    print(f"Saved {episodes} recorded {RELEASE} {profile} validation examples to {output}")
    return replays


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--profile", choices=("normal", "stress", "capability", "mixed"), default="capability")
    run_demo(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
