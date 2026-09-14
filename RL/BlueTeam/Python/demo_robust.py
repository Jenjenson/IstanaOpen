"""Record robust-v2 policy decisions and sensing outcomes in an offline replay.

The browser plays recorded simulator frames; it performs no policy inference.
Ground-truth trajectories and curriculum labels are visualization/audit data,
never policy observations. These validation examples are not final-test evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from demo_adaptive import export_html
from evaluate_robust import (
    PROFILES, checkpoint_files, implementation_fingerprints as evaluation_sources,
    validate_seed_range,
)
from triad_rl.adaptive_evaluation import canonical_hash, json_safe
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.robust_scenarios import RobustPlacementEnv


REPLAY_SCHEMA = "triad.robust_replay.v1"
RELEASE = "robust-v2"
DEFAULT_SEED = 1_000_990_600_000_000


def implementation_fingerprints() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {**evaluation_sources(), **{
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("demo_robust.py", "demo_adaptive.py", "adaptive_demo.html")}}


def _policy_state(policy: Any) -> dict[str, str | None]:
    state = {"weights_sha256": policy.weights_fingerprint(), "rng_sha256": None}
    if hasattr(policy, "rng"):
        state["rng_sha256"] = canonical_hash(policy.rng.bit_generator.state)
    return state


def collect_replay(policy: Any, *, seed: int, profile: str = "mixed") -> dict:
    """Run the actual actor on only public observations, then copy scored frames."""
    validate_seed_range(seed, 1, "validation", getattr(policy, "metadata", {}))
    if profile not in PROFILES:
        raise ValueError(f"Unknown robust profile: {profile}")
    sources_before = implementation_fingerprints()
    state_before = _policy_state(policy)
    env = RobustPlacementEnv(seed=seed, profile=profile)
    observation = env.reset(seed=seed)
    decisions, total_return = [], 0.0
    for _ in range(64):
        # Policy inputs are the same public adapter contract used for training
        # and recommendations. No env.scenario/case_metadata enters this call.
        public_before = deepcopy(observation["state"])
        action = policy.act(deepcopy(observation), deterministic=True)
        if (isinstance(action, bool) or not isinstance(action, (int, np.integer))
                or not 0 <= int(action) < len(observation["action_mask"])
                or not observation["action_mask"][int(action)]):
            raise ValueError("Replay policy returned an invalid or masked action")
        action = int(action)
        option = deepcopy(observation["options"][action])
        observation, reward, done, info = env.step(action)
        if not np.isfinite(reward):
            raise ValueError("Replay returned a nonfinite reward")
        decisions.append({"action_index": action, "action": option,
                          "reward": float(reward), "public_before": public_before})
        total_return += float(reward)
        if done:
            break
    else:
        raise RuntimeError("Robust replay exceeded the placement decision limit")
    sources_after = implementation_fingerprints()
    state_after = _policy_state(policy)
    if sources_before != sources_after:
        raise RuntimeError("Replay implementation changed during collection")
    if state_before != state_after:
        raise RuntimeError("Frozen replay policy weights or RNG changed during collection")
    if not info.get("frames"):
        raise ValueError("Scored replay did not contain frames")
    scenario = deepcopy(env.scenario)
    catalogue = deepcopy(env.catalogue)
    case_metadata = deepcopy(env.case_metadata)
    return json_safe({
        "schema": REPLAY_SCHEMA, "release": RELEASE, "seed": seed,
        "stage": "validation", "profile": case_metadata["profile"],
        "requested_profile": profile,
        # Existing offline template displays split in the selector/conditions.
        "split": f"{RELEASE} / validation / {case_metadata['profile']}",
        "scenario": scenario, "catalogue": catalogue,
        "case_metadata": case_metadata, "placements": deepcopy(env.placements),
        "decisions": decisions,
        "metrics": {**{key: deepcopy(value) for key, value in info.items() if key != "frames"},
                    "return": total_return},
        "frames": deepcopy(info["frames"]),
        "audit": {
            "schema": "triad.robust_replay_audit.v1",
            "recorded_simulator_replay": True, "browser_inference": False,
            "training_performed": False, "independent_final_test_evidence": False,
            "ground_truth_in_policy_observation": False,
            "scenario_catalogue_metadata_sha256": canonical_hash({
                "scenario": scenario, "catalogue": catalogue, "case_metadata": case_metadata}),
            "policy_state_before": state_before, "policy_state_after": state_after,
            "implementation_sha256_before": sources_before,
            "implementation_sha256_after": sources_after,
            "interpretation": "Initial sensor placement followed by recorded sensing; not sensor relocation, physical interception or real-world validation",
        },
    })


def export_robust_html(replays: list[dict], output: Path, *, checkpoint: str,
                       weights_sha256: str) -> None:
    """Reuse the existing escaped-data renderer without changing frozen assets."""
    if not replays:
        raise ValueError("At least one robust replay is required")
    for replay in replays:
        audit = replay.get("audit", {})
        if (replay.get("schema") != REPLAY_SCHEMA or replay.get("release") != RELEASE
                or replay.get("stage") != "validation"
                or audit.get("recorded_simulator_replay") is not True
                or audit.get("browser_inference") is not False
                or audit.get("policy_state_before", {}).get("weights_sha256") != weights_sha256
                or audit.get("policy_state_before") != audit.get("policy_state_after")
                or audit.get("implementation_sha256_before") != audit.get("implementation_sha256_after")
                or audit.get("scenario_catalogue_metadata_sha256") != canonical_hash({
                    "scenario": replay.get("scenario"), "catalogue": replay.get("catalogue"),
                    "case_metadata": replay.get("case_metadata")})):
            raise ValueError("Robust replay/checkpoint provenance does not match")
    # The outer triad.adaptive_demo.v1 envelope is the shared viewer contract;
    # each recorded case carries its new robust schema and full audit hashes.
    export_html(replays, output, checkpoint=f"{RELEASE} / {checkpoint}",
                weights_sha256=weights_sha256)


def run_demo(*, checkpoint: Path, output: Path, seed: int = DEFAULT_SEED,
             episodes: int = 6, profile: str = "mixed") -> list[dict]:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or not 1 <= episodes <= 50:
        raise ValueError("Demo episodes must be an integer between 1 and 50")
    validate_seed_range(seed, episodes, "validation")
    checkpoint, output = Path(checkpoint), Path(output)
    if profile not in PROFILES:
        raise ValueError(f"Unknown robust profile: {profile}")
    source_before = implementation_fingerprints()
    files_before = checkpoint_files(checkpoint)
    policy = AdaptivePolicy.load(checkpoint, feature_names=FEATURE_NAMES)
    validate_seed_range(seed, episodes, "validation", policy.metadata)
    state_before = _policy_state(policy)
    replays = [collect_replay(policy, seed=seed + index, profile=profile)
               for index in range(episodes)]
    if (source_before != implementation_fingerprints()
            or files_before != checkpoint_files(checkpoint)
            or state_before != _policy_state(policy)):
        raise RuntimeError("Replay source/checkpoint changed; refusing a mixed-version demo")
    for replay in replays:
        replay["audit"]["checkpoint_files_sha256_before"] = deepcopy(files_before)
        replay["audit"]["checkpoint_files_sha256_after"] = deepcopy(files_before)
    export_robust_html(replays, output, checkpoint=checkpoint.name,
                       weights_sha256=state_before["weights_sha256"])
    print(f"Saved {episodes} recorded {RELEASE} {profile} validation scenarios to {output}")
    return replays


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--profile", choices=PROFILES, default="mixed")
    run_demo(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
