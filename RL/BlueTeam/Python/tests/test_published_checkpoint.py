"""Verify the portable benchmark artifacts without Unreal or network access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from triad_rl.placement_policy import DynamicPlacementPolicy
from triad_rl.rollout import aggregate_metrics


BLUE_TEAM = Path(__file__).resolve().parents[2]
CHECKPOINT = BLUE_TEAM / "Checkpoints" / "toy-210"


def test_published_checkpoint_matches_native_config_and_training_metadata():
    # Loading verifies the feature-contract hash, NPZ hash, parameter hash,
    # tensor dimensions, and finite optimizer state (allow_pickle=False).
    policy, metadata = DynamicPlacementPolicy.load_checkpoint(CHECKPOINT)
    fingerprint = "md5:" + hashlib.md5(
        (BLUE_TEAM / "DefaultTrainingConfig.json").read_bytes()
    ).hexdigest()

    assert metadata["details"]["environment"]["config_fingerprint"] == fingerprint
    assert metadata["details"]["trainer"]["episodes_completed"] == 210
    assert metadata["details"]["trainer"]["dry_run"] is False
    assert metadata["details"]["trainer"]["updates_enabled"] is True
    assert policy.parameter_hash() == metadata["parameter_sha256"]
    assert metadata["feature_contract"]["observation_schema"] == "triad.policy_observation.v4"

    config = json.loads((BLUE_TEAM / "DefaultTrainingConfig.json").read_text(encoding="utf-8"))
    assert len(config["SensorCandidates"]) == 5
    assert all(candidate["bAllowDynamicPosition"] for candidate in config["SensorCandidates"])


@pytest.mark.parametrize("filename,source_kind,deterministic", [
    ("initial-stochastic.json", "initial", False),
    ("trained-stochastic.json", "checkpoint", False),
    ("trained-deterministic.json", "checkpoint", True),
])
def test_published_evaluation_is_frozen_and_summary_matches_episodes(
    filename, source_kind, deterministic
):
    report = json.loads((BLUE_TEAM / "Results" / filename).read_text(encoding="utf-8"))
    checkpoint = json.loads((CHECKPOINT / "checkpoint.json").read_text(encoding="utf-8"))

    assert report["schema"] == "triad.dynamic_placement_evaluation.v1"
    assert report["training_performed"] is False
    assert report["source"]["kind"] == source_kind
    assert report["deterministic"] is deterministic
    assert report["parameter_sha256_before"] == report["parameter_sha256_after"]
    assert report["environment"]["config_fingerprint"] == (
        checkpoint["details"]["environment"]["config_fingerprint"]
    )
    if source_kind == "checkpoint":
        assert report["parameter_sha256_before"] == checkpoint["parameter_sha256"]
        assert report["source"]["metadata"]["arrays_sha256"] == checkpoint["arrays_sha256"]

    assert report["held_out_seed_count"] == len(report["episodes"]) == 50
    assert report["held_out_seed_start"] == 1_500_000_000
    assert [row["seed"] for row in report["episodes"]] == list(range(1_500_000_000, 1_500_000_050))
    recomputed = aggregate_metrics(report["episodes"])
    assert report["summary"].keys() == recomputed.keys()
    for key, value in recomputed.items():
        if isinstance(value, float):
            assert report["summary"][key] == pytest.approx(value)
        else:
            assert report["summary"][key] == value
