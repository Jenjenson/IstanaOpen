"""Frozen-policy sensitivity probes remain public-input-only and reproducible."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_adaptivity as probe
from triad_rl.adaptive_env import AdaptivePlacementEnv
from triad_rl.adaptive_policy import AdaptivePolicy


def test_interventions_are_public_only_independent_and_keep_approved_sites():
    observation = AdaptivePlacementEnv(seed=8).reset(seed=8)
    state = observation["state"]
    original = deepcopy(state)
    variants = probe.public_variants(state)
    assert state == original
    assert set(variants) == {"direction", "emitter", "weather"}
    for left, right in variants.values():
        assert left["sites"] == right["sites"] == state["sites"]
        assert set(left) == set(right) == set(state)
        assert left["budget_remaining"] == right["budget_remaining"]
    left, right = variants["direction"]
    np.testing.assert_allclose(right["forecast"]["approach_weights"], np.roll(left["forecast"]["approach_weights"], 4))
    for a, b in zip(left["tracks"], right["tracks"]):
        assert b["position"][:2] == [-x for x in a["position"][:2]]
        assert b["velocity"][:2] == [-x for x in a["velocity"][:2]]
        assert b["position"][2] == a["position"][2]
    assert variants["emitter"][0]["forecast"]["emitter_probability"] == 0
    assert variants["emitter"][1]["forecast"]["emitter_probability"] == 1
    assert variants["weather"][0]["weather"]["rf_noise"] == variants["weather"][1]["weather"]["rf_noise"]
    assert state == original


@pytest.mark.parametrize("seed,episodes", [(0, 2), (2 * 10 ** 15, 2), (2 * 10 ** 15 - 1, 2),
                                          (probe.DEFAULT_PROBE_SEED, 0), (True, 2)])
def test_probes_refuse_training_and_final_test_bands(tmp_path, seed, episodes):
    with pytest.raises(ValueError, match="validation-only"):
        probe.run_probes(tmp_path / "unused", seed=seed, episodes=episodes)


def test_probe_reproducible_and_never_reads_truth_or_steps_simulation(tmp_path, monkeypatch):
    original_env = AdaptivePlacementEnv
    observation = original_env(seed=4).reset(seed=4)
    policy = AdaptivePolicy(observation["feature_names"], seed=4)
    policy.save(tmp_path / "checkpoint")
    fingerprint = policy.weights_fingerprint()

    class PublicOnlyEnvironment:
        def __init__(self, **kwargs):
            self._inner = original_env(**kwargs)

        def reset(self, **kwargs):
            return self._inner.reset(**kwargs)

        @property
        def scenario(self):
            raise AssertionError("Probe must not inspect hidden truth")

        def step(self, action):
            raise AssertionError("Recommendation probes must not run episode outcomes")

    monkeypatch.setattr(probe, "AdaptivePlacementEnv", PublicOnlyEnvironment)
    first = probe.run_probes(tmp_path / "checkpoint", episodes=3)
    second = probe.run_probes(tmp_path / "checkpoint", episodes=3)
    assert first == second
    assert first["weights_sha256"] == fingerprint
    assert not first["training_performed"] and not first["final_test_accessed"]
    assert len(first["episodes"]) == 9
    for variant in first["variants"].values():
        assert variant["count"] == 3
        assert 0 <= variant["sensor_change_rate"] <= 1
        assert 0 <= variant["mean_action_distribution_l1"] <= 2 + 1e-12
    assert AdaptivePolicy.load(tmp_path / "checkpoint").weights_fingerprint() == fingerprint
    assert probe.main(["--checkpoint", str(tmp_path / "checkpoint"), "--episodes", "2",
                       "--output", str(tmp_path / "probe.json")]) == 0
    saved = json.loads((tmp_path / "probe.json").read_text())
    assert saved["weights_sha256"] == fingerprint
    assert len(saved["episodes"]) == 6
