import numpy as np
import pytest

import probe_balanced as probe
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.robust_scenarios import RobustPlacementEnv


SEED = 1_000_992_720_000_000


def test_probe_uses_gate_rule_not_joint_argmax():
    obs = RobustPlacementEnv(seed=SEED, profile="normal").reset(seed=SEED)
    policy = BalancedPolicy(FEATURE_NAMES)
    # Synthetic public-only actor explicitly demonstrates the rule distinction.
    p = np.zeros(len(obs["options"]))
    legal = np.flatnonzero(obs["action_mask"][:-1])
    assert len(legal) > 2
    p[legal] = .7 / len(legal)
    p[-1] = .3

    class GateActor:
        def probabilities(self, _):
            return p.copy()

        def act(self, _, deterministic=True):
            return int(legal[0])

    row = probe.first_action_record(GateActor(), obs, seed=SEED)
    assert row["joint_argmax_stop"]
    assert not row["deterministic_stop"]
    assert row["first_action"]["index"] == legal[0]


def test_probe_never_steps_or_reads_private_truth(tmp_path, monkeypatch):
    BalancedPolicy(FEATURE_NAMES).save(tmp_path / "actor")
    real = RobustPlacementEnv

    class PublicOnly:
        def __init__(self, **kwargs):
            self._reset = real(**kwargs).reset

        def reset(self, **kwargs):
            return self._reset(**kwargs)

        def __getattr__(self, key):
            raise AssertionError(f"Forbidden private access: {key}")

    monkeypatch.setattr(probe, "RobustPlacementEnv", PublicOnly)
    report = probe.run_probe(tmp_path / "actor", seed=SEED, episodes=3)
    assert len(report["records"]) == 3
    assert report["weights_sha256"] == report["weights_sha256_after"]
    assert not report["private_truth_accessed"]
    assert not report["simulation_rollouts_performed"]


@pytest.mark.parametrize("seed,episodes", [(2 * 10**15, 1), (10**15 - 1, 1), (SEED, 0)])
def test_no_test_seed_access(tmp_path, seed, episodes):
    with pytest.raises(ValueError):
        probe.run_probe(tmp_path / "missing", seed=seed, episodes=episodes)
