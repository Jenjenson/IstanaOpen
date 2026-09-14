import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import recommend_ranking
from recommend_adaptive import recommend_layout
from triad_rl.adaptive_evaluation import canonical_hash
from triad_rl.adaptive_inputs import FEATURE_NAMES, LiveObservationAdapter
from triad_rl.ranking_policy import RankPolicy, POLICY_SCHEMA
from triad_rl.robust_scenarios import RobustPlacementEnv


@pytest.mark.parametrize("profile", ["normal", "stress", "capability"])
@pytest.mark.parametrize("learned", [False, True])
def test_simulated_and_external_deployments_are_exactly_equal(profile, learned):
    policy = RankPolicy(FEATURE_NAMES, seed=499)
    if learned:
        policy.parameters["wa"][:] = np.linspace(-.03, .03, policy.hidden_size)
    env = RobustPlacementEnv(seed=499000000, profile=profile)
    observation = env.reset(seed=499000000)
    public = copy.deepcopy(observation["state"])
    before = canonical_hash(public)
    rng = canonical_hash(policy.rng.bit_generator.state)
    adapter = LiveObservationAdapter(env.catalogue)
    external = adapter.observe(public)
    np.testing.assert_array_equal(external["option_features"], observation["option_features"])
    plan = recommend_layout(policy, public, catalogue=env.catalogue)
    decisions = []
    while not env.done:
        action = policy.act(observation, deterministic=True)
        decisions.append(adapter.recommendation(observation, action))
        observation, _, _, _ = env.step(action)
    assert plan["decisions"][:len(decisions)] == decisions
    # The environment may auto-finish on exhausted resources. The offline plan
    # then records the adapter's explicit STOP without adding a placement.
    assert all(row["stop"] for row in plan["decisions"][len(decisions):])
    assert plan["new_placements"] == [row for row in decisions if not row["stop"]]
    assert plan["physical_commands_sent"] is False
    assert canonical_hash(public) == before
    assert canonical_hash(policy.rng.bit_generator.state) == rng


def test_cli_preserves_existing_plan_and_labels_version(tmp_path):
    policy = RankPolicy(FEATURE_NAMES, seed=499)
    policy.save(tmp_path / "checkpoint")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "plan.json"
    args = ["--checkpoint", str(tmp_path / "checkpoint"), "--input", str(root / "Examples/public-snapshot.json"),
            "--catalogue", str(root / "Examples/sensor-catalogue.json"), "--now", "0", "--output", str(output)]
    assert recommend_ranking.main(args) == 0
    raw = output.read_bytes()
    report = json.loads(raw)
    assert report["policy_schema"] == POLICY_SCHEMA
    assert report["physical_commands_sent"] is False
    assert report["checkpoint_weights_sha256"] == policy.weights_fingerprint()
    with pytest.raises(ValueError, match="never overwritten"):
        recommend_ranking.main(args)
    assert output.read_bytes() == raw


def test_inference_entrypoint_imports_no_simulator_or_rollout_labels():
    result = subprocess.run([sys.executable, "-c", "import recommend_ranking,sys; "
        "assert not any(n in sys.modules for n in ('triad_rl.adaptive_env','triad_rl.robust_scenarios',"
        "'triad_rl.ranking_rollout','triad_rl.train_ranking'))"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
