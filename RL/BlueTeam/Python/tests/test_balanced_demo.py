import json
from pathlib import Path

import numpy as np
import pytest

import demo_balanced as demo
import recommend_balanced
from recommend_adaptive import recommend_layout
from triad_rl.adaptive_evaluation import canonical_hash
from triad_rl.adaptive_inputs import FEATURE_NAMES, LiveObservationAdapter
from triad_rl.balanced_policy import BalancedPolicy, POLICY_SCHEMA
from triad_rl.robust_scenarios import RobustPlacementEnv


SEED = 1_000_992_710_000_000


def checkpoint(tmp_path):
    policy = BalancedPolicy(FEATURE_NAMES, seed=91)
    policy.save(tmp_path / "checkpoint")
    return tmp_path / "checkpoint", policy


def test_replay_uses_actual_balanced_decisions_and_scored_frames(tmp_path):
    path, policy = checkpoint(tmp_path)
    before_rng = canonical_hash(policy.rng.bit_generator.state)
    target = tmp_path / "demo.html"
    replays = demo.run_demo(checkpoint=path, output=target, seed=SEED,
                            episodes=2, profile="capability")
    html = target.read_text(encoding="utf-8")
    payload = json.loads(html.split('<script id="replay-data" type="application/json">')[1].split("</script>")[0])
    assert payload["replays"] == replays
    assert "Recorded replay, not live inference" in html
    assert "fetch(" not in html
    assert 'threatLabel([x, y], compact ? `${index + 1}`' in html
    assert 'text([x + 10, y - 9], compact ?' not in html
    assert len({canonical_hash(case["catalogue"]) for case in replays}) == 2
    for i, replay in enumerate(replays):
        assert replay["schema"] == "triad.balanced_replay.v1"
        assert replay["release"] == "balanced-v3"
        assert replay["audit"]["policy_schema"] == POLICY_SCHEMA
        assert not replay["audit"]["independent_final_test_evidence"]
        assert not replay["audit"]["ground_truth_in_policy_observation"]
        assert "triad_rl/balanced_policy.py" in replay["audit"]["implementation_sha256_before"]
        env = RobustPlacementEnv(seed=SEED + i, profile="capability")
        obs = env.reset(seed=SEED + i)
        total = 0.
        for decision in replay["decisions"]:
            action = policy.act(obs)
            assert decision["action_index"] == action
            assert decision["public_before"] == obs["state"]
            obs, reward, done, info = env.step(action)
            total += reward
        assert done
        assert replay["placements"] == env.placements
        assert replay["frames"] == info["frames"]
        assert replay["metrics"]["return"] == total
    assert canonical_hash(policy.rng.bit_generator.state) == before_rng


@pytest.mark.parametrize("profile", ["normal", "stress", "capability"])
def test_simulated_and_external_adapters_produce_identical_full_balanced_plan(profile):
    policy = BalancedPolicy(FEATURE_NAMES, seed=87)
    env = RobustPlacementEnv(seed=SEED, profile=profile)
    obs = env.reset(seed=SEED)
    adapter = LiveObservationAdapter(env.catalogue)
    external = adapter.observe(obs["state"], now=obs["state"]["timestamp"])
    np.testing.assert_array_equal(external["option_features"], obs["option_features"])
    plan = recommend_layout(policy, obs["state"], catalogue=env.catalogue)
    actions = []
    while not env.done:
        action = policy.act(obs)
        actions.append(adapter.recommendation(obs, action))
        obs, _, _, _ = env.step(action)
    assert plan["decisions"] == actions
    assert plan["physical_commands_sent"] is False


def test_recommendation_cli_labels_version_and_sends_no_commands(tmp_path):
    path, policy = checkpoint(tmp_path)
    root = Path(__file__).resolve().parents[2]
    target = tmp_path / "plan.json"
    assert recommend_balanced.main([
        "--checkpoint", str(path), "--input", str(root / "Examples/public-snapshot.json"),
        "--catalogue", str(root / "Examples/sensor-catalogue.json"),
        "--now", "0", "--output", str(target)]) == 0
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["policy_schema"] == POLICY_SCHEMA
    assert result["physical_commands_sent"] is False
    assert result["checkpoint_weights_sha256"] == policy.weights_fingerprint()


@pytest.mark.parametrize("seed,episodes", [(2 * 10**15, 1), (10**15 - 1, 1),
                                           (2 * 10**15 - 1, 2), (SEED, 0), (SEED, 51)])
def test_replay_rejects_test_seeds_or_bad_counts_before_loading(tmp_path, seed, episodes):
    target = tmp_path / "none.html"
    with pytest.raises(ValueError):
        demo.run_demo(checkpoint=tmp_path / "missing", output=target, seed=seed, episodes=episodes)
    assert not target.exists()


def test_balanced_source_drift_fails_without_output(tmp_path, monkeypatch):
    path, _ = checkpoint(tmp_path)
    target = tmp_path / "none.html"
    calls = 0

    def drifting():
        nonlocal calls
        calls += 1
        return {"triad_rl/balanced_policy.py": str(calls)}

    monkeypatch.setattr(demo, "implementation_fingerprints", drifting)
    with pytest.raises(RuntimeError, match="drift"):
        demo.run_demo(checkpoint=path, output=target, seed=SEED, episodes=1)
    assert not target.exists()


def test_balanced_label_export_escapes_script_termination(tmp_path):
    target = tmp_path / "escaped.html"
    hostile = '</script><script>alert("no")</script>&'
    demo.export_balanced_html([{"frames": [{}], "note": hostile}], target,
                              checkpoint=hostile, weights_sha256="test")
    html = target.read_text(encoding="utf-8")
    assert '<script>alert' not in html
    payload = json.loads(html.split('<script id="replay-data" type="application/json">')[1].split("</script>")[0])
    assert payload["checkpoint"] == hostile
    assert payload["replays"][0]["note"] == hostile
