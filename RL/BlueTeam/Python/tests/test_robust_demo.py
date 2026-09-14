import json
from pathlib import Path

import pytest

import demo_robust as demo
from triad_rl.adaptive_evaluation import canonical_hash
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.robust_scenarios import RobustPlacementEnv


VALIDATION_SEED = 1_000_888_010_000_000
PUBLISHED = Path(__file__).resolve().parents[2] / "Checkpoints" / "adaptive-v1"


def embedded_data(path):
    html = path.read_text(encoding="utf-8")
    payload = html.split('<script id="replay-data" type="application/json">')[1].split("</script>")[0]
    return html, json.loads(payload)


def test_recorded_public_policy_decisions_match_actual_layout_frames_and_rewards():
    original = AdaptivePolicy.load(PUBLISHED)

    class PublicOnlyPolicy:
        rng = original.rng

        def weights_fingerprint(self):
            return original.weights_fingerprint()

        def act(self, observation, deterministic=True):
            assert deterministic is True
            assert "scenario" not in observation
            assert "case_metadata" not in observation
            assert "targets" not in observation["state"]
            return original.act(observation, deterministic=True)

    policy = PublicOnlyPolicy()
    before = canonical_hash(policy.rng.bit_generator.state)
    replay = demo.collect_replay(policy, seed=VALIDATION_SEED, profile="capability")
    env = RobustPlacementEnv(seed=VALIDATION_SEED, profile="capability")
    obs = env.reset(seed=VALIDATION_SEED)
    total = 0.
    for record in replay["decisions"]:
        action = original.act(obs, deterministic=True)
        assert action == record["action_index"]
        assert record["public_before"] == obs["state"]
        assert record["action"] == obs["options"][action]
        obs, reward, done, info = env.step(action)
        assert record["reward"] == reward
        total += reward
    assert done
    assert replay["placements"] == env.placements
    assert replay["frames"] == info["frames"]
    assert replay["metrics"]["return"] == total == info["return"]
    assert replay["metrics"]["success"] == info["success"]
    assert replay["metrics"]["reward_components"] == info["reward_components"]
    assert replay["catalogue"] == env.catalogue
    assert before == canonical_hash(policy.rng.bit_generator.state)
    assert replay["audit"]["policy_state_before"] == replay["audit"]["policy_state_after"]
    assert replay["audit"]["implementation_sha256_before"] == replay["audit"]["implementation_sha256_after"]
    assert replay["audit"]["scenario_catalogue_metadata_sha256"] == canonical_hash({
        "scenario": replay["scenario"], "catalogue": replay["catalogue"], "case_metadata": replay["case_metadata"]})


def test_capability_replays_record_heterogeneous_catalogues_and_sites_deterministically():
    policy = AdaptivePolicy(FEATURE_NAMES, seed=19)
    cases = [demo.collect_replay(policy, seed=VALIDATION_SEED + index, profile="capability")
             for index in range(3)]
    assert len({canonical_hash(case["catalogue"]) for case in cases}) == 3
    assert len({canonical_hash(case["scenario"]["public"]["sites"]) for case in cases}) == 3
    for index, case in enumerate(cases):
        assert case["case_metadata"]["synthetic_capability_variation"] is True
        assert case["profile"] == "capability"
        assert case == demo.collect_replay(policy, seed=VALIDATION_SEED + index, profile="capability")


@pytest.mark.parametrize("profile", ["normal", "stress", "mixed"])
def test_other_profiles_preserve_requested_and_resolved_labels(profile):
    policy = AdaptivePolicy(FEATURE_NAMES, seed=3)
    case = demo.collect_replay(policy, seed=VALIDATION_SEED + 20, profile=profile)
    assert case["requested_profile"] == profile
    assert case["profile"] == case["case_metadata"]["profile"]
    assert case["split"] == f"robust-v2 / validation / {case['profile']}"
    assert case["audit"]["independent_final_test_evidence"] is False
    assert case["audit"]["browser_inference"] is False
    assert case["frames"][0]["threats"]


def test_shared_viewer_escapes_robust_metadata_and_records_full_hashes(tmp_path):
    policy = AdaptivePolicy(FEATURE_NAMES, seed=5)
    replay = demo.collect_replay(policy, seed=VALIDATION_SEED, profile="capability")
    injection = '</script><script>alert("injection")</script>&'
    replay["case_metadata"]["note"] = injection
    replay["audit"]["scenario_catalogue_metadata_sha256"] = canonical_hash({
        "scenario": replay["scenario"], "catalogue": replay["catalogue"], "case_metadata": replay["case_metadata"]})
    target = tmp_path / "replay.html"
    demo.export_robust_html([replay], target, checkpoint=injection,
                            weights_sha256=policy.weights_fingerprint())
    html, payload = embedded_data(target)
    assert "<script>alert" not in html
    assert "fetch(" not in html
    assert "__REPLAY_DATA__" not in html
    assert payload["replays"] == [replay]
    assert payload["schema"] == "triad.adaptive_demo.v1"  # Shared viewer contract.
    assert payload["checkpoint"] == f"robust-v2 / {injection}"
    assert len(payload["weights_sha256"]) == 64
    assert payload["replays"][0]["schema"] == "triad.robust_replay.v1"
    assert 'id="components"' in html and 'id="placements"' in html
    assert "Recorded replay, not live inference" in html


def test_run_demo_with_frozen_published_checkpoint_keeps_files_unchanged(tmp_path):
    before = demo.checkpoint_files(PUBLISHED)
    target = tmp_path / "demo.html"
    replays = demo.run_demo(checkpoint=PUBLISHED, output=target,
                            seed=VALIDATION_SEED, episodes=2, profile="capability")
    _, payload = embedded_data(target)
    assert before == demo.checkpoint_files(PUBLISHED)
    assert replays == payload["replays"]
    for replay in replays:
        assert replay["audit"]["checkpoint_files_sha256_before"] == before
        assert replay["audit"]["checkpoint_files_sha256_after"] == before
        assert replay["audit"]["training_performed"] is False


def test_mutating_policy_is_rejected():
    policy = AdaptivePolicy(FEATURE_NAMES, seed=9)

    class MutatingPolicy:
        rng = policy.rng

        def weights_fingerprint(self):
            return policy.weights_fingerprint()

        def act(self, obs, deterministic=True):
            policy.parameters["wa"][0] += .01
            return len(obs["options"]) - 1

    with pytest.raises(RuntimeError, match="weights or RNG changed"):
        demo.collect_replay(MutatingPolicy(), seed=VALIDATION_SEED)


def test_source_drift_rejected(monkeypatch):
    count = 0

    def changed():
        nonlocal count
        count += 1
        return {"demo_robust.py": str(count)}

    monkeypatch.setattr(demo, "implementation_fingerprints", changed)
    with pytest.raises(RuntimeError, match="implementation changed"):
        demo.collect_replay(AdaptivePolicy(FEATURE_NAMES), seed=VALIDATION_SEED)


@pytest.mark.parametrize("seed,episodes", [
    (2 * 10 ** 15, 1), (10 ** 15 - 1, 1), (2 * 10 ** 15 - 1, 2),
    (VALIDATION_SEED, 0), (VALIDATION_SEED, 51),
])
def test_demo_rejects_test_seeds_and_invalid_counts_before_checkpoint_access(tmp_path, seed, episodes):
    target = tmp_path / "must-not-exist.html"
    with pytest.raises(ValueError):
        demo.run_demo(checkpoint=tmp_path / "unused", output=target, seed=seed, episodes=episodes)
    assert not target.exists()


def test_export_refuses_wrong_checkpoint_hash_or_nonfinite_values(tmp_path):
    policy = AdaptivePolicy(FEATURE_NAMES, seed=5)
    replay = demo.collect_replay(policy, seed=VALIDATION_SEED, profile="normal")
    target = tmp_path / "must-not-exist.html"
    with pytest.raises(ValueError, match="provenance"):
        demo.export_robust_html([replay], target, checkpoint="wrong", weights_sha256="wrong")
    replay["metrics"]["return"] = float("nan")
    with pytest.raises(ValueError):
        demo.export_robust_html([replay], target, checkpoint="policy", weights_sha256=policy.weights_fingerprint())
    assert not target.exists()
