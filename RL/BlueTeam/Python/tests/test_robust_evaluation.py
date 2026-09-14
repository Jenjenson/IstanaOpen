from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import evaluate_robust as robust
from triad_rl.adaptive_evaluation import RandomLegal, canonical_hash
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.adaptive_policy import AdaptivePolicy


VALIDATION_SEED = 1_000_888_000_000_000
BLUE_ROOT = Path(__file__).resolve().parents[2]


def observation():
    return {"options": [{"sensor_id": "rf", "position": [0., 100.]},
                        {"sensor_id": None, "position": [0., 0.], "stop": True}],
            "action_mask": np.ones(2, dtype=bool),
            "option_features": np.zeros((2, len(FEATURE_NAMES))),
            "feature_names": FEATURE_NAMES,
            "catalogue": [{"id": "rf", "cost": .8, "ranges": {"rf": 130.}}],
            "state": {"budget_total": 3., "placements": []}}


class OneStepRobustEnv:
    def __init__(self, *, seed, profile):
        self.seed = seed
        self.scenario = {"seed": seed, "weather": {"visibility": .8},
                         "targets": [{"altitude": 40., "speed": 10., "emitter_duty": .8}]}
        self.catalogue = observation()["catalogue"]
        self.case_metadata = {"profile": profile, "private_marker": "must_not_reach_actor"}

    def reset(self, *, seed):
        assert seed == self.seed
        return observation()

    def step(self, action):
        score = float(np.random.default_rng(self.seed).random())
        return observation(), score, True, {"success": score > .5, "coverage": score,
                                             "placements": [], "frames": [{"time": 0.}]}


def test_recursive_transfer_and_selection_seed_ranges_are_all_reserved():
    provenance = {"training_state": {"seed_provenance": {
        "training": {"start": 123, "count": 20},
        "validation": {"normal": {"start": VALIDATION_SEED, "count": 3}},
        "inherited_training": {"selection": [{"start": VALIDATION_SEED + 10, "count": 5}],
                               "source": {"validation": {"start": VALIDATION_SEED + 20, "count": 8}}}}}}
    ranges = robust.nested_seed_ranges(provenance)
    assert len(ranges) == 4
    for offset in (0, 2, 10, 14, 20, 27):
        with pytest.raises(ValueError, match="overlap"):
            robust.validate_seed_range(VALIDATION_SEED + offset, 1, "validation", provenance)
    robust.validate_seed_range(VALIDATION_SEED + 28, 2, "validation", provenance)


@pytest.mark.parametrize("entry", [
    {"start": 10}, {"count": 5}, {"start": True, "count": 2},
    {"start": -1, "count": 2}, {"start": 10, "count": -1},
    {"start": 10, "count": 1.5},
])
def test_malformed_nested_provenance_fails_closed(entry):
    with pytest.raises(ValueError, match="seed range"):
        robust.nested_seed_ranges({"source": {"validation": [entry]}})


@pytest.mark.parametrize("seed,count,stage", [
    (10 ** 15 - 1, 1, "validation"), (2 * 10 ** 15 - 1, 2, "validation"),
    (2 * 10 ** 15, 1, "validation"), (10 ** 15, 0, "validation"),
    (True, 1, "validation"), (10 ** 15, 1, "test"),
    (2 * 10 ** 15 - 1, 1, "test"), (10 ** 15, 1, "heldout"),
])
def test_stage_and_whole_interval_bounds(seed, count, stage):
    # Pure validation only: these tests never instantiate final-test scenarios.
    with pytest.raises(ValueError):
        robust.validate_seed_range(seed, count, stage)


def test_old_consumed_final_tests_are_reserved_and_new_ranges_are_allowed():
    for start in (robust.TEST_BASE, robust.TEST_BASE + 10_000):
        for seed, count in ((start, 1), (start + 199, 1), (start - 1, 2)):
            with pytest.raises(ValueError):
                robust.validate_seed_range(seed, count, "test")
    # Merely validate reservation arithmetic; do not open any test episodes.
    robust.validate_seed_range(robust.TEST_BASE + 200, 9_800, "test")
    robust.validate_seed_range(robust.VALIDATION_BASE, 1, "validation")


def test_pairing_covers_catalogue_and_case_metadata_without_observation_leakage():
    class PublicOnlyActor(RandomLegal):
        def act(self, obs, deterministic=True):
            assert "scenario" not in obs and "curriculum_metadata" not in obs
            assert "private_marker" not in json.dumps(obs["state"])
            assert obs["catalogue"][0]["ranges"]["rf"] == 130.
            obs["state"]["placements"].append({"sensor_id": "should_not_persist"})
            return 1

    report = robust.evaluate_robust_methods(
        {"adaptive": PublicOnlyActor, "baseline": RandomLegal}, episodes=3,
        seed=VALIDATION_SEED, profile="capability", stage="validation",
        env_factory=OneStepRobustEnv, bootstrap_samples=10)
    assert report["schema"] == "triad.robust_evaluation.v1"
    assert report["stage"] == report["split"] == "validation"
    assert report["protocol"]["test_tuning_allowed"] is True
    assert report["protocol"]["independent_final_test_evidence"] is False
    first, second = (report["methods"][name]["episodes"] for name in ("adaptive", "baseline"))
    assert [row["scenario_sha256"] for row in first] == [row["scenario_sha256"] for row in second]
    for row in first:
        assert row["scenario_sha256"] == canonical_hash(row["scenario"])
        assert row["scenario"]["evaluation_catalogue"] == observation()["catalogue"]
        assert row["scenario"]["curriculum_metadata"]["profile"] == "capability"
        assert row["placements"] == []
    assert report["methods"]["adaptive"]["subgroups"]["curriculum_profile"]["capability"]["episodes"] == 3
    assert report["implementation_sha256"] == report["implementation_sha256_after"]
    assert "triad_rl/robust_scenarios.py" in report["implementation_sha256"]
    assert "evaluate_robust.py" in report["implementation_sha256"]


@pytest.mark.parametrize("change", ["catalogue", "case_metadata"])
def test_different_capability_or_curriculum_with_same_truth_is_not_a_pair(change):
    counter = 0

    def broken_factory(*, seed, profile):
        nonlocal counter
        counter += 1
        env = OneStepRobustEnv(seed=seed, profile=profile)
        if change == "catalogue":
            env.catalogue[0]["ranges"]["rf"] += counter
        else:
            env.case_metadata["revision"] = counter
        return env

    with pytest.raises(RuntimeError, match="Paired scenario mismatch"):
        robust.evaluate_robust_methods({"adaptive": RandomLegal, "other": RandomLegal},
                                       seed=VALIDATION_SEED, episodes=1,
                                       env_factory=broken_factory, bootstrap_samples=5)


def test_frozen_weights_are_recorded_full_length_before_and_after(tmp_path):
    policy = AdaptivePolicy(FEATURE_NAMES, seed=7)
    path = tmp_path / "policy"
    policy.save(path)
    fingerprint = policy.weights_fingerprint()
    report = robust.evaluate_robust_methods({"adaptive": robust.frozen_factory(path, fingerprint)},
                                           seed=VALIDATION_SEED, episodes=2,
                                           env_factory=OneStepRobustEnv, bootstrap_samples=5)
    for row in report["methods"]["adaptive"]["episodes"]:
        assert row["weights_sha256_before"] == row["weights_sha256_after"] == fingerprint
        assert len(row["weights_sha256_before"]) == 64
    different = AdaptivePolicy(FEATURE_NAMES, seed=8)
    different.save(path)
    with pytest.raises(RuntimeError, match="changed between episodes"):
        robust.frozen_factory(path, fingerprint)(0)


def test_robust_source_drift_rejected(monkeypatch):
    count = 0

    def changed_source():
        nonlocal count
        count += 1
        return {"robust_scenarios.py": f"version-{count}"}

    monkeypatch.setattr(robust, "implementation_fingerprints", changed_source)
    with pytest.raises(RuntimeError, match="Robust implementation changed"):
        robust.evaluate_robust_methods({"adaptive": RandomLegal},
                                       seed=VALIDATION_SEED, episodes=1,
                                       env_factory=OneStepRobustEnv, bootstrap_samples=5)


def test_actual_capability_validation_all_methods_and_explicit_pretrained_initializer(tmp_path):
    # New validation-only smoke cases: no final-test scenarios are opened.
    path = BLUE_ROOT / "Checkpoints" / "adaptive-v1"
    output = tmp_path / "validation.json"
    assert robust.main(["--checkpoint", str(path), "--initialized-checkpoint", str(path),
                        "--seed", str(VALIDATION_SEED + 100), "--episodes", "2",
                        "--profile", "capability", "--stage", "validation",
                        "--bootstrap-samples", "10", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report["methods"]) == {
        "adaptive", "adaptive_v1", "initialized", "greedy_public", "random_legal",
        "uniform_fixed_rf_radar", "legacy_toy210_projected"}
    for method in report["methods"].values():
        assert method["summary"]["episodes"] == 2
        assert method["summary"]["mean_invalid_actions"] == 0
    assert report["checkpoint_files_sha256_before"] == report["checkpoint_files_sha256_after"]
    init = report["methods"]["initialized"]["metadata"]
    assert init["is_claimed_untrained"] is False
    assert init["same_weights_as_published_v1"] is True
    assert init["recorded_source_completed_episodes"] == 4000
    assert len(report["published_v1_selection_sha256"]) == 64
    assert report["protocol"]["independent_final_test_evidence"] is False


def test_cli_requires_explicit_initializer_and_rejects_consumed_range_without_output(tmp_path):
    output = tmp_path / "not-created.json"
    with pytest.raises(SystemExit):
        robust.main(["--checkpoint", "unused", "--seed", str(VALIDATION_SEED), "--output", str(output)])
    with pytest.raises(ValueError, match="overlap"):
        robust.main(["--checkpoint", "unused", "--initialized-checkpoint", "unused",
                     "--seed", str(robust.TEST_BASE), "--stage", "test", "--output", str(output)])
    assert not output.exists()


def test_cli_verifies_initializer_matches_robust_candidate(tmp_path):
    policy = AdaptivePolicy(FEATURE_NAMES, seed=11)
    path = tmp_path / "candidate"
    policy.save(path, training_state={"schema": "triad.robust_training.v1",
                                     "initialization": {"weights_sha256": "not-the-supplied-source"}})
    with pytest.raises(ValueError, match="pre-training weights"):
        robust.main(["--checkpoint", str(path), "--initialized-checkpoint",
                     str(BLUE_ROOT / "Checkpoints" / "adaptive-v1"),
                     "--seed", str(VALIDATION_SEED), "--episodes", "1",
                     "--output", str(tmp_path / "not-created.json")])


def test_cli_reserves_published_common_selection_not_only_checkpoint_train_seeds(tmp_path):
    source = BLUE_ROOT / "Checkpoints" / "adaptive-v1"
    selection = json.loads((BLUE_ROOT / "Results" / "adaptive-v1" / "common-validation-selection.json").read_text())
    ranges = robust.nested_seed_ranges(selection["seed_provenance"])
    used = next(entry for name, entry in ranges.items() if "validation" in name and entry["count"] == 300)
    with pytest.raises(ValueError, match="overlap"):
        robust.main(["--checkpoint", str(source), "--initialized-checkpoint", str(source),
                     "--seed", str(used["start"]), "--episodes", "1",
                     "--output", str(tmp_path / "not-created.json")])


def selection_report():
    return {"schema": "triad.robust_model_selection.v1", "stage": "validation",
            "final_test_accessed": False, "selected": {"seed": 101, "weights_sha256": robust.V1_WEIGHTS},
            "seed_provenance": {"selection_validation": [
                {"profile": name, "start": VALIDATION_SEED + index * 1000, "count": 100}
                for index, name in enumerate(("normal", "stress", "capability"))],
                "lineage": {"other_candidate": {"validation": {"start": VALIDATION_SEED + 5_000, "count": 20}}}}}


def test_selection_binds_weights_and_reserves_all_candidates_and_profiles(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection_report()), encoding="utf-8")
    provenance, evidence, raw = robust.read_selection_report(path, robust.V1_WEIGHTS)
    assert raw == path.read_bytes()
    assert evidence["selected"]["weights_sha256"] == robust.V1_WEIGHTS
    assert len(evidence["sha256"]) == 64
    for offset in (0, 99, 1000, 2099, 5000, 5019):
        with pytest.raises(ValueError, match="overlap"):
            robust.validate_seed_range(VALIDATION_SEED + offset, 1, "validation", provenance)
    with pytest.raises(ValueError, match="weights"):
        robust.read_selection_report(path, "different-policy")


@pytest.mark.parametrize("mutation", [
    lambda report: report.update(stage="test"),
    lambda report: report.update(final_test_accessed=True),
    lambda report: report.update(final_test_accessed=0),
    lambda report: report["seed_provenance"]["selection_validation"].pop(),
    lambda report: report["seed_provenance"]["selection_validation"][0].update(count=0),
    lambda report: report["seed_provenance"]["selection_validation"][0].update(start=robust.TEST_BASE),
    lambda report: report["seed_provenance"]["lineage"]["other_candidate"]["validation"].update(count=-1),
])
def test_selection_rejects_test_evidence_or_malformed_nested_ranges(tmp_path, mutation):
    report = selection_report()
    mutation(report)
    path = tmp_path / "bad-selection.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        robust.read_selection_report(path, robust.V1_WEIGHTS)


def test_cli_selection_overlap_fails_before_running_scenario(tmp_path):
    source = BLUE_ROOT / "Checkpoints" / "adaptive-v1"
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(selection_report()), encoding="utf-8")
    output = tmp_path / "not-created.json"
    with pytest.raises(ValueError, match="overlap"):
        robust.main(["--checkpoint", str(source), "--initialized-checkpoint", str(source),
                     "--selection-report", str(selection), "--seed", str(VALIDATION_SEED),
                     "--episodes", "1", "--output", str(output)])
    assert not output.exists()
