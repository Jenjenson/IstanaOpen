"""Temporal evaluation tests: hand-constructed cases, never pilot generation.

Endpoint-validation stubs are deliberate here; the trainer tests its own real
artifact contract. This suite tests evaluator ordering, exact pairing, dispatch,
frozen scoring, persistence and statistics without running any training.
"""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
import shutil

import numpy as np
import pytest

import evaluate_temporal as evaluator
from triad_rl import adaptive_env, robust_scenarios
from triad_rl.temporal_policy import TemporalPolicy
from test_temporal_env import scenario
from test_temporal_inputs import catalogue, config


def forbidden(*args, **kwargs):
    raise AssertionError("No scenario generator, ghost constructor, training or extra inference is allowed")


def _settings():
    training = {**evaluator.trainer.FIXED_TRAINING, "policy_seeds": [497, 498, 499],
                "episodes": 4, "batch_size": 2, "hidden_size": 2}
    validation = {**evaluator.VALIDATION_SETTINGS, "episodes_per_profile": 2,
                  "run_seeds": dict(zip(evaluator.PROFILES, (999400, 999401, 999402)))}
    return training, validation


def bindings(patch, fixture):
    patch.setattr(evaluator.trainer, "FIXED_TRAINING", fixture["training"])
    patch.setattr(evaluator, "VALIDATION_SETTINGS", fixture["validation"])
    patch.setattr(evaluator, "load_published_lineage", lambda: deepcopy(fixture["lineage"]))
    patch.setattr(evaluator.trainer, "validate_run", fixture["validate"])
    patch.setattr(adaptive_env, "generate_scenario", forbidden)
    patch.setattr(robust_scenarios, "make_case", forbidden)
    patch.setattr(adaptive_env.AdaptivePlacementEnv, "__init__", forbidden)
    patch.setattr(evaluator.trainer, "run_training", forbidden)
    patch.setattr(evaluator, "make_case", fixture["handmade_case"])


@pytest.fixture(scope="module")
def temporal_evaluation_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("temporal-handmade-evaluation")
    training, validation = _settings()
    protocol = {"schema": evaluator.trainer.PROTOCOL_SCHEMA, "experiment": "temporal-v6-pilot",
                "training": {**training, "scenario_seed_ranges": {
                    str(seed): {"start": seed * 1_000_000, "count": 4} for seed in training["policy_seeds"]}},
                "temporal_config": asdict(config()), "validation": validation, "scale_gate": deepcopy(evaluator.GATE_SPEC),
                "prior_publication": {"path": "Results/ranking-v5-pilot/artifact-manifest.json",
                                      "sha256": "a9d975e38dbdf8a5095c52e0767e2142ac5616dcd09d338c85247747517dd058"},
                "reserved_final_tests_unopened": {profile: {"start": 2 * 10**15 + (500 + index) * 1_000_000, "count": 200}
                                                  for index, profile in enumerate(evaluator.PROFILES)}}
    validation.update(methods=list(evaluator.method_names(protocol)), scenario_seed_ranges={profile: {
        "start": 10**15 + seed * 1_000_000, "count": 2} for profile, seed in validation["run_seeds"].items()})
    lineage = {"schema": "triad.temporal_published_lineage.v1", "consumed_seed_ranges": [],
               "reserved_final_seed_ranges": list(protocol["reserved_final_tests_unopened"].values()),
               "prior_pilot": {"publication_manifest": deepcopy(protocol["prior_publication"])}}
    runs, policies = {}, {}
    for seed in training["policy_seeds"]:
        runs[seed] = root / f"seed-{seed}"
        actor = TemporalPolicy(config(), seed=seed, hidden_size=2)
        actor.save(runs[seed] / "initialized")
        actor.save(runs[seed] / "last")
        policies[seed] = actor
        for name in evaluator.trainer.RUN_FILES:
            path = runs[seed] / name
            if not path.exists(): path.write_bytes(("TEST STUB, NOT TRAINING: " + name).encode())
    validated, generated = [], []

    def validate(path, protocol_path, *, lineage):
        seed = next(seed for seed, directory in runs.items() if directory == path)
        validated.append(seed)
        actor = deepcopy(policies[seed])
        return actor, {"weights_sha256": actor.weights_fingerprint(), "initialized_weights_sha256": actor.weights_fingerprint(),
                       "run_files_sha256": {name: evaluator._sha((path / name).read_bytes()) for name in evaluator.trainer.RUN_FILES},
                       "summary": {"policy_seed": seed, "completed_episodes": 4}, "verification_scope": "Handmade test stub only"}

    def handmade_case(seed, profile):
        interval = validation["scenario_seed_ranges"][profile]
        assert interval["start"] <= seed < interval["start"] + 2
        assert set(validated) == set(training["policy_seeds"]), "Every endpoint must be checked before any case"
        generated.append((seed, profile))
        case = scenario()
        case["seed"] = seed
        return case, catalogue(), {"profile": profile, "requested_profile": profile, "scenario_seed": seed, "scenario_filtered": False}

    fixture = {"root": root, "training": training, "validation": validation, "lineage": lineage, "runs": runs,
               "validate": validate, "handmade_case": handmade_case, "generated": generated, "validated": validated,
               "protocol": protocol, "protocol_path": root / "protocol.json", "output": root / "evaluation"}
    # Patches end before returning: no session/global monkeypatch leakage.
    with pytest.MonkeyPatch.context() as patch:
        bindings(patch, fixture)
        protocol["training_implementation_sha256"] = evaluator.trainer.source_provenance()
        protocol["evaluation_implementation_sha256"] = evaluator.source_provenance()
        fixture["protocol_path"].write_bytes(evaluator._bytes(protocol))
        fixture["aggregate"] = evaluator.run_evaluation(runs, protocol_path=fixture["protocol_path"], output=fixture["output"])
    return fixture


@pytest.fixture
def evaluated(temporal_evaluation_fixture, monkeypatch):
    bindings(monkeypatch, temporal_evaluation_fixture)
    return temporal_evaluation_fixture


def reports_at(output):
    return {profile: json.loads(gzip.decompress((output / f"validation-{profile}.json.gz").read_bytes()))
            for profile in evaluator.PROFILES}


def test_handmade_nine_method_integration_has_no_ghost_cases_and_bound_hashes(evaluated):
    assert len(evaluated["generated"]) == len(set(evaluated["generated"])) == 6
    assert set(evaluated["validated"]) == {497, 498, 499}
    assert set(path.name for path in evaluated["output"].iterdir()) == set(evaluator.EVALUATION_FILES)
    reports = reports_at(evaluated["output"])
    assert sum(len(method["episodes"]) for report in reports.values() for method in report["methods"].values()) == 54
    for profile, report in reports.items():
        assert len(report["methods"]) == 9
        identities = [[(row["seed"], row["scenario_sha256"]) for row in method["episodes"]] for method in report["methods"].values()]
        assert all(value == identities[0] for value in identities)
        for name, method in report["methods"].items():
            for row in method["episodes"]:
                assert row["scenario_sha256"] == evaluator.canonical_hash(row["scenario"])
                assert "evaluation_catalogue" in row["scenario"] and "curriculum_metadata" in row["scenario"]
                assert row["actions"] and row["metrics"]["invalid_actions"] == 0
                assert row["weights_sha256_before"] == row["weights_sha256_after"]
                if name not in ("greedy_public", "temporal_public"):
                    assert isinstance(row["weights_sha256_before"], str)
            assert method["metadata"]["temporal_public_features"] == name.startswith("temporal_")
    assert evaluated["aggregate"]["final_test_accessed"] is False
    assert set(evaluated["aggregate"]["scale_gate"]["comparators"]) == {"temporal_public", "greedy_public", "balanced_v3"}


def test_completed_resume_and_verifier_do_not_generate_infer_train_or_write(evaluated, monkeypatch):
    before = {path.name: path.read_bytes() for path in evaluated["output"].iterdir()}
    for name in ("make_case", "evaluate_robust_methods", "_new"):
        monkeypatch.setattr(evaluator, name, forbidden)
    monkeypatch.setattr(evaluator._PublicActor, "act", forbidden)
    monkeypatch.setattr(TemporalPolicy, "act", forbidden)
    monkeypatch.setattr(TemporalPolicy, "update", forbidden)
    result = evaluator.run_evaluation(evaluated["runs"], protocol_path=evaluated["protocol_path"], output=evaluated["output"], resume=True)
    verified = evaluator.verify_completed_evaluation(evaluated["runs"], protocol_path=evaluated["protocol_path"], output=evaluated["output"])
    assert result == verified == evaluated["aggregate"]
    assert before == {path.name: path.read_bytes() for path in evaluated["output"].iterdir()}


@pytest.mark.parametrize("mutation", ["inputs", "receipt", "compressed", "summary", "aggregate"])
def test_persisted_tamper_fails_before_any_sampling(evaluated, tmp_path, monkeypatch, mutation):
    output = tmp_path / "changed"
    shutil.copytree(evaluated["output"], output)
    if mutation == "inputs":
        path = output / "evaluation-inputs.json"
        data = json.loads(path.read_bytes()); data["weights_sha256"]["temporal_499"] = "0" * 64
        path.write_bytes(evaluator._bytes(data))
    elif mutation == "receipt":
        (output / "validation-stress.receipt.json").write_bytes(b"{}")
    elif mutation == "compressed":
        path = output / "validation-stress.json.gz"
        path.write_bytes(path.read_bytes() + b"changed")
    elif mutation == "aggregate":
        path = output / "aggregate.json"
        data = json.loads(path.read_bytes()); data["scale_gate"]["passed"] = not data["scale_gate"]["passed"]
        path.write_bytes(evaluator._bytes(data))
    else:
        path = output / "validation-stress.json.gz"
        report = json.loads(gzip.decompress(path.read_bytes()))
        report["methods"]["temporal_499"]["summary"]["episodes"] = 999
        data = gzip.compress(evaluator._bytes(report), mtime=0); path.write_bytes(data)
        receipt = json.loads((output / "validation-stress.receipt.json").read_bytes())
        receipt.update(sha256=evaluator._sha(data), bytes=len(data))
        (output / "validation-stress.receipt.json").write_bytes(evaluator._bytes(receipt))
    monkeypatch.setattr(evaluator, "make_case", forbidden)
    with pytest.raises(ValueError):
        evaluator.run_evaluation(evaluated["runs"], protocol_path=evaluated["protocol_path"], output=output, resume=True)


def test_late_profile_tamper_checked_before_missing_early_profile(evaluated, tmp_path, monkeypatch):
    output = tmp_path / "changed"
    shutil.copytree(evaluated["output"], output)
    for name in ("aggregate.json", "validation-normal.json.gz", "validation-normal.receipt.json"):
        (output / name).unlink()
    (output / "validation-stress.receipt.json").write_bytes(b"{}")
    monkeypatch.setattr(evaluator, "make_case", forbidden)
    with pytest.raises(ValueError, match="receipt"):
        evaluator.run_evaluation(evaluated["runs"], protocol_path=evaluated["protocol_path"], output=output, resume=True)


def test_failed_final_endpoint_preflight_never_generates_or_creates_output(evaluated, tmp_path, monkeypatch):
    called = []
    def incomplete(path, protocol_path, *, lineage):
        called.append(path)
        if len(called) == 3: raise ValueError("Endpoint incomplete")
        return evaluated["validate"](path, protocol_path, lineage=lineage)
    monkeypatch.setattr(evaluator.trainer, "validate_run", incomplete)
    monkeypatch.setattr(evaluator, "make_case", forbidden)
    with pytest.raises(ValueError, match="incomplete"):
        evaluator.run_evaluation(evaluated["runs"], protocol_path=evaluated["protocol_path"], output=tmp_path / "output")
    assert len(called) == 3 and not (tmp_path / "output").exists()


def synthetic():
    protocol = {"training": {"policy_seeds": [497, 498, 499]},
                "validation": {"bootstrap_samples": 2000, "bootstrap_seed": 73044}, "scale_gate": deepcopy(evaluator.GATE_SPEC)}
    reports = {}
    for profile in evaluator.PROFILES:
        methods = {}
        for name in evaluator.method_names(protocol):
            trained = name in evaluator.temporal_names(protocol)
            methods[name] = {"episodes": [{"seed": index, "scenario_sha256": str(index), "metrics": {
                "breached_fraction": .3 if trained else .5, "detected_fraction": .8 if trained else .7,
                "success": True, "return": 1. if trained else 0., "cost": 2.}} for index in range(4)]}
        reports[profile] = {"profile": profile, "stage": "validation", "methods": methods}
    return protocol, reports


def test_seed_average_then_paired_stratified_interval_matches_independent_bootstrap():
    protocol, reports = synthetic()
    offsets = {"temporal_497": [-.2, .1, .3, .4], "temporal_498": [.4, .3, .1, -.2], "temporal_499": [.2] * 4}
    for report in reports.values():
        for name, values in offsets.items():
            for row, delta in zip(report["methods"][name]["episodes"], values): row["metrics"]["breached_fraction"] = .5 - delta
    result = evaluator.aggregate_reports(reports, protocol)
    mean = np.mean(list(offsets.values()), axis=0)
    rng, bootstrap = np.random.default_rng(73044), np.zeros(2000)
    for _ in evaluator.PROFILES:
        bootstrap += mean[rng.integers(0, 4, size=(2000, 4))].mean(axis=1) / 3.
    actual = result["paired_stratified_differences"]["temporal_mean_minus_temporal_public"]["timely_fraction"]
    np.testing.assert_allclose([actual["difference"], actual["lower95"], actual["upper95"]],
                               [mean.mean(), *np.quantile(bootstrap, [.025, .975])], atol=1e-15)
    assert actual["pairs_per_profile"] == dict.fromkeys(evaluator.PROFILES, 4)
    assert len(result["paired_stratified_differences"]) == 4 * 6
    assert result["scale_gate"]["passed"] is True and result["scale_gate"]["not_policy_promotion"] is True


@pytest.mark.parametrize("mutation,check", [("small", "minimum_mean_timely_gain"), ("ci", "mean_timely_ci_lower_above_zero"),
    ("profile", "profile_timely_point_safeguard"), ("profile_detection", "profile_detection_point_safeguard"),
    ("seed", "individual_seed_timely_point_safeguard"), ("detection", "mean_detection_not_lower"),
    ("success", "mean_all_threat_success_not_lower"), ("return", "mean_return_not_lower")])
def test_every_predeclared_gate_guard(mutation, check):
    protocol, reports = synthetic()
    for profile, report in reports.items():
        for name in evaluator.temporal_names(protocol):
            for index, row in enumerate(report["methods"][name]["episodes"]):
                metrics = row["metrics"]
                if mutation == "small": metrics["breached_fraction"] = .495
                elif mutation == "ci": metrics["breached_fraction"] = 1. if index == 0 else .3
                elif mutation == "profile" and profile == "normal": metrics["breached_fraction"] = .52
                elif mutation == "profile_detection" and profile == "normal": metrics["detected_fraction"] = .68
                elif mutation == "seed" and name == "temporal_497": metrics["breached_fraction"] = .52
                elif mutation == "detection": metrics["detected_fraction"] = .69
                elif mutation == "success": metrics["success"] = False
                elif mutation == "return": metrics["return"] = -.1
    gate = evaluator.aggregate_reports(reports, protocol)["scale_gate"]
    assert gate["passed"] is False
    assert all(row["checks"][check] is False for row in gate["comparators"].values())


@pytest.mark.parametrize("reference", evaluator.GATE_SPEC["comparators"])
def test_all_three_controls_are_required(reference):
    protocol, reports = synthetic()
    for report in reports.values():
        for row in report["methods"][reference]["episodes"]: row["metrics"]["breached_fraction"] = .2
    gate = evaluator.aggregate_reports(reports, protocol)["scale_gate"]
    assert gate["passed"] is False and gate["comparators"][reference]["passed"] is False


@pytest.mark.parametrize("mutation", ["seed", "seed_type", "hash", "missing", "stage", "profile", "nonfinite"])
def test_unpaired_or_invalid_aggregation_rejected(mutation):
    protocol, reports = synthetic()
    report = reports["normal"]
    row = report["methods"]["temporal_499"]["episodes"][1]
    if mutation == "seed": row["seed"] = 100
    elif mutation == "seed_type": row["seed"] = True
    elif mutation == "hash": row["scenario_sha256"] = "different"
    elif mutation == "missing": del report["methods"]["temporal_499"]
    elif mutation == "stage": report["stage"] = "test"
    elif mutation == "profile": report["profile"] = "stress"
    else: row["metrics"]["return"] = float("nan")
    with pytest.raises(ValueError): evaluator.aggregate_reports(reports, protocol)


@pytest.mark.parametrize("mutation", ["sources", "reserved", "overlap", "settings"])
def test_protocol_lineage_or_source_drift_fails_before_sampling(evaluated, tmp_path, monkeypatch, mutation):
    protocol = deepcopy(evaluated["protocol"])
    if mutation == "sources": protocol["evaluation_implementation_sha256"]["evaluate_temporal.py"] = "0" * 64
    elif mutation == "reserved": protocol["reserved_final_tests_unopened"]["normal"]["start"] += 1
    elif mutation == "settings": protocol["validation"]["bootstrap_seed"] += 1
    else:
        lineage = deepcopy(evaluated["lineage"])
        lineage["consumed_seed_ranges"] = [deepcopy(protocol["validation"]["scenario_seed_ranges"]["normal"])]
        monkeypatch.setattr(evaluator, "load_published_lineage", lambda: lineage)
    path = tmp_path / "protocol.json"; path.write_bytes(evaluator._bytes(protocol))
    monkeypatch.setattr(evaluator, "make_case", forbidden)
    with pytest.raises(ValueError):
        evaluator.run_evaluation(evaluated["runs"], protocol_path=path, output=tmp_path / "output")
    assert not (tmp_path / "output").exists()
