"""Paired multi-seed statistics and one isolated real fixed-endpoint evaluation."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import evaluate_ranking as evaluator
from triad_rl import train_ranking as trainer


def draft_protocol():
    return json.loads((evaluator.BLUE_ROOT / "Results/ranking-v5-pilot/protocol.json").read_text())


def fixture_bindings(patch, lineage):
    patch.setattr(evaluator, "load_published_lineage", lambda: deepcopy(lineage))
    patch.setattr(trainer, "load_published_lineage", lambda: deepcopy(lineage))


@pytest.fixture(scope="session")
def ranking_evaluation_fixture(tmp_path_factory):
    """No patches survive return; only499 and999300/1/2 test slots are sampled."""
    root = tmp_path_factory.mktemp("ranking-evaluation")
    protocol = draft_protocol()
    train = protocol["training"]
    train.update(policy_seeds=[499], episodes=4, batch_size=2, passes_per_batch=2,
                 scenario_seed_ranges={"499": {"start": 499000000, "count": 4}})
    validation = protocol["validation"]
    validation.update(episodes_per_profile=2, methods=list(evaluator.method_names(protocol)),
                      run_seeds={profile: 999300 + index for index, profile in enumerate(evaluator.PROFILES)})
    validation["scenario_seed_ranges"] = {profile: {
        "start": evaluator.scenario_seed("validation", seed, 0), "count": 2}
        for profile, seed in validation["run_seeds"].items()}
    prior = {key: protocol["prior_publication"][key] for key in ("path", "sha256")}
    lineage = {"schema": "triad.ranking_published_lineage.v1", "consumed_seed_ranges": [],
               "reserved_final_seed_ranges": sorted(protocol["reserved_final_tests_unopened"].values(), key=lambda row: row["start"]),
               "publication_manifests": [prior], "prior_pilot": {"publication_manifest": prior}}
    protocol["training_implementation_sha256"] = trainer.source_provenance()
    protocol["evaluation_implementation_sha256"] = evaluator.source_provenance()
    protocol_path, output, runs = root / "protocol.json", root / "evaluation", {499: root / "seed-499"}
    protocol_path.write_bytes(evaluator._bytes(protocol))
    with pytest.MonkeyPatch.context() as patch:
        fixture_bindings(patch, lineage)
        trainer.run_training(protocol_path, 499, runs[499])
        aggregate = evaluator.run_evaluation(runs, protocol_path=protocol_path, output=output)
    return {"root": root, "runs": runs, "protocol_path": protocol_path, "output": output,
            "aggregate": aggregate, "lineage": lineage}


@pytest.fixture
def evaluation(ranking_evaluation_fixture, monkeypatch):
    fixture_bindings(monkeypatch, ranking_evaluation_fixture["lineage"])
    return ranking_evaluation_fixture


def reports_at(output):
    return {profile: json.loads(gzip.decompress((output / f"validation-{profile}.json.gz").read_bytes()))
            for profile in evaluator.PROFILES}


def synthetic():
    protocol = draft_protocol()
    protocol["training"]["policy_seeds"] = [7, 8, 9]
    reports = {}
    for profile in evaluator.PROFILES:
        methods = {}
        for method in evaluator.method_names(protocol):
            ranker = method.startswith("ranker_")
            methods[method] = {"episodes": [{"seed": index, "scenario_sha256": str(index), "metrics": {
                "breached_fraction": .3 if ranker else .5,
                "detected_fraction": .8 if ranker else .7, "success": True,
                "return": 1. if ranker else 0., "cost": 2.}} for index in range(4)]}
        reports[profile] = {"methods": methods}
    return protocol, reports


def test_seed_mean_is_formed_within_case_before_bootstrap():
    protocol, reports = synthetic()
    offsets = {"ranker_7": [-.2, .1, .3, .4], "ranker_8": [.4, .3, .1, -.2], "ranker_9": [.2, .2, .2, .2]}
    for report in reports.values():
        for name, deltas in offsets.items():
            for row, delta in zip(report["methods"][name]["episodes"], deltas):
                row["metrics"]["breached_fraction"] = .5 - delta
    result = evaluator.aggregate_reports(reports, protocol)
    mean = np.mean(list(offsets.values()), axis=0)
    expected = evaluator.stratified_interval({profile: mean for profile in evaluator.PROFILES}, samples=2000, seed=73043)
    actual = result["paired_stratified_differences"]["ranker_mean_minus_greedy_public"]["timely_fraction"]
    assert evaluator.derived_same(actual, expected)
    assert actual["pairs_per_profile"] == dict.fromkeys(evaluator.PROFILES, 4)
    assert result["training_seeds"] == [7, 8, 9]
    assert all(name in result["methods"] for name in offsets)
    assert "not full training-population" in result["interval_scope"]
    assert result["scale_gate"]["passed"] is True
    assert result["scale_gate"]["not_policy_promotion"] is True


@pytest.mark.parametrize("mutation,check", [
    ("small_gain", "minimum_mean_timely_gain"), ("wide_ci", "mean_timely_ci_lower_above_zero"),
    ("profile_timely", "profile_timely_point_safeguard"),
    ("profile_detection", "profile_detection_point_safeguard"),
    ("individual", "individual_seed_timely_point_safeguard"),
    ("detection", "mean_detection_not_lower"), ("success", "mean_all_threat_success_not_lower"),
    ("return", "mean_return_not_lower")])
def test_each_predeclared_scaling_guard(mutation, check):
    protocol, reports = synthetic()
    for profile, report in reports.items():
        for name in evaluator.ranker_names(protocol):
            for index, row in enumerate(report["methods"][name]["episodes"]):
                metrics = row["metrics"]
                if mutation == "small_gain": metrics["breached_fraction"] = .495
                elif mutation == "wide_ci": metrics["breached_fraction"] = 1. if index == 0 else .3
                elif mutation == "profile_timely" and profile == "normal": metrics["breached_fraction"] = .52
                elif mutation == "profile_detection" and profile == "normal": metrics["detected_fraction"] = .68
                elif mutation == "individual" and name == "ranker_7": metrics["breached_fraction"] = .52
                elif mutation == "detection": metrics["detected_fraction"] = .69
                elif mutation == "success": metrics["success"] = False
                elif mutation == "return": metrics["return"] = -.1
    gate = evaluator.aggregate_reports(reports, protocol)["scale_gate"]
    assert gate["passed"] is False
    assert gate["comparators"]["greedy_public"]["checks"][check] is False


def test_both_comparators_and_all_individual_seeds_required():
    protocol, reports = synthetic()
    for report in reports.values():
        for row in report["methods"]["balanced_v3"]["episodes"]:
            row["metrics"]["breached_fraction"] = .1
    gate = evaluator.aggregate_reports(reports, protocol)["scale_gate"]
    assert gate["comparators"]["greedy_public"]["passed"] is True
    assert gate["comparators"]["balanced_v3"]["passed"] is False
    assert gate["passed"] is False
    reports["normal"]["methods"].pop("ranker_9")
    with pytest.raises(ValueError): evaluator.aggregate_reports(reports, protocol)


@pytest.mark.parametrize("mutation", ["pair", "probability", "boolean", "nan", "success"])
def test_pairing_and_numeric_bounds(mutation):
    protocol, reports = synthetic()
    row = reports["normal"]["methods"]["ranker_7"]["episodes"][0]
    if mutation == "pair": row["scenario_sha256"] = "different"
    elif mutation == "probability": row["metrics"]["detected_fraction"] = 1.01
    elif mutation == "boolean": row["metrics"]["return"] = True
    elif mutation == "nan": row["metrics"]["cost"] = float("nan")
    else: row["metrics"]["success"] = 1
    with pytest.raises(ValueError): evaluator.aggregate_reports(reports, protocol)


def test_real_fixture_complete_bound_schema_and_exact_greedy_control(evaluation):
    result = evaluation["aggregate"]
    reports = reports_at(evaluation["output"])
    protocol = json.loads(evaluation["protocol_path"].read_text())
    inputs = json.loads((evaluation["output"] / "evaluation-inputs.json").read_text())
    assert set(path.name for path in evaluation["output"].iterdir()) == set(evaluator.EVALUATION_FILES)
    assert result["schema"] == evaluator.AGGREGATE_SCHEMA
    assert result["training_seeds"] == [499] and result["final_test_accessed"] is False
    assert len(inputs["endpoint_evidence"]["499"]["run_files_sha256"]) == 9
    for profile, report in reports.items():
        assert report["schema"] == evaluator.REPORT_SCHEMA
        assert set(report["methods"]) == set(evaluator.method_names(protocol))
        assert len(report["methods"]) == 6
        assert all(len(method["episodes"]) == 2 for method in report["methods"].values())
        assert report["methods"]["greedy_public"]["metadata"]["initialized_weights_sha256"] == {
            "499": inputs["endpoint_evidence"]["499"]["initialized_weights_sha256"]}
        receipt = json.loads((evaluation["output"] / f"validation-{profile}.receipt.json").read_text())
        assert receipt["sha256"] == evaluator._sha((evaluation["output"] / f"validation-{profile}.json.gz").read_bytes())
    derived = evaluator.aggregate_reports(reports, protocol)
    assert result["scale_gate"] == derived["scale_gate"]


def forbidden(*args, **kwargs):
    raise AssertionError("No new scenarios, inference or filesystem writes permitted")


def test_completed_verifier_is_read_only_and_resume_never_resamples(evaluation, monkeypatch):
    output = evaluation["output"]
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", forbidden)
    monkeypatch.setattr(evaluator.RankPolicy, "act", forbidden)
    monkeypatch.setattr(evaluator.BalancedPolicy, "act", forbidden)
    monkeypatch.setattr(evaluator.AdaptivePolicy, "act", forbidden)
    monkeypatch.setattr(trainer, "collect_episode", forbidden)
    monkeypatch.setattr(trainer, "RobustPlacementEnv", forbidden)
    monkeypatch.setattr(evaluator, "_new", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    result = evaluator.verify_completed_evaluation(evaluation["runs"], protocol_path=evaluation["protocol_path"], output=output)
    assert result == evaluation["aggregate"]
    assert evaluator.run_evaluation(evaluation["runs"], protocol_path=evaluation["protocol_path"], output=output, resume=True) == result
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


@pytest.mark.parametrize("mutation", ["gate", "ci", "profile", "seed_summary", "boolean"])
def test_derived_aggregate_tampering_rejected(evaluation, tmp_path, mutation):
    output = tmp_path / "evaluation"
    shutil.copytree(evaluation["output"], output)
    result = json.loads((output / "aggregate.json").read_text())
    if mutation == "gate": result["scale_gate"]["passed"] = not result["scale_gate"]["passed"]
    elif mutation == "ci": result["paired_stratified_differences"]["ranker_mean_minus_greedy_public"]["timely_fraction"]["lower95"] += .1
    elif mutation == "profile": result["methods"]["ranker_mean"]["profiles"]["normal"]["timely_fraction"] += .1
    elif mutation == "seed_summary": result["methods"]["ranker_499"]["equal_profile"]["return"] += .1
    else: result["final_test_accessed"] = 0
    (output / "aggregate.json").write_bytes(evaluator._bytes(result))
    with pytest.raises(ValueError, match="aggregate"):
        evaluator.verify_completed_evaluation(evaluation["runs"], protocol_path=evaluation["protocol_path"], output=output)


@pytest.mark.parametrize("mutation", ["compressed", "rehash_metric", "pair", "receipt", "missing"])
def test_all_cached_profiles_validated_before_any_missing_profile_sampling(evaluation, tmp_path, monkeypatch, mutation):
    output = tmp_path / "evaluation"
    shutil.copytree(evaluation["output"], output)
    (output / "aggregate.json").unlink()
    (output / "validation-normal.json.gz").unlink()
    (output / "validation-normal.receipt.json").unlink()
    path, receipt_path = output / "validation-stress.json.gz", output / "validation-stress.receipt.json"
    if mutation == "compressed": path.write_bytes(path.read_bytes() + b"x")
    elif mutation == "missing": receipt_path.unlink()
    elif mutation == "receipt": receipt_path.write_bytes(b"{}")
    else:
        report = json.loads(gzip.decompress(path.read_bytes()))
        row = report["methods"]["ranker_499"]["episodes"][0]
        if mutation == "rehash_metric": row["metrics"]["detected_fraction"] = 2.
        else: row["scenario_sha256"] = "0" * 64
        data = gzip.compress(evaluator._bytes(report), mtime=0)
        path.write_bytes(data)
        receipt = json.loads(receipt_path.read_text())
        receipt.update(sha256=evaluator._sha(data), bytes=len(data))
        receipt_path.write_bytes(evaluator._bytes(receipt))
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", forbidden)
    with pytest.raises(ValueError):
        evaluator.run_evaluation(evaluation["runs"], protocol_path=evaluation["protocol_path"], output=output, resume=True)
    assert not (output / "validation-normal.json.gz").exists()


@pytest.mark.parametrize("mutation", ["sources", "reserved", "prior", "missing_seed", "run_bytes"])
def test_input_drift_rejected_before_output(evaluation, tmp_path, monkeypatch, mutation):
    protocol_path = tmp_path / "protocol.json"
    protocol = json.loads(evaluation["protocol_path"].read_text())
    runs = evaluation["runs"]
    if mutation == "sources": protocol["evaluation_implementation_sha256"] = {}
    elif mutation == "reserved": protocol["reserved_final_tests_unopened"]["normal"]["count"] += 1
    elif mutation == "prior": protocol["prior_publication"]["sha256"] = "0" * 64
    elif mutation == "missing_seed": runs = {}
    else:
        run = tmp_path / "run"
        shutil.copytree(runs[499], run)
        (run / "summary.json").write_bytes((run / "summary.json").read_bytes() + b" ")
        runs = {499: run}
    protocol_path.write_bytes(evaluator._bytes(protocol))
    output = tmp_path / "new-output"
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", forbidden)
    with pytest.raises(ValueError): evaluator.run_evaluation(runs, protocol_path=protocol_path, output=output)
    assert not output.exists()


def test_float_tolerance_does_not_relax_structure_discrete_or_hash_equality():
    assert evaluator.derived_same({"x": .1}, {"x": .1 + 1e-13})
    assert not evaluator.derived_same({"x": .1}, {"x": .10001})
    assert not evaluator.derived_same({"x": False}, {"x": 0})
    assert not evaluator.derived_same({"x": 1.}, {"x": 1})
    assert not evaluator.derived_same({"x": "a" * 64}, {"x": "b" * 64})
    assert not evaluator.derived_same({"x": []}, {"x": [], "y": 0})
