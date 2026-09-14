"""Bounded fixture-only evaluation; never sample pilot or final-test slots."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from triad_rl import evaluate_credit_pilot as evaluator
from triad_rl import train_credit_pilot as trainer


def protocol_value():
    value = json.loads((evaluator.BLUE_ROOT / "Results/credit-v4-pilot/protocol.json").read_text())
    value["training"].update(policy_seed=499, scenario_seed_start=499_000_000, episodes=4, batch_size=2)
    value["validation"].update(episodes_per_profile=2,
                               run_seeds=dict(zip(evaluator.PROFILES, (999100, 999101, 999102))))
    value["validation"]["scenario_seed_ranges"] = {
        profile: {"start": evaluator.scenario_seed("validation", seed, 0), "count": 2}
        for profile, seed in value["validation"]["run_seeds"].items()}
    value["training_implementation_sha256"] = trainer.source_provenance()
    value["evaluation_implementation_sha256"] = evaluator.source_provenance()
    value["test_only"] = True
    return value


def write(path, value):
    path.write_bytes(evaluator._bytes(value))


def read(path):
    return json.loads(path.read_bytes())


def tree(path):
    return {str(p.relative_to(path)): evaluator._sha(p.read_bytes()) for p in path.rglob("*") if p.is_file()}


def synthetic(*, gain=.02, profile_gains=None, success_change=0., reference_gain=0.):
    reports = {}
    for profile in evaluator.PROFILES:
        methods = {}
        for method in evaluator.METHODS:
            gain_here = (profile_gains or {}).get(profile, gain) if method == "paired_stop" else 0.
            if method == "balanced_v3":
                gain_here = reference_gain
            success = bool(success_change >= 0 or method != "paired_stop")
            methods[method] = {"episodes": [{
                "seed": seed, "scenario_sha256": str(seed),
                "metrics": {"breached_fraction": .5 - gain_here, "success": success,
                            "detected_fraction": .7, "return": gain_here, "cost": 1.},
                "placements": [{"sensor": "fixture", "position": [1, 2]}]}
                for seed in range(12)]}
        reports[profile] = {"stage": "validation", "profile": profile, "methods": methods}
    return reports


def test_stratified_bootstrap_exactly_matches_independent_matrix_calculation():
    deltas = {"normal": np.array([0., 1.]), "stress": np.array([.25, -.25, .5]),
              "capability": np.array([0., 0., .5, 1.])}
    rng = np.random.default_rng(73041)
    expected = sum(values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1) / 3.
                   for values in deltas.values())
    actual = evaluator.stratified_interval(deltas)
    assert actual["difference"] == np.mean([x.mean() for x in deltas.values()])
    assert actual["lower95"] == np.quantile(expected, .025)
    assert actual["upper95"] == np.quantile(expected, .975)
    assert actual["pairs_per_profile"] == {"normal": 2, "stress": 3, "capability": 4}


@pytest.mark.parametrize("deltas", [{}, {p: [] for p in evaluator.PROFILES},
                                    {p: [np.nan] for p in evaluator.PROFILES},
                                    {p: [[1.]] for p in evaluator.PROFILES}])
def test_bootstrap_rejects_incomplete_nonfinite_or_nonscalar_pairs(deltas):
    with pytest.raises(ValueError):
        evaluator.stratified_interval(deltas)


def test_fixed_primary_gate_passes_only_both_references_and_reports_diagnostic_b():
    result = evaluator.aggregate_reports(synthetic(), protocol_value())
    assert result["scale_gate"]["passed"] is True
    assert result["scale_gate"]["decision"] == "eligible_for_multiseed_replication"
    assert result["scale_gate"]["not_policy_promotion"] is True
    comparisons = result["paired_stratified_differences"]
    assert len(comparisons) == 14
    assert comparisons["paired_stop_minus_no_critic_gradient"]["timely_fraction"]["difference"] == pytest.approx(.02)
    assert result["methods"]["paired_stop"]["equal_profile"]["timely_fraction"] == pytest.approx(.52)


@pytest.mark.parametrize("kwargs,failed_check", [
    ({"gain": .009}, "minimum_timely_gain"),
    ({"gain": 0.}, "timely_ci_lower_above_zero"),
    ({"profile_gains": {"normal": .1, "stress": -.011, "capability": .1}}, "no_profile_timely_decline_over_margin"),
    ({"success_change": -1.}, "all_threat_success_not_lower"),
])
def test_each_sensing_gate_condition_is_required(kwargs, failed_check):
    result = evaluator.aggregate_reports(synthetic(**kwargs), protocol_value())
    assert result["scale_gate"]["passed"] is False
    assert result["scale_gate"]["comparators"]["shared_critic"]["checks"][failed_check] is False


def test_cannot_win_only_one_reference_or_choose_diagnostic_b_as_primary():
    reports = synthetic(gain=.02, reference_gain=.04)
    for profile in evaluator.PROFILES:
        for row in reports[profile]["methods"]["no_critic_gradient"]["episodes"]:
            row["metrics"]["breached_fraction"] = 0.
    result = evaluator.aggregate_reports(reports, protocol_value())
    assert result["scale_gate"]["comparators"]["shared_critic"]["passed"] is True
    assert result["scale_gate"]["comparators"]["balanced_v3"]["passed"] is False
    assert result["scale_gate"]["primary_arm"] == "paired_stop"
    assert result["scale_gate"]["passed"] is False


@pytest.mark.parametrize("change", ["stage", "profile", "seed", "scenario", "method", "success_type", "finite"])
def test_aggregate_rejects_wrong_scope_unpaired_or_invalid_metrics(change):
    reports = synthetic()
    report = reports["normal"]
    row = report["methods"]["paired_stop"]["episodes"][0]
    if change == "stage": report["stage"] = "final_test"
    elif change == "profile": report["profile"] = "stress"
    elif change == "seed": row["seed"] += 1
    elif change == "scenario": row["scenario_sha256"] = "changed"
    elif change == "method": report["methods"].pop("adaptive_v1")
    elif change == "success_type": row["metrics"]["success"] = 1
    else: row["metrics"]["return"] = np.inf
    with pytest.raises(ValueError):
        evaluator.aggregate_reports(reports, protocol_value())


def test_layout_groups_explicitly_descriptive_and_partition_cases():
    reports = synthetic()
    records = reports["normal"]["methods"]["paired_stop"]["episodes"]
    records[0]["placements"] = []
    records[1]["placements"].append({"sensor": "later"})
    records[2]["placements"] = [{"sensor": "other_first"}]
    result = evaluator.aggregate_reports(reports, protocol_value())["exploratory_layout_groups"]
    groups = result["comparisons"]["paired_stop_minus_shared_critic"]["normal"]
    assert {k: v["pairs"] for k, v in groups.items()} == {
        "initial_stop_flip": 1, "same_first_placement_changed_later": 1,
        "changed_first_placement": 1, "unchanged_layout": 9}
    assert "not causal" in result["scope"]


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    root = tmp_path_factory.mktemp("credit-evaluation-fixture")
    declaration = root / "protocol.json"
    write(declaration, protocol_value())
    lineage = evaluator.load_published_lineage()
    patches = pytest.MonkeyPatch()
    patches.setattr(evaluator, "load_published_lineage", lambda: deepcopy(lineage))
    patches.setattr(trainer, "load_published_lineage", lambda: deepcopy(lineage))
    runs = {arm: root / arm for arm in evaluator.ARMS}
    for arm, directory in runs.items():
        trainer.run_training(declaration, arm, directory)
    before = {arm: tree(path) for arm, path in runs.items()}
    output = root / "evaluation"
    result = evaluator.run_evaluation(runs, protocol_path=declaration, output=output)
    assert before == {arm: tree(path) for arm, path in runs.items()}
    yield root, declaration, runs, output, result
    patches.undo()


def test_real_short_runs_evaluate_all_paired_fixture_profiles_and_bind_evidence(actual):
    root, declaration, runs, output, result = actual
    assert result["stage"] == "validation" and result["final_test_accessed"] is False
    assert len(list(output.iterdir())) == 8
    inputs = read(output / "evaluation-inputs.json")
    assert len({v["first_batch_trajectory_sha256"] for v in inputs["endpoint_evidence"].values()}) == 1
    assert inputs["implementation_sha256"] == evaluator.source_provenance()
    assert result["protocol_sha256"] == evaluator._sha(declaration.read_bytes())
    for profile in evaluator.PROFILES:
        report = json.loads(gzip.decompress((output / f"validation-{profile}.json.gz").read_bytes()))
        expected = inputs["validation_ranges"][profile]
        assert 1_000_999_100_000_000 <= expected["start"] <= 1_000_999_102_000_000
        assert set(report["methods"]) == set(evaluator.METHODS)
        for method in report["methods"].values():
            assert [row["seed"] for row in method["episodes"]] == list(range(expected["start"], expected["start"] + 2))
        assert report["seed_provenance"]["declared_lineage"] == inputs["lineage"]


def test_resume_completed_outputs_is_byte_exact_and_never_samples(actual, monkeypatch):
    _, declaration, runs, output, result = actual
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", lambda *a, **k: pytest.fail("resume sampled"))
    before = tree(output)
    assert evaluator.run_evaluation(runs, protocol_path=declaration, output=output, resume=True) == result
    assert tree(output) == before
    with pytest.raises(ValueError, match="Explicit resume"):
        evaluator.run_evaluation(runs, protocol_path=declaration, output=output)


@pytest.mark.parametrize("corrupt", ["bytes", "receipt_type", "scenario", "lineage", "summary", "seed_type"])
def test_all_existing_caches_prevalidated_before_any_missing_profile_sampling(actual, tmp_path, monkeypatch, corrupt):
    _, declaration, runs, original, _ = actual
    output = tmp_path / "copy"
    output.mkdir()
    for name in ("evaluation-inputs.json", "validation-stress.json.gz", "validation-stress.receipt.json"):
        shutil.copyfile(original / name, output / name)
    data_path, receipt_path = output / "validation-stress.json.gz", output / "validation-stress.receipt.json"
    data, receipt = data_path.read_bytes(), read(receipt_path)
    if corrupt == "bytes":
        data_path.write_bytes(data + b"tamper")
    elif corrupt == "receipt_type":
        receipt["bytes"] = float(receipt["bytes"])
        write(receipt_path, receipt)
    else:
        report = json.loads(gzip.decompress(data))
        method = report["methods"]["paired_stop"]
        if corrupt == "scenario": method["episodes"][0]["scenario"]["injected"] = True
        elif corrupt == "lineage": report["seed_provenance"]["declared_lineage"] = {}
        elif corrupt == "summary": method["summary"]["mean_return"] += 1.
        else: method["episodes"][0]["seed"] = float(method["episodes"][0]["seed"])
        data = gzip.compress(evaluator._bytes(report), mtime=0)
        data_path.write_bytes(data)
        receipt.update(sha256=evaluator._sha(data), bytes=len(data))
        write(receipt_path, receipt)
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", lambda *a, **k: pytest.fail("sampled before cache preflight"))
    before = tree(output)
    with pytest.raises(ValueError):
        evaluator.run_evaluation(runs, protocol_path=declaration, output=output, resume=True)
    assert tree(output) == before


@pytest.mark.parametrize("change", ["evaluation_source", "reservation", "validation_band", "validation_overlap",
                                    "validation_count_type", "run_seed_type", "gate_type", "primary"])
def test_invalid_protocol_fails_before_sampling_or_output(actual, tmp_path, monkeypatch, change):
    _, original, runs, _, _ = actual
    value = read(original)
    if change == "evaluation_source": value["evaluation_implementation_sha256"]["triad_rl/evaluate_credit_pilot.py"] = "0" * 64
    elif change == "reservation": value["reserved_final_tests_unopened"]["normal"]["count"] += 1
    elif change == "validation_band": value["validation"]["scenario_seed_ranges"]["normal"]["start"] = 2 * 10**15
    elif change == "validation_overlap":
        value["validation"]["run_seeds"]["normal"] = value["validation"]["run_seeds"]["stress"]
        value["validation"]["scenario_seed_ranges"]["normal"] = deepcopy(value["validation"]["scenario_seed_ranges"]["stress"])
    elif change == "validation_count_type": value["validation"]["episodes_per_profile"] = 2.
    elif change == "run_seed_type": value["validation"]["run_seeds"]["normal"] = 999100.
    elif change == "gate_type": value["scale_gate"]["require_paired_ci_lower_above_zero"] = 1
    else: value["scale_gate"]["primary_arm"] = "no_critic_gradient"
    declaration = tmp_path / "protocol.json"
    write(declaration, value)
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", lambda *a, **k: pytest.fail("invalid protocol sampled"))
    with pytest.raises(ValueError):
        evaluator.run_evaluation(runs, protocol_path=declaration, output=tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_source_drift_after_protocol_preflight_fails_before_output(actual, tmp_path, monkeypatch):
    _, declaration, runs, _, _ = actual
    original, count = evaluator.source_provenance, 0
    def changed():
        nonlocal count
        count += 1
        value = original()
        if count >= 2:
            value["triad_rl/evaluate_credit_pilot.py"] = "changed"
        return value
    monkeypatch.setattr(evaluator, "source_provenance", changed)
    with pytest.raises(ValueError, match="changed during preflight"):
        evaluator.run_evaluation(runs, protocol_path=declaration, output=tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("name", ["summary.json", "training.jsonl", "last/arrays.npz", "protocol.json"])
def test_tampered_endpoint_never_samples_or_creates_output(actual, tmp_path, monkeypatch, name):
    _, declaration, runs, _, _ = actual
    copied = tmp_path / "tampered"
    shutil.copytree(runs["paired_stop"], copied)
    target = copied / name
    target.write_bytes(target.read_bytes() + (b" " if name.endswith(".json") else b"tamper"))
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", lambda *a, **k: pytest.fail("tampered endpoint sampled"))
    with pytest.raises((ValueError, EOFError)):
        evaluator.run_evaluation({**runs, "paired_stop": copied}, protocol_path=declaration, output=tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()
