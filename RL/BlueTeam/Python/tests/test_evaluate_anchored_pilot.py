"""Fixture-only anchored evaluation: pairing, fixed gates and fail-closed resume."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil

import pytest

from triad_rl import evaluate_anchored_pilot as evaluator
from triad_rl import train_anchored_pilot as trainer


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_bytes(evaluator._bytes(value))


def tree(path):
    return {file.relative_to(path).as_posix(): evaluator._sha(file.read_bytes())
            for file in Path(path).rglob("*") if file.is_file()}


def fixture_protocol():
    protocol = read(evaluator.BLUE_ROOT / "Results/anchored-v4-pilot/protocol.json")
    # Replace every production sampling declaration before calling either runner.
    protocol["training"].update(policy_seed=499, scenario_seed_start=499_000_000, episodes=4, batch_size=2,
                                 endpoint="Test fixture only: four episodes, no checkpoint selection.")
    protocol["validation"].update(episodes_per_profile=2,
                                   run_seeds=dict(zip(evaluator.PROFILES, (999200, 999201, 999202))))
    protocol["validation"]["scenario_seed_ranges"] = {
        profile: {"start": evaluator.scenario_seed("validation", seed, 0), "count": 2}
        for profile, seed in protocol["validation"]["run_seeds"].items()}
    protocol["training_implementation_sha256"] = trainer.source_provenance()
    protocol["evaluation_implementation_sha256"] = evaluator.source_provenance()
    protocol["test_only"] = True
    return protocol


def synthetic(*, gain=.02, reference_gain=0., profile_gains=None, reward_gain=0., success_loss=False):
    reports = {}
    for profile in evaluator.PROFILES:
        methods = {}
        for method in evaluator.METHODS:
            delta = (profile_gains or {}).get(profile, gain) if method == evaluator.PRIMARY else 0.
            if method == "balanced_v3":
                delta = reference_gain
            methods[method] = {"episodes": [{
                "seed": seed, "scenario_sha256": str(seed),
                "metrics": {"breached_fraction": .5-delta,
                            "success": not (success_loss and method == evaluator.PRIMARY),
                            "detected_fraction": .7, "return": reward_gain if method == evaluator.PRIMARY else 0., "cost": 1.},
                "placements": [{"sensor": "fixture", "position": [1, 2]}]}
                for seed in range(12)]}
        reports[profile] = {"stage": "validation", "profile": profile, "methods": methods}
    return reports


def test_predeclared_primary_pass_requires_both_references_and_eight_methods():
    result = evaluator.aggregate_reports(synthetic(), fixture_protocol())
    assert set(result["methods"]) == set(evaluator.METHODS) and len(result["methods"]) == 8
    assert result["scale_gate"]["primary_arm"] == "anchored_later_stop"
    assert result["scale_gate"]["passed"] is True
    assert set(result["scale_gate"]["comparators"]) == {"no_critic_gradient", "balanced_v3"}
    assert result["scale_gate"]["not_policy_promotion"] is True
    assert len(result["paired_stratified_differences"]) == 17
    assert result["paired_stratified_differences"]["anchored_later_stop_minus_all_step_stop_pilot"]["timely_fraction"]["difference"] == pytest.approx(.02)


@pytest.mark.parametrize("kwargs,check", [
    ({"gain": .009}, "minimum_timely_gain"),
    ({"gain": 0.}, "timely_ci_lower_above_zero"),
    ({"profile_gains": {"normal": .1, "stress": -.011, "capability": .1}}, "profile_point_safeguard"),
    ({"success_loss": True}, "all_threat_success_point_safeguard"),
    ({"reward_gain": -.000001}, "mean_return_point_safeguard"),
])
def test_each_sensing_success_profile_and_return_guard_is_required(kwargs, check):
    result = evaluator.aggregate_reports(synthetic(**kwargs), fixture_protocol())
    assert result["scale_gate"]["passed"] is False
    for comparator in result["scale_gate"]["comparators"].values():
        assert comparator["checks"][check] is False


def test_single_reference_win_and_better_diagnostic_arm_do_not_pass_primary():
    reports = synthetic(reference_gain=.04)
    for profile in evaluator.PROFILES:
        for row in reports[profile]["methods"]["shared_critic"]["episodes"]:
            row["metrics"]["breached_fraction"] = 0.
            row["metrics"]["return"] = 100.
    result = evaluator.aggregate_reports(reports, fixture_protocol())
    assert result["scale_gate"]["comparators"]["no_critic_gradient"]["passed"] is True
    assert result["scale_gate"]["comparators"]["balanced_v3"]["passed"] is False
    assert result["scale_gate"]["passed"] is False


@pytest.mark.parametrize("change", ["seed", "seed_type", "scenario", "missing", "stage", "profile"])
def test_aggregate_rejects_inexact_pairing_incomplete_methods_and_wrong_scope(change):
    reports = synthetic()
    report = reports["normal"]
    row = report["methods"][evaluator.PRIMARY]["episodes"][0]
    if change == "seed": row["seed"] += 1
    elif change == "seed_type": row["seed"] = float(row["seed"])
    elif change == "scenario": row["scenario_sha256"] = "changed"
    elif change == "missing": report["methods"].pop("all_step_stop_pilot")
    elif change == "stage": report["stage"] = "test"
    else: report["profile"] = "stress"
    with pytest.raises(ValueError):
        evaluator.aggregate_reports(reports, fixture_protocol())


@pytest.fixture(scope="session")
def anchored_evaluation_fixture(tmp_path_factory):
    """Shared immutable fixture for evaluator/archive tests; copy before tampering."""
    root = tmp_path_factory.mktemp("anchored-evaluation")
    protocol_path = root / "protocol.json"
    declaration = fixture_protocol()
    assert declaration["training"]["scenario_seed_start"] == 499_000_000
    assert list(declaration["validation"]["run_seeds"].values()) == [999200, 999201, 999202]
    write(protocol_path, declaration)
    lineage = evaluator.load_published_lineage()
    patches = pytest.MonkeyPatch()
    patches.setattr(evaluator, "load_published_lineage", lambda: deepcopy(lineage))
    patches.setattr(trainer, "load_published_lineage", lambda: deepcopy(lineage))
    actual_evaluation, visited = evaluator.evaluate_robust_methods, []
    def guarded_evaluation(*args, **kwargs):
        profile = kwargs["profile"]
        interval = declaration["validation"]["scenario_seed_ranges"][profile]
        assert kwargs["stage"] == "validation" and kwargs["seed"] == interval["start"]
        assert kwargs["episodes"] == interval["count"] == 2
        visited.append(profile)
        return actual_evaluation(*args, **kwargs)
    patches.setattr(evaluator, "evaluate_robust_methods", guarded_evaluation)
    try:
        runs = {arm: root / arm for arm in evaluator.ARMS}
        summaries = {arm: trainer.run_training(protocol_path, arm, directory) for arm, directory in runs.items()}
        before = {arm: tree(path) for arm, path in runs.items()}
        output = root / "evaluation"
        aggregate = evaluator.run_evaluation(runs, protocol_path=protocol_path, output=output)
        assert visited == list(evaluator.PROFILES)
        assert before == {arm: tree(path) for arm, path in runs.items()}
        fixture = {"root": root, "protocol_path": protocol_path, "runs": runs, "output": output,
                   "aggregate": aggregate, "lineage": lineage, "summaries": summaries}
    finally:
        patches.undo()
    return fixture


@pytest.fixture
def evaluator_fixture(monkeypatch, anchored_evaluation_fixture):
    """Cache verified lineage only for this fixture-only test, never the session."""
    lineage = anchored_evaluation_fixture["lineage"]
    monkeypatch.setattr(evaluator, "load_published_lineage", lambda: deepcopy(lineage))
    monkeypatch.setattr(trainer, "load_published_lineage", lambda: deepcopy(lineage))
    return anchored_evaluation_fixture


def test_shared_fixture_restores_real_loaders_and_evaluator_before_return(anchored_evaluation_fixture):
    from evaluate_robust import evaluate_robust_methods
    from triad_rl.anchored_lineage import load_published_lineage
    assert evaluator.evaluate_robust_methods is evaluate_robust_methods
    assert evaluator.load_published_lineage is load_published_lineage
    assert trainer.load_published_lineage is load_published_lineage


def test_small_real_three_arm_evaluation_pairs_eight_methods_and_persists_sources(evaluator_fixture):
    fixture = evaluator_fixture
    output, result = fixture["output"], fixture["aggregate"]
    assert set(tree(output)) == set(evaluator.EVALUATION_FILES)
    assert result["final_test_accessed"] is False and result["stage"] == "validation"
    inputs = read(output / "evaluation-inputs.json")
    assert inputs["implementation_sha256"] == evaluator.source_provenance()
    assert result["protocol_sha256"] == evaluator._sha(fixture["protocol_path"].read_bytes())
    assert len({summary["first_batch_trajectory_sha256"] for summary in fixture["summaries"].values()}) == 1
    assert len({summary["first_batch_coefficient_identity"]["applied_first_coefficients_sha256"] for summary in fixture["summaries"].values()}) == 1
    for profile in evaluator.PROFILES:
        blob = (output / f"validation-{profile}.json.gz").read_bytes()
        report = json.loads(gzip.decompress(blob))
        receipt = read(output / f"validation-{profile}.receipt.json")
        assert receipt["sha256"] == evaluator._sha(blob) and receipt["bytes"] == len(blob)
        assert set(report["methods"]) == set(evaluator.METHODS)
        interval = inputs["validation_ranges"][profile]
        assert 1_000_999_200_000_000 <= interval["start"] <= 1_000_999_202_000_000
        assert report["seed_provenance"]["declared_lineage"] == inputs["lineage"]
        identities = []
        for method in report["methods"].values():
            rows = method["episodes"]
            assert [row["seed"] for row in rows] == [interval["start"], interval["start"]+1]
            assert [evaluator.canonical_hash(row["scenario"]) for row in rows] == [row["scenario_sha256"] for row in rows]
            identities.append([row["scenario_sha256"] for row in rows])
        assert all(value == identities[0] for value in identities)


def test_complete_resume_is_byte_exact_and_never_samples(evaluator_fixture, monkeypatch):
    fixture = evaluator_fixture
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", lambda *a, **k: pytest.fail("Completed resume sampled"))
    before = tree(fixture["output"])
    result = evaluator.run_evaluation(fixture["runs"], protocol_path=fixture["protocol_path"], output=fixture["output"], resume=True)
    assert result == fixture["aggregate"] and tree(fixture["output"]) == before
    with pytest.raises(ValueError, match="Explicit resume"):
        evaluator.run_evaluation(fixture["runs"], protocol_path=fixture["protocol_path"], output=fixture["output"])


@pytest.mark.parametrize("change", ["bytes", "receipt_type", "scenario", "seed_type", "lineage", "summary"])
def test_cached_later_profile_is_validated_before_any_missing_profile_sampling(evaluator_fixture, tmp_path, monkeypatch, change):
    fixture = evaluator_fixture
    output = tmp_path / "copy"
    output.mkdir()
    for name in ("evaluation-inputs.json", "validation-stress.json.gz", "validation-stress.receipt.json"):
        shutil.copyfile(fixture["output"] / name, output / name)
    report_path, receipt_path = output / "validation-stress.json.gz", output / "validation-stress.receipt.json"
    blob, receipt = report_path.read_bytes(), read(receipt_path)
    if change == "bytes": report_path.write_bytes(blob+b"tamper")
    elif change == "receipt_type":
        receipt["bytes"] = float(receipt["bytes"])
        write(receipt_path, receipt)
    else:
        report = json.loads(gzip.decompress(blob))
        method = report["methods"][evaluator.PRIMARY]
        if change == "scenario": method["episodes"][0]["scenario"]["injected"] = True
        elif change == "seed_type": method["episodes"][0]["seed"] = float(method["episodes"][0]["seed"])
        elif change == "lineage": report["seed_provenance"]["declared_lineage"] = {}
        else: method["summary"]["mean_return"] += 1.
        blob = gzip.compress(evaluator._bytes(report), mtime=0)
        report_path.write_bytes(blob)
        receipt.update(sha256=evaluator._sha(blob), bytes=len(blob))
        write(receipt_path, receipt)
    before = tree(output)
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", lambda *a, **k: pytest.fail("Cache preflight sampled"))
    with pytest.raises(ValueError):
        evaluator.run_evaluation(fixture["runs"], protocol_path=fixture["protocol_path"], output=output, resume=True)
    assert tree(output) == before


@pytest.mark.parametrize("change", ["lock", "protocol", "endpoint", "aggregate"])
def test_saved_input_or_aggregate_tamper_refused_without_sampling(evaluator_fixture, tmp_path, monkeypatch, change):
    fixture = evaluator_fixture
    output = tmp_path / "copy"
    shutil.copytree(fixture["output"], output)
    protocol_path, runs = fixture["protocol_path"], fixture["runs"]
    if change == "lock":
        value = read(output / "evaluation-inputs.json")
        value["protocol_sha256"] = "0"*64
        write(output / "evaluation-inputs.json", value)
    elif change == "protocol":
        protocol_path = tmp_path / "changed-protocol.json"
        protocol_path.write_bytes(fixture["protocol_path"].read_bytes()+b" ")
    elif change == "endpoint":
        copied = tmp_path / "changed-run"
        shutil.copytree(runs[evaluator.PRIMARY], copied)
        summary = read(copied / "summary.json")
        summary["first_batch_coefficient_identity"]["applied_first_coefficients_sha256"] = "0"*64
        write(copied / "summary.json", summary)
        runs = {**runs, evaluator.PRIMARY: copied}
    else:
        value = read(output / "aggregate.json")
        value["scale_gate"]["passed"] = not value["scale_gate"]["passed"]
        write(output / "aggregate.json", value)
    before = tree(output)
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", lambda *a, **k: pytest.fail("Tampered inputs sampled"))
    with pytest.raises(ValueError):
        evaluator.run_evaluation(runs, protocol_path=protocol_path, output=output, resume=True)
    assert tree(output) == before


@pytest.mark.parametrize("change", ["return_guard", "profile_margin", "prior_pilot", "reservation", "source"])
def test_changed_declaration_rejected_before_output_or_sampling(evaluator_fixture, tmp_path, monkeypatch, change):
    fixture = evaluator_fixture
    declaration = read(fixture["protocol_path"])
    if change == "return_guard": declaration["scale_gate"]["require_equal_profile_mean_return_not_lower"] = False
    elif change == "profile_margin": declaration["scale_gate"]["maximum_profile_point_decline"] = .02
    elif change == "prior_pilot": declaration["prior_pilot_publication"]["sha256"] = "0"*64
    elif change == "reservation": declaration["reserved_final_tests_unopened"]["normal"]["count"] += 1
    else: declaration["evaluation_implementation_sha256"]["triad_rl/evaluate_anchored_pilot.py"] = "0"*64
    protocol_path = tmp_path / "changed-protocol.json"
    write(protocol_path, declaration)
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", lambda *a, **k: pytest.fail("Invalid declaration sampled"))
    with pytest.raises(ValueError):
        evaluator.run_evaluation(fixture["runs"], protocol_path=protocol_path, output=tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()
