"""Paired balanced selection binds full training, transfer lineage and controls."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import evaluate_balanced as balanced
from triad_rl.adaptive_evaluation import canonical_hash, summarize
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.robust_scenarios import RobustPlacementEnv
from triad_rl.train_balanced import run_training, validation_seed_ranges


def evidence():
    candidates = {501: {"weights_sha256": "a", "initialized_weights_sha256": "i"},
                  502: {"weights_sha256": "b", "initialized_weights_sha256": "i"}}
    returns = {501: [8., -9., -3.], 502: [7., -7., -2.]}
    reports = {}
    for index, profile in enumerate(balanced.PROFILES):
        report = {"stage": "validation", "profile": profile, "schema": balanced.REPORT_SCHEMA,
                  "protocol": {"independent_final_test_evidence": False},
                  "implementation_sha256": {"source": "same"},
                  "implementation_sha256_after": {"source": "same"}, "methods": {},
                  "seed_provenance": {"evaluation": {"start": 1000992880000000 + index * 1000000, "count": 1}}}
        for seed, candidate in candidates.items():
            for kind in ("candidate", "initialized"):
                scenario = {"fixture": profile}
                fingerprint = candidate["weights_sha256"] if kind == "candidate" else "i"
                records = [{"weights_sha256_before": fingerprint, "weights_sha256_after": fingerprint,
                            "seed": report["seed_provenance"]["evaluation"]["start"],
                            "scenario": scenario, "scenario_sha256": canonical_hash(scenario),
                            "metrics": {"return": returns[seed][index] if kind == "candidate" else 0.}}]
                report["methods"][f"{kind}_{seed}"] = {"summary": summarize(records), "episodes": records}
        reports[profile] = report
    return reports, candidates


def test_equal_profile_selection_and_input_immutability():
    reports, candidates = evidence()
    before = deepcopy(reports)
    selected, scores = balanced.select_from_reports(reports, candidates)
    assert selected["seed"] == 502
    assert selected["balanced_mean_return"] == pytest.approx(-2 / 3)
    assert len(scores) == 2 and reports == before


@pytest.mark.parametrize("mutation", ["test", "old_schema", "missing_profile", "source_drift", "source_after",
                                       "weights", "initial_weights", "missing_control", "empty", "summary", "unpaired"])
def test_rejects_incomplete_unpaired_or_mixed_evidence(mutation):
    reports, candidates = evidence()
    report = reports["stress"]
    if mutation == "test":
        report["stage"] = "test"
    elif mutation == "old_schema":
        report["schema"] = "triad.robust_evaluation.v1"
    elif mutation == "missing_profile":
        reports.pop("capability")
    elif mutation == "source_drift":
        report["implementation_sha256"] = {"source": "changed"}
    elif mutation == "source_after":
        report["implementation_sha256_after"] = {"source": "changed"}
    elif mutation == "weights":
        report["methods"]["candidate_501"]["episodes"][0]["weights_sha256_after"] = "changed"
    elif mutation == "initial_weights":
        report["methods"]["initialized_501"]["episodes"][0]["weights_sha256_before"] = "changed"
    elif mutation == "missing_control":
        for value in reports.values():
            value["methods"].pop("initialized_501")
    elif mutation == "empty":
        report["methods"]["candidate_501"]["episodes"] = []
    elif mutation == "summary":
        report["methods"]["candidate_501"]["summary"]["mean_return"] = 123
    else:
        report["methods"]["candidate_501"]["episodes"][0]["scenario_sha256"] = "unpaired"
    with pytest.raises(ValueError):
        balanced.select_from_reports(reports, candidates)


def test_equal_scores_tie_break_on_seed():
    reports, candidates = evidence()
    for report in reports.values():
        other = report["methods"]["candidate_502"]
        other["episodes"][0]["metrics"] = deepcopy(report["methods"]["candidate_501"]["episodes"][0]["metrics"])
        other["summary"] = summarize(other["episodes"])
    selected, _ = balanced.select_from_reports(reports, dict(reversed(list(candidates.items()))))
    assert selected["seed"] == 501


def test_frozen_reference_bundle_and_full_exposure_are_bound():
    lineage, evidence = balanced._reference_evidence()
    assert evidence["v2_selection"]["sha256"] == balanced.V2_SELECTION_SHA256
    ranges = balanced.nested_seed_ranges(lineage)
    assert any(row == {"start": 103000000, "count": 8000} for row in ranges.values())
    assert any(row == {"start": 1000990200000000, "count": 300} for row in ranges.values())
    assert len(evidence["publication_files_sha256"]) == 41


def test_fixed_baselines_never_change_sensor_and_obey_masks():
    for sensor_id in ("rf", "radar"):
        env = RobustPlacementEnv(seed=1000992885000000, profile="capability")
        obs = env.observe()
        actor = balanced.FixedSingleModality(sensor_id)
        for _ in range(3):
            action = actor.act(obs)
            option = obs["options"][action]
            assert obs["action_mask"][action]
            assert option.get("stop") or option["sensor_id"] == sensor_id
            obs, _, done, _ = env.step(action)
            if done:
                break
        assert done


@pytest.fixture(scope="module")
def small_runs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("balanced-selection")
    paths = {}
    for seed in (501, 502):
        run = directory / str(seed)
        run_training(output=run, episodes=4, batch_size=2, seed=seed,
                     learning_rate=.002, entropy_coef=.015, gamma=1.,
                     validation_every=4, validation_episodes=1, validation_run_seed=992850,
                     initial_checkpoint=balanced.BLUE_ROOT / "Checkpoints" / "robust-v2-candidate",
                     initial_selection_report=balanced.BLUE_ROOT / "Results" / "robust-v2" / "selection.json")
        paths[seed] = run / "best"
    protocol = {
        "schema": balanced.PROTOCOL_SCHEMA,
        "frozen_reference": {"weights_sha256": balanced.V2_WEIGHTS,
                             "checkpoint": "Checkpoints/robust-v2-candidate",
                             "selection_report": "Results/robust-v2/selection.json"},
        "candidate_training": {"run_seeds": [501, 502], "episodes_per_run": 4,
                               "batch_size": 2, "learning_rate": .002, "entropy_coef": .015,
                               "gamma": 1., "validation_every": 4, "validation_episodes_per_profile": 1,
                               "training_profile": "mixed", "value_coef": .5, "max_grad_norm": 1.,
                               "training_seed_ranges": [{"start": seed * 1000000, "count": 4} for seed in (501, 502)],
                               "validation_seed_ranges": [{"profile": p, **r} for p, r in validation_seed_ranges(992850, 1).items()]},
        "candidate_selection": {"seed_ranges": [
            {"profile": profile, "start": 1000992880000000 + index * 1000000, "count": 2}
            for index, profile in enumerate(balanced.PROFILES)]},
    }
    protocol_path = directory / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return paths, protocol, protocol_path


def test_real_selection_preserves_full_paired_reports_and_controls(small_runs, tmp_path):
    paths, protocol, protocol_path = small_runs
    output = tmp_path / "selection"
    summary = balanced.run_selection(paths, protocol_path=protocol_path, output=output, bootstrap_samples=10)
    assert summary["schema"] == balanced.SCHEMA and summary["final_test_accessed"] is False
    assert summary["selected"]["seed"] in paths
    for seed, metadata in summary["candidate_metadata"].items():
        complete = metadata["complete_run"]
        assert complete["last_checkpoint_metadata"]["training_state"]["completed_episodes"] == 4
        assert len(complete["run_files_sha256"]) == 9
    assert len(summary["seed_provenance"]["selection_validation"]) == 3
    expected_methods = {"candidate_501", "candidate_502", "initialized_501", "initialized_502",
                        "robust_v2", "adaptive_v1", "greedy_public", "random_legal", "fixed_rf", "fixed_radar",
                        "uniform_fixed_rf_radar", "legacy_toy210_projected"}
    reports = {}
    for profile, artifact in summary["reports"].items():
        raw = (output / artifact["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
        report = json.loads(gzip.decompress(raw))
        reports[profile] = report
        assert report["schema"] == balanced.REPORT_SCHEMA and report["stage"] == "validation"
        assert set(report["methods"]) == expected_methods
        assert all(len(method["episodes"]) == 2 for method in report["methods"].values())
        initial_a, initial_b = [report["methods"][f"initialized_{seed}"] for seed in (501, 502)]
        assert initial_a["summary"] == initial_b["summary"]
        assert [row["actions"] for row in initial_a["episodes"]] == [row["actions"] for row in initial_b["episodes"]]
        assert initial_a["metadata"]["is_claimed_untrained"] is False
        assert len(report["selected_candidate_paired_differences"]) == len(expected_methods) - 1
    assert balanced.select_from_reports(reports, {int(k): v for k, v in summary["candidate_metadata"].items()})[0] == summary["selected"]
    with pytest.raises(ValueError, match="empty directory"):
        balanced.run_selection(paths, protocol_path=protocol_path, output=output)


@pytest.mark.parametrize("mutation", ["summary", "log", "source", "full_train", "validation", "value_coef", "init"])
def test_complete_run_tampering_rejected(small_runs, tmp_path, mutation):
    paths, original, _ = small_runs
    run = tmp_path / "candidate"
    shutil.copytree(paths[501].parent, run)
    protocol = deepcopy(original)
    if mutation == "summary":
        path = run / "summary.json"
        value = json.loads(path.read_text())
        value["completed_episodes"] = 2
        path.write_text(json.dumps(value))
    elif mutation == "log":
        with (run / "training.jsonl").open("a") as handle:
            handle.write('{}\n')
    elif mutation == "source":
        path = run / "config.json"
        value = json.loads(path.read_text())
        value["source_sha256"]["balanced_policy.py"] = "changed"
        path.write_text(json.dumps(value))
    elif mutation == "full_train":
        protocol["candidate_training"]["training_seed_ranges"][0]["count"] = 2
    elif mutation == "validation":
        protocol["candidate_training"]["validation_seed_ranges"][0]["start"] += 10
    elif mutation == "value_coef":
        protocol["candidate_training"]["value_coef"] = .75
    else:
        initial = BalancedPolicy.load(run / "initialized")
        initial.parameters["wa"][0] += .1
        initial.save(run / "initialized", initial.training_state)
    policy = BalancedPolicy.load(run / "best", feature_names=FEATURE_NAMES)
    with pytest.raises(ValueError):
        balanced._run_evidence(run / "best", policy, 501, protocol)


def test_frozen_factory_detects_parameter_replacement(small_runs, tmp_path):
    paths, _, _ = small_runs
    shutil.copytree(paths[501], tmp_path / "checkpoint")
    policy = BalancedPolicy.load(tmp_path / "checkpoint")
    factory = balanced.balanced_factory(tmp_path / "checkpoint", policy.weights_fingerprint())
    assert factory(42).weights_fingerprint() == policy.weights_fingerprint()
    policy.parameters["wa"][0] += .1
    policy.save(tmp_path / "checkpoint", policy.training_state)
    with pytest.raises(RuntimeError, match="changed"):
        factory(42)


@pytest.mark.parametrize("mutation", ["missing_batch", "wrong_cadence", "wrong_selected", "wrong_score", "unknown_event"])
def test_rehashed_log_still_requires_complete_honest_selection_trace(small_runs, tmp_path, mutation):
    paths, protocol, _ = small_runs
    run = tmp_path / "candidate"
    shutil.copytree(paths[501].parent, run)
    log_path = run / "training.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    if mutation == "missing_batch":
        rows = [row for row in rows if not (row["event"] == "training_batch" and row["episode"] == 2)]
    elif mutation == "wrong_cadence":
        next(row for row in rows if row["event"] == "validation" and row["episode"] == 4)["episode"] = 2
    elif mutation == "wrong_selected":
        rows[0]["selected"] = False
    elif mutation == "wrong_score":
        rows[0]["balanced_mean_return"] += 3
    else:
        rows.append({"event": "unknown"})
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    last = BalancedPolicy.load(run / "last")
    state = deepcopy(last.training_state)
    state["log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
    last.save(run / "last", state)
    policy = BalancedPolicy.load(run / "best")
    with pytest.raises(ValueError):
        balanced._run_evidence(run / "best", policy, 501, protocol)


def test_validation_only_rejects_final_before_loading_checkpoint(small_runs, tmp_path, monkeypatch):
    paths, original, _ = small_runs
    protocol = deepcopy(original)
    protocol["candidate_selection"]["seed_ranges"][0]["start"] = 2000000500000000
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(BalancedPolicy, "load", lambda *a, **kw: pytest.fail("Must reject final seeds before loading"))
    with pytest.raises(ValueError, match="Validation seeds"):
        balanced.run_selection(paths, protocol_path=path, output=tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field", ["weights_sha256", "checkpoint", "selection_report"])
def test_contradictory_frozen_reference_rejected_before_loading(small_runs, tmp_path, monkeypatch, field):
    paths, original, _ = small_runs
    protocol = deepcopy(original)
    protocol["frozen_reference"][field] = "changed"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(BalancedPolicy, "load", lambda *a, **kw: pytest.fail("Must reject reference before loading"))
    with pytest.raises(ValueError, match="transfer reference"):
        balanced.run_selection(paths, protocol_path=path, output=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_internal_validation_overlap_rejected_before_evaluation(small_runs, tmp_path, monkeypatch):
    paths, original, _ = small_runs
    protocol = deepcopy(original)
    protocol["candidate_selection"]["seed_ranges"][0]["start"] = 1000992850000000
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(balanced, "evaluate_robust_methods", lambda *a, **kw: pytest.fail("Overlap must fail before sampling"))
    with pytest.raises(ValueError, match="overlap"):
        balanced.run_selection(paths, protocol_path=path, output=tmp_path / "output")


def test_source_drift_mid_selection_leaves_no_report(small_runs, tmp_path, monkeypatch):
    paths, _, protocol_path = small_runs
    real = balanced.evaluate_robust_methods
    source_before = balanced.implementation_fingerprints()

    def changed(*args, **kwargs):
        report = real(*args, **kwargs)
        monkeypatch.setattr(balanced, "implementation_fingerprints", lambda: {**source_before, "extra": "changed"})
        return report

    monkeypatch.setattr(balanced, "evaluate_robust_methods", changed)
    with pytest.raises(RuntimeError, match="changed"):
        balanced.run_selection(paths, protocol_path=protocol_path, output=tmp_path / "output", bootstrap_samples=2)
    assert not (tmp_path / "output").exists()


def test_cli_duplicate_candidates_rejected():
    with pytest.raises(SystemExit) as error:
        balanced.main(["--candidate", "501=a", "--candidate", "501=b", "--output", "unused"])
    assert error.value.code == 2


@pytest.fixture(scope="module")
def recovery_evidence(small_runs, tmp_path_factory):
    """One real complete selection and one real interrupted profile snapshot."""
    paths, _, protocol = small_runs
    root = tmp_path_factory.mktemp("balanced-recovery")
    complete = root / "complete"
    balanced.run_selection(paths, protocol_path=protocol, output=complete, bootstrap_samples=2)
    partial = root / "partial"
    worker = balanced._profile_worker

    def interrupted(profile, context):
        if profile == "stress":
            raise RuntimeError("Simulated app restart after normal committed")
        return worker(profile, context)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(balanced, "_profile_worker", interrupted)
        with pytest.raises(RuntimeError, match="Simulated app restart"):
            balanced.run_selection(paths, protocol_path=protocol, output=partial, bootstrap_samples=2)
    assert sorted(json.loads((partial / "progress.json").read_text())["completed_profiles"]) == ["normal"]
    assert not (partial / "selection.json").exists()
    return complete, partial


def _assert_final_bytes_equal(a, b):
    for name in ("selection.json", *(f"common-validation-{p}.json.gz" for p in balanced.PROFILES)):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_real_parallel_resume_matches_serial_without_rewriting_completed_profile(small_runs, recovery_evidence, tmp_path):
    paths, _, protocol = small_runs
    complete, partial = recovery_evidence
    output = tmp_path / "resume"
    shutil.copytree(partial, output)
    normal = output / balanced._profile_filename("normal")
    before_bytes, before_mtime = normal.read_bytes(), normal.stat().st_mtime_ns
    balanced.run_selection(paths, protocol_path=protocol, output=output, bootstrap_samples=2, workers=2, resume=True)
    assert normal.read_bytes() == before_bytes and normal.stat().st_mtime_ns == before_mtime
    _assert_final_bytes_equal(complete, output)
    assert (complete / "progress.json").read_bytes() == (output / "progress.json").read_bytes()


def test_serial_resume_never_resamples_verified_profile(small_runs, recovery_evidence, tmp_path, monkeypatch):
    paths, _, protocol = small_runs
    complete, partial = recovery_evidence
    output = tmp_path / "resume"
    shutil.copytree(partial, output)
    called, worker = [], balanced._profile_worker

    def traced(profile, context):
        assert profile != "normal", "A committed completed profile must not be sampled again"
        called.append(profile)
        return worker(profile, context)

    monkeypatch.setattr(balanced, "_profile_worker", traced)
    balanced.run_selection(paths, protocol_path=protocol, output=output, bootstrap_samples=2, resume=True)
    assert called == ["stress", "capability"]
    _assert_final_bytes_equal(complete, output)
    monkeypatch.setattr(balanced, "_profile_worker", lambda *a: pytest.fail("Completed resume must not sample"))
    balanced.run_selection(paths, protocol_path=protocol, output=output, bootstrap_samples=2, workers=3, resume=True)
    _assert_final_bytes_equal(complete, output)


def test_recover_orphan_atomic_profile_and_keep_incomplete_temporary(small_runs, recovery_evidence, tmp_path, monkeypatch):
    paths, _, protocol = small_runs
    complete, partial = recovery_evidence
    output = tmp_path / "orphan"
    shutil.copytree(partial, output)
    progress = json.loads((output / "progress.json").read_text())
    progress["completed_profiles"] = {}  # Crash after atomic profile, before progress commit.
    (output / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    temporary = output / ".completed-validation-stress.json.gz.interrupted.tmp"
    temporary.write_bytes(b"incomplete owned atomic temporary")
    called, worker = [], balanced._profile_worker

    def traced(profile, context):
        assert profile != "normal"
        called.append(profile)
        return worker(profile, context)

    monkeypatch.setattr(balanced, "_profile_worker", traced)
    balanced.run_selection(paths, protocol_path=protocol, output=output, bootstrap_samples=2, resume=True)
    assert called == ["stress", "capability"] and temporary.read_bytes() == b"incomplete owned atomic temporary"
    _assert_final_bytes_equal(complete, output)


@pytest.mark.parametrize("mutation", ["bytes", "seed", "summary", "weights", "comparison", "subgroup", "missing_groups", "missing_metric", "progress", "unknown"])
def test_resume_rejects_corrupt_or_rehashed_tampered_profiles_before_sampling(
        small_runs, recovery_evidence, tmp_path, monkeypatch, mutation):
    paths, _, protocol = small_runs
    _, partial = recovery_evidence
    output = tmp_path / "tampered"
    shutil.copytree(partial, output)
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text())
    snapshot = output / balanced._profile_filename("normal")
    if mutation == "bytes":
        snapshot.write_bytes(snapshot.read_bytes() + b"changed")
    elif mutation == "progress":
        progress["experiment"]["bootstrap_samples"] = 3
        progress_path.write_text(json.dumps(progress))
    elif mutation == "unknown":
        (output / "notes.txt").write_text("unrecognized user-owned file")
    else:
        report = json.loads(gzip.decompress(snapshot.read_bytes()))
        if mutation == "seed":
            report["seed_provenance"]["evaluation"]["start"] += 10
        elif mutation == "summary":
            report["methods"]["candidate_501"]["summary"]["mean_return"] += 1
        elif mutation == "weights":
            report["methods"]["candidate_501"]["episodes"][0]["weights_sha256_after"] = "changed"
        elif mutation == "comparison":
            comparison = next(iter(report["paired_differences"].values()))
            comparison["return"]["difference"] += 1
        elif mutation == "missing_groups":
            report["methods"]["candidate_501"]["subgroups"] = {}
        elif mutation == "missing_metric":
            report["methods"]["candidate_501"]["episodes"][0]["metrics"].pop("cost")
            report["methods"]["candidate_501"]["summary"] = summarize(report["methods"]["candidate_501"]["episodes"])
        else:
            groups = report["methods"]["candidate_501"]["subgroups"]["curriculum_profile"]
            next(iter(groups.values()))["mean_return"] += 1
        data = gzip.compress(json.dumps(report, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
        snapshot.write_bytes(data)
        progress["completed_profiles"]["normal"] = balanced._profile_entry("normal", data, report)
        progress_path.write_text(json.dumps(progress))
    monkeypatch.setattr(balanced, "_profile_worker", lambda *a: pytest.fail("Tamper must fail before sampling"))
    with pytest.raises(ValueError):
        balanced.run_selection(paths, protocol_path=protocol, output=output, bootstrap_samples=2, resume=True)
    assert not (output / "selection.json").exists()


@pytest.mark.parametrize("mutation", ["protocol", "source", "checkpoint", "bootstrap"])
def test_resume_rejects_changed_frozen_experiment(small_runs, recovery_evidence, tmp_path, monkeypatch, mutation):
    original_paths, original_protocol, protocol_path = small_runs
    paths = original_paths.copy()
    _, partial = recovery_evidence
    output = tmp_path / "resume"
    shutil.copytree(partial, output)
    samples = 2
    if mutation == "protocol":
        protocol = deepcopy(original_protocol)
        protocol["new_note"] = "Changed source protocol bytes"
        protocol_path = tmp_path / "protocol.json"
        protocol_path.write_text(json.dumps(protocol))
    elif mutation == "source":
        sources = balanced.implementation_fingerprints()
        monkeypatch.setattr(balanced, "implementation_fingerprints", lambda: {**sources, "new_source.py": "changed"})
    elif mutation == "checkpoint":
        run = tmp_path / "changed_run"
        shutil.copytree(paths[501].parent, run)
        last = BalancedPolicy.load(run / "last")
        last.rng = np.random.default_rng(12345)
        last.save(run / "last", last.training_state)
        paths[501] = run / "best"
    else:
        samples = 3
    monkeypatch.setattr(balanced, "_profile_worker", lambda *a: pytest.fail("Changed binding must fail before sampling"))
    with pytest.raises(ValueError, match="binding changed"):
        balanced.run_selection(paths, protocol_path=protocol_path, output=output, bootstrap_samples=samples, resume=True)


def test_resume_finalization_reuses_matching_files_and_rejects_changed_final(small_runs, recovery_evidence, tmp_path, monkeypatch):
    paths, _, protocol = small_runs
    complete, _ = recovery_evidence
    output = tmp_path / "finalization"
    shutil.copytree(complete, output)
    (output / "selection.json").unlink()  # Simulate crash just before the final commit marker.
    monkeypatch.setattr(balanced, "_profile_worker", lambda *a: pytest.fail("All profiles already completed"))
    balanced.run_selection(paths, protocol_path=protocol, output=output, bootstrap_samples=2, resume=True)
    _assert_final_bytes_equal(complete, output)
    (output / "selection.json").write_text("{}")
    with pytest.raises(ValueError, match="Existing final"):
        balanced.run_selection(paths, protocol_path=protocol, output=output, bootstrap_samples=2, resume=True)


@pytest.mark.parametrize("workers", [0, 4, True, 1.5])
def test_bad_worker_count_rejected_without_output(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        balanced.run_selection({}, protocol_path=tmp_path / "missing", output=tmp_path / "none", workers=workers)
    assert not (tmp_path / "none").exists()


def test_resume_requires_existing_progress(small_runs, tmp_path):
    paths, _, protocol = small_runs
    with pytest.raises(ValueError, match="progress.json"):
        balanced.run_selection(paths, protocol_path=protocol, output=tmp_path / "missing", bootstrap_samples=2, resume=True)
