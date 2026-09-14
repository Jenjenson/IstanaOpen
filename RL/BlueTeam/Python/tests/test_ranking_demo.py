"""Offline ranking presentation uses only already-scored fixture cases."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import re
import shutil

import pytest

import demo_ranking as demo
from test_evaluate_ranking import fixture_bindings, ranking_evaluation_fixture
from triad_rl.adaptive_evaluation import canonical_hash


@pytest.fixture
def completed(ranking_evaluation_fixture, monkeypatch):
    fixture_bindings(monkeypatch, ranking_evaluation_fixture["lineage"])
    return ranking_evaluation_fixture


def args(fixture, output):
    return {"runs": fixture["runs"], "evaluation": fixture["output"],
            "protocol": fixture["protocol_path"], "output": output}


def files(fixture):
    return {path: path.read_bytes() for path in fixture["root"].rglob("*") if path.is_file()}


def test_real_demo_recreates_only_first_two_scored_cases_and_reuses_offline_viewer(completed, tmp_path, monkeypatch):
    fixture, calls = completed, []
    original = demo.collect_replay
    protocol = json.loads(fixture["protocol_path"].read_bytes())
    expected = [(profile, protocol["validation"]["scenario_seed_ranges"][profile]["start"] + case)
                for profile in demo.evaluator.PROFILES for case in range(2)]
    def collect(policy, *, seed, profile):
        assert (profile, seed) in expected
        calls.append((profile, seed))
        return original(policy, seed=seed, profile=profile)
    monkeypatch.setattr(demo, "collect_replay", collect)
    before = files(fixture)
    sources = demo.implementation_fingerprints()
    path = tmp_path / "replays.html"
    replays = demo.run_demo(**args(fixture, path))
    assert calls == expected and len(replays) == 6
    assert files(fixture) == before and demo.implementation_fingerprints() == sources
    text = path.read_text(encoding="utf-8")
    payload = json.loads(re.search(r'<script[^>]+id="replay-data"[^>]*>(.*?)</script>', text, re.S).group(1))
    assert payload["schema"] == "triad.adaptive_demo.v1" and payload["replays"] == replays
    assert "OFFLINE, NOT LIVE" in payload["checkpoint"] and "no promotion or physical commands" in payload["checkpoint"]
    assert not re.search(r'(?:src|href)=["\']https?://', text)
    for row in replays:
        audit = row["audit"]
        assert row["ranker_seed"] == 499 and row["case_index"] in (1, 2)
        assert f"ranker 499 / {row['profile']} / case {row['case_index']} of 2" in row["split"]
        assert row["schema"] == "triad.ranking_replay.v1" and row["release"] == "ranking-v5-pilot"
        assert audit["scored_actions_metrics_verified"] is True
        assert audit["experimental"] is True and audit["default_policy_changed"] is False
        assert audit["physical_commands"] is False and audit["live_feed"] is False
        assert audit["policy_state_before"] == audit["policy_state_after"]
        assert audit["checkpoint_files_sha256_before"] == audit["checkpoint_files_sha256_after"]
        assert audit["ranking_implementation_sha256"] == sources
    digest = canonical_hash({"499": replays[0]["audit"]["policy_state_before"]["weights_sha256"]})
    assert payload["weights_sha256"] == digest


def test_existing_output_refused_before_verification_or_collection(completed, tmp_path, monkeypatch):
    path = tmp_path / "existing.html"
    path.write_bytes(b"user-owned")
    def forbidden(*unused, **kwargs):
        pytest.fail("Existing output must fail before validation or replay")
    monkeypatch.setattr(demo.evaluator, "verify_completed_evaluation", forbidden)
    monkeypatch.setattr(demo, "collect_replay", forbidden)
    with pytest.raises(FileExistsError):
        demo.run_demo(**args(completed, path))
    assert path.read_bytes() == b"user-owned"


def test_incomplete_evaluation_rejected_before_any_replay(completed, tmp_path, monkeypatch):
    evaluation = tmp_path / "incomplete"
    shutil.copytree(completed["output"], evaluation)
    (evaluation / "aggregate.json").unlink()
    monkeypatch.setattr(demo, "collect_replay", lambda *a, **k: pytest.fail("No replay before completion"))
    output = tmp_path / "no.html"
    with pytest.raises((ValueError, FileNotFoundError)):
        demo.run_demo(**{**args(completed, output), "evaluation": evaluation})
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["actions", "metrics", "scenario", "placements", "targets"])
def test_recreated_mismatch_fails_before_any_output(completed, tmp_path, monkeypatch, mutation):
    original = demo.collect_replay
    def changed(*items, **kwargs):
        replay = original(*items, **kwargs)
        if mutation == "actions":
            replay["decisions"][0]["action_index"] += 1
        elif mutation == "metrics":
            replay["metrics"]["cost"] += .1
        elif mutation == "scenario":
            replay["scenario"]["targets"][0]["speed"] += .1
        elif mutation == "placements":
            replay["placements"] = [] if replay["placements"] else [{"sensor_id": "fake"}]
        else:
            replay["metrics"]["target_results"][0]["time_to_zone"] += 1.
        return replay
    monkeypatch.setattr(demo, "collect_replay", changed)
    output = tmp_path / "no.html"
    with pytest.raises(ValueError, match="differs from its scored"):
        demo.run_demo(**args(completed, output))
    assert not output.exists()


def test_actor_rng_drift_fails_before_output(completed, tmp_path, monkeypatch):
    original = demo.collect_replay
    def drift(policy, **kwargs):
        replay = original(policy, **kwargs)
        policy.rng.random()
        return replay
    monkeypatch.setattr(demo, "collect_replay", drift)
    output = tmp_path / "no.html"
    with pytest.raises(RuntimeError, match="RNG changed"):
        demo.run_demo(**args(completed, output))
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["source", "checkpoint"])
def test_source_or_checkpoint_drift_during_render_fails_before_output(completed, tmp_path, monkeypatch, mutation):
    original = demo.export_balanced_html
    source = demo.implementation_fingerprints()
    def drift(*items, **kwargs):
        original(*items, **kwargs)
        if mutation == "source":
            monkeypatch.setattr(demo, "implementation_fingerprints", lambda: {**source, "changed.py": "0" * 64})
        else:
            read = demo.evaluator._file
            target = completed["runs"][499] / "last/checkpoint.json"
            monkeypatch.setattr(demo.evaluator, "_file", lambda path: b"changed checkpoint" if path == target else read(path))
    monkeypatch.setattr(demo, "export_balanced_html", drift)
    output = tmp_path / "no.html"
    with pytest.raises(RuntimeError, match="source/checkpoint/evidence/RNG"):
        demo.run_demo(**args(completed, output))
    assert not output.exists()


def test_output_race_preserves_other_writer(completed, tmp_path, monkeypatch):
    output = tmp_path / "race.html"
    original = demo.export_balanced_html
    def race(*items, **kwargs):
        original(*items, **kwargs)
        output.write_bytes(b"other writer")
    monkeypatch.setattr(demo, "export_balanced_html", race)
    with pytest.raises(FileExistsError):
        demo.run_demo(**args(completed, output))
    assert output.read_bytes() == b"other writer"


def test_every_declared_endpoint_is_included_without_selection(completed, tmp_path, monkeypatch):
    # Synthetic endpoint identifiers only: reuse one already-scored fixture's
    # exact policy/reports. Never train or collect production403/9953 scenarios.
    evaluation, protocol_path = tmp_path / "evaluation", tmp_path / "protocol.json"
    shutil.copytree(completed["output"], evaluation)
    protocol = json.loads(completed["protocol_path"].read_bytes())
    protocol["training"]["policy_seeds"] = [497, 498, 499]
    protocol_path.write_bytes(demo.evaluator._bytes(protocol))
    for profile in demo.evaluator.PROFILES:
        path = evaluation / f"validation-{profile}.json.gz"
        report = json.loads(gzip.decompress(path.read_bytes()))
        report["methods"].update({f"ranker_{seed}": deepcopy(report["methods"]["ranker_499"])
                                   for seed in (497, 498)})
        path.write_bytes(gzip.compress(demo.evaluator._bytes(report), mtime=0))
    monkeypatch.setattr(demo.evaluator, "verify_completed_evaluation", lambda *a, **k: {})
    output = tmp_path / "all.html"
    replays = demo.run_demo(runs=dict.fromkeys((497, 498, 499), completed["runs"][499]),
                            evaluation=evaluation, protocol=protocol_path, output=output)
    assert len(replays) == 18
    assert {(r["ranker_seed"], r["profile"], r["case_index"]) for r in replays} == {
        (seed, profile, case) for seed in (497, 498, 499) for profile in demo.evaluator.PROFILES for case in (1, 2)}


def test_less_than_two_scored_cases_rejected_before_collection(completed, tmp_path, monkeypatch):
    evaluation = tmp_path / "evaluation"
    shutil.copytree(completed["output"], evaluation)
    path = evaluation / "validation-capability.json.gz"
    report = json.loads(gzip.decompress(path.read_bytes()))
    report["methods"]["ranker_499"]["episodes"] = report["methods"]["ranker_499"]["episodes"][:1]
    path.write_bytes(gzip.compress(demo.evaluator._bytes(report), mtime=0))
    monkeypatch.setattr(demo.evaluator, "verify_completed_evaluation", lambda *a, **k: {})
    monkeypatch.setattr(demo, "collect_replay", lambda *a, **k: pytest.fail("No partial case selection"))
    output = tmp_path / "no.html"
    with pytest.raises(ValueError, match="first two"):
        demo.run_demo(**{**args(completed, output), "evaluation": evaluation})
    assert not output.exists()
