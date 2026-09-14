"""Stored handmade cases only; no pilot seeds, generators or training.

The shared fixture scores hand-constructed cases with real frozen physics;
endpoint validation remains that fixture's explicit stub, tested by the trainer.
"""
from copy import deepcopy
import gzip
import json
import re
import shutil

import pytest

import demo_temporal as demo
from test_evaluate_temporal import bindings, forbidden, temporal_evaluation_fixture
from test_temporal_env import core


@pytest.fixture
def completed(temporal_evaluation_fixture, monkeypatch):
    bindings(monkeypatch, temporal_evaluation_fixture)
    # Even the evaluator's handmade factory must not be called by verification
    # or replay: the completed records, not seed regeneration, are the inputs.
    monkeypatch.setattr(demo.evaluator, "make_case", forbidden)
    return temporal_evaluation_fixture


def args(fixture, output):
    return {"runs": fixture["runs"], "evaluation": fixture["output"],
            "protocol": fixture["protocol_path"], "output": output}


def row_at(fixture):
    report = json.loads(gzip.decompress((fixture["output"] / "validation-normal.json.gz").read_bytes()))
    return report["methods"]["temporal_499"]["episodes"][0]


def actor_at(fixture):
    return demo.TemporalPolicy.load(fixture["runs"][499] / "last", config=demo.TemporalConfig(**fixture["protocol"]["temporal_config"]))


def collect(fixture, row=None, actor=None):
    return demo.collect_stored_replay(actor or actor_at(fixture), row or row_at(fixture),
        config=demo.TemporalConfig(**fixture["protocol"]["temporal_config"]), profile="normal")


def test_all_18_stored_replays_exact_scoring_no_generators_or_mutation(completed, tmp_path, monkeypatch):
    fixture, calls, verified = completed, [], []
    original = demo.TemporalPlacementEnv
    verify = demo.evaluator.verify_completed_evaluation
    def verified_first(*a, **k):
        result = verify(*a, **k)
        verified.append(True)
        return result
    def exact_case(**kwargs):
        assert verified, "No replay until all endpoints and evaluation are verified"
        assert set(kwargs) == {"scenario", "catalogue", "config"}
        assert "evaluation_catalogue" not in kwargs["scenario"]
        assert "curriculum_metadata" not in kwargs["scenario"]
        calls.append(deepcopy(kwargs["scenario"]))
        return original(**kwargs)
    monkeypatch.setattr(demo, "TemporalPlacementEnv", exact_case)
    monkeypatch.setattr(demo.evaluator, "verify_completed_evaluation", verified_first)
    original_act = demo.TemporalPolicy.act
    def public_only(self, observation, deterministic=True):
        assert deterministic is True
        assert not {"scenario", "targets", "frames", "target_results", "curriculum_metadata"} & observation.keys()
        assert "private-target-not-in-public-inputs" not in repr(observation)
        return original_act(self, observation, deterministic=True)
    monkeypatch.setattr(demo.TemporalPolicy, "act", public_only)
    before = {path: path.read_bytes() for path in fixture["root"].rglob("*") if path.is_file()}
    output = tmp_path / "all.html"
    replays = demo.run_demo(**args(fixture, output))
    assert len(replays) == len(calls) == 18 and verified == [True]
    assert {(r["temporal_seed"], r["profile"], r["case_index"]) for r in replays} == {
        (seed, profile, index) for seed in (497, 498, 499) for profile in demo.evaluator.PROFILES for index in (1, 2)}
    assert before == {path: path.read_bytes() for path in fixture["root"].rglob("*") if path.is_file()}
    text = output.read_text(encoding="utf-8")
    payload = json.loads(re.search(r'<script[^>]+id="replay-data"[^>]*>(.*?)</script>', text, re.S).group(1))
    assert payload["replays"] == replays and payload["schema"] == "triad.adaptive_demo.v1"
    assert "OFFLINE, NOT LIVE" in payload["checkpoint"] and "no promotion or physical commands" in payload["checkpoint"]
    assert not re.search(r'(?:src|href)=["\']https?://', text)
    for replay, supplied in zip(replays, calls):
        assert replay["scenario"] == supplied and replay["frames"]
        audit = replay["audit"]
        assert audit["stored_case_only"] and audit["scored_actions_metrics_verified"]
        assert not any(audit[key] for key in ("scenario_generation_performed", "training_performed", "browser_inference",
            "independent_final_test_evidence", "physical_commands", "default_policy_changed", "live_feed"))
        assert audit["policy_state_before"] == audit["policy_state_after"]
        assert audit["checkpoint_files_sha256_before"] == audit["checkpoint_files_sha256_after"]
        assert audit["implementation_sha256_before"] == audit["implementation_sha256_after"] == demo.implementation_fingerprints()


def test_collection_preserves_input_and_full_scored_frames(completed):
    row, actor = row_at(completed), actor_at(completed)
    before, state = deepcopy(row), demo._policy_state(actor)
    replay = collect(completed, row, actor)
    assert row == before and demo._policy_state(actor) == state
    # The compact evaluation omits frames. Independently step its same stored
    # case/actions through the frozen core, without another actor or generator.
    scenario = deepcopy(row["scenario"])
    catalogue = scenario.pop("evaluation_catalogue")
    scenario.pop("curriculum_metadata")
    env = core(scenario, catalogue)
    for action in row["actions"]: _, _, _, info = env.step(action)
    assert replay["frames"] == info["frames"]
    assert replay["metrics"]["target_results"] == row["target_results"]
    assert sum(decision["reward"] for decision in replay["decisions"]) == row["metrics"]["return"]


@pytest.mark.parametrize("mutation", ["scenario", "catalogue", "metadata", "actions", "placements", "metrics", "rewards", "targets", "outcome", "weights"])
def test_mismatch_rejected_without_changing_input(completed, mutation):
    row = deepcopy(row_at(completed))
    if mutation == "scenario": row["scenario"]["targets"][0]["speed"] += .1
    elif mutation == "catalogue": row["scenario"]["evaluation_catalogue"][0]["cost"] += .1
    elif mutation == "metadata":
        row["scenario"]["curriculum_metadata"]["profile"] = "stress"
        row["scenario_sha256"] = demo.canonical_hash(row["scenario"])
    elif mutation == "actions": row["actions"][0] += 1
    elif mutation == "placements": row["placements"] = [] if row["placements"] else [{"sensor_id": "fake"}]
    elif mutation == "metrics": row["metrics"]["cost"] += .1
    elif mutation == "rewards": row["reward_components"][next(iter(row["reward_components"]))] += .1
    elif mutation == "targets": row["target_results"][0]["time_to_zone"] += 1.
    elif mutation == "outcome": row["outcome"] = "changed"
    else: row["weights_sha256_before"] = "0" * 64
    before = deepcopy(row)
    with pytest.raises(ValueError): collect(completed, row)
    assert row == before


@pytest.mark.parametrize("action", [True, 1.5, -1, 999999])
def test_invalid_actor_action_rejected(completed, monkeypatch, action):
    monkeypatch.setattr(demo.TemporalPolicy, "act", lambda *a, **k: action)
    with pytest.raises(ValueError, match="invalid or masked"): collect(completed)


@pytest.mark.parametrize("mutation", ["rng", "weights"])
def test_actor_drift_rejected(completed, monkeypatch, mutation):
    original = demo.TemporalPolicy.act
    def drift(self, obs, deterministic=True):
        action = original(self, obs, deterministic=deterministic)
        if mutation == "rng": self.rng.random()
        else: self.parameters["wa"][0] += 1e-8
        return action
    monkeypatch.setattr(demo.TemporalPolicy, "act", drift)
    with pytest.raises(RuntimeError, match="weights, RNG or input changed"): collect(completed)


def test_existing_output_refused_before_verification(completed, tmp_path, monkeypatch):
    output = tmp_path / "existing.html"; output.write_bytes(b"user content")
    monkeypatch.setattr(demo.evaluator, "verify_completed_evaluation", forbidden)
    monkeypatch.setattr(demo, "collect_stored_replay", forbidden)
    with pytest.raises(FileExistsError): demo.run_demo(**args(completed, output))
    assert output.read_bytes() == b"user content"


def test_incomplete_evaluation_refused_before_replay(completed, tmp_path, monkeypatch):
    evaluation = tmp_path / "incomplete"; shutil.copytree(completed["output"], evaluation)
    (evaluation / "aggregate.json").unlink()
    monkeypatch.setattr(demo, "collect_stored_replay", forbidden)
    output = tmp_path / "no.html"
    with pytest.raises(FileNotFoundError): demo.run_demo(**{**args(completed, output), "evaluation": evaluation})
    assert not output.exists()


def test_less_than_two_scored_cases_refused_before_replay(completed, tmp_path, monkeypatch):
    evaluation = tmp_path / "incomplete"; shutil.copytree(completed["output"], evaluation)
    path = evaluation / "validation-capability.json.gz"
    report = json.loads(gzip.decompress(path.read_bytes()))
    report["methods"]["temporal_499"]["episodes"].pop()
    path.write_bytes(gzip.compress(demo.evaluator._bytes(report), mtime=0))
    monkeypatch.setattr(demo.evaluator, "verify_completed_evaluation", lambda *a, **k: {})
    monkeypatch.setattr(demo, "collect_stored_replay", forbidden)
    output = tmp_path / "no.html"
    with pytest.raises(ValueError, match="first two"): demo.run_demo(**{**args(completed, output), "evaluation": evaluation})
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["source", "evidence", "rng"])
def test_render_drift_refused_before_output(completed, tmp_path, monkeypatch, mutation):
    original, source = demo.export_balanced_html, demo.implementation_fingerprints()
    def drift(*a, **k):
        original(*a, **k)
        if mutation == "source": monkeypatch.setattr(demo, "implementation_fingerprints", lambda: {**source, "changed.py": "0" * 64})
        elif mutation == "evidence":
            read = demo._file_hash
            target = completed["runs"][499] / "last/arrays.npz"
            monkeypatch.setattr(demo, "_file_hash", lambda path: "0" * 64 if path == target else read(path))
        else:
            read = demo._policy_state
            monkeypatch.setattr(demo, "_policy_state", lambda policy: {**read(policy), "rng_sha256": "changed"})
    monkeypatch.setattr(demo, "export_balanced_html", drift)
    output = tmp_path / "no.html"
    with pytest.raises(RuntimeError, match="source/checkpoint/evidence/RNG"): demo.run_demo(**args(completed, output))
    assert not output.exists()


def test_output_race_preserves_other_writer(completed, tmp_path, monkeypatch):
    output = tmp_path / "race.html"
    original = demo.export_balanced_html
    def race(*a, **k):
        original(*a, **k); output.write_bytes(b"other writer")
    monkeypatch.setattr(demo, "export_balanced_html", race)
    with pytest.raises(FileExistsError): demo.run_demo(**args(completed, output))
    assert output.read_bytes() == b"other writer"


def test_html_data_escaping_uses_frozen_viewer(completed, tmp_path):
    replay = collect(completed)
    replay["catalogue"][0]["label"] = "</script><script>attack & text</script>"
    output = tmp_path / "escaped.html"
    demo.export_balanced_html([replay], output, checkpoint="offline test", weights_sha256="0" * 64)
    text = output.read_text(encoding="utf-8")
    assert "<script>attack" not in text and "\\u003c/script\\u003e" in text
