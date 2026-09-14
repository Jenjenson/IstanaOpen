"""Read-only published diagnostic audit and six explicitly supplied old cases."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import probe_temporal as probe
from triad_rl import adaptive_env, robust_scenarios
from triad_rl.adaptive_evaluation import canonical_hash


ARTIFACT = probe.BLUE / "Results/temporal-v6-development/reused-probe.json"
ARTIFACT_SHA256 = "1860fb3c1d520484fd868e88cb4044009b8eaab1d468980032a23ca944d4af1d"


def expected_metrics(info):
    return {"timely": 1. - info["breached_fraction"], "detected": info["detected_fraction"],
            "confirmed": info["confirmed_fraction"], "success": float(info["success"]),
            "cost": info["cost"], "return": info["return"], "invalid_actions": info["invalid_actions"]}


def forbidden(*args, **kwargs):
    raise AssertionError("No new scenarios, hidden constructors or unexpected inference")


@pytest.fixture(scope="module")
def evidence():
    data = ARTIFACT.read_bytes()
    assert probe.sha(data) == ARTIFACT_SHA256
    result = json.loads(data)
    reports = {profile: json.loads(gzip.decompress((probe.ARCHIVE / f"evaluation/validation-{profile}.json.gz").read_bytes()))
               for profile in probe.PROFILES}
    return result, reports


@pytest.fixture(autouse=True)
def no_new_cases(monkeypatch):
    monkeypatch.setattr(robust_scenarios, "make_case", forbidden)
    monkeypatch.setattr(adaptive_env, "generate_scenario", forbidden)
    monkeypatch.setattr(adaptive_env.AdaptivePlacementEnv, "__init__", forbidden)


def test_actual_sixty_case_sequence_all_hashes_and_paired_reference_metrics(evidence):
    result, reports = evidence
    assert result["schema"] == "triad.temporal_reused_development_probe.v1"
    assert result["training_performed"] is result["final_test_accessed"] is result["policy_promoted"] is False
    assert result["archive_manifest_sha256"] == probe.MANIFEST_SHA256
    assert result["temporal_config"] == asdict(probe.TemporalConfig())
    assert len(result["episodes"]) == 60
    assert Counter(row["profile"] for row in result["episodes"]) == dict.fromkeys(probe.PROFILES, 20)
    assert len(result["source_sha256"]) == 7 and len(result["input_sha256"]) == 4
    for path, digest in {**result["source_sha256"], **result["input_sha256"]}.items():
        assert probe.sha((probe.BLUE / path).read_bytes()) == digest
    for offset, profile in enumerate(probe.PROFILES):
        for index, row in enumerate(result["episodes"][offset * 20:(offset + 1) * 20]):
            assert row["profile"] == profile
            parent = reports[profile]["methods"]["greedy_public"]["episodes"][index]
            assert row["seed"] == parent["seed"]
            assert row["parent_scenario_sha256"] == parent["scenario_sha256"] == canonical_hash(parent["scenario"])
            assert set(row["references"]) == set(probe.REFERENCE_METHODS)
            for method in probe.REFERENCE_METHODS:
                other = reports[profile]["methods"][method]["episodes"][index]
                assert (other["seed"], other["scenario_sha256"]) == (row["seed"], row["parent_scenario_sha256"])
                assert row["references"][method] == expected_metrics(other["metrics"])
            assert row["temporal_public_control"]["invalid_actions"] == 0
            assert all(type(action) is int for action in row["actions"])


def test_actual_all_summaries_recomputed_from_paired_records(evidence):
    result, _ = evidence
    for method in ("temporal_public_control", *probe.REFERENCE_METHODS):
        profiles = {}
        for profile in probe.PROFILES:
            rows = [row[method] if method == "temporal_public_control" else row["references"][method]
                    for row in result["episodes"] if row["profile"] == profile]
            assert all(set(row) == set(rows[0]) for row in rows)
            assert all(np.isfinite(list(row.values())).all() for row in rows)
            profiles[profile] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
            assert result["summaries"][method][profile] == pytest.approx(profiles[profile], abs=1e-12, rel=1e-12)
        expected = {key: float(np.mean([profiles[p][key] for p in probe.PROFILES])) for key in profiles[probe.PROFILES[0]]}
        assert result["summaries"][method]["equal_profile"] == pytest.approx(expected, abs=1e-12, rel=1e-12)


def test_first_two_existing_cases_per_profile_replay_exactly(evidence, monkeypatch):
    result, reports = evidence
    before = ARTIFACT.read_bytes()
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    actor = probe.TemporalPublicGreedy()
    for offset, profile in enumerate(probe.PROFILES):
        for index in range(2):
            recorded = result["episodes"][offset * 20 + index]
            scenario = deepcopy(reports[profile]["methods"]["greedy_public"]["episodes"][index]["scenario"])
            catalogue = scenario.pop("evaluation_catalogue")
            scenario.pop("curriculum_metadata")
            env = probe.TemporalPlacementEnv(scenario=scenario, catalogue=catalogue, config=result["temporal_config"])
            observation, actions = env.observe(), []
            while not env.done:
                action = actor.act(observation)
                actions.append(action)
                observation, _, _, info = env.step(action)
            assert actions == recorded["actions"]
            assert env.placements == recorded["placements"]
            assert expected_metrics(info) == pytest.approx(recorded["temporal_public_control"], abs=1e-12, rel=1e-12)
    assert ARTIFACT.read_bytes() == before


def fake_driver(monkeypatch):
    seen = []
    class Actor:
        def act(self, observation): return 0
    class Env:
        def __init__(self, *, scenario, catalogue, config):
            seen.append(scenario["seed"])
            self.done, self.placements = False, []
        def observe(self): return {"temporal_config": asdict(probe.TemporalConfig())}
        def step(self, action):
            self.done = True
            return self.observe(), 0., True, {"breached_fraction": 1., "detected_fraction": 0., "confirmed_fraction": 0.,
                                             "success": False, "cost": 0., "return": -10.5, "invalid_actions": 0}
    monkeypatch.setattr(probe, "TemporalPlacementEnv", Env)
    monkeypatch.setattr(probe, "TemporalPublicGreedy", Actor)
    return seen


def test_driver_selects_only_fixed_first_twenty_and_refuses_overwrite(evidence, tmp_path, monkeypatch):
    seen = fake_driver(monkeypatch)
    output = tmp_path / "fake-driver.json"
    probe.run_probe(output)
    assert seen == [row["seed"] for row in evidence[0]["episodes"]]
    before = output.read_bytes()
    monkeypatch.setattr(probe, "TemporalPlacementEnv", forbidden)
    with pytest.raises(FileExistsError): probe.run_probe(output)
    assert output.read_bytes() == before


@pytest.mark.parametrize("relative", ["artifact-manifest.json", "evaluation/validation-capability.json.gz"])
def test_archive_preflight_rejects_bad_bytes_before_any_case(relative, tmp_path, monkeypatch):
    original = Path.read_bytes
    target = probe.ARCHIVE / relative
    monkeypatch.setattr(Path, "read_bytes", lambda path: original(path) + b"changed" if path == target else original(path))
    monkeypatch.setattr(probe, "TemporalPlacementEnv", forbidden)
    output = tmp_path / "not-written.json"
    with pytest.raises(ValueError, match="changed"): probe.run_probe(output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["source", "input"])
def test_mid_probe_source_or_input_drift_prevents_output(kind, tmp_path, monkeypatch):
    fake_driver(monkeypatch)
    target = Path(probe.__file__) if kind == "source" else probe.ARCHIVE / "artifact-manifest.json"
    original, calls = Path.read_bytes, Counter()
    def changed(path):
        calls[path] += 1
        return original(path) + (b"changed" if path == target and calls[path] > 1 else b"")
    monkeypatch.setattr(Path, "read_bytes", changed)
    output = tmp_path / "not-written.json"
    with pytest.raises(RuntimeError, match="changed"): probe.run_probe(output)
    assert not output.exists()
