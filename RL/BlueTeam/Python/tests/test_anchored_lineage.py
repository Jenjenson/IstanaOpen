"""Pinned additive pilot lineage: exact bytes, compact ranges, zero sampling."""
import builtins
from copy import deepcopy
import gzip
import hashlib
import json

import pytest

from triad_rl import anchored_lineage as lineage


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "repo" / "RL" / "BlueTeam"
    inherited = {"schema": lineage.prior.SCHEMA,
                 "source_checkpoint": {"path": "Checkpoints/balanced-v3-candidate", "weights_sha256": "f" * 64},
                 "publication_manifests": [{"path": "fixture-prior.json", "sha256": "a" * 64}],
                 "evidence_artifact_count": 1, "evidence_inventory_sha256": "b" * 64,
                 "consumed_seed_ranges": [{"start": 30_000_000, "count": 5}],
                 "reserved_final_seed_ranges": [{"start": 2 * 10**15 + n * 1000, "count": 3} for n in range(3)]}
    monkeypatch.setattr(lineage.prior, "load_published_lineage", lambda blue_root=None: deepcopy(inherited))
    training = {"scenario_seed_start": 880_000_000, "episodes": 4}
    validation = {p: {"start": 10**15 + 888_000_000 + i * 1000, "count": 2} for i, p in enumerate(lineage.PROFILES)}
    protocol = {"training": training, "validation": {"scenario_seed_ranges": validation},
                "reserved_final_tests_unopened": dict(zip(lineage.PROFILES, inherited["reserved_final_seed_ranges"])),
                "frozen_reference": {"weights_sha256": "f" * 64}}
    documents = {"protocol.json": protocol,
                 "evaluation/aggregate.json": {"final_test_accessed": False,
                                               "seed_provenance": {"pilot_validation": validation}}}
    for arm in lineage.ARMS:
        documents[f"training/{arm}/config.json"] = {"training": training, "inherited_lineage": inherited}
        documents[f"training/{arm}/summary.json"] = {"completed_episodes": 4, "seed_provenance": {
            "training": {"start": training["scenario_seed_start"], "count": 4}, "inherited": inherited}}
    for profile, row in validation.items():
        documents[f"evaluation/validation-{profile}.json.gz"] = {
            "stage": "validation", "training_performed": False,
            "protocol": {"independent_final_test_evidence": False},
            "seed_provenance": {"evaluation": row},
            "methods": {"fixture": {"episodes": [{"seed": seed} for seed in range(row["start"], row["start"] + 2)]}}}
    files = {lineage.PREFIX + name: gzip.compress(encoded(value), mtime=0) if name.endswith(".gz") else encoded(value)
             for name, value in documents.items()}
    files[lineage.PREFIX + "training/fixture/arrays.npz"] = b"opaque numeric checkpoint bytes"
    files[lineage.PREFIX + "training/fixture/training.jsonl"] = b'{"event":"complete"}\n'
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    sources = [root / "Python" / "fixture.py", root.parent.parent / "Tools" / "fixture.py"]
    for source in sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"# frozen source\n")
    manifest = {"schema": "triad.credit_pilot_publication.v1", "experiment": "credit-v4-pilot", "stage": "validation",
                "final_test_accessed": False, "default_policy_changed": False,
                "files": [{"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} for name, raw in files.items()],
                "evaluation_implementation_sha256": {"fixture.py": hashlib.sha256(sources[0].read_bytes()).hexdigest()},
                "archive_implementation_sha256": {"Tools/fixture.py": hashlib.sha256(sources[1].read_bytes()).hexdigest()}}
    manifest_path = root / lineage.MANIFEST_PATH
    def anchor():
        manifest_path.write_bytes(encoded(manifest))
        monkeypatch.setattr(lineage, "MANIFEST_SHA256", hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    monkeypatch.setattr(lineage, "ARTIFACT_COUNT", len(files))
    anchor()
    return root, inherited, sources, manifest, anchor


def test_compact_extension_keeps_v3_source_and_reservations_without_simulation(bundle, monkeypatch):
    root, inherited, _, manifest, _ = bundle
    original = builtins.__import__
    def no_simulation(name, *args, **kwargs):
        if name.startswith(("triad_rl.adaptive_env", "triad_rl.robust_scenarios", "triad_rl.balanced_policy")):
            pytest.fail("Lineage must never import a simulator or policy")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_simulation)
    value = lineage.load_published_lineage(root)
    assert value["schema"] == lineage.SCHEMA
    assert value["source_checkpoint"] == inherited["source_checkpoint"]
    assert value["reserved_final_seed_ranges"] == inherited["reserved_final_seed_ranges"]
    assert value["consumed_seed_ranges"] == [*inherited["consumed_seed_ranges"], {"start": 880_000_000, "count": 4},
        *[{"start": 10**15 + 888_000_000 + i * 1000, "count": 2} for i in range(3)]]
    assert value["evidence_artifact_count"] == inherited["evidence_artifact_count"] + len(manifest["files"]) + 1
    assert len(value["publication_manifests"]) == 2 and str(root) not in json.dumps(value)
    assert "prior_lineage_sha256" in value
    assert "prior_lineage_sha256" not in inherited


@pytest.mark.parametrize("kind", ["manifest", "artifact", "missing", "evaluation_source", "archive_source"])
def test_modified_or_missing_bytes_fail_closed(bundle, kind):
    root, _, sources, _, _ = bundle
    path = (root / lineage.MANIFEST_PATH if kind == "manifest" else
            sources[0] if kind == "evaluation_source" else sources[1] if kind == "archive_source" else
            root / lineage.PREFIX / "training/fixture/arrays.npz")
    if kind == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises((ValueError, FileNotFoundError)):
        lineage.load_published_lineage(root)


@pytest.mark.parametrize("mutation", ["reservation", "incomplete", "case_seed"])
def test_reanchored_fixture_with_inconsistent_exposure_fails(bundle, mutation):
    root, _, _, manifest, anchor = bundle
    name = {"reservation": "protocol.json", "incomplete": "training/shared_critic/summary.json",
            "case_seed": "evaluation/validation-normal.json.gz"}[mutation]
    path = root / lineage.PREFIX / name
    value = json.loads(gzip.decompress(path.read_bytes()) if name.endswith(".gz") else path.read_bytes())
    if mutation == "reservation":
        value["reserved_final_tests_unopened"]["normal"]["start"] += 10
    elif mutation == "incomplete":
        value["completed_episodes"] = 3
    else:
        value["methods"]["fixture"]["episodes"][0]["seed"] += 1
    raw = gzip.compress(encoded(value), mtime=0) if name.endswith(".gz") else encoded(value)
    path.write_bytes(raw)
    entry = next(row for row in manifest["files"] if row["path"] == lineage.PREFIX + name)
    entry.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    anchor()
    with pytest.raises(ValueError):
        lineage.load_published_lineage(root)


def test_source_drift_during_extraction_is_rejected(bundle, monkeypatch):
    root, _, sources, _, _ = bundle
    original = lineage._pilot_ranges
    def drift(documents, inherited):
        value = original(documents, inherited)
        sources[0].write_bytes(b"# changed during read\n")
        return value
    monkeypatch.setattr(lineage, "_pilot_ranges", drift)
    with pytest.raises(RuntimeError, match="evidence changed"):
        lineage.load_published_lineage(root)


def test_prior_lineage_drift_is_rejected(bundle, monkeypatch):
    root, inherited, _, _, _ = bundle
    calls = []
    def drift(blue_root=None):
        calls.append(True)
        return deepcopy(inherited) if len(calls) == 1 else {**inherited, "scope": "changed"}
    monkeypatch.setattr(lineage.prior, "load_published_lineage", drift)
    with pytest.raises(RuntimeError, match="Prior published lineage changed"):
        lineage.load_published_lineage(root)


def test_disjoint_checker_covers_new_old_and_reserved_intervals_and_preserves_input(bundle):
    value = lineage.load_published_lineage(bundle[0])
    before = deepcopy(value)
    for row in [*value["consumed_seed_ranges"], *value["reserved_final_seed_ranges"]]:
        with pytest.raises(ValueError, match="overlap"):
            lineage.assert_disjoint(row["start"] - 1, 2, value)
    lineage.assert_disjoint(880_000_004, 5, value)
    assert value == before
    with pytest.raises(ValueError, match="schema"):
        lineage.assert_disjoint(1, 1, {**value, "schema": lineage.prior.SCHEMA})


def test_actual_published_pilot_extends_evidence_without_sampling():
    value = lineage.load_published_lineage()
    assert value["evidence_artifact_count"] == 141
    assert len(value["publication_manifests"]) == 4
    assert len(value["consumed_seed_ranges"]) == 34
    assert value["publication_manifests"][-1]["artifact_count"] == 33
    assert len(value["reserved_final_seed_ranges"]) == 3
    assert len(json.dumps(value)) < 4500
    assert value["source_checkpoint"]["weights_sha256"] == lineage.prior.SOURCE_WEIGHTS
    for row in value["reserved_final_seed_ranges"]:
        assert row not in value["consumed_seed_ranges"]
