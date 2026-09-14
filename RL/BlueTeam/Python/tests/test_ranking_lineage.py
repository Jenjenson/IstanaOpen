"""Compact ranking references extend pinned history without sampling scenarios."""
import builtins
from copy import deepcopy
import gzip
import hashlib
import json

import pytest

from test_anchored_lineage import bundle as inherited_bundle
from triad_rl import ranking_lineage as lineage


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


@pytest.fixture
def bundle(inherited_bundle, monkeypatch):
    root, _, sources, old_manifest, _ = inherited_bundle
    inherited = lineage.prior.load_published_lineage(root)
    monkeypatch.setattr(lineage.prior, "load_published_lineage", lambda blue_root=None: deepcopy(inherited))
    training = {"scenario_seed_start": 881_000_000, "episodes": 6}
    validation = {profile: {"start": 10**15 + 889_000_000 + index * 1000, "count": 3}
                  for index, profile in enumerate(lineage.PROFILES)}
    protocol = {"training": training, "validation": {"scenario_seed_ranges": validation},
                "reserved_final_tests_unopened": dict(zip(lineage.PROFILES, inherited["reserved_final_seed_ranges"])),
                "frozen_reference": {"weights_sha256": inherited["source_checkpoint"]["weights_sha256"]}}
    documents = {"protocol.json": protocol,
                 "evaluation/aggregate.json": {"final_test_accessed": False,
                                               "seed_provenance": {"pilot_validation": validation}}}
    for arm in lineage.ARMS:
        documents[f"training/{arm}/config.json"] = {"training": training, "inherited_lineage": inherited}
        documents[f"training/{arm}/summary.json"] = {"completed_episodes": 6, "seed_provenance": {
            "training": {"start": training["scenario_seed_start"], "count": 6}, "inherited": inherited}}
    for profile, row in validation.items():
        documents[f"evaluation/validation-{profile}.json.gz"] = {
            "stage": "validation", "profile": profile, "training_performed": False,
            "protocol": {"independent_final_test_evidence": False}, "seed_provenance": {"evaluation": row},
            "methods": {method: {"episodes": [{"seed": seed} for seed in range(row["start"], row["start"] + row["count"])]}
                        for method in lineage.METHODS}}
    files = {lineage.PREFIX + name: gzip.compress(encoded(value), mtime=0) if name.endswith(".gz") else encoded(value)
             for name, value in documents.items()}
    files[lineage.PREFIX + "training/fixture/arrays.npz"] = b"opaque hash-bound numeric fixture"
    files[lineage.PREFIX + "training/fixture/training.jsonl"] = b'{"event":"complete"}\n'
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    manifest = {"schema": "triad.anchored_pilot_publication.v1", "experiment": "anchored-v4-pilot", "stage": "validation",
                "final_test_accessed": False, "default_policy_changed": False,
                "files": [{"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} for name, raw in files.items()],
                "evaluation_implementation_sha256": old_manifest["evaluation_implementation_sha256"],
                "archive_implementation_sha256": old_manifest["archive_implementation_sha256"]}
    manifest_path = root / lineage.MANIFEST_PATH
    def anchor():
        manifest_path.write_bytes(encoded(manifest))
        monkeypatch.setattr(lineage, "MANIFEST_SHA256", hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    monkeypatch.setattr(lineage, "ARTIFACT_COUNT", len(files))
    anchor()
    return root, inherited, sources, manifest, anchor


def test_compact_reference_only_lineage_keeps_complete_history_and_reservations(bundle, monkeypatch):
    root, inherited, _, manifest, _ = bundle
    original = builtins.__import__
    def no_simulation(name, *args, **kwargs):
        if name.startswith(("triad_rl.adaptive_env", "triad_rl.robust_scenarios", "triad_rl.balanced_policy")):
            pytest.fail("Ranking lineage may not import a simulator or policy")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_simulation)
    value = lineage.load_published_lineage(root)
    assert value["schema"] == lineage.SCHEMA and value["source_checkpoint_role"] == lineage.REFERENCE_ROLE
    assert value["source_checkpoint"] == inherited["source_checkpoint"]
    assert value["reserved_final_seed_ranges"] == inherited["reserved_final_seed_ranges"]
    assert value["consumed_seed_ranges"] == lineage.utils.merge_ranges([
        *inherited["consumed_seed_ranges"], {"start": 881_000_000, "count": 6},
        *[{"start": 10**15 + 889_000_000 + index * 1000, "count": 3} for index in range(3)]])
    assert value["evidence_artifact_count"] == inherited["evidence_artifact_count"] + len(manifest["files"]) + 1
    assert len(value["publication_manifests"]) == len(inherited["publication_manifests"]) + 1
    assert value["prior_pilot"]["publication_manifest"]["path"] == lineage.MANIFEST_PATH
    assert str(root) not in json.dumps(value) and "source_checkpoint_role" not in inherited


@pytest.mark.parametrize("kind", ["manifest", "artifact", "missing", "evaluation_source", "archive_source"])
def test_changed_or_missing_pinned_bytes_are_rejected(bundle, kind):
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


@pytest.mark.parametrize("mutation", ["reservation", "incomplete", "case_seed", "missing_method", "wrong_profile"])
def test_reanchored_fixture_must_have_complete_consistent_exposure(bundle, mutation):
    root, _, _, manifest, anchor = bundle
    name = ("protocol.json" if mutation == "reservation" else "training/shared_critic/summary.json"
            if mutation == "incomplete" else "evaluation/validation-normal.json.gz")
    path = root / lineage.PREFIX / name
    value = json.loads(gzip.decompress(path.read_bytes()) if name.endswith(".gz") else path.read_bytes())
    if mutation == "reservation":
        value["reserved_final_tests_unopened"]["normal"]["start"] += 10
    elif mutation == "incomplete":
        value["completed_episodes"] = 5
    elif mutation == "case_seed":
        value["methods"]["shared_critic"]["episodes"][0]["seed"] += 1
    elif mutation == "missing_method":
        del value["methods"]["greedy_public"]
    else:
        value["profile"] = "stress"
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


def test_inherited_lineage_drift_is_rejected(bundle, monkeypatch):
    root, inherited, _, _, _ = bundle
    calls = []
    def drift(blue_root=None):
        calls.append(True)
        return deepcopy(inherited) if len(calls) == 1 else {**inherited, "scope": "changed"}
    monkeypatch.setattr(lineage.prior, "load_published_lineage", drift)
    with pytest.raises(RuntimeError, match="Inherited lineage changed"):
        lineage.load_published_lineage(root)


def test_disjoint_checks_cover_all_exposure_and_do_not_mutate_input(bundle):
    value = lineage.load_published_lineage(bundle[0])
    before = deepcopy(value)
    for row in [*value["consumed_seed_ranges"], *value["reserved_final_seed_ranges"]]:
        with pytest.raises(ValueError, match="overlap"):
            lineage.assert_disjoint(row["start"] - 1, 2, value)
    lineage.assert_disjoint(881_000_006, 4, value)
    assert value == before
    with pytest.raises(ValueError, match="schema"):
        lineage.assert_disjoint(1, 1, {**value, "schema": lineage.prior.SCHEMA})


def test_actual_five_publications_are_compact_reference_evidence_without_sampling():
    value = lineage.load_published_lineage()
    assert value["evidence_artifact_count"] == 175
    assert len(value["publication_manifests"]) == 5
    assert len(value["consumed_seed_ranges"]) == 38
    assert len(value["reserved_final_seed_ranges"]) == 3
    assert value["publication_manifests"][-1]["artifact_count"] == 33
    assert value["source_checkpoint"]["weights_sha256"] == lineage.utils.SOURCE_WEIGHTS
    assert value["source_checkpoint_role"] == lineage.REFERENCE_ROLE
    assert len(json.dumps(value)) < 5000
    assert all(row not in value["consumed_seed_ranges"] for row in value["reserved_final_seed_ranges"])
