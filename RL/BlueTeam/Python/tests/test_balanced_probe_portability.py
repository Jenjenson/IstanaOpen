"""Exact archived identities plus portable numeric generator/actor verification."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import verify_balanced_probes as portable


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    path = tmp_path_factory.mktemp("portable-probe") / "inputs.json.gz"
    # This fixture is producer-independent: CI normally uses the supplement
    # captured on Windows, while local pre-publication tests create it there.
    published = portable.BLUE_ROOT / portable.RESULTS / "portable-public-inputs.json.gz"
    if published.exists():
        path = path.with_name(published.name)
        shutil.copyfile(published, path)
        shutil.copyfile(portable._manifest_path(published), portable._manifest_path(path))
    else:
        portable.export_inputs(output=path)
    return path


def copy_archive(source, directory):
    path = directory / source.name
    shutil.copyfile(source, path)
    shutil.copyfile(portable._manifest_path(source), portable._manifest_path(path))
    return path


def rewrite_corpus(path, mutate):
    corpus = json.loads(gzip.decompress(path.read_bytes()))
    mutate(corpus)
    raw = gzip.compress(json.dumps(corpus, allow_nan=False).encode(), mtime=0)
    path.write_bytes(raw)
    manifest_path = portable._manifest_path(path)
    manifest = json.loads(manifest_path.read_text())
    manifest["corpus"].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_actual_archive_recomputes_all_three_original_actors_and_preserves_bytes(archive):
    before = archive.read_bytes(), portable._manifest_path(archive).read_bytes()
    audit = portable.verify_inputs(archive)
    assert audit["schema"] == portable.AUDIT_SCHEMA
    assert audit["verified"] and audit["observations"] == 60 and audit["probes"] == 3
    assert audit["physics_rollouts"] == 0 and audit["policy_rng_unchanged"]
    assert audit["original_public_input_hashes_exact"]
    assert before == (archive.read_bytes(), portable._manifest_path(archive).read_bytes())


def test_each_archived_public_hash_matches_all_original_probes_exactly(archive):
    corpus = json.loads(gzip.decompress(archive.read_bytes()))
    metadata, probes = portable._frozen_evidence()
    assert corpus["metadata"] == metadata
    for index, row in enumerate(corpus["observations"]):
        assert row["public_input_sha256"] == portable._public_hash(row["observation"])
        assert all(row["public_input_sha256"] == probe["records"][index]["public_input_sha256"] for probe in probes)
        decoded = portable._decode_observation(row["observation"])
        assert decoded["option_features"].dtype == np.float32
        assert decoded["action_mask"].dtype == bool


def public_only_factory(real, mutate=None):
    class PublicOnly:
        def __init__(self, **kwargs):
            self._wrapped = real(**kwargs)

        def reset(self, **kwargs):
            observation = self._wrapped.reset(**kwargs)
            if mutate:
                mutate(observation)
            return observation

        def step(self, *args, **kwargs):
            pytest.fail("Portable verification must never perform a physics rollout")

        def __getattr__(self, name):
            pytest.fail(f"Private/non-public environment access: {name}")
    return PublicOnly


def test_verifier_only_calls_public_reset_and_accepts_last_bit_generator_drift(archive, monkeypatch):
    real = portable.RobustPlacementEnv
    def jitter(observation):
        observation["state"]["sites"][0][0] += 1e-13
        observation["options"][0]["position"][0] += 1e-13
    monkeypatch.setattr(portable, "RobustPlacementEnv", public_only_factory(real, jitter))
    assert portable.verify_inputs(archive)["verified"]


@pytest.mark.parametrize("change", ["coordinate", "mask", "sensor_id", "feature"])
def test_material_runtime_generator_or_discrete_drift_rejected(archive, monkeypatch, change):
    real = portable.RobustPlacementEnv
    def mutate(observation):
        if change == "coordinate":
            observation["state"]["sites"][0][0] += .001
        elif change == "mask":
            observation["action_mask"][0] = not observation["action_mask"][0]
        elif change == "sensor_id":
            observation["options"][0]["sensor_id"] = "changed"
        else:
            observation["option_features"][0, 0] = np.float32(.9)
    monkeypatch.setattr(portable, "RobustPlacementEnv", public_only_factory(real, mutate))
    with pytest.raises(ValueError, match="Runtime public generator"):
        portable.verify_inputs(archive)


def test_export_requires_exact_original_hashes_even_for_tolerable_public_drift(tmp_path, monkeypatch):
    real = portable.RobustPlacementEnv
    def jitter(observation):
        observation["state"]["sites"][0][0] += 1e-13
    monkeypatch.setattr(portable, "RobustPlacementEnv", public_only_factory(real, jitter))
    path = tmp_path / "new" / "inputs.json.gz"
    with pytest.raises(ValueError, match="Producer public input hash mismatch"):
        portable.export_inputs(output=path)
    assert not path.parent.exists()


@pytest.mark.parametrize("change", ["state_last_bit", "catalogue", "hash", "features", "mask_type", "shape",
                                   "names", "private_field", "order", "missing", "source", "schema"])
def test_rehashed_corpus_tampering_still_rejected(archive, tmp_path, change):
    path = copy_archive(archive, tmp_path)
    def mutate(corpus):
        row = corpus["observations"][0]
        if change == "state_last_bit":
            row["observation"]["state"]["sites"][0][0] += 1e-13
        elif change == "catalogue":
            row["observation"]["catalogue"][0]["cost"] += .01
        elif change == "hash":
            row["public_input_sha256"] = "0" * 64
        elif change == "features":
            row["observation"]["option_features"][0][0] = float(np.float32(.9))
        elif change == "mask_type":
            row["observation"]["action_mask"][0] = 1
        elif change == "shape":
            row["observation"]["option_features"][0].pop()
        elif change == "names":
            row["observation"]["feature_names"].reverse()
        elif change == "private_field":
            row["observation"]["scenario"] = {"hidden": True}
        elif change == "order":
            corpus["observations"].reverse()
        elif change == "missing":
            corpus["observations"].pop()
        elif change == "source":
            corpus["metadata"]["verifier_source_sha256"] = "0" * 64
        else:
            corpus["schema"] = "changed"
    rewrite_corpus(path, mutate)
    with pytest.raises(ValueError):
        portable.verify_inputs(path)


@pytest.mark.parametrize("field", ["index", "position", "cost"])
def test_original_actions_are_not_replaced_by_joint_argmax_or_forged_recommendations(archive, monkeypatch, field):
    original = portable.first_action_record
    def changed(*args, **kwargs):
        record = original(*args, **kwargs)
        if field == "position":
            record["first_action"]["position"][0] += 1e-13
        else:
            record["first_action"][field] += 1 if field == "index" else 1e-13
        return record
    monkeypatch.setattr(portable, "first_action_record", changed)
    with pytest.raises(ValueError, match="Actual versioned actor"):
        portable.verify_inputs(archive)


def test_inference_rng_drift_detected(archive, monkeypatch):
    original = portable.first_action_record
    def changed(policy, *args, **kwargs):
        policy.rng.random()
        return original(policy, *args, **kwargs)
    monkeypatch.setattr(portable, "first_action_record", changed)
    with pytest.raises(ValueError, match="weights or RNG"):
        portable.verify_inputs(archive)


@pytest.mark.parametrize("change", ["compressed", "manifest_hash", "manifest_size", "manifest_source", "manifest_audit",
                                   "bool_as_int", "int_as_float", "metadata_int_as_float", "metadata_float_as_int"])
def test_exact_artifact_and_manifest_integrity(archive, tmp_path, change):
    path = copy_archive(archive, tmp_path)
    manifest_path = portable._manifest_path(path)
    manifest = json.loads(manifest_path.read_text())
    if change == "compressed":
        path.write_bytes(path.read_bytes() + b"tampered")
    elif change == "manifest_hash":
        manifest["corpus"]["sha256"] = "0" * 64
    elif change == "manifest_size":
        manifest["corpus"]["bytes"] += 1
    elif change == "manifest_source":
        manifest["metadata"]["verifier_source_sha256"] = "0" * 64
    elif change == "bool_as_int":
        manifest["audit"]["verified"] = 1
    elif change == "int_as_float":
        manifest["corpus"]["bytes"] = float(manifest["corpus"]["bytes"])
    elif change == "metadata_int_as_float":
        manifest["metadata"]["seed_range"]["count"] = 60.0
    elif change == "metadata_float_as_int":
        manifest["metadata"]["tolerance"]["absolute"] = 0
    else:
        manifest["audit"]["probes"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        portable.verify_inputs(path)


def test_rehashed_corpus_metadata_keeps_exact_discrete_types(archive, tmp_path):
    path = copy_archive(archive, tmp_path)
    def mutate(corpus):
        corpus["metadata"]["seed_range"]["count"] = 60.0
    rewrite_corpus(path, mutate)
    with pytest.raises(ValueError, match="binding"):
        portable.verify_inputs(path)


def test_bounded_gzip_expansion(archive, monkeypatch):
    monkeypatch.setattr(portable, "MAX_EXPANDED_BYTES", 128)
    with pytest.raises(ValueError, match="expanded size"):
        portable.verify_inputs(archive)


def test_bounded_compressed_input(archive, monkeypatch):
    monkeypatch.setattr(portable, "MAX_COMPRESSED_BYTES", 1)
    with pytest.raises(ValueError, match="bounded size"):
        portable.verify_inputs(archive)


@pytest.mark.parametrize("token", ['"schema":"duplicate",', '"overflow":1e999,'])
def test_strict_json_manifest_rejects_duplicate_keys_and_overflow(archive, tmp_path, token):
    path = copy_archive(archive, tmp_path)
    manifest = portable._manifest_path(path)
    manifest.write_text("{" + token + manifest.read_text()[1:], encoding="utf-8")
    with pytest.raises(ValueError):
        portable.verify_inputs(path)


def test_export_conflict_does_not_touch_existing_output_or_manifest(archive):
    before = archive.read_bytes(), portable._manifest_path(archive).read_bytes()
    with pytest.raises(FileExistsError):
        portable.export_inputs(output=archive)
    assert before == (archive.read_bytes(), portable._manifest_path(archive).read_bytes())


def test_export_manifest_conflict_leaves_corpus_absent(tmp_path):
    path = tmp_path / "inputs.json.gz"
    manifest = portable._manifest_path(path)
    manifest.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        portable.export_inputs(output=path)
    assert not path.exists() and manifest.read_bytes() == b"existing"


def test_source_drift_before_export_write_leaves_no_directory(tmp_path, monkeypatch, archive):
    original = portable._frozen_evidence
    calls = 0
    def changed():
        nonlocal calls
        calls += 1
        metadata, probes = original()
        if calls > 1:
            metadata["verifier_source_sha256"] = "changed"
        return metadata, probes
    # Reuse the bound producer snapshots to make this preflight-drift test
    # independent of whether the test host reproduces producer libm bits.
    corpus = json.loads(gzip.decompress(archive.read_bytes()))
    by_seed = {row["seed"]: row["observation"] for row in corpus["observations"]}
    class ProducerInputs:
        def __init__(self, **kwargs):
            pass
        def reset(self, *, seed):
            return portable._decode_observation(deepcopy(by_seed[seed]))
    monkeypatch.setattr(portable, "RobustPlacementEnv", ProducerInputs)
    monkeypatch.setattr(portable, "_frozen_evidence", changed)
    path = tmp_path / "new" / "inputs.json.gz"
    with pytest.raises(ValueError, match="changed during export"):
        portable.export_inputs(output=path)
    assert not path.parent.exists()
