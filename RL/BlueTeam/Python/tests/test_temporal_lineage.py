"""Flat publication hashing, overlap guards and no recursive model validation."""
from collections import Counter
import hashlib
import io
from pathlib import Path

import pytest

from triad_rl import temporal_lineage as lineage


def forbidden(*args, **kwargs):
    raise AssertionError("No recursive lineage loading or new filesystem writes")


@pytest.fixture(scope="module")
def published():
    return lineage.load_published_lineage()


def test_actual_flat_inventory_single_read_and_no_recursive_loaders(monkeypatch):
    from triad_rl import credit_lineage, anchored_lineage, ranking_lineage
    for module in (credit_lineage, anchored_lineage, ranking_lineage):
        monkeypatch.setattr(module, "load_published_lineage", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    original, opened = Path.open, Counter()
    def counted(path, mode="r", *args, **kwargs):
        assert mode == "rb", "The preflight must only read bytes"
        opened[path] += 1
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", counted)
    result = lineage.load_published_lineage()
    assert set(opened.values()) == {1}
    assert len(opened) == 249  # 212 artifact/manifest paths plus37 unique source paths.
    assert result["evidence_artifact_count"] == 212
    assert len(result["publication_manifests"]) == 6
    assert len(result["source_files_sha256"]) == 37
    assert len(result["consumed_seed_ranges"]) == 44
    assert len(result["reserved_final_seed_ranges"]) == 3
    assert set(result["ranking_endpoints"]) == {"403", "404", "405"}
    assert result["source_checkpoint"]["weights_sha256"] == credit_lineage.SOURCE_WEIGHTS


def test_proposed_slots_are_fresh_without_sampling(published):
    # Interval arithmetic only: never instantiate/reset an environment here.
    for seed in (406, 407, 408):
        lineage.assert_disjoint(seed * 1_000_000, 512, published)
    for seed in (995400, 995401, 995402):
        lineage.assert_disjoint(10**15 + seed * 1_000_000, 200, published)
    for row in (*published["consumed_seed_ranges"], *published["reserved_final_seed_ranges"]):
        with pytest.raises(ValueError, match="overlap"):
            lineage.assert_disjoint(row["start"], 1, published)
        with pytest.raises(ValueError, match="overlap"):
            lineage.assert_disjoint(row["start"] - 1, 2, published)


@pytest.mark.parametrize("start,count", [(True, 1), (0, False), (-1, 1), (1, 0), (1., 2)])
def test_invalid_interval_types_and_counts(published, start, count):
    with pytest.raises(ValueError): lineage.assert_disjoint(start, count, published)


@pytest.mark.parametrize("target", [lineage.MANIFEST_PATH,
    "Results/robust-v2/artifact-manifest.json", "Results/ranking-v5-pilot/training/seed-404/last/arrays.npz",
    "Results/balanced-v3/selection.json", "Python/triad_rl/adaptive_env.py"])
def test_any_pinned_manifest_artifact_or_source_byte_change_fails(monkeypatch, target):
    original = Path.open
    chosen = (lineage.BLUE_ROOT / target).resolve()
    def altered(path, mode="r", *args, **kwargs):
        if path == chosen and mode == "rb":
            with original(path, mode, *args, **kwargs) as handle:
                return io.BytesIO(handle.read() + b"tampered")
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", altered)
    with pytest.raises(ValueError, match="Pinned lineage bytes differ"):
        lineage.load_published_lineage()


def test_file_snapshot_detects_mid_read_and_later_changes(tmp_path, monkeypatch):
    path = tmp_path / "evidence"
    path.write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    reader = lineage._Files()
    reader.check(path, digest)
    path.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="changed during"):
        reader.stable()
    real_stamp, calls = lineage._stamp, []
    def changing_stamp(target):
        value = real_stamp(target)
        calls.append(target)
        return (*value[:-1], value[-1] + len(calls))
    monkeypatch.setattr(lineage, "_stamp", changing_stamp)
    with pytest.raises(ValueError, match="changed while"):
        lineage._Files().check(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_bad_pin_cannot_be_bypassed_by_bound_internal_hashes(monkeypatch):
    monkeypatch.setattr(lineage, "MANIFEST_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="Pinned"):
        lineage.load_published_lineage()
