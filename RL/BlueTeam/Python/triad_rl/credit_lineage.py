"""Compact, hash-pinned published exposure for a future credit-assignment pilot.

Reads evidence only: no environment/policy imports, inference or sampling.
The three pinned manifests transitively retain every exact artifact hash;
new checkpoints need not duplicate their deeply nested provenance. Consumed
scenario seeds and unopened final reservations are deliberately separate.
This is a union of published evidence, not a claim to enumerate unlogged
experiments or unit-test executions on every workstation.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re


SCHEMA = "triad.credit_published_lineage.v1"
SOURCE_WEIGHTS = "a48d3c3a5b43e99dd4545cfb7321eb64077dceb09432a60658c2649dd19f21c0"
MANIFESTS = {
    "adaptive-v1": "cf3e69a664c8b0d9331199169b3e65351d323d6d35570098ec4aefc022bd74e2",
    "robust-v2": "5ad308a6a4516ac30cf6aafdcc7ab7d92f019be18a27992405edb9aab6ac1393",
    "balanced-v3": "6c05b4004f1cc427bebb253a08afefe70bec68b821fb44d18737c285ab9a6e2a",
}
BLUE_ROOT = Path(__file__).resolve().parents[2]
MAX_JSON_BYTES = 256 * 1024 * 1024


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _strict(data):
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("Lineage JSON exceeds the bounded evidence size")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate lineage evidence JSON key")
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=unique)
    _canonical(value)
    return value


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def merge_ranges(ranges):
    """Canonical union of overlapping/adjacent nonnegative integer ranges."""
    intervals = []
    for row in ranges:
        if not isinstance(row, dict) or "start" not in row or "count" not in row:
            raise ValueError("Seed ranges require integer start/count")
        start, count = _integer(row["start"], "start"), _integer(row["count"], "count")
        if count:
            intervals.append((start, start + count))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [{"start": start, "count": end - start} for start, end in merged]


def _collect(value, consumed, reserved):
    """Collect declared ranges and recorded scenario seeds, never policy RNGs."""
    if isinstance(value, dict):
        if "start" in value and "count" in value:
            row = {"start": _integer(value["start"], "start"), "count": _integer(value["count"], "count")}
            consumed.append(row)
        if "scenario_seed" in value:
            consumed.append({"start": _integer(value["scenario_seed"], "scenario_seed"), "count": 1})
        # Config.seed and policy_seed are RNG/run identifiers, not scenario
        # seeds. A scalar seed is consumed only in a recorded case/replay.
        if "seed" in value and {"scenario", "scenario_sha256", "targets", "case_metadata"}.intersection(value):
            consumed.append({"start": _integer(value["seed"], "scenario seed"), "count": 1})
        for key, child in value.items():
            if key == "reserved_final_tests":
                _collect(child, reserved, [])
            elif key not in ("rng_state", "policy_rng_state"):
                if isinstance(child, (dict, list)):
                    _collect(child, consumed, reserved)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                _collect(child, consumed, reserved)


def _decode(path, data):
    if path.name.endswith(".json.gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
            return [_strict(handle.read(MAX_JSON_BYTES + 1))]
    if path.suffix == ".json":
        return [_strict(data)]
    if path.suffix == ".jsonl":
        return [_strict(line) for line in data.splitlines() if line.strip()]
    if path.suffix == ".html":
        blocks = re.findall(rb'<script\b(?=[^>]*\bid=[\"\']replay-data[\"\'])[^>]*>(.*?)</script\s*>', data, re.S | re.I)
        if len(blocks) != 1:
            raise ValueError("Published demo must contain exactly one replay-data payload")
        return [_strict(blocks[0])]
    if path.suffix == ".npz":
        return []  # Exact bytes are still bound; no numeric-array deserialization.
    raise ValueError(f"Unsupported publication evidence extension: {path.suffix}")


def _safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("Lineage artifact must be a portable relative path")
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError("Lineage artifact escapes its repository root")
    return path


def load_published_lineage(blue_root=None):
    """Verify all three immutable bundles and return portable compact lineage.

    Manifest hashes are explicit release anchors, not mutable discovery. Each
    listed artifact and source implementation is checked before and after
    extraction. The inventory digest binds the transitive manifest expansion.
    """
    root = (Path(blue_root) if blue_root is not None else BLUE_ROOT).resolve()
    consumed, reserved, manifests, inventory, snapshots = [], [], [], {}, {}
    selected = None
    for release, expected in MANIFESTS.items():
        name = f"Results/{release}/artifact-manifest.json"
        path = _safe_path(root, name)
        data = path.read_bytes()
        if _sha(data) != expected:
            raise ValueError(f"Pinned {release} publication manifest changed")
        manifest = _strict(data)
        snapshots[path] = expected
        manifests.append({"path": name, "sha256": expected, "bytes": len(data), "artifact_count": len(manifest["files"])})
        inventory[name] = {"path": name, "sha256": expected, "bytes": len(data)}
        for artifact in manifest["files"]:
            artifact_path = _safe_path(root, artifact["path"])
            blob = artifact_path.read_bytes()
            if _sha(blob) != artifact["sha256"] or len(blob) != artifact["bytes"]:
                raise ValueError(f"Published lineage artifact changed: {artifact['path']}")
            entry = {key: artifact[key] for key in ("path", "sha256", "bytes")}
            if entry["path"] in inventory and inventory[entry["path"]] != entry:
                raise ValueError("Publication manifests disagree about an artifact")
            inventory[entry["path"]] = entry
            snapshots[artifact_path] = entry["sha256"]
            for value in _decode(artifact_path, blob):
                _collect(value, consumed, reserved)
                # Completed run configs independently cover the full run,
                # not just a validation-selected checkpoint's early prefix.
                if artifact_path.name == "config.json" and "training" in artifact_path.parts:
                    seed = _integer(value["seed"], "training run seed")
                    count = _integer(value["target_episodes"], "training episodes")
                    consumed.append({"start": seed * 1_000_000, "count": count})
            if entry["path"] == "Checkpoints/balanced-v3-candidate/checkpoint.json":
                selected = _strict(blob)
        source_files = manifest.get("source_files_sha256", {})
        if release == "adaptive-v1":
            source_files = {f"RL/BlueTeam/Python/triad_rl/{k}": v
                            for k, v in manifest["training_implementation_sha256"].items()}
        for source, digest in source_files.items():
            source_path = _safe_path(root.parent.parent, source)
            if _sha(source_path.read_bytes()) != digest:
                raise ValueError(f"Published lineage implementation changed: {source}")
            snapshots[source_path] = digest
    if not selected or selected.get("weights_sha256") != SOURCE_WEIGHTS:
        raise ValueError("Published balanced source weights differ from the declared reference")
    if any(_sha(path.read_bytes()) != digest for path, digest in snapshots.items()):
        raise RuntimeError("Published evidence changed while deriving compact lineage")
    return {
        "schema": SCHEMA,
        "source_checkpoint": {"path": "Checkpoints/balanced-v3-candidate", "weights_sha256": SOURCE_WEIGHTS,
                              "checkpoint_files_sha256": {name: inventory[f"Checkpoints/balanced-v3-candidate/{name}"]["sha256"]
                                                          for name in ("checkpoint.json", "arrays.npz")},
                              "selection_report": {"path": "Results/balanced-v3/selection.json",
                                                   "sha256": inventory["Results/balanced-v3/selection.json"]["sha256"]}},
        "publication_manifests": manifests,
        "evidence_inventory_sha256": _sha(_canonical([inventory[k] for k in sorted(inventory)])),
        "evidence_artifact_count": len(inventory),
        "consumed_seed_ranges": merge_ranges(consumed),
        "reserved_final_seed_ranges": merge_ranges(reserved),
        "scope": "Published v1/v2/v3 scenario exposure, including complete runs, selection, old finals, diagnostics and replays; excludes policy RNG identifiers and unlogged workstation/test history.",
        "reference_contract": "Pinned manifests transitively identify every exact artifact and source hash; full evidence remains in those release bundles.",
    }


def assert_disjoint(start, count, lineage, *, label="proposed scenarios"):
    """Reject overlap with either consumed evidence or unopened final suites."""
    start, count = _integer(start, "start"), _integer(count, "count")
    if not count:
        raise ValueError("Proposed scenario count must be positive")
    if lineage.get("schema") != SCHEMA:
        raise ValueError("Unsupported compact published lineage schema")
    for kind in ("consumed_seed_ranges", "reserved_final_seed_ranges"):
        for row in merge_ranges(lineage[kind]):
            if max(start, row["start"]) < min(start + count, row["start"] + row["count"]):
                raise ValueError(f"{label} overlap {kind}: {row['start']} + {row['count']}")
