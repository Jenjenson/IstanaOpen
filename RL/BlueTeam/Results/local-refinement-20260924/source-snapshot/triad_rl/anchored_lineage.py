"""Compact published exposure including the completed, unpromoted credit pilot.

Only reads hash-pinned evidence. No policy/environment imports or sampling.
The frozen credit lineage remains unchanged; this additive schema chains its
inventory digest to the exact pilot manifest and its 33 archived artifacts.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import credit_lineage as prior


SCHEMA = "triad.anchored_published_lineage.v1"
MANIFEST_PATH = "Results/credit-v4-pilot/artifact-manifest.json"
MANIFEST_SHA256 = "c144bb680aa2714efc815b6b27b08dd69dfb0a0c98322428e7075641df267613"
ARTIFACT_COUNT = 33
PROFILES = ("normal", "stress", "capability")
ARMS = ("shared_critic", "no_critic_gradient", "paired_stop")
PREFIX = "Results/credit-v4-pilot/"


def _same(left, right):
    return prior._canonical(left) == prior._canonical(right)


def _read(path):
    with path.open("rb") as handle:
        data = handle.read(prior.MAX_JSON_BYTES + 1)
    if len(data) > prior.MAX_JSON_BYTES:
        raise ValueError("Anchored lineage evidence exceeds bounded file size")
    return data


def _pilot_ranges(documents, inherited):
    """Extract actual complete-run/report exposure, not nested reservations."""
    protocol = documents[PREFIX + "protocol.json"]
    train = protocol["training"]
    training = {"start": prior._integer(train["scenario_seed_start"], "training start"),
                "count": prior._integer(train["episodes"], "training episodes")}
    validation = protocol["validation"]["scenario_seed_ranges"]
    reservations = protocol["reserved_final_tests_unopened"]
    if (not training["count"] or set(validation) != set(PROFILES)
            or set(reservations) != set(PROFILES)
            or not _same(prior.merge_ranges(reservations.values()), inherited["reserved_final_seed_ranges"])):
        raise ValueError("Published pilot ranges or unopened final reservations differ")
    if protocol["frozen_reference"]["weights_sha256"] != inherited["source_checkpoint"]["weights_sha256"]:
        raise ValueError("Published pilot did not start from the frozen v3 reference")
    ranges = [training]
    for arm in ARMS:
        config = documents[PREFIX + f"training/{arm}/config.json"]
        summary = documents[PREFIX + f"training/{arm}/summary.json"]
        if (not _same(config["training"], train)
                or type(summary["completed_episodes"]) is not int
                or summary["completed_episodes"] != training["count"]
                or not _same(summary["seed_provenance"]["training"], training)
                or not _same(config["inherited_lineage"], inherited)
                or not _same(summary["seed_provenance"]["inherited"], inherited)):
            raise ValueError("Published pilot lacks complete matching training exposure")
    for profile in PROFILES:
        row = validation[profile]
        if not _same(prior.merge_ranges([row]), [row]) or not row["count"]:
            raise ValueError("Published validation requires a positive exact seed range")
        report = documents[PREFIX + f"evaluation/validation-{profile}.json.gz"]
        expected_seeds = list(range(row["start"], row["start"] + row["count"]))
        if (report["stage"] != "validation" or report["training_performed"] is not False
                or report["protocol"]["independent_final_test_evidence"] is not False
                or not _same(report["seed_provenance"]["evaluation"], row)
                or not report["methods"]
                or any([episode["seed"] for episode in method["episodes"]] != expected_seeds
                       for method in report["methods"].values())):
            raise ValueError("Published pilot validation exposure differs from scored cases")
        ranges.append(row)
    aggregate = documents[PREFIX + "evaluation/aggregate.json"]
    if (aggregate["final_test_accessed"] is not False
            or not _same(aggregate["seed_provenance"]["pilot_validation"], validation)):
        raise ValueError("Published aggregate exposure differs from profile reports")
    for row in ranges:
        prior.assert_disjoint(row["start"], row["count"], inherited, label="published credit pilot")
    return prior.merge_ranges(ranges)


def load_published_lineage(blue_root=None):
    """Verify v1/v2/v3 and the exact v4 pilot, returning compact portable lineage."""
    root = (Path(blue_root) if blue_root is not None else prior.BLUE_ROOT).resolve()
    inherited = prior.load_published_lineage(root)
    path = prior._safe_path(root, MANIFEST_PATH)
    raw = _read(path)
    if prior._sha(raw) != MANIFEST_SHA256:
        raise ValueError("Pinned credit pilot publication manifest changed")
    manifest = prior._strict(raw)
    if (manifest.get("schema") != "triad.credit_pilot_publication.v1"
            or manifest.get("experiment") != "credit-v4-pilot" or manifest.get("stage") != "validation"
            or manifest.get("final_test_accessed") is not False
            or manifest.get("default_policy_changed") is not False
            or len(manifest["files"]) != ARTIFACT_COUNT):
        raise ValueError("Credit pilot publication scope or artifact inventory differs")
    inventory = {MANIFEST_PATH: {"path": MANIFEST_PATH, "sha256": MANIFEST_SHA256, "bytes": len(raw)}}
    snapshots, documents = {path: MANIFEST_SHA256}, {}
    for entry in manifest["files"]:
        name = entry["path"]
        if not name.startswith(PREFIX) or name in inventory:
            raise ValueError("Credit pilot inventory path is duplicate or outside its bundle")
        artifact = prior._safe_path(root, name)
        data = _read(artifact)
        if prior._sha(data) != entry["sha256"] or len(data) != entry["bytes"]:
            raise ValueError(f"Published credit pilot artifact changed: {name}")
        snapshots[artifact] = entry["sha256"]
        inventory[name] = {key: entry[key] for key in ("path", "sha256", "bytes")}
        # The archive hash binds all log/array bytes. Only decode authoritative
        # range-bearing documents; nested provenance also includes reservations.
        if artifact.suffix == ".json" or artifact.name.endswith(".json.gz"):
            documents[name] = prior._decode(artifact, data)[0]
    for field, source_root in (("evaluation_implementation_sha256", root / "Python"),
                               ("archive_implementation_sha256", root.parent.parent)):
        source_map = manifest[field]
        if not source_map:
            raise ValueError("Published credit pilot source inventory is missing")
        for name, digest in source_map.items():
            source = prior._safe_path(source_root, name)
            if prior._sha(_read(source)) != digest:
                raise ValueError(f"Published credit pilot implementation changed: {name}")
            snapshots[source] = digest
    added_ranges = _pilot_ranges(documents, inherited)
    if not _same(prior.load_published_lineage(root), inherited):
        raise RuntimeError("Prior published lineage changed during anchored extraction")
    if any(prior._sha(_read(path)) != digest for path, digest in snapshots.items()):
        raise RuntimeError("Credit pilot evidence changed during anchored extraction")
    result = deepcopy(inherited)
    result.update({
        "schema": SCHEMA,
        "prior_lineage_sha256": prior._sha(prior._canonical(inherited)),
        "prior_pilot": {"publication_manifest": {"path": MANIFEST_PATH, "sha256": MANIFEST_SHA256},
                        "protocol": {"path": PREFIX + "protocol.json", "sha256": inventory[PREFIX + "protocol.json"]["sha256"]}},
        "publication_manifests": [*inherited["publication_manifests"],
                                  {**inventory[MANIFEST_PATH], "artifact_count": ARTIFACT_COUNT}],
        "evidence_inventory_sha256": prior._sha(prior._canonical({
            "prior_inventory_sha256": inherited["evidence_inventory_sha256"],
            "additional_artifacts": [inventory[name] for name in sorted(inventory)]})),
        "evidence_artifact_count": inherited["evidence_artifact_count"] + len(inventory),
        "consumed_seed_ranges": prior.merge_ranges([*inherited["consumed_seed_ranges"], *added_ranges]),
        "scope": "Published v1/v2/v3 and completed credit-v4 pilot scenario exposure; excludes policy RNG identifiers and unlogged workstation/test history.",
        "reference_contract": "Four pinned manifests retain exact artifact/source hashes. The inventory digest chains the prior inventory digest to this pilot manifest and all its artifacts. No pilot checkpoint was promoted; source_checkpoint remains frozen v3.",
    })
    return result


def load_anchored_lineage(blue_root=None):
    """Explicitly named alias for callers alongside the frozen prior loader."""
    return load_published_lineage(blue_root)


def assert_disjoint(start, count, lineage, *, label="proposed scenarios"):
    """Reject consumed or reserved ranges, using the unchanged interval checker."""
    if lineage.get("schema") != SCHEMA:
        raise ValueError("Unsupported anchored published lineage schema")
    prior.assert_disjoint(start, count, {**lineage, "schema": prior.SCHEMA}, label=label)
