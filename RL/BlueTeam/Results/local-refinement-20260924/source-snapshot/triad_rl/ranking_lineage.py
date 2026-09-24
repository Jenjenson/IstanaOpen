"""Hash-pinned development exposure for a prospective public-option ranker.

No training, policy or environment imports, inference or scenario sampling.
The preserved v3 source_checkpoint is a comparison reference only: it is not
the initialization of a ranker with a greedy prior and fresh residual weights.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import anchored_lineage as prior


SCHEMA = "triad.ranking_published_lineage.v1"
MANIFEST_PATH = "Results/anchored-v4-pilot/artifact-manifest.json"
MANIFEST_SHA256 = "e0842860986efe40b06e6d6b7a7f7210ccb5d6802037defc3ee414c7f075bab5"
PREFIX = "Results/anchored-v4-pilot/"
ARTIFACT_COUNT = 33
PROFILES = prior.PROFILES
ARMS = ("shared_critic", "no_critic_gradient", "anchored_later_stop")
METHODS = (*ARMS, "balanced_v3", "robust_v2", "adaptive_v1", "all_step_stop_pilot", "greedy_public")
REFERENCE_ROLE = "frozen_v3_comparison_reference_only_not_ranker_initialization"
utils = prior.prior


def _pilot_ranges(documents, inherited):
    """Derive complete training and scored validation exposure, never reservations."""
    protocol = documents[PREFIX + "protocol.json"]
    train = protocol["training"]
    training = {"start": utils._integer(train["scenario_seed_start"], "training start"),
                "count": utils._integer(train["episodes"], "training episodes")}
    validation = protocol["validation"]["scenario_seed_ranges"]
    reservations = protocol["reserved_final_tests_unopened"]
    if (not training["count"] or set(validation) != set(PROFILES) or set(reservations) != set(PROFILES)
            or not prior._same(utils.merge_ranges(reservations.values()), inherited["reserved_final_seed_ranges"])
            or protocol["frozen_reference"]["weights_sha256"] != inherited["source_checkpoint"]["weights_sha256"]):
        raise ValueError("Published anchored reference or seed reservations differ")
    for arm in ARMS:
        config = documents[PREFIX + f"training/{arm}/config.json"]
        summary = documents[PREFIX + f"training/{arm}/summary.json"]
        if (not prior._same(config["training"], train)
                or type(summary["completed_episodes"]) is not int
                or summary["completed_episodes"] != training["count"]
                or not prior._same(summary["seed_provenance"]["training"], training)
                or not prior._same(config["inherited_lineage"], inherited)
                or not prior._same(summary["seed_provenance"]["inherited"], inherited)):
            raise ValueError("Published anchored run lacks complete matching exposure")
    ranges = [training]
    for profile in PROFILES:
        row = validation[profile]
        if not prior._same(utils.merge_ranges([row]), [row]) or not row["count"]:
            raise ValueError("Published anchored validation range must be positive and exact")
        report = documents[PREFIX + f"evaluation/validation-{profile}.json.gz"]
        seeds = list(range(row["start"], row["start"] + row["count"]))
        if (report["stage"] != "validation" or report["profile"] != profile
                or report["training_performed"] is not False
                or report["protocol"]["independent_final_test_evidence"] is not False
                or not prior._same(report["seed_provenance"]["evaluation"], row)
                or set(report["methods"]) != set(METHODS)
                or any([episode["seed"] for episode in method["episodes"]] != seeds
                       for method in report["methods"].values())):
            raise ValueError("Published anchored validation differs from scored paired cases")
        ranges.append(row)
    aggregate = documents[PREFIX + "evaluation/aggregate.json"]
    if (aggregate["final_test_accessed"] is not False
            or not prior._same(aggregate["seed_provenance"]["pilot_validation"], validation)):
        raise ValueError("Published anchored aggregate exposure differs")
    for row in ranges:
        prior.assert_disjoint(row["start"], row["count"], inherited, label="published anchored pilot")
    return utils.merge_ranges(ranges)


def load_published_lineage(blue_root=None):
    """Verify five immutable bundles and return compact, portable reference lineage."""
    root = (Path(blue_root) if blue_root is not None else utils.BLUE_ROOT).resolve()
    inherited = prior.load_published_lineage(root)
    path = utils._safe_path(root, MANIFEST_PATH)
    raw = prior._read(path)
    if utils._sha(raw) != MANIFEST_SHA256:
        raise ValueError("Pinned anchored publication manifest changed")
    manifest = utils._strict(raw)
    if (manifest.get("schema") != "triad.anchored_pilot_publication.v1"
            or manifest.get("experiment") != "anchored-v4-pilot" or manifest.get("stage") != "validation"
            or manifest.get("final_test_accessed") is not False
            or manifest.get("default_policy_changed") is not False
            or len(manifest["files"]) != ARTIFACT_COUNT):
        raise ValueError("Published anchored inventory or scope differs")
    inventory = {MANIFEST_PATH: {"path": MANIFEST_PATH, "sha256": MANIFEST_SHA256, "bytes": len(raw)}}
    snapshots, documents = {path: MANIFEST_SHA256}, {}
    for entry in manifest["files"]:
        name = entry["path"]
        if not name.startswith(PREFIX) or name in inventory:
            raise ValueError("Anchored artifact is duplicate or outside its publication")
        artifact = utils._safe_path(root, name)
        data = prior._read(artifact)
        if utils._sha(data) != entry["sha256"] or len(data) != entry["bytes"]:
            raise ValueError(f"Published anchored artifact changed: {name}")
        snapshots[artifact] = entry["sha256"]
        inventory[name] = {key: entry[key] for key in ("path", "sha256", "bytes")}
        # Exact manifest hashes bind all log/array bytes without deserialization.
        if artifact.suffix == ".json" or artifact.name.endswith(".json.gz"):
            documents[name] = utils._decode(artifact, data)[0]
    for field, source_root in (("evaluation_implementation_sha256", root / "Python"),
                               ("archive_implementation_sha256", root.parent.parent)):
        source_map = manifest[field]
        if not source_map:
            raise ValueError("Published anchored source inventory is missing")
        for name, digest in source_map.items():
            source = utils._safe_path(source_root, name)
            if utils._sha(prior._read(source)) != digest:
                raise ValueError(f"Published anchored implementation changed: {name}")
            snapshots[source] = digest
    added_ranges = _pilot_ranges(documents, inherited)
    if not prior._same(prior.load_published_lineage(root), inherited):
        raise RuntimeError("Inherited lineage changed during ranking extraction")
    if any(utils._sha(prior._read(path)) != digest for path, digest in snapshots.items()):
        raise RuntimeError("Anchored evidence changed during ranking extraction")
    result = deepcopy(inherited)
    result.update({
        "schema": SCHEMA, "source_checkpoint_role": REFERENCE_ROLE,
        "prior_lineage_sha256": utils._sha(utils._canonical(inherited)),
        "prior_pilot": {"publication_manifest": {"path": MANIFEST_PATH, "sha256": MANIFEST_SHA256},
                        "protocol": {"path": PREFIX + "protocol.json", "sha256": inventory[PREFIX + "protocol.json"]["sha256"]}},
        "publication_manifests": [*inherited["publication_manifests"],
                                  {**inventory[MANIFEST_PATH], "artifact_count": ARTIFACT_COUNT}],
        "evidence_inventory_sha256": utils._sha(utils._canonical({
            "prior_inventory_sha256": inherited["evidence_inventory_sha256"],
            "additional_artifacts": [inventory[name] for name in sorted(inventory)]})),
        "evidence_artifact_count": inherited["evidence_artifact_count"] + len(inventory),
        "consumed_seed_ranges": utils.merge_ranges([*inherited["consumed_seed_ranges"], *added_ranges]),
        "scope": "Published v1/v2/v3, credit-v4 and anchored-v4 development exposure; excludes policy RNG identifiers and unlogged workstation/test history.",
        "reference_contract": "Five pinned manifests retain exact artifact/source hashes; the inventory digest chains prior evidence to the anchored publication. source_checkpoint identifies the frozen v3 comparator only, not ranker initialization or transferred weights.",
    })
    return result


def assert_disjoint(start, count, lineage, *, label="proposed scenarios"):
    """Reject inherited consumed or unopened final intervals without modifying them."""
    if lineage.get("schema") != SCHEMA:
        raise ValueError("Unsupported ranking published lineage schema")
    prior.assert_disjoint(start, count, {**lineage, "schema": prior.SCHEMA}, label=label)
