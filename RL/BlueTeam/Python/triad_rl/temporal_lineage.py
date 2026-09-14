"""One-pass, hash-pinned published exposure; no replay, inference or sampling.

The ranking publication pins its already-audited inherited range declaration.
All six manifests, their exact artifact bytes and source maps are checked once
per unique path. Final file-stat guards detect ordinary concurrent changes;
this is a preflight snapshot, not a promise that files cannot change afterward.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from .credit_lineage import _canonical, _integer, _safe_path, _sha, _strict, merge_ranges

SCHEMA = "triad.temporal_published_lineage.v1"
BLUE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = "Results/ranking-v5-pilot/artifact-manifest.json"
MANIFEST_SHA256 = "a9d975e38dbdf8a5095c52e0767e2142ac5616dcd09d338c85247747517dd058"
PREFIX = "Results/ranking-v5-pilot/"
PROFILES = ("normal", "stress", "capability")


def _stamp(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class _Files:
    def __init__(self):
        self.seen = {}

    def check(self, path, digest, size=None, *, keep=False):
        if path not in self.seen:
            before, hasher, chunks = _stamp(path), hashlib.sha256(), []
            if before[2] > 512 * 1024 * 1024:
                raise ValueError("Lineage artifact exceeds bounded file size")
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    hasher.update(chunk)
                    if keep: chunks.append(chunk)
            if _stamp(path) != before:
                raise ValueError("Lineage file changed while hashing")
            self.seen[path] = (hasher.hexdigest(), before, b"".join(chunks) if keep else None)
        actual, stamp, data = self.seen[path]
        if actual != digest or size is not None and stamp[2] != size:
            raise ValueError(f"Pinned lineage bytes differ: {path.name}")
        if keep and data is None:
            raise ValueError("Range document was not retained on its first read")
        return data

    def stable(self):
        if any(_stamp(path) != row[1] for path, row in self.seen.items()):
            raise ValueError("Lineage file changed during preflight")


def assert_disjoint(start, count, lineage, *, label="proposed scenarios"):
    start, count = _integer(start, "start"), _integer(count, "count")
    if not count or lineage.get("schema") != SCHEMA:
        raise ValueError("Need positive count and temporal lineage schema")
    for kind in ("consumed_seed_ranges", "reserved_final_seed_ranges"):
        for row in merge_ranges(lineage[kind]):
            if max(start, row["start"]) < min(start + count, row["start"] + row["count"]):
                raise ValueError(f"{label} overlaps {kind}: {row['start']} + {row['count']}")


def load_published_lineage(blue_root=None):
    """Verify flat bytes once; reuse pinned exposure without recursive loaders."""
    root = (Path(blue_root) if blue_root is not None else BLUE_ROOT).resolve()
    files, inventory, sources, documents = _Files(), {}, {}, {}
    raw = files.check(_safe_path(root, MANIFEST_PATH), MANIFEST_SHA256, keep=True)
    current = _strict(raw)
    if (current.get("schema") != "triad.ranking_pilot_publication.v1" or len(current["files"]) != 36
            or current.get("final_test_accessed") is not False or current.get("default_policy_changed") is not False):
        raise ValueError("Pinned ranking publication scope differs")
    config_name = PREFIX + "training/seed-403/config.json"
    entry = next(row for row in current["files"] if row["path"] == config_name)
    inherited = _strict(files.check(_safe_path(root, config_name), entry["sha256"], entry["bytes"], keep=True))["inherited_lineage"]
    if inherited.get("schema") != "triad.ranking_published_lineage.v1" or len(inherited["publication_manifests"]) != 5:
        raise ValueError("Pinned inherited range declaration differs")
    manifests = [*inherited["publication_manifests"], {"path": MANIFEST_PATH, "sha256": MANIFEST_SHA256,
                                                      "bytes": len(raw), "artifact_count": 36}]
    for record in manifests:
        name = record["path"]
        manifest_raw = files.check(_safe_path(root, name), record["sha256"], record["bytes"], keep=True)
        manifest = _strict(manifest_raw)
        if len(manifest["files"]) != record["artifact_count"]:
            raise ValueError("Manifest artifact count differs")
        inventory[name] = {key: record[key] for key in ("path", "sha256", "bytes")}
        for row in manifest["files"]:
            artifact = row["path"]
            if artifact in inventory and inventory[artifact] != row:
                raise ValueError("Conflicting artifact declarations")
            inventory[artifact] = {key: row[key] for key in ("path", "sha256", "bytes")}
            keep = artifact.startswith(PREFIX) and artifact.endswith(".json")
            data = files.check(_safe_path(root, artifact), row["sha256"], row["bytes"], keep=keep)
            if keep: documents[artifact] = _strict(data)
        maps = [manifest.get("source_files_sha256", {}), manifest.get("archive_implementation_sha256", {})]
        if name == "Results/adaptive-v1/artifact-manifest.json":
            maps.append({"RL/BlueTeam/Python/triad_rl/" + key: value for key, value in manifest["training_implementation_sha256"].items()})
        else:
            maps.extend({"RL/BlueTeam/Python/" + key: value for key, value in manifest.get(field, {}).items()}
                        for field in ("training_implementation_sha256", "evaluation_implementation_sha256"))
        for mapping in maps:
            for source, digest in mapping.items():
                if source in sources and sources[source] != digest:
                    raise ValueError("Conflicting source declarations")
                sources[source] = digest
                files.check(_safe_path(root.parent.parent, source), digest)
    if len(inventory) != inherited["evidence_artifact_count"] + 37:
        raise ValueError("Flat inherited/current artifact inventory differs")
    protocol = documents[PREFIX + "protocol.json"]
    train, validation = protocol["training"], protocol["validation"]["scenario_seed_ranges"]
    reserved = merge_ranges(protocol["reserved_final_tests_unopened"].values())
    if (reserved != inherited["reserved_final_seed_ranges"] or len(reserved) != 3
            or set(validation) != set(PROFILES) or current["protocol_sha256"] != inventory[PREFIX + "protocol.json"]["sha256"]):
        raise ValueError("Ranking protocol exposure/reservations differ")
    ranges, endpoints = [], {}
    for seed in train["policy_seeds"]:
        path = PREFIX + f"training/seed-{seed}/"
        config, summary = documents[path + "config.json"], documents[path + "summary.json"]
        row = {"start": seed * 1_000_000, "count": train["episodes"]}
        if (_canonical(config["inherited_lineage"]) != _canonical(inherited) or config["training"] != train
                or summary["completed_episodes"] != row["count"] or summary["seed_provenance"]["training"] != row
                or train["scenario_seed_ranges"][str(seed)] != row):
            raise ValueError("Ranking complete training exposure differs")
        endpoints[str(seed)] = {"path": path + "last", "weights_sha256": summary["last_weights_sha256"],
                                "checkpoint_files_sha256": {key: inventory[path + "last/" + key]["sha256"] for key in ("checkpoint.json", "arrays.npz")}}
        ranges.append(row)
    for profile in PROFILES:
        row = validation[profile]
        if row != {"start": 10**15 + protocol["validation"]["run_seeds"][profile] * 1_000_000,
                   "count": protocol["validation"]["episodes_per_profile"]}:
            raise ValueError("Ranking validation exposure differs")
        ranges.append(row)
    result = {"schema": SCHEMA, "consumed_seed_ranges": merge_ranges(inherited["consumed_seed_ranges"]),
              "reserved_final_seed_ranges": reserved}
    for row in ranges: assert_disjoint(row["start"], row["count"], result, label="published ranking")
    result.update(consumed_seed_ranges=merge_ranges([*result["consumed_seed_ranges"], *ranges]),
                  publication_manifests=deepcopy(manifests), source_checkpoint=deepcopy(inherited["source_checkpoint"]),
                  source_checkpoint_role="frozen_v3_reference_only_not_temporal_initialization", ranking_endpoints=endpoints,
                  prior_pilot={"publication_manifest": {"path": MANIFEST_PATH, "sha256": MANIFEST_SHA256},
                               "protocol": {"path": PREFIX + "protocol.json", "sha256": current["protocol_sha256"]}},
                  inherited_inventory_sha256=inherited["evidence_inventory_sha256"],
                  inherited_exposure_declaration={"path": config_name, "sha256": inventory[config_name]["sha256"], "field": "inherited_lineage"},
                  evidence_inventory_sha256=_sha(_canonical([inventory[key] for key in sorted(inventory)])),
                  evidence_artifact_count=len(inventory), source_files_sha256=dict(sorted(sources.items())),
                  verification_scope="One content hash per unique artifact/source path, final stat guards; no optimization replay, inference, scenario sampling or recursive lineage loading.",
                  scope="Published v1/v2/v3 and credit/anchored/ranking pilots; repeated old-case diagnostics add no new ranges. Unlogged workstation/test history is outside this declaration.")
    files.stable()
    return result
