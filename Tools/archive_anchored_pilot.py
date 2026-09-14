"""Archive exact terminal anchored-pilot evidence without sampling or promotion.

Uses only read-only checkpoint/report validators and paired arithmetic. It never
invokes training, inference or the evaluator's potentially sampling resume path.
Every input, privacy constraint and destination is checked before the first copy;
identical existing artifacts are idempotent and differing files are not replaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from publish_robust_rl import _canonical_bytes, _gzip_json, _json_object, privacy_check, sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
BLUE_ROOT = REPO_ROOT / "RL/BlueTeam"
sys.path.insert(0, str(BLUE_ROOT / "Python"))
from triad_rl import evaluate_anchored_pilot as evaluator

RESULTS = "Results/anchored-v4-pilot"
ARCHIVE_SOURCES = ("Tools/archive_anchored_pilot.py", "Tools/publish_robust_rl.py")


def _verify(paths, protocol_path, documents, artifacts, sources):
    """Reconstruct terminal inputs and statistics, with no scenario-producing call."""
    protocol, raw = evaluator.load_protocol(protocol_path)
    if raw != artifacts[f"{RESULTS}/protocol.json"] or not evaluator._same(sources, protocol["evaluation_implementation_sha256"]):
        raise ValueError("Archive protocol or evaluation source changed during preflight")
    lineage = evaluator.load_published_lineage()
    reservations = protocol.get("reserved_final_tests_unopened", {})
    declared_prior = protocol.get("prior_pilot_publication", {})
    if (set(reservations) != set(evaluator.PROFILES)
            or not evaluator._same(sorted(reservations.values(), key=lambda row: row["start"]), lineage["reserved_final_seed_ranges"])
            or not evaluator._same({"path": declared_prior.get("manifest"), "sha256": declared_prior.get("sha256")},
                                   lineage["prior_pilot"]["publication_manifest"])):
        raise ValueError("Archive prior pilot or unopened final reservations differ")
    train = protocol["training"]
    exposure = {"inherited": lineage, "pilot_training": {"start": train["scenario_seed_start"], "count": train["episodes"]}}
    for row in protocol["validation"]["scenario_seed_ranges"].values():
        evaluator.assert_disjoint(row["start"], row["count"], lineage, label="archived anchored validation")
        evaluator.validate_seed_range(row["start"], row["count"], "validation", exposure)
    endpoints = {arm: evaluator.trainer.validate_run(path, protocol_path, arm, lineage) for arm, path in paths.items()}
    if len({value[1]["first_batch_trajectory_sha256"] for value in endpoints.values()}) != 1:
        raise ValueError("Archived arms do not share the same first sampled batch")
    identities = [evaluator._bytes(value[1]["first_batch_coefficient_identity"]) for value in endpoints.values()]
    if len(set(identities)) != 1:
        raise ValueError("Archived arms do not share identical first-batch coefficient evidence")
    reference_files = {}
    weights = {arm: policy.weights_fingerprint() for arm, (policy, _) in endpoints.items()}
    for name, (path, expected, kind) in evaluator.REFERENCES.items():
        target = evaluator.BLUE_ROOT / path
        reference_files[name] = evaluator._checkpoint_files(target)
        if kind.load(target, feature_names=evaluator.FEATURE_NAMES).weights_fingerprint() != expected:
            raise ValueError("Archived reference fingerprint differs")
        weights[name] = expected
    inputs = {"schema": "triad.anchored_pilot_inputs.v1", "protocol_sha256": sha256(raw), "implementation_sha256": sources,
              "endpoint_evidence": {arm: evidence for arm, (_, evidence) in endpoints.items()},
              "reference_files_sha256": reference_files, "weights_sha256": weights,
              "lineage": exposure, "validation_ranges": protocol["validation"]["scenario_seed_ranges"]}
    if artifacts[f"{RESULTS}/evaluation/evaluation-inputs.json"] != evaluator._bytes(inputs):
        raise ValueError("Archived evaluation inputs differ from complete validated endpoints")
    reports, receipts = {}, {}
    for profile in evaluator.PROFILES:
        name = f"{RESULTS}/evaluation/validation-{profile}"
        data, report = artifacts[name + ".json.gz"], documents[name + ".json.gz"]
        receipt = {"profile": profile, "sha256": sha256(data), "bytes": len(data), "inputs_sha256": evaluator.canonical_hash(inputs)}
        if not evaluator._same(documents[name + ".receipt.json"], receipt):
            raise ValueError("Archived profile compressed-byte receipt differs")
        evaluator.validate_report(report, profile, inputs, weights)
        reports[profile], receipts[profile] = report, receipt
    verified = {**evaluator.aggregate_reports(reports, protocol), "protocol_sha256": sha256(raw),
                "inputs_sha256": evaluator.canonical_hash(inputs), "implementation_sha256": sources, "reports": receipts,
                "seed_provenance": {**exposure, "pilot_validation": inputs["validation_ranges"]}}
    if artifacts[f"{RESULTS}/evaluation/aggregate.json"] != evaluator._bytes(verified):
        raise ValueError("Archived aggregate, paired statistics or scaling gate differs")
    if not evaluator._same(evaluator.load_published_lineage(), lineage):
        raise ValueError("Inherited publication changed during archive verification")
    if any(evaluator._checkpoint_files(evaluator.BLUE_ROOT / path) != reference_files[name]
           for name, (path, _, _) in evaluator.REFERENCES.items()):
        raise ValueError("Frozen reference changed during archive verification")
    return verified


def archive(*, runs, evaluation, protocol, output=BLUE_ROOT):
    """Verify complete terminal evidence, then copy exact bytes without promotion."""
    if set(runs) != set(evaluator.ARMS):
        raise ValueError("Exactly the three anchored pilot arms are required")
    paths = {arm: Path(path).resolve() for arm, path in runs.items()}
    evaluation, protocol, output = Path(evaluation).resolve(), Path(protocol).resolve(), Path(output).resolve()
    if (not evaluation.is_dir()
            or {path.name for path in evaluation.iterdir() if path.is_file()} != set(evaluator.EVALUATION_FILES)
            or any(not path.is_file() for path in evaluation.iterdir())):
        raise ValueError("Archive requires the exact complete terminal evaluation file set")
    sources = evaluator.source_provenance()
    archive_sources = {name: sha256((REPO_ROOT / name).read_bytes()) for name in ARCHIVE_SOURCES}
    snapshots, artifacts, documents = {}, {}, {}

    def add(name, path):
        data = evaluator._file(path)
        snapshots[path] = data
        if name.endswith(".json.gz"):
            documents[name] = _gzip_json(data)
        elif name.endswith(".jsonl"):
            for line in data.splitlines():
                _json_object(line)
        elif name.endswith(".json"):
            documents[name] = _json_object(data)
        elif not name.endswith(".npz"):
            raise ValueError("Unexpected anchored artifact extension")
        artifacts[name] = data

    # All artifacts must exist and pass privacy checks before any deeper work.
    for name in evaluator.EVALUATION_FILES:
        add(f"{RESULTS}/evaluation/{name}", evaluation / name)
    for arm, path in paths.items():
        for name in evaluator.trainer.RUN_FILES:
            add(f"{RESULTS}/training/{arm}/{name}", path / name)
    add(f"{RESULTS}/protocol.json", protocol)
    verified = _verify(paths, protocol, documents, artifacts, sources)
    manifest = {"schema": "triad.anchored_pilot_publication.v1", "experiment": "anchored-v4-pilot",
                "stage": "validation", "final_test_accessed": False, "default_policy_changed": False,
                "status": "Development pilot; no selected or promoted replacement policy",
                "scale_gate": verified["scale_gate"], "protocol_sha256": sha256(artifacts[f"{RESULTS}/protocol.json"]),
                "evaluation_implementation_sha256": sources, "archive_implementation_sha256": archive_sources,
                "copy_contract": "Exact input bytes; no checkpoint repacking, redaction, scenario sampling or best-arm selection",
                "manifest_self_hash_excluded": True,
                "files": [{"path": name, "bytes": len(data), "sha256": sha256(data)} for name, data in sorted(artifacts.items())]}
    privacy_check(manifest)
    artifacts[f"{RESULTS}/artifact-manifest.json"] = _canonical_bytes(manifest)
    targets = {}
    for name, data in artifacts.items():
        target = (output / name).resolve()
        if not target.is_relative_to(output) or target in targets.values():
            raise ValueError("Archive target escapes its root or aliases another target")
        for ancestor in (target.parent, *target.parent.parents):
            if ancestor.exists() and not ancestor.is_dir():
                raise ValueError("Archive parent is not a directory")
            if ancestor == output:
                break
        if target.exists() and (not target.is_file() or target.read_bytes() != data):
            raise FileExistsError(f"Differing artifact will not be replaced: {name}")
        targets[name] = target
    if (any(path.read_bytes() != data for path, data in snapshots.items())
            or evaluator.source_provenance() != sources
            or any(sha256((REPO_ROOT / name).read_bytes()) != digest for name, digest in archive_sources.items())):
        raise ValueError("Archive source or input changed during preflight")
    for name, data in artifacts.items():
        target = targets[name]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=RUN_DIRECTORY")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BLUE_ROOT)
    args = vars(parser.parse_args())
    runs = {}
    for item in args.pop("run"):
        arm, separator, path = item.partition("=")
        if not separator or arm not in evaluator.ARMS or arm in runs or not path:
            parser.error("Each run must be a unique known ARM=RUN_DIRECTORY")
        runs[arm] = Path(path)
    result = archive(runs=runs, **args)
    print(f"Archived {len(result['files'])} exact anchored pilot artifacts; no policy promoted.")


if __name__ == "__main__":
    main()
