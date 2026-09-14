"""Byte-exact publication of completed temporal development evidence.

Read-only endpoint/evaluation verification precedes all writes; no historical
optimizer replay, inference, scenario sampling, selection or promotion. Files
are exclusively created and the manifest is published last. An interrupted
write leaves a visibly incomplete archive requiring manual recovery; it is never
silently reused or overwritten. Existing identical complete archives are audited.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
BLUE_ROOT = REPO_ROOT / "RL/BlueTeam"
sys.path.insert(0, str(BLUE_ROOT / "Python"))
import evaluate_temporal as evaluator

RESULTS = "Results/temporal-v6-pilot"
SCHEMA = "triad.temporal_pilot_publication.v1"
PREDECLARED_SOURCE_COMMIT = "e2d4478c9eb5352640b317c4507a85667bce53c0"


def source_provenance():
    return {"Tools/archive_temporal_pilot.py": evaluator._sha(Path(__file__).read_bytes())}


def _no_links(path):
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Archive paths must not traverse symlinks")


def _tree(path):
    if not path.exists(): return {}
    if not path.is_dir(): raise FileExistsError("Archive destination is not a directory")
    result = {}
    for item in path.rglob("*"):
        if item.is_symlink(): raise ValueError("Archive destination contains a symlink")
        if item.is_file(): result[item.relative_to(path).as_posix()] = evaluator._file(item)
        elif not item.is_dir(): raise ValueError("Archive contains a nonregular object")
    return result


def _new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    _no_links(path.parent)
    with path.open("xb") as handle: handle.write(data)


def archive(*, runs, evaluation, protocol, output=BLUE_ROOT):
    """Validate completed inputs, then publish exact bytes under Results/.

    ``output`` is a BlueTeam-shaped root, not the result directory itself.
    A pre-existing result directory may contain only the byte-identical protocol
    or an entirely identical completed publication. No files are ever removed.
    """
    protocol, evaluation, output = (Path(path).absolute() for path in (protocol, evaluation, output))
    runs = {seed: Path(path).absolute() for seed, path in runs.items()}
    destination = output / RESULTS
    for path in (protocol, evaluation, destination, *runs.values()): _no_links(path)
    protocol_bytes = evaluator._file(protocol)
    document = evaluator._strict_json(protocol_bytes)
    seeds = document["training"]["policy_seeds"]
    if (len(seeds) != 3 or len(set(seeds)) != 3 or any(type(seed) is not int for seed in seeds)
            or any(type(seed) is not int for seed in runs) or set(runs) != set(seeds)):
        raise ValueError("Exactly every declared temporal endpoint is required")
    paths = {"protocol.json": protocol, **{f"evaluation/{name}": evaluation / name for name in evaluator.EVALUATION_FILES},
             **{f"training/seed-{seed}/{name}": runs[seed] / name for seed in sorted(seeds) for name in evaluator.trainer.RUN_FILES}}
    if len(paths) != 36: raise ValueError("Temporal publication must contain exactly 36 inputs")
    blobs = {name: evaluator._file(path) for name, path in paths.items()}
    if blobs["protocol.json"] != protocol_bytes: raise ValueError("Protocol changed during archive preflight")
    sources = source_provenance()
    result = evaluator.verify_completed_evaluation(runs, protocol_path=protocol, output=evaluation)
    if (result.get("schema") != evaluator.AGGREGATE_SCHEMA or result.get("stage") != "validation"
            or result.get("final_test_accessed") is not False
            or result.get("scale_gate", {}).get("not_policy_promotion") is not True):
        raise ValueError("Temporal evaluation scope does not permit this publication")
    manifest = {"schema": SCHEMA, "experiment": "temporal-v6-pilot", "stage": "validation",
                "predeclared_source_commit": PREDECLARED_SOURCE_COMMIT,
                "status": "Development experiment; all fixed endpoints retained, no selection or promoted replacement",
                "default_policy_changed": False, "final_test_accessed": False, "manifest_self_hash_excluded": True,
                "policy_seeds": sorted(seeds), "protocol_sha256": evaluator._sha(blobs["protocol.json"]),
                "archive_implementation_sha256": sources,
                "training_implementation_sha256": document["training_implementation_sha256"],
                "evaluation_implementation_sha256": document["evaluation_implementation_sha256"],
                "seed_provenance": {"training": document["training"]["scenario_seed_ranges"],
                                    "validation": document["validation"]["scenario_seed_ranges"],
                                    "reserved_final_tests_unopened": document["reserved_final_tests_unopened"]},
                "scale_gate": result["scale_gate"],
                "copy_contract": "Exact input bytes including compressed JSONL/reports; no repacking; manifest published last",
                "verification_scope": "Complete current endpoint/evaluation metadata, hashes, paired statistics; no optimizer replay or sampling",
                "runtime_reporting": "Per-run config.json records runtime; runtime details are not a claim of compute-matched comparisons",
                "files": [{"path": f"{RESULTS}/{name}", "sha256": evaluator._sha(data), "bytes": len(data)}
                          for name, data in sorted(blobs.items())]}
    expected = {**blobs, "artifact-manifest.json": evaluator._bytes(manifest)}

    def stable():
        if (sources != source_provenance() or any(evaluator._file(path) != blobs[name] for name, path in paths.items())
                or not evaluator._same(evaluator.source_provenance(), document["evaluation_implementation_sha256"])):
            raise ValueError("Publication source or verified input changed")

    stable()
    existing = _tree(destination)
    if existing == expected:
        stable()
        return manifest
    if existing not in ({}, {"protocol.json": blobs["protocol.json"]}):
        raise FileExistsError("Refusing mixed, incomplete or conflicting existing publication")
    # Reject unexpected empty directories too; they may belong to another task.
    if destination.exists() and any(path.is_dir() for path in destination.iterdir()):
        raise FileExistsError("Protocol-only destination contains unrelated directories")
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(blobs.items()):
        if name not in existing: _new(destination / name, data)
    stable()
    if _tree(destination) != blobs: raise ValueError("Archive bytes changed before completion; no manifest written")
    _new(destination / "artifact-manifest.json", expected["artifact-manifest.json"])
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=RUN_DIRECTORY")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BLUE_ROOT)
    args, runs = vars(parser.parse_args(argv)), {}
    for item in args.pop("run"):
        seed, separator, path = item.partition("=")
        if not separator or not seed.isdecimal() or int(seed) in runs or not path:
            parser.error("Each run must be a unique integer SEED=RUN_DIRECTORY")
        runs[int(seed)] = Path(path)
    print(evaluator._bytes(archive(runs=runs, **args)).decode())


if __name__ == "__main__":
    main()
