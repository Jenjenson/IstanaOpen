"""Archive a fully completed credit pilot byte-for-byte, without promotion.

Requires every profile and aggregate to exist before invoking the evaluator's
read-only completed-resume verification. This command never starts missing
evaluation work, training, network uploads, or overwrites differing artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from publish_robust_rl import _canonical_bytes, _gzip_json, _json_object, privacy_check, sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
BLUE_ROOT = REPO_ROOT / "RL/BlueTeam"
sys.path.insert(0, str(BLUE_ROOT / "Python"))
from triad_rl import evaluate_credit_pilot as evaluator

RESULTS = "Results/credit-v4-pilot"
EVALUATION_FILES = ("evaluation-inputs.json", "aggregate.json", *(
    f"validation-{profile}{suffix}" for profile in evaluator.PROFILES
    for suffix in (".json.gz", ".receipt.json")))


def archive(*, runs, evaluation, protocol, output=BLUE_ROOT):
    """Verify complete evidence and copy only after every target passes checks."""
    if set(runs) != set(evaluator.ARMS):
        raise ValueError("Exactly the three pilot arms are required")
    evaluation, protocol, output = Path(evaluation).resolve(), Path(protocol).resolve(), Path(output).resolve()
    paths = {arm: Path(path).resolve() for arm, path in runs.items()}
    sources, snapshots, artifacts = evaluator.source_provenance(), {}, {}
    archive_sources = {name: sha256((REPO_ROOT / name).read_bytes()) for name in (
        "Tools/archive_credit_pilot.py", "Tools/publish_robust_rl.py")}

    def read(path):
        data = evaluator._file(path)
        snapshots[path] = data
        return data

    def add(name, data):
        if name.endswith(".json.gz"):
            _gzip_json(data)
        elif name.endswith(".jsonl"):
            for line in data.splitlines():
                _json_object(line)
        elif name.endswith(".json"):
            _json_object(data)
        elif not name.endswith(".npz"):
            raise ValueError("Unexpected pilot artifact extension")
        artifacts[name] = data

    # Reading ALL completed files first prevents resume from sampling a missing
    # profile. The evaluator then recomputes summaries/gate and checks bindings.
    for name in EVALUATION_FILES:
        add(f"{RESULTS}/evaluation/{name}", read(evaluation / name))
    for arm, path in paths.items():
        for name in evaluator.RUN_FILES:
            add(f"{RESULTS}/training/{arm}/{name}", read(path / name))
    add(f"{RESULTS}/protocol.json", read(protocol))
    verified = evaluator.run_evaluation(paths, protocol_path=protocol, output=evaluation, resume=True)
    if (verified["final_test_accessed"] is not False
            or any(path.read_bytes() != data for path, data in snapshots.items())
            or evaluator.source_provenance() != sources):
        raise ValueError("Completed pilot evidence/source changed during verification")
    manifest = {
        "schema": "triad.credit_pilot_publication.v1", "experiment": "credit-v4-pilot",
        "stage": "validation", "final_test_accessed": False, "default_policy_changed": False,
        "status": "Diagnostic pilot; no selected or promoted replacement policy",
        "scale_gate": verified["scale_gate"],
        "protocol_sha256": sha256(artifacts[f"{RESULTS}/protocol.json"]),
        "evaluation_implementation_sha256": sources, "archive_implementation_sha256": archive_sources,
        "copy_contract": "Exact input bytes; no checkpoint repacking, redaction, or best-arm selection",
        "manifest_self_hash_excluded": True,
        "files": [{"path": name, "bytes": len(data), "sha256": sha256(data)}
                  for name, data in sorted(artifacts.items())],
    }
    privacy_check(manifest)
    add(f"{RESULTS}/artifact-manifest.json", _canonical_bytes(manifest))
    targets = {}
    for name, data in artifacts.items():
        target = (output / name).resolve()
        if not target.is_relative_to(output) or target in targets.values():
            raise ValueError("Archive output escapes its root or aliases another target")
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
        raise ValueError("Archive source/input changed during preflight")
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
    print(f"Archived {len(result['files'])} exact pilot artifacts; no policy promoted.")


if __name__ == "__main__":
    main()
