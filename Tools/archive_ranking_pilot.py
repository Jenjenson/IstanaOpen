"""Publish completed ranking evidence without new scenarios or policy promotion.

Read-only evaluator validation precedes every write. Existing compressed reports
and JSONL labels are copied, never repacked. A complete staging tree is prepared
first. A new archive is atomically installed; an existing identical protocol-only
directory receives complete subtrees and its manifest last, with rollback of
unchanged newly installed subtrees if installation fails. A manifest is the
completion marker. Existing mixed, partial or conflicting evidence is refused.
Validation may reconstruct recorded optimizer arithmetic on detached parameters;
it never runs a new training experiment or new behavior/simulator trajectory.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import gzip
import importlib
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile

from publish_robust_rl import _canonical_bytes, _gzip_json, _json_object, privacy_check, sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
BLUE_ROOT = REPO_ROOT / "RL/BlueTeam"
sys.path.insert(0, str(BLUE_ROOT / "Python"))
RESULTS = "Results/ranking-v5-pilot"
SCHEMA = "triad.ranking_pilot_publication.v1"
ARCHIVE_SOURCES = ("Tools/archive_ranking_pilot.py", "Tools/publish_robust_rl.py")
EVALUATION_FILES = ("evaluation-inputs.json", "aggregate.json",
                    *(f"validation-{profile}{suffix}" for profile in ("normal", "stress", "capability")
                      for suffix in (".json.gz", ".receipt.json")))
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024


def _evaluator():
    # Importing the module is read-only; never call its sampling runner.
    return importlib.import_module("evaluate_ranking")


def source_provenance():
    return {name: sha256((REPO_ROOT / name).read_bytes()) for name in ARCHIVE_SOURCES}


def _no_links(path, checked=None):
    checked = set() if checked is None else checked
    for item in (path, *path.parents):
        if item in checked:
            break
        if item.is_symlink():
            raise ValueError("Archive inputs and destinations must not traverse symlinks")
        checked.add(item)


def _read(path, *, checked=None):
    _no_links(path, checked)
    if not path.is_file():
        raise FileNotFoundError(f"Missing regular archive input: {path.name}")
    with path.open("rb") as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Archive input exceeds bounded size")
    return data


def _json_lines(data, *, compressed):
    stream = gzip.GzipFile(fileobj=io.BytesIO(data)) if compressed else io.BytesIO(data)
    total, count = 0, 0
    try:
        with stream:
            while True:
                line = stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                total += len(line)
                if len(line) > MAX_LINE_BYTES or total > MAX_EXPANDED_BYTES:
                    raise ValueError("JSONL evidence exceeds bounded expanded/line size")
                _json_object(line)
                count += 1
    except (OSError, EOFError) as error:
        raise ValueError("Invalid compressed JSONL evidence") from error
    if not count:
        raise ValueError("JSONL evidence must not be empty")


def _tree(path):
    if not path.exists():
        return None
    checked = set()
    _no_links(path, checked)
    if not path.is_dir():
        raise FileExistsError("Archive destination is not a directory")
    result = {}
    for item in path.rglob("*"):
        _no_links(item, checked)
        if item.is_file():
            result[item.relative_to(path).as_posix()] = _read(item, checked=checked)
        elif not item.is_dir():
            raise ValueError("Archive contains a nonregular entry")
        elif not any(item.iterdir()):
            # Empty unexpected subdirectories are not silently absorbed.
            result[item.relative_to(path).as_posix() + "/"] = None
    return result


def _rename_noreplace(source, target):
    """Atomic no-replacement rename on supported publication hosts."""
    if target.exists() or target.is_symlink():
        raise FileExistsError("Archive target appeared during publication")
    if os.name == "nt":
        os.rename(source, target)  # Windows refuses an existing file or directory.
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise OSError("Atomic no-replacement rename unavailable; publication refused")
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
            code = ctypes.get_errno()
            if code in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError("Archive target appeared during publication")
            raise OSError(code, os.strerror(code))
        return
    raise OSError("Atomic no-replacement archive publication supports Windows and Linux only")


def _verify(runs, evaluation, protocol, documents, sources):
    evaluator = _evaluator()
    verified = evaluator.verify_completed_evaluation(runs, protocol_path=protocol, output=evaluation)
    if (_canonical_bytes(verified) != _canonical_bytes(documents["evaluation/aggregate.json"])
            or verified.get("final_test_accessed") is not False
            or not isinstance(verified.get("scale_gate"), dict)
            or type(verified["scale_gate"].get("passed")) is not bool):
        raise ValueError("Completed ranking aggregate or development-only scope differs")
    if _canonical_bytes(documents["protocol.json"].get("evaluation_implementation_sha256")) != _canonical_bytes(sources):
        raise ValueError("Predeclared ranking evaluation sources differ")
    return verified


def _install(target, artifacts, check_stable):
    existing = _tree(target)
    protocol_only = {"protocol.json": artifacts["protocol.json"]}
    if existing == artifacts:
        check_stable()
        return
    if existing is not None and existing != protocol_only:
        raise FileExistsError("Only an identical complete archive or identical protocol-only directory is accepted")
    checked = set()
    for ancestor in (target.parent, *target.parent.parents):
        _no_links(ancestor, checked)
        if ancestor.exists() and not ancestor.is_dir():
            raise ValueError("Archive parent is not a directory")
    check_stable()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ranking-v5-pilot-staging-", dir=target.parent)).resolve()
    installed = []
    try:
        for name, data in artifacts.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
        if _tree(staging) != artifacts:
            raise ValueError("Staged ranking evidence differs from validated bytes")
        check_stable()
        if existing is None:
            _rename_noreplace(staging, target)
            return
        if _tree(target) != protocol_only:
            raise FileExistsError("Protocol-only destination changed during staging")
        # Each subtree is complete before it appears; the manifest commits them.
        for name in ("training", "evaluation", "artifact-manifest.json"):
            _rename_noreplace(staging / name, target / name)
            installed.append(name)
        if _tree(target) != artifacts:
            raise ValueError("Published ranking evidence changed before completion check")
    except BaseException as error:
        # Never remove or move newly modified user bytes during rollback.
        rollback_failed = False
        for name in reversed(installed):
            destination, original = target / name, staging / name
            expected = ({key[len(name) + 1:]: value for key, value in artifacts.items() if key.startswith(name + "/")}
                        if name != "artifact-manifest.json" else artifacts[name])
            try:
                actual = _tree(destination) if destination.is_dir() else _read(destination)
                if actual != expected:
                    raise ValueError("Newly installed evidence changed; not safe to roll back")
                _rename_noreplace(destination, original)
            except (OSError, ValueError):
                rollback_failed = True
        if rollback_failed:
            raise RuntimeError("Archive installation failed; changed partial evidence was preserved for manual recovery") from error
        raise
    finally:
        if staging.exists():
            # A generated, exact sibling staging directory only; never a broad
            # directory or an unresolved caller-supplied destructive target.
            if (staging.parent != target.parent.resolve()
                    or not staging.name.startswith(".ranking-v5-pilot-staging-") or staging.is_symlink()):
                raise RuntimeError("Refusing unsafe staging cleanup")
            shutil.rmtree(staging)


def archive(*, runs, evaluation, protocol, output=BLUE_ROOT):
    """Validate terminal reports, then archive their exact bytes without selection."""
    if (not isinstance(runs, dict) or not 1 <= len(runs) <= 3
            or any(type(seed) is not int or not 0 <= seed <= 999999 for seed in runs)):
        raise ValueError("Runs must map one to three integer policy seeds to directories")
    evaluator = _evaluator()
    paths = {seed: Path(path).absolute() for seed, path in runs.items()}
    evaluation, protocol, output = Path(evaluation).absolute(), Path(protocol).absolute(), Path(output).absolute()
    checked = set()
    for path in (*paths.values(), evaluation, protocol, output):
        _no_links(path, checked)
    if (not evaluation.is_dir() or {path.name for path in evaluation.iterdir()} != set(EVALUATION_FILES)
            or any(not path.is_file() for path in evaluation.iterdir())
            or set(evaluator.EVALUATION_FILES) != set(EVALUATION_FILES)):
        raise ValueError("Archive requires the exact complete terminal evaluation file set")
    sources, archive_sources = evaluator.source_provenance(), source_provenance()
    snapshots, artifacts, documents = {}, {}, {}

    def add(name, path):
        data = _read(path, checked=checked)
        snapshots[path] = data
        if name.endswith(".jsonl.gz"):
            _json_lines(data, compressed=True)
        elif name.endswith(".json.gz"):
            documents[name] = _gzip_json(data)
        elif name.endswith(".jsonl"):
            _json_lines(data, compressed=False)
        elif name.endswith(".json"):
            documents[name] = _json_object(data)
        elif not name.endswith(".npz"):
            raise ValueError("Unexpected ranking artifact extension")
        artifacts[name] = data

    add("protocol.json", protocol)
    declared = documents["protocol.json"].get("training", {}).get("policy_seeds")
    if (not isinstance(declared, list) or any(type(seed) is not int for seed in declared)
            or len(declared) != len(paths) or set(declared) != set(paths)):
        raise ValueError("Archive run seeds must exactly match the declared policy seeds")
    for name in EVALUATION_FILES:
        add(f"evaluation/{name}", evaluation / name)
    for seed, path in sorted(paths.items()):
        for name in evaluator.trainer.RUN_FILES:
            add(f"training/seed-{seed}/{name}", path / name)
    verified = _verify(paths, evaluation, protocol, documents, sources)
    manifest = {"schema": SCHEMA, "experiment": "ranking-v5-pilot", "stage": "validation",
                "final_test_accessed": False, "default_policy_changed": False,
                "status": "Development ranking experiment; no selected or promoted replacement policy",
                "policy_seeds": declared, "scale_gate": verified["scale_gate"],
                "protocol_sha256": sha256(artifacts["protocol.json"]),
                "training_implementation_sha256": documents["protocol.json"]["training_implementation_sha256"],
                "evaluation_implementation_sha256": sources, "archive_implementation_sha256": archive_sources,
                "copy_contract": "Exact input bytes, including existing deterministic gzip; no checkpoint repacking, redaction, new scenarios, new training experiment, new behavior trajectory or endpoint selection",
                "completion_contract": "Manifest published last; identical protocol-only destination permitted; no differing artifact replacement",
                "manifest_self_hash_excluded": True,
                "files": [{"path": f"{RESULTS}/{name}", "bytes": len(data), "sha256": sha256(data)}
                          for name, data in sorted(artifacts.items())]}
    privacy_check(manifest)
    artifacts["artifact-manifest.json"] = _canonical_bytes(manifest)

    def stable():
        # Parent-link checks are shared only within this single fresh pass;
        # every subsequent stability pass starts a new checked-path set.
        current_checked = set()
        if (any(_read(path, checked=current_checked) != data for path, data in snapshots.items())
                or evaluator.source_provenance() != sources or source_provenance() != archive_sources):
            raise ValueError("Ranking source or input changed during archive preflight")

    stable()
    _install(output / RESULTS, artifacts, stable)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=RUN_DIRECTORY")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BLUE_ROOT)
    args = vars(parser.parse_args())
    runs = {}
    for item in args.pop("run"):
        seed, separator, path = item.partition("=")
        try:
            seed = int(seed)
        except ValueError:
            parser.error("Each run must be SEED=RUN_DIRECTORY")
        if not separator or seed in runs or not path:
            parser.error("Each run must have a unique policy seed and a directory")
        runs[seed] = Path(path)
    result = archive(runs=runs, **args)
    print(f"Archived {len(result['files'])} exact ranking artifacts; no policy promoted.")


if __name__ == "__main__":
    main()
