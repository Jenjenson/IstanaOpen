"""Archive the robust-v2 validation experiment without promoting its candidate.

No network, training, inference, source edits or default-policy changes occur.
Complete run and selection artifacts are copied byte-for-byte only after all
integrity, privacy and destination-conflict checks pass. Differing existing
files are never replaced. Repeating the same publication is idempotent.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BLUE_ROOT = REPO_ROOT / "RL" / "BlueTeam"
PYTHON_ROOT = BLUE_ROOT / "Python"
sys.path.insert(0, str(PYTHON_ROOT))

import select_robust
from evaluate_robust import V1_WEIGHTS, checkpoint_files, implementation_fingerprints, validate_seed_range
from triad_rl.adaptive_evaluation import canonical_hash
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.train_robust import source_provenance


SCHEMA = "triad.robust_publication.v1"
RESULTS = "Results/robust-v2"
PROFILES = ("normal", "stress", "capability")
RUN_FILES = ("config.json", "summary.json", "training.jsonl",
             "initialized/checkpoint.json", "initialized/arrays.npz",
             "best/checkpoint.json", "best/arrays.npz", "last/checkpoint.json", "last/arrays.npz")
MAX_JSON_BYTES = 256 * 1024 * 1024
_PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|file://|\\\\[^\\\s]+[\\/]|^/|"
                           r"(?:^|[\s='\"])/(?:home|Users|mnt|tmp|var|private|root|workspace|workspaces)/",
                           flags=re.IGNORECASE)
_SECRET = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
                      r"AKIA[A-Z0-9]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|"
                      r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+[A-Za-z0-9._~+/-]{16,})")
_SECRET_KEYS = {"password", "passwd", "access_token", "refresh_token", "api_key",
                "client_secret", "private_key", "authorization"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not publishable")
        result[key] = value
    return result


def strict_json(data: bytes | str) -> Any:
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON evidence exceeds publication size limit")
    result = json.loads(data, object_pairs_hook=_unique_object,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON is not publishable")))
    json.dumps(result, allow_nan=False)  # Also rejects exponent overflow, e.g. 1e999.
    return result


def privacy_check(value: Any) -> None:
    """Fail closed; never redact bytes that underpin checkpoint/run hashes."""
    if isinstance(value, str):
        if _PRIVATE_PATH.search(value) or _SECRET.search(value):
            raise ValueError("Evidence contains a workstation-absolute path or potential secret")
    elif isinstance(value, dict):
        for key, item in value.items():
            privacy_check(key)
            if key.lower().replace("-", "_") in _SECRET_KEYS and item not in (None, "", False):
                raise ValueError("Evidence contains a populated credential field")
            privacy_check(item)
    elif isinstance(value, list):
        for item in value:
            privacy_check(item)


def _json_object(data: bytes | str) -> dict:
    value = strict_json(data)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object evidence")
    privacy_check(value)
    return value


def _gzip_json(data: bytes) -> dict:
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        decoded = handle.read(MAX_JSON_BYTES + 1)
    return _json_object(decoded)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _source_files(additional: Sequence[str] = ()) -> dict[str, str]:
    sources = {f"RL/BlueTeam/Python/{name}": digest for name, digest in implementation_fingerprints().items()}
    sources["RL/BlueTeam/Python/select_robust.py"] = sha256(Path(select_robust.__file__).read_bytes())
    sources["Tools/publish_robust_rl.py"] = sha256(Path(__file__).read_bytes())
    for name in additional:
        sources[f"RL/BlueTeam/Python/{name}"] = sha256((PYTHON_ROOT / name).read_bytes())
    return sources


def _validate_demo(data: bytes, weights: str, selected_files: dict[str, str], lineage: dict) -> None:
    from demo_robust import implementation_fingerprints as demo_sources

    html = data.decode("utf-8")
    if "<html" not in html.lower() or _PRIVATE_PATH.search(html) or _SECRET.search(html):
        raise ValueError("Replay must be a portable HTML document without workstation paths or secrets")
    if re.search(r"(?:src|href)\s*=\s*['\"](?:https?:)?//", html, flags=re.IGNORECASE):
        raise ValueError("Replay must be self-contained, without external assets")
    blocks = re.findall(r"<script\b(?=[^>]*\bid\s*=\s*['\"]replay-data['\"])"
                        r"(?=[^>]*\btype\s*=\s*['\"]application/json['\"])[^>]*>(.*?)</script\s*>",
                        html, flags=re.IGNORECASE | re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("Replay requires exactly one application/json replay-data block")
    envelope = _json_object(blocks[0])
    if (envelope.get("schema") != "triad.adaptive_demo.v1"
            or envelope.get("weights_sha256") != weights or not envelope.get("replays")):
        raise ValueError("Replay envelope differs from selected candidate")
    for replay in envelope["replays"]:
        audit = replay.get("audit", {})
        before, after = audit.get("policy_state_before", {}), audit.get("policy_state_after", {})
        if (replay.get("schema") != "triad.robust_replay.v1" or replay.get("release") != "robust-v2"
                or replay.get("stage") != "validation" or not replay.get("frames")
                or audit.get("training_performed") is not False
                or audit.get("independent_final_test_evidence") is not False
                or audit.get("ground_truth_in_policy_observation") is not False
                or before != after or before.get("weights_sha256") != weights
                or audit.get("checkpoint_files_sha256_before") != selected_files
                or audit.get("checkpoint_files_sha256_after") != selected_files
                or audit.get("implementation_sha256_before") != demo_sources()
                or audit.get("implementation_sha256_after") != demo_sources()):
            raise ValueError("Replay is not frozen validation evidence for the selected checkpoint/current sources")
        case = {key: replay[key] for key in ("scenario", "catalogue", "case_metadata")}
        if canonical_hash(case) != audit.get("scenario_catalogue_metadata_sha256"):
            raise ValueError("Replay case fingerprint mismatch")
        validate_seed_range(replay["seed"], 1, "validation", lineage)


def _validate_stop_probe(data: bytes, allowed: list[dict]) -> dict:
    """Optional diagnostic; its declared hashes/seed ranges never become tests."""
    from probe_stop_exploration import source_fingerprints, summarize_records

    evidence = _json_object(data)
    if (evidence.get("schema") != "triad.stop_exploration_probe.v1"
            or evidence.get("training_performed") is not False
            or evidence.get("final_test_accessed") is not False
            or evidence.get("independent_unseen_evidence") is not False
            or evidence.get("simulation_rollouts_performed") is not False
            or evidence.get("private_truth_accessed") is not False
            or evidence.get("weights_sha256") != evidence.get("weights_sha256_after")
            or evidence.get("checkpoint_files_sha256_before") != evidence.get("checkpoint_files_sha256_after")
            or evidence.get("policy_rng_sha256_before") != evidence.get("policy_rng_sha256_after")):
        raise ValueError("Stop probe must identify the selected frozen candidate and validation-only scope")
    if (evidence.get("source_sha256_before") != source_fingerprints()
            or evidence.get("source_sha256_after") != source_fingerprints()):
        raise ValueError("Stop probe differs from frozen training sources")
    if not any(evidence.get("weights_sha256") == item["weights_sha256"]
               and evidence.get("checkpoint_files_sha256_before") == item["files"]
               and evidence.get("policy_rng_sha256_before") == item["rng_sha256"] for item in allowed):
        raise ValueError("Stop probe checkpoint differs from selected candidate or frozen-v1 reference")
    interval = evidence.get("seed_provenance", {}).get("validation_diagnostic", {})
    # Intentional validation reuse is legitimate for this explicitly labeled
    # diagnostic. Do not misrepresent its inputs as new independent scenarios.
    validate_seed_range(interval.get("start"), interval.get("count"), "validation")
    records = evidence.get("records", [])
    if (evidence.get("profile") not in (*PROFILES, "mixed")
            or [row.get("scenario_seed") for row in records] != list(range(interval["start"], interval["start"] + interval["count"]))
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(row.get("public_input_sha256", ""))) for row in records)):
        raise ValueError("Stop probe records are incomplete or have invalid public-input fingerprints")
    groups = {"all": summarize_records(records)}
    for label, threshold in (("coverage_le_0.01", .01), ("coverage_le_0.001", .001), ("coverage_zero", 0.)):
        groups[label] = summarize_records([row for row in records if row["max_legal_marginal_coverage"] <= threshold])
    if groups != evidence.get("groups"):
        raise ValueError("Stop probe summaries differ from its complete public-only records")
    return evidence


def publish(*, runs: Mapping[int, str | Path], selection_dir: str | Path,
            protocol: str | Path = BLUE_ROOT / RESULTS / "protocol.json",
            demo: str | Path | None = None, stop_probes: Sequence[str | Path] = (),
            output: str | Path = BLUE_ROOT) -> dict:
    """Publish a complete protocol-declared three-run validation experiment.

    Production robust-v2 declares seeds 101/102/103 and 8000 episodes each.
    Counts are verified against the exact supplied protocol, permitting small
    independent fixture protocols for publisher tests without running 24000
    training episodes. No final-test data is accepted by this publisher.
    """
    output, selection_dir, protocol = Path(output).resolve(), Path(selection_dir).resolve(), Path(protocol).resolve()
    paths = {seed: Path(path).resolve() for seed, path in runs.items()}
    if (len(paths) != 3 or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in paths)):
        raise ValueError("Publication requires exactly three explicitly keyed training runs")
    extra_sources = (["demo_robust.py", "demo_adaptive.py", "adaptive_demo.html"] if demo is not None else [])
    if stop_probes:
        extra_sources.append("probe_stop_exploration.py")
    sources_before = _source_files(extra_sources)
    source_inputs: dict[Path, bytes] = {}
    artifacts: dict[str, bytes] = {}

    def read(path: Path) -> bytes:
        data = path.read_bytes()
        if path in source_inputs and source_inputs[path] != data:
            raise ValueError("Input artifact changed during publication preflight")
        source_inputs[path] = data
        return data

    def add(name: str, data: bytes) -> None:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Publication paths must remain inside the BlueTeam bundle")
        if name in artifacts and artifacts[name] != data:
            raise ValueError("Duplicate publication destination has differing content")
        artifacts[name] = data

    protocol_bytes = read(protocol)
    declared = _json_object(protocol_bytes)
    if (declared.get("schema") != "triad.robust_experiment_protocol.v1"
            or sorted(declared.get("candidate_training", {}).get("run_seeds", [])) != sorted(paths)):
        raise ValueError("Protocol must declare exactly the supplied three training run seeds")
    selection_bytes = read(selection_dir / "selection.json")
    selection = _json_object(selection_bytes)
    if (selection.get("schema") != "triad.robust_model_selection.v1"
            or selection.get("stage") != "validation" or selection.get("final_test_accessed") is not False
            or selection.get("protocol_sha256") != sha256(protocol_bytes)
            or selection.get("implementation_sha256") != implementation_fingerprints()
            or selection.get("selection_implementation_sha256") != sources_before["RL/BlueTeam/Python/select_robust.py"]):
        raise ValueError("Selection schema, validation scope, protocol or implementation differs")
    if set(selection.get("candidate_metadata", {})) != {str(seed) for seed in paths}:
        raise ValueError("Selection must bind all and only the supplied candidate runs")
    actual_candidates = {}
    for seed, run in sorted(paths.items()):
        blobs = {name: read(run / name) for name in RUN_FILES}
        for name, data in blobs.items():
            if name.endswith(".json"):
                _json_object(data)
            elif name.endswith(".jsonl"):
                for line in data.splitlines():
                    if line.strip():
                        _json_object(line)
        policy = AdaptivePolicy.load(run / "best", feature_names=FEATURE_NAMES)
        info = {"weights_sha256": policy.weights_fingerprint(), "metadata": deepcopy(policy.metadata),
                "checkpoint_files_sha256": checkpoint_files(run / "best"),
                "complete_run": select_robust._run_evidence(run / "best", policy, seed, declared)}
        if info != selection["candidate_metadata"][str(seed)]:
            raise ValueError("Selection candidate metadata differs from the exact complete run/checkpoints")
        actual_candidates[seed] = info
        for name, data in blobs.items():
            add(f"{RESULTS}/training/seed-{seed}/{name}", data)
    reference_path = BLUE_ROOT / "Checkpoints" / "adaptive-v1"
    for name in ("checkpoint.json", "arrays.npz"):
        read(reference_path / name)
    reference = AdaptivePolicy.load(reference_path, feature_names=FEATURE_NAMES)
    old_selection = _json_object(read(BLUE_ROOT / "Results" / "adaptive-v1" / "common-validation-selection.json"))
    if reference.weights_fingerprint() != V1_WEIGHTS or old_selection.get("selected_weights_sha256") != V1_WEIGHTS:
        raise ValueError("Published v1 reference or its selection report changed")
    lineage = {"candidate_checkpoints": {str(seed): info["metadata"] for seed, info in actual_candidates.items()},
               "complete_training_runs": {str(seed): info["complete_run"]["last_checkpoint_metadata"]
                                          for seed, info in actual_candidates.items()},
               "adaptive_v1_checkpoint": reference.metadata,
               "adaptive_v1_selection": old_selection["seed_provenance"]}
    if selection.get("seed_provenance", {}).get("lineage") != lineage:
        raise ValueError("Selection seed lineage differs from complete candidate/reference evidence")
    range_rows = declared.get("candidate_selection", {}).get("seed_ranges", [])
    if len(range_rows) != 3 or {row.get("profile") for row in range_rows} != set(PROFILES):
        raise ValueError("Protocol needs exactly one common-validation range per profile")
    ranges = {row["profile"]: row for row in range_rows}
    if selection.get("seed_provenance", {}).get("selection_validation") != range_rows:
        raise ValueError("Selection validation seed ranges differ from protocol")
    if set(selection.get("reports", {})) != set(PROFILES):
        raise ValueError("Selection requires all three complete compressed validation reports")
    reports = {}
    for profile in PROFILES:
        entry = selection["reports"][profile]
        filename = f"common-validation-{profile}.json.gz"
        if entry.get("path") != filename:
            raise ValueError("Unexpected compressed report path")
        data = read(selection_dir / filename)
        if sha256(data) != entry.get("sha256") or len(data) != entry.get("bytes"):
            raise ValueError("Compressed validation report checksum or size differs from selection")
        report = _gzip_json(data)
        expected_range = {key: ranges[profile][key] for key in ("start", "count")}
        if (report.get("implementation_sha256") != implementation_fingerprints()
                or report.get("implementation_sha256_after") != implementation_fingerprints()
                or report.get("training_performed") is not False
                or report.get("seed_provenance", {}).get("evaluation") != expected_range
                or report.get("seed_provenance", {}).get("declared_lineage") != lineage
                or set(report.get("methods", {})) != {*(f"candidate_{seed}" for seed in paths), "adaptive_v1", "greedy_public"}):
            raise ValueError("Validation report source, exposure or compared methods differs")
        validate_seed_range(expected_range["start"], expected_range["count"], "validation", lineage)
        validate_seed_range(expected_range["start"], expected_range["count"], "validation",
                            {p: row for p, row in ranges.items() if p != profile})
        for record in report["methods"]["adaptive_v1"]["episodes"]:
            if any(record.get(key) != V1_WEIGHTS for key in ("weights_sha256_before", "weights_sha256_after")):
                raise ValueError("Validation reference weights differ from frozen v1")
        reports[profile] = report
        add(f"{RESULTS}/{filename}", data)
    selected, scores = select_robust.select_from_reports(reports, actual_candidates)
    if selected != selection.get("selected") or scores != selection.get("candidates"):
        raise ValueError("Recomputed paired-validation candidate ranking differs from selection")
    expected_references = {profile: {name: reports[profile]["methods"][name]["summary"]
                                     for name in ("adaptive_v1", "greedy_public")} for profile in PROFILES}
    if selection.get("reference_summaries") != expected_references:
        raise ValueError("Selection reference summaries differ from complete reports")
    selected_seed, weights = selected["seed"], selected["weights_sha256"]
    for role, destination in (("best", "robust-v2-candidate"), ("initialized", "robust-v2-candidate-initialized")):
        for filename in ("checkpoint.json", "arrays.npz"):
            add(f"Checkpoints/{destination}/{filename}", source_inputs[paths[selected_seed] / role / filename])
    add(f"{RESULTS}/protocol.json", protocol_bytes)
    add(f"{RESULTS}/selection.json", selection_bytes)
    full_lineage = {"training_and_selection": lineage, "common_validation": range_rows}
    if demo is not None:
        data = read(Path(demo).resolve())
        _validate_demo(data, weights, actual_candidates[selected_seed]["checkpoint_files_sha256"], full_lineage)
        add(f"{RESULTS}/demo.html", data)
    probe_pairings = {}
    probe_descriptions = []
    selected_metadata = actual_candidates[selected_seed]["metadata"]
    allowed_probes = [{"weights_sha256": weights,
                       "files": actual_candidates[selected_seed]["checkpoint_files_sha256"],
                       "rng_sha256": canonical_hash(selected_metadata["rng_state"])},
                      {"weights_sha256": V1_WEIGHTS, "files": checkpoint_files(reference_path),
                       "rng_sha256": canonical_hash(reference.metadata["rng_state"])}]
    for index, probe_path in enumerate(stop_probes):
        data = read(Path(probe_path).resolve())
        probe = _validate_stop_probe(data, allowed_probes)
        interval = probe["seed_provenance"]["validation_diagnostic"]
        key = (probe["profile"], interval["start"], interval["count"])
        pairing = [row["public_input_sha256"] for row in probe["records"]]
        if key in probe_pairings and probe_pairings[key] != pairing:
            raise ValueError("Matched STOP probe scenarios have different public-input fingerprints")
        probe_pairings[key] = pairing
        name = f"{RESULTS}/stop-probe-{index + 1}.json"
        add(name, data)
        probe_descriptions.append({"path": name, "weights_sha256": probe["weights_sha256"],
                                   "profile": probe["profile"], "seed_range": interval,
                                   "independent_unseen_evidence": False})
    manifest = {"schema": SCHEMA, "release": "robust-v2", "stage": "validation",
                "status": "experimental candidate; not promoted as a robust policy",
                "independent_final_test_evidence": False,
                "selected_seed": selected_seed, "selected_weights_sha256": weights,
                "initialized_weights_sha256": V1_WEIGHTS, "archived_run_seeds": sorted(paths),
                "protocol_sha256": sha256(protocol_bytes), "selection_sha256": sha256(selection_bytes),
                "source_files_sha256": sources_before,
                "evidence_scope": "Complete training and common validation only; no new independent robustness-success claim",
                "copy_contract": "All input artifact bytes preserved exactly; no redaction or checkpoint repacking",
                "default_policy_changed": False, "manifest_self_hash_excluded": True,
                "stop_probes": probe_descriptions,
                "files": [{"path": name, "sha256": sha256(data), "bytes": len(data)}
                          for name, data in sorted(artifacts.items())]}
    privacy_check(manifest)
    add(f"{RESULTS}/artifact-manifest.json", _canonical_bytes(manifest))
    if _source_files(extra_sources) != sources_before or any(path.read_bytes() != data for path, data in source_inputs.items()):
        raise ValueError("Source or input artifact changed during publication preflight")
    targets = {}
    for name, data in artifacts.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output):
            raise ValueError("Output symlink escapes the BlueTeam publication directory")
        if path.exists() and not path.is_file():
            raise ValueError("Publication target is not a file")
        for ancestor in (path.parent, *path.parent.parents):
            if ancestor.exists() and not ancestor.is_dir():
                raise ValueError("Publication parent is not a directory")
            if ancestor == output:
                break
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Differing existing artifact will not be replaced: {name}")
        targets[name] = path
    # No destination directory/file is created until every artifact has passed
    # all validation and conflict checks. Existing identical files stay intact.
    for name, data in artifacts.items():
        path = targets[name]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=RUN_DIRECTORY")
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=BLUE_ROOT / RESULTS / "protocol.json")
    parser.add_argument("--demo", type=Path)
    parser.add_argument("--stop-probe", dest="stop_probes", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path, default=BLUE_ROOT)
    args = vars(parser.parse_args(argv))
    runs = {}
    for entry in args.pop("run"):
        seed, separator, path = entry.partition("=")
        if not separator or not seed.isdigit() or not path or int(seed) in runs:
            parser.error("Each --run must be a unique SEED=RUN_DIRECTORY pair")
        runs[int(seed)] = Path(path)
    manifest = publish(runs=runs, **args)
    print(json.dumps({"status": manifest["status"], "selected_seed": manifest["selected_seed"],
                      "weights_sha256": manifest["selected_weights_sha256"],
                      "files": len(manifest["files"]) + 1}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
