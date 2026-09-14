"""Publish reproducible adaptive Blue RL evidence without workstation paths.

This maintainer tool has no network calls and does not touch the legacy
source-manifest.json. It verifies source/weight provenance before exporting
the selected best and original initialized checkpoint, complete compressed
evaluation reports, readable summaries, training logs and a self-contained
demo. Existing differing files require --replace; every conflict is checked
before any artifact is written. Identical reruns are byte-for-byte idempotent.
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
from typing import Any, Mapping, Sequence


PUBLICATION_SCHEMA = "triad.adaptive_publication.v1"
CORE_FILES = ("adaptive_env.py", "adaptive_inputs.py", "adaptive_policy.py", "train_adaptive.py")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "RL" / "BlueTeam"
_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[/\\]|[/\\])")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant {value} is not publishable")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(data: str | bytes) -> Any:
    return json.loads(data, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def deterministic_gzip(data: bytes) -> bytes:
    buffer = io.BytesIO()
    # GzipFile avoids platform-specific OS headers from gzip.compress(mtime=0)
    # in Python 3.11. No original filename or current timestamp is embedded.
    with gzip.GzipFile(fileobj=buffer, filename="", mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def portable(value: Any, paths: Mapping[str, str] | None = None) -> Any:
    """Remove informational absolute paths; preserve numeric values and hashes."""
    mappings = paths or {}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            safe_key = portable(key, mappings)
            if safe_key in result:
                raise ValueError("Path redaction would collide JSON keys; refusing evidence loss")
            result[safe_key] = portable(item, mappings)
        return result
    if isinstance(value, list):
        return [portable(item, mappings) for item in value]
    if isinstance(value, str) and value.lower().startswith("file://"):
        return "<local-path-redacted>"
    if isinstance(value, str) and _ABSOLUTE.match(value):
        normalized = value.replace("\\", "/").rstrip("/")
        # Longest prefix wins, allowing selected checkpoint paths to map more
        # specifically than their parent run directory.
        for source, destination in sorted(mappings.items(), key=lambda item: -len(item[0])):
            if normalized.casefold() == source.casefold():
                return destination
            if normalized.casefold().startswith(source.casefold() + "/"):
                return destination + normalized[len(source):]
        return "<local-path-redacted>"
    return value


def _read_json(path: Path) -> dict:
    result = strict_json(path.read_bytes())
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return result


def _valid_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"Invalid SHA256 for {name}")
    return value


def implementation_hashes(source_root: Path | None = None) -> dict[str, str]:
    root = source_root or REPO_ROOT / "RL" / "BlueTeam" / "Python" / "triad_rl"
    return {name: sha256((root / name).read_bytes()) for name in CORE_FILES}


def _load_run(path: Path, source_hashes: Mapping[str, str]) -> dict:
    config = _read_json(path / "config.json")
    summary = _read_json(path / "summary.json")
    if config.get("schema") != "triad.adaptive_training.v1":
        raise ValueError("Unsupported adaptive training config")
    if config.get("source_sha256") != source_hashes:
        raise ValueError("Training source drift: run config differs from current four core modules")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1_000_000:
        raise ValueError("Run seed must be an integer in [0,1000000)")
    basis = {key: value for key, value in config.items()
             if key not in {"target_episodes", "config_sha256", "python", "numpy"}}
    expected = sha256(json.dumps(basis, sort_keys=True, allow_nan=False).encode())
    if config.get("config_sha256") != expected or summary.get("config_sha256") != expected:
        raise ValueError("Run configuration fingerprint does not match config/summary")
    events = [strict_json(line) for line in (path / "training.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events or any(not isinstance(event, dict) for event in events):
        raise ValueError("Training log must contain JSON object events")
    best_episode = summary.get("best_episode")
    completed = summary.get("completed_episodes")
    if not isinstance(completed, int) or not isinstance(best_episode, int) or not 0 <= best_episode <= completed:
        raise ValueError("Invalid completed/best episode counts")
    best_events = [event for event in events if event.get("event") == "validation"
                   and event.get("episode") == best_episode and event.get("selected") is True]
    if not best_events:
        raise ValueError("Selected checkpoint is missing its validation selection event")
    event = best_events[-1]
    if (event.get("success_rate") != summary.get("best_validation_success_rate")
            or event.get("mean_return") != summary.get("best_validation_mean_return")):
        raise ValueError("Run summary differs from its validation selection event")
    return {"path": path, "seed": seed, "config": config, "summary": summary,
            "events": events, "selected_event": event}


def _load_checkpoint(path: Path, run: Mapping[str, Any], initialized: bool) -> tuple[dict, bytes]:
    metadata = _read_json(path / "checkpoint.json")
    arrays = (path / "arrays.npz").read_bytes()
    if metadata.get("schema") != "triad.adaptive_checkpoint.v1":
        raise ValueError("Unsupported checkpoint schema")
    if sha256(arrays) != metadata.get("arrays_sha256"):
        raise ValueError("Checkpoint array checksum mismatch")
    _valid_hash(metadata.get("weights_sha256"), "checkpoint weights")
    state = metadata.get("training_state", {})
    config = run["config"]
    if (state.get("config_sha256") != config["config_sha256"]
            or state.get("config", {}).get("source_sha256") != config["source_sha256"]
            or state.get("config", {}).get("seed") != run["seed"]):
        raise ValueError("Checkpoint training provenance differs from selected run")
    expected_episode = 0 if initialized else run["summary"]["best_episode"]
    if state.get("completed_episodes") != expected_episode:
        raise ValueError("Checkpoint episode differs from selected/initialized episode")
    if not initialized and metadata["weights_sha256"] != run["selected_event"].get("weights_sha256"):
        raise ValueError("Best checkpoint weights differ from validation selection event")
    return metadata, arrays


def _ranges(value: Any) -> list[tuple[int, int]]:
    ranges = []
    if isinstance(value, dict):
        if "start" in value and "count" in value:
            start, count = value["start"], value["count"]
            if isinstance(start, int) and isinstance(count, int) and count > 0:
                ranges.append((start, start + count))
        else:
            for item in value.values():
                ranges.extend(_ranges(item))
    elif isinstance(value, list):
        for item in value:
            ranges.extend(_ranges(item))
    return ranges


def _selection(selected: Mapping[str, Any], runs: Sequence[Mapping[str, Any]],
               evidence: Mapping[str, Any] | None) -> dict:
    if evidence is None:
        if len(runs) > 1:
            raise ValueError("Multiple runs require --selection-report using common validation scenarios")
        return {"method": "best checkpoint selected by this run's validation only",
                "selected_seed": selected["seed"], "selected_episode": selected["summary"]["best_episode"],
                "selection_split": "validation", "heldout_used_for_selection": False,
                "seed_provenance": selected["summary"].get("seed_provenance", {}),
                "validation_score": {"success_rate": selected["summary"]["best_validation_success_rate"],
                                     "mean_return": selected["summary"]["best_validation_mean_return"]}}
    if evidence.get("selection_split") != "validation" or evidence.get("selected_seed") != selected["seed"]:
        raise ValueError("Selection report must identify the selected seed and validation split")
    rows = evidence.get("runs", [])
    seeds = {run["seed"] for run in runs}
    if not isinstance(rows, list) or {row.get("seed") for row in rows} != seeds or len(rows) != len(seeds):
        raise ValueError("Common validation selection report must cover exactly the archived run seeds")
    by_seed = {run["seed"]: run for run in runs}
    for row in rows:
        if row.get("weights_sha256") != by_seed[row["seed"]]["selected_event"].get("weights_sha256"):
            raise ValueError("Selection report weight fingerprint differs from run's best checkpoint")
        for metric in ("success_rate", "mean_return"):
            if isinstance(row.get(metric), bool) or not isinstance(row.get(metric), (int, float)):
                raise ValueError(f"Common validation row missing numeric {metric}")
    winner = next(row for row in rows if row["seed"] == selected["seed"])
    if (winner["success_rate"], winner["mean_return"]) != max((row["success_rate"], row["mean_return"]) for row in rows):
        raise ValueError("Selected seed does not maximize the declared validation ordering")
    common_ranges = _ranges(evidence.get("seed_provenance", {}).get("validation", {}))
    if len(common_ranges) != 1:
        raise ValueError("Common validation selection requires a declared scenario seed range")
    old_ranges = [interval for run in runs for interval in _ranges(run["summary"].get("seed_provenance", {}))]
    if any(max(a, c) < min(b, d) for a, b in common_ranges for c, d in old_ranges):
        raise ValueError("Common model-selection validation overlaps training/candidate-validation seeds")
    return {"method": "common unseen validation success_rate, then mean_return; no heldout selection",
            "selection_split": "validation", "heldout_used_for_selection": False,
            "selected_seed": selected["seed"], "selected_episode": selected["summary"]["best_episode"],
            "common_validation_evidence": deepcopy(dict(evidence))}


def _validate_report(report: Mapping[str, Any], split: str, best: Mapping[str, Any],
                     source_hashes: Mapping[str, str], excluded_ranges: Sequence[tuple[int, int]]) -> None:
    if (report.get("schema") != "triad.adaptive_evaluation.v1"
            or report.get("split") != split or report.get("training_performed") is not False):
        raise ValueError(f"Expected frozen {split} evaluation report")
    if report.get("checkpoint", {}).get("metadata", {}).get("weights_sha256") != best["weights_sha256"]:
        raise ValueError("Evaluation checkpoint does not match the selected best weights")
    implementation = report.get("implementation_sha256", {})
    for name in ("adaptive_env.py", "adaptive_inputs.py", "adaptive_policy.py"):
        if implementation.get(name) != source_hashes[name]:
            raise ValueError("Evaluation implementation differs from training source")
    protocol = report.get("protocol", {})
    if not protocol.get("paired_scenarios") or protocol.get("test_tuning_allowed") is not False:
        raise ValueError("Evaluation must declare paired scenarios with no test tuning")
    evaluation = report.get("seed_provenance", {}).get("evaluation", {})
    intervals = _ranges(evaluation)
    if len(intervals) != 1:
        raise ValueError("Evaluation requires one explicit seed range")
    first, last = intervals[0]
    if any(max(first, a) < min(last, b) for a, b in excluded_ranges):
        raise ValueError("Final evaluation seeds overlap training or checkpoint/model-selection validation")
    methods = report.get("methods", {})
    if not isinstance(methods, dict) or "adaptive" not in methods:
        raise ValueError("Evaluation lacks adaptive method evidence")
    reference_hashes = [episode.get("scenario_sha256") for episode in methods["adaptive"].get("episodes", [])]
    for name, method in methods.items():
        episodes = method.get("episodes", [])
        if len(episodes) != last - first:
            raise ValueError(f"Evaluation episode count mismatch: {name}")
        if [episode.get("seed") for episode in episodes] != list(range(first, last)):
            raise ValueError(f"Evaluation seed order mismatch: {name}")
        hashes = [episode.get("scenario_sha256") for episode in episodes]
        if hashes != reference_hashes or any(not isinstance(h, str) or not _SHA256.fullmatch(h) for h in hashes):
            raise ValueError(f"Evaluation scenarios are not paired: {name}")
        if any(sha256(json.dumps(episode.get("scenario"), sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()) != episode["scenario_sha256"] for episode in episodes):
            raise ValueError(f"Evaluation scenario checksum mismatch: {name}")
        if name == "adaptive" and any(episode.get(key) != best["weights_sha256"] for episode in episodes
                                      for key in ("weights_sha256_before", "weights_sha256_after")):
            raise ValueError("Adaptive evaluation weights changed or differ from checkpoint")


def publish(*, run_dir: str | Path, heldout: str | Path, demo: str | Path,
            runs: Sequence[str | Path] = (), stress: str | Path | None = None,
            selection_report: str | Path | None = None,
            sensitivity: str | Path | None = None, variability: str | Path | None = None,
            feasibility: str | Path | None = None,
            output: str | Path = DEFAULT_OUTPUT, replace: bool = False,
            source_root: Path | None = None) -> dict:
    """Verify and export the exact supplied run/evaluation set, without network I/O."""
    source_hashes = implementation_hashes(source_root)
    selected_path = Path(run_dir).resolve()
    run_paths = sorted({selected_path, *(Path(path).resolve() for path in runs)}, key=str)
    loaded_runs = [_load_run(path, source_hashes) for path in run_paths]
    if len({run["seed"] for run in loaded_runs}) != len(loaded_runs):
        raise ValueError("Archived independent runs must have distinct seeds")
    selected = next(run for run in loaded_runs if run["path"] == selected_path)
    best, best_arrays = _load_checkpoint(selected_path / "best", selected, False)
    initialized, initial_arrays = _load_checkpoint(selected_path / "initialized", selected, True)
    common_validation = _read_json(Path(selection_report)) if selection_report is not None else None
    selection = _selection(selected, loaded_runs, common_validation)
    excluded = []
    for run in loaded_runs:
        excluded.extend(_ranges(run["summary"].get("seed_provenance", {})))
    if common_validation:
        excluded.extend(_ranges(common_validation.get("seed_provenance", {})))
    paths = {str(run["path"]).replace("\\", "/"): f"RL/BlueTeam/Results/adaptive-v1/training/seed-{run['seed']}"
             for run in loaded_runs}
    paths.update({str(selected_path / "best").replace("\\", "/"): "RL/BlueTeam/Checkpoints/adaptive-v1",
                  str(selected_path / "initialized").replace("\\", "/"): "RL/BlueTeam/Checkpoints/adaptive-v1-initialized",
                  str(REPO_ROOT).replace("\\", "/"): "."})
    artifacts: dict[str, bytes] = {}
    origins: dict[str, str] = {}

    def add(name: str, data: bytes, original: bytes | None = None) -> None:
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("Export path must stay within the BlueTeam bundle")
        artifacts[name] = data
        if original is not None:
            origins[name] = sha256(original)

    for name, metadata, arrays in (("adaptive-v1", best, best_arrays), ("adaptive-v1-initialized", initialized, initial_arrays)):
        add(f"Checkpoints/{name}/arrays.npz", arrays, arrays)
        checkpoint_source = selected_path / ("best" if name == "adaptive-v1" else "initialized") / "checkpoint.json"
        add(f"Checkpoints/{name}/checkpoint.json", canonical_json(portable(metadata, paths)), checkpoint_source.read_bytes())
    for run in sorted(loaded_runs, key=lambda item: item["seed"]):
        prefix = f"Results/adaptive-v1/training/seed-{run['seed']}"
        for filename, value in (("config.json", run["config"]), ("summary.json", run["summary"])):
            add(f"{prefix}/{filename}", canonical_json(portable(value, paths)), (run["path"] / filename).read_bytes())
        add(f"{prefix}/training.jsonl", b"".join(canonical_json(portable(event, paths)) for event in run["events"]),
            (run["path"] / "training.jsonl").read_bytes())
    evaluation_ranges = []
    source_reports: dict[str, tuple[dict, str]] = {}
    for name, input_path in (("heldout", heldout), ("stress", stress)):
        if input_path is None:
            continue
        source_bytes = Path(input_path).read_bytes()
        report = strict_json(source_bytes)
        _validate_report(report, name, best, source_hashes, excluded)
        source_reports[name] = (report, sha256(source_bytes))
        interval = _ranges(report["seed_provenance"]["evaluation"])[0]
        if any(max(interval[0], a) < min(interval[1], b) for a, b in evaluation_ranges):
            raise ValueError("Heldout and stress must use distinct evaluation seed ranges")
        evaluation_ranges.append(interval)
        report = portable(report, paths)
        full = canonical_json(report)
        add(f"Results/adaptive-v1/{name}.json.gz", deterministic_gzip(full), source_bytes)
        summary = deepcopy(report)
        for method in summary["methods"].values():
            method.pop("episodes", None)
        summary["detailed_evidence"] = {"path": f"{name}.json.gz", "canonical_json_sha256": sha256(full),
                                        "note": "Complete episode/scenario/replay evidence in deterministic gzip JSON"}
        add(f"Results/adaptive-v1/{name}-summary.json", canonical_json(summary))
    for name, input_path in (("validation-sensitivity", sensitivity), ("training-run-variability", variability)):
        if input_path is None:
            continue
        source_bytes = Path(input_path).read_bytes()
        evidence = strict_json(source_bytes)
        if not isinstance(evidence, dict):
            raise ValueError(f"{name} evidence must be a JSON object")
        selection_reference = evidence.get("selection_report", {})
        weights = evidence.get("selected_weights_sha256", evidence.get("weights_sha256",
                     selection_reference.get("selected_weights_sha256", selection_reference.get("selected", {}).get("weights_sha256"))))
        if weights != best["weights_sha256"]:
            raise ValueError(f"{name} evidence does not identify the selected checkpoint weights")
        if not _ranges(evidence.get("seed_provenance", {})):
            raise ValueError(f"{name} evidence must declare scenario seed provenance")
        if name == "validation-sensitivity":
            if evidence.get("schema") != "triad.adaptive_public_input_sensitivity.v1":
                raise ValueError("Unsupported sensitivity evidence schema")
            if not evidence.get("interpretation"):
                raise ValueError("Sensitivity evidence must distinguish interventions from defence outcomes")
            if evidence.get("source_sha256") != source_hashes:
                raise ValueError("Sensitivity evidence differs from frozen training implementation")
            if evidence.get("training_performed") is not False or evidence.get("final_test_accessed") is not False:
                raise ValueError("Sensitivity evidence must be frozen and validation-only")
            if any(max(a, c) < min(b, d) for a, b in _ranges(evidence["seed_provenance"])
                   for c, d in evaluation_ranges):
                raise ValueError("Sensitivity probes overlap final evaluation seeds")
        else:
            if evidence.get("heldout_used_for_selection", evidence.get("protocol", {}).get("test_tuning_allowed")) is not False:
                raise ValueError("Variability evidence must explicitly declare no heldout model selection")
            if evidence.get("selected_training_seed_fixed_before_test", selected["seed"]) != selected["seed"]:
                raise ValueError("Variability evidence changed the validation-selected run")
            if evidence.get("training_performed") is not False:
                raise ValueError("Variability audit must evaluate frozen weights")
            if selection_report is not None and selection_reference.get("sha256") is not None:
                if selection_reference["sha256"] != sha256(Path(selection_report).read_bytes()):
                    raise ValueError("Variability audit references a different common-validation selection report")
            methods = evidence.get("methods", {})
            expected_weights = {run["selected_event"]["weights_sha256"] for run in loaded_runs}
            actual_weights = set()
            for method in methods.values():
                episodes = method.get("episodes", [])
                if not episodes:
                    raise ValueError("Variability audit is missing per-episode evidence")
                fingerprints = {episode.get(key) for episode in episodes
                                for key in ("weights_sha256_before", "weights_sha256_after")}
                if len(fingerprints) != 1:
                    raise ValueError("Variability audit weights changed across episodes")
                actual_weights.update(fingerprints)
            if actual_weights != expected_weights or len(methods) != len(loaded_runs):
                raise ValueError("Variability audit does not cover exactly the archived selected checkpoints")
        safe_evidence = portable(evidence, paths)
        if name == "training-run-variability":
            full = canonical_json(safe_evidence)
            add(f"Results/adaptive-v1/{name}.json.gz", deterministic_gzip(full), source_bytes)
            summary = deepcopy(safe_evidence)
            for method in summary["methods"].values():
                method.pop("episodes", None)
            summary["detailed_evidence"] = {"path": f"{name}.json.gz", "canonical_json_sha256": sha256(full)}
            add(f"Results/adaptive-v1/{name}-summary.json", canonical_json(summary))
        else:
            add(f"Results/adaptive-v1/{name}.json", canonical_json(safe_evidence), source_bytes)
    if feasibility is not None:
        if "stress" not in source_reports:
            raise ValueError("--feasibility requires the matching --stress report")
        source_bytes = Path(feasibility).read_bytes()
        audit = strict_json(source_bytes)
        stress_report, stress_original_hash = source_reports["stress"]
        if (not isinstance(audit, dict) or audit.get("schema") != "triad.adaptive_feasibility_audit.v1"
                or audit.get("split") != "stress" or audit.get("training_performed") is not False
                or audit.get("new_independent_test") is not False or audit.get("private_truth_used") is not True):
            raise ValueError("Feasibility evidence must be an explicitly post-hoc private-truth stress audit")
        if audit.get("source_report", {}).get("sha256") != stress_original_hash:
            raise ValueError("Feasibility source evaluation hash differs from provided stress report")
        if (audit.get("implementation_sha256") != stress_report["implementation_sha256"]
                or audit.get("source_scenario_sequence_sha256") != stress_report["scenario_sequence_sha256"]):
            raise ValueError("Feasibility implementation/scenario sequence differs from evaluated stress scenarios")
        episodes = audit.get("episodes", [])
        stress_episodes = stress_report["methods"]["adaptive"]["episodes"]
        if [(e.get("seed"), e.get("scenario_sha256")) for e in episodes] != [(e["seed"], e["scenario_sha256"]) for e in stress_episodes]:
            raise ValueError("Feasibility episode evidence differs from evaluated stress scenarios")
        if not audit.get("interpretation") or not audit.get("scenario_reuse"):
            raise ValueError("Feasibility audit requires interpretation and scenario-reuse limitations")
        audit = portable(audit, paths)
        audit["publication_context"] = {"selected_weights_sha256": best["weights_sha256"],
                                         "source_stress_original_sha256": stress_original_hash,
                                         "note": "Diagnostic only; not used to retrain or select the published checkpoint"}
        full = canonical_json(audit)
        add("Results/adaptive-v1/stress-feasibility.json.gz", deterministic_gzip(full), source_bytes)
        summary = deepcopy(audit)
        summary.pop("episodes", None)
        summary["detailed_evidence"] = {"path": "stress-feasibility.json.gz", "canonical_json_sha256": sha256(full)}
        add("Results/adaptive-v1/stress-feasibility-summary.json", canonical_json(summary))
    selection.update(schema=PUBLICATION_SCHEMA, selected_weights_sha256=best["weights_sha256"],
                     initialized_weights_sha256=initialized["weights_sha256"],
                     archived_run_seeds=sorted(run["seed"] for run in loaded_runs),
                     training_implementation_sha256=source_hashes)
    add("Results/adaptive-v1/selection.json", canonical_json(portable(selection, paths)))
    if common_validation is not None:
        # Keep the raw versioned input available to evaluate_adaptive.py's
        # --selection-report, in addition to the publication selection wrapper.
        add("Results/adaptive-v1/common-validation-selection.json",
            canonical_json(portable(common_validation, paths)), Path(selection_report).read_bytes())
    demo_bytes = Path(demo).read_bytes()
    if b"<html" not in demo_bytes.lower() or b"file://" in demo_bytes.lower():
        raise ValueError("Demo must be a self-contained HTML document without file:// links")
    demo_blocks = re.findall(
        r"<script\b(?=[^>]*\bid\s*=\s*['\"]replay-data['\"])(?=[^>]*\btype\s*=\s*['\"]application/json['\"])[^>]*>(.*?)</script\s*>",
        demo_bytes.decode("utf-8"), flags=re.IGNORECASE | re.DOTALL)
    if len(demo_blocks) != 1:
        raise ValueError("Demo must contain exactly one application/json replay-data block")
    demo_data = strict_json(demo_blocks[0])
    if (not isinstance(demo_data, dict) or demo_data.get("schema") != "triad.adaptive_demo.v1"
            or demo_data.get("weights_sha256") != best["weights_sha256"]):
        raise ValueError("Demo replay checkpoint differs from selected best weights")
    add("Results/adaptive-v1/demo.html", demo_bytes, demo_bytes)
    manifest = {"schema": PUBLICATION_SCHEMA, "selected_seed": selected["seed"],
                "selected_weights_sha256": best["weights_sha256"],
                "training_implementation_sha256": source_hashes,
                "portability": "Informational absolute paths redacted/mapped; numbers, model arrays and hash values preserved",
                "compression": "canonical strict UTF-8 JSON, gzip level9, mtime0, empty filename",
                "files": [{"path": name, "sha256": sha256(data), "bytes": len(data),
                           **({"original_sha256": origins[name]} if name in origins else {})}
                          for name, data in sorted(artifacts.items())],
                "manifest_self_hash_excluded": True}
    add("Results/adaptive-v1/artifact-manifest.json", canonical_json(manifest))
    destination = Path(output).resolve()
    # Check source did not change while reports were loaded/compressed.
    if implementation_hashes(source_root) != source_hashes:
        raise ValueError("Core source changed during publication")
    targets: dict[str, Path] = {}
    conflicts = []
    for name, data in artifacts.items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination):
            raise ValueError("Output symlink resolves outside the publication directory")
        if path.exists() and not path.is_file():
            raise IsADirectoryError(f"Publication file target is not a file: {name}")
        for ancestor in (path.parent, *path.parent.parents):
            if ancestor.exists() and not ancestor.is_dir():
                raise NotADirectoryError("Publication parent is not a directory")
            if ancestor == destination:
                break
        if path.exists() and path.read_bytes() != data:
            conflicts.append(name)
        targets[name] = path
    if conflicts and not replace:
        raise FileExistsError("Differing publication artifacts exist; review and pass --replace: " + ", ".join(conflicts))
    for name, data in artifacts.items():
        path = targets[name]
        if path.exists() and path.is_file() and path.read_bytes() == data:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run whose validation-selected best checkpoint will be published")
    parser.add_argument("--runs", type=Path, nargs="*", default=[], help="All independent final training runs to archive")
    parser.add_argument("--selection-report", type=Path, help="Common validation comparison required when archiving multiple runs")
    parser.add_argument("--sensitivity", type=Path, help="Validation-only controlled public-input sensitivity evidence")
    parser.add_argument("--variability", type=Path, help="Post-selection independent training-run variation evidence")
    parser.add_argument("--feasibility", type=Path, help="Post-hoc feasibility audit bound to the supplied stress report")
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--stress", type=Path)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true", help="Explicitly permit replacement of differing targeted artifacts")
    result = publish(**vars(parser.parse_args(argv)))
    print(json.dumps({"exported_files": len(result["files"]) + 1,
                      "selected_seed": result["selected_seed"],
                      "weights_sha256": result["selected_weights_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
