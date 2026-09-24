"""Build or verify this evidence archive; never contacts the simulator or trains.

Verification needs only Python's standard library. --load-policies additionally
uses the repository's NumPy-based policy loader to check offline deployment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STUDY = Path("Saved/Experiments/four-sensor-20260924")
ORIGINAL_ROOT = "D:/triad/IstanaOpen-LearningReview/"
OMIT_KEYS = {"consoleUrl", "consoleUrls", "bridgePort"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {portable(key): portable(item) for key, item in value.items()
                if key not in OMIT_KEYS}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        return (value.replace(ORIGINAL_ROOT.replace("/", "\\\\"), "")
                .replace(ORIGINAL_ROOT.replace("/", "\\"), "").replace(ORIGINAL_ROOT, ""))
    return value


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def source_id(value):
    return portable(value).replace("\\", "/")


def build(root):
    if (HERE / "manifest.json").exists():
        raise ValueError("Archive already exists; refusing to overwrite evidence")
    files = []

    def save(destination, source, mode="copy", pointer=None):
        source = Path(source)
        raw = (root / source).read_bytes()
        if mode == "json":
            data = json_bytes(portable(json.loads(raw)))
        elif mode == "text":
            data = portable(raw.decode("utf-8")).encode("utf-8")
        elif mode == "gzip":
            data = gzip.compress(raw, compresslevel=9, mtime=0)
        elif mode == "extract":
            value = json.loads(raw)
            for key in pointer:
                value = value[key]
            data = json_bytes(portable(value))
        else:
            data = raw
        target = HERE / destination
        if target.exists():
            raise ValueError(f"Refusing to overwrite {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        row = {"path": destination, "bytes": len(data), "sha256": digest(data),
               "source": source.as_posix(), "sourceSha256": digest(raw), "transformation": mode}
        if pointer:
            row["jsonPointerKeys"] = pointer
        if mode == "gzip":
            row["uncompressedSha256"] = digest(raw)
        files.append(row)

    results = read(root / STUDY / "results.json")
    if results["phase"] != "complete" or len(results["trials"]) != 3:
        raise ValueError("Only the completed three-trial study can be archived")
    trials = []
    for trial in results["trials"]:
        letter, seed = trial["trial"], trial["seed"]
        directory = f"trials/{letter}-{seed}"
        run = Path(source_id(trial["outputDirectory"]))
        model = Path("Saved/WarningTraining/models") / trial["modelId"]
        for name in ("configuration.json", "summary.json", "evaluation-summary.json", "test-evaluation.json"):
            save(f"{directory}/{name}", run / name, "json")
        for name in ("training.jsonl", "evaluations.jsonl"):
            save(f"{directory}/{name}.gz", run / name, "gzip")
        save(f"{directory}/final-policy.json", run / "final-policy.json")
        save(f"{directory}/best-policy.json", model / "best-policy.json")
        save(f"{directory}/native-context.json", run / "recording-manifest.json", "extract", ["context"])
        save(f"{directory}/model-metadata.json", model / "model.json", "json")
        final_hash = digest((root / run / "final-policy.json").read_bytes())
        best_hash = digest((root / model / "best-policy.json").read_bytes())
        if final_hash != digest((root / run / "policy-0500.json").read_bytes()):
            raise ValueError("Final policy differs from episode-500 checkpoint")
        if best_hash != digest((root / run / "policy-0000.json").read_bytes()):
            raise ValueError("Selected policy differs from the initial checkpoint")
        trials.append({"trial": letter, "seed": seed, "directory": directory,
                       "sourceRun": run.as_posix(), "sourceModelId": trial["modelId"],
                       "completedEpisodes": 500, "selectedEpisode": 0,
                       "finalPolicy": f"{directory}/final-policy.json",
                       "selectedPolicy": f"{directory}/best-policy.json",
                       "sourceCheckpointMatches": {"policy-0500.json": final_hash,
                                                   "policy-0000.json": best_hash}})
    for name in ("protocol.json", "results.json", "diagnostics.json", "reporting-recovery.json"):
        save(name, STUDY / name, "json")
    for name in ("report.md", "diagnostics.md"):
        save(name, STUDY / name, "text")
    for subdir, names in {
        "attempt-01-startup-failure": ("ATTEMPT.md", "results.json", "report.md"),
        "reporting-failure-evidence": ("README.md", "results.json", "report.md"),
    }.items():
        for name in names:
            save(f"history/{subdir}/{name}", STUDY / subdir / name,
                 "json" if name.endswith(".json") else "text")
    for name in ("README.md", "archive.py", ".gitattributes"):
        data = (HERE / name).read_bytes()
        files.append({"path": name, "bytes": len(data), "sha256": digest(data),
                      "transformation": "archive-support"})
    manifest = {
        "schema": "istana.portable_four_sensor_study.v1",
        "archivedUtc": datetime.now(timezone.utc).isoformat(),
        "archiveAssemblyCommit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "trainingCommit": None,
        "trainingCommitNote": "The original run did not record a training commit; assembly commit is not training provenance.",
        "sourceIdsAreDependencies": False,
        "portability": "JSON/text remove the original machine root and operational console/bridge keys; policies and decompressed JSONL retain exact original bytes. Embedded historical hashes refer to originals. Source IDs identify provenance, not required paths.",
        "trials": trials,
        "files": files,
    }
    (HERE / "manifest.json").write_bytes(json_bytes(manifest))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def layout_key(placements):
    return tuple(sorted((row["profileId"], row["siteId"], row["yawDeg"], row["pitchDeg"])
                        for row in placements))


def verify(source_root=None, load_policies=False):
    manifest = read(HERE / "manifest.json")
    for row in manifest["files"]:
        path = HERE / row["path"]
        data = path.read_bytes()
        check(len(data) == row["bytes"] and digest(data) == row["sha256"], f"Hash/size mismatch: {path}")
        unpacked = gzip.decompress(data) if row["transformation"] == "gzip" else data
        if row["transformation"] == "gzip":
            check(digest(unpacked) == row["uncompressedSha256"] == row["sourceSha256"], "Log source mismatch")
        if row["path"] not in {"archive.py", "README.md"}:
            check(not re.search(rb"(?<![A-Za-z])[A-Za-z]:[\\/]", unpacked), f"Absolute machine path: {path}")
        if source_root and "source" in row:
            original = (source_root / row["source"]).read_bytes()
            check(digest(original) == row["sourceSha256"], f"Original changed: {row['source']}")
            mode = row["transformation"]
            expected = original
            if mode == "json":
                expected = json_bytes(portable(json.loads(original)))
            elif mode == "text":
                expected = portable(original.decode("utf-8")).encode("utf-8")
            elif mode == "extract":
                value = json.loads(original)
                for key in row["jsonPointerKeys"]:
                    value = value[key]
                expected = json_bytes(portable(value))
            check(unpacked == expected, f"Source transformation mismatch: {row['path']}")
    protocol, results = read(HERE / "protocol.json"), read(HERE / "results.json")
    check(results["phase"] == "complete", "Incomplete result report")
    check([r["seed"] for r in manifest["trials"]] == [917, 918, 919], "Unexpected trial set")
    report = []
    shared_test = None
    for trial, result in zip(manifest["trials"], results["trials"]):
        directory = HERE / trial["directory"]
        config, summary = read(directory / "configuration.json"), read(directory / "summary.json")
        final, best = read(directory / "final-policy.json"), read(directory / "best-policy.json")
        model = read(directory / "model-metadata.json")
        selected = read(directory / "evaluation-summary.json")
        test = read(directory / "test-evaluation.json")
        for key, value in protocol["baseConfiguration"].items():
            check(config[key] == value, f"Protocol mismatch: {trial['trial']} {key}")
        check(config["seed"] == trial["seed"], "Wrong training seed")
        check(summary["completedEpisodes"] == 500 and not summary["stoppedEarly"], "Incomplete native run")
        check(summary["bestEpisode"] == selected["bestEpisode"] == model["bestEpisode"] == result["bestEpisode"] == 0,
              "Selected checkpoint changed")
        check(summary["modelId"] == model["id"] == result["modelId"] == trial["sourceModelId"], "Model identity mismatch")
        check(final["updates"] == 25 and final["optimizer_steps"] == 100, "Wrong final optimizer counters")
        check(best["updates"] == 0 and best["optimizer_steps"] == 0, "Wrong selected optimizer counters")
        check(final["option_logits"] != best["option_logits"], "Final and initial weights unexpectedly equal")
        for policy in (final, best):
            check(policy["sensor_count"] == 4 and policy["allowed_sensor_ids"] == ["thermal"], "Wrong selected sensor contract")
            check(policy["schema"] == "istana.warning_directional_masked_ppo.v1", "Wrong policy schema")
        training = [json.loads(line) for line in gzip.decompress((directory / "training.jsonl.gz").read_bytes()).splitlines()]
        evaluations = [json.loads(line) for line in gzip.decompress((directory / "evaluations.jsonl.gz").read_bytes()).splitlines()]
        check([row["episode"] for row in training] == list(range(1, 501)), "Incomplete training log")
        check([row["episode"] for row in evaluations] == list(range(20, 501, 20)), "Incomplete validation log")
        for row in training:
            check(row["metrics"]["targets"] == 5 and len(row["placements"]) == 4, "Wrong native target/sensor count")
            check({sensor["profileId"] for sensor in row["placements"]} == {"thermal"}, "Unselected sensor type")
        check(test["usedForSelection"] is False, "Held-out test used for selection")
        panels = [(selected["best"], selected["contractor"], protocol["validationSeeds"], result["validation"]),
                  (test["policy"], test["contractor"], protocol["testSeeds"], result["test"])]
        for candidate, baseline, seeds, aggregate in panels:
            check([row["seed"] for row in candidate["cases"]] == seeds, "Panel seed mismatch")
            check([row["seed"] for row in baseline["cases"]] == seeds, "Baseline panel mismatch")
            deltas = []
            for learned, naive in zip(candidate["cases"], baseline["cases"]):
                check(learned["targets"] == naive["targets"] == 5, "Wrong panel target count")
                for key in ("trajectorySha256", "constraintsSha256"):
                    check(learned[key] == naive[key], "Unmatched physical/scenario panel")
                deltas.append(learned["mean_drone_warning_s"] - naive["mean_drone_warning_s"])
            check(deltas == aggregate["pairedDeltasSeconds"] == [0.0] * 10, "Unexpected warning gains")
            mean = sum(row["mean_drone_warning_s"] for row in candidate["cases"]) / len(seeds)
            detected = sum(row["detected_fraction"] for row in candidate["cases"]) / len(seeds)
            check(abs(mean - aggregate["meanWarningSeconds"]) < 1e-10, "Mean report mismatch")
            check(abs(detected - aggregate["detected_fraction"]) < 1e-10, "Detection report mismatch")
        current_test = [(row["seed"], row["trajectorySha256"]) for row in test["policy"]["cases"]]
        if shared_test is None:
            shared_test = current_test
        check(shared_test == current_test, "Trials do not share the declared test scenarios")
        for evaluation in evaluations:
            check(evaluation["evaluation"]["cases"] == selected["initial"]["cases"], "Validation checkpoint differs from reported unchanged panel")
        if source_root:
            for name, expected_hash in trial["sourceCheckpointMatches"].items():
                check(digest((source_root / trial["sourceRun"] / name).read_bytes()) == expected_hash, "Checkpoint provenance mismatch")
        if load_policies:
            sys.path.insert(0, str(HERE.parents[1] / "Python"))
            from triad_rl.warning_algorithms import TRAINING_ALGORITHMS
            context = read(directory / "native-context.json")
            for filename in ("final-policy.json", "best-policy.json"):
                policy = TRAINING_ALGORITHMS["ppo"]["factory"].load(directory / filename, context)
                placements, _ = policy.plan(context, deterministic=True)
                check(layout_key(placements) == layout_key(config["initialLayout"]["placements"]), "Greedy policy deployment differs from recorded diagnostics")
        report.append({"trial": trial["trial"], "episodes": len(training), "validationCheckpoints": len(evaluations),
                       "uniqueLayouts": len({layout_key(row["placements"]) for row in training}),
                       "selectedEpisode": 0, "freshUpdates": final["updates"], "adamSteps": final["optimizer_steps"],
                       "testMeanWarningSeconds": result["test"]["meanWarningSeconds"],
                       "testDeltaSeconds": result["test"]["deltaSeconds"]})
    print(json.dumps({"verified": True, "files": len(manifest["files"]),
                      "bytes": sum(row["bytes"] for row in manifest["files"]),
                      "comparedOriginals": source_root is not None, "loadedPolicies": load_policies,
                      "sharedTestScenarios": len(shared_test), "trials": report}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--source-root", type=Path, help="Optional original repository root for source hash comparison")
    parser.add_argument("--load-policies", action="store_true", help="Also test offline policy loading; requires repository Python dependencies")
    args = parser.parse_args()
    if args.command == "build":
        if args.source_root is None:
            parser.error("build requires --source-root containing original Saved study outputs")
        build(args.source_root.resolve())
    verify(args.source_root.resolve() if args.source_root else None, args.load_policies)
