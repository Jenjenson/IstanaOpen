"""Export the selected Blue RL source and evidence from an existing TRIAD checkout.

Maintainer utility; consumers do not need the original checkout. Uses an explicit
allowlist and rewrites informational machine-local paths in published metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triad-root", required=True, type=Path)
    args = parser.parse_args()
    source = args.triad_root.resolve()
    target = Path(__file__).resolve().parents[1]
    if not (source / "TRIAD.uproject").is_file():
        raise ValueError("Expected an existing TRIAD project root")
    if not (target / "IstanaOpen.uproject").is_file():
        raise ValueError("Run the exporter from the IstanaOpen Tools directory")
    bundle = target / "RL/BlueTeam"
    records = []

    def copy(relative, destination):
        src, dst = source / relative, bundle / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        records.append({"source": relative, "published": str(dst.relative_to(target)).replace("\\", "/"),
                        "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})

    def metadata(relative, destination):
        src, dst = source / relative, bundle / destination
        value = json.loads(src.read_text(encoding="utf-8"))

        def portable(item):
            if isinstance(item, dict):
                return {key: portable(val) for key, val in item.items()}
            if isinstance(item, list):
                return [portable(val) for val in item]
            if isinstance(item, str):
                normalized = item.replace("\\", "/")
                if normalized.lower().endswith("/content/triad/rl/defaulttrainingconfig.json"):
                    return "RL/BlueTeam/DefaultTrainingConfig.json"
                if normalized.lower().endswith("/pilot/episode_000210_final"):
                    return "RL/BlueTeam/Checkpoints/toy-210"
            return item

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(portable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        records.append({"source": relative, "published": str(dst.relative_to(target)).replace("\\", "/"),
                        "original_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                        "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                        "change": "Informational absolute paths changed to repository-relative paths; numeric results and model arrays unchanged."})

    python_root = source / "Content/TRIAD/RL/Python"
    for folder in (Path("triad_rl"), Path("tests")):
        for item in sorted((python_root / folder).glob("*.py")):
            copy(str(item.relative_to(source)).replace("\\", "/"), "Python/" + item.relative_to(python_root).as_posix())
    for name in ("train_blue_placement.py", "train_self_play.py", "evaluate_checkpoint.py", "benchmark_checkpoint.py", "pyproject.toml"):
        copy("Content/TRIAD/RL/Python/" + name, "Python/" + name)
    copy("Content/TRIAD/RL/DefaultTrainingConfig.json", "DefaultTrainingConfig.json")

    native = "Plugins/TRIADSensorFusion/Source/TRIADSensorFusion/"
    for name in ("TRIADRLTrainingTypes.h", "TRIADRLTrainingModel.h", "TRIADAdversarialTrainingManager.h", "TRIADSwarmControllerComponent.h", "TRIADProtectedZoneComponent.h"):
        copy(native + "Public/" + name, "NativeReference/Public/" + name)
    for name in ("TRIADRLTrainingModel.cpp", "TRIADAdversarialTrainingManager.cpp", "TRIADSwarmControllerComponent.cpp", "TRIADProtectedZoneComponent.cpp", "Tests/TRIADRLTrainingModelTests.cpp"):
        copy(native + "Private/" + name, "NativeReference/Private/" + name)
    for suffix in ("h", "cpp"):
        name = "TRIADRLAssetBootstrapCommandlet." + suffix
        copy("Plugins/TRIADSensorFusion/Source/TRIADSensorFusionEditor/Private/" + name, "NativeReference/Editor/" + name)

    run = "Saved/TRIAD/RLFundamentals/DynamicTraining/"
    copy(run + "pilot/episode_000210_final/arrays.npz", "Checkpoints/toy-210/arrays.npz")
    metadata(run + "pilot/episode_000210_final/checkpoint.json", "Checkpoints/toy-210/checkpoint.json")
    copy(run + "pilot/metrics.jsonl", "Results/training.jsonl")
    copy(run + "native-dynamic-oracle.json", "Results/native-oracle.json")
    for folder, name in (("eval-initial-v2", "initial-stochastic"), ("eval-trained-210-stochastic", "trained-stochastic"), ("eval-trained-210-deterministic", "trained-deterministic")):
        metadata(run + folder + "/evaluation.json", "Results/" + name + ".json")

    manifest = {"schema": "istana.blue_rl_export.v1", "source_training_date": "2026-09-10",
                "note": "Selected TRIAD source and measured synthetic-arena results. NativeReference is not compiled by IstanaOpen.",
                "files": records}
    (bundle / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Exported {len(records)} selected files to {bundle}")


if __name__ == "__main__":
    main()
