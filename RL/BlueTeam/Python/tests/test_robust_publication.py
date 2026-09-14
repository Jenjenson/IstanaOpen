"""Exact-byte validation-only publication, with real small-run fixtures."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("publish_robust_rl", REPO_ROOT / "Tools" / "publish_robust_rl.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)

from demo_robust import run_demo
from probe_stop_exploration import run_probe
from select_robust import PROFILES, run_selection
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.train_robust import run_training, validation_seed_ranges


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hashes(path):
    return {file.relative_to(path).as_posix(): publisher.sha256(file.read_bytes())
            for file in path.rglob("*") if file.is_file()}


TRAIN_OPTIONS = {"batch_size": 2, "learning_rate": .002, "entropy_coef": .015,
                 "gamma": 1., "validation_every": 4, "validation_episodes": 1,
                 "validation_run_seed": 990870}


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("robust-publication-inputs")
    runs = {seed: root / "training" / str(seed) for seed in (101, 102, 103)}
    for seed, path in runs.items():
        run_training(output=path, episodes=4, seed=seed, **TRAIN_OPTIONS,
                     initial_checkpoint=publisher.BLUE_ROOT / "Checkpoints" / "adaptive-v1",
                     initial_selection_report=publisher.BLUE_ROOT / "Results" / "adaptive-v1" / "common-validation-selection.json")
    protocol = {
        "schema": "triad.robust_experiment_protocol.v1", "experiment": "small publisher fixture",
        "candidate_training": {"run_seeds": list(runs), "episodes_per_run": 4,
                               "batch_size": 2, "learning_rate": .002, "entropy_coef": .015,
                               "gamma": 1., "validation_every": 4, "validation_episodes_per_profile": 1,
                               "training_profile": "mixed",
                               "training_seed_ranges": [{"start": seed * 1000000, "count": 4} for seed in runs],
                               "validation_seed_ranges": [{"profile": p, **r} for p, r in validation_seed_ranges(990870, 1).items()]},
        "candidate_selection": {"seed_ranges": [
            {"profile": p, "start": 1000990880000000 + index * 1000000, "count": 2}
            for index, p in enumerate(PROFILES)]},
    }
    write_json(root / "protocol.json", protocol)
    selection = run_selection({seed: path / "best" for seed, path in runs.items()},
                              protocol_path=root / "protocol.json", output=root / "selection", bootstrap_samples=10)
    selected = runs[selection["selected"]["seed"]] / "best"
    run_demo(checkpoint=selected, output=root / "demo.html", seed=1000990890000000, episodes=2, profile="capability")
    for name, checkpoint in (("v1", publisher.BLUE_ROOT / "Checkpoints" / "adaptive-v1"), ("v2", selected)):
        # Intentionally reuse trainer validation inputs: this public-only
        # diagnostic explicitly permits reuse and is not independent evidence.
        probe = run_probe(checkpoint, seed=1000990871000000, episodes=2, profile="stress")
        write_json(root / f"stop-{name}.json", probe)
    return root


def arguments(root, output, optional=True):
    options = {"runs": {seed: root / "training" / str(seed) for seed in (101, 102, 103)},
               "selection_dir": root / "selection", "protocol": root / "protocol.json", "output": output}
    if optional:
        options.update(demo=root / "demo.html", stop_probes=[root / "stop-v1.json", root / "stop-v2.json"])
    return options


def copied(experiment, tmp_path):
    destination = tmp_path / "inputs"
    shutil.copytree(experiment, destination)
    return destination


def test_publishes_exact_complete_runs_selected_aliases_and_full_reports_idempotently(experiment, tmp_path):
    output = tmp_path / "public"
    before_inputs = hashes(experiment)
    result = publisher.publish(**arguments(experiment, output))
    assert result["stage"] == "validation" and result["independent_final_test_evidence"] is False
    assert result["default_policy_changed"] is False
    assert "experimental candidate" in result["status"]
    assert result["archived_run_seeds"] == [101, 102, 103]
    for seed in (101, 102, 103):
        for name in publisher.RUN_FILES:
            assert (output / publisher.RESULTS / "training" / f"seed-{seed}" / name).read_bytes() == (
                experiment / "training" / str(seed) / name).read_bytes()
    for role, alias in (("best", "robust-v2-candidate"), ("initialized", "robust-v2-candidate-initialized")):
        for name in ("checkpoint.json", "arrays.npz"):
            assert (output / "Checkpoints" / alias / name).read_bytes() == (
                experiment / "training" / str(result["selected_seed"]) / role / name).read_bytes()
    for name in ("selection.json", *(f"common-validation-{p}.json.gz" for p in PROFILES)):
        assert (output / publisher.RESULTS / name).read_bytes() == (experiment / "selection" / name).read_bytes()
    for row in result["files"]:
        data = (output / row["path"]).read_bytes()
        assert publisher.sha256(data) == row["sha256"] and len(data) == row["bytes"]
    assert result["protocol_sha256"] == publisher.sha256((experiment / "protocol.json").read_bytes())
    assert all(publisher.sha256((REPO_ROOT / name).read_bytes()) == digest
               for name, digest in result["source_files_sha256"].items())
    assert len(result["stop_probes"]) == 2
    before_output = hashes(output)
    assert publisher.publish(**arguments(experiment, output)) == result
    assert hashes(output) == before_output and hashes(experiment) == before_inputs


def test_published_complete_training_directory_can_resume_exactly(experiment, tmp_path):
    output = tmp_path / "public"
    publisher.publish(**arguments(experiment, output, optional=False))
    published_run = output / publisher.RESULTS / "training" / "seed-101"
    reference = tmp_path / "resumed-original"
    shutil.copytree(experiment / "training" / "101", reference)
    for path in (published_run, reference):
        run_training(output=path, episodes=6, seed=101, resume=path / "last", **TRAIN_OPTIONS)
    left, right = AdaptivePolicy.load(published_run / "last"), AdaptivePolicy.load(reference / "last")
    assert left.weights_fingerprint() == right.weights_fingerprint()
    assert left.rng.bit_generator.state == right.rng.bit_generator.state
    assert left.update_count == right.update_count == 3
    for key in left.parameters:
        assert (left.adam_m[key] == right.adam_m[key]).all()
        assert (left.adam_v[key] == right.adam_v[key]).all()


@pytest.mark.parametrize("target", ["Results/robust-v2/protocol.json", "Results/robust-v2/artifact-manifest.json",
                                   "Checkpoints/robust-v2-candidate/checkpoint.json"])
def test_conflicts_reject_before_any_new_output_files(experiment, tmp_path, target):
    output = tmp_path / "public"
    existing = output / target
    existing.parent.mkdir(parents=True)
    existing.write_text("User-owned existing bytes")
    before = hashes(output)
    with pytest.raises(FileExistsError, match="will not be replaced"):
        publisher.publish(**arguments(experiment, output))
    assert hashes(output) == before


@pytest.mark.parametrize("mutation", ["protocol", "stage", "winner", "candidate_metadata", "compressed_sha", "compressed_bytes",
                                     "summary", "pairing", "report_path", "missing_run", "source"])
def test_corrupt_or_incomplete_evidence_fails_before_output(experiment, tmp_path, mutation):
    root = copied(experiment, tmp_path)
    path = root / "selection" / "selection.json"
    selection = read_json(path)
    kwargs = arguments(root, tmp_path / "public")
    if mutation == "protocol":
        selection["protocol_sha256"] = "0" * 64
    elif mutation == "stage":
        selection["stage"] = "test"
    elif mutation == "winner":
        selection["selected"]["balanced_mean_return"] += 1
    elif mutation == "candidate_metadata":
        selection["candidate_metadata"]["101"]["metadata"]["training_state"]["completed_episodes"] += 1
    elif mutation == "compressed_sha":
        selection["reports"]["normal"]["sha256"] = "0" * 64
    elif mutation == "compressed_bytes":
        selection["reports"]["normal"]["bytes"] += 1
    elif mutation in ("summary", "pairing"):
        entry = selection["reports"]["normal"]
        report_path = root / "selection" / entry["path"]
        report = json.loads(gzip.decompress(report_path.read_bytes()))
        if mutation == "summary":
            report["methods"]["candidate_101"]["summary"]["mean_return"] += 10
        else:
            report["methods"]["candidate_101"]["episodes"][0]["scenario_sha256"] = "0" * 64
        data = gzip.compress(json.dumps(report).encode(), mtime=0)
        report_path.write_bytes(data)
        entry.update(sha256=publisher.sha256(data), bytes=len(data))
    elif mutation == "report_path":
        selection["reports"]["normal"]["path"] = "../outside.json.gz"
    elif mutation == "missing_run":
        kwargs["runs"].pop(103)
    else:
        selection["implementation_sha256"]["triad_rl/robust_scenarios.py"] = "0" * 64
    write_json(path, selection)
    with pytest.raises(ValueError):
        publisher.publish(**kwargs)
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("value", [r"C:\Users\Alice\private", "D:/work/run", "/home/alice/run",
                                  "See /Users/alice/run for files", r"\\server\share\run", "file:///private/run",
                                  "ghp_" + "a" * 36, "-----BEGIN PRIVATE KEY-----", {"api_key": "do-not-publish"}])
def test_workstation_paths_and_secrets_rejected_without_echoing_them(value):
    with pytest.raises(ValueError) as caught:
        publisher.privacy_check({"nested": [value]})
    assert "Alice" not in str(caught.value) and "do-not-publish" not in str(caught.value)


def test_public_web_namespace_and_relative_source_paths_are_not_windows_drives():
    publisher.privacy_check({"url": "http://www.w3.org/2000/svg", "reference": "https://example.org/path",
                             "source": "triad_rl/train_robust.py", "note": "No physical controls were sent."})


@pytest.mark.parametrize("where", ["config", "jsonl", "gzip", "demo"])
def test_privacy_scans_nested_json_jsonl_decoded_gzip_and_replay(experiment, tmp_path, where):
    root = copied(experiment, tmp_path)
    secret = "do not publish D:/private/evidence"
    if where == "config":
        path = root / "training" / "101" / "config.json"
        data = read_json(path)
        data["private_note"] = secret
        write_json(path, data)
    elif where == "jsonl":
        path = root / "training" / "101" / "training.jsonl"
        path.write_text(path.read_text() + json.dumps({"private_note": secret}) + "\n")
    elif where == "gzip":
        selection_path = root / "selection" / "selection.json"
        selection = read_json(selection_path)
        entry = selection["reports"]["normal"]
        path = root / "selection" / entry["path"]
        data = json.loads(gzip.decompress(path.read_bytes()))
        data["private_note"] = secret
        compressed = gzip.compress(json.dumps(data).encode(), mtime=0)
        path.write_bytes(compressed)
        entry.update(sha256=publisher.sha256(compressed), bytes=len(compressed))
        write_json(selection_path, selection)
    else:
        path = root / "demo.html"
        path.write_text(path.read_text(encoding="utf-8") + f"<!-- {secret} -->", encoding="utf-8")
    with pytest.raises(ValueError, match="path|secret|portable"):
        publisher.publish(**arguments(root, tmp_path / "public"))
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("mutation", ["weights", "rng", "source", "groups", "pairing", "private"])
def test_probe_integrity_and_paired_public_inputs_checked(experiment, tmp_path, mutation):
    root = copied(experiment, tmp_path)
    path = root / "stop-v2.json"
    probe = read_json(path)
    if mutation == "weights":
        probe["weights_sha256_after"] = "0" * 64
    elif mutation == "rng":
        probe["policy_rng_sha256_after"] = "0" * 64
    elif mutation == "source":
        probe["source_sha256_before"]["probe_stop_exploration.py"] = "0" * 64
    elif mutation == "groups":
        probe["groups"]["all"]["count"] += 1
    elif mutation == "pairing":
        probe["records"][0]["public_input_sha256"] = "0" * 64
    else:
        probe["private_truth_accessed"] = True
    write_json(path, probe)
    with pytest.raises(ValueError):
        publisher.publish(**arguments(root, tmp_path / "public"))
    assert not (tmp_path / "public").exists()


def test_source_drift_during_preflight_does_not_publish(experiment, tmp_path, monkeypatch):
    original = publisher._source_files
    calls = 0
    def drifting(additional=()):
        nonlocal calls
        calls += 1
        result = original(additional)
        if calls > 1:
            result["Tools/publish_robust_rl.py"] = "0" * 64
        return result
    monkeypatch.setattr(publisher, "_source_files", drifting)
    with pytest.raises(ValueError, match="changed during"):
        publisher.publish(**arguments(experiment, tmp_path / "public"))
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("data", ['{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}'])
def test_strict_json_rejects_nonfinite_overflow_and_duplicate_keys(data):
    with pytest.raises(ValueError):
        publisher.strict_json(data)
