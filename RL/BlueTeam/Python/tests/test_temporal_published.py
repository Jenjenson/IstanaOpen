"""Static published temporal evidence checks; never regenerate or replay runs."""
import gzip
import hashlib
import io
from pathlib import Path

import evaluate_temporal as evaluator

BLUE = Path(__file__).resolve().parents[2]
RESULTS = BLUE / "Results/temporal-v6-pilot"
PREFIX = "Results/temporal-v6-pilot/"
SEEDS = (406, 407, 408)
PROTOCOL_SHA = "a35865fe99306013c3ce9c6d4ebc176af057144c2f9fcaecfb8d597060af6bea"
MANIFEST_SHA = "529c7ce21238003fcaae0eb8aa73c44d1bb43f696446d2a7987c9e4dc49a1031"


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(1024 * 1024): digest.update(data)
    return digest.hexdigest()


def read(path):
    return evaluator._strict_json(path.read_bytes())


def test_published_temporal_bundle_exact_bytes_ranges_pairing_and_aggregate(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Published regression must not sample, infer, replay, train or recursively validate")
    for owner, names in ((evaluator, ("make_case", "run_evaluation", "prepare_evaluation", "verify_completed_evaluation", "evaluate_robust_methods")),
                         (evaluator.trainer, ("run_training", "validate_run", "load_published_lineage")),
                         (evaluator.AdaptivePlacementEnv, ("__init__", "reset", "step"))):
        for name in names: monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr(evaluator, "load_published_lineage", forbidden)
    manifest_path = RESULTS / "artifact-manifest.json"
    assert sha_file(manifest_path) == MANIFEST_SHA
    manifest, protocol = read(manifest_path), read(RESULTS / "protocol.json")
    assert manifest["schema"] == "triad.temporal_pilot_publication.v1" and manifest["stage"] == "validation"
    assert manifest["predeclared_source_commit"] == "e2d4478c9eb5352640b317c4507a85667bce53c0"
    assert manifest["default_policy_changed"] is False and manifest["final_test_accessed"] is False
    assert manifest["manifest_self_hash_excluded"] is True
    assert sha_file(RESULTS / "protocol.json") == manifest["protocol_sha256"] == PROTOCOL_SHA
    training = {str(seed): {"start": seed * 1_000_000, "count": 512} for seed in SEEDS}
    validation = {profile: {"start": 10**15 + (995400 + index) * 1_000_000, "count": 200} for index, profile in enumerate(evaluator.PROFILES)}
    reserved = {profile: {"start": 2 * 10**15 + (500 + index) * 1_000_000, "count": 200} for index, profile in enumerate(evaluator.PROFILES)}
    assert evaluator._same(protocol["training"], {**evaluator.trainer.FIXED_TRAINING, "scenario_seed_ranges": training})
    assert evaluator._same(protocol["training"]["policy_seeds"], list(SEEDS))
    assert evaluator._same(manifest["seed_provenance"], {"training": training, "validation": validation, "reserved_final_tests_unopened": reserved})
    assert evaluator._same(protocol["validation"]["scenario_seed_ranges"], validation)
    assert evaluator._same(protocol["reserved_final_tests_unopened"], reserved)
    expected = {PREFIX + "protocol.json", *(PREFIX + "evaluation/" + name for name in evaluator.EVALUATION_FILES),
                *(PREFIX + f"training/seed-{seed}/" + name for seed in SEEDS for name in evaluator.trainer.RUN_FILES)}
    assert len(manifest["files"]) == len(expected) == 36
    assert {row["path"] for row in manifest["files"]} == expected
    for row in manifest["files"]:
        path = BLUE / row["path"]
        assert path.resolve().is_relative_to(RESULTS.resolve())
        assert type(row["bytes"]) is int and path.stat().st_size == row["bytes"]
        assert sha_file(path) == row["sha256"]
    for field in ("training_implementation_sha256", "evaluation_implementation_sha256"):
        assert evaluator._same(manifest[field], protocol[field])
        for name, digest in manifest[field].items(): assert sha_file(BLUE / "Python" / name) == digest
    for name, digest in manifest["archive_implementation_sha256"].items(): assert sha_file(BLUE.parents[1] / name) == digest
    inputs, aggregate = read(RESULTS / "evaluation/evaluation-inputs.json"), read(RESULTS / "evaluation/aggregate.json")
    assert inputs["protocol_sha256"] == aggregate["protocol_sha256"] == PROTOCOL_SHA
    assert aggregate["inputs_sha256"] == evaluator.canonical_hash(inputs)
    for seed in SEEDS:
        directory = RESULTS / f"training/seed-{seed}"
        summary, endpoint = read(directory / "summary.json"), read(directory / "last/checkpoint.json")
        assert type(summary["completed_episodes"]) is int and summary["completed_episodes"] == 512
        assert summary["optimizer_updates"] == endpoint["update_count"] == endpoint["actor_update_count"] == endpoint["critic_update_count"] == 32
        assert summary["policy_seed"] == endpoint["training_state"]["policy_seed"] == seed
        assert summary["last_weights_sha256"] == endpoint["weights_sha256"] == inputs["weights_sha256"][f"temporal_{seed}"]
        assert sha_file(directory / "protocol.json") == PROTOCOL_SHA
        assert summary["last_files_sha256"] == {name: sha_file(directory / "last" / name) for name in ("checkpoint.json", "arrays.npz")}
    prepared = {"inputs": inputs, "protocol": protocol, "weights": inputs["weights_sha256"]}
    reports, total = {}, 0
    for profile in evaluator.PROFILES:
        path = RESULTS / f"evaluation/validation-{profile}.json.gz"
        with gzip.GzipFile(fileobj=io.BytesIO(path.read_bytes())) as handle: raw = handle.read(evaluator.MAX_BYTES + 1)
        assert len(raw) <= evaluator.MAX_BYTES
        report = evaluator._strict_json(raw)
        receipt = read(RESULTS / f"evaluation/validation-{profile}.receipt.json")
        assert evaluator._same(receipt, {"profile": profile, "bytes": path.stat().st_size, "sha256": sha_file(path), "inputs_sha256": evaluator.canonical_hash(inputs)})
        evaluator.validate_report(report, profile, prepared)  # Pure stored-row/hash/summary arithmetic only.
        assert set(report["methods"]) == set(evaluator.method_names(protocol)) and len(report["methods"]) == 9
        assert all(len(method["episodes"]) == 200 for method in report["methods"].values())
        total += sum(len(method["episodes"]) for method in report["methods"].values())
        reports[profile] = report
    assert total == 5400 and sum(row["count"] for row in validation.values()) == 600
    recomputed = evaluator.aggregate_reports(reports, protocol)  # No inference or optimizer reconstruction.
    assert evaluator.derived_same({key: aggregate[key] for key in recomputed}, recomputed)
    assert evaluator._same(aggregate["scale_gate"], manifest["scale_gate"])
    assert aggregate["final_test_accessed"] is False and aggregate["scale_gate"]["not_policy_promotion"] is True
