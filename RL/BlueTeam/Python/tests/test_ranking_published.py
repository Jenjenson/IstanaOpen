"""Static published-byte/replay regression: no project imports or execution.

Only read recorded artifacts. Never sample scenarios, infer policies, reconstruct
optimizer updates, or regenerate the separately derived offline HTML.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re

import pytest

BLUE = Path(__file__).resolve().parents[2]
PYTHON = BLUE / "Python"
RESULTS = BLUE / "Results/ranking-v5-pilot"
PREFIX = "Results/ranking-v5-pilot/"
SEEDS = (403, 404, 405)
PROFILES = ("normal", "stress", "capability")
MANIFEST_SHA = "a9d975e38dbdf8a5095c52e0767e2142ac5616dcd09d338c85247747517dd058"
DEMO_SHA = "bd5ac8ac194cdf8154859cfb2a7d7fcb7497822a172df717be00705ac66dce34"
RUN_FILES = ("config.json", "protocol.json", "training.jsonl", "comparisons.jsonl.gz",
             "summary.json", "initialized/checkpoint.json", "initialized/arrays.npz",
             "last/checkpoint.json", "last/arrays.npz")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    # Also distinguishes JSON booleans/integers/floats for exact comparisons.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture(scope="module")
def published():
    manifest_bytes = (RESULTS / "artifact-manifest.json").read_bytes()
    html = (RESULTS / "demo.html").read_bytes()
    assert digest(manifest_bytes) == MANIFEST_SHA
    assert digest(html) == DEMO_SHA
    payloads = re.findall(r'<script[^>]+id="replay-data"[^>]*>(.*?)</script>', html.decode(), re.S)
    assert len(payloads) == 1
    return (json.loads(manifest_bytes), json.loads(payloads[0]),
            json.loads((RESULTS / "protocol.json").read_bytes()),
            json.loads((RESULTS / "evaluation/evaluation-inputs.json").read_bytes()))


def test_published_manifest_exact_inventory_sources_and_no_promotion(published):
    manifest, _, protocol, inputs = published
    expected = {PREFIX + "protocol.json", PREFIX + "evaluation/evaluation-inputs.json",
                PREFIX + "evaluation/aggregate.json"}
    expected.update(PREFIX + f"evaluation/validation-{profile}.{suffix}"
                    for profile in PROFILES for suffix in ("json.gz", "receipt.json"))
    expected.update(PREFIX + f"training/seed-{seed}/{name}" for seed in SEEDS for name in RUN_FILES)
    assert manifest["schema"] == "triad.ranking_pilot_publication.v1"
    assert manifest["experiment"] == "ranking-v5-pilot" and manifest["stage"] == "validation"
    assert len(manifest["files"]) == len(expected) == 36
    assert {row["path"] for row in manifest["files"]} == expected
    for row in manifest["files"]:
        relative = PurePosixPath(row["path"])
        assert not relative.is_absolute() and ".." not in relative.parts and "\\" not in row["path"]
        path = (BLUE / row["path"]).resolve()
        assert path.is_relative_to(RESULTS.resolve())
        data = path.read_bytes()
        assert type(row["bytes"]) is int and len(data) == row["bytes"]
        assert digest(data) == row["sha256"]
    for field, root in (("archive_implementation_sha256", BLUE.parents[1]),
                        ("training_implementation_sha256", PYTHON),
                        ("evaluation_implementation_sha256", PYTHON)):
        for name, expected_hash in manifest[field].items():
            assert digest((root / name).read_bytes()) == expected_hash
    assert manifest["default_policy_changed"] is False and manifest["final_test_accessed"] is False
    assert manifest["manifest_self_hash_excluded"] is True
    assert canonical(manifest["policy_seeds"]) == canonical(list(SEEDS))
    assert manifest["protocol_sha256"] == digest((RESULTS / "protocol.json").read_bytes())
    assert inputs["protocol_sha256"] == manifest["protocol_sha256"]
    aggregate = json.loads((RESULTS / "evaluation/aggregate.json").read_bytes())
    assert canonical(aggregate["scale_gate"]) == canonical(manifest["scale_gate"])
    assert aggregate["final_test_accessed"] is False
    assert aggregate["scale_gate"]["passed"] is False
    assert aggregate["scale_gate"]["not_policy_promotion"] is True
    assert aggregate["scale_gate"]["decision"] == "do_not_scale_from_this_experiment"
    assert canonical(aggregate["inputs_sha256"]) == canonical(digest(canonical(inputs)))
    assert canonical(protocol["training_implementation_sha256"]) == canonical(manifest["training_implementation_sha256"])
    assert canonical(protocol["evaluation_implementation_sha256"]) == canonical(manifest["evaluation_implementation_sha256"])


def test_published_demo_has_all_fixed_endpoints_and_unselected_first_cases(published):
    _, demo, _, inputs = published
    assert demo["schema"] == "triad.adaptive_demo.v1"
    assert "OFFLINE, NOT LIVE" in demo["checkpoint"]
    assert "no promotion or physical commands" in demo["checkpoint"]
    assert len(demo["replays"]) == 18
    selections = [(row["ranker_seed"], row["profile"], row["case_index"]) for row in demo["replays"]]
    assert canonical(selections) == canonical([(seed, profile, case) for profile in PROFILES
                                               for seed in SEEDS for case in (1, 2)])
    weights = {str(seed): inputs["weights_sha256"][f"ranker_{seed}"] for seed in SEEDS}
    assert demo["weights_sha256"] == digest(canonical(weights))


@pytest.mark.parametrize("profile", PROFILES)
def test_published_demo_matches_scored_actions_layouts_metrics_and_audit(published, profile):
    manifest, demo, protocol, inputs = published
    compressed = (RESULTS / f"evaluation/validation-{profile}.json.gz").read_bytes()
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as handle:
        expanded = handle.read(128 * 1024 * 1024 + 1)
    assert len(expanded) <= 128 * 1024 * 1024
    report = json.loads(expanded)
    receipt = json.loads((RESULTS / f"evaluation/validation-{profile}.receipt.json").read_bytes())
    assert canonical(receipt) == canonical({"bytes": len(compressed), "sha256": digest(compressed),
        "profile": profile, "inputs_sha256": digest(canonical(inputs))})
    assert report["schema"] == "triad.ranking_evaluation.v1"
    assert report["profile"] == profile and report["stage"] == "validation"
    assert report["training_performed"] is False
    assert report["protocol"]["independent_final_test_evidence"] is False
    assert report["ranking_inputs_sha256"] == receipt["inputs_sha256"]
    start = protocol["validation"]["scenario_seed_ranges"][profile]["start"]
    by_path = {row["path"]: row["sha256"] for row in manifest["files"]}
    for replay in (row for row in demo["replays"] if row["profile"] == profile):
        seed, case = replay["ranker_seed"], replay["case_index"]
        scored = report["methods"][f"ranker_{seed}"]["episodes"][case - 1]
        assert type(replay["seed"]) is int and replay["seed"] == scored["seed"] == start + case - 1
        assert replay["schema"] == "triad.ranking_replay.v1" and replay["release"] == "ranking-v5-pilot"
        assert replay["stage"] == "validation" and replay["requested_profile"] == profile
        assert replay["split"] == f"EXPERIMENTAL ranker {seed} / {profile} / case {case} of 2 / offline replay"
        assert canonical([row["action_index"] for row in replay["decisions"]]) == canonical(scored["actions"])
        assert canonical(replay["placements"]) == canonical(scored["placements"])
        for key in ("return", "reward", "cost", "total_cost", "success", "detected_fraction",
                    "confirmed_fraction", "breached_fraction", "coverage", "early_detection", "invalid_actions"):
            assert canonical(replay["metrics"][key]) == canonical(scored["metrics"][key])
        assert len(replay["placements"]) == scored["metrics"]["sensors_placed"]
        for key in ("outcome", "target_results", "reward_components"):
            assert canonical(replay["metrics"][key]) == canonical(scored[key])
        scenario = {**replay["scenario"], "evaluation_catalogue": replay["catalogue"],
                    "curriculum_metadata": replay["case_metadata"]}
        assert canonical(scenario) == canonical(scored["scenario"])
        audit = replay["audit"]
        assert audit["schema"] == "triad.ranking_replay_audit.v1"
        assert audit["policy_schema"] == "triad.greedy_residual_rank_policy.v1"
        assert audit["scored_scenario_sha256"] == scored["scenario_sha256"] == digest(canonical(scenario))
        assert audit["scenario_catalogue_metadata_sha256"] == digest(canonical(
            {key: replay[key] for key in ("scenario", "catalogue", "case_metadata")}))
        assert audit["evaluation_protocol_sha256"] == manifest["protocol_sha256"]
        for flag in ("recorded_simulator_replay", "scored_actions_metrics_verified", "experimental"):
            assert audit[flag] is True
        for flag in ("browser_inference", "training_performed", "independent_final_test_evidence",
                     "ground_truth_in_policy_observation", "default_policy_changed", "physical_commands", "live_feed"):
            assert audit[flag] is False
        assert canonical(audit["policy_state_before"]) == canonical(audit["policy_state_after"])
        assert re.fullmatch("[0-9a-f]{64}", audit["policy_state_before"]["rng_sha256"])
        assert (audit["policy_state_before"]["weights_sha256"] == scored["weights_sha256_before"]
                == scored["weights_sha256_after"] == inputs["weights_sha256"][f"ranker_{seed}"])
        checkpoint = {name: by_path[PREFIX + f"training/seed-{seed}/last/{name}"]
                      for name in ("checkpoint.json", "arrays.npz")}
        assert canonical(audit["checkpoint_files_sha256_before"]) == canonical(checkpoint)
        assert canonical(audit["checkpoint_files_sha256_after"]) == canonical(checkpoint)
        assert canonical(audit["implementation_sha256_before"]) == canonical(audit["implementation_sha256_after"])
        for source_map in (audit["implementation_sha256_before"], audit["ranking_implementation_sha256"]):
            for name, source_hash in source_map.items():
                assert digest((PYTHON / name).read_bytes()) == source_hash
