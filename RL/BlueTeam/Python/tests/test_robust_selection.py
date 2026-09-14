"""Common candidate selection is validation-only and follows the declared score."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from select_robust import PROFILES, run_selection, select_from_reports
from triad_rl.adaptive_inputs import FEATURE_NAMES
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.adaptive_evaluation import canonical_hash, summarize
from triad_rl.train_robust import run_training, validation_seed_ranges
from select_robust import BLUE_ROOT


def evidence():
    candidates = {101: {"weights_sha256": "a"}, 102: {"weights_sha256": "b"}}
    # Candidate101 wins normal but loses the equally weighted three-profile score.
    returns = {101: [8., -9., -3.], 102: [7., -7., -2.]}
    reports = {}
    for index, profile in enumerate(PROFILES):
        reports[profile] = {
            "stage": "validation", "profile": profile, "schema": "triad.robust_evaluation.v1",
            "protocol": {"independent_final_test_evidence": False},
            "implementation_sha256": {"source": "same"}, "methods": {},
            "seed_provenance": {"evaluation": {"start": 1000990700000000 + index * 1000000, "count": 1}},
        }
        for seed, candidate in candidates.items():
            scenario = {"fixture": profile}
            records = [{"weights_sha256_before": candidate["weights_sha256"],
                        "weights_sha256_after": candidate["weights_sha256"],
                        "seed": reports[profile]["seed_provenance"]["evaluation"]["start"],
                        "scenario": scenario, "scenario_sha256": canonical_hash(scenario),
                        "metrics": {"return": returns[seed][index]}}]
            reports[profile]["methods"][f"candidate_{seed}"] = {
                "summary": summarize(records), "episodes": records,
            }
    return reports, candidates


def test_balanced_selection_not_normal_only_and_inputs_unchanged():
    reports, candidates = evidence()
    before = deepcopy(reports)
    selected, scores = select_from_reports(reports, candidates)
    assert selected["seed"] == 102
    assert selected["balanced_mean_return"] == pytest.approx(-2 / 3)
    assert len(scores) == 2 and reports == before


@pytest.mark.parametrize("mutation", ["test", "missing", "drift", "weights", "empty", "summary", "unpaired"])
def test_rejects_wrong_stage_incomplete_or_mixed_evidence(mutation):
    reports, candidates = evidence()
    if mutation == "test":
        reports["stress"]["stage"] = "test"
    elif mutation == "missing":
        reports.pop("capability")
    elif mutation == "drift":
        reports["stress"]["implementation_sha256"] = {"source": "other"}
    elif mutation == "weights":
        reports["stress"]["methods"]["candidate_101"]["episodes"][0]["weights_sha256_after"] = "changed"
    elif mutation == "empty":
        reports["stress"]["methods"]["candidate_101"]["episodes"] = []
    elif mutation == "summary":
        reports["stress"]["methods"]["candidate_101"]["summary"]["mean_return"] = 123
    else:
        reports["stress"]["methods"]["candidate_101"]["episodes"][0]["scenario_sha256"] = "unpaired"
    with pytest.raises(ValueError):
        select_from_reports(reports, candidates)


def test_equal_scores_tie_break_on_seed():
    reports, candidates = evidence()
    for report in reports.values():
        other = report["methods"]["candidate_102"]
        other["episodes"][0]["metrics"] = deepcopy(report["methods"]["candidate_101"]["episodes"][0]["metrics"])
        other["summary"] = summarize(other["episodes"])
    selected, _ = select_from_reports(reports, dict(reversed(list(candidates.items()))))
    assert selected["seed"] == 101


def test_real_selection_writes_complete_paired_validation_evidence(tmp_path):
    candidate_paths = {}
    for seed in (101, 102):
        run = tmp_path / str(seed)
        run_training(output=run, episodes=4, batch_size=2, seed=seed,
                     learning_rate=.002, entropy_coef=.015, gamma=1.,
                     validation_every=4, validation_episodes=1, validation_run_seed=990850,
                     initial_checkpoint=BLUE_ROOT / "Checkpoints" / "adaptive-v1")
        candidate_paths[seed] = run / "best"
    protocol = {
        "schema": "triad.robust_experiment_protocol.v1",
        "candidate_training": {"run_seeds": [101, 102], "episodes_per_run": 4,
                               "batch_size": 2, "learning_rate": .002, "entropy_coef": .015,
                               "gamma": 1., "validation_every": 4, "validation_episodes_per_profile": 1,
                               "training_profile": "mixed",
                               "training_seed_ranges": [{"start": seed * 1000000, "count": 4} for seed in (101, 102)],
                               "validation_seed_ranges": [{"profile": p, **r} for p, r in validation_seed_ranges(990850, 1).items()]},
        "candidate_selection": {"seed_ranges": [
            {"profile": profile, "start": 1000990800000000 + index * 1000000, "count": 2}
            for index, profile in enumerate(PROFILES)]},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "selection"
    summary = run_selection(candidate_paths, protocol_path=protocol_path, output=output,
                            bootstrap_samples=10)
    assert summary["selected"]["seed"] in candidate_paths
    assert summary["final_test_accessed"] is False
    assert len(summary["seed_provenance"]["selection_validation"]) == 3
    for profile, artifact in summary["reports"].items():
        compressed = (output / artifact["path"]).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == artifact["sha256"]
        report = json.loads(gzip.decompress(compressed))
        assert report["stage"] == "validation" and report["profile"] == profile
        assert set(report["methods"]) == {"candidate_101", "candidate_102", "adaptive_v1", "greedy_public"}
        assert all(len(method["episodes"]) == 2 for method in report["methods"].values())
    with pytest.raises(ValueError, match="empty directory"):
        run_selection(candidate_paths, protocol_path=protocol_path, output=output)
