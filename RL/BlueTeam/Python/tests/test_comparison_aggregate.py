"""Stored held-out panels stay separate from illustrative episode comparisons."""
from copy import deepcopy
import math
from pathlib import Path
import shutil
import subprocess
import threading

import pytest

from native_comparison import NativeComparisons, held_out_summary
from simulation_console import ConsoleState, make_server
from test_console_comparison import request
from triad_rl.trained_models import COMPARISON_SCHEMA, TrainedModelRegistry


def panel():
    return {"caseCount": 3, "meanWarningSeconds": 12., "baselineMeanWarningSeconds": 10.,
            "deltaSeconds": 2., "pairedDeltasSeconds": [2., 1., 3.], "seeds": [101, 102, 103],
            "cases": [{"seed": seed, "mean_drone_warning_s": warning}
                      for seed, warning in zip([101, 102, 103], [10., 12., 14.])]}


def test_complete_paired_panel_uses_scenario_standard_error_not_individual_drone_count():
    summary = held_out_summary(panel())
    assert summary["caseCount"] == summary["pairedCaseCount"] == 3
    assert summary["deltaSeconds"] == 2.
    interval = summary["interval95"]
    assert interval["method"] == "normal_approximation"
    assert interval["standardErrorSeconds"] == pytest.approx(1 / math.sqrt(3))
    assert interval["lowerSeconds"] == pytest.approx(2 - 1.96 / math.sqrt(3))
    assert interval["upperSeconds"] == pytest.approx(2 + 1.96 / math.sqrt(3))
    assert "cases" not in summary and "pairedDeltasSeconds" not in summary


def test_zero_warning_and_keep_policy_are_legitimate_results():
    evaluation = {"caseCount": 10, "meanWarningSeconds": 0., "baselineMeanWarningSeconds": 0.,
                  "deltaSeconds": 0., "pairedDeltasSeconds": [0.] * 10}
    summary = held_out_summary(evaluation)
    assert summary["meanWarningSeconds"] == summary["deltaSeconds"] == 0.
    assert summary["interval95"]["lowerSeconds"] == summary["interval95"]["upperSeconds"] == 0.


def test_one_case_or_legacy_means_cannot_invent_uncertainty():
    evaluation = {"caseCount": 1, "meanWarningSeconds": 5., "baselineMeanWarningSeconds": 5.,
                  "deltaSeconds": 0., "pairedDeltasSeconds": [0.]}
    assert held_out_summary(evaluation)["interval95"] is None
    del evaluation["pairedDeltasSeconds"]
    evaluation["caseCount"] = 10
    summary = held_out_summary(evaluation)
    assert summary["caseCount"] == 10 and summary["pairedCaseCount"] == 0
    assert summary["interval95"] is None


@pytest.mark.parametrize("change", [
    {"caseCount": 4}, {"caseCount": 0}, {"caseCount": True}, {"caseCount": 3.5},
    {"meanWarningSeconds": float("nan")}, {"baselineMeanWarningSeconds": float("inf")},
    {"deltaSeconds": 99}, {"meanWarningSeconds": "12"}, {"pairedDeltasSeconds": [2.]},
    {"pairedDeltasSeconds": [2., float("nan"), 3.]}, {"pairedDeltasSeconds": [2., 2., 5.]},
    {"pairedDeltasSeconds": None}, {"pairedDeltasSeconds": [True, 2., 3.]},
    {"seeds": [101, 101, 103]}, {"seeds": [101, 102]}, {"cases": []},
    {"cases": [{"seed": seed, "mean_drone_warning_s": 2.} for seed in [101, 102, 103]]},
])
def test_inconsistent_or_partial_panel_is_not_displayed(change):
    assert held_out_summary({**panel(), **change}) is None


@pytest.mark.parametrize("evaluation", [None, {}, [], {"meanWarningSeconds": 99}])
def test_absent_aggregate_never_uses_an_episode_or_training_score(evaluation):
    assert held_out_summary(evaluation) is None


def registered_store(tmp_path, evaluation):
    registry = TrainedModelRegistry(tmp_path / "models")
    episode = deepcopy(NativeComparisons()._workbench_bundle()["episodes"][0])
    episode["rl"]["placements"] = deepcopy(episode["baseline"]["placements"])
    episode["rl"]["metrics"]["cost"] = episode["baseline"]["metrics"]["cost"]
    identifier = registry.identifier_for("Held-out zero gain")
    episode.update(schema=COMPARISON_SCHEMA, id=identifier, policy=identifier,
                   policyLabel="Held-out zero gain", case=1, label="Illustrative replay")
    observed = deepcopy(episode)
    observed.update(id=f"observed-{identifier}", policy=f"observed-{identifier}")
    observed["audit"].update(exactTrainingEpisode=7, exactTrainingReplay=True)
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    registry.register(name="Held-out zero gain", policy_path=policy, comparison_episode=episode,
                      observed_episode=observed,
                      metadata={"bestEpisode": 0, "bestWarningSeconds": 99.,
                                "bestObservedEpisode": 7, "bestObservedWarningSeconds": 101.,
                                "sensorCount": 8, "sensorIds": ["thermal"],
                                "testEvaluation": evaluation})
    return NativeComparisons(registry=registry), identifier


def test_named_model_endpoint_exposes_test_panel_but_observed_and_historical_do_not(tmp_path):
    evaluation = {"caseCount": 64, "meanWarningSeconds": 27., "baselineMeanWarningSeconds": 27.,
                  "deltaSeconds": 0., "pairedDeltasSeconds": [0.] * 64}
    store, identifier = registered_store(tmp_path, evaluation)
    state = ConsoleState()
    state.view = {"existing_live_episode": True}
    server = make_server(0, state=state, comparisons=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        code, result = request(server, "/api/comparison/run", {"episodeId": identifier})
        assert code == 200 and result["heldOutSummary"]["caseCount"] == 64
        assert result["heldOutSummary"]["meanWarningSeconds"] == 27.
        assert result["heldOutSummary"]["deltaSeconds"] == 0.
        assert result["metrics"]["rl"]["mean_warning_s"] != 27.
        for selection in (f"observed-{identifier}", "native-406-1"):
            code, result = request(server, "/api/comparison/run", {"episodeId": selection})
            assert code == 200 and "heldOutSummary" not in result
        assert state.view == {"existing_live_episode": True}
        assert not state.status()["connected"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("evaluation", [None, {**panel(), "caseCount": 10}])
def test_missing_or_inconsistent_test_panel_keeps_legacy_replay_available(tmp_path, evaluation):
    store, identifier = registered_store(tmp_path, evaluation)
    result = store.get(identifier)
    assert result["trainedModel"] and "heldOutSummary" not in result


def test_comparison_summary_ui_behavior():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    subprocess.run([node, str(Path(__file__).with_name("console_comparison_summary.test.cjs"))],
                   check=True, capture_output=True, text=True, timeout=30)
