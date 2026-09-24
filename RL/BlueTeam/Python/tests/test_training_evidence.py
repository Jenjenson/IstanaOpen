"""Selection and recording semantics with controlled native-shaped episode evidence."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from test_training_workbench import native_context
from triad_rl.directional_inputs import apply_placement, build_observation
from triad_rl.istana_live import public_planning_inputs
from triad_rl.training_evidence import (
    evaluation_summary, layout_changes, layout_key, legal_layout_probes, rollout_entropy)
from triad_rl.training_workbench import TrainingManager, common_sense_start


VALIDATION = [2700000, 2700001]
TEST = [3800000, 3800001]


class Policy:
    """A controlled sequence of policies; optimizer behavior is tested separately."""
    def __init__(self, context, *, sensor_count=1, allowed_sensor_ids=None, **_):
        self.updates = 0
        self.sensor_count = sensor_count
        self.allowed_sensor_ids = allowed_sensor_ids
        self.placements = [{"profileId": "thermal", "siteId": 16,
                            "yawDeg": 0., "pitchDeg": 10.}]

    def initialize_from_placements(self, placements, **_):
        self.placements = deepcopy(placements)
        return {"trainable": True}

    def update(self, episodes):
        assert episodes
        self.updates += 1
        return {"update": self.updates}

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump({"updates": self.updates, "sensor_count": self.sensor_count,
                       "allowed_sensor_ids": self.allowed_sensor_ids}, stream)
        return "test-checkpoint"

    @classmethod
    def load(cls, path, context):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(context, sensor_count=data["sensor_count"],
                     allowed_sensor_ids=data["allowed_sensor_ids"])
        policy.updates = data["updates"]
        return policy


def evidence(seed, warning, *, context=None, placements=None):
    ctx = deepcopy(context if context is not None else native_context())
    placements = deepcopy(placements or [{"profileId": "thermal", "siteId": 16,
                                        "yawDeg": 0., "pitchDeg": 10.}])
    public = [{"sensor_id": row["profileId"], "sensor_index": 0,
               "position": ctx["publicSnapshot"]["sites"][row["siteId"]],
               "cost": 1., "yaw_deg": row["yawDeg"], "pitch_deg": row["pitchDeg"]}
              for row in placements]
    cost = float(len(placements))
    native_reward = 10. - cost / ctx["publicSnapshot"]["budget_total"]
    rows = [{"droneId": index, "firstDetectionSeconds": 10.,
             "firstConfirmationSeconds": 11., "zoneEntrySeconds": 10. + warning,
             "warningSeconds": warning} for index in range(5)]
    targets = [{"id": f"drone-{index}", "first_detection": 10., "first_confirmation": 11.,
                "time_to_zone": 10. + warning, "warning_time": warning,
                "timely_confirmed": True, "unresolved": False} for index in range(5)]
    return {"context": ctx, "seed": seed, "placements": placements, "public_placements": public,
        "sensor_count": len(placements), "run_id": f"controlled-{seed}", "steps": 1000,
        "trajectory_sha256": f"trajectory-for-seed-{seed}",
        "metrics": {"mean_drone_warning_s": float(warning), "team_warning_s": float(warning),
                    "first_detection_s": 10., "detected_fraction": 1., "targets": 5, "cost": cost},
        "native_metrics": {"mean_drone_warning_seconds_lower_bound": float(warning),
            "team_warning_seconds_lower_bound": float(warning), "detected_fraction": 1.,
            "confirmed_fraction": 1., "timely_fraction": 1., "breached_fraction": 0., "cost": cost},
        "reward": native_reward, "red_native_reward": -native_reward,
        "red_policy": "fixed_radius_random_sectors", "red_decision": {
            "formation": "randomized_eight_sector_fixed_radius", "radius_cm": 57000.,
            "angles_degrees": [5., 50., 95., 185., 275.],
            "sector_centers_degrees": [0., 45., 90., 180., 270.]},
        "elapsed_seconds": 60., "warning_evidence": rows, "target_results": targets,
        "frames": [{"time": 0., "completedSteps": 0, "threats": [], "detections": [], "tracks": []}]}


class Harness:
    def __init__(self, directory, validation_scores=None, stop_during_validation=False,
                 stop_during_test=False):
        self.context = native_context()
        self.validation_scores = validation_scores or {0: [5., 5.], 1: [20., 0.], 2: [11., 11.]}
        self.stop_during_validation = stop_during_validation
        self.stop_during_test = stop_during_test
        self.calls, self.clients = [], []
        harness = self

        class Client:
            def __init__(self, **_):
                self.closed = False
                harness.clients.append(self)
            def reset(self, _): return {}
            def get_blue_context(self): return deepcopy(harness.context)
            def close(self): self.closed = True

        self.manager = TrainingManager(8765, directory, client_factory=Client,
                                      policy_factory=Policy, episode_runner=self.episode)

    def episode(self, _client, policy, seed, **kwargs):
        generation = policy.updates if isinstance(policy, Policy) else None
        self.calls.append({"seed": seed, "generation": generation, **kwargs})
        if ((self.stop_during_validation and generation == 1 and seed == VALIDATION[0])
                or (self.stop_during_test and seed == TEST[0])):
            self.manager.stop()
            assert kwargs["should_stop"]()
            raise InterruptedError("Controlled stop inside a native evaluation action")
        placements = deepcopy(policy.placements)
        if generation is None:
            warning = 6.
        elif seed in VALIDATION:
            warning = self.validation_scores[generation][VALIDATION.index(seed)]
        elif seed in TEST:
            warning = 1.  # Worse unseen results must not retroactively change selection.
        else:
            warning = 40. + generation  # Lucky sampled scores never choose the checkpoint.
        if generation is not None:
            placements[0]["yawDeg"] = (generation * 45.) % 360
        run = evidence(seed, warning, context=self.context, placements=placements)
        run["action_seed"] = kwargs.get("action_seed")
        return run, []

    def start(self):
        self.manager.start({"name": "Evidence policy", "algorithm": "reinforce", "episodes": 4,
            "batchSize": 2, "sensorCount": 1, "seed": 31, "initialization": "untrained",
            "validationCases": 2, "headroomProbes": 0, "checkpointInterval": 2})
        self.manager.thread.join(timeout=10)
        assert not self.manager.thread.is_alive()
        assert self.manager.status()["phase"] in ("complete", "stopped"), self.manager.status()
        return Path(self.manager.status()["outputDirectory"])


def test_validation_panel_selects_mean_not_lucky_case_and_test_cannot_reselect(tmp_path):
    harness = Harness(tmp_path)
    output = harness.start()
    status = harness.manager.status()
    assert status["bestEpisode"] == 4
    assert status["bestWarningSeconds"] == 11.
    # Earlier checkpoint had the best individual validation case (20), but mean10 loses to11.
    logged = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert logged[1]["validationWarningSeconds"] == 10.
    assert logged[3]["validationWarningSeconds"] == 11.
    selection = json.loads((output / "evaluation-summary.json").read_text())
    assert selection["bestEpisode"] == 4 and selection["initial"]["meanWarningSeconds"] == 5.
    assert selection["validationSeeds"] == VALIDATION and selection["testSeeds"] == TEST
    assert set(VALIDATION).isdisjoint(TEST)
    test = json.loads((output / "test-evaluation.json").read_text())
    assert test["usedForSelection"] is False
    assert test["policy"]["meanWarningSeconds"] == 1.
    assert test["policy"]["deltaSeconds"] == -5.
    assert status["bestWarningSeconds"] == 11. and status["testEvaluation"]["meanWarningSeconds"] == 1.
    # Deployment really contains the checkpoint selected by the panel.
    model = harness.manager.registry.list()[0]
    checkpoint = json.loads(harness.manager.registry.policy_path(model["id"]).read_text())
    assert checkpoint["updates"] == 2
    assert model["bestEpisode"] == 4 and model["testEvaluation"]["caseCount"] == 2
    assert all(row["generation"] == 2 for row in harness.calls
               if row["seed"] in TEST and row["generation"] is not None)
    assert all(client.closed for client in harness.clients)


def test_initial_policy_is_retained_when_later_checkpoints_tie_or_regress(tmp_path):
    harness = Harness(tmp_path, validation_scores={0: [5., 5.], 1: [0., 0.], 2: [5., 5.]})
    output = harness.start()
    status = harness.manager.status()
    assert status["bestEpisode"] == 0 and status["bestWarningSeconds"] == 5.
    assert not list(output.glob("best-policy-*.json"))
    assert harness.manager.replay("best")["episode"] == 0
    model = harness.manager.registry.list()[0]
    selected = json.loads(harness.manager.registry.policy_path(model["id"]).read_text())
    assert selected["updates"] == 0
    assert json.loads((output / "final-policy.json").read_text())["updates"] == 2
    assert all(row["generation"] == 0 for row in harness.calls
               if row["seed"] in TEST and row["generation"] is not None)


def test_stop_inside_evaluation_retains_completed_episode_without_more_native_calls(tmp_path):
    harness = Harness(tmp_path, stop_during_validation=True)
    output = harness.start()
    status = harness.manager.status()
    assert status["phase"] == "stopped" and status["episode"] == 2
    assert status["bestEpisode"] == 0 and status["registeredModel"] is None
    assert len(harness.calls) == 7  # Four initial panel cases, two training, one interrupted evaluation.
    assert not any(row["seed"] in TEST for row in harness.calls)
    assert harness.manager.registry.list() == []
    logged = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert [row["episode"] for row in logged] == [1, 2]
    assert "validationWarningSeconds" not in logged[-1]
    assert len(list((output / "episodes").glob("episode-*.json"))) == 2
    assert harness.manager.replay("2")["episode"] == 2
    assert harness.manager.replay("initial")["episode"] == 0
    assert len(harness.manager.export()["history"]) == 2
    assert json.loads((output / "stopped-policy.json").read_text())["updates"] == 1
    stopped = json.loads((output / "stopped-summary.json").read_text())
    assert stopped["completedEpisodes"] == 2 and stopped["bestPolicy"] == "policy-0000.json"
    assert stopped["published"] is False and stopped["pendingBatchEpisodes"] == 0
    assert all(client.closed for client in harness.clients)


def test_stopped_run_can_explicitly_publish_its_retained_episode_zero_policy(tmp_path):
    harness = Harness(tmp_path, stop_during_validation=True)
    output = harness.start()
    harness.stop_during_validation = False
    calls_before = len(harness.calls)
    model = harness.manager.publish_saved_run(output)
    assert model["bestEpisode"] == 0 and model["stoppedEarly"] is True
    assert model["completedEpisodes"] == 2
    assert model["bestWarningSeconds"] == 5.
    selected = json.loads(harness.manager.registry.policy_path(model["id"]).read_text())
    assert selected["updates"] == 0
    assert len(harness.calls) == calls_before + 5  # Two matched test panels plus observed baseline.
    assert len(list((output / "episodes").glob("episode-*.json"))) == 2
    assert harness.manager.registry.observed_comparison(model["id"])["audit"]["exactTrainingReplay"]
    assert all(client.closed for client in harness.clients)


def test_legacy_recovery_selects_latest_best_checkpoint_numerically(tmp_path, monkeypatch):
    harness = Harness(tmp_path, stop_during_validation=True)
    output = harness.start()
    configuration = json.loads((output / "configuration.json").read_text())
    configuration["episodes"] = 10000
    (output / "configuration.json").write_text(json.dumps(configuration))
    (output / "evaluation-summary.json").unlink()
    row = json.loads((output / "training.jsonl").read_text().splitlines()[-1])
    rows = [{**row, "episode": episode, "validationWarningSeconds": float(episode)}
            for episode in (9996, 10000)]
    (output / "training.jsonl").write_text("\n".join(json.dumps(item) for item in rows))
    for episode in (9996, 10000):
        (output / f"best-policy-{episode:04d}.json").write_bytes(
            (output / "stopped-policy.json").read_bytes())
    monkeypatch.setattr(harness.manager, "_publish", lambda **kwargs: {
        "episode": kwargs["best_episode"], "path": kwargs["best_path"].name})
    selected = harness.manager.publish_saved_run(output)
    assert selected == {"episode": 10000, "path": "best-policy-10000.json"}


def test_stop_during_final_test_retains_training_without_finishing_publication(tmp_path):
    harness = Harness(tmp_path, stop_during_test=True)
    output = harness.start()
    status = harness.manager.status()
    assert status["phase"] == "stopped" and status["episode"] == 4
    assert status["bestEpisode"] == 4 and status["registeredModel"] is None
    # Twelve initial/training/validation calls, then the first interrupted test case.
    assert len(harness.calls) == 13
    assert [row["seed"] for row in harness.calls if row["seed"] in TEST] == [TEST[0]]
    assert not (output / "test-evaluation.json").exists()
    assert (output / "final-policy.json").is_file()
    assert len(harness.manager.export()["history"]) == 4
    assert harness.manager.registry.list() == []
    assert all(client.closed for client in harness.clients)


def test_layout_diversity_ignores_order_and_counts_changed_sensor_choices():
    rows = [{"profileId": "thermal", "siteId": 1, "yawDeg": 0., "pitchDeg": 10.},
            {"profileId": "thermal", "siteId": 2, "yawDeg": 90., "pitchDeg": 10.}]
    assert layout_key(rows) == layout_key(list(reversed(rows)))
    assert layout_changes(list(reversed(rows)), rows) == 0
    changed = deepcopy(rows)
    changed[0]["pitchDeg"] = 20.
    assert layout_changes(changed, rows) == 1
    changed[1]["siteId"] = 3
    assert layout_changes(changed, rows) == 2


def test_entropy_diagnostic_excludes_forced_stop_and_supports_both_record_formats():
    expected = {"entropy": pytest.approx(np.log(2)), "normalizedEntropy": pytest.approx(1.)}
    assert rollout_entropy([(0, [.5, .5, 0.]), (2, [0., 0., 1.])]) == expected
    assert rollout_entropy([{"probabilities": [.5, .5, 0.]}, {"probabilities": [0., 0., 1.]}]) == expected
    assert rollout_entropy([]) == {"entropy": None, "normalizedEntropy": None}


def test_probes_offer_legal_site_and_orientation_changes_under_same_sensor_limit():
    ctx = native_context()
    original = common_sense_start(ctx, "directional_balanced_8", count=3)["placements"]
    before = deepcopy((ctx, original))
    probes = legal_layout_probes(ctx, original, limit=4)
    assert len(probes) == 4 and len({layout_key(row) for row in probes}) == 4
    site_sets = lambda rows: {row["siteId"] for row in rows}
    assert any(site_sets(rows) != site_sets(original) for rows in probes)
    assert any(site_sets(rows) == site_sets(original) for rows in probes)
    for rows in probes:
        assert len(rows) == 3 and layout_changes(rows, original) == 1
        assert all(row["profileId"] == "thermal" for row in rows)
        state, catalogue, _ = public_planning_inputs(ctx)
        state["max_sites"] = 3
        for placement in rows:
            observation = build_observation(state, catalogue)
            index = next(index for index, option in enumerate(observation["options"][:-1])
                         if (option["sensor_id"], option["site_index"], option["yaw_deg"], option["pitch_deg"])
                         == (placement["profileId"], placement["siteId"], placement["yawDeg"], placement["pitchDeg"]))
            assert observation["action_mask"][index]
            state = apply_placement(state, index, catalogue)
        assert state["budget_remaining"] == 5.
    assert (ctx, original) == before
    assert legal_layout_probes(ctx, original, limit=0) == []


def test_evaluation_reports_paired_deltas_including_losses_and_ties():
    baseline = evaluation_summary([evidence(11, 5.), evidence(12, 7.), evidence(13, 10.)])
    result = evaluation_summary([evidence(11, 6.), evidence(12, 7.), evidence(13, 8.)], baseline)
    assert result["meanWarningSeconds"] == 7.
    assert result["pairedDeltasSeconds"] == [1., 0., -2.]
    assert result["deltaSeconds"] == pytest.approx(-1 / 3)
    assert result["improvedCases"] == result["tiedCases"] == result["worseCases"] == 1


@pytest.mark.parametrize("mismatch", ["seed", "trajectory", "budget", "catalogue", "sensor_selection", "weather"])
def test_evaluation_rejects_unpaired_scenarios_or_deployment_constraints(mismatch):
    original = evidence(11, 5.)
    changed = deepcopy(original)
    if mismatch == "seed": changed["seed"] = 12
    elif mismatch == "trajectory": changed["trajectory_sha256"] = "different-route"
    elif mismatch == "budget": changed["context"]["publicSnapshot"]["budget_total"] = 10.
    elif mismatch == "catalogue": changed["context"]["catalogue"][0]["cost"] = .5
    elif mismatch == "sensor_selection": changed["context"]["publicSnapshot"]["available_sensor_ids"] = []
    elif mismatch == "weather": changed["context"]["publicSnapshot"]["weather"] = {"rain": .5}
    with pytest.raises(ValueError, match="seed|trajector|constraints"):
        evaluation_summary([changed], evaluation_summary([original]))
