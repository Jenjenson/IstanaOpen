from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from triad_rl.warning_policy import WarningPolicy, warning_metrics
from triad_rl.istana_live import public_planning_inputs
from train_warning_live import red_centers, paired_interval


@pytest.fixture
def context():
    folder = Path(__file__).resolve().parents[2] / "Examples"
    def read(name): return json.loads((folder / name).read_text())
    state = read("public-snapshot.json")
    state.update(timestamp=0, placements=[], done=False, budget_remaining=state["budget_total"])
    return {"completedSteps": 0, "committed": False, "coordinateSystem": "unreal_xy_relative_m_z_up",
            "catalogue": read("sensor-catalogue.json"), "publicSnapshot": state,
            "temporalConfig": read("temporal-config.json")}


def test_policy_uses_only_public_input_and_respects_constraints(context):
    policy = WarningPolicy(context)
    before = deepcopy(context)
    for seed in range(30):
        placements, records = policy.plan(context, rng=np.random.default_rng(seed))
        assert len(placements) <= context["publicSnapshot"]["max_sites"]
        assert len({p["siteId"] for p in placements}) == len(placements)
        assert all(p["siteId"] not in context["publicSnapshot"]["blocked_sites"] for p in placements)
        costs = {s["id"]: s["cost"] for s in context["catalogue"]}
        assert sum(costs[p["profileId"]] for p in placements) <= context["publicSnapshot"]["budget_total"] + 1e-9
        points = [np.asarray(context["publicSnapshot"]["sites"][p["siteId"]]) for p in placements]
        assert all(np.linalg.norm(a-b) >= context["publicSnapshot"]["min_separation"] - 1e-9
                   for i,a in enumerate(points) for b in points[i+1:])
        for action, probabilities in records:
            assert probabilities[action] > 0 and np.isclose(probabilities.sum(), 1)
    poisoned = deepcopy(context)
    poisoned.update(private_red_truth={"position": [900, 900]}, reward=999999,
                    warningEvidenceForEvaluationOnly=[{"firstDetectionSeconds": 0}])
    a, _ = policy.plan(context, rng=np.random.default_rng(77))
    b, _ = policy.plan(poisoned, rng=np.random.default_rng(77))
    assert a == b and context == before


def test_reinforce_updates_toward_higher_return_and_roundtrips(context, tmp_path):
    p = WarningPolicy(context)
    probs = np.ones(len(p.logits())) / len(p.logits())
    high, low = 0, p.n_sites
    p.update([([(high, probs)], 10.), ([(low, probs)], 0.)])
    assert p.logits()[high] > p.logits()[low]
    path = tmp_path / "checkpoint.json"
    p.save(path)
    loaded = WarningPolicy.load(path, context)
    assert np.array_equal(p.logits(), loaded.logits())
    assert p.plan(context)[0] == loaded.plan(context)[0]
    with pytest.raises(FileExistsError): p.save(path)
    changed = deepcopy(context)
    changed["publicSnapshot"]["sites"][0][0] += 1
    with pytest.raises(ValueError): p.plan(changed)


def observation(rows):
    mean = np.mean([r["warningSeconds"] for r in rows])
    detections = [r["firstDetectionSeconds"] for r in rows if r["firstDetectionSeconds"] is not None]
    team = max(0, min(r["zoneEntrySeconds"] for r in rows) - min(detections)) if detections else 0
    return {"metricsAvailable": True, "warningEvidenceForEvaluationOnly": rows,
            "metrics": {"mean_drone_warning_seconds_lower_bound": mean,
                        "team_warning_seconds_lower_bound": team, "cost": 3}}


def test_warning_undetected_zero_and_distinct_team_mean():
    obs = observation([{"droneId": 1, "firstDetectionSeconds": 2, "zoneEntrySeconds": 20, "warningSeconds": 18},
                       {"droneId": 2, "firstDetectionSeconds": None, "zoneEntrySeconds": 10, "warningSeconds": 0}])
    metrics = warning_metrics(obs)
    assert metrics["mean_drone_warning_s"] == 9
    assert metrics["team_warning_s"] == 8
    assert metrics["detected_fraction"] == .5
    obs["warningEvidenceForEvaluationOnly"][0]["warningSeconds"] = 99
    with pytest.raises(ValueError, match="mismatch"): warning_metrics(obs)


@pytest.mark.parametrize("first,arrival,expected", [(None, 10, 0), (15, 10, 0), (0, 0, 0), (1, 11.75, 10.75)])
def test_warning_edge_cases(first, arrival, expected):
    metrics = warning_metrics(observation([{"droneId": 1, "firstDetectionSeconds": first,
                                            "zoneEntrySeconds": arrival, "warningSeconds": expected}]))
    assert metrics["team_warning_s"] == metrics["mean_drone_warning_s"] == expected


def test_warning_unresolved_or_nonterminal_rejected():
    with pytest.raises(ValueError): warning_metrics({"metricsAvailable": False})
    with pytest.raises(ValueError, match="unresolved"):
        warning_metrics({"metricsAvailable": True, "warningEvidenceForEvaluationOnly": [{"zoneEntrySeconds": None}]})


def test_red_rotation_reproducible_within_current_constraints():
    context = {"minRadiusCm": 3000, "maxRadiusCm": 10000, "objectiveWorldCm": {"x": 10, "y": 90, "z": 2560},
               "heightOffsetCm": 0, "groupCount": 5}
    a, b = red_centers(context, 91), red_centers(context, 92)
    assert a == red_centers(context, 91) and a != b
    assert all(np.isclose(np.hypot(x - 10, y - 90), 6500) and z == 2560 for x, y, z in a)
    assert paired_interval([0, 0, 0])["bootstrap_95_ci"] == [0., 0.]
