"""Behavior and shared legality contracts for sensible and human placements."""
from copy import deepcopy
import json

import pytest

from triad_rl.adaptive_inputs import INPUT_SCHEMA, DEFAULT_CATALOGUE
from triad_rl.common_sense import plan_common_sense, validate_manual_layout


@pytest.fixture
def state():
    return {"schema": INPUT_SCHEMA, "timestamp": 0., "sites": [[100., 0.], [0., 100.], [-100., 0.], [0., -100.]],
            "placements": [], "budget_total": 3., "budget_remaining": 3., "max_sites": 3,
            "min_separation": 20., "weather": {}, "tracks": [],
            "forecast": {"approach_weights": [1., 0., 0., 0., 0., 0., 0., 0.]}}


def test_ingress_priority_and_deterministic_public_only_plan(state):
    original, catalogue = deepcopy(state), deepcopy(DEFAULT_CATALOGUE)
    plan = plan_common_sense(state, catalogue)
    assert plan == plan_common_sense(state, catalogue)
    assert state == original and catalogue == DEFAULT_CATALOGUE
    assert plan["new_placements"][0]["site_index"] == 0
    assert plan["decisions"][-1]["stop"] and plan["final_public_state"]["done"]
    assert all(type(row["action_index"]) is int and row["reason"] for row in plan["decisions"])
    assert plan["selection"]["kind"] == "common_sense"
    assert plan["public_only"] and not plan["physical_commands_sent"]
    json.dumps(plan, allow_nan=False)
    with pytest.raises(ValueError, match="Unknown public"):
        plan_common_sense({**state, "private_trajectories": []}, catalogue)


def test_rotated_public_approach_rotates_first_site(state):
    state["forecast"]["approach_weights"] = [0., 0., 0., 0., 1., 0., 0., 0.]
    assert plan_common_sense(state)["new_placements"][0]["site_index"] == 2


def test_weather_changes_sensor_choice(state):
    catalogue = [dict(id="eo", cost=1., ranges={"eo": 120.}, strengths={"eo": .95}),
                 dict(id="radar", cost=1., ranges={"radar": 120.}, strengths={"radar": .85})]
    state["max_sites"] = 1
    clear = plan_common_sense(state, catalogue)
    state["weather"] = {"visibility": .1, "illumination": 0.}
    dark = plan_common_sense(state, catalogue)
    assert clear["new_placements"][0]["sensor_id"] == "eo"
    assert dark["new_placements"][0]["sensor_id"] == "radar"


def test_spreads_coverage_across_opposite_likely_approaches(state):
    state["forecast"]["approach_weights"] = [.5, 0., 0., 0., .5, 0., 0., 0.]
    state["sites"].insert(1, [100., 25.])
    state["max_sites"] = 2
    catalogue = [dict(id="radar", cost=1., ranges={"radar": 100.})]
    sites = [row["site_index"] for row in plan_common_sense(state, catalogue)["new_placements"]]
    assert set(sites) == {0, 3}


def test_budget_blocking_availability_and_separation(state):
    state.update(blocked_sites=[0], available_sensor_ids=["eo"], budget_total=1.5,
                 budget_remaining=1.5, min_separation=180.)
    plan = plan_common_sense(state)
    assert all(row["site_index"] != 0 and row["sensor_id"] == "eo" for row in plan["new_placements"])
    assert len(plan["new_placements"]) <= 2
    assert plan["final_public_state"]["budget_remaining"] >= 0
    validated = validate_manual_layout(state, DEFAULT_CATALOGUE, [
        {key: row[key] for key in ("sensor_id", "site_index")} for row in plan["new_placements"]])
    assert validated["final_public_state"] == plan["final_public_state"]


@pytest.mark.parametrize("change", [dict(blocked_sites=[0, 1, 2, 3]), dict(available_sensor_ids=[]),
                                  dict(budget_remaining=0.), dict(deployment_min_radius=120.)])
def test_no_legal_choice_stops_with_empty_layout(state, change):
    assert plan_common_sense({**state, **change})["new_placements"] == []


def test_no_useful_sensing_stops_without_wasting_budget(state):
    catalogue = [dict(id="inactive", cost=1., ranges={"rf": 130.}, strengths={"rf": 0.})]
    assert plan_common_sense(state, catalogue)["new_placements"] == []


def test_manual_layout_resolves_catalogue_cost_and_commits(state):
    original = deepcopy(state)
    rows = [{"sensor_id": "radar", "site_index": 2}, {"sensor_id": "eo", "site_index": 0}]
    plan = validate_manual_layout(state, DEFAULT_CATALOGUE, rows)
    assert state == original and len(rows) == 2
    assert plan["new_placements"][0]["position"] == [-100., 0.]
    assert plan["final_public_state"]["budget_remaining"] == pytest.approx(1.1)
    assert plan["selection"]["kind"] == "manual" and plan["decisions"][-1]["stop"]
    empty = validate_manual_layout(state, DEFAULT_CATALOGUE, [])
    assert empty["new_placements"] == [] and empty["final_public_state"]["done"]


@pytest.mark.parametrize("rows", [None, {}, [{"sensor_id": "rf"}],
    [{"sensor_id": "rf", "site_index": True}], [{"sensor_id": "rf", "site_index": 1.0}],
    [{"sensor_id": "rf", "site_index": -1}], [{"sensor_id": "rf", "site_index": 4}],
    [{"sensor_id": "unknown", "site_index": 0}], [{"sensor_id": [], "site_index": 0}],
    [{"sensor_id": "rf", "site_index": 0, "cost": 0}],
    [{"sensor_id": "rf", "site_index": 0}] * 2,
    [{"sensor_id": "fused", "site_index": 0}, {"sensor_id": "fused", "site_index": 2}],
    [{"sensor_id": "eo", "site_index": i} for i in range(4)]])
def test_manual_rejects_invalid_or_illegal_layout_atomically(state, rows):
    before = deepcopy(state)
    with pytest.raises(ValueError):
        validate_manual_layout(state, DEFAULT_CATALOGUE, rows)
    assert state == before


@pytest.mark.parametrize("change", [dict(blocked_sites=[0]), dict(available_sensor_ids=["eo"]),
                                  dict(deployment_min_radius=120.)])
def test_manual_shares_placement_constraints(state, change):
    with pytest.raises(ValueError, match="violates"):
        validate_manual_layout({**state, **change}, DEFAULT_CATALOGUE, [{"sensor_id": "radar", "site_index": 0}])


def test_manual_rejects_too_close_sites_even_with_remaining_budget(state):
    state["sites"][1] = [100., 10.]
    with pytest.raises(ValueError, match="violates"):
        validate_manual_layout(state, DEFAULT_CATALOGUE, [
            {"sensor_id": "eo", "site_index": 0}, {"sensor_id": "eo", "site_index": 1}])


def test_finished_public_snapshot_cannot_be_replanned(state):
    state["done"] = True
    with pytest.raises(ValueError, match="done"):
        plan_common_sense(state)
    with pytest.raises(ValueError, match="done"):
        validate_manual_layout(state, DEFAULT_CATALOGUE, [])


def test_existing_placements_consume_budget_and_sites(state):
    state.update(placements=[{"sensor_id": "radar", "position": [100., 0.]}],
                 budget_remaining=1.8, max_sites=2)
    plan = plan_common_sense(state)
    assert len(plan["new_placements"]) <= 1
    assert plan["final_public_state"]["placements"][0]["sensor_id"] == "radar"
    assert all(row["site_index"] != 0 for row in plan["new_placements"])
    with pytest.raises(ValueError, match="at most 1"):
        validate_manual_layout(state, DEFAULT_CATALOGUE, [
            {"sensor_id": "eo", "site_index": 1}, {"sensor_id": "eo", "site_index": 2}])


def test_zero_separation_still_prohibits_duplicate_coordinates(state):
    state.update(min_separation=0.)
    state["sites"][1] = list(state["sites"][0])
    with pytest.raises(ValueError, match="violates"):
        validate_manual_layout(state, DEFAULT_CATALOGUE, [
            {"sensor_id": "eo", "site_index": 0}, {"sensor_id": "eo", "site_index": 1}])
