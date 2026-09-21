"""Public common-sense decisions must respect the current directional sensors.

These scenarios exercise geometry, cost, redundancy and native plan hand-off;
they do not substitute a fake sensing environment or inspect private trajectories.
"""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from triad_rl.adaptive_inputs import INPUT_SCHEMA
from triad_rl.common_sense import RULE, plan_common_sense
from triad_rl.directional_inputs import (
    BOSON_PLUS_640_18MM, apply_placement, best_orientation, build_observation,
    to_legacy_inputs, validate_catalogue, validate_public_state,
)
from triad_rl.istana_live import make_plan
from triad_rl.temporal_inputs import TemporalConfig


@pytest.fixture
def public_state():
    return {
        "schema": INPUT_SCHEMA, "timestamp": 0.,
        "sites": [[30., 0.], [0., 30.], [-30., 0.], [0., -30.], [100., 0.]],
        "placements": [], "budget_total": 3., "budget_remaining": 3.,
        "max_sites": 3, "min_separation": 20., "weather": {}, "tracks": [],
        "forecast": {"approach_weights": [1., 0., 0., 0., 0., 0., 0., 0.],
                     "altitude": 45., "target_size_m": .4, "angular_uncertainty": .1},
    }


def native_context(state, catalogue):
    return {
        "runId": "directional-common-sense-contract", "revision": 1,
        "completedSteps": 0, "committed": False,
        "coordinateSystem": "unreal_xy_relative_m_z_up",
        "worldOriginCm": {"x": 0., "y": 0., "z": 0.},
        "catalogue": deepcopy(catalogue), "publicSnapshot": deepcopy(state),
        "temporalConfig": asdict(TemporalConfig()),
    }


def test_directional_plan_is_repeatable_and_leaves_public_inputs_unchanged(public_state):
    catalogue = [deepcopy(BOSON_PLUS_640_18MM)]
    before = deepcopy((public_state, catalogue))
    result = plan_common_sense(public_state, catalogue)
    assert result == plan_common_sense(public_state, catalogue)
    assert (public_state, catalogue) == before
    assert result["public_only"] and not result["physical_commands_sent"]
    assert result["selection"]["rule"] == RULE
    assert "joint type/site/yaw/pitch" in result["selection"]["sensor_model"]
    assert result["decisions"][-1]["stop"]
    assert result["final_public_state"]["done"]
    json.dumps(result, allow_nan=False)


def test_pointing_away_stops_even_when_all_approaches_are_within_500m(public_state):
    public_state["sites"] = [[-30., 0.]]
    sensor = deepcopy(BOSON_PLUS_640_18MM)
    sensor.update(yaw_bins_deg=[180.], pitch_bins_deg=[0.])
    assert sensor["ranges"]["thermal"] == 500.
    assert plan_common_sense(public_state, [sensor])["new_placements"] == []

    # The previous radial interpretation falsely considered this useful. Keeping
    # that contrast in the regression prevents reintroducing a 500m coverage disc.
    radial_state, radial_catalogue = to_legacy_inputs(public_state, [sensor])
    assert plan_common_sense(radial_state, radial_catalogue)["new_placements"]


def test_public_approach_rotation_changes_both_site_and_yaw(public_state):
    public_state["max_sites"] = 1
    east = plan_common_sense(public_state, [BOSON_PLUS_640_18MM])["new_placements"][0]
    public_state["forecast"]["approach_weights"] = [0., 0., 1., 0., 0., 0., 0., 0.]
    north = plan_common_sense(public_state, [BOSON_PLUS_640_18MM])["new_placements"][0]
    assert (east["site_index"], east["yaw_deg"], east["pitch_deg"]) == (0, 0., 20.)
    assert (north["site_index"], north["yaw_deg"], north["pitch_deg"]) == (1, 90., 20.)


def test_useful_coverage_per_actual_cost_can_outweigh_stronger_sensor(public_state):
    public_state.update(max_sites=1, budget_total=3., budget_remaining=3.)
    strongest = deepcopy(BOSON_PLUS_640_18MM)
    strongest.update(id="strongest", cost=2.)
    economical = deepcopy(BOSON_PLUS_640_18MM)
    economical.update(id="economical", cost=.8, strengths={"thermal": .4})
    chosen = plan_common_sense(public_state, [strongest, economical])["new_placements"][0]
    assert chosen["sensor_id"] == "economical"
    assert chosen["rationale"]["cost"] == .8
    assert chosen["rationale"]["new_coverage_estimate"] > 0
    assert chosen["rationale"]["overlap_estimate"] == 0

    economical["cost"] = 2.5
    assert plan_common_sense(public_state, [strongest, economical])["new_placements"][0]["sensor_id"] == "strongest"


def test_existing_east_facing_sensor_makes_west_the_next_useful_direction(public_state):
    public_state["forecast"]["approach_weights"] = [.5, 0., 0., 0., .5, 0., 0., 0.]
    public_state.update(max_sites=2, budget_remaining=2., placements=[
        {"sensor_id": "thermal", "position": [30., 0.], "yaw_deg": 0., "pitch_deg": 20.}])
    result = plan_common_sense(public_state, [BOSON_PLUS_640_18MM])
    assert len(result["new_placements"]) == 1
    row = result["new_placements"][0]
    assert (row["site_index"], row["yaw_deg"], row["pitch_deg"]) == (2, 180., 20.)
    assert row["rationale"]["overlap_estimate"] == 0.
    retained = result["final_public_state"]["placements"][0]
    assert retained["position"] == [30., 0.] and retained["yaw_deg"] == 0.
    assert result["final_public_state"]["budget_remaining"] == 1.


def test_layout_honors_budget_availability_sites_spacing_and_advertised_angle_bins(public_state):
    sensor = deepcopy(BOSON_PLUS_640_18MM)
    sensor.update(yaw_bins_deg=[90., 0.], pitch_bins_deg=[10.])
    unavailable = deepcopy(sensor)
    unavailable.update(id="unavailable", cost=.1)
    catalogue = [unavailable, sensor]
    public_state.update(available_sensor_ids=["thermal"], blocked_sites=[0],
                        min_separation=55., budget_total=1.5, budget_remaining=1.5)
    result = plan_common_sense(public_state, catalogue)
    assert len(result["new_placements"]) == 1
    state = deepcopy(public_state)
    for decision in result["decisions"]:
        observation = build_observation(state, catalogue)
        index = decision["action_index"]
        assert observation["action_mask"][index]
        option = observation["options"][index]
        assert all(decision[key] == value for key, value in option.items())
        if not decision["stop"]:
            assert decision["sensor_id"] == "thermal" and decision["site_index"] != 0
            assert decision["yaw_deg"] in sensor["yaw_bins_deg"]
            assert decision["pitch_deg"] == 10.
        state = apply_placement(state, index, catalogue)
    assert state == result["final_public_state"]
    assert state["budget_remaining"] == .5


def test_directional_spacing_can_force_stop_with_budget_still_available(public_state):
    public_state["min_separation"] = 75.
    result = plan_common_sense(public_state, [BOSON_PLUS_640_18MM])
    assert len(result["new_placements"]) == 1
    assert result["new_placements"][0]["site_index"] == 0
    # Every remaining candidate is less than 75m from this approved site.
    assert result["final_public_state"]["budget_remaining"] == 2.
    assert result["decisions"][-1]["stop"]


@pytest.mark.parametrize("overrides", [
    {"blocked_sites": [0, 1, 2, 3, 4]}, {"available_sensor_ids": []},
    {"budget_remaining": .5}, {"deployment_min_radius": 120.},
])
def test_directional_no_legal_gain_stops_without_spending(public_state, overrides):
    public_state.update(overrides)
    result = plan_common_sense(public_state, [BOSON_PLUS_640_18MM])
    assert result["new_placements"] == []
    assert len(result["decisions"]) == 1 and result["decisions"][0]["stop"]
    assert result["final_public_state"]["budget_remaining"] == public_state["budget_remaining"]


def test_native_common_sense_keeps_jointly_chosen_angles_instead_of_frozen_policy_adapter(public_state, monkeypatch):
    public_state["forecast"]["approach_weights"] = [.6, 0., 0., 0., .4, 0., 0., 0.]
    catalogue = [deepcopy(BOSON_PLUS_640_18MM)]
    expected = plan_common_sense(public_state, catalogue)
    validated = validate_public_state(public_state, validate_catalogue(catalogue))
    # The last sensor faces west because the earlier ones already cover east.
    # The legacy per-site adapter ignores those placements and would turn it east.
    last = expected["new_placements"][-1]
    assert last["yaw_deg"] == 180.
    assert best_orientation(validated, catalogue[0], last["position"])[0] == 0.

    def forbidden_adapter(*args, **kwargs):
        raise AssertionError("Common sense must retain its jointly selected orientation")
    monkeypatch.setattr("triad_rl.directional_inputs.best_orientation", forbidden_adapter)
    result = make_plan(native_context(public_state, catalogue), common_sense=True)
    assert result["recommendation"] == expected
    assert "orientation_adapter" not in result["recommendation"]["selection"]
    assert result["placements"] == [
        {"profileId": row["sensor_id"], "siteId": row["site_index"],
         "yawDeg": row["yaw_deg"], "pitchDeg": row["pitch_deg"]}
        for row in expected["new_placements"]]


def test_native_common_sense_uses_no_private_truth_or_temporal_reward_forecast(public_state, monkeypatch):
    context = native_context(public_state, [BOSON_PLUS_640_18MM])
    expected = make_plan(context, common_sense=True)
    context.update(private_red_truth={"future_paths": [[999., 888., 777.]]},
                   warningEvidenceForEvaluationOnly=[{"warningSeconds": 999.}],
                   reward=-1234., metrics={"detected_fraction": 0.})
    context["temporalConfig"].update(lead_time_s=10., horizon_s=120.)

    def forbidden_forecast(*args, **kwargs):
        raise AssertionError("The fixed baseline must not optimize the temporal reward forecast")
    monkeypatch.setattr("triad_rl.temporal_inputs.TemporalObservationBuilder.observe", forbidden_forecast)
    assert make_plan(context, common_sense=True) == expected
    with pytest.raises(ValueError, match="Unknown public"):
        plan_common_sense({**public_state, "private_trajectories": []}, [BOSON_PLUS_640_18MM])
