"""Hand-built public inputs only: no scenario generator, simulator or seed draws."""
from copy import deepcopy
import json

import numpy as np
import pytest

from triad_rl import adaptive_inputs as legacy
from triad_rl import temporal_inputs as temporal
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.balanced_policy import BalancedPolicy
from triad_rl.ranking_policy import RankPolicy


def catalogue():
    return [
        {"id": "public-rf", "label": "Custom RF", "cost": .8,
         "ranges": {"rf": 150.}, "strengths": {"rf": .9}, "height_m": 4.},
        {"id": "public-dual", "label": "Custom dual", "cost": 1.4,
         "ranges": {"radar": 150., "thermal": 150.},
         "strengths": {"radar": .85, "thermal": .85}, "height_m": 7.},
    ]


def public_state():
    return {
        "schema": legacy.INPUT_SCHEMA, "timestamp": 0., "source": "external",
        "episode_id": "hand-constructed-public-case", "max_track_age": 10.,
        "sites": [[120., 0.], [60., 0.], [0., 120.]], "placements": [],
        "budget_total": 5., "budget_remaining": 5., "max_sites": 3,
        "min_separation": 20., "deployment_min_radius": 30., "deployment_max_radius": 150.,
        "weather": {"visibility": .8, "rain": .2, "illumination": .6,
                    "humidity": .4, "rf_noise": .1},
        "forecast": {"approach_weights": [1., 0., 0., 0., 0., 0., 0., 0.],
                     "altitude": 50., "speed": 20., "emitter_probability": .5,
                     "swarm_size": 1., "angular_uncertainty": .1},
        "tracks": [{"id": "public-track", "position": [220., 0., 50.],
                    "velocity": [-20., 0., 0.], "confidence": 1., "timestamp": 0.,
                    "confirmed": False, "emitter_probability": .5}],
        "available_sensor_ids": [row["id"] for row in catalogue()], "blocked_sites": [], "done": False,
    }


def config(**changes):
    return temporal.TemporalConfig(**{"horizon_s": 24., "prior_spawn_radius_m": 220.,
                                      "altitude_uncertainty_m": 0., **changes})


def build(state=None, sensors=None, **kwargs):
    return temporal.build_observation(public_state() if state is None else state,
        catalogue() if sensors is None else sensors, config=kwargs.pop("config", config()), **kwargs)


def column(observation, name):
    return observation["option_features"][:, temporal.FEATURE_NAMES.index(name)]


def assert_observations_equal(left, right):
    np.testing.assert_array_equal(left["option_features"], right["option_features"])
    np.testing.assert_array_equal(left["action_mask"], right["action_mask"])
    for name in ("schema", "feature_schema", "feature_names", "options", "state", "catalogue",
                 "temporal_config", "temporal_forecast"):
        assert left[name] == right[name]


def assert_legacy_prefix(observation, state, sensors, *, now=None):
    frozen = legacy.build_observation(state, sensors, now=now)
    np.testing.assert_array_equal(observation["option_features"][:, :len(legacy.FEATURE_NAMES)],
                                  frozen["option_features"])
    np.testing.assert_array_equal(observation["action_mask"], frozen["action_mask"])
    for name in ("options", "state", "catalogue"):
        assert observation[name] == frozen[name]


def test_versioned_schema_exact_legacy_prefix_actions_masks_and_public_audit():
    state, sensors = public_state(), catalogue()
    observation = build(state, sensors)
    assert observation["schema"] == observation["feature_schema"] == temporal.FEATURE_SCHEMA
    assert observation["feature_names"] == temporal.FEATURE_NAMES
    assert temporal.FEATURE_NAMES[:len(legacy.FEATURE_NAMES)] == legacy.FEATURE_NAMES
    assert len(set(temporal.FEATURE_NAMES)) == len(temporal.FEATURE_NAMES)
    assert observation["option_features"].dtype == np.float32
    assert_legacy_prefix(observation, state, sensors)
    assert observation["temporal_forecast"]["schema"] == temporal.FORECAST_SCHEMA
    assert observation["temporal_forecast"]["public_only"] is True
    assert observation["temporal_forecast"]["physical_commands"] is False


@pytest.mark.parametrize("change", ["blocked", "availability", "budget", "done"])
def test_resource_and_legality_changes_preserve_frozen_masks(change):
    state, sensors = public_state(), catalogue()
    if change == "blocked":
        state["blocked_sites"] = [0, 2]
    elif change == "availability":
        state["available_sensor_ids"] = [sensors[0]["id"]]
    elif change == "budget":
        state["budget_remaining"] = .7
    else:
        state["done"] = True
    observation = build(state, sensors)
    assert_legacy_prefix(observation, state, sensors)
    assert observation["action_mask"][-1]
    legal_gain = column(observation, "temporal_marginal_timely")[observation["action_mask"]].max()
    np.testing.assert_array_equal(column(observation, "temporal_best_legal_timely_gain"), legal_gain)


def test_input_immutability_and_returned_arrays_do_not_poison_builder_cache():
    state, sensors = public_state(), catalogue()
    before = deepcopy((state, sensors))
    builder = temporal.TemporalObservationBuilder(config())
    first = builder.observe(state, sensors)
    expected = builder.observe(state, sensors)
    first["option_features"][:] = -999
    first["action_mask"][:] = False
    first["state"]["weather"]["rain"] = .99
    first["catalogue"][0]["strengths"]["rf"] = 0.
    first["temporal_config"]["lead_time_s"] = 30.
    assert_observations_equal(builder.observe(state, sensors), expected)
    assert (state, sensors) == before


def test_public_track_speed_changes_temporal_features_but_not_legacy_prefix():
    slow, fast = public_state(), public_state()
    slow["tracks"][0]["velocity"] = [-8., 0., 0.]
    fast["tracks"][0]["velocity"] = [-35., 0., 0.]
    left, right = build(slow), build(fast)
    np.testing.assert_array_equal(left["option_features"][:, :len(legacy.FEATURE_NAMES)],
                                  right["option_features"][:, :len(legacy.FEATURE_NAMES)])
    assert not np.array_equal(column(left, "temporal_timely"), column(right, "temporal_timely"))
    assert column(left, "temporal_timely").max() > column(right, "temporal_timely").max()


def test_public_track_altitude_changes_temporal_features_but_not_legacy_prefix():
    low, high = public_state(), public_state()
    low["tracks"][0]["position"][2] = 10.
    high["tracks"][0]["position"][2] = 250.
    left, right = build(low), build(high)
    np.testing.assert_array_equal(left["option_features"][:, :len(legacy.FEATURE_NAMES)],
                                  right["option_features"][:, :len(legacy.FEATURE_NAMES)])
    assert np.all(column(left, "temporal_detection") >= column(right, "temporal_detection"))
    assert column(left, "temporal_detection").max() > column(right, "temporal_detection").max()


def test_public_track_emitter_estimate_changes_rf_features_not_legacy_prefix():
    quiet, emitting = public_state(), public_state()
    quiet["tracks"][0]["emitter_probability"] = 0.
    emitting["tracks"][0]["emitter_probability"] = 1.
    left, right = build(quiet), build(emitting)
    np.testing.assert_array_equal(left["option_features"][:, :len(legacy.FEATURE_NAMES)],
                                  right["option_features"][:, :len(legacy.FEATURE_NAMES)])
    assert column(right, "temporal_timely")[0] > column(left, "temporal_timely")[0]
    np.testing.assert_array_equal(column(left, "temporal_timely")[3:], column(right, "temporal_timely")[3:])


def test_public_mission_deadline_changes_only_timely_not_detection_or_confirmation():
    early, late = build(config=config(lead_time_s=0.)), build(config=config(lead_time_s=9.))
    for key in ("temporal_detection", "temporal_confirmation", "temporal_early"):
        np.testing.assert_array_equal(column(early, key), column(late, key))
    assert np.all(column(early, "temporal_timely") >= column(late, "temporal_timely"))
    assert column(early, "temporal_timely").max() > column(late, "temporal_timely").max()


def test_public_confirmation_rules_change_repeat_requirement():
    single = build(config=config(required_confirmations=1, confirmation_window=1))
    repeated = build(config=config(required_confirmations=2, confirmation_window=3))
    np.testing.assert_array_equal(column(single, "temporal_detection"), column(repeated, "temporal_detection"))
    np.testing.assert_array_equal(column(single, "temporal_confirmation"), column(single, "temporal_detection"))
    assert np.any(column(repeated, "temporal_confirmation") < column(single, "temporal_confirmation"))


def test_stale_track_filter_and_explicit_empty_track_forecast_parity():
    state = public_state()
    observation = build(state, now=11.)
    assert_legacy_prefix(observation, state, catalogue(), now=11.)
    assert observation["state"]["tracks"] == []
    np.testing.assert_array_equal(column(observation, "temporal_track_mass"), 0.)
    empty = deepcopy(state)
    empty["tracks"] = []
    expected = build(empty, now=11.)
    np.testing.assert_array_equal(observation["option_features"], expected["option_features"])


def test_track_order_and_ids_are_not_physical_features():
    state = public_state()
    state["forecast"]["swarm_size"] = 3.
    state["tracks"].extend([
        {**deepcopy(state["tracks"][0]), "id": "second", "position": [0., 240., 70.],
         "velocity": [0., -15., 0.], "confidence": .6, "emitter_probability": .2},
        {**deepcopy(state["tracks"][0]), "id": "third", "position": [-200., 0., 30.],
         "velocity": [18., 0., 0.], "confidence": .3, "emitter_probability": .8},
    ])
    left = build(state)
    state["tracks"].reverse()
    for index, track in enumerate(state["tracks"]):
        track["id"] = f"renamed-{index}"
    right = build(state)
    np.testing.assert_array_equal(left["option_features"], right["option_features"])
    assert left["temporal_forecast"] == right["temporal_forecast"]


def test_custom_catalogue_ids_labels_and_order_preserve_physical_rows():
    state, sensors = public_state(), catalogue()
    left = build(state, sensors)
    renamed = deepcopy(sensors)
    for index, sensor in enumerate(renamed):
        sensor.update(id=f"arbitrary-{index}", label=f"Changed label {index}")
    state["available_sensor_ids"] = [row["id"] for row in renamed]
    right = build(state, renamed)
    np.testing.assert_array_equal(left["option_features"], right["option_features"])
    renamed.reverse()
    permuted = build(state, renamed)
    np.testing.assert_array_equal(right["option_features"][[3, 4, 5, 0, 1, 2, 6]], permuted["option_features"])
    assert column(right, "sensor_height")[0] == pytest.approx(.04)
    assert column(right, "calibrated_strength_rf")[0] == pytest.approx(.9)


def test_hard_slant_range_no_hits_and_resource_cost_still_count():
    sensors = catalogue()
    for sensor in sensors:
        sensor["ranges"] = {name: 5. for name in sensor["ranges"]}
    observation = build(sensors=sensors)
    for name in ("detection", "confirmation", "timely", "early", "window_support", "predeadline_hits"):
        np.testing.assert_array_equal(column(observation, "temporal_" + name), 0.)
    expected = [-(.55 * sensor["cost"] + .3) / 20. for sensor in sensors for _ in range(3)] + [0.]
    np.testing.assert_allclose(column(observation, "temporal_marginal_return"), expected, rtol=0., atol=2e-8)
    assert temporal.TemporalPublicGreedy().act(observation) == len(observation["options"]) - 1


def test_two_modalities_on_one_tick_cannot_double_confirm():
    sensors = [catalogue()[1]]
    state = public_state()
    state["available_sensor_ids"] = [sensors[0]["id"]]
    two = build(state, sensors, config=config(horizon_s=.1, lead_time_s=0.))
    one = build(state, sensors, config=config(horizon_s=.1, lead_time_s=0., required_confirmations=1))
    assert column(two, "temporal_detection").max() > 0.
    np.testing.assert_array_equal(column(two, "temporal_confirmation"), 0.)
    np.testing.assert_array_equal(column(two, "temporal_window_support"), 0.)
    np.testing.assert_array_equal(column(one, "temporal_confirmation"), column(one, "temporal_detection"))


def test_shared_rf_emission_mixes_after_sensor_union():
    forecast = {"times": np.array([0.]), "zone_times": np.array([10.]), "emitters": np.array([.25]),
                "confirmed": np.array([False]), "weights": np.array([1.]), "enters": np.array([True])}
    no_hit = np.ones((1, 1, 4))
    no_hit[..., 0] = (1. - .8) ** 2  # Two RF sensors, one shared on/off state.
    actual = temporal._statistics(no_hit, forecast, config())
    assert actual["detected"] == pytest.approx(.25 * (1. - .2 ** 2))
    assert actual["detected"] != pytest.approx(1. - (1. - .25 * .8) ** 2)
    assert actual["confirmed"] == 0.


def test_independent_and_persistent_emission_two_tick_mixture_is_explicit():
    forecast = {"times": np.array([0., 1.]), "zone_times": np.array([10.]), "emitters": np.array([.25]),
                "confirmed": np.array([False]), "weights": np.array([1.]), "enters": np.array([True])}
    no_hit = np.ones((1, 2, 4))
    no_hit[..., 0] = .2
    actual = temporal._statistics(no_hit, forecast, config(rf_persistent_weight=.5))
    assert actual["iid_timely"] == pytest.approx((.25 * .8) ** 2)
    assert actual["persistent_timely"] == pytest.approx(.25 * .8 ** 2)
    assert actual["timely"] == pytest.approx(.1)
    assert actual["predeadline_hits"] == pytest.approx(.4)


def test_stop_existing_values_and_marginal_zero_after_public_placement():
    state, sensors = public_state(), catalogue()
    before = build(state, sensors)
    placed = legacy._apply_legal_option(before["state"], before["options"][0], before["catalogue"])
    after = build(placed, sensors)
    assert_legacy_prefix(after, placed, sensors)
    for name in ("detection", "confirmation", "timely", "early"):
        assert column(after, "temporal_" + name)[-1] == column(before, "temporal_" + name)[0]
        assert column(after, "temporal_existing_" + name)[-1] == column(after, "temporal_" + name)[-1]
        assert column(after, "temporal_marginal_" + name)[-1] == 0.
    assert column(after, "temporal_marginal_return")[-1] == 0.
    assert column(after, "sensor_height")[-1] == 0.
    assert not after["action_mask"][0]  # Same occupied site remains prohibited.


def test_cost_change_affects_return_proxy_without_changing_sensing():
    sensors = catalogue()
    before = build(sensors=sensors)
    sensors[0]["cost"] += .4
    after = build(sensors=sensors)
    for name in ("temporal_detection", "temporal_confirmation", "temporal_timely", "temporal_early"):
        np.testing.assert_array_equal(column(before, name), column(after, name))
    np.testing.assert_allclose(column(after, "temporal_marginal_return")[:3] - column(before, "temporal_marginal_return")[:3],
                               -.55 * .4 / 20., rtol=0., atol=1e-7)


def test_reused_builder_equals_stateless_after_placement_weather_catalogue_clock_and_resources():
    state, sensors, now = public_state(), catalogue(), None
    builder = temporal.TemporalObservationBuilder(config())
    observation = builder.observe(state, sensors)
    for mutation in ("placement", "weather", "catalogue", "clock", "resources"):
        previous_key = builder._key
        if mutation == "placement":
            state = legacy._apply_legal_option(observation["state"], observation["options"][0], observation["catalogue"])
        elif mutation == "weather":
            state["weather"]["rain"] = .8
        elif mutation == "catalogue":
            sensors[1]["ranges"]["radar"] = 190.
        elif mutation == "clock":
            now = 1.
        else:
            state["blocked_sites"] = [2]
        observation = builder.observe(state, sensors, now=now)
        assert_observations_equal(observation, build(state, sensors, now=now))
        assert (builder._key == previous_key) == (mutation in ("placement", "resources"))


def test_public_json_external_simulation_source_and_dictionary_parity():
    state, sensors = public_state(), catalogue()
    builder = temporal.TemporalObservationBuilder(config())
    expected = builder.observe(state, sensors)
    assert_observations_equal(builder.observe(json.dumps(state), sensors), expected)
    state["source"] = "simulation"
    simulation = builder.observe(state, sensors)
    np.testing.assert_array_equal(simulation["option_features"], expected["option_features"])
    np.testing.assert_array_equal(simulation["action_mask"], expected["action_mask"])


@pytest.mark.parametrize("kind", [AdaptivePolicy, BalancedPolicy, RankPolicy])
def test_frozen_policy_contracts_reject_new_temporal_observation(kind):
    policy = kind(legacy.FEATURE_NAMES, seed=499)
    with pytest.raises(ValueError):
        policy.act(build())


def test_horizon_truncation_and_quadrature_reduction_are_exposed():
    short = build(config=config(horizon_s=1.))
    long = build()
    np.testing.assert_array_equal(column(short, "temporal_horizon_truncated_mass"), 1.)
    np.testing.assert_array_equal(column(long, "temporal_horizon_truncated_mass"), 0.)
    state = public_state()
    state["forecast"]["approach_weights"] = [.125] * 8
    reduced = build(state, config=config(max_hypotheses=8))
    full = build(state, config=config(max_hypotheses=128))
    assert reduced["temporal_forecast"]["hypotheses"] <= 8
    assert reduced["temporal_forecast"]["quadrature_reduced"] is True
    assert full["temporal_forecast"]["hypotheses"] > 8
    assert full["temporal_forecast"]["quadrature_reduced"] is False
    np.testing.assert_array_equal(column(reduced, "temporal_quadrature_reduced"), 1.)


def test_probability_ordering_finite_features_and_calibrated_strength_bounds():
    observation = build()
    assert np.isfinite(observation["option_features"]).all()
    for name in ("detection", "confirmation", "timely", "early", "window_support", "iid_timely", "persistent_timely"):
        values = column(observation, "temporal_" + name)
        assert ((0 <= values) & (values <= 1)).all()
    assert np.all(column(observation, "temporal_timely") <= column(observation, "temporal_confirmation"))
    assert np.all(column(observation, "temporal_confirmation") <= column(observation, "temporal_detection"))
    for modality in legacy.MODALITIES:
        assert ((0 <= column(observation, "calibrated_strength_" + modality))
                & (column(observation, "calibrated_strength_" + modality) <= 1)).all()


def test_invalid_mission_configuration_and_private_input_fields_fail_closed():
    invalid = [{"lead_time_s": True}, {"lead_time_s": -1.}, {"lead_time_s": float("nan")},
               {"look_interval_s": 0.}, {"required_confirmations": True}, {"required_confirmations": 4},
               {"confirmation_window": 9}, {"max_hypotheses": 7}, {"max_hypotheses": 129},
               {"rf_persistent_weight": 1.1}, {"altitude_uncertainty_m": -1.},
               {"horizon_s": 300., "look_interval_s": .1}]
    for changes in invalid:
        with pytest.raises(ValueError):
            config(**changes)
    with pytest.raises((ValueError, TypeError)):
        temporal.TemporalObservationBuilder({"unknown_private_rule": 1})
    for name in ("targets", "scenario_seed", "future_detections", "private_truth"):
        state = public_state()
        state[name] = []
        with pytest.raises(ValueError):
            build(state)


def test_temporal_greedy_stops_on_nonpositive_gain_and_stable_positive_tie():
    observation = build()
    actor = temporal.TemporalPublicGreedy()
    observation["option_features"][:, temporal.FEATURE_NAMES.index("temporal_marginal_return")] = 0.
    observation["action_mask"][0] = False
    assert actor.act(observation) == len(observation["options"]) - 1
    observation["option_features"][:-1, temporal.FEATURE_NAMES.index("temporal_marginal_return")] = 1.
    assert actor.act(observation) == 1
    observation["action_mask"][:-1] = False
    assert actor.act(observation) == len(observation["options"]) - 1
    observation["action_mask"] = observation["action_mask"].astype(int)
    with pytest.raises(ValueError):
        actor.act(observation)


def test_forecast_chunk_boundary_matches_identical_option_physics():
    state = public_state()
    state["sites"] = [[float(radius), 0.] for radius in range(35, 53)]
    observation = build(state)
    assert len(observation["options"]) == 37  # The second sensor spans the 32-row chunk boundary.
    assert np.isfinite(observation["option_features"]).all()
    for index in (0, 17, 18, 31, 32, 35):
        small = deepcopy(state)
        small["sites"] = [state["sites"][index % 18]]
        expected = build(small)
        keep = [i for i, name in enumerate(temporal.FEATURE_NAMES) if name != "temporal_best_legal_timely_gain"]
        np.testing.assert_array_equal(observation["option_features"][index, keep], expected["option_features"][index // 18, keep])


@pytest.mark.parametrize("velocity", [[20., 0., 0.], [0., 0., 0.], [0., 0., 20.]])
def test_outbound_stationary_and_vertical_tracks_do_not_invent_arrival_deadlines(velocity):
    inbound = build()
    state = public_state()
    state["tracks"][0]["velocity"] = velocity
    nonclosing = build(state)
    np.testing.assert_array_equal(inbound["option_features"][:, :len(legacy.FEATURE_NAMES)],
                                  nonclosing["option_features"][:, :len(legacy.FEATURE_NAMES)])
    np.testing.assert_array_equal(column(inbound, "temporal_no_predicted_entry_mass"), 0.)
    np.testing.assert_array_equal(column(nonclosing, "temporal_no_predicted_entry_mass"), .75)
    np.testing.assert_array_equal(column(nonclosing, "temporal_horizon_truncated_mass"), 0.)
    assert column(nonclosing, "temporal_timely").max() < column(inbound, "temporal_timely").max()
    validated = legacy.validate_public_state(state, legacy.validate_catalogue(catalogue()))
    forecast = temporal._forecast(validated, config(), 0.)
    outside = ~forecast["enters"]
    assert forecast["weights"][outside].sum() == pytest.approx(.75)
    # Only the independent public ingress prior retains a threat deadline.
    no_hit = np.full((*forecast["points"].shape[:2], 4), .2)
    only_nonclosing = {name: value[outside] if isinstance(value, np.ndarray) and name != "times" else value
                       for name, value in forecast.items()}
    only_nonclosing["weights"] /= only_nonclosing["weights"].sum()
    metrics = temporal._statistics(no_hit[outside], only_nonclosing, config())
    assert metrics["detected"] > 0. and metrics["confirmed"] > 0.
    assert metrics["timely"] == metrics["early"] == metrics["support"] == metrics["predeadline_hits"] == 0.


def test_known_confirmed_public_track_starts_confirmed_without_claiming_past_time():
    state = public_state()
    state["tracks"][0]["confirmed"] = True
    observation = build(state)
    for name in ("detection", "confirmation", "timely", "early"):
        assert column(observation, "temporal_existing_" + name)[-1] == pytest.approx(.75)
    assert "current planning time" in observation["temporal_forecast"]["confirmed_track"]
