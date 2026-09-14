"""Curriculum reproducibility, diversity and shared public/scoring boundary."""
from copy import deepcopy
import json

import numpy as np
import pytest

from triad_rl.adaptive_env import AdaptivePlacementEnv, generate_scenario
from triad_rl.adaptive_inputs import (
    DEFAULT_CATALOGUE, FEATURE_NAMES, LiveObservationAdapter, apply_placement,
    build_observation, validate_catalogue,
)
from triad_rl.robust_scenarios import (
    PROFILES, ROBUST_SCENARIO_VERSION, RobustPlacementEnv, curriculum_manifest, make_case,
)


def assert_observation_equal(a, b):
    np.testing.assert_array_equal(a["option_features"], b["option_features"])
    np.testing.assert_array_equal(a["action_mask"], b["action_mask"])
    for key in ("state", "catalogue", "options", "feature_names", "feature_schema"):
        assert a[key] == b[key]


@pytest.mark.parametrize("profile", PROFILES)
def test_seeded_factory_and_wrapper_reproducible(profile):
    case = make_case(47, profile)
    assert case == make_case(47, profile)
    json.dumps(case, allow_nan=False)
    env = RobustPlacementEnv(seed=47, profile=profile)
    first = env.observe()
    assert env.case_metadata == case[2]
    assert env.scenario["targets"] == case[0]["targets"]
    assert env.catalogue == case[1]
    env.step(len(first["options"]) - 1)
    env.reset(seed=91)
    assert_observation_equal(first, env.reset(seed=47))
    assert_observation_equal(first, RobustPlacementEnv(seed=47, profile=profile).observe())


def test_unseeded_sequence_independent_of_explicit_resets():
    a, b = RobustPlacementEnv(seed=11), RobustPlacementEnv(seed=11)
    a.reset(seed=9234)
    for _ in range(3):
        assert_observation_equal(a.reset(), b.reset())
        assert a.case_metadata == b.case_metadata


@pytest.mark.parametrize("profile,split", [("normal", "train"), ("stress", "stress")])
def test_unmodified_base_family_and_sensing_results(profile, split):
    for seed in (4, 48, 800):
        scenario, catalogue, metadata = make_case(seed, profile)
        assert scenario == generate_scenario(seed, split)
        assert catalogue == validate_catalogue(DEFAULT_CATALOGUE)
        assert not metadata["scenario_filtered"]
        wrapper = RobustPlacementEnv(seed=seed, profile=profile)
        original = AdaptivePlacementEnv(seed=seed, split=split)
        for _ in range(4):
            observation = wrapper.observe()
            assert_observation_equal(observation, original.observe())
            legal = np.flatnonzero(observation["action_mask"][:-1])
            action = int(legal[0]) if len(legal) else len(observation["options"]) - 1
            _, reward_a, done_a, info_a = wrapper.step(action)
            _, reward_b, done_b, info_b = original.step(action)
            assert reward_a == reward_b and done_a == done_b and info_a == info_b
            if done_a:
                break
        assert wrapper.done


def test_mixture_and_capability_family_are_not_selected_from_outcomes():
    counts = dict.fromkeys(("normal", "stress", "capability"), 0)
    cap_splits = {"train": 0, "stress": 0}
    for seed in range(400):
        mixed = make_case(seed, "mixed")
        resolved = mixed[2]["profile"]
        counts[resolved] += 1
        # Explicitly selecting the resolved profile yields identical physics
        # and public constraints; the profile-selection draw consumes neither.
        explicit = make_case(seed, resolved)
        assert mixed[:2] == explicit[:2]
        cap = make_case(seed, "capability")
        cap_splits[cap[2]["threat_split"]] += 1
        original = generate_scenario(seed, cap[2]["threat_split"])
        for key in set(original) - {"public"}:
            assert cap[0][key] == original[key]
        for key in ("tracks", "forecast", "weather"):
            assert cap[0]["public"][key] == original["public"][key]
    assert .32 < counts["normal"] / 400 < .48
    assert .23 < counts["stress"] / 400 < .37
    assert .23 < counts["capability"] / 400 < .37
    assert .42 < cap_splits["train"] / 400 < .58


def test_catalogue_and_legal_offered_geometry_diversity():
    cases = [make_case(seed, "capability") for seed in range(50)]
    assert {case[2]["geometry"] for case in cases} == {"jittered_rings", "annular_cloud", "partial_arc"}
    assert len({len(case[0]["public"]["sites"]) for case in cases}) > 10
    assert len({tuple(case[0]["public"]["available_sensor_ids"]) for case in cases}) > 10
    assert {case[0]["public"]["max_sites"] for case in cases} == {2, 3, 4}
    assert any(case[0]["public"]["blocked_sites"] for case in cases)
    for scenario, catalogue, _ in cases:
        positions = np.asarray(scenario["public"]["sites"])
        radii = np.linalg.norm(positions, axis=1)
        assert radii.min() >= 35 and radii.max() <= 145
        assert len(np.unique(positions, axis=0)) == len(positions)
        assert [row["id"] for row in catalogue] == [row["id"] for row in DEFAULT_CATALOGUE]
        assert [row["label"] for row in catalogue] == [row["label"] for row in DEFAULT_CATALOGUE]
        for original, changed in zip(validate_catalogue(), catalogue):
            active = [m for m, r in original["ranges"].items() if r > 0]
            assert active == [m for m, r in changed["ranges"].items() if r > 0]
            assert .6 <= changed["cost"] / original["cost"] <= 1.5
            assert 2 <= changed["height_m"] <= 18
            for modality in active:
                assert .65 <= changed["ranges"][modality] / original["ranges"][modality] <= 1.75
                assert .55 <= changed["strengths"][modality] <= .98
    rf_ranges = [case[1][0]["ranges"]["rf"] for case in cases]
    assert max(rf_ranges) - min(rf_ranges) > 120


def test_catalogue_constraints_not_coupled_to_private_truth(monkeypatch):
    import triad_rl.robust_scenarios as module
    before = make_case(139, "capability")

    def changed_truth(seed, split):
        value = generate_scenario(seed, split)
        for target in value["targets"]:
            target.update(altitude=950., speed=190., emitter_duty=0., emitter_phase=.981)
        return value

    monkeypatch.setattr(module, "generate_scenario", changed_truth)
    after = make_case(139, "capability")
    assert before[0]["targets"] != after[0]["targets"]
    assert before[0]["public"] == after[0]["public"]
    assert before[1:] == after[1:]


def test_unchanged_scoring_uses_actual_varied_catalogue():
    scenario, catalogue, _ = make_case(21, "capability")
    wrapper = RobustPlacementEnv(seed=21, profile="capability")
    original = AdaptivePlacementEnv(catalogue=catalogue)
    original.reset(scenario=scenario)
    while not wrapper.done:
        observation = wrapper.observe()
        gain = observation["option_features"][:, FEATURE_NAMES.index("marginal_coverage")]
        action = int(np.argmax(np.where(observation["action_mask"], gain, -np.inf)))
        a = wrapper.step(action)
        b = original.step(action)
        assert_observation_equal(a[0], b[0])
        assert a[1:] == b[1:]
    assert wrapper.info["catalogue"] == catalogue
    assert wrapper.info["cost"] == pytest.approx(sum(p["cost"] for p in wrapper.placements))


@pytest.mark.parametrize("seed", [7, 19, 57, 89])
def test_external_snapshot_parity_and_budget_masks_all_steps(seed):
    env = RobustPlacementEnv(seed=seed, profile="capability")
    adapter = LiveObservationAdapter(env.catalogue)
    assert env.observe()["option_features"].shape[1] == len(FEATURE_NAMES)
    while not env.done:
        sim = env.observe()
        external = adapter.observe(json.dumps(env.public_state), now=0)
        assert_observation_equal(sim, external)
        for index, option in enumerate(sim["options"][:-1]):
            if sim["action_mask"][index]:
                assert option["sensor_id"] in env.public_state["available_sensor_ids"]
                assert option["site_index"] not in env.public_state["blocked_sites"]
                assert env.catalogue[option["sensor_index"]]["cost"] <= env.public_state["budget_remaining"] + 1e-9
                for placed in env.placements:
                    assert np.linalg.norm(np.array(placed["position"]) - option["position"]) >= env.public_state["min_separation"] - 1e-9
        legal = np.flatnonzero(sim["action_mask"][:-1])
        action = int(legal[0]) if len(legal) else len(sim["options"]) - 1
        planned = apply_placement(env.public_state, action, env.catalogue)
        after, _, done, _ = env.step(action)
        if done:
            # Simulator also automatically finishes when resource exhaustion
            # leaves STOP as the only legal option; provider steps it explicitly.
            planned["done"] = True
        assert_observation_equal(after, build_observation(planned, env.catalogue))
        assert sum(p["cost"] for p in env.placements) + env.public_state["budget_remaining"] == pytest.approx(env.public_state["budget_total"])
        assert len(env.placements) <= env.public_state["max_sites"]
    assert env.info["invalid_actions"] == 0
    assert env.info["return"] == pytest.approx(sum(env.info["reward_components"].values()))


def test_future_truth_and_audit_metadata_do_not_enter_observations():
    env = RobustPlacementEnv(seed=29, profile="capability")
    initial = env.observe()
    changed = deepcopy(env.scenario)
    changed["seed"] += 200
    for target in changed["targets"]:
        target.update(altitude=600., speed=100., emitter_duty=0., emitter_phase=.93)
    env.case_metadata["test_private_value"] = "must not appear"
    after = env.reset(scenario=changed)
    assert_observation_equal(initial, after)
    assert "targets" not in initial["state"]
    assert ROBUST_SCENARIO_VERSION not in json.dumps(initial["state"])
    assert "case_metadata" not in after
    with pytest.raises(ValueError, match="Unknown public"):
        LiveObservationAdapter(env.catalogue).observe({**env.public_state, "targets": changed["targets"]})


def test_no_affordable_or_available_sensors_remains_valid_stop_case():
    env = RobustPlacementEnv(seed=14, profile="capability")
    scenario = deepcopy(env.scenario)
    scenario["public"]["available_sensor_ids"] = []
    scenario["public"].update(budget_total=.001, budget_remaining=.001)
    observation = env.reset(scenario=scenario)
    assert not observation["action_mask"][:-1].any()
    _, reward, done, info = env.step(len(observation["options"]) - 1)
    assert done and info["cost"] == 0 and not info["success"]
    assert np.isfinite(reward)


def test_manifest_is_auditable_and_mutation_safe():
    manifest = curriculum_manifest()
    assert manifest["schema"] == ROBUST_SCENARIO_VERSION
    assert sum(manifest["mixed_probabilities"].values()) == 1
    assert manifest["capability"]["rejection_sampling"] is False
    assert manifest["stress"]["retain_impossible_cases"] is True
    assert len(set(manifest["stream_namespaces"].values())) == len(manifest["stream_namespaces"])
    json.dumps(manifest, allow_nan=False)
    manifest["stream_namespaces"]["catalogue"] = 0
    assert curriculum_manifest()["stream_namespaces"]["catalogue"] != 0


@pytest.mark.parametrize("seed", [True, -1, .5, "1", None])
def test_invalid_seeds_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        make_case(seed)
    with pytest.raises(ValueError, match="seed"):
        RobustPlacementEnv(seed=seed)


def test_invalid_profile_and_ambiguous_overrides_rejected():
    with pytest.raises(ValueError, match="profile"):
        make_case(1, "cherry-picked")
    with pytest.raises(ValueError, match="profile"):
        RobustPlacementEnv(profile="unknown")
    env = RobustPlacementEnv(seed=1)
    with pytest.raises(ValueError, match="override"):
        env.reset(seed=4, catalogue=DEFAULT_CATALOGUE)
    with pytest.raises(ValueError, match="match"):
        env.reset(seed=2, scenario=env.scenario)
