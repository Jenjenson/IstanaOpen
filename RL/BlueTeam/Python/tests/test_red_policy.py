import json

import numpy as np
import pytest

from triad_rl.red_policy import (DispersedRandomRedPolicy, FixedRadiusRandomBearingRedPolicy,
    FixedRadiusSectorRedPolicy, LearnedRedPlacementPolicy, RandomLegalRedPolicy, RedLayoutSpec,
    ScriptedRadialRedPolicy, layout_catalogue, public_approach_exposure)


@pytest.fixture
def context():
    return {"objectiveWorldCm": {"x": 100., "y": -200., "z": 300.},
            "groupCount": 3, "minRadiusCm": 3000., "maxRadiusCm": 6000.,
            "spreadRadiusCm": 400., "heightOffsetCm": 500.,
            "movement": {"spacingCm": 150.}}


def test_catalogue_is_stable_and_every_template_obeys_contract(context):
    layouts, mask = layout_catalogue(context)
    again, again_mask = layout_catalogue(context)
    assert layouts == again and np.array_equal(mask, again_mask)
    assert len(layouts) == 16 and mask.all()
    for action, layout in enumerate(layouts):
        assert layout["action"] == action and len(layout["centers"]) == 3
        for center in layout["centers"]:
            assert center[2] == 800.
            radius = np.linalg.norm(np.asarray(center[:2]) - [100., -200.])
            assert 3000 <= radius <= 6000


def test_impossible_inner_radius_is_masked():
    context = {"objectiveWorldCm": {"x": 0, "y": 0, "z": 0}, "groupCount": 8,
               "minRadiusCm": 100, "maxRadiusCm": 3000, "spreadRadiusCm": 500,
               "heightOffsetCm": 0, "movement": {"spacingCm": 200}}
    _, mask = layout_catalogue(context, RedLayoutSpec(radius_fractions=(0., 1.)))
    assert not mask[:16].any() and mask[16:].all()


def test_baselines_return_one_center_per_group(context):
    scripted = ScriptedRadialRedPolicy().select(context)
    random_a = RandomLegalRedPolicy(7).select(context)
    random_b = RandomLegalRedPolicy(7).select(context)
    assert len(scripted["centers"]) == len(random_a["centers"]) == 3
    assert random_a == random_b


def test_dispersed_random_is_seeded_and_legally_separated(context):
    first = DispersedRandomRedPolicy(17).select(context)
    second = DispersedRandomRedPolicy(17).select(context)
    assert first == second and first["formation"] == "dispersed"
    assert len(first["centers"]) == context["groupCount"]
    required = 2 * context["spreadRadiusCm"] + context["movement"]["spacingCm"]
    for index, center in enumerate(first["centers"]):
        radius = np.linalg.norm(np.asarray(center[:2]) - [100., -200.])
        assert context["minRadiusCm"] <= radius <= context["maxRadiusCm"]
        for other in first["centers"][:index]:
            assert np.linalg.norm(np.asarray(center[:2]) - other[:2]) >= required
    assert len({round(angle, 6) for angle in first["angles_degrees"]}) == context["groupCount"]


def test_fixed_radius_random_bearings_are_seeded_full_circle_and_separated(context):
    first = FixedRadiusRandomBearingRedPolicy(31).select(context)
    second = FixedRadiusRandomBearingRedPolicy(31).select(context)
    different = FixedRadiusRandomBearingRedPolicy(32).select(context)
    assert first == second and first != different
    assert first["formation"] == "random_bearings_fixed_radius" and not first["learned"]
    midpoint = (context["minRadiusCm"] + context["maxRadiusCm"]) / 2
    required = 2 * context["spreadRadiusCm"] + context["movement"]["spacingCm"]
    assert first["radius_cm"] == midpoint
    for index, center in enumerate(first["centers"]):
        assert np.linalg.norm(np.asarray(center[:2]) - [100., -200.]) == pytest.approx(midpoint)
        for other in first["centers"][:index]:
            assert np.linalg.norm(np.asarray(center[:2]) - other[:2]) >= required
    assert all(0 <= angle < 360 for angle in first["angles_degrees"])


def test_fixed_radius_sectors_are_seeded_distinct_jittered_and_separated(context):
    context["groupCount"] = 5
    first = FixedRadiusSectorRedPolicy(31).select(context)
    second = FixedRadiusSectorRedPolicy(31).select(context)
    different = FixedRadiusSectorRedPolicy(32).select(context)
    assert first == second and first != different
    assert first["formation"] == "randomized_eight_sector_fixed_radius"
    assert len(first["sector_indices"]) == len(set(first["sector_indices"])) == 5
    assert all(0 <= sector < 8 for sector in first["sector_indices"])
    assert all(abs(jitter) <= 4 for jitter in first["jitter_degrees"])
    midpoint = (context["minRadiusCm"] + context["maxRadiusCm"]) / 2
    required = 2 * context["spreadRadiusCm"] + context["movement"]["spacingCm"]
    for index, (center, angle, sector_center) in enumerate(zip(
            first["centers"], first["angles_degrees"], first["sector_centers_degrees"])):
        assert np.linalg.norm(np.asarray(center[:2]) - [100., -200.]) == pytest.approx(midpoint)
        assert abs((angle - sector_center + 180) % 360 - 180) <= 4
        for other in first["centers"][:index]:
            assert np.linalg.norm(np.asarray(center[:2]) - other[:2]) >= required


def test_fixed_radius_sector_policy_rejects_more_than_eight_groups(context):
    context["groupCount"] = 9
    with pytest.raises(ValueError, match="at most eight"):
        FixedRadiusSectorRedPolicy(1).select(context)


def test_positive_reward_increases_selected_action_probability(context):
    policy = LearnedRedPlacementPolicy(seed=4)
    warmup = policy.select(context, deterministic=False)
    policy.update(warmup["training_record"], 0., entropy_coefficient=0.)
    decision = policy.select(context, deterministic=False)
    action = decision["training_record"]["action"]
    before = decision["training_record"]["probabilities"][action]
    policy.update(decision["training_record"], 5., entropy_coefficient=0.)
    after = policy.select(context, deterministic=False)["training_record"]["probabilities"][action]
    assert after > before


def test_first_negative_return_bootstraps_baseline_without_biasing_action(context):
    policy = LearnedRedPlacementPolicy(seed=4)
    decision = policy.select(context, deterministic=False)
    before = policy.logits.copy()
    update = policy.update(decision["training_record"], -28., entropy_coefficient=0.)
    assert np.array_equal(policy.logits, before)
    assert update["advantage"] == 0. and policy.baseline == -28.


def test_public_exposure_changes_with_approach_direction(context):
    blue = {"catalogue": [{"id": "radar", "height_m": 0,
             "ranges": {"rf": 0, "radar": 60, "eo": 0, "thermal": 0},
             "strengths": {"rf": 0, "radar": 1, "eo": 0, "thermal": 0}}],
            "publicSnapshot": {"sites": [[30, 0]],
                "weather": {"visibility": 1, "rain": 0, "illumination": 1,
                            "humidity": 0, "rf_noise": 0}}}
    near = [[4000, -200, 300]]
    far = [[-4000, -200, 300]]
    placement = [{"profileId": "radar", "siteId": 0}]
    assert public_approach_exposure(context, blue, placement, near) > public_approach_exposure(
        context, blue, placement, far)


def test_checkpoint_round_trip_and_integrity(context, tmp_path):
    policy = LearnedRedPlacementPolicy(seed=9)
    decision = policy.select(context, deterministic=False)
    policy.update(decision["training_record"], 2.)
    path = tmp_path / "checkpoint"
    policy.save(path)
    loaded = LearnedRedPlacementPolicy.load(path)
    assert np.array_equal(loaded.logits, policy.logits)
    assert loaded.baseline == policy.baseline and loaded.episodes == 1
    assert loaded.select(context)["action"] == policy.select(context)["action"]
    metadata = json.loads((path / "checkpoint.json").read_text())
    metadata["arrays_sha256"] = "0" * 64
    (path / "checkpoint.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash differs"):
        LearnedRedPlacementPolicy.load(path)


@pytest.mark.parametrize("field", ["groupCount", "minRadiusCm", "objectiveWorldCm"])
def test_bad_context_is_rejected(context, field):
    del context[field]
    with pytest.raises((TypeError, ValueError)):
        layout_catalogue(context)
