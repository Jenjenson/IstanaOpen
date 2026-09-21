from copy import deepcopy

import numpy as np

from triad_rl.adaptive_env import AdaptivePlacementEnv
from triad_rl.directional_inputs import (
    BOSON_PLUS_640_18MM, FEATURE_SCHEMA, apply_placement, build_observation,
    pixels_on_target, sensing_probabilities, validate_catalogue,
    vertical_fov_from_horizontal,
)


def state_for_directional():
    state = deepcopy(AdaptivePlacementEnv(seed=4).public_state)
    state["available_sensor_ids"] = ["thermal"]
    state["budget_total"] = state["budget_remaining"] = 3.
    state["forecast"]["target_size_m"] = .4
    return state


def test_boson_facts_calculation_and_assumptions_are_separate():
    sensor = validate_catalogue([BOSON_PLUS_640_18MM])[0]
    hardware, assumptions = sensor["manufacturer_specifications"], sensor["simulation_assumptions"]
    assert hardware == {"manufacturer": "Teledyne FLIR", "model": "Boson+ 640, 24deg HFOV, 18 mm",
                        "modality": "uncooled LWIR thermal",
                        "source": "https://oem.flir.com/products/boson-plus/?model=22640A024",
                        "resolution": [640, 512], "pixel_pitch_um": 12., "horizontal_fov_deg": 24.,
                        "ifov_mrad": .667, "frame_rate_hz": 60., "selectable_frame_rate_hz": 30.,
                        "industrial_nedt_mk_max": 20.}
    assert sensor["calculated_geometry"]["vertical_fov_deg"] == vertical_fov_from_horizontal(24, 640, 512)
    assert np.isclose(sensor["calculated_geometry"]["vertical_fov_deg"], 19.31, atol=.01)
    assert assumptions["max_evaluation_distance_m"] == 500.
    assert assumptions["detection_model"] == "pixels_on_target_v1"


def test_pixels_probability_frustum_weather_and_los():
    sensor = validate_catalogue([BOSON_PLUS_640_18MM])[0]
    assert np.allclose(pixels_on_target(.5, np.array([100., 200., 300., 400., 500.]), .667),
                       [7.496, 3.748, 2.499, 1.874, 1.499], atol=.01)
    points = np.array([[100., 0., 4.], [200., 0., 4.], [300., 0., 4.], [400., 0., 4.],
                       [500., 0., 4.], [100., 50., 4.]])
    weather = {"visibility": 1., "rain": 0., "illumination": 1., "humidity": 0., "rf_noise": 0.}
    kwargs = dict(orientations=np.array([[0., 0.]]), target_size_m=.5)
    clear = sensing_probabilities(np.array([[0., 0.]]), [sensor], points, weather, 1., **kwargs)[0, :, 3]
    assert np.all(np.diff(clear[:5]) < 0) and clear[0] < 1 and clear[3] > 0 and clear[4] < 1e-20
    assert clear[5] == 0
    degraded = sensing_probabilities(np.array([[0., 0.]]), [sensor], points,
        {**weather, "visibility": .5, "rain": .8, "humidity": .9}, 1., **kwargs)[0, :, 3]
    assert np.all(degraded[:4] < clear[:4])
    blocked = sensing_probabilities(np.array([[0., 0.]]), [sensor], points, weather, 1.,
        line_of_sight=np.zeros((1, len(points)), dtype=bool), **kwargs)[0, :, 3]
    assert not blocked.any()


def test_directional_actions_store_orientation_and_stop():
    state = state_for_directional()
    observation = build_observation(state, [BOSON_PLUS_640_18MM])
    assert observation["schema"] == FEATURE_SCHEMA
    assert len(observation["options"]) == len(state["sites"]) * 8 * 3 + 1
    assert {(row["yaw_deg"], row["pitch_deg"]) for row in observation["options"][:-1]} == {
        (yaw, pitch) for yaw in BOSON_PLUS_640_18MM["yaw_bins_deg"]
        for pitch in BOSON_PLUS_640_18MM["pitch_bins_deg"]}
    selected = int(np.flatnonzero(observation["action_mask"][:-1])[0])
    placed = apply_placement(state, selected, [BOSON_PLUS_640_18MM])
    assert set(placed["placements"][0]) >= {"sensor_id", "position", "yaw_deg", "pitch_deg"}
    rebuilt = build_observation(placed, [BOSON_PLUS_640_18MM])
    assert not rebuilt["action_mask"][selected]
