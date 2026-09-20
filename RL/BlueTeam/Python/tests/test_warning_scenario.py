from copy import deepcopy
import math

import numpy as np
import pytest

from triad_rl.warning_scenario import approach_centers, digest, start_diagnostics, saturation_metrics
from train_warning_approach import paired_check


def red_context():
    return {"minRadiusCm": 26000, "maxRadiusCm": 30000,
            "objectiveWorldCm": {"x": 10, "y": 90, "z": 2560},
            "heightOffsetCm": 12000, "groupCount": 5, "spreadRadiusCm": 1000,
            "movement": {"spacingCm": 140}}


def test_approaches_reproducible_spaced_and_far():
    red = red_context()
    routes = []
    for seed in range(100):
        centers = approach_centers(red, seed)
        assert centers == approach_centers(red, seed)
        assert all(np.isclose(math.hypot(x-10, y-90), 28000) and z == 14560 for x,y,z in centers)
        assert min(np.linalg.norm(np.array(a)-b) for i,a in enumerate(centers) for b in centers[i+1:]) > 2140
        routes.append(round(math.atan2(centers[2][1]-90, centers[2][0]-10)/(math.pi/2)))
    assert len(set(routes)) >= 3
    red["minRadiusCm"] = 3000
    with pytest.raises(ValueError, match="IstanaWarningApproachV2"): approach_centers(red, 4)


def test_overcrowded_fan_rejected_not_silently_respawned():
    red = red_context()
    red["groupCount"] = 100
    with pytest.raises(ValueError, match="accommodate"): approach_centers(red, 4)


def test_distances_use_world_surface_height_and_do_not_leak_to_actor():
    context = {"catalogue": [{"id": "eo", "height_m": 4, "ranges": {"eo": 100}}],
               "siteSurfacesWorldCm": [[1000, 0, 2000], None],
               "publicSnapshot": {"blocked_sites": [1], "available_sensor_ids": ["eo"]}}
    drones = [{"droneId": 1, "positionCm": {"x": 31000, "y": 0, "z": 2400}}]
    before = deepcopy(context)
    result = start_diagnostics(context, [{"profileId": "eo", "siteId": 0}], drones)
    assert result["drones"][0]["nearest_sensor_distance_m"] == 300
    assert result["minimum_universal_clearance_m"] == 200
    assert result["spawned_in_range_count"] == 0 and context == before
    empty = start_diagnostics(context, [], drones)
    assert empty["drones"][0]["nearest_sensor_distance_m"] is None
    assert empty["minimum_universal_clearance_m"] == 200
    drones[0]["positionCm"]["x"] = 5000
    assert start_diagnostics(context, [{"profileId": "eo", "siteId": 0}], drones)["spawned_in_range_count"] == 1


def test_saturation_includes_zero_and_first_look_but_not_undetected():
    evidence = [{"firstDetectionSeconds": t} for t in (None, 0., 1., 2.)]
    assert saturation_metrics(evidence, 1.)["first_look_detected_fraction"] == .5
    assert not saturation_metrics([{"firstDetectionSeconds": None}], 1.)["team_first_look_saturated"]


@pytest.mark.parametrize("key", ["seed", "initial_states_sha256", "trajectory_sha256", "physical_contract_sha256", "red_centers_world_cm"])
def test_pairing_fails_on_any_physical_drift(key):
    row = {"seed": 1, "initial_states_sha256": "a", "trajectory_sha256": "b", "physical_contract_sha256": "c",
           "red_centers_world_cm": [[1,2,3]], "warning_evidence": [{"droneId": 1, "zoneEntrySeconds": 10}]}
    paired_check([row], [deepcopy(row)])
    changed = deepcopy(row)
    changed[key] = "different"
    with pytest.raises(ValueError, match="Unfair"): paired_check([row], [changed])
    changed = deepcopy(row)
    changed["warning_evidence"][0]["zoneEntrySeconds"] = 11
    with pytest.raises(ValueError, match="arrival"): paired_check([row], [changed])


def test_hash_is_key_order_independent():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
