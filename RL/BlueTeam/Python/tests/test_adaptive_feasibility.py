import importlib.util
from pathlib import Path

import numpy as np
import pytest

from triad_rl.adaptive_env import generate_scenario
from triad_rl.adaptive_evaluation import canonical_hash, implementation_fingerprints


path = Path(__file__).resolve().parents[1] / "audit_adaptive_feasibility.py"
spec = importlib.util.spec_from_file_location("audit_adaptive_feasibility", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_confirmation_requires_distinct_supported_ticks_in_window_before_deadline():
    times = np.arange(7.)
    assert audit.confirmation_support([False, True, False, True, False, False, False], times, 3., window=3, required=2)["possible"]
    assert not audit.confirmation_support([False, True, False, False, True, False, False], times, 5., window=3, required=2)["possible"]
    assert not audit.confirmation_support([False, True, False, True, False, False, False], times, 2.9, window=3, required=2)["possible"]
    assert audit.confirmation_support([True] * 7, times, -1., window=3, required=2)["max_positive_ticks_in_window_before_deadline"] == 0


def test_no_emission_means_rf_support_is_zero_not_fabricated():
    scenario = generate_scenario(17)
    scenario["public"]["available_sensor_ids"] = ["rf"]
    for target in scenario["targets"]:
        target["emitter_duty"] = 0.
    result = audit.audit_scenario(scenario, [{"id": "rf", "cost": .5, "ranges": {"rf": 2000}, "strengths": {"rf": 1}}])
    assert not result["all_targets_possibly_timely_confirmable"]
    assert all(t["classification"] == "zero_sensing_support_before_deadline" for t in result["targets"])
    assert all(t["maximum_single_modality_single_look_probability_before_deadline"] == 0 for t in result["targets"])


def test_sufficient_radar_range_has_positive_support_without_promising_success():
    scenario = generate_scenario(17)
    scenario["public"]["available_sensor_ids"] = ["radar"]
    result = audit.audit_scenario(scenario, [{"id": "radar", "cost": .5, "ranges": {"radar": 2000}, "strengths": {"radar": .000001}}])
    assert result["all_targets_possibly_timely_confirmable"]
    assert all(t["classification"] == "positive_support_only_not_a_feasible_deployment_proof" for t in result["targets"])
    assert all(t["earliest_possible_confirmation"] == scenario["dt"] for t in result["targets"])


def test_empty_legal_catalogue_detects_no_initially_legal_options():
    scenario = generate_scenario(17)
    scenario["public"]["available_sensor_ids"] = []
    result = audit.audit_scenario(scenario, [{"id": "radar", "cost": .5, "ranges": {"radar": 2000}}])
    assert result["initially_legal_option_count"] == 0
    assert all(t["classification"] == "no_initially_legal_sensor_site_options" for t in result["targets"])


def test_short_slant_range_cannot_observe_high_altitude():
    scenario = generate_scenario(17)
    scenario["public"]["available_sensor_ids"] = ["radar"]
    for target in scenario["targets"]:
        target["altitude"] = 200.
        target["altitude_amplitude"] = 0.
    result = audit.audit_scenario(scenario, [{"id": "radar", "cost": .5, "ranges": {"radar": 50}}])
    assert all(not t["possible_any_sighting_before_arrival"] for t in result["targets"])


def test_individually_unaffordable_sensor_is_not_an_initially_legal_option():
    scenario = generate_scenario(17)
    scenario["public"]["available_sensor_ids"] = ["radar"]
    result = audit.audit_scenario(scenario, [{"id": "radar", "cost": 100, "ranges": {"radar": 2000}}])
    assert result["initially_legal_option_count"] == 0


def test_audit_refuses_mixed_source_version_before_processing_private_truth():
    with pytest.raises(ValueError, match="implementation"):
        audit.audit_report({"schema": "triad.adaptive_evaluation.v1", "implementation_sha256": {}})


def test_audit_refuses_guessing_missing_recorded_catalogue():
    with pytest.raises(ValueError, match="catalogue"):
        audit.audit_report({"schema": "triad.adaptive_evaluation.v1",
                            "implementation_sha256": implementation_fingerprints(),
                            "methods": {"adaptive": {"episodes": [{"seed": 17}]}}})


def test_audit_upper_bound_must_not_contradict_recorded_success():
    scenario = generate_scenario(17)
    scenario["public"]["available_sensor_ids"] = ["radar"]
    for target in scenario["targets"]:
        target["altitude"] = 200.
        target["altitude_amplitude"] = 0.
    catalogue = [{"id": "radar", "cost": .5, "ranges": {"radar": 50}}]
    row = {"seed": 17, "scenario": scenario, "scenario_sha256": canonical_hash(scenario),
           "metrics": {"success": True}, "replay": {"catalogue": catalogue}}
    with pytest.raises(RuntimeError, match="contradicts"):
        audit.audit_report({"schema": "triad.adaptive_evaluation.v1",
                            "implementation_sha256": implementation_fingerprints(),
                            "methods": {"adaptive": {"episodes": [row]}}})
