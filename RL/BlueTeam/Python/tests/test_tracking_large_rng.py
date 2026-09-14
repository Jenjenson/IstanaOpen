from copy import deepcopy

from track_adaptive import dashboard_config


def test_dashboard_preserves_large_rng_integer_digits_without_mutating_evidence():
    source = {"rng": {"state": 2**127 + 19}, "nested": [2**63, -(2**63) - 1],
              "seed": 201, "validation": 1000991300000000, "enabled": True}
    before = deepcopy(source)
    result = dashboard_config(source)
    assert source == before
    assert result["rng"]["state"] == str(2**127 + 19)
    assert result["nested"] == [str(2**63), str(-(2**63) - 1)]
    assert result["seed"] == 201
    assert result["validation"] == 1000991300000000
    assert result["enabled"] is True


def test_dashboard_preserves_signed_64_bit_boundaries_and_nulls():
    source = [-(2**63), 2**63 - 1, None, False, .5, "unchanged"]
    assert dashboard_config(source) == source
