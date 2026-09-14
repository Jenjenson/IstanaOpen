"""Independent exhaustive sequence checks; no simulator, RNG or inference."""
from itertools import product

import numpy as np
import pytest

from triad_rl.temporal_confirmation import confirmation_statistics

KEYS = ("detected", "confirmed", "timely", "early", "mean_confirmation_time_censored")


def enumerate_statistics(probabilities, times, zone, required, window, lead):
    result = dict.fromkeys(KEYS, 0.)
    for bits in product((0, 1), repeat=len(times)):
        weight = float(np.prod([p if hit else 1. - p for p, hit in zip(probabilities, bits)]))
        eligible = [hit if time <= zone else 0 for hit, time in zip(bits, times)]
        first = next((time for i, time in enumerate(times) if time <= zone
                      and sum(eligible[max(0, i - window + 1):i + 1]) >= required), None)
        values = (bool(any(eligible)), first is not None,
                  first is not None and first <= zone - lead,
                  0. if first is None else max(0., 1. - first / zone), zone if first is None else first)
        for key, value in zip(KEYS, values):
            result[key] += weight * value
    return result


@pytest.mark.parametrize("window,required", [(w, k) for w in range(1, 5) for k in range(1, w + 1)])
@pytest.mark.parametrize("zone,lead", product((2., 4., 10.), (0., 1., 4.)))
def test_every_tiny_sequence_matches_independent_enumeration(window, required, zone, lead):
    probabilities, times = [.05, .2, .7, 1., 0., .35], [0., .5, 1., 2., 4., 7.]
    actual = confirmation_statistics(probabilities, times, zone, required_confirmations=required,
                                     confirmation_window=window, lead_time=lead)
    expected = enumerate_statistics(probabilities, times, zone, required, window, lead)
    assert set(actual) == set(KEYS)
    for key in KEYS:
        assert isinstance(actual[key], np.ndarray)
        assert actual[key].shape == () and actual[key].dtype == np.float64
        assert actual[key] == pytest.approx(expected[key], abs=2e-14)


@pytest.mark.parametrize("probabilities,confirmed", [([1, 0, 1], 1), ([1, 0, 0, 1], 0),
                                                        ([1, 0, 0, 1, 0, 1], 1)])
def test_two_of_three_gaps(probabilities, confirmed):
    result = confirmation_statistics(probabilities, np.arange(len(probabilities)), 10.)
    assert result["detected"] == 1 and result["confirmed"] == confirmed


def test_confirmation_absorbs_once_and_window_counts_ticks_not_seconds():
    result = confirmation_statistics([1, 1, 1, 1], [0, 100, 101, 102], 200., lead_time=100.)
    assert result == dict(detected=1., confirmed=1., timely=1., early=.5, mean_confirmation_time_censored=100.)


def test_inclusive_zone_and_timely_boundaries_and_ignored_late_hits():
    for time, expected in ((5.999, 1.), (6., 1.), (6.001, 0.)):
        result = confirmation_statistics([1, 1], [0, time], 10., lead_time=4.)
        assert result["timely"] == expected
    at_zone = confirmation_statistics([1, 1], [0, 10], 10., lead_time=0.)
    after_zone = confirmation_statistics([1, 1], [0, 10.001], 10., lead_time=0.)
    assert at_zone["confirmed"] == at_zone["timely"] == 1.
    assert at_zone["early"] == 0. and at_zone["mean_confirmation_time_censored"] == 10.
    assert after_zone["detected"] == 1. and after_zone["confirmed"] == 0.
    assert after_zone["mean_confirmation_time_censored"] == 10.
    assert confirmation_statistics([1], [11], 10., required_confirmations=1)["detected"] == 0.


def test_zero_one_probabilities_and_empty_ticks():
    zero = confirmation_statistics(np.zeros((2, 5)), np.arange(5), [5, 10])
    for key in KEYS[:-1]:
        np.testing.assert_array_equal(zero[key], [0., 0.])
    np.testing.assert_array_equal(zero[KEYS[-1]], [5., 10.])
    empty = confirmation_statistics(np.empty((2, 0)), [], [5, 10])
    for key in KEYS:
        np.testing.assert_array_equal(empty[key], zero[key])
    one = confirmation_statistics([1], [0], 10., required_confirmations=1, confirmation_window=1)
    assert one == dict(detected=1., confirmed=1., timely=1., early=1., mean_confirmation_time_censored=0.)


def test_broadcasting_matches_each_scalar_case_and_never_mutates_inputs():
    probabilities = np.linspace(0., 1., 30).reshape(2, 3, 5)
    times, zones = np.array([0., 1., 3., 5., 8.]), np.array([3., 7., 10.])
    originals = [item.copy() for item in (probabilities, times, zones)]
    actual = confirmation_statistics(probabilities, times, zones)
    for i, j in product(range(2), range(3)):
        expected = enumerate_statistics(probabilities[i, j], times, zones[j], 2, 3, 4.)
        for key in KEYS:
            assert actual[key].shape == (2, 3)
            assert actual[key][i, j] == pytest.approx(expected[key], abs=2e-14)
    for item, original in zip((probabilities, times, zones), originals):
        np.testing.assert_array_equal(item, original)


def test_increasing_hit_probabilities_is_monotone():
    values = [confirmation_statistics(np.linspace(.1, .5, 8) + shift, np.arange(8), 10.)
              for shift in (0., .1, .4)]
    for before, after in zip(values, values[1:]):
        for key in KEYS[:-1]:
            assert after[key] >= before[key]
        assert after[KEYS[-1]] <= before[KEYS[-1]]


def test_larger_window_or_fewer_required_hits_improves_confirmation():
    kwargs = dict(hit_probabilities=[.2, .4, .1, .8, .9], times=np.arange(5), zone_times=10.)
    baseline = confirmation_statistics(**kwargs, required_confirmations=3, confirmation_window=3)
    for improved in (confirmation_statistics(**kwargs, required_confirmations=2, confirmation_window=3),
                     confirmation_statistics(**kwargs, required_confirmations=3, confirmation_window=5)):
        assert improved["detected"] == baseline["detected"]
        for key in KEYS[1:-1]:
            assert improved[key] >= baseline[key]
        assert improved[KEYS[-1]] <= baseline[KEYS[-1]]


def test_supported_upper_bounds_and_numpy_integer_controls():
    result = confirmation_statistics(np.full(512, .2), np.arange(512), 512.,
                                     required_confirmations=np.int64(8), confirmation_window=np.int32(8))
    for key in KEYS[:-1]:
        assert np.isfinite(result[key]) and 0. <= result[key] <= 1.
    assert 0. <= result[KEYS[-1]] <= 512.
    assert result["timely"] <= result["confirmed"] <= result["detected"]


@pytest.mark.parametrize("name,value", [
    ("required_confirmations", value) for value in (True, np.bool_(False), 2., 0, -1, 4, "2", None)
] + [("confirmation_window", value) for value in (True, 3., 0, -1, 1, 9, "3", None)] + [
    ("lead_time", value) for value in (True, -1., np.nan, np.inf, [4.], "4", None)])
def test_invalid_controls_fail_closed(name, value):
    with pytest.raises((ValueError, TypeError)):
        confirmation_statistics([.5, .5], [0., 1.], 10., **{name: value})


@pytest.mark.parametrize("probabilities,times,zones", [
    (.5, [0.], 10.), ([.5] * 513, list(range(513)), 1000.),
    ([-.01], [0.], 10.), ([1.01], [0.], 10.), ([np.nan], [0.], 10.), ([np.inf], [0.], 10.),
    ([True], [0.], 10.), ([.5, True], [0., 1.], 10.), ([".5"], [0.], 10.), ([1j], [0.], 10.),
    ([.5], [], 10.), ([.5], [[0.]], 10.), ([.5], [-1.], 10.), ([.5], [np.nan], 10.),
    ([.5], [np.inf], 10.), ([.5], [True], 10.), ([.5, .5], [1., 1.], 10.),
    ([.5, .5], [2., 1.], 10.), ([.5], [0.], 0.), ([.5], [0.], -1.),
    ([.5], [0.], np.inf), ([.5], [0.], np.nan), ([.5], [0.], True),
    ([.5], [0.], [10.]), (np.full((2, 3), .5), [0., 1., 2.], [10., 10., 10.]),
])
def test_invalid_inputs_fail_closed(probabilities, times, zones):
    with pytest.raises((ValueError, TypeError)):
        confirmation_statistics(probabilities, times, zones)
