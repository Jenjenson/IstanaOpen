"""Exact finite-state expectations for independent Bernoulli hit sequences.

The rolling window counts ticks, not elapsed seconds. First confirmation is
absorbing; a hit at the zone time is eligible, but a later hit is ignored. This
standalone numerical primitive uses no RNG, simulator state, or policy inputs.
"""
from numbers import Integral

import numpy as np


def _numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers, not booleans or strings")
    if not isinstance(value, np.ndarray) and any(
            isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
        raise ValueError(f"{name} must not contain booleans")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def confirmation_statistics(hit_probabilities, times, zone_times, *,
                            required_confirmations=2, confirmation_window=3, lead_time=4.):
    """Return five float64 arrays with the probabilities' leading shape.

    ``hit_probabilities`` has shape ``[..., ticks]`` (at most 512 ticks),
    ``times`` is a nonnegative, strictly increasing tick vector, and positive
    ``zone_times`` broadcasts to the leading shape. Empty tick axes are valid.
    Integer controls obey ``1 <= required_confirmations <= confirmation_window
    <= 8``; booleans are not numeric inputs. ``lead_time`` is a finite,
    nonnegative scalar. Inputs are never modified.

    ``detected`` is the probability of any eligible hit; ``confirmed`` is the
    probability of any confirmation; ``timely`` is the probability of first
    confirmation at or before ``zone_time - lead_time``. ``early`` is expected
    max(0, 1 - first_confirmation / zone_time), with zero for no confirmation.
    ``mean_confirmation_time_censored`` substitutes zone time when unconfirmed.
    Computation sums every sequence through an exact finite-state recurrence,
    subject only to float64 rounding, not simulation or an independence
    approximation between overlapping confirmation windows.
    """
    for name, value in (("required_confirmations", required_confirmations),
                        ("confirmation_window", confirmation_window)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer, not a boolean")
    required, window = int(required_confirmations), int(confirmation_window)
    if not 1 <= required <= window <= 8:
        raise ValueError("Require 1 <= required_confirmations <= confirmation_window <= 8")
    probabilities = _numeric(hit_probabilities, "hit_probabilities")
    ticks = _numeric(times, "times")
    zones = _numeric(zone_times, "zone_times")
    lead = _numeric(lead_time, "lead_time")
    if probabilities.ndim < 1 or probabilities.shape[-1] > 512:
        raise ValueError("hit_probabilities must have a tick axis of length at most 512")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("hit_probabilities must be in [0, 1]")
    if (ticks.ndim != 1 or len(ticks) != probabilities.shape[-1]
            or (ticks < 0).any() or (ticks[1:] <= ticks[:-1]).any()):
        raise ValueError("times must match ticks and be nonnegative and strictly increasing")
    if (zones <= 0).any():
        raise ValueError("zone_times must be positive")
    if lead.ndim != 0 or lead < 0:
        raise ValueError("lead_time must be a nonnegative scalar")
    shape = probabilities.shape[:-1]
    try:
        zones = np.broadcast_to(zones, shape)
    except ValueError as error:
        raise ValueError("zone_times must broadcast to the probabilities' leading shape") from error

    # Unconfirmed mass indexed by the last (window - 1) hit bits. Two source
    # states differing only in their oldest bit merge after each left shift.
    count = 1 << (window - 1)
    state = np.zeros((*shape, count), dtype=np.float64)
    state[..., 0] = 1.
    confirming = np.array([index.bit_count() >= required - 1 for index in range(count)])
    detected, confirmed, timely, early, first_time = (np.zeros(shape) for _ in range(5))
    for index, time in enumerate(ticks):
        probability = np.where(time <= zones, probabilities[..., index], 0.)
        detected += (1. - detected) * probability
        mass = state[..., confirming].sum(axis=-1) * probability
        confirmed += mass
        timely += mass * (time <= zones - lead)
        early += mass * (1. - np.minimum(time, zones) / zones)
        first_time += mass * time
        if window == 1:
            state *= (1. - probability)[..., None]
            continue
        half = count // 2
        following = np.empty_like(state)
        following[..., ::2] = (state[..., :half] + state[..., half:]) * (1. - probability)[..., None]
        # Absorbed mass cannot confirm again or flow back into a later window.
        state[..., confirming] = 0.
        following[..., 1::2] = (state[..., :half] + state[..., half:]) * probability[..., None]
        state = following
    return {"detected": np.asarray(np.clip(detected, 0., 1.)),
            "confirmed": np.asarray(np.clip(confirmed, 0., 1.)),
            "timely": np.asarray(np.clip(timely, 0., 1.)),
            "early": np.asarray(np.clip(early, 0., 1.)),
            "mean_confirmation_time_censored": np.asarray(np.clip(first_time + state.sum(axis=-1) * zones, 0., zones))}
