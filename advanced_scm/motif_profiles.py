"""Sensor trajectories as sequences of motifs (trajectory_model: motif_sequence).

A sensor's batch trajectory is a fixed sequence of 2-4 motifs -- its process
profile, e.g. ramp up -> plateau -> step down -- placed in consecutive time
slots, so segments are ordered and never overlap:

    x(t) = level + sum_k  fired_k * amplitude_k * BASE * shape_k(t; start_k, duration_k)
           + GP(t; noise_amplitude, lengthscale) + white noise

The motif types and their order are STRUCTURE (drawn once per graph). The
numbers are generative parameters (advanced_scm/parameter_layer.py), drawn per
batch -- from the SCM when causal, at random otherwise:

    s<k>_onset     where segment k starts inside its slot (fraction of the slot)
    s<k>_duration  how long segment k takes (fraction of the slot)
    s<k>_amplitude how big segment k is (multiplier of BASE)
    level, noise_amplitude, lengthscale   per sensor

Shapes (all deterministic given their parameters):
    step     smooth transition to a new level, which persists
    ramp     linear change over the duration, then the new level persists
    spike    transient Gaussian bump, back to where it was
    plateau  rise, hold, fall: a transient excursion
each with a direction (up/down) fixed in the structure.
"""
from typing import Dict, List, Optional

import numpy as np

from time_series.pattern.mvn import _cholesky  # cached, unique factor

MOTIFS = ["step", "ramp", "spike", "plateau"]
BASE = 1.5  # event size at amplitude 1.0, in units of the sensor's raw scale
WHITE_NOISE = 0.05


_UNIT_FACTORS: Dict = {}


def _unit_texture_factor(lengthscale: float, T: int) -> np.ndarray:
    """Cholesky factor of the unit-amplitude texture kernel, cached by
    (lengthscale, T). The amplitude scales the draw instead of the kernel
    (chol(a^2 K) = a chol(K)), so the factorisation runs once per lengthscale
    -- lengthscales are snapped to a 48-value grid -- not once per batch and
    sensor: the cache keyed on the full kernel never hit, because the
    amplitude differs every batch, and the factorisations dominated both
    generation and the audit's replays. Jitter 1e-6 on the unit kernel."""
    key = (float(lengthscale), int(T))
    L = _UNIT_FACTORS.get(key)
    if L is None:
        t = np.arange(T, dtype=float)
        K = np.exp(-0.5 * ((t[:, None] - t[None, :]) / (lengthscale * T)) ** 2)
        L = _cholesky(K + 1e-6 * np.eye(T))
        if len(_UNIT_FACTORS) >= 256:
            _UNIT_FACTORS.clear()
        _UNIT_FACTORS[key] = L
    return L


def plan_profile(rng: np.random.Generator, n_min: int = 2, n_max: int = 4) -> List[Dict]:
    """The sensor's fixed motif sequence (structure)."""
    k = int(rng.integers(n_min, n_max + 1))
    return [
        {"motif": str(rng.choice(MOTIFS)), "direction": int(rng.choice([-1, 1]))}
        for _ in range(k)
    ]


def _shape(motif: str, t: np.ndarray, start: float, dur: float) -> np.ndarray:
    dur = max(dur, 1.0)
    if motif == "step":
        width = max(1.0, 0.15 * dur)
        return 1.0 / (1.0 + np.exp(-(t - start - dur / 2) / width))
    if motif == "ramp":
        return np.clip((t - start) / dur, 0.0, 1.0)
    if motif == "spike":
        return np.exp(-0.5 * ((t - start - dur / 2) / (dur / 4)) ** 2)
    if motif == "plateau":
        edge = max(1.0, 0.15 * dur)
        up = np.clip((t - start) / edge, 0.0, 1.0)
        down = np.clip((start + dur - t) / edge, 0.0, 1.0)
        return np.minimum(up, down)
    raise ValueError(f"unknown motif {motif!r}")


def sensor_series(
    profile: List[Dict],
    params: Dict[str, float],
    T: int,
    rng: np.random.Generator,
    fired: Optional[List[bool]] = None,
    timing_shift: Optional[List[int]] = None,
    amplitude_factor: Optional[List[float]] = None,
    residual_scale: float = 1.0,
    white_noise: float = WHITE_NOISE,
) -> Dict:
    """One batch of one sensor. `rng` drives the noise only (seeded per batch
    and sensor by the caller); `fired`, `timing_shift` and `amplitude_factor`
    are the per-batch variation (stochastic firing and jitter) per segment."""
    t = np.arange(T, dtype=float)
    K = len(profile)
    slot = T / K
    mean = np.full(T, params["level"])
    windows = []
    for k, seg in enumerate(profile):
        start = k * slot + params[f"s{k}_onset"] * slot + (timing_shift[k] if timing_shift else 0)
        start = float(np.clip(start, k * slot, (k + 1) * slot - 1))
        dur = params[f"s{k}_duration"] * slot
        amp = params[f"s{k}_amplitude"] * (amplitude_factor[k] if amplitude_factor else 1.0)
        on = True if fired is None else bool(fired[k])
        if on:
            mean = mean + seg["direction"] * amp * BASE * _shape(seg["motif"], t, start, dur)
        windows.append((int(start), int(min(T - 1, start + dur)), on))
    texture = params["noise_amplitude"] * (
        _unit_texture_factor(params["lengthscale"], T) @ rng.standard_normal(T)
    )
    residual = residual_scale * (texture + rng.normal(0.0, white_noise, T))
    return {
        "series": mean + residual,
        "windows": windows,
        "mean_var": float(np.var(mean)),
        "residual_var": float(np.var(residual)),
    }
