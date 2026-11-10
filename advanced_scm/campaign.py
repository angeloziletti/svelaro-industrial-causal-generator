"""Raw-material lots and campaign drift (vector_scm.shares.lot, .lot_persistence, .drift).

Batches are generated in order; the batch index is the time of the campaign. Two latent
batch-level processes act on every sensor:

    lot    L_b  stationary AR(1) over the batch index with persistence rho
                (consecutive batches share a raw-material lot)
    drift  R_b  a linear trend over the campaign (fouling, catalyst ageing, instrument drift)

Each is set by a share of the sensor's batch-to-batch variation: the variance over batches
of the sensor's batch mean m_b. That variation includes everything that changes between
batches (the batch parameters, whether a motif occurs, onset and size jitter), so the shares
mean what they say in the recorded data. The batch mean is mixed as in shares.py,

    m_b -> sqrt(1 - s_l - s_d) m^_b + sqrt(s_l) g L^_b + sqrt(s_d) h R^_b     (standardised units)

by adding a constant per batch to the sensor's own trajectory, with signs g, h per sensor.
The step runs on the sensors' own trajectories, before the sensor coupling, the control
loops and the target, so lots and drift propagate like any other part of a trajectory.
The batch-mean scale and the processes are frozen on the first call (the baseline
simulation), so a replay with an intervention (ground_truth_audit.py) is exact.
"""
from typing import Dict, Optional

import numpy as np

from advanced_scm.shares import draw

SALT_LOT, SALT_SIGN = 1103, 1201


def settings(shares: Optional[Dict], seed: int) -> Optional[Dict]:
    """The lot and drift shares of a plant, or None when both are off."""
    shares = shares or {}
    s_l = draw(shares.get("lot"), seed, "lot")
    s_d = draw(shares.get("drift"), seed, "drift")
    if not s_l and not s_d:
        return None
    s_l, s_d = float(s_l or 0.0), float(s_d or 0.0)
    if s_l + s_d > 1.0:
        raise ValueError(f"shares.lot + shares.drift must be <= 1, got {s_l} + {s_d}")
    return {"lot": s_l, "drift": s_d, "lot_persistence": float(shares.get("lot_persistence", 0.9))}


def processes(n_batches: int, rho: float, data_seed: int, structure_seed: int):
    """Standardised lot process L^ and trend R^ over the batches (R^ has a random sign)."""
    rng = np.random.default_rng(int(data_seed) + 7919 * SALT_LOT)
    v = rng.standard_normal(n_batches)
    lot = np.empty(n_batches)
    lot[0] = v[0]
    for b in range(1, n_batches):
        lot[b] = rho * lot[b - 1] + np.sqrt(1.0 - rho * rho) * v[b]
    ramp = np.arange(n_batches, dtype=float)
    sign = float(np.random.default_rng(int(structure_seed) + 7919 * SALT_SIGN).choice([-1.0, 1.0]))
    z = lambda x: (x - x.mean()) / (x.std() or 1.0)  # noqa: E731
    return z(lot), sign * z(ramp)


def apply(series: Dict[str, np.ndarray], cfg: Dict, frozen: Optional[Dict], data_seed: int,
          structure_seed: int):
    """Lots and drift on every sensor's own trajectory (n_batches, T). Returns the new series
    and the frozen state (scales, signs and processes of the baseline call)."""
    n = next(iter(series.values())).shape[0]
    if frozen is None:
        lot, trend = processes(n, cfg["lot_persistence"], data_seed, structure_seed)
        rng = np.random.default_rng(int(structure_seed) + 7919 * SALT_SIGN + 1)
        frozen = {"lot": lot.tolist(), "trend": trend.tolist(), "sensors": {}}
        for name in sorted(series):
            m = series[name].mean(axis=1)
            frozen["sensors"][name] = {"mean": float(m.mean()), "sd": float(m.std()),
                                       "g": float(rng.choice([-1.0, 1.0]))}
    lot, trend = np.asarray(frozen["lot"]), np.asarray(frozen["trend"])
    s_l, s_d = cfg["lot"], cfg["drift"]
    keep = np.sqrt(1.0 - s_l - s_d)
    out = {}
    for name, x in series.items():
        f = frozen["sensors"].get(name)
        if f is None or f["sd"] <= 0:
            out[name] = x
            continue
        m = x.mean(axis=1)
        shift = (keep - 1.0) * (m - f["mean"]) + f["sd"] * (np.sqrt(s_l) * f["g"] * lot + np.sqrt(s_d) * trend)
        out[name] = x + shift[:, None]
    return out, frozen


def realised(series: Dict[str, np.ndarray], frozen: Dict) -> Dict:
    """Per sensor, the share of its batch-mean variance explained by the lot process and by
    the trend (squared correlations, as recorded in the exported data)."""
    lot, trend = np.asarray(frozen["lot"]), np.asarray(frozen["trend"])
    out = {}
    for name in frozen["sensors"]:
        if name not in series:
            continue
        m = series[name].mean(axis=1)
        if m.std() == 0:
            continue
        out[name] = {"lot": float(np.corrcoef(m, lot)[0, 1] ** 2), "drift": float(np.corrcoef(m, trend)[0, 1] ** 2)}
    return out
