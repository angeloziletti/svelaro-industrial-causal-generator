"""Edge strengths as explained-variance shares (vector_scm.shares).

A share is the fraction of a node's variance that comes from one parent group;
per node the shares and the node's own remainder sum to 1, and a share means the
same thing on every node whatever the mechanism (dynamic edge, nonlinearity,
linear parameter edge). Each contribution is standardised to unit variance on the
baseline simulation -- scales frozen there, so interventions stay exact -- and
weighted by the square root of its share:

    node = sqrt(s) * contribution_hat + sqrt(1 - s) * rest_hat

Shares are structure: drawn once per plant from the structure seed. A spec is a
number (fixed share) or [lo, hi] (uniform per plant).

    vector_scm.shares:
      defect_timeseries: [0.5, 0.8]   # the defect: causes' time-series route vs noise
      consequence: [0.1, 0.4]         # a consequence: echo of the true quality vs its own process
      confounder: [0.1, 0.5]          # a confounded parameter: hidden confounder vs the rest
      unexplained: process            # the defect's unexplained part: "white" (per-step noise, the
                                      # default) or "process" (same time scale as the causes' signal,
                                      # as unmeasured causes would be); see process_noise
      echo: [0.1, 0.5]                # a sensor downstream of another: its parents' signal vs its own
                                      # (both routes: recipe edges between batch parameters, and
                                      # dynamic edges between curves)
"""
from typing import Any, Optional

import numpy as np

SALT = {"defect_timeseries": 101, "consequence": 211, "confounder": 307, "echo": 401, "echo_recipe": 503,
        "interaction": 601, "measurement": 701, "cumulative": 809,
        "operator": 907, "phase": 1009, "lot": 1103, "drift": 1109}


def draw(spec: Any, seed: int, name: str, index: int = 0) -> Optional[float]:
    """The share for one node (index distinguishes nodes of the same kind)."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        s = float(spec)
    else:
        lo, hi = (float(v) for v in spec)
        s = float(np.random.default_rng(int(seed) + 7919 * SALT[name] + index).uniform(lo, hi))
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"share {name} must be in [0, 1], got {s}")
    return s


def standardise(x: np.ndarray, scale) -> np.ndarray:
    """(x - mean) / sd with a frozen [mean, sd]."""
    mu, sd = scale
    return (x - mu) / (sd if sd > 0 else 1.0)


def scale_of(x: np.ndarray):
    return [float(np.mean(x)), float(np.std(x))]


def mix(contribution_hat: np.ndarray, rest_hat: np.ndarray, share: float) -> np.ndarray:
    return np.sqrt(share) * contribution_hat + np.sqrt(1.0 - share) * rest_hat


def calibrate_confounder_shares(scm, spec, seed: int, n_pilot: int = 3000) -> dict:
    edges = sorted(
        (z, p) for z, p in scm.graph.edges
        if scm.node_types.get(z) == "shadow_confounder" and scm.node_types.get(p) == "gp_parameter"
        and not str(z).startswith("phase_")
    )
    return calibrate_edge_shares(scm, spec, seed, edges, "confounder", n_pilot)


def calibrate_phase_shares(scm, spec, seed: int, n_pilot: int = 3000) -> dict:
    """Shared batch phases: each phase node carries share `spec` of every onset it times
    (graph_generators._inject_phase_nodes)."""
    edges = sorted(
        (z, p) for z, p in scm.graph.edges
        if str(z).startswith("phase_") and not str(z).startswith("phase_root_")
        and scm.node_types.get(p) == "gp_parameter"
    )
    return calibrate_edge_shares(scm, spec, seed, edges, "phase", n_pilot)


def calibrate_recipe_shares(scm, spec, seed: int, n_pilot: int = 3000) -> dict:
    """Recipe edges (one sensor's batch parameter -> another's): the echo share,
    batch-parameter route."""
    sg = scm.graph.graph.get("sensor_graph") or {}
    edges = sorted(tuple(e["via"]) for e in sg.get("edges") or [] if scm.graph.has_edge(*e["via"]))
    return calibrate_edge_shares(scm, spec, seed, edges, "echo_recipe", n_pilot)


def calibrate_edge_shares(scm, spec, seed: int, edges, name: str, n_pilot: int = 3000) -> dict:
    """Set every hidden-confounder edge Z -> parameter so that Z carries a drawn
    share of the parameter's variance.

    The parameter mechanisms are linear (core.py: gp_parameter nodes), so a pilot
    sample of the parameter SCM gives, per confounded parameter p with current
    weight w0, rest = p - w0 * Z exactly; the weight w solving
        w^2 var(Z) = s * var(w Z + rest)
    (covariance with the rest included) gives share s. The weight is changed in
    place in the mechanism's own weight dict: redefining the mechanisms would
    redraw every other random weight. The global numpy state is saved and
    restored, so the data drawn afterwards are unaffected by the pilot."""
    if spec is None or not edges:
        return {}
    state = np.random.get_state()
    try:
        pilot = scm.generate_dataset(n_pilot, apply_post_processing=False)
    finally:
        np.random.set_state(state)
    out = {}
    for i, (z, p) in enumerate(edges):
        weights = scm.node_parameters[p]["weights"]
        w0 = float(weights[z])
        zv = pilot[z].to_numpy(dtype=float)
        rest = pilot[p].to_numpy(dtype=float) - w0 * zv
        vz, vr = float(np.var(zv)), float(np.var(rest))
        c = float(np.cov(zv, rest)[0, 1])
        s = draw(spec, seed, name, i)
        if vz <= 0 or s >= 1.0:
            continue
        a, b, cc = vz * (1.0 - s), -2.0 * s * c, -s * vr
        w = (-b + np.sqrt(b * b - 4 * a * cc)) / (2 * a)
        weights[z] = float(w)
        out[f"{z}->{p}"] = {"share": s, "weight": float(w), "weight_before": w0}
    return out


def decorrelation_steps(x: np.ndarray, max_lag: int = 256, within_batch: bool = False,
                        noise_robust: bool = False) -> int:
    """First lag at which the batch-averaged ACF of x (n_batches, T) drops below 1/e.
    The ONE definition used by the generator and the evaluation (docs section 0, "one tool,
    two time scales"):
      within_batch=False  global mean removed: batch-level offsets count as slow variation
                          (the time scale the unexplained noise copies; per-batch centring
                          would also bias a slow AR(1) down: time scale 40 reads as ~29)
      within_batch=True   each batch centred on its own mean: within-batch dynamics only
                          (resolution and look-back of every method; consequence delays)
      noise_robust=True   ACF normalised by its lag-1 value, so white measurement noise
                          (which only lowers lags >= 1 by a constant) does not shorten it"""
    x = np.asarray(x, dtype=float)
    x = x - (x.mean(axis=1, keepdims=True) if within_batch else x.mean())
    T = x.shape[1]
    n = 1 << int(np.ceil(np.log2(2 * T)))
    f = np.fft.rfft(x, n=n, axis=1)
    acf = np.fft.irfft(f * np.conj(f), n=n, axis=1)[:, : min(max_lag, T - 1) + 1]
    acf = acf.mean(axis=0)
    if noise_robust:
        if acf[1] <= 0:
            return 1
        r = acf / acf[1]
        below = np.nonzero(r[1:] < np.exp(-1))[0]
        return int(below[0] + 1) if len(below) else int(len(r) - 1)
    acf = acf / (acf[0] or 1.0)
    below = np.nonzero(acf < np.exp(-1))[0]
    return int(below[0]) if len(below) else int(len(acf) - 1)


def process_tau(x: np.ndarray) -> int:
    """The process time scale the benchmark works at: within-batch, noise-robust."""
    return decorrelation_steps(x, within_batch=True, noise_robust=True)


def reading(true: np.ndarray, share: float, scale, tau: float, seed: int) -> np.ndarray:
    """A sensor's reading with measurement error: share `share` of its variance is error
    drifting on time scale `tau` (process_noise), the rest the true value; same mean and
    spread as the true value (frozen `scale`), so units are kept."""
    mu, sd = scale
    err = process_noise(true.shape, tau, seed)
    return mu + sd * mix(err, standardise(true, scale), share)


def process_noise(shape, tau: float, seed: int) -> np.ndarray:
    """Unit-variance noise with the process's time scale: a stationary AR(1) per batch
    whose ACF falls to 1/e after `tau` steps (a = exp(-1/tau)), started from its
    stationary distribution. The unexplained part of a node (unmeasured causes) lives
    on the same time scale as the process, so averaging over time does not remove it
    the way it removes white noise."""
    rng = np.random.default_rng(int(seed))
    e = rng.standard_normal(shape)
    a = float(np.exp(-1.0 / max(float(tau), 1e-9)))
    b = np.sqrt(1.0 - a * a)
    y = np.empty_like(e)
    y[:, 0] = e[:, 0]
    for t in range(1, shape[1]):
        y[:, t] = a * y[:, t - 1] + b * e[:, t]
    return y
