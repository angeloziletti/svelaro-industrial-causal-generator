"""The defect as a node of the time-series SCM (vector_scm.edge_model: dynamic).

Each cause reaches the defect through the same kind of edge that couples sensors
(advanced_scm/sensor_dynamics.py), followed by a static nonlinearity -- the
Hammerstein-Wiener block of process identification:

    defect(t) = sum_k w_k * g_k( FOPDT_{tau_k, theta_k}[x_k](t) ) / sqrt(n_causes)

    theta_k   dead time: the cause reaches the quality point later (transport)
    tau_k     time constant: the quality responds through a mixing volume
    g_k       nonlinearity, drawn from NONLINEARITIES:
                linear      proportional
                saturation  tanh: the effect levels off
                optimum     x^2 - 1: quality is best at a setpoint, worse either side
                threshold   max(0, x - 1): no effect inside the window, then proportional

With vector_scm.shares.interaction, products f_i * f_j of filtered causes that cover
every cause (temperature x concentration) carry that share of the causes' signal.
Every parameter is drawn per plant from the structure seed, like the sensor
graph's dynamic edges. Parents enter on a dataset-wide unit scale (frozen from
the baseline, so interventions do not rescale other batches).
"""
from typing import Any, Dict, List, Tuple

import numpy as np

from advanced_scm.control_loops import fopdt

NONLINEARITIES = {
    "linear": lambda x: x,
    "saturation": np.tanh,
    "optimum": lambda x: x ** 2 - 1.0,
    "threshold": lambda x: np.maximum(0.0, x - 1.0),
}

DEFAULTS = {
    "time_constant": [1.0, 30.0],
    "dead_time": [0, 15],
    "gain": [0.6, 1.2],
    "nonlinearities": {"linear": 1.0, "saturation": 1.0, "optimum": 1.0, "threshold": 1.0},
    # The old random single interaction: must stay 0 (interactions come from
    # vector_scm.shares.interaction).
    "interaction_probability": 0.0,
}


def plan_target_edges(truths: List[str], cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Structure: one edge per cause."""
    c = {**DEFAULTS, **(cfg or {})}
    rng = np.random.default_rng(int(seed) + 6829)
    names = list(c["nonlinearities"])
    p = np.array([c["nonlinearities"][n] for n in names], float)
    p = p / p.sum()
    edges = {}
    for s in truths:
        edges[s] = {
            "time_constant": float(rng.uniform(*c["time_constant"])),
            "dead_time": int(rng.integers(int(c["dead_time"][0]), int(c["dead_time"][1]) + 1)),
            "gain": float(rng.choice([-1, 1]) * rng.uniform(*c["gain"])),
            "nonlinearity": str(rng.choice(names, p=p)),
        }
    if float(c["interaction_probability"]) != 0.0:
        raise ValueError(
            "dynamic_target.interaction_probability is not part of this release (must be 0); "
            "use vector_scm.shares.interaction"
        )
    return {"edges": edges, "interaction": None}


def cover_pairs(truths: List[str]) -> List[List[str]]:
    """Interaction pairs that involve every cause (so with interactions only, each cause still
    acts on the defect): 2 causes a-b; 3 a-b, b-c; 4 a-b, c-d; one cause has no pair."""
    t = list(truths)
    if len(t) < 2:
        return []
    if len(t) == 3:
        return [[t[0], t[1]], [t[1], t[2]]]
    return [[t[i], t[i + 1]] for i in range(0, len(t) - 1, 2)] + ([[t[-2], t[-1]]] if len(t) % 2 else [])


def plan_interactions(truths: List[str], seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + 7417)
    return [{"pair": p, "gain": float(rng.choice([-1, 1]) * rng.uniform(0.6, 1.2))} for p in cover_pairs(truths)]


def cause_response(x: np.ndarray, s: str, e: Dict[str, Any], plan: Dict[str, Any],
                   ps: Dict[str, Any]) -> np.ndarray:
    """One cause's filtered trajectory as the defect reads it: dead time, first-order lag,
    and two optional realism knobs (both off reproduce FOPDT exactly):

      delay jitter      plan["delay_jitter"] = j: each batch adds its own extra hold-up of
                        up to j * tau_cause steps to the dead time (flow rate varies batch
                        to batch), so a fixed-lag method sees the effect smeared over lags
      cumulative share  e["cumulative_share"] = k: share k of the cause's (unit-variance)
                        response is its dose -- the running mean over the last 2 * tau_cause
                        steps -- the rest the instantaneous response
      context           plan["context"] = c: each batch has a recorded grade g = +-1
                        (batch_grades) and each cause a sign; the response is scaled by
                        (1 + c * sign * g) / sqrt(1 + c^2): at c = 1 a cause acts only in
                        the grade its sign picks, so pooling batches dilutes it

    tau_cause (process_tau of the cause), the dose window and the scales are frozen in `ps`
    on the baseline, so an intervention changes only what it touches."""
    from advanced_scm import shares

    j = float(plan.get("delay_jitter") or 0.0)
    k = e.get("cumulative_share")
    if j > 0 or k:
        tau = ps.setdefault(f"tau::{s}", float(max(shares.process_tau(x), 1)))
    if j > 0:
        n_b, T = x.shape
        rng = np.random.default_rng(int(plan["jitter_seed"]) + 7907 * sorted(plan["edges"]).index(s))
        d = int(e["dead_time"]) + np.rint(j * tau * rng.uniform(0.0, 1.0, n_b)).astype(int)
        idx = np.clip(np.arange(T)[None, :] - d[:, None], 0, None)
        f = fopdt(np.take_along_axis(x, idx, axis=1), e["time_constant"], 0)
    else:
        f = fopdt(x, e["time_constant"], e["dead_time"])
    if k:
        w = int(ps.setdefault(f"window::{s}", int(max(round(2 * tau), 1))))
        c = np.cumsum(f, axis=1)
        dose = c.copy()
        dose[:, w:] = c[:, w:] - c[:, :-w]
        dose = dose / np.minimum(np.arange(1, f.shape[1] + 1), w)[None, :]
        cs = ps.setdefault(f"dose::{s}", shares.scale_of(dose))
        fs = ps.setdefault(f"direct::{s}", shares.scale_of(f))
        f = shares.mix(shares.standardise(dose, cs), shares.standardise(f, fs), float(k))
    c = float(plan.get("context") or 0.0)
    if c > 0:
        g = batch_grades(plan, f.shape[0])
        f = f * ((1.0 + c * e["context_sign"] * g) / np.sqrt(1.0 + c * c))[:, None]
    return f


def batch_grades(plan: Dict[str, Any], n_batches: int) -> np.ndarray:
    """The recorded grade (+1 / -1) of every batch for the context knob."""
    return np.where(np.random.default_rng(int(plan["context_seed"])).random(n_batches) < 0.5, 1.0, -1.0)


def realism_delays(plan: Dict[str, Any], ps: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Per cause, the delay terms the realism knobs add (for the look-back check):
    the largest extra hold-up and the dose window."""
    j = float(plan.get("delay_jitter") or 0.0)
    out = {}
    for s, e in plan["edges"].items():
        tau = (ps or {}).get(f"tau::{s}")
        out[s] = {"delay_extra_max": float(round(j * tau)) if j > 0 and tau else 0.0,
                  "cumulative_window": float((ps or {}).get(f"window::{s}") or 0.0) if e.get("cumulative_share") else 0.0}
    return out


def dynamic_target(
    parents: List[np.ndarray], truths: List[str], plan: Dict[str, Any],
    interaction_share: float = None, part_scales: Dict[str, List[float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """(n_batches, T) noise-free defect signal from unit-scale parent trajectories.

    With `interaction_share` s (plan["interactions"] covering every cause): the main effects
    and the interaction products are each standardised (scales frozen on the baseline via
    `part_scales`) and mixed as sqrt(s) * interactions + sqrt(1 - s) * main effects, so the
    interactions carry share s of the causes' signal; s = 1 leaves interactions only."""
    knob_scales = dict(part_scales or {})
    filtered = {s: cause_response(x, s, plan["edges"][s], plan, knob_scales) for s, x in zip(truths, parents)}
    knobs = {"part_scales": knob_scales, "realism_delays": realism_delays(plan, knob_scales)} if knob_scales else {}
    if interaction_share is not None and plan.get("interactions"):
        from advanced_scm import shares

        main = sum(plan["edges"][s]["gain"] * NONLINEARITIES[plan["edges"][s]["nonlinearity"]](filtered[s])
                   for s in truths) / np.sqrt(len(truths))
        inter = sum(i["gain"] * filtered[i["pair"][0]] * filtered[i["pair"][1]]
                    for i in plan["interactions"]) / np.sqrt(len(plan["interactions"]))
        ps = {**knob_scales, **(part_scales or {"main": shares.scale_of(main), "interaction": shares.scale_of(inter)})}
        out = shares.mix(shares.standardise(inter, ps["interaction"]), shares.standardise(main, ps["main"]),
                         float(interaction_share))
        return out, {"edge_model": "dynamic", **plan, **knobs, "interaction_share": float(interaction_share),
                     "part_scales": ps}
    out = np.zeros_like(parents[0], dtype=float)
    n_terms = 0
    for s in truths:
        e = plan["edges"][s]
        out += e["gain"] * NONLINEARITIES[e["nonlinearity"]](filtered[s])
        n_terms += 1
    out = out / np.sqrt(max(n_terms, 1))
    return out, {"edge_model": "dynamic", **plan, **knobs}


def plan_consequences(
    candidates: List[str], cfg: Dict[str, Any], seed: int,
) -> List[Dict[str, Any]]:
    """Sensors downstream of the defect (structure): each receives the defect
    through its own dynamic edge. A consequence moves with the defect and causes
    nothing -- a method that names it has the arrow backwards."""
    if not cfg or not candidates:
        return []
    rng = np.random.default_rng(int(seed) + 4789)
    n = cfg.get("n", 2)
    n = int(rng.integers(int(n[0]), int(n[1]) + 1)) if isinstance(n, (list, tuple)) else int(n)
    n = min(n, len(candidates))
    g_lo, g_hi = cfg.get("gain", [0.5, 1.0])
    tc_lo, tc_hi = cfg.get("time_constant", [3.0, 20.0])
    dt_lo, dt_hi = cfg.get("dead_time", [0, 10])
    # dead_time_tau: [lo, hi] -- the delay as a multiple of the defect's process time scale,
    # resolved when the defect exists (apply time); takes precedence over dead_time.
    tau_rng = cfg.get("dead_time_tau")
    chosen = sorted(str(s) for s in rng.choice(sorted(candidates), size=n, replace=False))
    return [
        {
            "sensor": s,
            "gain": float(rng.choice([-1, 1]) * rng.uniform(g_lo, g_hi)),
            "time_constant": float(rng.uniform(tc_lo, tc_hi)),
            "dead_time": int(rng.integers(int(dt_lo), int(dt_hi) + 1)),
            **({"dead_time_factor": float(rng.uniform(float(tau_rng[0]), float(tau_rng[1])))} if tau_rng else {}),
        }
        for s in chosen
    ]


def apply_consequences(
    series: Dict[str, np.ndarray], defect: np.ndarray, consequences: List[Dict[str, Any]],
    scale: List[float], frozen: Dict[str, Dict[str, Any]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """Add each consequence's response to the (unit-scale) true defect.

    Each consequence carries a share s (vector_scm.shares.consequence): the
    sensor becomes sqrt(s) * echo_hat + sqrt(1 - s) * own_hat, both standardised
    with scales frozen on the baseline (`frozen`, returned for the next call), so
    s is the fraction of the consequence's variance that is the defect's echo."""
    from advanced_scm import shares

    mu, sd = scale
    z = (defect - mu) / (sd if sd > 0 else 1.0)
    out = dict(series)
    frozen = dict(frozen or {})
    for c in consequences:
        echo = fopdt(z, c["time_constant"], c["dead_time"])
        s = c.get("share")
        if s is None:
            raise ValueError("consequences require vector_scm.shares.consequence")
        f = frozen.setdefault(c["sensor"], {
            "own": shares.scale_of(out[c["sensor"]]), "echo": shares.scale_of(echo),
        })
        out[c["sensor"]] = shares.mix(
            np.sign(c["gain"]) * shares.standardise(echo, f["echo"]),
            shares.standardise(out[c["sensor"]], f["own"]), s,
        )
    return out, frozen


def unique_shares(curves: Dict[str, np.ndarray], defect: np.ndarray, plan: Dict[str, Any]) -> Dict[str, float]:
    """Unique share of the defect per true cause: the drop in R^2 when the cause's
    terms (its own dead time, lag and nonlinearity; an interaction counts towards both
    of its causes) are left out of the true defect equation, refit by least squares.
    curves: (n_batches, T) per cause; defect: (n_batches, T); plan: edges (+ interaction)."""

    def r2(cols, y):
        X = np.column_stack([np.ones(len(y))] + cols)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(1.0 - np.var(y - X @ b) / np.var(y))

    y = np.asarray(defect, dtype=float).ravel()
    filt, terms = {}, {}
    for s, e in plan["edges"].items():
        z = curves[s]
        filt[s] = cause_response((z - z.mean()) / (z.std() or 1.0), s, e, plan, dict(plan.get("part_scales") or {}))
        terms[s] = NONLINEARITIES[e["nonlinearity"]](filt[s]).ravel()
    pairs = [i["pair"] for i in plan.get("interactions") or []]
    for k, (a, b) in enumerate(pairs):
        terms[f"__interaction_{k}__"] = (filt[a] * filt[b]).ravel()
    full = r2(list(terms.values()), y)
    out = {}
    for s in plan["edges"]:
        drop = {s} | {f"__interaction_{k}__" for k, p in enumerate(pairs) if s in p}
        out[s] = max(full - r2([v for k, v in terms.items() if k not in drop], y), 0.0)
    return out
