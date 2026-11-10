"""Dynamic sensor -> sensor edges (route 3): a child follows its parent's
trajectory over time, delayed and smoothed.

    echo(t) = sum over parents of gain * FOPDT_{time_constant, dead_time}[parent](t)

and the child mixes its own curve with the echo at the echo share
(vector_scm.shares.echo, apply_dynamic_edges_shared).

Every movement of the parent -- events, drifts, small wiggles -- reaches the
child a few steps later, so the dependence is visible at every timestep of the
batch, not only at event moments. Example: jacket temperature -> reactor
temperature. Edges are drawn forward in the sensor order (upstream -> truths ->
others), so the coupling is acyclic and the target's roles are preserved:
upstream sensors become causes the truths mediate, and sensors downstream of a
truth become its echoes -- correlated with the target, causing nothing.
"""
from typing import Dict, List

import numpy as np

from advanced_scm.control_loops import fopdt


def plan_dynamic_edges(order: List[str], cfg: Dict, rng: np.random.Generator) -> List[Dict]:
    """Forward edges with probability 2 * expected_in_degree / (n - 1)."""
    if not (cfg or {}).get("enabled", False):
        return []
    n = len(order)
    p = min(1.0, 2.0 * float(cfg.get("expected_in_degree", 1.0)) / max(n - 1, 1))
    g_lo, g_hi = cfg.get("gain", [0.3, 0.8])
    tc_lo, tc_hi = cfg.get("time_constant", [3, 20])
    dt_lo, dt_hi = cfg.get("dead_time", [0, 5])
    edges = []
    for j in range(n):
        for i in range(j):
            if rng.random() >= p:
                continue
            a, b = order[i], order[j]
            edges.append(
                {
                    "source": a,
                    "target": b,
                    "gain": float(rng.choice([-1, 1]) * rng.uniform(g_lo, g_hi)),
                    "time_constant": float(rng.uniform(tc_lo, tc_hi)),
                    "dead_time": int(rng.integers(dt_lo, dt_hi + 1)),
                }
            )
    return edges


def apply_dynamic_edges_shared(series, edges, order, spec, seed, frozen=None):
    """Couple (n_samples, T) trajectories along the edges, parents first, so a child
    receives its parents' already-coupled trajectories. Echo share (advanced_scm/shares.py): a child with
    dynamic parents becomes sqrt(s) * echo_hat + sqrt(1 - s) * own_hat, where echo is
    the gain-weighted sum of its parents' delayed, smoothed curves and own is the
    child's curve before coupling; both standardised with scales frozen on the
    baseline (returned for the next call)."""
    from advanced_scm import shares

    out = dict(series)
    frozen = dict(frozen or {})
    for i, child in enumerate(order):
        mine = [e for e in edges if e["target"] == child]
        if not mine:
            continue
        echo = sum(e["gain"] * fopdt(out[e["source"]], e["time_constant"], e["dead_time"]) for e in mine)
        s = shares.draw(spec, seed, "echo", i)
        f = frozen.setdefault(child, {"own": shares.scale_of(out[child]), "echo": shares.scale_of(echo), "share": s})
        out[child] = shares.mix(shares.standardise(echo, f["echo"]), shares.standardise(out[child], f["own"]), s)
    return out, frozen


