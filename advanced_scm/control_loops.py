"""Closed-loop control inside each batch (config `control_loops`).

A loop has three sensors:

    controlled  C  a truth sensor (a direct parent of defect_ts). Its generated
                   trajectory is the setpoint profile r(t): the planned
                   recipe trajectory its parameters describe.
    disturbance D  a non-cause sensor whose trajectory pushes C off its setpoint
                   through process dynamics -- dead time theta and a first-order
                   lag tau (both in timesteps; 0 = instant):
                   d(t) = disturbance_gain * FOPDT_{tau,theta}[x_D](t).
    actuator    A  a non-cause sensor: the valve. Its own trajectory x_A(t) is
                   the valve's drift; the controller adds its moves u(t).

Per batch, along time (discrete, one-step actuator delay):

    y(t) = r(t) + d(t) + a(t-1)             C as measured (closed loop)
    u(t) = u(t-1) + tightness * (r(t) - y(t))   integral action
    a(t) = x_A(t) + u(t)                    A as measured (valve position)

tightness (the integral gain, in [0, 1]) is the difficulty dial: 0 = manual (no
control: the valve drifts, C carries the full disturbance), small = loose, 1 =
tight (the controller removes the slow part of d from C, and that part shows
up in A instead). The closed loop is stable for 0 < tightness <= 1.

Causal structure this adds (variable level): D -> C, A -> C, and the feedback
C -> A when tightness > 0. D stays an ancestor of defect_ts, but control
cancels part of its effect -- the textbook faithfulness violation the
ground-truth labels flag.
"""
from typing import Dict, List

import numpy as np


def plan_control_loops(
    cfg: Dict, truths: List[str], candidates: List[str], rng: np.random.Generator
) -> List[Dict]:
    """Choose loop members (structure): each loop controls a distinct truth and
    uses two distinct candidate sensors (non-causes) as disturbance and actuator."""
    if not (cfg or {}).get("enabled", False):
        return []
    n = int(cfg.get("n_loops", 1))
    if n > len(truths) or 2 * n > len(candidates):
        raise ValueError(
            f"control_loops.n_loops={n} needs {n} truth sensors and {2 * n} non-cause "
            f"sensors; have {len(truths)} and {len(candidates)}"
        )
    tightness = float(cfg.get("tightness", 0.8))
    if not 0.0 <= tightness <= 1.0:
        raise ValueError(f"control_loops.tightness must be in [0, 1], got {tightness}")
    controlled = [str(s) for s in rng.choice(truths, size=n, replace=False)]
    others = [str(s) for s in rng.choice(candidates, size=2 * n, replace=False)]
    return [
        {
            "controlled": controlled[i],
            "disturbance": others[2 * i],
            "actuator": others[2 * i + 1],
            "tightness": tightness,
            "disturbance_gain": float(cfg.get("disturbance_gain", 1.0)),
            "disturbance_time_constant": float(cfg.get("disturbance_time_constant", 0.0)),
            "disturbance_dead_time": int(cfg.get("disturbance_dead_time", 0)),
        }
        for i in range(n)
    ]


def fopdt(x: np.ndarray, time_constant: float, dead_time: int) -> np.ndarray:
    """First-order-plus-dead-time response of (n_samples, T) inputs along time:
    delay by dead_time steps (holding the first value), then
    y(t) = y(t-1) + (u(t) - y(t-1)) / time_constant. time_constant <= 1 and
    dead_time 0 return the input unchanged."""
    u = x
    if dead_time > 0:
        u = np.concatenate([np.repeat(x[:, :1], dead_time, axis=1), x[:, :-dead_time]], axis=1)
    if time_constant <= 1.0:
        return u.copy() if u is x else u
    y = np.empty_like(u)
    y[:, 0] = u[:, 0]
    for t in range(1, u.shape[1]):
        y[:, t] = y[:, t - 1] + (u[:, t] - y[:, t - 1]) / time_constant
    return y


def apply_control_loops(series: Dict[str, np.ndarray], loops: List[Dict]) -> Dict[str, np.ndarray]:
    """Close the loops on open-loop (n_samples, T) trajectories. Returns a new dict;
    only the controlled and actuator series change."""
    out = dict(series)
    for loop in loops or []:
        r = series[loop["controlled"]]
        d = loop["disturbance_gain"] * fopdt(
            series[loop["disturbance"]],
            loop.get("disturbance_time_constant", 0.0),
            loop.get("disturbance_dead_time", 0),
        )
        x_a = series[loop["actuator"]]
        k = loop["tightness"]
        n, T = r.shape
        y = np.empty_like(r)
        a = np.empty_like(r)
        u = np.zeros(n)
        a_prev = x_a[:, 0]
        for t in range(T):
            y[:, t] = r[:, t] + d[:, t] + a_prev
            u = u + k * (r[:, t] - y[:, t])
            a[:, t] = x_a[:, t] + u
            a_prev = a[:, t]
        out[loop["controlled"]] = y
        out[loop["actuator"]] = a
    return out
