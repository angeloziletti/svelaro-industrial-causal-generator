"""The generative parameters of a sensor trajectory.

Every sensor trajectory in a batch is

    x(t) = signal_strength * shape(t; onset, amplitude, duration) + level
           + GP(t; noise_amplitude, lengthscale) + interference(t)

`shape` is the sensor's motif (step, spike, ramp, baseline shift), the GP is the
sensor's texture. The parameters in `causal_params` (config
parameter_layer.causal_params) are nodes of the SCM: they have causes, and a
change in a cause changes the trajectory through them. The others are drawn at
random per batch, so they vary without carrying causal information. Which
subset is causal is the mean-level / kernel-level ablation:

    shape params   -> "mean" edges   (a cause moves level, timing, size, width)
    kernel params  -> "kernel" edges (a cause changes variability / roughness)

An SCM node value z (real line) maps monotonically to the physical range with a
sigmoid; lengthscale is interpolated on a log scale.
"""
from typing import Dict, Iterable, List, Optional

import numpy as np

PARAMETERS: Dict[str, Dict] = {
    # shape (mean function)
    "onset": {"kind": "shape", "lo": 0.10, "hi": 0.80},  # event start, fraction of the batch (= jitter clamp)
    "amplitude": {"kind": "shape", "lo": 0.5, "hi": 2.0},  # event size multiplier
    "duration": {"kind": "shape", "lo": 0.03, "hi": 0.08},  # event width, fraction of the batch
    # additive offset of the whole batch. +/-0.5: batches of one recipe sit at
    # similar levels (IndPenSim between-batch share of variance ~0.09; +/-1.5
    # gave 0.29)
    "level": {"kind": "shape", "lo": -0.5, "hi": 0.5},
    # kernel (GP residual)
    # GP texture: fast-but-smooth process variation. Calibrated against IndPenSim,
    # whose trajectories carry ~28% of their within-batch power above 5 cycles per
    # batch while staying smooth step to step (lag-1 autocorrelation 0.99).
    "noise_amplitude": {"kind": "kernel", "lo": 0.2, "hi": 1.0},  # GP standard deviation
    "lengthscale": {"kind": "kernel", "lo": 0.005, "hi": 0.05, "log": True},  # fraction of the batch
}
DEFAULT_CAUSAL = ["onset", "amplitude"]

# trajectory_model: motif_sequence (advanced_scm/motif_profiles.py). Each segment
# k of a sensor's profile has its own onset/duration/amplitude, named
# "s<k>_<name>", with ranges relative to the segment's time slot (ordered,
# non-overlapping segments). level and the kernel parameters stay per sensor.
SEGMENT_PARAMETERS: Dict[str, Dict] = {
    "onset": {"kind": "shape", "lo": 0.0, "hi": 0.5},  # start inside the slot
    "duration": {"kind": "shape", "lo": 0.2, "hi": 0.5},  # fraction of the slot
    "amplitude": {"kind": "shape", "lo": 0.5, "hi": 2.0},  # multiplier of the motif's base size
}
SENSOR_LEVEL_PARAMETERS = ["level", "noise_amplitude", "lengthscale"]

# Lengthscales snap to this grid so the GP covariance (and its Cholesky factor)
# repeats across batches and is cached; the snapped value is what is used.
_LS = PARAMETERS["lengthscale"]
LENGTHSCALE_GRID = np.geomspace(_LS["lo"], _LS["hi"], 48)


def node_name(sensor: str, param: str) -> str:
    return f"{sensor}_{param}"


def base_of(param: str) -> str:
    """"s2_onset" -> "onset"; sensor-level names are their own base."""
    head, _, rest = param.partition("_")
    if head.startswith("s") and head[1:].isdigit() and rest in SEGMENT_PARAMETERS:
        return rest
    return param


def spec(param: str) -> Dict:
    base = base_of(param)
    return SEGMENT_PARAMETERS[base] if base != param else PARAMETERS[param]


def param_of(node: str, sensor: str) -> Optional[str]:
    p = node[len(sensor) + 1 :] if node.startswith(sensor + "_") else None
    if p is None:
        return None
    try:
        spec(p)
    except KeyError:
        return None
    return p


def kind(param: str) -> str:
    return spec(param)["kind"]


def sensor_param_names(n_segments: Optional[int] = None) -> List[str]:
    """Every generative parameter of a sensor, in a fixed order."""
    if n_segments is None:
        return list(PARAMETERS)
    return [f"s{k}_{b}" for k in range(n_segments) for b in SEGMENT_PARAMETERS] + list(
        SENSOR_LEVEL_PARAMETERS
    )


def expand_causal(causal_params: Iterable[str], n_segments: Optional[int]) -> List[str]:
    """Causal base names -> this sensor's parameter names (one per segment for
    segment parameters)."""
    out = []
    for b in causal_params:
        if n_segments is not None and b in SEGMENT_PARAMETERS:
            out.extend(f"s{k}_{b}" for k in range(n_segments))
        else:
            out.append(b)
    return out


def validate(causal_params: Iterable[str]) -> List[str]:
    params = list(causal_params)
    unknown = [p for p in params if p not in PARAMETERS]
    if unknown:
        raise ValueError(
            f"Unknown parameter_layer.causal_params {unknown}; known: {list(PARAMETERS)}"
        )
    return params


def to_physical(param: str, z: float) -> float:
    """SCM node value -> physical value. A non-finite z is a bug upstream, not a
    value to replace with a default (that silently cut the causal link before)."""
    if not np.isfinite(z):
        raise ValueError(f"non-finite SCM value {z} for parameter '{param}'")
    sp = spec(param)
    u = 1.0 / (1.0 + np.exp(-float(z)))
    if sp.get("log"):
        return float(np.exp(np.log(sp["lo"]) + u * (np.log(sp["hi"]) - np.log(sp["lo"]))))
    return float(sp["lo"] + u * (sp["hi"] - sp["lo"]))


def random_physical(param: str, rng: np.random.Generator) -> float:
    """A non-causal parameter: uniform over its physical range (log-uniform for
    lengthscale), independent of everything else."""
    sp = spec(param)
    if sp.get("log"):
        return float(np.exp(rng.uniform(np.log(sp["lo"]), np.log(sp["hi"]))))
    return float(rng.uniform(sp["lo"], sp["hi"]))


def snap_lengthscale(value: float) -> float:
    return float(LENGTHSCALE_GRID[np.argmin(np.abs(np.log(LENGTHSCALE_GRID) - np.log(value)))])


def sensor_parameters(
    scm_sample: Dict[str, float],
    sensor: str,
    nodes: List[str],
    rng: np.random.Generator,
    n_segments: Optional[int] = None,
) -> Dict[str, float]:
    """All physical parameters of one sensor in one batch: SCM-driven where the
    sensor has a node for the parameter, random otherwise. n_segments=None is
    the single-pattern model; an int is the motif-sequence model."""
    causal = {param_of(n, sensor): n for n in nodes if param_of(n, sensor)}
    out = {}
    # fixed order: the random draws do not depend on which are causal
    for param in sensor_param_names(n_segments):
        draw = random_physical(param, rng)
        out[param] = to_physical(param, scm_sample[causal[param]]) if param in causal else draw
    out["lengthscale"] = snap_lengthscale(out["lengthscale"])
    return out
