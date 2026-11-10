"""Which SCM columns a method may see.

The generator keeps every SCM node value internally (the mechanisms need them),
but the exported data a benchmarked method reads must contain only what a plant
would record. One rule, used by both exports (the long-format timeseries and the
tabular dataset's static passthrough):

- event parameters (`gp_parameter` nodes, incl. `causal_event_time`): always hidden;
  they are the shape parameters a method should have to infer. Exported
  separately as oracle data.
- confounders (`shadow_confounder` nodes, Z_k): hidden unless
  `observability.confounders: true`.
- batch conditions (`static` nodes): observed unless
  `observability.static_attributes: false`.
- defect nodes are targets, never hidden here.

Sensor scale: exported sensor trajectories are z-scored per sensor over the
whole dataset unless `observability.standardize_sensors: false`. Otherwise a
sensor's marginal variance can betray its role (truth sensors carry extra
confounder signal), and ranking by variance finds the causes -- the
varsortability artifact of synthetic benchmarks (Reisach et al.).
"""
from typing import Dict, List, Optional

import numpy as np


def standardize_sensors(
    series: Dict[str, "np.ndarray"], sensors: List[str], observability: Optional[Dict] = None
):
    """Return (exported series, {sensor: (mean, sd)}): sensors z-scored over all
    samples and timesteps, everything else (e.g. defect_ts) unchanged."""
    if not (observability or {}).get("standardize_sensors", True):
        return series, {}
    out, scales = dict(series), {}
    for s in sensors:
        if s in series:
            mu, sd = float(np.mean(series[s])), float(np.std(series[s]))
            out[s] = (series[s] - mu) / (sd if sd > 0 else 1.0)
            scales[s] = [mu, sd]
    return out, scales


def hidden_scm_columns(
    node_types: Dict[str, str], observability: Optional[Dict] = None
) -> List[str]:
    cfg = observability or {}
    hidden_types = {"gp_parameter"}
    if not cfg.get("confounders", False):
        hidden_types.add("shadow_confounder")
    if not cfg.get("static_attributes", True):
        hidden_types.add("static")
    return sorted(n for n, t in node_types.items() if t in hidden_types)
