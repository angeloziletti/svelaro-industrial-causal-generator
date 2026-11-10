"""
Vector-SCM defect generator: produces the `defect_ts` time series from the
truth-sensor time series, as a node of the time-series SCM
(advanced_scm/target_edges.py) mixed with unexplained noise at the configured
share (advanced_scm/shares.py).

This is the bridge between the static SCM's sensor time series and the
time-series defect column: the static SCM owns the sensor nodes and their
parameters; `defect_ts` is added here, from the realised trajectories.

Call site: `time_series/unified_temporal_processor.py` Step 2.4, after
the synthetic pipeline has produced `structured_time_series` for all
truth sensors.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from advanced_scm.unified_config import VectorSCMConfig


# Offset added to the seed to derive the noise RNG seed.
# Picked as a prime so it doesn't collide with the per-sample pattern seed
# offset (sample_idx * 1000) at
# time_series/generation/temporal_generator.py:132.
EDGE_PARAMS_SEED_OFFSET = 7919


def generate_defect_ts(
    cfg: VectorSCMConfig,
    structured_time_series: Dict[str, np.ndarray],
    truth_sensors: List[str],
    seed: int,
    noise_seed: Optional[int] = None,
    parent_scales: Optional[Dict[str, List[float]]] = None,
    signal_scale: Optional[List[float]] = None,
    noise_tau: Optional[float] = None,
    part_scales: Optional[Dict[str, List[float]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Produce `defect_ts` shape (n_samples, T) from truth-sensor time series.

    Args:
        cfg: vector-SCM config block (edge model, dynamic_target ranges, shares).
        structured_time_series: dict mapping sensor name → array of shape
            (n_samples, T). Must contain every name in `truth_sensors`.
        truth_sensors: ordered list of sensor names that are direct parents
            of the defect node. Order matters: edge_params[i] applies to
            truth_sensors[i]. Caller-derived (typically via
            TabularDatasetGenerator.derive_causal_sensors).
        seed: structure seed: the cause -> defect edges and the shares are
            drawn from it. Edge params are structure: pass the graph seed.
        noise_seed: data seed for the observation noise on defect_ts. None =
            derived from `seed` (the old behaviour).
        parent_scales: {sensor: [mean, sd]} for cfg.dataset_standardize_parents.
            None = computed from these parents (over all batches and timesteps).
            Pass the baseline's scales when replaying an intervention, so the
            counterfactual does not rescale every other batch.

    Returns:
        (defect_ts, audit) — defect_ts shape (n_samples, T); audit is a
        JSON-serialisable dict describing the cause -> defect edges, the
        shares and the frozen scales. Intended for inclusion in
        dataset_metadata.json.

    Raises:
        ValueError: if truth_sensors is empty, if any name is missing from
            structured_time_series, if shapes are inconsistent, or if the
            config selects a defect model that is not part of this release.
    """
    if not truth_sensors:
        raise ValueError(
            "vector-SCM defect generation requires at least one truth sensor; "
            "got an empty list. Check that the static SCM has at least one "
            "GP-parameter node with an edge to a defect node."
        )

    missing = [s for s in truth_sensors if s not in structured_time_series]
    if missing:
        raise ValueError(
            f"truth sensors not present in structured_time_series: {missing}. "
            f"Available keys (first 10): {list(structured_time_series.keys())[:10]}"
        )

    parents: List[np.ndarray] = [structured_time_series[s] for s in truth_sensors]
    # Dataset-wide standardization: every parent enters the mechanism on a unit
    # scale, so a change that rescales a sensor (e.g. a control loop removing a
    # disturbance) does not move the nonlinearities into another regime. Unlike
    # standardize_parents (per sample), batch-to-batch differences in level and
    # size are kept.
    used_scales: Dict[str, List[float]] = {}
    if getattr(cfg, "dataset_standardize_parents", False):
        for i, s in enumerate(truth_sensors):
            mu, sd = (parent_scales or {}).get(s) or (
                float(np.mean(parents[i])), float(np.std(parents[i]))
            )
            used_scales[s] = [mu, sd]
            parents[i] = (parents[i] - mu) / (sd if sd > 0 else 1.0)
    shapes = {s: p.shape for s, p in zip(truth_sensors, parents)}
    first_shape = parents[0].shape
    if not all(p.shape == first_shape for p in parents):
        raise ValueError(
            f"truth-sensor time series have inconsistent shapes: {shapes}"
        )

    edge_seed = int(seed) + EDGE_PARAMS_SEED_OFFSET

    # Noise is data: seeded from the data seed when given, so realizations of
    # one graph share edge params but not noise.
    noise_rng_seed = (
        edge_seed + 1
        if noise_seed is None
        else int(noise_seed) + EDGE_PARAMS_SEED_OFFSET + 1
    )
    # ---- Defect as a node of the time-series SCM ----------------------
    if getattr(cfg, "edge_model", "operators") != "dynamic":
        raise ValueError(
            f"vector_scm.edge_model must be 'dynamic' (the operator models are not part of this "
            f"release), got {getattr(cfg, 'edge_model', None)!r}"
        )
    from advanced_scm import shares
    from advanced_scm.target_edges import dynamic_target, plan_target_edges

    plan = plan_target_edges(list(truth_sensors), getattr(cfg, "dynamic_target", None) or {}, seed)
    # Realism knobs on the cause -> defect edges (target_edges.cause_response):
    # per-batch delay jitter (fraction of each cause's time scale) and the cumulative
    # (dose) share of each cause's effect, drawn per cause.
    sh = getattr(cfg, "shares", None) or {}
    if sh.get("delay_jitter"):
        plan = {**plan, "delay_jitter": float(sh["delay_jitter"]), "jitter_seed": int(seed) + 9311}
    if sh.get("context"):
        # effect switched by a recorded batch attribute (grade); one sign per cause
        rs = np.random.default_rng(int(seed) + 9419)
        plan = {**plan, "context": float(sh["context"]), "context_seed": int(seed) + 9421,
                "edges": {s: {**e, "context_sign": float(rs.choice([-1.0, 1.0]))} for s, e in plan["edges"].items()}}
    if sh.get("cumulative") is not None:
        plan = {**plan, "edges": {s: {**e, "cumulative_share": shares.draw(sh["cumulative"], seed, "cumulative", i)}
                                  for i, (s, e) in enumerate(plan["edges"].items())}}
    share = shares.draw((getattr(cfg, "shares", None) or {}).get("defect_timeseries"), seed, "defect_timeseries")
    if share is None:
        raise ValueError("vector_scm.shares.defect_timeseries is required (the causes' share of the defect)")
    # Interaction share (advanced_scm/shares.py): the causes act through products that
    # cover every cause, carrying this share of the causes' signal. A plant with one
    # cause has no pair (share recorded as 0).
    inter_share = shares.draw((getattr(cfg, "shares", None) or {}).get("interaction"), seed, "interaction")
    if inter_share is not None:
        from advanced_scm.target_edges import plan_interactions

        plan = {**plan, "interaction": None, "interactions": plan_interactions(list(truth_sensors), seed)}
        if not plan["interactions"]:
            inter_share = None
    # Explained-variance share (advanced_scm/shares.py): the causes' signal,
    # standardised on the baseline (frozen scale), carries `share` of the
    # defect's variance; unit noise carries the rest.
    signal, dyn_audit = dynamic_target(parents, list(truth_sensors), plan,
                                       interaction_share=inter_share, part_scales=part_scales)
    sc = signal_scale or shares.scale_of(signal)
    unexplained = (getattr(cfg, "shares", None) or {}).get("unexplained", "white")
    if unexplained == "process":
        # Same time scale as the causes' signal (frozen on the baseline, like
        # its scale), so the share holds at any time resolution.
        noise_tau = noise_tau or float(max(shares.decorrelation_steps(signal), 1))
        noise = shares.process_noise(signal.shape, noise_tau, noise_rng_seed)
    elif unexplained == "white":
        noise = np.random.default_rng(noise_rng_seed).standard_normal(signal.shape)
    else:
        raise ValueError(f"shares.unexplained must be 'white' or 'process', got {unexplained!r}")
    defect_ts = shares.mix(shares.standardise(signal, sc), noise, share)
    extra = {"share_defect_timeseries": share, "signal_scale": sc, "unexplained": unexplained,
             "noise_tau": noise_tau if unexplained == "process" else None}
    return defect_ts, {
        "enabled": True, "version": "v3_dynamic", "defect_operator": "DYNAMIC",
        "noise_sigma": float(cfg.noise_sigma), "parent_scales": used_scales,
        "truth_sensors": list(truth_sensors), **dyn_audit, **extra,
    }
