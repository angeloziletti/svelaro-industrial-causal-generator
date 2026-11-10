"""Interventional audit of a generated realization's ground truth.

The labels in dataset_metadata.json (target_driver_labels) are derived from the
generator's graph. This audit checks them against the generator's BEHAVIOUR:
it rebuilds the realization from its config and seeds, replays it, and runs
counterfactuals -- the same noise, one variable changed -- measuring how much
defect_ts moves.

For every observed input X:
  sensor        do(X := X + perturbation) on its final trajectory, and
                do(p := p + sd(p)) on each of its event parameters (timing,
                magnitude), propagated through the SCM
  static        do(X := X + sd(X)), propagated through the SCM

Effect = RMS(defect_ts' - defect_ts) / SD(defect_ts). Pass criteria:
  replay              the replayed defect_ts equals the exported one
  direct_parent       some effect > 0 (reported as weak below min_effect)
  indirect_ancestor   some effect > 0 (reported as weak below min_effect)
  non_cause           every effect exactly 0
  every sensor        each of its parameter nodes changes its own trajectory
                      (a parameter that does nothing is a false edge)
"""
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

ZERO_TOL = 1e-12


def _load(realization_dir: Path):
    from advanced_scm.unified_config import UnifiedSCMConfig

    meta = json.loads(
        (realization_dir / "datasets" / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    experiment_root = realization_dir.parents[2]
    config = UnifiedSCMConfig.from_yaml(str(experiment_root / "experiment_config.yaml"))
    used = meta["configuration_used"]
    config.seed = int(used["seed"])
    config.graph_seed = int(used["graph_seed"])
    return meta, config


class _Replay:
    """Rebuilds one realization exactly as TabularDatasetGenerator.generate_dataset did."""

    def __init__(self, config, workdir: Path, sensor_n_jobs: int = -1):
        from advanced_scm.tabular_dataset_generator import TabularDatasetGenerator

        # Sensors are generated independently with per-(sample, sensor) seeds, so
        # parallel generation gives identical series -- only faster.
        config.sensor_n_jobs = sensor_n_jobs
        config.write_validation_exports = False  # replays write no side-outputs
        self.config = config
        self.workdir = workdir
        np.random.seed(config.structure_seed())
        self.scm = TabularDatasetGenerator(str(workdir / "tdg"))._initialize_scm(config)
        if config.auto_select_temporal_sensors and self.scm.gp_parameter_nodes:
            config.temporal_sensors = list(self.scm.gp_parameter_nodes.keys())

    def defect_ts(
        self,
        intervention: Optional[Dict[str, Any]] = None,
        open_loop_overrides: Optional[Dict[str, Any]] = None,
    ):
        from time_series.unified_temporal_processor import UnifiedTemporalProcessor

        np.random.seed(self.config.seed)
        utp = UnifiedTemporalProcessor(
            unified_config=copy.deepcopy(self.config), export_dir=self.workdir / "utp"
        )
        scm_df = self.scm.generate_dataset(
            self.config.n_samples, apply_post_processing=True, intervention=intervention
        )
        series, _, _ = utp.simulate_trajectories(
            self.scm, scm_df, self.config.n_samples, open_loop_overrides=open_loop_overrides
        )
        return series["defect_ts"], scm_df, series


def _effect(changed: np.ndarray, base: np.ndarray) -> float:
    return float(np.sqrt(np.mean((changed - base) ** 2)) / np.std(base))


def _variable_effects(replay, var, entry, base, series_var, scm_df):
    """Interventions on one variable: (effects on the defect, each parameter's
    effect on its own sensor)."""
    effects: Dict[str, float] = {}
    on_sensor: Dict[str, float] = {}  # a parameter's effect on its own sensor
    if entry["kind"] == "sensor":
        sd = float(np.std(series_var))
        noise = np.random.default_rng(0).normal(0.0, sd, size=series_var.shape)
        # A full replay, so the perturbation also propagates through
        # any control loop the sensor is part of.
        effects["trajectory"] = _effect(
            replay.defect_ts(open_loop_overrides={var: lambda a, n=noise: a + n})[0],
            base,
        )
        for p in replay.scm.gp_parameter_nodes.get(var, []):
            shift = float(np.nanstd(scm_df[p])) or 1.0
            changed, _, changed_series = replay.defect_ts(
                intervention={p: lambda v, s=shift: v + s}
            )
            effects[p] = _effect(changed, base)
            on_sensor[p] = _effect(changed_series[var], series_var)
    else:
        shift = float(np.nanstd(scm_df[var])) or 1.0
        effects["value"] = _effect(
            replay.defect_ts(intervention={var: lambda v, s=shift: v + s})[0],
            base,
        )
    return effects, on_sensor


def _variable_worker(realization_dir, var, entry, base, series_var, scm_df):
    """One variable's interventions in its own process: its own replay,
    sensors generated sequentially (no nested worker pools)."""
    _, config = _load(Path(realization_dir))
    with tempfile.TemporaryDirectory() as tmp:
        replay = _Replay(config, Path(tmp), sensor_n_jobs=1)
        # The baseline first: the generator freezes its scales (parent scales,
        # consequence scale, dose thresholds) on its first computation, so an
        # intervention must never be a fresh replay's first call. Effects are
        # measured against THIS process's baseline: a worker with one BLAS thread
        # rounds the last bits differently from the main process (~1e-9 SD), and a
        # non-cause must show exactly zero effect, so baseline and intervention
        # must come from the same arithmetic. The worker's baseline must still
        # agree with the main one to the export tolerance.
        own_base, _, own_series = replay.defect_ts()
        if not np.allclose(own_base, base, rtol=0, atol=1e-6 * float(np.std(base))):
            raise RuntimeError(
                f"audit worker for {var}: baseline replay differs from the main replay by "
                f"{float(np.max(np.abs(own_base - base))) / float(np.std(base)):.2e} SD"
            )
        own_var = own_series[var] if series_var is not None else None
        return _variable_effects(replay, var, entry, own_base, own_var, scm_df)


def audit_realization(
    realization_dir: Path, min_effect: float = 0.01, sensor_n_jobs: int = -1
) -> Dict:
    """sensor_n_jobs: worker processes, one variable's interventions each
    (-1 = all cores). Every replay inside a worker is sequential."""
    realization_dir = Path(realization_dir)
    meta, config = _load(realization_dir)
    labels = meta["target_driver_labels"]["labels"]

    with tempfile.TemporaryDirectory() as tmp:
        replay = _Replay(config, Path(tmp), sensor_n_jobs=1)
        base, scm_df, series = replay.defect_ts()

        exported = pd.read_parquet(
            realization_dir / "full_timeseries.parquet", columns=["defect_ts"]
        )["defect_ts"].to_numpy().reshape(base.shape)
        # Exact up to float rounding: the cached Cholesky factor is unique but its
        # last bits depend on the BLAS thread count, so a parallel generation and
        # a sequential replay agree to ~1e-8, not bit for bit.
        replay_ok = bool(np.allclose(exported, base, rtol=0, atol=1e-6 * float(np.std(base))))

        from joblib import Parallel, delayed

        # Every variable's interventions are independent: one worker process per
        # variable, each replaying sequentially.
        names = list(labels)
        computed = Parallel(n_jobs=sensor_n_jobs)(
            delayed(_variable_worker)(
                str(realization_dir), var, labels[var], base,
                series[var] if labels[var]["kind"] == "sensor" else None, scm_df,
            )
            for var in names
        )
        results: Dict[str, Dict] = {}
        for var, (effects, on_sensor) in zip(names, computed):
            entry = labels[var]
            label = entry["label"]
            biggest = max(effects.values())
            if label in ("direct_parent", "indirect_ancestor"):
                # A direct cause acts through its trajectory (trajectory target),
                # through its parameters (batch-level target), or both. Any effect
                # confirms the label; a faint cause is a hard one, not a wrong one.
                ok = biggest > ZERO_TOL
            else:
                ok = biggest <= ZERO_TOL
            # Every parameter node claims an edge into its sensor: it must move it.
            dead_params = sorted(p for p, e in on_sensor.items() if e <= ZERO_TOL)
            ok = ok and not dead_params
            results[var] = {
                "label": label,
                "effects": {k: round(v, 6) for k, v in effects.items()},
                "effect_on_sensor": {k: round(v, 6) for k, v in on_sensor.items()},
                "dead_params": dead_params,
                "passed": bool(ok),
                "weak": bool(label in ("direct_parent", "indirect_ancestor") and biggest < min_effect),
            }

    failed = sorted(v for v, r in results.items() if not r["passed"])
    report = {
        "realization": str(realization_dir),
        "min_effect": min_effect,
        "replay_matches_export": replay_ok,
        "passed": replay_ok and not failed,
        "failed": failed,
        "weak_direct_causes": sorted(
            v for v, r in results.items() if r["weak"] and r["label"] == "direct_parent"
        ),
        "weak_ancestors": sorted(
            v for v, r in results.items() if r["weak"] and r["label"] == "indirect_ancestor"
        ),
        "variables": results,
    }
    (realization_dir / "datasets" / "ground_truth_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
