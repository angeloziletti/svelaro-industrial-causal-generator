"""
Temporal Generator: the sensor trajectories of a plant (each sensor a fixed
motif sequence, advanced_scm/motif_profiles.py, with parameters from the SCM)
and the SCM dataset they are driven by.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from advanced_scm.core import AdvancedExtrusionSCM
from logger_config import get_logger
from advanced_scm.parameter_layer import sensor_parameters

logger = get_logger(__name__)


def _generate_one_sensor(
    feature_idx: int,
    sensor_name: str,
    scm_sample_dicts: List[Dict[str, Any]],
    gp_parameter_nodes: Dict[str, List[str]],
    n_samples: int,
    time_length: int,
    seed: Optional[int],
    jitter_cfg: Optional[Dict[str, Any]] = None,
    residual_scale: float = 1.0,
    profile: Optional[List[Dict[str, Any]]] = None,
    measurement_noise: float = 0.05,
) -> Dict[str, Any]:
    """Generate the full time series (all ``n_samples`` samples) for ONE sensor.

    Worker for ``joblib.Parallel`` over the per-sensor loop in
    ``generate_gp_controlled_pattern_timeseries``; accumulators are returned to
    the master rather than mutated in place.

    Determinism: every draw is seeded from
    ``pattern_seed = seed + sample_idx * 1000 + feature_idx``, fully determined
    by the args we receive — output is identical regardless of n_jobs.

    Returns:
        A dict the caller merges into the global accumulators. Keys:
          - sensor_name (str)
          - feature_idx (int)
          - structured_array (np.ndarray [n_samples, time_length])
          - pattern_windows_array (np.ndarray [n_samples, 2], int)
          - pattern_fired_array (np.ndarray [n_samples], bool)
          - snr, physical_params
          - feature_metadata_entry (dict)
          - gp_stats (dict: controlled/random counters)
    """
    if profile is None:
        raise ValueError(
            f"sensor {sensor_name!r} has no motif profile: this release generates "
            f"motif_sequence trajectories only"
        )

    from advanced_scm.motif_profiles import sensor_series

    sensor_time_series_list: List[np.ndarray] = []
    pattern_windows_array = np.zeros((n_samples, 2), dtype=int)
    # False where every event was suppressed (stochastic firing): the window is still
    # recorded, flagged as not fired.
    pattern_fired_array = np.ones(n_samples, dtype=bool)
    mean_var_sum = 0.0
    residual_var_sum = 0.0
    physical_rows: List[Dict[str, float]] = []
    gp_stats = {"controlled": 0, "random": 0}

    for sample_idx in range(n_samples):
        scm_sample = scm_sample_dicts[sample_idx]

        # Per-(sample, sensor) seed, so output is deterministic regardless of
        # worker count. Jitter, firing and noise derive their RNGs from it.
        pattern_seed = (
            seed + sample_idx * 1000 + feature_idx if seed is not None else None
        )

        # Motif-sequence model (advanced_scm/motif_profiles.py): firing and
        # jitter per segment, noise from a per-(batch, sensor) stream.
        K = len(profile)
        physical = sensor_parameters(
            scm_sample, sensor_name, gp_parameter_nodes.get(sensor_name, []),
            np.random.default_rng(None if pattern_seed is None else int(pattern_seed) + 5051),
            n_segments=K,
        )
        seed_base = 0 if pattern_seed is None else int(pattern_seed)
        p_fire = float((jitter_cfg or {}).get("event_fire_probability", 1.0))
        fired = [
            np.random.default_rng(seed_base + 8629 + 97 * k).random() < p_fire
            for k in range(K)
        ]
        shifts, factors = None, None
        if jitter_cfg is not None and jitter_cfg.get("enabled", False):
            jr = np.random.default_rng(seed_base + 31337)
            shifts = [int(jr.integers(-int(jitter_cfg.get("timing_delta", 8)),
                                      int(jitter_cfg.get("timing_delta", 8)) + 1)) for _ in range(K)]
            factors = [max(0.1, 1.0 + float(jr.normal(0.0, float(jitter_cfg.get("amplitude_sigma", 0.15)))))
                       for _ in range(K)]
        out = sensor_series(
            profile, physical, time_length,
            np.random.default_rng(seed_base + 7001),
            fired=fired, timing_shift=shifts, amplitude_factor=factors,
            residual_scale=residual_scale, white_noise=measurement_noise,
        )
        sensor_time_series_list.append(out["series"].astype(np.float32))
        on = [(a, b) for a, b, f in out["windows"] if f]
        pattern_windows_array[sample_idx] = (
            (on[0][0], on[-1][1]) if on else (out["windows"][0][0], out["windows"][-1][1])
        )
        pattern_fired_array[sample_idx] = bool(on)
        mean_var_sum += out["mean_var"]
        residual_var_sum += out["residual_var"]
        gp_stats["controlled" if gp_parameter_nodes.get(sensor_name) else "random"] += 1
        physical_rows.append(dict(physical, fired=[bool(f) for f in fired]))

    structured_array = np.array(sensor_time_series_list)
    feature_metadata_entry = {
        "feature_index": feature_idx,
        "feature_name": sensor_name,
        "generation_method": "gp_controlled_pattern",
        "n_series": n_samples,
        "timesteps": time_length,
    }

    return {
        "sensor_name": sensor_name,
        "feature_idx": feature_idx,
        "structured_array": structured_array,
        "pattern_windows_array": pattern_windows_array,
        "pattern_fired_array": pattern_fired_array,
        # Realized SNR: within-sample variance of the event shape over that of
        # the stochastic residual, pooled over samples.
        "snr": mean_var_sum / residual_var_sum if residual_var_sum > 0 else None,
        "physical_params": physical_rows,
        "feature_metadata_entry": feature_metadata_entry,
        "gp_stats": gp_stats,
    }


class TemporalGenerator:
    """Generates the sensor trajectories of a plant from its SCM samples."""

    def __init__(self, unified_config, exporter, logger_instance=None):
        """
        Initialize the temporal generator.

        Args:
            unified_config: Unified configuration object
            exporter: TemporalDataExporter instance for sample export
            logger_instance: Optional logger instance
        """
        self.unified_config = unified_config
        self.exporter = exporter
        self.logger = logger_instance or logger

    def generate_gp_controlled_pattern_timeseries(
        self,
        scm_df: pd.DataFrame,
        scm: AdvancedExtrusionSCM,
        n_samples: int,
        export_dir,
        sensor_n_jobs: int = 1,
    ) -> Tuple[Dict, Dict]:
        """
        Generate pattern time series controlled by GP parameters from SCM samples.

        This method implements the GP parameter layer for causal validation:
        - SCM samples contain GP parameter values (event_timing, event_magnitude)
        - GP parameters control pattern generation (position, magnitude, coverage)
        - Enables testing if lossy temporal compression preserves causal information

        Parameters:
        -----------
        scm_df : pd.DataFrame
            SCM dataset containing GP parameter values for each sample
        scm : AdvancedExtrusionSCM
            SCM object with gp_parameter_nodes dict
        n_samples : int
            Number of samples to generate (must match scm_df length)
        export_dir : Path
            Directory for exporting samples

        Returns:
        --------
        metadata : Dict
            Generation metadata
        structured_time_series : Dict
            Structured time series by sensor name
        """
        self.logger.info("   🎛️ GP-Controlled Pattern Generation")
        self.logger.info(f"      GP parameter nodes: {scm.gp_parameter_nodes}")

        if len(scm_df) != n_samples:
            raise ValueError(
                f"SCM dataframe length ({len(scm_df)}) must match n_samples ({n_samples})"
            )

        # Get configuration
        pattern_selection = self.unified_config.patterns
        time_length = self.unified_config.sequence_length
        seed = self.unified_config.seed
        temporal_sensors = self.unified_config.temporal_sensors

        self.logger.info(f"      Pattern selection: {pattern_selection}")
        self.logger.info(f"      Time length: {time_length}")
        self.logger.info(f"      Temporal sensors: {temporal_sensors}")

        # Pattern names (recorded per sensor in the metadata; the trajectories are
        # motif sequences)
        if pattern_selection == "critical":
            # Use only GP-controlled critical patterns (importance 9-10)
            critical_patterns = [
                "Sharp_Upward_Spike",
                "Sharp_Downward_Spike",
                "Step_Change_Up",
                "Step_Change_Down",
                "Monotonic_Increase",
                "Monotonic_Decrease",
                "Baseline_Shift",
            ]
            available_patterns = critical_patterns
        elif pattern_selection == "all":
            raise NotImplementedError("pattern_selection 'all' is not part of this release")
        else:
            # Assume comma-separated list
            available_patterns = [p.strip() for p in pattern_selection.split(",")]

        self.logger.info(f"      Using patterns: {available_patterns}")

        # One pattern name per sensor, assigned by a seeded shuffle (cycled when
        # n_sensors > n_patterns).
        n_sensors = len(temporal_sensors)
        n_patterns = len(available_patterns)
        # Which sensor gets which shape is structure, not data.
        rng = np.random.default_rng(self.unified_config.structure_seed())
        shuffled_patterns = list(available_patterns)
        rng.shuffle(shuffled_patterns)
        sensor_pattern_assignment: Dict[str, str] = {
            temporal_sensors[i]: shuffled_patterns[i % n_patterns]
            for i in range(n_sensors)
        }
        for sensor, pattern in sensor_pattern_assignment.items():
            self.logger.info(f"      📌 {sensor} → {pattern}")
        if n_sensors > n_patterns:
            self.logger.info(
                f"      ℹ️ {n_sensors} sensors > {n_patterns} patterns — "
                f"some sensors share a pattern (cycled assignment)"
            )

        # Generate time series for each sensor
        structured_time_series = {}
        feature_metadata = []
        gp_control_stats = {"controlled": 0, "random": 0}

        # Event window per (sensor, sample), recorded in the metadata.
        # Shape per sensor: (n_samples, 2) with [start_idx, end_idx] in timestep coordinates.
        pattern_windows: Dict[str, np.ndarray] = {
            sensor: np.zeros((n_samples, 2), dtype=int) for sensor in temporal_sensors
        }
        # Per sensor, per sample: did the event fire (False = flattened to baseline)?
        pattern_fired: Dict[str, np.ndarray] = {
            sensor: np.ones(n_samples, dtype=bool) for sensor in temporal_sensors
        }
        sensor_snr: Dict[str, Optional[float]] = {}
        physical_params: Dict[str, List[Dict[str, float]]] = {}

        # Per-sensor parallelism. The outer "for sensor in temporal_sensors"
        # loop is embarrassingly parallel: each sensor's body only reads
        # shared state (scm_df, gp_parameter_nodes, the sensor profiles)
        # and writes its own per-sensor slice. We farm 20 sensors across
        # `sensor_n_jobs` loky workers; the master merges results below.
        #
        # Determinism preserved: pattern_seed inside the worker is a pure
        # function of (seed, sample_idx, feature_idx), so output bytes are
        # identical to the sequential version regardless of n_jobs.
        #
        # scm_df.to_dict('records') materializes once outside the call so we
        # don't pay it per-sensor; each worker only reads from this list.
        scm_sample_dicts = scm_df.to_dict("records")
        self.logger.info(
            f"      📊 Generating {len(temporal_sensors)} sensors with "
            f"sensor_n_jobs={sensor_n_jobs} (loky backend)"
        )

        from joblib import Parallel, delayed

        # Build a tiny picklable dict for per-sample variation. Workers can't
        # rely on the unified_config dataclass directly because loky spawns
        # fresh processes that don't import the full advanced_scm tree.
        vec_cfg = getattr(self.unified_config, "vector_scm", None)
        jitter_cfg: Optional[Dict[str, Any]] = None
        if vec_cfg is not None and vec_cfg.per_sample_variation.enabled:
            jitter_cfg = {
                "enabled": True,
                "amplitude_sigma": float(vec_cfg.per_sample_variation.amplitude_sigma),
                "timing_delta": int(vec_cfg.per_sample_variation.timing_delta),
                # Per (sample, sensor) firing probability. 1.0 = current
                # behaviour (every sample fires). p < 1.0 means worker skips
                # pattern injection with prob (1 - p) and falls back to the
                # baseline + kernel-noise path.
                "event_fire_probability": float(
                    getattr(vec_cfg.per_sample_variation, "event_fire_probability", 1.0)
                ),
            }

        sensor_results = Parallel(n_jobs=sensor_n_jobs, backend="loky")(
            delayed(_generate_one_sensor)(
                feature_idx=i,
                sensor_name=name,
                scm_sample_dicts=scm_sample_dicts,
                gp_parameter_nodes=scm.gp_parameter_nodes,
                n_samples=n_samples,
                time_length=time_length,
                seed=seed,
                jitter_cfg=jitter_cfg,
                residual_scale=float(
                    getattr(self.unified_config, "pattern_residual_scale", 1.0)
                ),
                profile=(scm.graph.graph.get("sensor_profiles") or {}).get(name),
                measurement_noise=float(getattr(self.unified_config, "measurement_noise", 0.05)),
            )
            for i, name in enumerate(temporal_sensors)
        )

        # Merge worker results in feature_idx order.
        sensor_results.sort(key=lambda r: r["feature_idx"])
        for res in sensor_results:
            sensor_name = res["sensor_name"]
            structured_time_series[sensor_name] = res["structured_array"]
            pattern_windows[sensor_name] = res["pattern_windows_array"]
            pattern_fired[sensor_name] = res["pattern_fired_array"]
            sensor_snr[sensor_name] = res["snr"]
            physical_params[sensor_name] = res["physical_params"]
            feature_metadata.append(res["feature_metadata_entry"])
            gp_control_stats["controlled"] += res["gp_stats"]["controlled"]
            gp_control_stats["random"] += res["gp_stats"]["random"]
            self.logger.info(
                f"         ✅ Generated {sensor_name}: "
                f"{res['structured_array'].shape}"
            )
        self.logger.info(
            f"      ✅ GP Control Stats: {gp_control_stats['controlled']} controlled, "
            f"{gp_control_stats['random']} random"
        )

        # Validation side-outputs (an Excel sample, the GP diagnostic). Off in the
        # ground-truth audit's replays, which regenerate a plant hundreds of times.
        write_exports = bool(getattr(self.unified_config, "write_validation_exports", True))
        if write_exports:
            self.exporter.export_temporal_sample_from_structured(
                structured_time_series, temporal_sensors, export_dir
            )

        # GP-control diagnostic: per-sample event records of the single-pattern
        # model; motif-sequence plants have none, so the file holds an empty list
        # (kept so the output folder is unchanged).
        if write_exports:
            diag_dir = Path(export_dir) / "datasets"
            diag_dir.mkdir(parents=True, exist_ok=True)
            diag_path = diag_dir / "gp_diagnostic.json"
            with open(diag_path, "w") as f:
                json.dump([], f, indent=2)

        metadata = {
            "method": "gp_controlled_pattern",
            "n_samples_requested": n_samples,
            "n_features_requested": len(temporal_sensors),
            "total_time_series_generated": 0,  # no flattened rows (the exporter writes the long format)
            "timesteps_per_series": time_length,
            "pattern_selection": pattern_selection,
            "gp_control_stats": gp_control_stats,
            "gp_parameters": [],  # per-sample records of the single-pattern model: none here
            "gp_parameter_nodes": scm.gp_parameter_nodes,  # Node structure
            "feature_metadata": feature_metadata,
            "generation_structure": f"{n_samples} samples × {len(temporal_sensors)} features",
            "temporal_feature_names": temporal_sensors,
            # Per-sensor per-sample event window in timestep coordinates [start, end].
            "pattern_windows": pattern_windows,
            # Per-sensor per-sample: True if the event fired, False if stochastic firing
            # suppressed it.
            "pattern_fired": pattern_fired,
            "sensor_snr": sensor_snr,
            # Per sensor, per sample: the physical parameters actually used.
            "physical_params": physical_params,
            # Sensor → pattern-name assignment (cycled across the configured pattern set).
            "sensor_pattern_assignment": sensor_pattern_assignment,
        }

        return metadata, structured_time_series


    def generate_scm_dataset_without_constraints(
        self, scm: AdvancedExtrusionSCM, n_samples: int
    ) -> pd.DataFrame:
        """Generate proper SCM dataset when no constraints are specified."""

        # Generate dataset using SCM's proven methodology
        scm_df = scm.generate_dataset(n_samples, apply_post_processing=True)

        self.logger.info(
            f"   Generated SCM dataset without constraints: {scm_df.shape}"
        )

        return scm_df

