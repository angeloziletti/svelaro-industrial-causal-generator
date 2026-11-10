"""
Unified Configuration System for Advanced SCM
Single source of truth with YAML → defaults → usage flow
"""

from dataclasses import dataclass, fields, field
import dataclasses
from typing import Dict, List, Optional, Any, Union
import yaml
from logger_config import get_logger

logger = get_logger(__name__)


def generate_default_defect_rates(n_defects: int = 4) -> Dict[str, float]:
    """
    Generate default defect configurations with varying rates.

    Default defect types (in priority order):
    1. gel - Most common (15%)
    2. black_spec - Common (8%)
    3. fish_eye - Moderate (5%)
    4. die_lines - Less common (4%)
    5. contamination - Rare (3%)
    6. burn_marks - Very rare (2%)

    Parameters:
    -----------
    n_defects : int
        Number of defect types to include (default: 4, max: 6)

    Returns:
    --------
    defect_rates : Dict[str, float]
        Dictionary mapping defect names to target rates
    """
    # Defect types with decreasing frequency (realistic manufacturing distribution)
    available_defects = [
        ("gel", 0.15),           # Most common
        ("black_spec", 0.08),    # Common
        ("fish_eye", 0.05),      # Moderate
        ("die_lines", 0.04),     # Less common
        ("contamination", 0.03), # Rare
        ("burn_marks", 0.02),    # Very rare
    ]

    # Clamp n_defects to valid range
    n_defects = max(1, min(n_defects, len(available_defects)))

    # Return the first n_defects
    return dict(available_defects[:n_defects])














@dataclass
class VectorSCMPerSampleVariationConfig:
    """Per-sample jitter on GP-driven pattern parameters so the samples in a
    context show varied manifestations of the same underlying cause."""
    enabled: bool = True
    amplitude_sigma: float = 0.15   # event_magnitude *= (1 + N(0, sigma))
    timing_delta: int = 8           # event_start shifted by Uniform{-delta..delta} timesteps
    # Per (sample, sensor) firing probability. 1.0 = every sample fires the
    # GP-controlled pattern (current behaviour). p < 1.0 = with probability
    # (1 - p) the pattern is *not* injected for this (sample, sensor) pair
    # — the time series collapses to baseline + kernel noise. Independent
    # draws per sensor mean defect_ts now depends on WHICH truths fired,
    # not just their event timing → breaks the within-context peak-co-
    # occurrence alignment that Pearson exploits. a lag-based method still sees
    # ~p * samples_per_context events per context, plenty of signal.
    event_fire_probability: float = 1.0


@dataclass
class VectorSCMOperatorConfig:
    """Legacy per-edge ranges of the operator models (vector_scm.operators).
    Parsed so existing configs load; the operator models are not part of this
    release, so they have no effect."""
    lag_max: int = 5
    weight_range: List[float] = field(default_factory=lambda: [0.6, 1.2])
    weight_sign_random: bool = True
    nonlinearity_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "linear":                 0.0,
            "tanh":                   0.15,
            "softplus":               0.05,
            "sine":                   0.40,
            "gaussian_bump_shifted":  0.40,
        }
    )
    windows: List[int] = field(default_factory=lambda: [4, 8, 16])


@dataclass
class TemporalConfounderConfig:
    """
    Temporal confounder Z_k(t), an extension of shadow red herrings.

    A scalar Z_k per truth sensor that modulates the GP
    parameters of both truth and shadow red herrings. This produces SAMPLE-
    LEVEL confounding (sample-aggregated methods see
    confounded shadow correlations matching truth) but NOT within-sample
    temporal confounding — a scalar can't transmit temporal shape.

    Here Z_k becomes a time series Z_k(t), generated per (truth
    sensor, sample) as a smooth Gaussian-process-like signal, and added
    as an ADDITIVE contribution to BOTH truth's and shadow's pattern-
    injected time series at the temporal level:

        truth_ts(t)  = pattern_T_k(t) + α_truth  · Z_{k,s}(t) + noise
        shadow_ts(t) = pattern_R(t)   + α_shadow · Z_{k,s}(t) + noise

    The vector mechanism takes truth_ts (including the Z_k(t) component)
    and produces defect_ts. Both shadow_ts and defect_ts now share within-
    sample temporal structure via Z_{k,s}(t), so long-format Pearson sees
    confounded shadow correlations.

    Default off.
    """
    enabled: bool = False
    # Length scale of Z_k(t) smoothness in TIMESTEPS. Must be ≥ vector
    # mechanism lag_max (so Z_k(t) correlations survive the lag). Smaller
    # values → choppier Z; larger → smoother. 15 is ~12% of T=128.
    correlation_length: int = 15
    # Fixed contribution of Z_k(t) to truth_ts (added on top of the
    # pattern-injection output). Modest by default so truth's pattern
    # signal still dominates the vector mechanism — Z_k(t) is a
    # confounder, not the primary causal driver.
    # A [lo, hi] list draws it per truth sensor (graph seed) instead; give it the
    # same range as alpha_shadow_range so a sensor's confounder share does not
    # reveal whether it is a cause.
    alpha_truth: Union[float, List[float]] = 0.5
    # Distribution range for α_shadow (sampled per shadow red herring per
    # realisation, uniform). Skewed slightly higher than alpha_truth so
    # the strongest shadows can match truth's correlation magnitude.
    alpha_shadow_range: List[float] = field(default_factory=lambda: [0.3, 1.2])
    # Tie the within-batch route to the confounder share (vector_scm.shares.confounder): the strengths above are
    # multiplied by g(s) = sqrt(s/(1-s)) / sqrt(r/(1-r)) with r = share_reference, so g = 1 at the reference share
    # and the data there are unchanged. None = not tied (the strengths ignore the share).
    share_reference: Optional[float] = None
    # Normalise Z_k(t) to unit variance per sample before scaling by
    # alpha. Makes alpha interpretable as variance-fraction contribution.
    normalise_to_unit_variance: bool = True
    # RNG offset for Z_k(t) sampling, independent of scalar-Z (1693) and
    # other RNG streams in the pipeline.
    seed_offset: int = 4093


@dataclass
class ShadowRedHerringConfig:
    """
    Shadow red herring confounder structure.

    When enabled, every red-herring sensor is assigned as a "shadow" of
    exactly one truth sensor through a per-truth confounder Z_k. Z_k feeds
    into both:
      (a) the truth sensor's GP parameter nodes (event_timing, event_magnitude)
      (b) the shadow red herrings' GP parameter nodes (stochastically per param)

    This introduces sample-level confounding so non-causal sensors correlate
    with defect_ts through a shared upstream cause. Default off.
    """
    enabled: bool = False
    confounder_strength_distribution: str = "uniform"
    # Edge-weight range for Z_k → {truth,shadow} GP-param edges. Higher
    # values produce stronger confounding (red herrings more correlated
    # with defect_ts via the Z_k → T_k → defect_ts path).
    confounder_strength_range: List[float] = field(
        default_factory=lambda: [0.1, 0.9]
    )
    # Which GP parameters can be modulated by Z_k. Default both.
    modulated_gp_params: List[str] = field(
        default_factory=lambda: ["onset", "amplitude"]
    )
    # Per shadow red herring and per GP param, independent Bernoulli draw
    # at this probability decides whether to add a Z_k edge. At 0.7, ~49%
    # of shadows get both params modulated, ~42% get one, ~9% get neither.
    modulation_probability: float = 0.7
    # RNG offset for confounder sampling (independent of pattern seeds).
    seed_offset: int = 1693
    # Temporal confounder Z_k(t): adds within-sample temporal
    # confounding on top of the sample-level scalar Z_k. Default disabled.
    temporal_confounder: TemporalConfounderConfig = field(
        default_factory=TemporalConfounderConfig
    )


@dataclass
class VectorSCMMixContribution:
    """One contribution of the legacy operator mix (defect_operator_mix).

    Parsed so existing configs load; the operator models are not part of this
    release (edge_model must be "dynamic"), so it has no effect.
    """
    operator: str = "lagged_pointwise"
    weight: float = 1.0


@dataclass
class VectorSCMMixConfig:
    """Legacy mixed-operator block (defect_operator_mix). Parsed so existing
    configs load; ignored, since the operator models are not part of this
    release (edge_model must be "dynamic").
    """
    enabled: bool = False
    contributions: List[VectorSCMMixContribution] = field(default_factory=list)


@dataclass
class VectorSCMConfig:
    """
    Vector-SCM defect generation.

    Adds a `defect_ts` column (length-T per-sample time series) to the dataset,
    produced from the truth sensors' trajectories (advanced_scm/vector_scm.py).

    Some fields are kept only so existing configs load; this release supports
    target_mechanism "trajectory", edge_model "dynamic", target_path null and
    per_sample_center false, and raises on any other value
    (TabularDatasetGenerator._initialize_scm, vector_scm.generate_defect_ts).
    """
    enabled: bool = True
    # "trajectory": defect_ts(t) is a dynamic function of the truths' trajectories.
    # (The "batch", "both" and "scalar" targets are not part of this release.)
    target_mechanism: str = "trajectory"
    defect_operator: str = "lagged_pointwise"      # legacy operator name: recorded, not used
    noise_sigma: float = 0.10                       # recorded in the audit
    # Legacy per-sample z-scoring of the operator inputs: parsed, not used.
    standardize_parents: bool = False
    # Divide each parent by its mean/SD over ALL batches and timesteps before
    # the defect mechanism (keeps between-batch differences; fixes the input scale so
    # e.g. control tightness does not change the nonlinearities' regime).
    dataset_standardize_parents: bool = False
    # The record path (transport delay, response, lab measurement): must be null,
    # not part of this release.
    target_path: Optional[Dict[str, Any]] = None
    # For the scalar target (not part of this release): "parameters" or "dose".
    scalar_mechanism: str = "parameters"
    # How the causes reach the defect (advanced_scm/target_edges.py): must be
    # "dynamic" (the defect is a node of the time-series SCM: dead time +
    # first-order lag + nonlinearity per cause); the operator models are not
    # part of this release.
    edge_model: str = "operators"
    dynamic_target: Optional[Dict[str, Any]] = None  # ranges for edge_model dynamic
    # Sensors downstream of the defect: {n: [lo, hi], gain, time_constant,
    # dead_time}. None = none.
    consequences: Optional[Dict[str, Any]] = None
    # Edge strengths as explained-variance shares (advanced_scm/shares.py):
    # {defect_timeseries, consequence, confounder}: a number or [lo, hi]. None = gains.
    shares: Optional[Dict[str, Any]] = None
    # Per-sample centering of the sensor curves: must be false, not part of this release.
    per_sample_center: bool = False
    operators: VectorSCMOperatorConfig = field(default_factory=VectorSCMOperatorConfig)
    per_sample_variation: VectorSCMPerSampleVariationConfig = field(
        default_factory=VectorSCMPerSampleVariationConfig
    )
    # Legacy mixed-operator block: parsed, not used (see VectorSCMMixConfig).
    defect_operator_mix: VectorSCMMixConfig = field(default_factory=VectorSCMMixConfig)


@dataclass
class TargetLabelsConfig:
    """
    Explicit configuration for target/label columns.

    Defines which columns are targets and their types (binary or continuous).
    This is the single source of truth — no magic string matching on column names.

    Config format:
        target_labels:
          label_dryer:
            type: "binary"
          gel_defect:
            type: "continuous"
    """
    columns: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # e.g. {"label_dryer": {"type": "binary"}, "gel_defect": {"type": "continuous"}}














@dataclass
class UnifiedSCMConfig:
    """
    Single source of truth for ALL advanced_scm configuration.
    All defaults are defined here once - no duplication!
    """

    # === CORE SCM SETTINGS ===
    seed: int = 42
    # Seed of the causal STRUCTURE: graph, mechanism weights, vector-SCM edge
    # params, shadow strengths, sensor->pattern assignment. `seed` then drives
    # only the data drawn from it, so realizations of one graph share the graph.
    # Set per graph by generate_batch_datasets; None = same as `seed`.
    graph_seed: Optional[int] = None
    use_dynamic_graph: bool = False
    vector_dim: int = 8
    min_base_nodes: int = 15  # Minimum number of base SCM nodes (0 = temporal-only mode)
    max_base_nodes: int = 40  # Maximum number of base SCM nodes
    # Edges among the base (batch-condition) nodes. true: a connected scale-free
    # graph, so every sensor shares a hidden ancestor with every other. false:
    # independent roots (SCM ablation: confounding through batch conditions off).
    base_edges: bool = True

    # === DATASET GENERATION ===
    n_samples: int = 1000
    sequence_length: int = 128
    n_defects: int = 4  # Default number of defect types (max 4 recommended for simplicity)
    target_defect_rates: Optional[Dict[str, float]] = (
        None  # No default - triggers unconstrained if missing
    )

    # === SENSOR DATA IMPORT SETTINGS ===
    sensor_data: Optional[Dict[str, Any]] = None
    time_scale_analysis: Optional[Dict[str, Any]] = None
    # === CLUSTERING ANALYSIS ===


    # === TARGET LABELS (explicit definition of target/label columns) ===
    target_labels: TargetLabelsConfig = field(
        default_factory=TargetLabelsConfig
    )

    # === VECTOR-SCM (time-series defect: defect_ts generated via vector
    #     mechanism on truth-sensor time series; parallel to the static SCM's
    #     scalar process_defect). ===
    vector_scm: VectorSCMConfig = field(default_factory=VectorSCMConfig)

    # === SHADOW RED HERRINGS  ===
    # Per-truth confounders Z_k introduce sample-level confounding between
    # truth sensors and red-herring "shadows". Default off; opt-in via YAML.
    shadow_red_herrings: ShadowRedHerringConfig = field(
        default_factory=ShadowRedHerringConfig
    )

    # === TEMPORAL SETTINGS ===
    enable_temporal: bool = True
    # feature matrix that tabular methods consume; lag-based methods read the raw
    # resampled series, so a lag-based run needs none of them.
    temporal_generation_method: str = "cauker"
    export_temporal_data: bool = True
    time_interval_minutes: int = 5
    # K: consecutive synthetic samples concatenated into one context.
    # Written as a `context_index` column in full_timeseries.xlsx; the
    # loader splits on it. K=10 with N=100 → 10 contexts × 1280 steps × 10 events.
    samples_per_context: int = 10

    # === GP PARAMETER LAYER SETTINGS (for Causal Validation) ===
    enable_gp_layer: bool = False  # Enable GP parameter layer for temporal control
    auto_select_temporal_sensors: bool = False  # Auto-select temporal sensors (for dynamic graphs)
    min_temporal_sensors: int = 3  # Minimum number of temporal sensors to auto-select (if auto_select=True)
    max_temporal_sensors: int = 6  # Maximum number of temporal sensors to auto-select (if auto_select=True)
    min_gp_defect_connections: int = 1  # Minimum number of defects each GP node connects to
    max_gp_defect_connections: int = 2  # Maximum number of defects each GP node connects to
    n_temporal_red_herrings: int = 0  # Number of temporal sensors that are NON-causal (specificity test)
    # [lo, hi]: draw the number of true causes per plant (structure seed);
    # overrides n_temporal_red_herrings. None = fixed by n_temporal_red_herrings.
    n_truths: Optional[List[int]] = None
    # Validation side-outputs during generation (Excel sample, GP diagnostic).
    # The ground-truth audit turns them off for its replays.
    write_validation_exports: bool = True
    # Per-realization parallelism — # of loky workers across the per-sensor
    # loop in generate_gp_controlled_pattern_timeseries. 1 = sequential.
    sensor_n_jobs: int = 1

    # === CAUKER SETTINGS ===
    cauker_features: int = 4
    cauker_max_parents: int = 6
    cauker_num_nodes: int = 18

    # === PATTERN SETTINGS ===
    patterns: str = "critical"  # "critical", "all", "high", or comma-separated list
    pattern_noise_level: str = "moderate"  # "clean", "moderate", "heavy"
    # Signal-to-noise dial: multiplies each sensor's stochastic residual (GP
    # kernel draw + background interference) relative to its mean function
    # (the event shape). 1.0 = unchanged; realized SNR per sensor is recorded
    # in dataset_metadata.json (sensor_snr).
    pattern_residual_scale: float = 1.0
    # Which generative parameters of each sensor are SCM nodes
    # (advanced_scm/parameter_layer.py): shape = onset, amplitude, duration,
    # level; kernel = noise_amplitude, lengthscale. The rest are random per batch.
    causal_params: List[str] = field(default_factory=lambda: ["onset", "amplitude"])
    # "single_pattern" (one motif per sensor; not part of this release) or
    # "motif_sequence": a fixed sequence of segments_per_sensor = [min, max] motifs
    # per sensor, each with its own onset / duration / amplitude
    # (advanced_scm/motif_profiles.py).
    trajectory_model: str = "single_pattern"
    segments_per_sensor: List[int] = field(default_factory=lambda: [2, 4])
    # motif_sequence: SD of white measurement noise per timestep (raw scale).
    measurement_noise: float = 0.05
    # The generator writes raw curves and ground truth; the tabular table is not built.
    export_tabular: bool = True
    # Parameter-mode edges between sensors (graph_generators._add_sensor_graph):
    # {enabled, n_upstream, expected_in_degree}. None = no sensor-to-sensor edges.
    sensor_graph: Optional[Dict[str, Any]] = None
    # Closed-loop control inside each batch (advanced_scm/control_loops.py):
    # {enabled, n_loops, tightness, disturbance_gain}. None = no control loops.
    control_loops: Optional[Dict[str, Any]] = None




    # === OUTPUT SETTINGS ===
    output_directory: str = "output"
    create_visualizations: bool = True

    # Plot switches: diagnostic plots are off by default.
    diagnostic_plots: bool = False
    headline_plots: bool = True
    export_timeseries_plots: bool = True  # Export raw time series with defect overlays
    export_tabular_excel: bool = False
    output_format: str = "excel"

    temporal_sensors: Optional[List[str]] = None  # Must be provided in YAML
    static_features: Optional[List[str]] = None  # Must be provided in YAML

    # Multi-graph validation settings
    multi_graph_validation: Optional[Dict[str, Any]] = None

    # Batch generation settings
    batch_generation: Optional[Dict[str, Any]] = None

    # Quality filter: post-generation gate that requires every truth (causal)
    # sensor to be observably correlated with the defect target above a
    # configurable threshold. Used by generate_batch_datasets.py to retry
    # seeds / redraw graphs when realizations come out too weak. Anchored
    # against Pi rig empirical truth |r| distribution.
    quality_filter: Optional[Dict[str, Any]] = None

    # What methods see. Event-parameter nodes are always hidden (exported only as
    # oracle data); `confounders` (Z_k, default false) and `static_attributes`
    # (batch conditions, default true) choose whether those SCM columns are
    # observed. See advanced_scm/observability.py.
    observability: Optional[Dict[str, Any]] = None

    # Multi-source combination settings

    def __post_init__(self):
        """Initialize default values for mutable defaults."""
        pass

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "UnifiedSCMConfig":
        """
        Load config from YAML using dataclass defaults - NO DUPLICATION!
        """
        logger.info(f"📁 UNIFIED_CONFIG: Loading from {yaml_path}")

        # Load YAML
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        logger.info(f"📋 UNIFIED_CONFIG: YAML sections: {list(yaml_data.keys())}")

        # Get dataclass field defaults automatically
        field_defaults = {
            f.name: (
                f.default
                if f.default is not dataclasses.MISSING
                else (
                    f.default_factory()
                    if f.default_factory is not dataclasses.MISSING
                    else None
                )
            )
            for f in fields(cls)
        }

        config_values = {}
        used_from_yaml = []
        used_defaults = []

        # Define YAML path mappings (where to find each field in YAML)
        yaml_paths = {
            # Core SCM
            "seed": "dataset_generation.seed",
            "use_dynamic_graph": "scm_settings.use_dynamic_graph",
            "vector_dim": "scm_settings.vector_dim",
            "min_base_nodes": "scm_settings.min_base_nodes",
            "max_base_nodes": "scm_settings.max_base_nodes",
            "base_edges": "scm_settings.base_edges",
            # Dataset generation
            "n_samples": "dataset_generation.n_samples",
            "sequence_length": "dataset_generation.length",  # Note: YAML uses 'length'
            "n_defects": "dataset_generation.n_defects",
            # Sensor data import
            # Temporal
            "enable_temporal": "temporal.enable_temporal",
            "temporal_generation_method": "temporal.temporal_generation_method",
            "export_temporal_data": "temporal.export_temporal_data",
            "time_interval_minutes": "temporal.time_interval_minutes",
            "samples_per_context": "temporal.samples_per_context",
            # GP Parameter Layer
            "enable_gp_layer": "gp_parameter_layer.enable_gp_layer",
            "auto_select_temporal_sensors": "gp_parameter_layer.auto_select_temporal_sensors",
            "min_temporal_sensors": "gp_parameter_layer.min_temporal_sensors",
            "max_temporal_sensors": "gp_parameter_layer.max_temporal_sensors",
            "min_gp_defect_connections": "gp_parameter_layer.min_gp_defect_connections",
            "max_gp_defect_connections": "gp_parameter_layer.max_gp_defect_connections",
            "n_temporal_red_herrings": "gp_parameter_layer.n_temporal_red_herrings",
            "n_truths": "gp_parameter_layer.n_truths",
            # CAUKER
            "cauker_features": "dataset_generation_cauker.features",
            "cauker_max_parents": "dataset_generation_cauker.max_parents",
            "cauker_num_nodes": "dataset_generation_cauker.num_nodes",
            # Pattern
            "patterns": "pattern_settings.patterns",
            "pattern_noise_level": "pattern_settings.noise_level",
            "pattern_residual_scale": "pattern_settings.residual_scale",
            "causal_params": "parameter_layer.causal_params",
            "trajectory_model": "parameter_layer.trajectory_model",
            "segments_per_sensor": "parameter_layer.segments_per_sensor",
            "measurement_noise": "parameter_layer.measurement_noise",
            "export_tabular": "output.export_tabular",
            "sensor_graph": "sensor_graph",
            "control_loops": "control_loops",
            # Cross-sensor (under time_scale_analysis in YAML)
            # Output
            "output_directory": "output.directory",
            "create_visualizations": "output.create_visualizations",
            # MUST be here: this dict is a whitelist and a YAML key absent from
            # it is silently dropped -- test_config_keys_are_live calls it
            # "the machine that manufactures dead keys".
            "diagnostic_plots": "output.diagnostic_plots",
            "headline_plots": "output.headline_plots",
            "export_timeseries_plots": "output.export_timeseries_plots",
            "export_tabular_excel": "output.export_tabular_excel",
            "output_format": "output.format",
            # Complete sections (stored as full dictionaries)
            "sensor_data": "sensor_data",
            "time_scale_analysis": "time_scale_analysis",
            "batch_generation": "batch_generation",
            "quality_filter": "quality_filter",
            "observability": "observability",
            "sensor_n_jobs": "gp_parameter_layer.sensor_n_jobs",
        }

        # Extract values using the mappings
        for field_name, yaml_path in yaml_paths.items():
            default_value = field_defaults.get(field_name)

            # Handle top-level vs nested paths
            if "." in yaml_path:
                yaml_value = get_nested_value(yaml_data, yaml_path)
            else:
                yaml_value = yaml_data.get(yaml_path)

            if yaml_value is not None:
                config_values[field_name] = yaml_value
                # Handle logging for dict vs simple values
                if isinstance(yaml_value, dict):
                    used_from_yaml.append(f"{field_name} = {list(yaml_value.keys())}")
                    logger.info(
                        f"✅ UNIFIED_CONFIG: Using YAML {field_name}: {list(yaml_value.keys())}"
                    )
                else:
                    used_from_yaml.append(f"{field_name} = {yaml_value}")
                    logger.debug(
                        f"✅ UNIFIED_CONFIG: Using YAML {field_name}: {yaml_value}"
                    )
            else:
                config_values[field_name] = default_value
                used_defaults.append(f"{field_name} = {default_value}")

        # Handle target_defect_rates - generate from n_defects if not specified
        defect_rates = get_nested_value(
            yaml_data, "dataset_generation.target_defect_rates"
        )
        if defect_rates:
            config_values["target_defect_rates"] = defect_rates
            used_from_yaml.append(f"target_defect_rates = {defect_rates}")
            logger.info(f"✅ UNIFIED_CONFIG: Using YAML defect rates: {defect_rates}")
        elif defect_rates == {}:
            # Explicit `{}`: no scalar defect node and no process_label. The
            # vector-SCM defect_ts is then the only target, and its parents are
            # the non-red-herring sensors (graph_generators / tabular_dataset_generator).
            config_values["target_defect_rates"] = {}
            used_from_yaml.append("target_defect_rates = {} (defect_ts is the only target)")
            logger.info(
                "✅ UNIFIED_CONFIG: target_defect_rates = {} -> no scalar defect; "
                "defect_ts is the only target"
            )
        else:
            # Generate default defect rates based on n_defects parameter
            n_defects = config_values.get("n_defects", 4)
            default_rates = generate_default_defect_rates(n_defects)
            config_values["target_defect_rates"] = default_rates
            used_defaults.append(f"target_defect_rates = {default_rates} (generated from n_defects={n_defects})")
            logger.info(f"✅ UNIFIED_CONFIG: Generated default defect rates ({n_defects} defects): {default_rates}")

        # Handle temporal_sensors - single source of truth at sensor_data.temporal_sensors
        temporal_sensors = get_nested_value(yaml_data, "sensor_data.temporal_sensors")
        if temporal_sensors:
            config_values["temporal_sensors"] = temporal_sensors
            used_from_yaml.append(f"temporal_sensors = {len(temporal_sensors)} items")
            logger.info(
                f"✅ UNIFIED_CONFIG: Using YAML temporal_sensors: {len(temporal_sensors)} items"
            )
        else:
            config_values["temporal_sensors"] = []
            used_defaults.append("temporal_sensors = [] (will auto-detect)")
            logger.info("✅ UNIFIED_CONFIG: temporal_sensors not specified (will auto-detect)")

        # Handle static_features (optional)


        # Handle multi-graph validation (optional)

        # Handle clustering_analysis (optional, nested under time_scale_analysis)


        # Handle feature_classification (optional, can be at root level or under time_scale_analysis)

        # Handle anomaly_detection (optional, under temporal section)


        # Handle vector_scm (optional, top-level block)
        vector_scm_dict = get_nested_value(yaml_data, "vector_scm")
        if vector_scm_dict:
            # Nested sub-blocks parsed into their dataclasses; top-level fields
            # passed positionally so unknown YAML keys raise a clean error.
            psv_dict = vector_scm_dict.get("per_sample_variation") or {}
            ops_dict = vector_scm_dict.get("operators") or {}
            psv = VectorSCMPerSampleVariationConfig(**psv_dict)
            ops = VectorSCMOperatorConfig(**ops_dict)

            # Mixed-operator config (optional). Parse the nested
            # `defect_operator_mix.contributions` list of dicts into
            # VectorSCMMixContribution dataclasses, leaving other top-level
            # fields (enabled) to flow through the dataclass kwargs.
            mix_dict = vector_scm_dict.get("defect_operator_mix") or {}
            if mix_dict:
                contrib_dicts = mix_dict.get("contributions") or []
                contributions = [
                    VectorSCMMixContribution(**c) for c in contrib_dicts
                ]
                mix_top = {
                    k: v for k, v in mix_dict.items() if k != "contributions"
                }
                mix_cfg = VectorSCMMixConfig(
                    **mix_top, contributions=contributions
                )
            else:
                mix_cfg = VectorSCMMixConfig()

            top_level = {
                k: v for k, v in vector_scm_dict.items()
                if k not in ("per_sample_variation", "operators", "defect_operator_mix")
            }
            vector_scm_config = VectorSCMConfig(
                **top_level,
                operators=ops,
                per_sample_variation=psv,
                defect_operator_mix=mix_cfg,
            )
            config_values["vector_scm"] = vector_scm_config
            used_from_yaml.append(
                f"vector_scm = enabled={vector_scm_config.enabled} "
                f"op={vector_scm_config.defect_operator} "
                f"jitter={vector_scm_config.per_sample_variation.enabled}"
            )
            logger.info(
                f"✅ UNIFIED_CONFIG: Vector-SCM: enabled={vector_scm_config.enabled}, "
                f"op={vector_scm_config.defect_operator}, "
                f"noise_sigma={vector_scm_config.noise_sigma}, "
                f"jitter={vector_scm_config.per_sample_variation.enabled}"
            )
        else:
            config_values["vector_scm"] = VectorSCMConfig()
            logger.info(
                "ℹ️  UNIFIED_CONFIG: Vector-SCM using defaults "
                "(enabled=True, op=lagged_pointwise)"
            )

        # Handle shadow_red_herrings (optional, top-level block),
        # including the nested `temporal_confounder` sub-block.
        shadow_dict = get_nested_value(yaml_data, "shadow_red_herrings")
        if shadow_dict:
            # Pop the nested temporal_confounder block so it doesn't conflict
            # with the **shadow_dict kwarg expansion below.
            shadow_kwargs = dict(shadow_dict)
            tc_dict = shadow_kwargs.pop("temporal_confounder", None)
            tc_config = (
                TemporalConfounderConfig(**tc_dict)
                if tc_dict
                else TemporalConfounderConfig()
            )
            shadow_config = ShadowRedHerringConfig(
                **shadow_kwargs, temporal_confounder=tc_config,
            )
            config_values["shadow_red_herrings"] = shadow_config
            used_from_yaml.append(
                f"shadow_red_herrings = enabled={shadow_config.enabled} "
                f"strength={shadow_config.confounder_strength_range} "
                f"p={shadow_config.modulation_probability} "
                f"temporal={tc_config.enabled}"
            )
            logger.info(
                f"✅ UNIFIED_CONFIG: Shadow red herrings: "
                f"enabled={shadow_config.enabled}, "
                f"strength={shadow_config.confounder_strength_range}, "
                f"modulation_probability={shadow_config.modulation_probability}, "
                f"temporal_confounder={tc_config.enabled}"
            )
            if tc_config.enabled:
                logger.info(
                    f"   Temporal confounder Z_k(t): "
                    f"correlation_length={tc_config.correlation_length}, "
                    f"alpha_truth={tc_config.alpha_truth}, "
                    f"alpha_shadow_range={tc_config.alpha_shadow_range}"
                )
        else:
            config_values["shadow_red_herrings"] = ShadowRedHerringConfig()
            logger.info(
                "ℹ️  UNIFIED_CONFIG: Shadow red herrings disabled (default)"
            )


        # Resume + sensor augmentation cannot coexist.
        #
        # `apply_augmentation` mints a fresh `augmented_inputs/aug_<ts>/` on every
        # run and rewrites each xlsx into it, then redirects
        # `sensor_data.folders` at it. So on a resumed run BOTH the source paths
        # and every source mtime change -- and `sensor_data` is part of the
        # feature fingerprint while `file_identity` is name|size|mtime. Every
        # phase-1 checkpoint would be refused, one by one, after the work to
        # discover it.
        #
        # Augmentation is documented as validation-only ("never enable in
        # production") and is enabled in exactly one config. Refuse the
        # combination outright rather than support it.

        # Handle target_labels (explicit label/target column definitions)
        target_labels_dict = get_nested_value(yaml_data, "target_labels")
        if target_labels_dict:
            # Parse from YAML: {column_name: {type: "binary"}, ...}
            target_labels_config = TargetLabelsConfig(columns=target_labels_dict)
            config_values["target_labels"] = target_labels_config
            used_from_yaml.append(
                f"target_labels = {list(target_labels_dict.keys())}"
            )
            logger.info(
                f"✅ UNIFIED_CONFIG: Target labels (explicit): {list(target_labels_dict.keys())}"
            )
        elif config_values.get("target_defect_rates"):
            # Auto-derive from target_defect_rates for SCM generation path
            rates = config_values["target_defect_rates"]
            derived_columns = {}
            for base_name in rates:
                derived_columns[f"{base_name}_defect"] = {"type": "continuous"}
                derived_columns[f"{base_name}_label"] = {"type": "binary"}
            config_values["target_labels"] = TargetLabelsConfig(columns=derived_columns)
            used_defaults.append(
                f"target_labels = derived from target_defect_rates: {list(derived_columns.keys())}"
            )
            logger.info(
                f"✅ UNIFIED_CONFIG: Target labels (derived from target_defect_rates): "
                f"{list(derived_columns.keys())}"
            )
        else:
            config_values["target_labels"] = TargetLabelsConfig()
            logger.info("ℹ️  UNIFIED_CONFIG: No target labels configured")

        # Create config object
        config = cls(**config_values)

        # Log summary
        logger.info(f"📊 UNIFIED_CONFIG: Configuration Summary:")
        logger.info(f"   Values from YAML: {len(used_from_yaml)}")
        logger.info(f"   Values from defaults: {len(used_defaults)}")

        if used_from_yaml:
            logger.info(f"✅ Used from YAML:")
            for item in used_from_yaml:
                logger.info(f"     {item}")

        if used_defaults:
            logger.info(f"⚠️  Used defaults:")
            for item in used_defaults:
                logger.info(f"     {item}")

        return config


    def to_scm_hyperparams(self) -> Dict[str, Any]:
        """
        Convert to hyperparams format for AdvancedExtrusionSCM.
        This REPLACES the complex ExtrustionSyntheticGenerator.
        """
        # Start with essential SCM hyperparams
        hyperparams = {
            # Core SCM settings
            "use_dynamic_graph": self.use_dynamic_graph,
            "vector_dim": self.vector_dim,
            # Keep the static defaults that SCM needs
            "num_samples_range": (100, 5000),
            "noise_level_range": (0.01, 0.2),
            "min_nodes": 15,
            "max_nodes": 40,
            "graph_alpha": 0.95,
            "graph_beta": 8.0,
            "edge_probability": 0.3,
            "max_parents": 5,
            # Material node parameters
            "recyclate_pct_range": (0, 100),
            "mfi_range": (0.2, 30),
            "moisture_range": (0.01, 2.0),
            "barrel_temp_range": (150, 350),
            "screw_speed_range": (20, 120),
            "pressure_range": (5, 100),
            "degassing_efficacy_range": (50, 99),
            "voc_content_range": (10, 2000),
            # Defect parameters
            "base_defect_probability": 0.2,
            "severity_scale": 10.0,
            "init_sigma": 1.0,
            "init_range": 1.0,
            "twin_screw_probability": 0.7,
            # Key relationships
            "key_relationships": {
                "moisture_to_fish_eye": 0.7,
                "contamination_to_black_spec": 0.85,
                "melt_temp_to_gel": 0.65,
                "die_temp_to_die_lines": 0.75,
            },
            # Defect thresholds
            "defect_thresholds": {
                "moisture_content": 0.8,
                "contamination_level": 0.5,
                "melt_temp_low": 220,
                "melt_temp_high": 300,
            },
        }

        return hyperparams


    def structure_seed(self) -> int:
        return self.seed if self.graph_seed is None else self.graph_seed



def get_nested_value(data: dict, path: str):
    """
    Get nested value from dict using dot notation.
    Returns None if path doesn't exist.
    """
    keys = path.split(".")
    current = data

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return None


# Replacement function for the old config loading
