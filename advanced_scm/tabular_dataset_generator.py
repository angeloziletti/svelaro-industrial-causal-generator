import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from io_utils import find_timeseries
from logger_config import get_logger
from advanced_scm.core import AdvancedExtrusionSCM
from advanced_scm.unified_config import UnifiedSCMConfig
from time_series.unified_temporal_processor import UnifiedTemporalProcessor

logger = get_logger(__name__)


class TabularDatasetGenerator:

    def __init__(self, output_dir: str, logger=None):
        """Initialize tabular dataset generator."""
        self.logger = get_logger(__name__)

        # Setup timestamped base directory
        self.base_output_dir = self._setup_output_directory(output_dir)

        # These will be set per variant in _setup_variant_directories()
        self.output_dir = None
        self.datasets_dir = None
        self.models_dir = None
        self.temporal_validation_dir = None
        self.reports_dir = None

    def _setup_variant_directories(self, variant_id: int, override_dir: Path = None) -> Path:
        """
        Setup variant-specific output directories.

        Parameters:
        -----------
        variant_id : int
            Variant ID (0 for base graph, >0 for variants)
        override_dir : Path, optional
            Override output directory (used by batch generator)

        Returns:
        --------
        variant_dir : Path
            Root directory for this variant
        """
        # Create variant-specific directory
        variant_dir = override_dir if override_dir is not None else self.base_output_dir / f"variant_{variant_id}"

        # Setup subdirectories for this variant
        self.output_dir = variant_dir
        self.datasets_dir = variant_dir / "datasets"
        self.models_dir = variant_dir / "models"
        self.temporal_validation_dir = variant_dir / "temporal_validation"
        self.reports_dir = variant_dir / "reports"

        # Create all directories
        for dir_path in [
            self.datasets_dir,
            self.models_dir,
            self.temporal_validation_dir,
            self.reports_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"📁 Created variant directory structure: {variant_dir}")

        return variant_dir

    def generate_dataset(
        self,
        unified_config: UnifiedSCMConfig,
        variant_id: int = 0,
        output_dir: Path = None,
    ) -> Dict[str, Any]:
        """
        Generate tabular dataset using unified configuration.

        Parameters:
        -----------
        unified_config : UnifiedSCMConfig
            Single source of truth configuration loaded from YAML
        variant_id : int
            Graph variant ID (0 = base graph; graph variants are not part of this release)
        output_dir : Path, optional
            Override output directory (used by batch generator)

        Returns:
        --------
        Dict containing generation results and metadata
        """
        self.logger.info(f"🚀 Generating tabular dataset (variant {variant_id})")
        if variant_id != 0:
            raise NotImplementedError("graph variants (variant_id > 0) are not part of this release")
        if unified_config.create_visualizations:
            raise NotImplementedError(
                "output.create_visualizations is not part of this release; set it to false"
            )
        generation_start_time = datetime.now()

        # Setup variant-specific directories
        variant_dir = self._setup_variant_directories(variant_id, override_dir=output_dir)

        # Graph, mechanisms and red-herring draw come from the structure seed;
        # the data seed is set below, after the SCM is fully built.
        np.random.seed(unified_config.structure_seed() + variant_id)

        # Initialize SCM using unified config
        scm = self._initialize_scm(unified_config)

        # Generate the plant. From here on every draw is data, not structure.
        np.random.seed(unified_config.seed + variant_id)
        dataset_info = self.generate_with_temporal_features(scm, unified_config)

        # No tabular feature dataset is written (dataset_path is empty), so the
        # label columns have no counted values.
        dataset_info["label_n_unique"] = {}

        # Save metadata
        generation_end_time = datetime.now()
        generation_duration = (
            generation_end_time - generation_start_time
        ).total_seconds()

        metadata = self._create_metadata(
            dataset_info,
            scm,
            unified_config,
            generation_start_time,
            generation_end_time,
            generation_duration,
            variant_id=variant_id,  # Pass variant_id to metadata
        )

        # Save metadata with standard name (not variant-specific)
        metadata_path = (
            self.datasets_dir / "dataset_metadata.json"
        )  # Changed from variant_0
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"✅ Dataset generation complete ({generation_duration:.1f}s)")
        self.logger.info(f"   Metadata saved: {metadata_path}")

        return {
            "output_directory": str(self.output_dir),
            "variant_directory": str(variant_dir),
            "datasets_directory": str(self.datasets_dir),
            "metadata_path": str(metadata_path),
            "generation_info": dataset_info,
            "metadata": metadata,
            "variant_id": variant_id,
        }




    def _initialize_scm(self, unified_config: UnifiedSCMConfig) -> AdvancedExtrusionSCM:
        """Initialize SCM using unified config - SIMPLIFIED."""
        self.logger.info("🔧 Initializing SCM from unified config...")

        # Convert unified config to SCM hyperparams format
        hyperparams = unified_config.to_scm_hyperparams()

        # Create SCM
        scm = AdvancedExtrusionSCM(
            hyperparams=hyperparams,
            vector_dim=unified_config.vector_dim,
            use_dynamic_graph=unified_config.use_dynamic_graph,
            target_defect_rates=unified_config.target_defect_rates,
            # GP parameter layer settings (for causal validation)
            enable_gp_layer=unified_config.enable_gp_layer,
            auto_select_temporal_sensors=unified_config.auto_select_temporal_sensors,
            min_temporal_sensors=unified_config.min_temporal_sensors,
            max_temporal_sensors=unified_config.max_temporal_sensors,
            min_base_nodes=unified_config.min_base_nodes,
            max_base_nodes=unified_config.max_base_nodes,
            temporal_sensors=unified_config.temporal_sensors,
            n_temporal_red_herrings=unified_config.n_temporal_red_herrings,
            # Shadow red-herring confounder structure (opt-in via config).
            shadow_red_herring_config=getattr(unified_config, "shadow_red_herrings", None),
            seed=unified_config.structure_seed(),
        )

        from advanced_scm import parameter_layer as pl

        scm.graph_generator.causal_params = pl.validate(unified_config.causal_params)
        if unified_config.trajectory_model not in ("single_pattern", "motif_sequence"):
            raise ValueError(
                f"parameter_layer.trajectory_model must be single_pattern or "
                f"motif_sequence, got {unified_config.trajectory_model!r}"
            )
        if unified_config.trajectory_model != "motif_sequence":
            raise NotImplementedError(
                "parameter_layer.trajectory_model single_pattern is not part of this release; "
                "use motif_sequence"
            )
        scm.graph_generator.trajectory_model = unified_config.trajectory_model
        scm.graph_generator.segments_per_sensor = tuple(unified_config.segments_per_sensor)
        vec = getattr(unified_config, "vector_scm", None)
        mech = getattr(vec, "target_mechanism", "trajectory") if vec is not None else "trajectory"
        if mech not in ("trajectory", "batch", "both", "scalar"):
            raise ValueError(
                f"vector_scm.target_mechanism must be trajectory/batch/both/scalar, got {mech!r}"
            )
        if mech != "trajectory":
            raise NotImplementedError(
                f"vector_scm.target_mechanism {mech!r} is not part of this release; use trajectory"
            )
        scm.graph_generator.base_edges = bool(getattr(unified_config, "base_edges", True))
        smech = getattr(vec, "scalar_mechanism", "parameters") if vec is not None else "parameters"
        if smech not in ("parameters", "dose"):
            raise ValueError(f"vector_scm.scalar_mechanism must be parameters/dose, got {smech!r}")
        if vec is not None and getattr(vec, "target_path", None):
            raise NotImplementedError(
                "vector_scm.target_path (transport delay, response, lab measurement) is not part "
                "of this release; set it to null"
            )
        if vec is not None and getattr(vec, "per_sample_center", False):
            raise NotImplementedError("vector_scm.per_sample_center is not part of this release")
        scm.graph_generator.sensor_graph = getattr(unified_config, "sensor_graph", None)
        scm.graph_generator.control_loops = getattr(unified_config, "control_loops", None)
        scm.graph_generator.n_truths = getattr(unified_config, "n_truths", None)
        scm.graph_generator.consequences = getattr(vec, "consequences", None) if vec is not None else None
        # a confounder share calibrates the Z_k route alone: Z_k gets a private root
        scm.graph_generator.independent_confounders = bool(
            vec is not None and (getattr(vec, "shares", None) or {}).get("confounder") is not None
        )
        # shared batch phases (vector_scm.shares.phase; number of phases: shares.phase_count, default 4)
        _sh = (getattr(vec, "shares", None) or {}) if vec is not None else {}
        scm.graph_generator.n_phases = int(_sh.get("phase_count", 4)) if _sh.get("phase") is not None else 0

        # Build causal graph
        scm.build_causal_graph()
        scm.define_node_mechanisms()
        # Hidden-confounder edges as explained-variance shares (advanced_scm/shares.py)
        if vec is not None and (getattr(vec, "shares", None) or {}).get("confounder") is not None:
            from advanced_scm import shares

            scm.graph.graph["confounder_shares"] = shares.calibrate_confounder_shares(
                scm, vec.shares["confounder"], unified_config.structure_seed()
            )
        # Shared batch phases: phase node -> segment onsets (graph_generators._inject_phase_nodes)
        if vec is not None and (getattr(vec, "shares", None) or {}).get("phase") is not None:
            from advanced_scm import shares

            scm.graph.graph["phase_shares"] = shares.calibrate_phase_shares(
                scm, vec.shares["phase"], unified_config.structure_seed()
            )
        # Recipe edges (echo share, batch-parameter route)
        if vec is not None and (getattr(vec, "shares", None) or {}).get("echo") is not None:
            from advanced_scm import shares

            scm.graph.graph["recipe_shares"] = shares.calibrate_recipe_shares(
                scm, vec.shares["echo"], unified_config.structure_seed()
            )

        self.logger.info("   Graph type: dynamic (TabPFN-inspired)")
        self.logger.info(
            f"   Graph: {len(scm.graph.nodes())} nodes, {len(scm.graph.edges())} edges"
        )

        return scm

    def generate_with_temporal_features(
        self,
        scm: AdvancedExtrusionSCM,
        unified_config: UnifiedSCMConfig,
    ) -> Dict:
        """Generate the plant: SCM, trajectories, target and ground-truth metadata."""
        self.logger.info(f"🚀 TEMPORAL PROCESSOR: Starting dataset processing...")
        self.logger.info(f"🔬 Method: {unified_config.temporal_generation_method}")

        temporal_processor = UnifiedTemporalProcessor(
            unified_config=unified_config,
            export_dir=self.temporal_validation_dir,
            logger_instance=self.logger,
        )

        gp_layer_path = (
            unified_config.enable_gp_layer
            and unified_config.temporal_generation_method == "pattern"
            and (
                bool(unified_config.temporal_sensors)
                or unified_config.auto_select_temporal_sensors
            )
        )
        if not gp_layer_path:
            raise NotImplementedError(
                "The generator requires enable_gp_layer=True + "
                "temporal_generation_method='pattern' + temporal sensors "
                "(or auto_select_temporal_sensors=True)."
            )

        pre_result = temporal_processor.generate_scm_and_export_timeseries(
            scm=scm,
            n_samples=unified_config.n_samples,
        )
        timeseries_path = pre_result["timeseries_path"]
        if timeseries_path is None:
            raise RuntimeError(
                "The time series export failed. Check earlier warnings."
            )

        run_result = {}
        result = self._merge_synthetic_metadata_into_result(
            run_result=run_result,
            scm=scm,
            pre_result=pre_result,
            unified_config=unified_config,
        )

        self.logger.info(f"   Method: {result['generation_method']}")
        return result




    def _merge_synthetic_metadata_into_result(
        self,
        run_result: Dict[str, Any],
        scm: AdvancedExtrusionSCM,
        pre_result: Dict[str, Any],
        unified_config: UnifiedSCMConfig,
    ) -> Dict:
        """Collect the result of the run and merge in the ground truth (vector_scm_audit,
        causal_sensors, causal_relationships, observable_graph, graph_structure,
        defect_registry)."""
        gen_info = run_result.get("generation_info", {}) or {}
        run_metadata = gen_info.get("pipeline_metadata", {}) or {}
        feature_metadata = run_metadata.get(
            "processing_result", {}
        ).get("feature_metadata", {}) or {}

        # Synthetic generation_metadata: keep causal_sensors + vector_scm_audit
        # (set when the target is generated) so _create_metadata can
        # surface them at the top level.
        synthetic_gen_metadata = dict(pre_result.get("generation_metadata", {}) or {})
        # Ensure causal_sensors is present (already in generation_metadata since
        # pre-extraction sets it, but be defensive).
        synthetic_gen_metadata.setdefault(
            "causal_sensors", pre_result.get("causal_sensors", []),
        )

        return {
            "generation_method": "synthetic_unified_pipeline",
            "dataset_path": gen_info.get("dataset_path", ""),
            "dataset_shape": gen_info.get("dataset_shape"),
            "feature_composition": gen_info.get("feature_composition", {}),
            "visualization": None,
            "pipeline_metadata": {
                "generation_metadata": synthetic_gen_metadata,
                "feature_metadata": feature_metadata,
                "export_metadata": run_metadata.get("export_metadata", {}),
                "processing_duration": run_metadata.get(
                    "processing_duration", 0.0
                ),
                "gp_ground_truth": synthetic_gen_metadata.get("gp_parameters"),
            },
        }

    def _build_defect_registry(
        self, scm: AdvancedExtrusionSCM, unified_config: UnifiedSCMConfig
    ) -> Dict:
        """Defect registry for the metadata. This release has no scalar defect
        nodes (target_defect_rates = {}, checked in core.AdvancedExtrusionSCM):
        defect_ts is the only target, so the registry is empty."""
        return {
            "defect_columns": [],
            "binary_label_columns": [],
            "config_base_names": [],
            "config_mapping": {},
            "terminal_validated": True,
            "generation_mode": "dynamic",
        }

    def _create_evaluation_grouping(
        self,
        scm: AdvancedExtrusionSCM,
        feature_classification: Dict[str, List[str]]
    ) -> Dict:
        """
        Create evaluation grouping metadata for temporal sensors.

        This maps each temporal sensor (temp_1, temp_2, etc.) to all related features:
        - GP parameter nodes (hidden in dataset, but in graph)
        - features extracted from time series (in dataset)

        For evaluation: discovering ANY feature related to temp_1 counts as discovering temp_1.

        Parameters:
        -----------
        scm : AdvancedExtrusionSCM
            The SCM instance with gp_parameter_nodes
        feature_classification : Dict
            Feature classification with temporal_features list

        Returns:
        --------
        evaluation_grouping : Dict
            Mapping of temporal sensors to their related features
        """
        grouping = {}
        temporal_features = feature_classification.get("temporal_features", [])

        for sensor_name, gp_nodes in scm.gp_parameter_nodes.items():
            # They will have names like: temp_1_bucket_1_cluster_*, temp_1_bucket_1_dim*
            related_features = [
                feat for feat in temporal_features
                if feat.startswith(f"{sensor_name}_")
            ]

            grouping[sensor_name] = {
                "gp_parameter_nodes": gp_nodes,
                "derived_features": related_features,
                "n_derived_features": len(related_features),
                "note": f"Discovering any of these features counts as discovering {sensor_name}"
            }

        self.logger.info(
            f"   📋 Created evaluation grouping for {len(grouping)} temporal sensors"
        )

        return grouping


    def _classify_features_by_type(self, dataset_info: Dict) -> Dict[str, List[str]]:
        """Feature classification of the tabular feature dataset. The plant writes
        no such dataset (dataset_path is empty), so every list is empty."""
        self.logger.warning("⚠️ No valid dataset path, cannot classify features")
        return {
            "temporal_features": [],
            "static_features": [],
            "defect_features": [],
            "total_features": 0,
        }

    @staticmethod
    def derive_causal_sensors(scm: AdvancedExtrusionSCM) -> List[str]:
        """The truth sensors: defect_ts is the only target (no scalar defect node)
        and its parents are the sensors that are not red herrings.

        This is the single source of truth for "truth sensors" in the SCM.
        Used by `_create_metadata` (to populate `causal_sensors` in
        dataset_metadata.json) and by the vector-SCM defect generator (to
        pick parents for the `defect_ts` mechanism).
        """
        red = set(scm.graph.graph.get("red_herring_sensors", []))
        return [s for s in scm.gp_parameter_nodes if s not in red]

    def _create_metadata(
        self,
        dataset_info: Dict,
        scm: AdvancedExtrusionSCM,
        unified_config: UnifiedSCMConfig,
        start_time: datetime,
        end_time: datetime,
        duration: float,
        variant_id: int = 0,
    ) -> Dict:
        """Create comprehensive metadata using unified config."""

        # Get causal relationships
        causal_relationships = scm.get_causal_relationships(
            estimate_strengths=True, max_path_length=5
        )

        # causal_sensors: the truth set for causal scoring AND the parent set for
        # the vector-SCM defect_ts mechanism (same derivation as in
        # unified_temporal_processor).
        causal_sensors: List[str] = self.derive_causal_sensors(scm)

        # Vector-SCM `defect_ts` is added downstream of the static SCM graph
        # (in UnifiedTemporalProcessor) so `scm.get_causal_relationships`
        # doesn't know about it — leaving `causal_relationships[defect_ts]`
        # absent. Downstream consumers (`true_direct_parents` lookup,
        # benchmark recovery scoring) silently get `[]` and report zero on
        # all metrics that depend on it.
        #
        # Mirror `process_defect`'s direct-cause schema (GP-parameter nodes
        # of the truth sensors). The vector mechanism operates on truth-
        # sensor time series, which are themselves controlled by these GP
        # params, so the causal parameter nodes of each truth sensor are
        # the right direct-cause set under this convention.
        vector_scm_cfg = getattr(unified_config, "vector_scm", None)
        if (
            vector_scm_cfg is not None
            and getattr(vector_scm_cfg, "enabled", False)
            and causal_sensors
            and "defect_ts" not in causal_relationships
        ):
            direct_causes = []
            for s in causal_sensors:
                direct_causes.extend(scm.gp_parameter_nodes.get(s, []))
            causal_relationships["defect_ts"] = {
                "direct_causes": direct_causes,
                "indirect_causes": [],
                "direct_strengths": {},
                "direct_strengths_source": {
                    n: "vector_scm_truth_sensor" for n in direct_causes
                },
            }

        # Build defect registry for metadata
        defect_registry = self._build_defect_registry(scm, unified_config)

        # Classify features by type (temporal vs static)
        feature_classification = self._classify_features_by_type(dataset_info)

        # Enhance feature_composition with classification
        enhanced_feature_composition = dataset_info.get("feature_composition", {}).copy()
        enhanced_feature_composition.update({
            "temporal_features": feature_classification["temporal_features"],
            "static_features": feature_classification["static_features"],
            "defect_features": feature_classification["defect_features"],
            "n_temporal_features": len(feature_classification["temporal_features"]),
            "n_static_features": len(feature_classification["static_features"]),
            "n_defect_features": len(feature_classification["defect_features"]),
        })

        # Auto-populate source_files with the long-format timeseries if it exists.
        source_files: List[str] = []
        timeseries_file = find_timeseries(self.output_dir)
        if timeseries_file is not None:
            source_files = [str(timeseries_file.resolve())]

        extra_target_labels: Dict[str, Dict] = {}

        # Surface vector-SCM audit (operator, edge_params, seed) and register
        # the new defect_ts target so downstream consumers can discover it via
        # metadata.target_labels.columns.
        pipeline_metadata = dataset_info.get("pipeline_metadata", {}) or {}
        gen_metadata = pipeline_metadata.get("generation_metadata", {}) or {}
        vector_scm_audit = gen_metadata.get("vector_scm_audit")

        # Variable-level ground truth: the full generative graph (SCM + sensors +
        # defect_ts + Z_k(t)) projected onto what a method observes, and the role
        # of every observed input for defect_ts (advanced_scm/ground_truth.py).
        observable_graph = None
        target_driver_labels = None
        if scm.gp_parameter_nodes:
            from advanced_scm import ground_truth as gt

            full_graph = gt.generative_graph(
                scm.graph,
                scm.node_types,
                scm.gp_parameter_nodes,
                causal_sensors,
                target="defect_ts" if vector_scm_audit else None,
                temporal_confounder_audit=gen_metadata.get("temporal_confounder_audit"),
                control_loops=scm.graph.graph.get("control_loops"),
                dynamic_edges=(scm.graph.graph.get("sensor_graph") or {}).get("dynamic_edges"),
                consequences=scm.graph.graph.get("consequences"),
                campaign=scm.graph.graph.get("campaign"),
            )
            hidden = gt.hidden_nodes(
                full_graph, getattr(unified_config, "observability", None)
            )
            observable_graph = gt.observable_projection(full_graph, hidden)
            shadow_audit = scm.graph.graph.get("shadow_red_herring_audit") or {}
            shadow_of = {
                r: t
                for t, group in (shadow_audit.get("shadow_groups") or {}).items()
                for r in group
            }
            if vector_scm_audit:
                target_driver_labels = gt.target_driver_labels(
                    full_graph,
                    hidden,
                    "defect_ts",
                    shadow_of=shadow_of,
                    control_loops=scm.graph.graph.get("control_loops"),
                    # sensors read with measurement error (Step 2.45)
                    measurement_noisy=[s for s, m in (gen_metadata.get("measurement_noise") or {}).items() if m],
                    # operator reaction to quality closes a loop defect -> cause (Step 2.41)
                    operator_reaction=gen_metadata.get("operator_reaction"),
                )
                self.logger.info(
                    f"   🎯 defect_ts driver labels: {target_driver_labels['counts']}"
                )
                # A planned consequence must come out of the ground truth as a
                # consequence (downstream of the defect only). Anything else means the
                # planner and the graph disagree: fail, never label silently.
                wrong = {
                    c["sensor"]: target_driver_labels["labels"].get(c["sensor"], {}).get("label")
                    for c in scm.graph.graph.get("consequences") or []
                    if target_driver_labels["labels"].get(c["sensor"], {}).get("label") != "consequence"
                }
                if wrong:
                    raise RuntimeError(f"planned consequences not labelled consequence: {wrong}")
        if vector_scm_audit is not None:
            extra_target_labels["defect_ts"] = {
                "type": "continuous_timeseries",
                "purpose": "vector_scm_target",
                "detection_method": "derived_from_vector_mechanism",
            }
            extra_target_labels["defect_ts_mean"] = {
                "type": "continuous",
                "purpose": "vector_scm_scalar_target",
                "detection_method": "post_discretization_aggregation",
            }

        return {
            "generation_info": {
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "generator_version": "unified_v1.0",
                "generator_type": "TabularDatasetGenerator",
                "config_source": "UnifiedSCMConfig",
                "variant_id": variant_id,
            },
            "source_files": source_files,
            "dataset_info": {
                "generation_method": dataset_info["generation_method"],
                "dataset_shape": dataset_info["dataset_shape"],
                "feature_composition": enhanced_feature_composition,
                "dataset_path": dataset_info["dataset_path"],
            },
            "configuration_used": {
                # Key config values that were actually used
                "sequence_length": unified_config.sequence_length,
                "n_samples": unified_config.n_samples,
                "seed": unified_config.seed,  # data seed
                "graph_seed": unified_config.structure_seed(),
                "temporal_generation_method": unified_config.temporal_generation_method,
                "target_defect_rates": unified_config.target_defect_rates,
                "use_dynamic_graph": unified_config.use_dynamic_graph,
                "vector_dim": unified_config.vector_dim,
            },
            "graph_info": {
                "nodes": len(scm.graph.nodes()),
                "edges": len(scm.graph.edges()),
                "graph_type": "dynamic",
            },
            "graph_structure": scm.get_graph_structure(),
            "causal_relationships": causal_relationships,
            "causal_sensors": causal_sensors,
            "observable_graph": observable_graph,
            "target_driver_labels": target_driver_labels,
            "evaluation_grouping": self._create_evaluation_grouping(scm, feature_classification) if scm.gp_parameter_nodes else None,
            "target_labels": {
                "columns": {
                    **{
                        col: {
                            **props,
                            "n_unique": dataset_info.get("label_n_unique", {}).get(col),
                            "detection_method": "explicit_config",
                        }
                        for col, props in unified_config.target_labels.columns.items()
                    },
                    **extra_target_labels,
                }
            },
            "defect_registry": defect_registry,
            # Vector-SCM audit: operator, per-edge params, seeds, jitter config.
            # Surfaced top-level for easy inspection / reproducibility. Same dict
            # is also under pipeline_metadata.generation_metadata.vector_scm_audit.
            "vector_scm_audit": vector_scm_audit,
            # Shadow red-herring audit. Lists per-truth confounder
            # nodes, shadow group assignments, and per-edge Z_k → GP-param
            # weights. None when shadow_red_herrings.enabled = False.
            "shadow_red_herring_audit": (
                scm.graph.graph.get("shadow_red_herring_audit")
                if hasattr(scm, "graph") and hasattr(scm.graph, "graph")
                else None
            ),
            # Temporal confounder audit. Per-truth Z_k(t) seeds
            # used + per-shadow alpha values. Lives in
            # generation_metadata["temporal_confounder_audit"] set by
            # unified_temporal_processor Step 2.35. None when
            # temporal_confounder.enabled = False.
            "temporal_confounder_audit": gen_metadata.get(
                "temporal_confounder_audit"
            ),
            # Realized signal-to-noise per sensor (event shape vs stochastic
            # residual; pattern_settings.residual_scale is the dial) and the
            # (mean, sd) each exported sensor was z-scored with.
            "sensor_snr": gen_metadata.get("sensor_snr"),
            "measurement_noise": gen_metadata.get("measurement_noise"),
            "operator_reaction": gen_metadata.get("operator_reaction"),
            # Parameter-mode sensor -> sensor edges (order, upstream ancestors, and
            # which parameter feeds which); None when sensor_graph is off.
            "sensor_graph": scm.graph.graph.get("sensor_graph"),
            # motif_sequence model: each sensor's fixed motif sequence (structure).
            "sensor_profiles": scm.graph.graph.get("sensor_profiles"),
            # Control loops: controlled / disturbance / actuator sensors and tightness.
            "control_loops": scm.graph.graph.get("control_loops"),
            # Every sensor a method may use as data (valves included) ...
            "sensor_universe": list(scm.gp_parameter_nodes or []),
            # ... its a-priori role (actuator = controller output, known from the P&ID) ...
            "variable_tags": {
                s: (
                    "actuator"
                    if any(l["actuator"] == s for l in scm.graph.graph.get("control_loops") or [])
                    else "process"
                )
                for s in (scm.gp_parameter_nodes or {})
            },
            # ... and the inputs that are ranked and scored as candidate causes
            # (observed inputs minus actuators).
            "candidate_causes": (target_driver_labels or {}).get("candidate_causes"),
            "sensor_standardization": gen_metadata.get("sensor_standardization"),
            "pipeline_metadata": dataset_info.get("pipeline_metadata", {}),
            # Extract feature_metadata to top level for causal analyzer (includes dim_features_by_bucket)
            "feature_metadata": dataset_info.get("pipeline_metadata", {}).get("feature_metadata", {}),
            "file_info": {
                "datasets_directory": str(self.datasets_dir),
                "models_directory": str(self.models_dir),
                "reports_directory": str(self.reports_dir),
                "temporal_validation_directory": str(self.temporal_validation_dir),
            },
        }

    def _setup_output_directory(self, output_dir: str) -> Path:
        """Setup output directory with timestamping if needed.
        not"""
        output_path = Path(output_dir)

        if output_path.exists() and any(output_path.iterdir()):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_dir = output_path / f"dataset_{timestamp}"
            self.logger.info(
                f"📁 Directory exists, creating timestamped: {timestamped_dir}"
            )
            return timestamped_dir
        else:
            return output_path
