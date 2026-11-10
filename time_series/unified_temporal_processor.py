"""Turns a plant's SCM parameters into batch trajectories and the target, and exports them."""

import pandas as pd
import numpy as np

import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from logger_config import get_logger
from advanced_scm.core import AdvancedExtrusionSCM
from time_series.generation.temporal_generator import TemporalGenerator
from time_series.export.data_exporter import TemporalDataExporter

logger = get_logger(__name__)




class UnifiedTemporalProcessor:
    """Generates the trajectories of one plant from its SCM."""

    def __init__(self, unified_config, export_dir: Path, logger_instance=None):
        """Initialize with unified config only."""
        self.unified_config = unified_config
        self.export_dir = Path(export_dir)
        self.logger = logger_instance or logger

        self.method = unified_config.temporal_generation_method
        self.export_enabled = unified_config.export_temporal_data

        # Initialize the extracted export module
        self.exporter = TemporalDataExporter(
            unified_config=unified_config,
            export_dir=export_dir,
            logger_instance=self.logger,
        )

        self.generator = TemporalGenerator(
            unified_config=unified_config,
            exporter=self.exporter,
            logger_instance=self.logger,
        )

        self.logger.debug(f"   Method: {self.method}")
        self.logger.debug(f"   Export enabled: {self.export_enabled}")

    def _has_temporal_sensors(self) -> bool:
        """Check if temporal sensors are configured."""
        has_sensors = (
            self.unified_config.temporal_sensors is not None
            and len(self.unified_config.temporal_sensors) > 0
        )
        return has_sensors



    def generate_scm_and_export_timeseries(
        self,
        scm: "AdvancedExtrusionSCM",
        n_samples: int,
    ) -> Dict[str, Any]:
        """Generates the SCM dataset, the trajectories and the target, and writes the
        long-format time series with the datasets folder.

        Returns:
            {
                "scm_df": pd.DataFrame,             # sample-level scalars
                "structured_time_series": Dict[str, np.ndarray],
                "generation_metadata": Dict,
                "timeseries_path": Optional[Path],
                "causal_sensors": List[str],
            }
        """
        # Auto-populate temporal sensors from SCM if GP layer uses auto-selection
        if (
            self.unified_config.enable_gp_layer
            and self.unified_config.auto_select_temporal_sensors
            and scm.gp_parameter_nodes
        ):
            temporal_sensors_from_scm = list(scm.gp_parameter_nodes.keys())
            if temporal_sensors_from_scm:
                self.logger.info(
                    f"🎯 Auto-selected temporal sensors from SCM: {temporal_sensors_from_scm}"
                )
                self.unified_config.temporal_sensors = temporal_sensors_from_scm

        gp_layer_enabled = (
            self.unified_config.enable_gp_layer
            and self.method == "pattern"
            and self._has_temporal_sensors()
        )
        if not gp_layer_enabled:
            raise NotImplementedError(
                "generate_scm_and_export_timeseries requires enable_gp_layer=True "
                "+ method='pattern' + temporal sensors."
            )

        self.logger.info("🎛️ GP PARAMETER LAYER ENABLED — Reversed flow")
        self.logger.info(
            "   Flow: SCM → GP-Controlled Patterns → Vector-SCM defect_ts → Excel"
        )

        # Step 1: Generate SCM dataset (sample-level scalars + GP parameter values)
        self.logger.info("🎯 Step 1: Generating SCM with GP parameters")
        self.logger.info("   Using SCM defaults")
        scm_df = self.generator.generate_scm_dataset_without_constraints(
            scm=scm, n_samples=n_samples
        )

        # Steps 2-2.4: trajectories and defect_ts from the SCM values. A separate
        # method so the ground-truth audit can replay it under interventions.
        structured_time_series, generation_metadata, causal_sensors = (
            self.simulate_trajectories(scm, scm_df, n_samples)
        )

        # Step 2.5: Export the long-format time series + datasets folder
        timeseries_path: Optional[Path] = None
        if self.export_enabled:
            self.logger.info(
                "💾 Step 2.5: Exporting the time series + datasets folder"
            )
            datasets_dir = self.export_dir / "datasets"
            datasets_dir.mkdir(parents=True, exist_ok=True)

            scm_metadata = scm_df.to_dict("records")
            self.exporter.export_temporal_sample_from_structured(
                structured_time_series,
                temporal_feature_names=self.unified_config.temporal_sensors,
                export_dir=datasets_dir,
                export_full_dataset=True,
                include_scm_metadata=scm_metadata,
            )

            # Methods see only what a plant would record (advanced_scm/observability.py).
            # The hidden SCM values go to an oracle sidecar, never into the
            # timeseries a method reads.
            from advanced_scm.observability import hidden_scm_columns

            hidden_cols = [
                c
                for c in hidden_scm_columns(
                    scm.node_types,
                    getattr(self.unified_config, "observability", None),
                )
                if c in scm_df.columns
            ]
            observed_scm_df = scm_df.drop(columns=hidden_cols)

            from advanced_scm.observability import standardize_sensors

            exported_series, sensor_scales = standardize_sensors(
                structured_time_series,
                list(scm.gp_parameter_nodes.keys()),
                getattr(self.unified_config, "observability", None),
            )
            generation_metadata["sensor_standardization"] = sensor_scales
            if hidden_cols:
                oracle_path = datasets_dir / "oracle_parameters.parquet"
                scm_df[hidden_cols].rename_axis("sample_index").reset_index().to_parquet(
                    oracle_path, index=False
                )
                self.logger.info(
                    f"   🙈 {len(hidden_cols)} SCM column(s) hidden from methods, "
                    f"exported as oracle: {oracle_path.name}"
                )

            try:
                label_cols = list(
                    self.unified_config.target_labels.columns.keys()
                )
                timeseries_path = (
                    self.exporter.export_timeseries(
                        structured_time_series=exported_series,
                        tabular_df=observed_scm_df,
                        label_columns=label_cols,
                        samples_per_context=int(
                            getattr(self.unified_config, "samples_per_context", 10)
                        ),
                    )
                )
                self._timeseries_path = timeseries_path
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Failed to export long-format timeseries: {e}"
                )
                self._timeseries_path = None

            # Export GP parameter ground-truth sidecar
            if "gp_parameters" in generation_metadata:
                gp_ground_truth_path = (
                    datasets_dir / "gp_parameters_ground_truth.json"
                )
                with open(gp_ground_truth_path, "w") as f:
                    json.dump(
                        {
                            "gp_parameters": generation_metadata["gp_parameters"],
                            "gp_parameter_nodes": generation_metadata[
                                "gp_parameter_nodes"
                            ],
                            "gp_control_stats": generation_metadata[
                                "gp_control_stats"
                            ],
                        },
                        f,
                        indent=2,
                    )
                self.logger.info(
                    f"✅ GP parameters ground truth exported: {gp_ground_truth_path}"
                )

        return {
            "scm_df": scm_df,
            "structured_time_series": structured_time_series,
            "generation_metadata": generation_metadata,
            "timeseries_path": timeseries_path,
            "causal_sensors": causal_sensors,
        }

    def simulate_trajectories(
        self,
        scm: "AdvancedExtrusionSCM",
        scm_df: pd.DataFrame,
        n_samples: int,
        open_loop_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Steps 2-2.4: sensor trajectories and defect_ts from the SCM sample table.

        Deterministic given (scm, scm_df, seeds): every draw is seeded per sample
        and sensor, so replaying it with a modified scm_df is a counterfactual
        with the same noise (advanced_scm/ground_truth_audit.py).
        `open_loop_overrides` maps a sensor to a function applied to its
        (n_samples, T) open-loop trajectory before control loops close and
        defect_ts is computed: an intervention on that observed sensor.

        Returns (structured_time_series, generation_metadata, causal_sensors).
        """
        # Step 2: Generate GP-controlled pattern time series
        import os as _os
        sensor_n_jobs = int(getattr(self.unified_config, "sensor_n_jobs", 1))
        resolved_cores = (
            _os.cpu_count() if sensor_n_jobs == -1 else sensor_n_jobs
        )
        self.logger.info(
            f"🔬 Step 2: Generating GP-controlled pattern time series "
            f"(sensor_n_jobs={sensor_n_jobs} → {resolved_cores} workers; "
            f"cpu_count={_os.cpu_count()})"
        )
        generation_metadata, structured_time_series = (
            self.generator.generate_gp_controlled_pattern_timeseries(
                scm_df, scm, n_samples, self.export_dir,
                sensor_n_jobs=sensor_n_jobs,
            )
        )

        # Derive causal_sensors — single source of truth, used by the
        # vector-SCM defect generator and the metadata.
        from advanced_scm.tabular_dataset_generator import TabularDatasetGenerator
        causal_sensors: List[str] = TabularDatasetGenerator.derive_causal_sensors(scm)
        generation_metadata["causal_sensors"] = causal_sensors

        # Step 2.35: Temporal confounder Z_k(t) — applied to BOTH
        # truth and shadow time series BEFORE the vector mechanism runs, so
        # defect_ts inherits Z_k(t) via truth_ts and shadow_ts share Z_k(t)
        # directly. Gated on shadow_red_herrings.temporal_confounder.enabled.
        shadow_cfg = getattr(self.unified_config, "shadow_red_herrings", None)
        tc_cfg = (
            getattr(shadow_cfg, "temporal_confounder", None)
            if shadow_cfg is not None else None
        )
        if (
            tc_cfg is not None
            and getattr(tc_cfg, "enabled", False)
            and hasattr(scm, "graph")
            and hasattr(scm.graph, "graph")
        ):
            shadow_audit = scm.graph.graph.get("shadow_red_herring_audit")
            if shadow_audit and shadow_audit.get("enabled"):
                from advanced_scm.temporal_confounder import (
                    apply_temporal_confounders_to_timeseries,
                )
                temporal_audit = apply_temporal_confounders_to_timeseries(
                    structured_time_series=structured_time_series,
                    shadow_groups=shadow_audit.get("shadow_groups", {}),
                    cfg=tc_cfg,
                    seed=int(self.unified_config.seed),
                    structure_seed=int(self.unified_config.structure_seed()),
                    logger=self.logger,
                    confounder_shares=scm.graph.graph.get("confounder_shares"),
                )
                generation_metadata["temporal_confounder_audit"] = temporal_audit
            else:
                self.logger.warning(
                    "⚠️ Step 2.35: temporal_confounder.enabled=True but no "
                    "shadow_red_herring_audit on the SCM graph. Skipping Z_k(t)."
                )

        # Step 2.36: raw-material lots and campaign drift (advanced_scm/campaign.py): shares of
        # every sensor's batch-to-batch variation, on its own trajectory, before the coupling,
        # the control loops and the target. Scales frozen on the first (baseline) call.
        _vec = getattr(self.unified_config, "vector_scm", None)
        from advanced_scm import campaign

        camp = campaign.settings(getattr(_vec, "shares", None) if _vec is not None else None,
                                 self.unified_config.structure_seed())
        if camp is not None:
            structured_time_series, _frozen = campaign.apply(
                structured_time_series, camp, scm.graph.graph.get("campaign_frozen"),
                self.unified_config.seed, self.unified_config.structure_seed(),
            )
            scm.graph.graph.setdefault("campaign_frozen", _frozen)
            scm.graph.graph.setdefault("campaign_realised", campaign.realised(structured_time_series, _frozen))
            scm.graph.graph["campaign"] = camp
            self.logger.info(f"🏭 Step 2.36: lots (share {camp['lot']}, persistence {camp['lot_persistence']}) "
                             f"and drift (share {camp['drift']}) on {len(structured_time_series)} sensors")

        vec_cfg = getattr(self.unified_config, "vector_scm", None)

        # Interventions on observed sensors act on each sensor's own trajectory,
        # before the dynamic coupling and the control loops propagate them
        # (advanced_scm/ground_truth_audit.py).
        for sname, fn in (open_loop_overrides or {}).items():
            structured_time_series[sname] = fn(structured_time_series[sname])

        # Step 2.39b: dynamic sensor -> sensor edges (advanced_scm/sensor_dynamics.py):
        # each child follows its parents' trajectories, delayed and smoothed.
        sg = scm.graph.graph.get("sensor_graph") or {}
        if sg.get("dynamic_edges"):
            from advanced_scm.sensor_dynamics import apply_dynamic_edges_shared

            echo_spec = (getattr(getattr(self.unified_config, "vector_scm", None), "shares", None) or {}).get("echo")
            if echo_spec is None:
                raise ValueError("sensor_graph.dynamic_edges require vector_scm.shares.echo")
            structured_time_series, frozen = apply_dynamic_edges_shared(
                structured_time_series, sg["dynamic_edges"], sg["order"], echo_spec,
                self.unified_config.structure_seed(), scm.graph.graph.get("echo_frozen_scales"),
            )
            scm.graph.graph.setdefault("echo_frozen_scales", frozen)
            self.logger.info(
                f"🔗 Step 2.39b: {len(sg['dynamic_edges'])} dynamic sensor edge(s) applied"
            )

        # Step 2.395: closed-loop control (advanced_scm/control_loops.py): the
        # controlled sensors track their setpoint profile, the actuators absorb
        # the disturbance.
        loops = scm.graph.graph.get("control_loops") or []
        open_loop_series = None
        open_loop_defect = None
        if loops:
            from advanced_scm.control_loops import apply_control_loops

            # The same loops left open (tightness 0: the disturbance flows in, nothing
            # corrects it). The defect's scales are frozen on this plant, so a share
            # describes the causes' physical strength and a tight loop lowers a
            # controlled cause's realised share, as in a real plant.
            open_loop_series = apply_control_loops(
                structured_time_series, [{**l, "tightness": 0.0} for l in loops]
            )
            structured_time_series = apply_control_loops(structured_time_series, loops)
            self.logger.info(
                f"🎛️ Step 2.395: {len(loops)} control loop(s) closed: "
                + ", ".join(
                    f"{l['controlled']} (disturbance {l['disturbance']}, actuator "
                    f"{l['actuator']}, tightness {l['tightness']})"
                    for l in loops
                )
            )

        # Step 2.4: The target as a time series, defect_ts.
        # target_mechanism (trajectory), target_path (none) and per_sample_center (off)
        # are validated in TabularDatasetGenerator._initialize_scm.
        mech = getattr(vec_cfg, "target_mechanism", "trajectory") if vec_cfg is not None else "trajectory"
        if vec_cfg is not None and vec_cfg.enabled and causal_sensors:
            from advanced_scm.vector_scm import generate_defect_ts

            path = getattr(vec_cfg, "target_path", None) or {}
            if open_loop_series is not None and scm.graph.graph.get("target_parent_scales") is None:
                # Freeze the defect's scales on the open-loop plant (see Step 2.395);
                # at tightness 0 this is the plant itself, so nothing changes.
                open_loop_defect, open_audit = generate_defect_ts(
                    cfg=vec_cfg, structured_time_series=open_loop_series,
                    truth_sensors=causal_sensors, seed=self.unified_config.structure_seed(),
                    noise_seed=self.unified_config.seed,
                )
                scm.graph.graph["target_parent_scales"] = open_audit.get("parent_scales") or None
                scm.graph.graph["target_signal_scale"] = open_audit.get("signal_scale")
                scm.graph.graph["target_noise_tau"] = open_audit.get("noise_tau")
                scm.graph.graph["target_part_scales"] = open_audit.get("part_scales")
            defect_ts, vector_scm_audit = generate_defect_ts(
                cfg=vec_cfg,
                structured_time_series=structured_time_series,
                truth_sensors=causal_sensors,
                seed=self.unified_config.structure_seed(),
                noise_seed=self.unified_config.seed,
                # Frozen after the first (baseline) computation on this SCM, so
                # the audit's interventions reuse the baseline's scales.
                parent_scales=scm.graph.graph.get("target_parent_scales"),
                signal_scale=scm.graph.graph.get("target_signal_scale"),
                noise_tau=scm.graph.graph.get("target_noise_tau"),
                part_scales=scm.graph.graph.get("target_part_scales"),
            )
            scm.graph.graph.setdefault(
                "target_parent_scales", vector_scm_audit.get("parent_scales") or None
            )
            scm.graph.graph.setdefault("target_signal_scale", vector_scm_audit.get("signal_scale"))
            scm.graph.graph.setdefault("target_noise_tau", vector_scm_audit.get("noise_tau"))
            scm.graph.graph.setdefault("target_part_scales", vector_scm_audit.get("part_scales"))
            # Step 2.41: operator reaction to quality (vector_scm.shares.operator = r). The
            # operator watches the defect and, one defect time scale later, moves the
            # setpoint of one true cause against it: the cause's reading becomes
            # sqrt(1 - r) * own + sqrt(r) * reaction, reaction = -sign(gain) * the defect
            # through a first-order lag of tau_defect after a dead time of tau_defect. A
            # feedback loop defect -> cause (sigma-separation in the ground truth). Solved
            # forward in time: each pass fixes another tau_defect steps, so ceil(T / tau) + 1
            # passes reproduce the closed loop exactly. Scales frozen on the reaction-free plant.
            op_spec = (getattr(vec_cfg, "shares", None) or {}).get("operator")
            # The lever must act monotonically on the defect (linear, saturation, threshold):
            # on an optimum (U-shaped) cause "against the defect" has no direction and the
            # loop can run away. A plant whose causes are all U-shaped gets no reaction.
            levers = sorted(s for s, e in (vector_scm_audit.get("edges") or {}).items()
                            if e.get("nonlinearity") in ("linear", "saturation", "threshold"))
            if op_spec is not None and vector_scm_audit.get("edges") and not levers:
                generation_metadata["operator_reaction"] = {"sensor": None, "reason": "no monotone cause"}
                self.logger.info("🧑‍🏭 Step 2.41: operator reaction skipped (every cause is U-shaped)")
            if op_spec is not None and levers:
                from advanced_scm import shares
                from advanced_scm.control_loops import fopdt

                r = shares.draw(op_spec, self.unified_config.structure_seed(), "operator")
                g = scm.graph.graph

                def seen(d, tau):
                    # what the operator has seen: the defect through a first-order lag, tau
                    # steps late; nothing before the batch starts (no reaction yet), so the
                    # loop has no instantaneous path at t = 0
                    y = fopdt(d, float(tau), 0)
                    return np.concatenate([np.zeros_like(y[:, :tau]), y[:, :-tau]], axis=1)

                if "operator_frozen" not in g:
                    rng = np.random.default_rng(self.unified_config.structure_seed() + 9433)
                    lever = levers[int(rng.integers(len(levers)))]
                    tau_d = int(max(shares.process_tau(defect_ts), 1))
                    react0 = seen(defect_ts, tau_d)[:, tau_d:]  # scale of a running reaction
                    g["operator_frozen"] = {"sensor": lever, "tau": tau_d, "share": r,
                                            "sign": -float(np.sign(vector_scm_audit["edges"][lever]["gain"]) or 1.0),
                                            "react_scale": shares.scale_of(react0),
                                            "own_scale": shares.scale_of(structured_time_series[lever])}
                op = g["operator_frozen"]
                own = structured_time_series[lever := op["sensor"]]
                T = own.shape[1]
                d = defect_ts
                for _ in range(int(np.ceil(T / op["tau"])) + 1):
                    r_ = seen(d, op["tau"])
                    r_[:, :op["tau"]] = op["react_scale"][0]  # standardises to zero: no reaction yet
                    react = op["sign"] * shares.standardise(r_, op["react_scale"])
                    mu, sd = op["own_scale"]
                    moved = dict(structured_time_series)
                    moved[lever] = mu + sd * shares.mix(react, shares.standardise(own, op["own_scale"]), op["share"])
                    d_new, vector_scm_audit = generate_defect_ts(
                        cfg=vec_cfg, structured_time_series=moved, truth_sensors=causal_sensors,
                        seed=self.unified_config.structure_seed(), noise_seed=self.unified_config.seed,
                        parent_scales=g.get("target_parent_scales"), signal_scale=g.get("target_signal_scale"),
                        noise_tau=g.get("target_noise_tau"), part_scales=g.get("target_part_scales"),
                    )
                    done = np.array_equal(d_new, d)
                    d = d_new
                    if done:
                        break
                if not done or not np.all(np.isfinite(d)):
                    raise RuntimeError(f"operator reaction on {lever}: closed loop did not settle "
                                       f"(converged={done}, finite={bool(np.all(np.isfinite(d)))})")
                structured_time_series = moved
                defect_ts = d
                vector_scm_audit["operator_reaction"] = {k: op[k] for k in ("sensor", "tau", "share", "sign")}
                generation_metadata["operator_reaction"] = vector_scm_audit["operator_reaction"]
                self.logger.info(f"🧑‍🏭 Step 2.41: operator reacts to quality on {lever} (share {r}, tau {op['tau']})")
            vector_scm_audit["target_mechanism"] = mech
            if vector_scm_audit.get("edges"):
                # Ground truth: each cause's unique share of the quality, from the
                # equation and the curves it acted on.
                from advanced_scm.target_edges import unique_shares

                # Ground truth: the shares this plant drew (fixed or from a range), so a
                # difficulty cloud can be analysed against them.
                g = scm.graph.graph
                vector_scm_audit["realised_shares"] = {
                    "defect_timeseries": vector_scm_audit.get("share_defect_timeseries"),
                    "interaction": vector_scm_audit.get("interaction_share"),
                    "confounder": {k: v.get("share") for k, v in (g.get("confounder_shares") or {}).items()},
                    "echo_recipe": {k: v.get("share") for k, v in (g.get("recipe_shares") or {}).items()},
                    "echo_dynamic": {k: v.get("share") for k, v in (g.get("echo_frozen_scales") or {}).items()},
                    # shared batch phases: phase node -> segment onset (share of the onset's variance)
                    "phase": {k: v.get("share") for k, v in (g.get("phase_shares") or {}).items()} or None,
                    # raw-material lots and campaign drift across episodes
                    "lot": (g.get("campaign") or {}).get("lot"),
                    "lot_persistence": (g.get("campaign") or {}).get("lot_persistence"),
                    "drift": (g.get("campaign") or {}).get("drift"),
                    # per sensor, measured on its own trajectory right after the step
                    "campaign_realised": g.get("campaign_realised"),
                    # realism knobs on the cause -> defect edges (target_edges.cause_response)
                    "delay_jitter": vector_scm_audit.get("delay_jitter"),
                    "cumulative": {k: e.get("cumulative_share") for k, e in vector_scm_audit["edges"].items()
                                   if e.get("cumulative_share") is not None} or None,
                    "context": vector_scm_audit.get("context"),
                }
                if vector_scm_audit.get("context"):
                    from advanced_scm.target_edges import batch_grades

                    vector_scm_audit["batch_grades"] = batch_grades(vector_scm_audit, defect_ts.shape[0]).astype(int).tolist()
                vector_scm_audit["cause_unique_share"] = unique_shares(
                    {c: structured_time_series[c] for c in vector_scm_audit["edges"]},
                    defect_ts, vector_scm_audit,
                )
                if open_loop_series is not None and open_loop_defect is not None:
                    # The same causes with the loops left open (tightness 0): how much of
                    # each cause's strength the control loops remove.
                    vector_scm_audit["cause_unique_share_open_loop"] = unique_shares(
                        {c: open_loop_series[c] for c in vector_scm_audit["edges"]},
                        open_loop_defect, vector_scm_audit,
                    )
            vector_scm_audit["target_path"] = path
            consequences = scm.graph.graph.get("consequences") or []
            if consequences:
                # Sensors downstream of the defect read the quality. Scale frozen from
                # the baseline, like the parent scales, so an intervention does not
                # rescale other batches.
                from advanced_scm import shares
                from advanced_scm.target_edges import apply_consequences

                scale = scm.graph.graph.get("consequence_scale") or [
                    float(np.mean(defect_ts)), float(np.std(defect_ts))
                ]
                scm.graph.graph.setdefault("consequence_scale", scale)
                spec = (getattr(vec_cfg, "shares", None) or {}).get("consequence")
                if any("dead_time_factor" in c for c in consequences):
                    # Delays as a multiple of the defect's process time scale (frozen on the
                    # baseline), with the same function the evaluation uses for the resolution
                    # (one point per tau rounded down): a delay >= tau is always >= one step, so
                    # a consequence's lag is visible to every method and time order holds.
                    tau_d = scm.graph.graph.get("consequence_defect_tau") or int(shares.process_tau(defect_ts))
                    scm.graph.graph.setdefault("consequence_defect_tau", tau_d)
                    consequences = [
                        {**c, "dead_time": int(np.ceil(c["dead_time_factor"] * tau_d)), "defect_tau": tau_d}
                        if "dead_time_factor" in c else c for c in consequences
                    ]
                consequences = [
                    {**c, "share": shares.draw(spec, self.unified_config.structure_seed(), "consequence", i)}
                    for i, c in enumerate(consequences)
                ]
                updated, frozen = apply_consequences(
                    structured_time_series, defect_ts, consequences, scale,
                    scm.graph.graph.get("consequence_frozen_scales"),
                )
                scm.graph.graph.setdefault("consequence_frozen_scales", frozen)
                structured_time_series.update(updated)
                vector_scm_audit["consequences"] = consequences
                if "realised_shares" in vector_scm_audit:
                    vector_scm_audit["realised_shares"]["consequence"] = [c.get("share") for c in consequences]
            structured_time_series["defect_ts"] = defect_ts
            generation_metadata["vector_scm_audit"] = vector_scm_audit
            self.logger.info(
                f"🧮 Step 2.4: defect_ts generated via "
                f"{vec_cfg.edge_model} edges on {len(causal_sensors)} truth "
                f"sensor(s); shape={defect_ts.shape}, "
                f"noise_sigma={vec_cfg.noise_sigma}"
            )
        elif vec_cfg is not None and vec_cfg.enabled and not causal_sensors:
            self.logger.warning(
                "⚠️ Step 2.4: vector_scm.enabled but no causal sensors found "
                "(gp_parameter_nodes empty or no defect edges). Skipping defect_ts."
            )

        # Step 2.45: measurement noise (vector_scm.shares.measurement). Everything above --
        # the defect, echoes, consequences, control -- acts on the TRUE values; methods
        # see each sensor's READING: sqrt(1 - m) * true_hat + sqrt(m) * error_hat, with
        # share m of the reading's variance being error. The error drifts on the sensor's
        # own time scale (an AR(1) at its decorrelation time, like the defect's unexplained
        # part), so averaging to the analysis resolution does not remove it. Scales and time
        # scales frozen on the baseline (the ground-truth audit replays this step). The
        # target is not touched (its measurement is out of scope).
        m_spec = (getattr(vec_cfg, "shares", None) or {}).get("measurement") if vec_cfg is not None else None
        if m_spec is not None:
            from advanced_scm import shares

            frozen = scm.graph.graph.setdefault("measurement_frozen", {})
            applied = {}
            names = sorted(k for k, v in structured_time_series.items()
                           if k != "defect_ts" and isinstance(v, np.ndarray) and v.ndim == 2)
            for i, s in enumerate(names):
                m = shares.draw(m_spec, self.unified_config.structure_seed(), "measurement", i)
                if not m:
                    continue
                true = structured_time_series[s]
                f = frozen.setdefault(s, {"scale": shares.scale_of(true),
                                          "tau": float(max(shares.decorrelation_steps(true), 1))})
                structured_time_series[s] = shares.reading(
                    true, m, f["scale"], f["tau"], int(self.unified_config.seed) + 9973 * (i + 1)
                )
                applied[s] = float(m)
            generation_metadata["measurement_noise"] = applied
            if isinstance(generation_metadata.get("vector_scm_audit"), dict):
                generation_metadata["vector_scm_audit"].setdefault("realised_shares", {})["measurement"] = applied
            self.logger.info(f"📏 Step 2.45: measurement noise on {len(applied)} sensor(s), share {m_spec}")

        return structured_time_series, generation_metadata, causal_sensors
