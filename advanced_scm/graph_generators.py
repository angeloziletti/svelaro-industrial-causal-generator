"""
Graph generation components for the Advanced Extrusion SCM: the dynamic
(random scale-free) graph with the sensors' parameter nodes.

graph_generators.py
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List

from advanced_scm import parameter_layer as pl
from logger_config import get_logger


class GraphGeneratorBase:
    """Base class for graph generators."""

    def __init__(self, hyperparams: Dict = None):
        """
        Initialize the graph generator.

        Parameters:
        -----------
        hyperparams : Dict
            Hyperparameters for controlling graph generation
        """
        self.hyperparams = hyperparams or {}

    def generate_graph(self) -> Tuple[nx.DiGraph, Dict, Dict]:
        """
        Generate a causal graph.

        Returns:
        --------
        graph : nx.DiGraph
            The generated directed acyclic graph
        node_types : Dict
            Dictionary mapping node names to their types
        gp_parameter_nodes : Dict
            Dictionary mapping temporal sensor names to their GP parameter node names
            Format: {sensor_name: [gp_node_names]}
            Example: {"barrel_temp_zone1": ["barrel_temp_zone1_event_timing", "barrel_temp_zone1_event_magnitude"]}
        """
        raise NotImplementedError(
            "This method should be implemented by derived classes"
        )


class DynamicGraphGenerator(GraphGeneratorBase):
    """
    Generates a dynamic random graph structure similar to TabPFN's approach.
    Uses growing network with redirection method to generate random scale-free networks.
    """

    def __init__(
        self,
        hyperparams: Dict = None,
        temporal_sensors: Optional[List[str]] = None,
        enable_gp_layer: bool = False,
        auto_select_temporal_sensors: bool = False,
        min_temporal_sensors: int = 3,
        max_temporal_sensors: int = 6,
        min_base_nodes: int = 15,
        max_base_nodes: int = 40,
        n_temporal_red_herrings: int = 0
    ):
        """
        Initialize the dynamic graph generator with optional GP parameter support.

        Parameters:
        -----------
        hyperparams : Dict
            Hyperparameters for controlling graph generation
        temporal_sensors : List[str], optional
            Explicit list of temporal sensor node names (ignored if auto_select=True)
        enable_gp_layer : bool
            Whether to add GP parameter nodes for temporal sensors
        auto_select_temporal_sensors : bool
            Must be True: the number of temporal sensors is drawn per plant
        min_temporal_sensors : int
            Minimum number of process nodes to randomly select as temporal sensors (used if auto_select=True)
        max_temporal_sensors : int
            Maximum number of process nodes to randomly select as temporal sensors (used if auto_select=True)
        min_base_nodes : int
            Minimum number of static (base) nodes to generate; must be > 0
        max_base_nodes : int
            Maximum number of static (base) nodes to generate
        """
        super().__init__(hyperparams)
        self.temporal_sensors = temporal_sensors or []
        self.enable_gp_layer = enable_gp_layer
        self.auto_select_temporal_sensors = auto_select_temporal_sensors
        self.min_temporal_sensors = min_temporal_sensors
        self.max_temporal_sensors = max_temporal_sensors
        self.min_base_nodes = min_base_nodes
        self.max_base_nodes = max_base_nodes
        self.n_temporal_red_herrings = n_temporal_red_herrings
        # Shadow red herring config (None or ShadowRedHerringConfig)
        self.shadow_red_herring_config = None
        # Generative parameters that are SCM nodes (parameter_layer.causal_params).
        self.causal_params = list(pl.DEFAULT_CAUSAL)
        # "motif_sequence": a fixed sequence of 2-4 motifs per sensor
        # (advanced_scm/motif_profiles.py); set by the caller.
        self.trajectory_model = "single_pattern"
        self.segments_per_sensor = (2, 4)
        self.seed = None
        self.shadow_audit: Optional[Dict] = None  # populated by _inject_shadow_confounders
        self.logger = get_logger(__name__)

    def generate_graph(self) -> Tuple[nx.DiGraph, Dict, Dict]:
        """
        Generate a dynamic random graph structure.

        Returns:
        --------
        graph : nx.DiGraph
            The generated directed acyclic graph
        node_types : Dict
            Dictionary mapping node names to their types
        gp_parameter_nodes : Dict
            Dictionary mapping temporal sensor names to their GP parameter node names
            Format: {sensor_name: [gp_node_names]}
        """
        node_types = {}

        # GP parameters require static nodes as parents: no temporal-only mode.
        if self.min_base_nodes <= 0:
            raise ValueError(
                f"min_base_nodes must be > 0 (sensor parameters need static parents), "
                f"got {self.min_base_nodes}"
            )
        # Generate base SCM nodes
        # Sample number of nodes using log-uniform distribution
        min_nodes = self.min_base_nodes
        max_nodes = self.max_base_nodes
        num_nodes = int(np.exp(np.random.uniform(np.log(min_nodes), np.log(max_nodes))))

        # Sample redirection probability using gamma distribution
        alpha = self.hyperparams.get("graph_alpha", 0.95)
        beta = self.hyperparams.get("graph_beta", 8.0)
        redirection_prob = np.random.gamma(alpha, 1 / beta)

        # Generate a scale-free network (growing network with redirection)
        G = nx.DiGraph()

        # Start with a single node
        G.add_node(0)

        # Add nodes one by one
        for i in range(1, num_nodes):
            # Select a target node to connect to
            if len(G) > 0:
                target = np.random.choice(list(G.nodes()))

                # Apply redirection with probability p
                if np.random.random() < redirection_prob:
                    # Redirect to a random neighbor
                    neighbors = list(G.successors(target)) + list(
                        G.predecessors(target)
                    )
                    if neighbors:
                        target = np.random.choice(neighbors)

                # Add the new node with an edge to the target (same draws
                # either way; base_edges false keeps the roots independent)
                G.add_node(i)
                if getattr(self, "base_edges", True):
                    G.add_edge(i, target)

        # Rename nodes to static_node_1, static_node_2, etc.
        # All base SCM nodes are "static" type (scalar values only, no time series)
        node_mapping = {}

        for i in range(num_nodes):
            # Generate simple sequential naming
            name = f"static_node_{i+1}"

            # Add to mapping and assign static type
            node_mapping[i] = name
            node_types[name] = "static"

        # Rename the nodes in the graph
        G = nx.relabel_nodes(G, node_mapping)

        self.logger.info(f"✅ Generated {num_nodes} base SCM nodes (range: {min_nodes}-{max_nodes})")

        # No scalar defect node: the vector-SCM defect_ts is the only target;
        # its parents are the truth sensors.
        self.logger.info(
            "   🎯 No scalar defect node: defect_ts is the only target"
        )

        # Add GP parameter nodes if enabled
        gp_parameter_nodes = {}
        if self.enable_gp_layer:
            # Determine which static nodes to add time series to
            temporal_sensors_to_use = self._determine_temporal_sensors(G, node_types)

            if temporal_sensors_to_use:
                gp_parameter_nodes = self._add_gp_parameter_nodes_dynamic(
                    G, node_types, temporal_sensors_to_use
                )

        return G, node_types, gp_parameter_nodes

    def _determine_temporal_sensors(self, G: nx.DiGraph, node_types: Dict) -> List[str]:
        """
        Determine how many independent temporal sensors to create (dynamic graph).

        Creates independent temp_N identifiers (NOT tied to static nodes).
        These identifiers are used for:
        - GP parameter node naming (temp_1_event_timing, temp_1_event_magnitude)
        - Time series generation naming
        - feature naming (temp_1_bucket_1_cluster_*)

        Parameters:
        -----------
        G : nx.DiGraph
            The generated graph
        node_types : Dict
            Dictionary mapping node names to their types

        Returns:
        --------
        temporal_sensors : List[str]
            List of temporal sensor identifiers (e.g., ['temp_1', 'temp_2', ...])
        """
        if not self.auto_select_temporal_sensors:
            raise NotImplementedError(
                "auto_select_temporal_sensors: false (an explicit sensor list) is not part of this release"
            )

        # Sample number of temporal sensors from uniform range
        n_temporal_sensors = np.random.randint(
            self.min_temporal_sensors,
            self.max_temporal_sensors + 1
        )

        if n_temporal_sensors == 0:
            self.logger.info("✅ No temporal sensors created (sampled 0)")
            return []

        # Create independent temporal sensor identifiers
        temporal_sensors = [f"temp_{i+1}" for i in range(n_temporal_sensors)]

        self.logger.info(
            f"✅ Creating {n_temporal_sensors} independent temporal sensors "
            f"(range: {self.min_temporal_sensors}-{self.max_temporal_sensors}): {temporal_sensors}"
        )
        self.logger.info(
            f"   These will generate time series controlled by GP parameters, "
            f"separate from static nodes"
        )

        return temporal_sensors

    def _add_gp_parameter_nodes_dynamic(
        self,
        G: nx.DiGraph,
        node_types: Dict,
        temporal_sensors: List[str]
    ) -> Dict[str, List[str]]:
        """
        Add GP parameter nodes for temporal sensors in dynamic graph.

        Creates independent GP parameter nodes (temp_N_<parameter>) that are NOT
        tied to any existing graph node. These GP parameters control time series
        generation; the truth sensors' trajectories are the parents of defect_ts.

        Parameters:
        -----------
        G : nx.DiGraph
            The graph to modify
        node_types : Dict
            Dictionary mapping node names to their types
        temporal_sensors : List[str]
            List of temporal sensor identifiers (e.g., ['temp_1', 'temp_2'])

        Returns:
        --------
        gp_parameter_nodes : Dict
            Dictionary mapping sensor names to their GP parameter node names
            Format: {sensor_name: [timing_node, magnitude_node]}
        """
        gp_nodes_dict = {}

        # Pre-pick red-herring sensors: these get GP nodes (so their time series is
        # still driven by upstream conditions) but are not parents of defect_ts, so
        # they appear in the dataset as non-causal confusers.
        n_red = getattr(self, "n_temporal_red_herrings", 0)
        n_truths = getattr(self, "n_truths", None)
        if n_truths:
            # The number of true causes is structure: drawn per plant.
            lo, hi = (int(v) for v in n_truths)
            n_red = len(temporal_sensors) - int(
                np.random.default_rng(int(self.seed or 0) + 8761).integers(lo, hi + 1)
            )
        red_herring_sensors = set()
        if n_red > 0 and len(temporal_sensors) > n_red:
            red_herring_sensors = set(np.random.choice(
                list(temporal_sensors), size=n_red, replace=False
            ))
            self.logger.info(
                f"   🃏 Designated {n_red} non-causal red-herring sensor(s): "
                f"{sorted(red_herring_sensors)}"
            )
        elif n_red > 0:
            self.logger.warning(
                f"   ⚠️ n_temporal_red_herrings={n_red} >= n_temporal_sensors="
                f"{len(temporal_sensors)}; ignoring (would leave no causal sensors)."
            )
        # Recorded on the graph: the truth sensors are exactly the complement of
        # this set.
        G.graph["red_herring_sensors"] = sorted(red_herring_sensors)

        # Bug 8 fix: when red herrings are present, partition the static-node pool into
        # two DISJOINT subsets so red-herring GP timing/magnitude is statistically
        # independent from causal-sensor GP timing/magnitude. Otherwise red herrings
        # share static-node parents with causal sensors and end up observationally
        # correlated with the defect even though they have no SCM edge to it.
        red_herring_parent_pool: Optional[List[str]] = None
        causal_parent_pool: Optional[List[str]] = None
        if red_herring_sensors:
            static_node_pool = [
                n for n, t in node_types.items() if t == "static"
            ]
            MIN_PARENTS_PER_GP = 2
            if len(static_node_pool) < 2 * MIN_PARENTS_PER_GP:
                raise ValueError(
                    f"Cannot create disjoint static-parent pools for red-herring "
                    f"sensors: only {len(static_node_pool)} static nodes available, "
                    f"need at least {2 * MIN_PARENTS_PER_GP}. Increase "
                    f"min_base_nodes / max_base_nodes in the SCM config, OR set "
                    f"n_temporal_red_herrings: 0 to disable disjoint pools."
                )
            shuffled = list(static_node_pool)
            np.random.shuffle(shuffled)
            n_red_pool = max(MIN_PARENTS_PER_GP, len(shuffled) // 2)
            red_herring_parent_pool = shuffled[:n_red_pool]
            causal_parent_pool = shuffled[n_red_pool:]
            if len(red_herring_parent_pool) < MIN_PARENTS_PER_GP:
                raise ValueError(
                    f"Red-herring static-parent pool has only "
                    f"{len(red_herring_parent_pool)} node(s); need at least "
                    f"{MIN_PARENTS_PER_GP}. Increase min_base_nodes / max_base_nodes "
                    f"or reduce n_temporal_red_herrings."
                )
            if len(causal_parent_pool) < MIN_PARENTS_PER_GP:
                raise ValueError(
                    f"Causal static-parent pool has only "
                    f"{len(causal_parent_pool)} node(s); need at least "
                    f"{MIN_PARENTS_PER_GP}. Increase min_base_nodes / max_base_nodes "
                    f"or reduce n_temporal_red_herrings."
                )
            self.logger.info(
                f"   🔀 Disjoint static-parent pools — "
                f"red-herring pool ({len(red_herring_parent_pool)}): "
                f"{sorted(red_herring_parent_pool)} | "
                f"causal pool ({len(causal_parent_pool)}): "
                f"{sorted(causal_parent_pool)}"
            )

        for sensor_name in temporal_sensors:
            # One SCM node per causal generative parameter of this sensor
            # (advanced_scm/parameter_layer.py); the other parameters are drawn at
            # random per batch by the trajectory generator.
            n_segments = None
            if self.trajectory_model == "motif_sequence":
                from advanced_scm.motif_profiles import plan_profile

                profile_rng = np.random.default_rng(
                    int(self.seed or 0) + 4409 + temporal_sensors.index(sensor_name)
                )
                profile = plan_profile(profile_rng, *self.segments_per_sensor)
                G.graph.setdefault("sensor_profiles", {})[sensor_name] = profile
                n_segments = len(profile)
            param_nodes = {
                param: pl.node_name(sensor_name, param)
                for param in pl.expand_causal(self.causal_params, n_segments)
            }
            for param, node in param_nodes.items():
                G.add_node(node, param=param, param_base=pl.base_of(param), param_kind=pl.kind(param))
                node_types[node] = "gp_parameter"

            # Track
            gp_nodes_dict[sensor_name] = list(param_nodes.values())

            # Connect static nodes → GP parameters
            # Select random static nodes as parents — drawn from a sensor-specific
            # pool when disjoint pools are active (Bug 8 fix), otherwise from all statics.
            if red_herring_parent_pool is not None and causal_parent_pool is not None:
                pool_for_sensor = (
                    red_herring_parent_pool
                    if sensor_name in red_herring_sensors
                    else causal_parent_pool
                )
            else:
                pool_for_sensor = None  # any static node is a candidate
            relevant_scm_nodes = self._select_gp_parameter_parents(
                G, node_types, parent_pool=pool_for_sensor
            )

            # Validate: GP parameter nodes MUST have parents
            if not relevant_scm_nodes:
                raise ValueError(
                    f"Cannot create GP parameter nodes for temporal sensor '{sensor_name}': "
                    f"no static nodes found to connect as parents. GP parameter nodes "
                    f"require at least one parent node to receive causal influence."
                )

            # Connect static nodes → GP parameters.
            for param, node in param_nodes.items():
                for scm_node in relevant_scm_nodes:
                    G.add_edge(scm_node, node)

            # Red-herring sensors get GP nodes (time series still driven by upstream
            # statics) but are not parents of defect_ts — non-causal by construction.
            if sensor_name in red_herring_sensors:
                self.logger.info(
                    f"   🃏 Red-herring sensor (non-causal): {sensor_name} "
                    f"(parents: {relevant_scm_nodes}, no defect edges)"
                )
            else:
                # A truth sensor: a parent of defect_ts.
                self.logger.info(
                    f"   ✅ Truth sensor (defect_ts parent): {sensor_name} "
                    f"(parents: {relevant_scm_nodes})"
                )

        # Shadow red-herring confounder structure (opt-in): reverses the disjoint pools above by
        # adding per-truth confounder nodes Z_k that feed into BOTH the
        # truth sensor's GP params AND the shadow red herrings' GP params,
        # introducing sample-level confounding for methodology testing.
        shadow_cfg = getattr(self, "shadow_red_herring_config", None)
        if (
            shadow_cfg is not None
            and getattr(shadow_cfg, "enabled", False)
            and red_herring_sensors
            and gp_nodes_dict
        ):
            self._inject_shadow_confounders(
                G=G,
                node_types=node_types,
                gp_nodes_dict=gp_nodes_dict,
                temporal_sensors=temporal_sensors,
                red_herring_sensors=set(red_herring_sensors),
                config=shadow_cfg,
            )

        # Shared batch phases (vector_scm.shares.phase): latent per-phase timing nodes that
        # move the onset of every sensor segment falling in that phase together.
        n_phases = int(getattr(self, "n_phases", 0) or 0)
        if n_phases > 0 and gp_nodes_dict:
            self._inject_phase_nodes(G, node_types, gp_nodes_dict, n_phases)

        self._add_sensor_graph(G, gp_nodes_dict, temporal_sensors, set(red_herring_sensors))
        self._plan_control_loops(G, temporal_sensors, set(red_herring_sensors))
        self._plan_consequences(G, temporal_sensors, set(red_herring_sensors))

        return gp_nodes_dict

    def _plan_consequences(self, G, temporal_sensors, red_herring_sensors):
        """Sensors downstream of the defect (advanced_scm/target_edges.py). Drawn
        among the non-causes that are not ancestors of a cause (through recipe
        edges, dynamic edges or a control loop: a sensor feeding a loop's
        disturbance drives the controlled cause), not in a control loop, and
        have no outgoing dynamic edge (their children were coupled before the
        defect exists, so they would not carry its echo). A consequence is then
        downstream of the defect only, never also upstream of it."""
        cfg = getattr(self, "consequences", None)
        if not cfg:
            return
        from advanced_scm.target_edges import plan_consequences

        sg = G.graph.get("sensor_graph") or {}
        excluded = set(sg.get("upstream", []))
        excluded |= {e["source"] for e in sg.get("dynamic_edges") or []}
        S = nx.DiGraph()
        S.add_edges_from((e["source"], e["target"]) for e in sg.get("edges") or [])
        S.add_edges_from((e["source"], e["target"]) for e in sg.get("dynamic_edges") or [])
        for loop in G.graph.get("control_loops") or []:
            excluded |= {loop["controlled"], loop["disturbance"], loop["actuator"]}
            S.add_edge(loop["disturbance"], loop["controlled"])
            S.add_edge(loop["actuator"], loop["controlled"])
        for t in (s for s in temporal_sensors if s not in red_herring_sensors):
            if t in S:
                excluded |= nx.ancestors(S, t)
        candidates = sorted(s for s in red_herring_sensors if s not in excluded)
        G.graph["consequences"] = plan_consequences(candidates, cfg, int(self.seed or 0))
        self.logger.info(f"   ↩️ Consequences of the defect: {[c['sensor'] for c in G.graph['consequences']]}")

    def _plan_control_loops(
        self, G: nx.DiGraph, temporal_sensors: List[str], red_herring_sensors: set
    ) -> None:
        """Choose control-loop members (config `control_loops`; see
        advanced_scm/control_loops.py): a truth sensor is controlled, two
        non-cause sensors (not upstream causes) are its disturbance and actuator."""
        from advanced_scm.control_loops import plan_control_loops

        cfg = getattr(self, "control_loops", None) or {}
        if not cfg.get("enabled", False):
            return
        rng = np.random.default_rng(int(self.seed or 0) + 3307)
        truths = [s for s in temporal_sensors if s not in red_herring_sensors]
        upstream = set((G.graph.get("sensor_graph") or {}).get("upstream", []))
        candidates = sorted(s for s in red_herring_sensors if s not in upstream)
        G.graph["control_loops"] = plan_control_loops(cfg, truths, candidates, rng)
        # A loop's members interact only through the loop itself: drop any
        # sensor-graph edge between two members of one loop (it would nest a
        # second dependence inside the loop, e.g. the controlled truth driving
        # its own disturbance).
        sg = G.graph.get("sensor_graph")
        if sg and G.graph["control_loops"]:
            groups = [
                {l["controlled"], l["disturbance"], l["actuator"]}
                for l in G.graph["control_loops"]
            ]
            kept = []
            for e in sg["edges"]:
                if any(e["source"] in g and e["target"] in g for g in groups):
                    pa, pb = e["via"]
                    if G.has_edge(pa, pb):
                        G.remove_edge(pa, pb)
                else:
                    kept.append(e)
            removed = len(sg["edges"]) - len(kept)
            sg["edges"] = kept
            dyn = sg.get("dynamic_edges") or []
            dyn_kept = [
                e for e in dyn
                if not any(e["source"] in g and e["target"] in g for g in groups)
            ]
            removed += len(dyn) - len(dyn_kept)
            sg["dynamic_edges"] = dyn_kept
            if removed:
                self.logger.info(
                    f"   🎛️ Removed {removed} sensor-graph edge(s) between loop members"
                )
        self.logger.info(f"   🎛️ Control loops: {G.graph['control_loops']}")

    def _add_sensor_graph(
        self,
        G: nx.DiGraph,
        gp_nodes_dict: Dict[str, List[str]],
        temporal_sensors: List[str],
        red_herring_sensors: set,
    ) -> None:
        """Parameter-mode edges between sensors (config `sensor_graph`).

        A sensor edge A -> B makes one of A's causal parameters a parent of one of
        B's (both drawn at random): A's batch shape shifts B's. Sensors are ordered

            upstream (n_upstream red herrings) -> truths -> every other sensor

        and edges only go forward, so the roles for defect_ts are guaranteed:
        truths stay its direct parents, upstream sensors are its only sensor
        ancestors (each gets one edge into a random truth), and the other sensors
        are never causes -- they may be descendants of a cause or confounded with
        it, the correlated non-causes. Forward edges are drawn with probability
        2 * expected_in_degree / (n - 1), giving that mean in-degree.
        """
        cfg = getattr(self, "sensor_graph", None) or {}
        if not cfg.get("enabled", False):
            return
        rng = np.random.default_rng(int(self.seed or 0) + 2711)
        truths = [s for s in temporal_sensors if s not in red_herring_sensors]
        reds = sorted(red_herring_sensors)
        n_up = max(0, min(int(cfg.get("n_upstream", 1)), len(reds) - 1))  # keep a non-cause
        upstream = [str(s) for s in rng.choice(reds, size=n_up, replace=False)] if n_up else []
        others = [s for s in reds if s not in upstream]
        order = upstream + [str(s) for s in rng.permutation(truths)] + [
            str(s) for s in rng.permutation(others)
        ]
        n = len(order)
        p_edge = min(1.0, 2.0 * float(cfg.get("expected_in_degree", 1.0)) / max(n - 1, 1))
        edges = {
            (order[i], order[j]) for j in range(n) for i in range(j) if rng.random() < p_edge
        }
        for u in upstream:
            edges.add((u, str(rng.choice(truths))))

        param_edges = []
        for a, b in sorted(edges, key=lambda e: (order.index(e[0]), order.index(e[1]))):
            pa = str(rng.choice(gp_nodes_dict[a]))
            pb = str(rng.choice(gp_nodes_dict[b]))
            G.add_edge(pa, pb)
            param_edges.append({"source": a, "target": b, "via": [pa, pb]})
        from advanced_scm.sensor_dynamics import plan_dynamic_edges

        dynamic_edges = plan_dynamic_edges(
            order, cfg.get("dynamic_edges"), np.random.default_rng(int(self.seed or 0) + 5113)
        )
        G.graph["sensor_graph"] = {
            "order": order,
            "upstream": upstream,
            "expected_in_degree": float(cfg.get("expected_in_degree", 1.0)),
            "edges": param_edges,
            # route 3: child(t) += gain * FOPDT[parent](t) (advanced_scm/sensor_dynamics.py)
            "dynamic_edges": dynamic_edges,
        }
        self.logger.info(
            f"   🔀 Sensor graph: {len(param_edges)} parameter-mode edges among {n} "
            f"sensors; upstream ancestors of the truths: {upstream}"
        )

    def _inject_phase_nodes(self, G: nx.DiGraph, node_types: Dict, gp_nodes_dict: Dict[str, List[str]],
                            n_phases: int) -> None:
        """Shared batch phases (ISA-88 recipe phases): the batch is divided into `n_phases`
        equal phases, and each phase k gets a latent timing node `phase_<k>` (one value per
        batch: how early or late this batch runs through phase k). Every sensor segment whose
        slot centre falls in phase k gets an edge phase_<k> -> its onset parameter, so all
        sensors switch together at the phase boundaries. Each phase node has a private
        exogenous root, so the phase share controls this route alone. The edge weights are
        set afterwards by shares.calibrate_phase_shares; the node type is the one the
        mechanism layer already evaluates as a weighted linear node (latent confounders)."""
        audit_edges = []
        phase_nodes = []
        for k in range(n_phases):
            ph, root = f"phase_{k}", f"phase_root_{k}"
            G.add_node(root)
            node_types[root] = "static"
            G.add_node(ph)
            node_types[ph] = "shadow_confounder"
            G.add_edge(root, ph)
            phase_nodes.append(ph)
        for sensor, nodes in sorted(gp_nodes_dict.items()):
            onsets = sorted(n for n in nodes if G.nodes[n].get("param_base") == "onset"
                            and str(G.nodes[n].get("param", "")).startswith("s"))
            K = len(onsets)
            for n in onsets:
                k = int(str(G.nodes[n]["param"]).split("_")[0][1:])
                phase = min(n_phases - 1, int((k + 0.5) / K * n_phases))
                G.add_edge(f"phase_{phase}", n, weight=1.0)
                audit_edges.append({"source": f"phase_{phase}", "target": n})
        G.graph["phase_audit"] = {"n_phases": n_phases, "phase_nodes": phase_nodes, "edges": audit_edges}
        self.logger.info(f"   Shared batch phases: {n_phases} phase nodes, {len(audit_edges)} onset edges")

    def _inject_shadow_confounders(
        self,
        G: nx.DiGraph,
        node_types: Dict,
        gp_nodes_dict: Dict[str, List[str]],
        temporal_sensors: List[str],
        red_herring_sensors: set,
        config,
    ) -> None:
        """Inject per-truth shadow confounder structure.

        For each truth sensor T_k:
          1. Create a confounder node `shadow_confounder_z_{T_k}` with type
             "shadow_confounder" (new node type, distinct from "static").
          2. Give Z_k a private exogenous root `shadow_root_{T_k}` (requires
             vector_scm.shares.confounder), so the confounder share controls
             this route alone.
          3. Add edges Z_k → T_k's event_timing and event_magnitude GP params
             with weight ~ uniform(*confounder_strength_range).
          4. Partition red herrings across truth sensors (ceil(N/K) per group,
             deterministic by sorted sensor name).
          5. For each shadow red herring in T_k's group, independently per GP
             param in `modulated_gp_params`, with probability
             `modulation_probability`, add edge Z_k → R_{gp_param} with
             weight ~ uniform(*confounder_strength_range).
          6. Build audit dict at `G.graph["shadow_red_herring_audit"]`.

        Mutates G and node_types in place. Mechanism layer reads edge
        attributes via `G.edges[parent, child].get("weight")` to apply
        per-parent weights (see EdgeMappingFactory.create_linear_combination_mapping).
        """
        import math

        # Identify truth sensors = temporal_sensors NOT in red_herring_sensors
        truth_sensors = [s for s in temporal_sensors if s not in red_herring_sensors]
        if not truth_sensors:
            self.logger.warning(
                "   ⚠️ Shadow confounder requested but no truth sensors found; skipping."
            )
            return

        # Deterministic RNG keyed off (seed + config offset). Independent of
        # other RNG streams in the pipeline.
        scm_seed = getattr(self, "seed", None)
        if scm_seed is None:
            self.logger.warning(
                "   ⚠️ Shadow confounder enabled but SCM seed is None; "
                "results will not be reproducible. Setting seed=0."
            )
            scm_seed = 0
        rng = np.random.default_rng(int(scm_seed) + int(config.seed_offset))

        # Partition red herrings across truth sensors. Deterministic chunked
        # assignment with sorted sensor names for reproducibility.
        sorted_truth = sorted(truth_sensors)
        sorted_red = sorted(red_herring_sensors)
        K, N = len(sorted_truth), len(sorted_red)
        n_per_group = max(1, math.ceil(N / K))
        shadow_groups: Dict[str, List[str]] = {}
        # (K * ceil(N / K) >= N, so the chunks cover every red herring.)
        for k, t in enumerate(sorted_truth):
            chunk = sorted_red[k * n_per_group : (k + 1) * n_per_group]
            shadow_groups[t] = chunk

        # Build the confounder structure
        weight_lo, weight_hi = (
            float(config.confounder_strength_range[0]),
            float(config.confounder_strength_range[1]),
        )
        modulated_params = list(config.modulated_gp_params)
        p_mod = float(config.modulation_probability)

        confounder_nodes: List[str] = []
        confounder_edges: List[Dict] = []

        if not getattr(self, "independent_confounders", False):
            raise NotImplementedError(
                "shadow confounders without vector_scm.shares.confounder are not part of this release"
            )

        for t_k in sorted_truth:
            z_k = f"shadow_confounder_z_{t_k}"
            G.add_node(z_k)
            node_types[z_k] = "shadow_confounder"
            confounder_nodes.append(z_k)

            # Explained-variance shares (advanced_scm/shares.py): Z_k is its own
            # latent driver, with a private exogenous root, so the confounder
            # share controls this route alone; the batch-condition route has
            # its own switch (scm_settings.base_edges).
            z_root = f"shadow_root_{t_k}"
            G.add_node(z_root)
            node_types[z_root] = "static"
            G.add_edge(z_root, z_k)

            # Z_k → T_k's GP params (always — Z_k is T_k's confounder by construction).
            t_gp_nodes = gp_nodes_dict.get(t_k, [])
            for gp_node in t_gp_nodes:
                w = float(rng.uniform(weight_lo, weight_hi))
                G.add_edge(z_k, gp_node, weight=w)
                confounder_edges.append({
                    "source": z_k,
                    "target": gp_node,
                    "weight": round(w, 4),
                    "kind": "truth",
                })

            # Z_k → shadow red herrings' GP params (stochastic per param).
            for r in shadow_groups[t_k]:
                r_gp_nodes = gp_nodes_dict.get(r, [])
                # Map gp node names to which GP param (event_timing/event_magnitude)
                for gp_node in r_gp_nodes:
                    # Exact parameter match from the node attribute: a suffix test
                    # would let "amplitude" also match "<sensor>_noise_amplitude".
                    matched_param = G.nodes[gp_node].get("param_base")
                    if matched_param not in modulated_params:
                        continue  # GP param not in modulated list
                    if rng.random() >= p_mod:
                        continue  # failed Bernoulli draw
                    w = float(rng.uniform(weight_lo, weight_hi))
                    G.add_edge(z_k, gp_node, weight=w)
                    confounder_edges.append({
                        "source": z_k,
                        "target": gp_node,
                        "weight": round(w, 4),
                        "kind": "shadow",
                        "shadow_of": t_k,
                    })

        # Stash audit dict on the graph for downstream metadata export.
        G.graph["shadow_red_herring_audit"] = {
            "enabled": True,
            "n_truth": K,
            "n_red_herrings": N,
            "shadow_groups": shadow_groups,
            "confounder_nodes": confounder_nodes,
            "confounder_edges": confounder_edges,
            "config": {
                "confounder_strength_range": [weight_lo, weight_hi],
                "modulation_probability": p_mod,
                "modulated_gp_params": modulated_params,
                "seed_offset": int(config.seed_offset),
                "rng_seed_used": int(scm_seed) + int(config.seed_offset),
            },
        }
        # Also store on the generator for direct access by callers (without
        # having to inspect G.graph attributes).
        self.shadow_audit = G.graph["shadow_red_herring_audit"]
        n_truth_edges = sum(1 for e in confounder_edges if e["kind"] == "truth")
        n_shadow_edges = sum(1 for e in confounder_edges if e["kind"] == "shadow")
        self.logger.info(
            f"   🕶️  Shadow confounders injected: {K} confounder node(s), "
            f"{n_truth_edges} Z_k→truth edges + {n_shadow_edges} Z_k→shadow edges "
            f"(modulation_p={p_mod}, weight∈[{weight_lo}, {weight_hi}])"
        )
        for t_k, shadows in shadow_groups.items():
            self.logger.info(f"      {t_k} shadows: {shadows}")

    def _select_gp_parameter_parents(
        self,
        G: nx.DiGraph,
        node_types: Dict,
        parent_pool: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Select static nodes to connect as parents to GP parameters.

        Randomly selects up to 3 static nodes from a candidate pool to serve
        as causal parents for GP parameter nodes.

        Parameters:
        -----------
        G : nx.DiGraph
            The graph
        node_types : Dict
            Dictionary mapping node names to their types
        parent_pool : Optional[List[str]]
            Restrict parent selection to this pool of static-node names.
            When None, falls back to ALL static nodes in the graph
            (default behavior, used when n_temporal_red_herrings == 0).

        Returns:
        --------
        parent_nodes : List[str]
            List of static node names (up to 3)
        """
        if parent_pool is not None:
            candidate_pool = list(parent_pool)
        else:
            candidate_pool = [
                node for node, node_type in node_types.items()
                if node_type == "static"
            ]

        if not candidate_pool:
            return []

        # Randomly select up to 3 static nodes as parents
        n_select = min(3, len(candidate_pool))
        selected_parents = list(np.random.choice(
            candidate_pool,
            size=n_select,
            replace=False
        ))

        return selected_parents
