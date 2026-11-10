import numpy as np
import pandas as pd

from advanced_scm.base import AdvancedExtrusionSCMBase
from advanced_scm.graph_generators import DynamicGraphGenerator
from advanced_scm.edge_mappings import EdgeMappingFactory
from advanced_scm.node_mechanisms import MechanismFactory
from advanced_scm.initialization import InitializationSampler
from advanced_scm.post_processing import PostProcessor

from logger_config import get_logger

logger = get_logger(__name__)


class AdvancedExtrusionSCM(AdvancedExtrusionSCMBase):
    """
    Structural causal model of a plant.
    """

    def __init__(
        self,
        hyperparams=None,
        vector_dim=8,
        use_dynamic_graph=False,
        target_defect_rates=None,
        temporal_sensors=None,
        enable_gp_layer=False,
        auto_select_temporal_sensors=False,
        min_temporal_sensors=3,
        max_temporal_sensors=6,
        min_base_nodes=15,
        max_base_nodes=40,
        n_temporal_red_herrings=0,
        shadow_red_herring_config=None,    # ShadowRedHerringConfig
        seed=None,                          # required when shadow_red_herring_config is provided (RNG determinism)
    ):
        """
        Initialize the SCM.

        Parameters:
        -----------
        hyperparams : dict or None
            Hyperparameters for the model
        vector_dim : int
            Dimension of vector representations for nodes
        use_dynamic_graph : bool
            Must be True: the dynamic graph is the only graph of this release
        target_defect_rates : dict or None
            Must be empty: this release has no scalar defect nodes (defect_ts,
            added by the temporal processor, is the only target)
        temporal_sensors : list or None
            List of temporal sensor node names for GP parameter nodes
        enable_gp_layer : bool
            Whether to enable GP parameter layer for causal validation
        auto_select_temporal_sensors : bool
            Whether to auto-select temporal sensors (for dynamic graphs)
        n_temporal_sensors : int
            Number of temporal sensors to auto-select (for dynamic graphs)
        """
        super().__init__(hyperparams, vector_dim)

        if not use_dynamic_graph:
            raise NotImplementedError(
                "use_dynamic_graph: false (the fixed domain-specific graph) is not part of this release"
            )
        if target_defect_rates:
            raise NotImplementedError(
                "scalar defect nodes (target_defect_rates) are not part of this release; "
                "set dataset_generation.target_defect_rates: {}"
            )

        # Graph generation options
        self.use_dynamic_graph = use_dynamic_graph
        self.target_defect_rates = {}

        # GP parameter layer options
        self.enable_gp_layer = enable_gp_layer
        self.temporal_sensors = temporal_sensors or []
        self.gp_parameter_nodes = {}  # Will be populated by generate_graph()

        # Initialize components
        self.graph_generator = DynamicGraphGenerator(
            hyperparams,
            temporal_sensors=temporal_sensors,
            enable_gp_layer=enable_gp_layer,
            auto_select_temporal_sensors=auto_select_temporal_sensors,
            min_temporal_sensors=min_temporal_sensors,
            max_temporal_sensors=max_temporal_sensors,
            min_base_nodes=min_base_nodes,
            max_base_nodes=max_base_nodes,
            n_temporal_red_herrings=n_temporal_red_herrings,
        )
        # Attach shadow red-herring config + seed to the generator
        # so _inject_shadow_confounders can access them.
        self.shadow_red_herring_config = shadow_red_herring_config
        self.seed = seed
        self.graph_generator.shadow_red_herring_config = shadow_red_herring_config
        self.graph_generator.seed = seed

        self.initialization_sampler = InitializationSampler(vector_dim, hyperparams)
        self.post_processor = PostProcessor(hyperparams)

    def build_causal_graph(self):
        """Build the causal graph structure."""
        # Generate the graph and node types
        self.graph, self.node_types, self.gp_parameter_nodes = self.graph_generator.generate_graph()

        static_nodes = [
            node for node, node_type in self.node_types.items() if node_type == "static"
        ]
        logger.info(f"🔍: Static nodes found: {static_nodes}")

        # Note: Static nodes CAN have other static nodes as parents in the scale-free graph
        # (some static nodes will have time series data added via GP parameters)

        # Initialize vector representations for nodes
        self.node_vectors = self.initialization_sampler.initialize_node_vectors(
            self.graph.nodes()
        )

        return self.graph

    def define_node_mechanisms(self):
        """Define the functional relationships for each node in the graph."""
        for node in self.graph.nodes():
            # Get parent nodes
            parents = list(self.graph.predecessors(node))

            # Get node type (default to "static" for SCM nodes)
            node_type = self.node_types.get(node, "static")

            # Handle static features
            if node_type == "static":
                if parents:
                    # Static node with parents (from scale-free graph) → use linear combination
                    mapping_func, linear_comb = (
                        EdgeMappingFactory.create_linear_combination_mapping(
                            parents, bias=0.0
                        )
                    )
                    linear_comb.noise_scale = 0.2  # Moderate noise

                    self.node_mechanisms[node] = (
                        lambda parent_values, params, f=mapping_func: f(parent_values, None)
                    )
                    self.node_parameters[node] = {
                        "mechanism_type": "linear_combination",
                        "parents": parents
                    }
                else:
                    # Static node without parents (root node) → use static mechanism
                    self.node_mechanisms[node] = (
                        MechanismFactory.create_static_node_mechanism(
                            node, self.hyperparams
                        )
                    )
                    self.node_parameters[node] = {"mechanism_type": "static"}

            # ========== Handle GP parameter nodes ==========
            # `shadow_confounder` Z_k nodes use the SAME predictable
            # linear mechanism as gp_parameter nodes. They need stable, well-
            # scaled values per sample so they can transmit signal from their
            # static parents through to truth+shadow GP params via the
            # per-parent weights set by _inject_shadow_confounders. Without
            # this branch, Z_k would fall through to a random nonlinear /
            # xgboost mechanism and Z_k's variance becomes unpredictable.
            elif node_type in ("gp_parameter", "shadow_confounder"):
                # GP parameter nodes are hidden control parameters
                # They should have simple linear mechanisms to ensure numerical stability
                if not parents:
                    raise ValueError(f"GP parameter node {node} must have parents")

                # Check whether any incoming edge has an explicit
                # `weight` attribute (set by _inject_shadow_confounders). If
                # so, use per-parent weights to give Z_k → GP-param edges
                # their configured strength. Other parents fall back to 1.0
                # so the relative magnitude of Z_k vs other parents reflects
                # the configured confounder strength range.
                per_parent_weights = None
                edges_with_weight = [
                    (p, self.graph.edges[p, node].get("weight"))
                    for p in parents
                    if "weight" in self.graph.edges[p, node]
                ]
                if edges_with_weight:
                    per_parent_weights = {
                        p: float(w) for p, w in edges_with_weight
                    }
                # Create simple linear combination with moderate noise.
                # Use equal_weights so GP-parameter values inherit positively
                # from their parents — random U[-1,1] weights would let some
                # sensors anti-correlate with the shared `causal_event_time`
                # (Option 1 latent), defeating the synchronized-timing fix.
                # Single parent → weight=1.0 (identity); multi-parent → 1/n.
                mapping_func, linear_comb = (
                    EdgeMappingFactory.create_linear_combination_mapping(
                        parents,
                        bias=0.0,
                        equal_weights=(per_parent_weights is None),
                        per_parent_weights=per_parent_weights,
                    )
                )

                # Set reasonable noise for GP parameters
                linear_comb.noise_scale = 0.2

                self.node_mechanisms[node] = (
                    lambda parent_values, params, f=mapping_func: f(parent_values, None)
                )

                self.node_parameters[node] = {
                    "weights": linear_comb.weights,
                    "bias": linear_comb.bias,
                    "noise_scale": linear_comb.noise_scale,
                    "mechanism_type": "gp_parameter",
                }

                logger.debug(
                    f"Created GP parameter mechanism: {parents} → {node} "
                    f"(weights={linear_comb.weights}, noise={linear_comb.noise_scale})"
                )
            # ==========================================

            # Any other node type
            else:
                raise NotImplementedError(
                    f"Node {node!r} has no mechanism: this release generates vector-SCM plants only."
                )

    def _apply_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing to the generated dataset.

        Only observed static attributes are post-processed (as observation noise).
        Event parameters, confounders and defects drive downstream mechanisms, so
        their exported values must be exactly the ones those mechanisms used.
        """
        protected = [
            node
            for node, node_type in self.node_types.items()
            if node_type in ("gp_parameter", "shadow_confounder", "defect")
        ]
        return self.post_processor.apply_mixed_post_processing(
            df, protected_columns=protected
        )

    def get_causal_relationships(self, estimate_strengths=True, max_path_length=5):
        """
        Causal relationships of the scalar defect nodes of the SCM graph: none in
        this release (defect_ts, the only target, is added by the caller).

        The edge-strength sampling still runs when estimate_strengths is True: it
        draws from the global RNG, and get_graph_structure, called next for the
        metadata, samples from the state it leaves.
        """
        if estimate_strengths:
            self._compute_edge_strengths_universal({"n_samples": 1000, "method": "sampling"})
        return {}

    def get_graph_structure(self):
        """
        Get complete graph structure with edge strengths and their sources
        (correlations from samples of the SCM).

        Returns dict with nodes, edges, and their properties including strength source.
        """
        edge_config = self.hyperparams.get("edge_strength_computation", {})

        if not edge_config.get("enabled", True):
            # No edge strength computation
            return {
                "nodes": list(self.graph.nodes()),
                "edges": [
                    {"source": src, "target": dst} for src, dst in self.graph.edges()
                ],
                "n_nodes": len(self.graph.nodes()),
                "n_edges": len(self.graph.edges()),
            }

        # Compute sampling-based strengths
        sampling_strengths = self._compute_edge_strengths_universal(edge_config)

        # Build edge list with strengths and sources
        edges = []
        for src, dst in self.graph.edges():
            # Get destination node parameters
            dst_params = self.node_parameters.get(dst, {})
            mechanism_type = dst_params.get("mechanism_type", "unknown")

            edge_info = {
                "source": src,
                "target": dst,
                "mechanism_type": mechanism_type,
            }

            # Sampling-based strength (no node of this release has the
            # "linear_combination"/"nonlinear"/"mediator" weights record that
            # would give an analytical one)
            sampling_strength = sampling_strengths.get((src, dst))
            if sampling_strength is None:
                raise ValueError(
                    f"No sampling strength found for edge {src} → {dst}"
                )

            edge_info["strength"] = float(sampling_strength)
            edge_info["strength_source"] = "sampling_correlation"
            edge_info["strength_sampling"] = float(sampling_strength)

            edges.append(edge_info)

        return {
            "nodes": list(self.graph.nodes()),
            "edges": edges,
            "n_nodes": len(self.graph.nodes()),
            "n_edges": len(self.graph.edges()),
            "edge_strength_methods": {
                "explicit_weights": "Normalized absolute weights from mechanism parameters",
                "sampling_correlation": "Correlation from generated samples",
            },
            "validation_info": {
                "description": "When both analytical and sampling strengths exist, ratio = sampling/analytical",
                "good_ratio_range": [0.8, 1.2],
                "interpretation": "Ratio close to 1.0 indicates sampling validates analytical strengths",
            },
        }







    def _compute_edge_strengths_universal(self, edge_strength_config=None):
        """
        Compute edge strengths by SAMPLING from the generative model.

        Parameters:
        -----------
        edge_strength_config : dict, optional
            Configuration with keys:
            - 'n_samples': number of samples to generate (default 1000)
            - 'method': 'sampling' (interventions removed)

        This works for ALL mechanism types (linear, nonlinear, neural, XGBoost).
        """
        # Extract n_samples from config
        n_samples = (edge_strength_config or {}).get("n_samples", 1000)

        logger.info(f"📊 Computing edge strengths from {n_samples} samples...")

        # Generate samples from the SCM
        samples = []
        for _ in range(n_samples):
            sample = self.generate_sample()
            samples.append(sample)

        df = pd.DataFrame(samples)

        # Compute edge strengths
        edge_strengths = {}

        for src, dst in self.graph.edges():
            # Get parent and child values
            parent_values = df[src].values
            child_values = df[dst].values

            # Compute absolute correlation as strength proxy
            correlation = np.abs(np.corrcoef(parent_values, child_values)[0, 1])

            # Handle NaN (constant variables)
            if np.isnan(correlation):
                correlation = 0.0

            edge_strengths[(src, dst)] = correlation

        # Normalize strengths per child node (so strengths to each node sum to 1.0)
        normalized_strengths = {}

        for node in self.graph.nodes():
            parents = list(self.graph.predecessors(node))
            if parents:
                # Get strengths for all edges into this node
                parent_strengths = {
                    p: edge_strengths.get((p, node), 0.0) for p in parents
                }

                # Normalize
                total_strength = sum(parent_strengths.values())
                if total_strength > 0:
                    for parent, strength in parent_strengths.items():
                        normalized_strengths[(parent, node)] = strength / total_strength
                else:
                    # All zero - use uniform
                    for parent in parents:
                        normalized_strengths[(parent, node)] = 1.0 / len(parents)

        logger.info(f"✅ Computed strengths for {len(normalized_strengths)} edges")

        return normalized_strengths
