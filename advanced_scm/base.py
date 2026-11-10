"""
Base class for the Advanced Extrusion SCM

base.py
"""

import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, Optional
from logger_config import get_logger

logger = get_logger(__name__)


class AdvancedExtrusionSCMBase:
    """
    Base class for Advanced Extrusion Structural Causal Model (SCM).
    Provides the basic structure for implementing TabPFN-inspired causal modeling.

    This class maintains the state of the causal graph and provides
    methods for dataset generation.
    """

    def __init__(self, hyperparams: Optional[Dict] = None, vector_dim: int = 8):
        """
        Initialize the base SCM.

        Parameters:
        -----------
        hyperparams : Dict or None
            Hyperparameters for controlling the model's behavior
        vector_dim : int
            Dimension of the vector representation for nodes
        """
        self.hyperparams = hyperparams or {}
        self.vector_dim = vector_dim

        # Core graph components
        self.graph = None  # The networkx graph representing the causal structure
        self.node_types = (
            {}
        )  # Tracks node types: static (scalar values, some with time series), defect, gp_parameter
        self.node_vectors = {}  # Stores vector representations of nodes

        # Mechanism components
        self.node_mechanisms = {}  # Stores the computation functions for each node
        self.node_parameters = {}  # Stores parameters for the mechanisms

        # NaN tracking for generation diagnostics
        self._nan_counts = {}  # Tracks NaN occurrences per node
        self._first_nan_logged = set()  # Tracks which nodes have had first NaN logged

    def generate_sample(self, intervention: Optional[Dict] = None):
        """
        Generate a single sample by propagating through the causal graph.

        intervention: optional {node: value or function of the natural value}.
            The node's mechanism still runs -- so it draws the same noise and
            every other node sees the same random stream -- and its value is then
            replaced. Descendants are computed from the intervened value: a
            counterfactual do(node := value) with all noise held fixed.

        Returns:
        --------
        sample : Dict
            Dictionary of node values representing a single data point
        """
        sample = {}

        # Topological sort ensures we process nodes in order (parents before children)
        for node in nx.topological_sort(self.graph):
            # Get parent nodes and their values
            parents = list(self.graph.predecessors(node))
            parent_values = {p: sample[p] for p in parents if p in sample}

            # Calculate node value using its mechanism and parameters
            node_func = self.node_mechanisms.get(node)
            node_params = self.node_parameters.get(node, {})

            if not parents:  # Root node
                value = node_func(node_params)
            else:  # Non-root node
                value = node_func(parent_values, node_params)

            if intervention and node in intervention:
                v = intervention[node]
                value = v(value) if callable(v) else v

            # Store the value
            sample[node] = value

            # Validate node value (catch NaN/Inf early)
            if not np.isfinite(value):
                node_type = self.node_types.get(node, "unknown")

                # Track NaN occurrence
                if node not in self._nan_counts:
                    self._nan_counts[node] = 0
                self._nan_counts[node] += 1

                # Log first occurrence only to avoid spam
                if node not in self._first_nan_logged:
                    if parents:
                        parent_info = {p: sample[p] for p in parents}
                        logger.warning(
                            f"❌ Node '{node}' produced non-finite value: {value}\n"
                            f"   Type: {node_type}\n"
                            f"   Parents: {parent_info}\n"
                            f"   Mechanism params: {node_params}\n"
                            f"   (subsequent occurrences will be counted silently)"
                        )
                    else:
                        logger.warning(
                            f"❌ Root node '{node}' produced non-finite value: {value}\n"
                            f"   Type: {node_type}\n"
                            f"   Mechanism params: {node_params}\n"
                            f"   (subsequent occurrences will be counted silently)"
                        )
                    self._first_nan_logged.add(node)

                # Set to a safe default value to allow debugging
                sample[node] = 0.0

        return sample

    def get_nan_summary(self) -> str:
        """
        Get a summary of NaN occurrences during dataset generation.

        Returns:
        --------
        summary : str
            Human-readable summary of NaN counts per node, or empty string if no NaN
        """
        if not self._nan_counts:
            return ""

        total_nans = sum(self._nan_counts.values())
        summary_lines = [f"\n⚠️ Dataset Generation NaN Summary: {total_nans} total non-finite values"]

        # Sort by count (descending) and show top offenders
        sorted_nodes = sorted(self._nan_counts.items(), key=lambda x: x[1], reverse=True)

        for node, count in sorted_nodes[:10]:  # Show top 10
            node_type = self.node_types.get(node, "unknown")
            summary_lines.append(f"   {node} ({node_type}): {count} occurrences")

        if len(sorted_nodes) > 10:
            summary_lines.append(f"   ... and {len(sorted_nodes) - 10} more nodes")

        summary_lines.append("   All non-finite values were replaced with 0.0")

        return "\n".join(summary_lines)

    def reset_nan_tracking(self):
        """Reset NaN tracking counters for a new dataset generation."""
        self._nan_counts = {}
        self._first_nan_logged = set()

    def generate_dataset(
        self,
        n_samples: int,
        apply_post_processing: bool = True,
        intervention: Optional[Dict] = None,
    ):
        """
        Generate a dataset of n samples.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        apply_post_processing : bool
            Whether to apply post-processing to the generated data

        Returns:
        --------
        dataset : DataFrame
            Pandas DataFrame containing the generated samples
        """
        # Reset NaN tracking for this dataset generation
        self.reset_nan_tracking()

        dataset = []
        for _ in range(n_samples):
            sample = self.generate_sample(intervention)
            dataset.append(sample)

        # Create a DataFrame from the samples
        df = pd.DataFrame(dataset)

        # Log NaN summary if any occurred
        nan_summary = self.get_nan_summary()
        if nan_summary:
            logger.warning(nan_summary)

        if apply_post_processing:
            # Post-processing will be implemented by derived class or post_processor
            df = self._apply_post_processing(df)

        return df


