"""
Edge mapping components for the Advanced Extrusion SCM.
Linear combinations of the parents.

edge_mappings.py
"""

import numpy as np
from typing import Dict, List, Optional, Callable





class LinearCombination:
    """
    Linear combination of parent values with optional noise.
    """

    def __init__(
        self, weights: Dict[str, float], bias: float = 0.0, noise_scale: float = 0.05
    ):
        """
        Initialize the linear combination.

        Parameters:
        -----------
        weights : Dict[str, float]
            Dictionary mapping parent names to their weights
        bias : float
            Bias term
        noise_scale : float
            Scale of the random noise to add to the output
        """
        self.weights = weights
        self.bias = bias
        self.noise_scale = noise_scale

    def apply(self, parent_values: Dict[str, float]) -> float:
        """
        Apply the linear combination to the input values.

        Parameters:
        -----------
        parent_values : Dict[str, float]
            Dictionary of parent node values

        Returns:
        --------
        result : float
            Output value
        """
        # Compute weighted sum of parent values with validation
        weighted_sum = 0.0
        for p, v in parent_values.items():
            weight = self.weights.get(p, 0)
            # Validate parent value is finite, use 0.0 if not
            if not np.isfinite(v):
                v = 0.0
            weighted_sum += weight * v

        # Clip weighted sum to prevent overflow
        weighted_sum = np.clip(weighted_sum, -100, 100)

        # Add bias and noise
        noise = np.random.normal(0, self.noise_scale)
        result = weighted_sum + self.bias + noise

        # Ensure result is finite, clip if not
        if not np.isfinite(result):
            result = np.clip(weighted_sum + self.bias, -100, 100)

        return result






class EdgeMappingFactory:
    """
    Factory class for creating edge mapping components based on configuration.
    """



    @staticmethod
    def create_linear_combination_mapping(
        parents: List[str],
        bias: float = 0.0,
        noise_scale: float = 0.05,
        equal_weights: bool = False,
        per_parent_weights: Optional[Dict[str, float]] = None,
    ) -> Callable:
        """
        Create a linear combination edge mapping function.

        Parameters:
        -----------
        parents : List[str]
            List of parent node names
        bias : float
            Bias term
        noise_scale : float
            Scale of the random noise to add to the output
        equal_weights : bool
            Whether to use equal weights for all parents
        per_parent_weights : Optional[Dict[str, float]]
            Explicit per-parent weights, overriding equal_weights.
            Used by shadow red-herring confounder edges to give Z_k variable
            strength per shadow target. Parents missing from this dict
            default to 1.0 (e.g., when only some edges have weight attrs).

        Returns:
        --------
        mapping_func : Callable
            Function that applies the linear combination mapping
        """
        # Generate weights
        if per_parent_weights is not None:
            # Explicit per-parent override (shadow confounders).
            # Parents NOT in per_parent_weights fall back to the equal-weight
            # baseline (1/n). This preserves the equal-weight invariant for
            # the unchanged parents — adding a Z_k edge to a GP node ONLY
            # changes Z_k's weight, leaving the other parents' contributions
            # at their 1/n baseline. The shadow "strength" w_Z is then
            # interpretable as the absolute coefficient; comparing it to 1/n
            # tells you Z_k's relative strength vs a single regular parent.
            default = 1.0 / len(parents)
            weights = {
                p: float(per_parent_weights.get(p, default)) for p in parents
            }
        elif equal_weights:
            weights = {p: 1.0 / len(parents) for p in parents}
        else:
            weights = {p: np.random.uniform(-1, 1) for p in parents}

        # Create the linear combination
        linear_comb = LinearCombination(weights, bias, noise_scale)

        # Create the mapping function
        def linear_combination_mapping(
            parent_values: Dict[str, float], _: List[str]
        ) -> float:
            return linear_comb.apply(parent_values)

        return linear_combination_mapping, linear_comb

