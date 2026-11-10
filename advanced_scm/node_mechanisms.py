"""
Node mechanism definitions for the Advanced Extrusion SCM: the root (static)
nodes. Nodes with parents use the linear combinations of edge_mappings.py.

node_mechanisms.py
"""

import numpy as np
from typing import Dict


class MechanismFactory:
    """
    Factory class for creating node mechanisms based on configuration.
    """

    @staticmethod
    def create_static_node_mechanism(node_name: str, hyperparams: Dict):
        """Create the mechanism of a static root node: a standard normal draw."""

        def static_mechanism(params: Dict) -> float:
            # Static nodes are root nodes, so they only take params (like other root nodes)
            return np.random.normal(0, 1)

        return static_mechanism
