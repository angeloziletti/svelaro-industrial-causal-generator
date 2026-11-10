# column_registry.py
from typing import Dict, List, Set, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ColumnTypeInfo:
    patterns: List[str]
    description: str
    data_type: str  # "continuous", "binary", "categorical", "metadata"
    validation_fn: Optional[Callable] = None
    processing_hints: Optional[Dict] = None


class ColumnTypeRegistry:
    """Centralized registry for column types and their characteristics."""

    _instance = None
    _registry: Dict[str, ColumnTypeInfo] = {}
    _pattern_to_types: Dict[str, Set[str]] = defaultdict(set)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(
        cls,
        type_name: str,
        patterns: List[str],
        description: str,
        data_type: str,
        validation_fn: Optional[Callable] = None,
        processing_hints: Optional[Dict] = None,
    ):
        """Register a new column type."""
        info = ColumnTypeInfo(
            patterns, description, data_type, validation_fn, processing_hints
        )
        cls._registry[type_name] = info

        # Build reverse lookup
        for pattern in patterns:
            cls._pattern_to_types[pattern].add(type_name)

    @classmethod
    def get_patterns(cls, type_name: str) -> List[str]:
        """Get patterns for a column type."""
        return cls._registry.get(type_name, ColumnTypeInfo([], "", "")).patterns

    @classmethod
    def get_all_patterns(cls, type_names: List[str]) -> List[str]:
        """Get all patterns for multiple column types."""
        all_patterns = []
        for type_name in type_names:
            all_patterns.extend(cls.get_patterns(type_name))
        return list(set(all_patterns))  # Remove duplicates










    @classmethod
    def register_hierarchical(cls, parent_type: str, child_types: List[str]):
        """Register a parent type that includes all patterns from child types."""
        all_patterns = []
        all_descriptions = []

        for child_type in child_types:
            if child_type in cls._registry:
                child_info = cls._registry[child_type]
                all_patterns.extend(child_info.patterns)
                all_descriptions.append(f"{child_type}({child_info.description})")

        # Register parent with combined patterns
        cls.register(
            parent_type,
            all_patterns,
            f"Combined: {', '.join(all_descriptions)}",
            "mixed",  # Special data type for hierarchical
            processing_hints={"child_types": child_types},
        )





# Initialize default column types
def setup_default_column_types():
    registry = ColumnTypeRegistry()

    # First register the specific types
    registry.register(
        "defect_continuous",
        ["_defect"],
        "Continuous defect measurements",
        "continuous",
    )

    registry.register(
        "defect_binary",
        ["_label"],
        "Binary defect indicators",
        "binary",
    )

    # Contains-based defect type
    registry.register(
        "defect_words",
        ["defect", "label"],
        "Columns containing defect-related words",
        "mixed",
    )

    registry.register(
        "timeseries_timestamp",
        [
            "sequence_id",
            "timestep",
            "timestamp",
            "timestamp_key",
            "context_end_timestamp",
            "context_start_timestamp",
            "context_end_original",
            "context_start_original",
            "context_start",  # Debugging column: start of context window
            "context_end",  # Debugging column: end of context window
            "context_length_min",  # Debugging column: length of context in minutes
            # Hierarchical context time boundaries (for debugging/visualization)
            "label_start_sec",  # Label discretization window start
            "label_end_sec",  # Label discretization window end
            "max_context_start_sec",  # Max context window start
            "max_context_end_sec",  # Max context window end
            "chunk_start_sec",  # Alias for label discretization start
            "chunk_end_sec",  # Alias for label discretization end
            "chunk_idx",  # Label chunk index
            "context_index",  # Independent-run id (per-run split metadata, never a feature)
            "label_discretization_idx",  # Label chunk index (alias of chunk_idx)
            "label_discretization_start",  # Label chunk window start (timestamp)
            "label_discretization_end",  # Label chunk window end (timestamp)
            "label_discretization_duration_sec",  # Label chunk window duration
            # Sample time boundaries for analysis/debugging
            "sample_start_timestamp",  # Start timestamp of context window
            "sample_end_timestamp",  # End timestamp of context window
            # Bucket boundaries (B1-B10 covers typical use cases)
            "B1_start_sec", "B1_end_sec",
            "B2_start_sec", "B2_end_sec",
            "B3_start_sec", "B3_end_sec",
            "B4_start_sec", "B4_end_sec",
            "B5_start_sec", "B5_end_sec",
            "B6_start_sec", "B6_end_sec",
            "B7_start_sec", "B7_end_sec",
            "B8_start_sec", "B8_end_sec",
            "B9_start_sec", "B9_end_sec",
            "B10_start_sec", "B10_end_sec",
            # Multi-source tracking columns (metadata, not features)
            "source_folder",  # Parent folder of source Excel file
            "source_file",  # Source Excel filename
        ],
        "Time series timestamp and sequence columns",
        "metadata",
    )

    # Then register the combined type
    registry.register_hierarchical("defect_any", ["defect_continuous", "defect_binary"])


# Auto-setup on import
setup_default_column_types()
