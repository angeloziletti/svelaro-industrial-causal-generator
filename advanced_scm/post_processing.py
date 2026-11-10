import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from configs.column_registry import ColumnTypeRegistry
from logger_config import get_logger

logger = get_logger(__name__)

registry = ColumnTypeRegistry()


class PostProcessor:
    """
    Handles post-processing of generated data.
    Implements TabPFN-inspired post-processing procedures.
    """

    def __init__(self, hyperparams: Optional[Dict] = None):
        """
        Initialize the post-processor.

        Parameters:
        -----------
        hyperparams : Dict or None
            Hyperparameters for controlling post-processing
        """
        self.hyperparams = hyperparams or {}

    def apply_kumaraswamy_warping(
        self, values: np.ndarray, a: float = 0.5, b: float = 0.5
    ) -> np.ndarray:
        """
        Apply Kumaraswamy feature warping, introducing nonlinear distortions.

        TabPFN comparison:
        - TabPFN uses Kumaraswamy feature warping
        - This function implements that approach

        Parameters:
        -----------
        values : np.ndarray
            Values to transform
        a, b : float
            Kumaraswamy distribution parameters

        Returns:
        --------
        warped_values : np.ndarray
            Transformed values
        """
        # Handle empty arrays or None
        if values is None or len(values) == 0:
            return values

        # Scale values to [0, 1]
        min_val = np.min(values)
        max_val = np.max(values)

        # Handle case where all values are the same
        if max_val == min_val:
            return values

        scaled = (values - min_val) / (max_val - min_val + 1e-10)

        # Apply Kumaraswamy CDF: 1 - (1 - x^a)^b
        warped = 1 - (1 - np.power(scaled, a)) ** b

        # Scale back to original range
        result = warped * (max_val - min_val) + min_val

        return result

    def quantize_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        max_bins: int = 10,
        convert_to_category: bool = False,  # Added parameter
        exclude: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Quantize continuous features into discrete bins.

        TabPFN comparison:
        - TabPFN quantizes continuous features into buckets
        - This function implements that approach

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to process
        features : List[str] or None
            Features to quantize (if None, randomly select a subset)
        max_bins : int
            Maximum number of bins for quantization
        convert_to_category : bool
            Whether to convert the result to categorical (True) or keep as numeric bin indices (False)

        Returns:
        --------
        processed_df : pd.DataFrame
            Processed dataset
        """
        result_df = df.copy()

        # If features not specified, randomly select a subset
        if features is None:
            numeric_features = [
                c
                for c in df.select_dtypes(include=np.number).columns
                if c not in set(exclude or [])
            ]

            # Handle empty dataframe or no numeric features
            if not numeric_features:
                return result_df

            n_features = min(len(numeric_features), np.random.randint(1, 4))
            features = np.random.choice(
                numeric_features, size=n_features, replace=False
            )

        # Quantize each selected feature
        for feature in features:
            if feature in df.columns:
                # Check if there are enough unique values to quantize
                unique_values = df[feature].nunique()
                if unique_values <= 1:  # Skip features with only one unique value
                    continue

                # Determine number of bins (no more than unique values)
                n_bins = min(np.random.randint(2, max_bins + 1), unique_values)

                # Create bin edges
                values = df[feature].values
                bins = np.linspace(np.min(values), np.max(values), n_bins + 1)

                # Quantize the feature
                if convert_to_category:
                    # Convert to categorical with labels
                    labels = [f"{feature}_bin_{i}" for i in range(n_bins)]
                    result_df[feature] = pd.cut(df[feature], bins=bins, labels=labels)
                else:
                    # Keep as numeric bin indices (0 to n_bins-1)
                    result_df[feature] = pd.cut(df[feature], bins=bins, labels=False)

        return result_df

    def add_missing_values(
        self,
        df: pd.DataFrame,
        missing_fraction: Optional[float] = None,
        missing_columns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        result_df = df.copy()

        if missing_fraction is None:
            missing_fraction = np.random.uniform(0.01, 0.2)

        if missing_columns is None:
            missing_columns = df.columns.tolist()

        if exclude_patterns is None:
            exclude_patterns = registry.get_all_patterns(["defect_any"])

        original_missing_columns = missing_columns.copy()
        missing_columns = [
            col
            for col in missing_columns
            if not any(col.endswith(suffix) for suffix in exclude_patterns)
        ]

        excluded_cols = [
            col for col in original_missing_columns if col not in missing_columns
        ]
        if excluded_cols:
            logger.debug(f"🔒 EXCLUDED from missing values: {excluded_cols}")
        logger.debug(f"📝 WILL add missing values to: {missing_columns}")

        # Add missing values to selected columns
        for column in missing_columns:
            if column in df.columns:
                mask = np.random.random(len(df)) < missing_fraction
                result_df.loc[mask, column] = np.nan

        return result_df

    def apply_mixed_post_processing(
        self, df: pd.DataFrame, protected_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply a mix of post-processing techniques to the dataset.

        TabPFN comparison:
        - TabPFN applies a mix of post-processing techniques
        - This function applies multiple techniques with random parameters

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to process
        protected_columns : List[str] or None
            Columns no technique may touch. The SCM passes the nodes whose values
            drive downstream mechanisms (event parameters, confounders, defects):
            post-processing them would make the exported value differ from the one
            the mechanism used, and a NaN there silently breaks the causal link.

        Returns:
        --------
        processed_df : pd.DataFrame
            Processed dataset
        """
        result_df = df.copy()
        protected = set(protected_columns or [])

        # Initialize logging info
        applied_techniques = []
        technique_details = {}

        # Randomly decide which post-processing techniques to apply
        apply_warping = np.random.random() < 0.3
        apply_quantization = np.random.random() < 0.4
        # apply_missing = np.random.random() < 0.2
        apply_missing = np.random.random() < 0.2 and not self.hyperparams.get(
            "disable_missing_values", False
        )

        logger.debug(
            f"Post-processing decisions: warping={apply_warping}, quantization={apply_quantization}, missing={apply_missing}"
        )

        # Apply selected techniques (with error handling)
        if apply_warping:
            try:
                # Select a subset of numerical features to warp
                numeric_features = [
                    c
                    for c in df.select_dtypes(include=np.number).columns
                    if c not in protected
                ]
                if numeric_features:  # Only proceed if there are numeric features
                    n_features = max(1, int(np.random.random() * len(numeric_features)))
                    features_to_warp = np.random.choice(
                        numeric_features, size=n_features, replace=False
                    )

                    # Apply warping to selected features
                    warping_params = {}
                    for feature in features_to_warp:
                        a = np.random.uniform(0.3, 3.0)
                        b = np.random.uniform(0.3, 3.0)
                        result_df[feature] = self.apply_kumaraswamy_warping(
                            result_df[feature].values, a, b
                        )
                        warping_params[feature] = {"a": a, "b": b}

                    applied_techniques.append("kumaraswamy_warping")
                    technique_details["kumaraswamy_warping"] = {
                        "features_warped": list(features_to_warp),
                        "n_features_warped": len(features_to_warp),
                        "total_numeric_features": len(numeric_features),
                        "warping_parameters": warping_params,
                    }
                    logger.debug(
                        f"✓ Applied Kumaraswamy warping to {len(features_to_warp)} features: {list(features_to_warp)}"
                    )
                else:
                    logger.debug(
                        "⚠ Skipped Kumaraswamy warping: No numeric features found"
                    )
            except Exception as e:
                logger.warning(f"✗ Error applying Kumaraswamy warping: {e}")

        if apply_quantization:
            try:
                # Use numeric bin indices instead of categorical labels for compatibility
                result_df = self.quantize_features(
                    result_df, convert_to_category=False, exclude=list(protected)
                )
                applied_techniques.append("feature_quantization")
                technique_details["feature_quantization"] = {
                    "convert_to_category": False
                }
                logger.debug("✓ Applied feature quantization (numeric bin indices)")
            except Exception as e:
                logger.warning(f"✗ Error applying feature quantization: {e}")

        if apply_missing:
            try:
                exclude_suffixes = registry.get_all_patterns(["defect_any"])

                result_df = self.add_missing_values(
                    result_df,
                    missing_columns=[c for c in df.columns if c not in protected],
                    exclude_patterns=exclude_suffixes,
                )
                applied_techniques.append("missing_values")
                technique_details["missing_values"] = {
                    "method": "random_missing_values",
                    "excluded_patterns": exclude_suffixes,
                }
                logger.debug(
                    "✓ Applied missing value injection (excluding target variables)"
                )
            except Exception as e:
                logger.warning(f"✗ Error adding missing values: {e}")

        # Summary logging
        if applied_techniques:
            logger.debug(f"\nPost-processing Summary:")
            logger.debug(f"   Applied techniques: {', '.join(applied_techniques)}")
            logger.debug(f"   Total techniques applied: {len(applied_techniques)}/3")

            # Store technique details as dataframe attribute for later access
            if not hasattr(result_df, "attrs"):
                result_df.attrs = {}
            result_df.attrs["postprocessing_applied"] = applied_techniques
            result_df.attrs["postprocessing_details"] = technique_details
        else:
            logger.debug("Post-processing Summary: No techniques were applied")
            if not hasattr(result_df, "attrs"):
                result_df.attrs = {}
            result_df.attrs["postprocessing_applied"] = []
            result_df.attrs["postprocessing_details"] = {}

        return result_df
