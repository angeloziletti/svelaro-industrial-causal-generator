"""
Temporal Data Exporter - Extracted export methods from UnifiedTemporalProcessor.

This module contains all data export logic for temporal processing results,
providing cleaner separation of concerns.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from io_utils import timeseries_write_path, write_timeseries
from logger_config import get_logger

logger = get_logger(__name__)


class TemporalDataExporter:
    """
    Handles all data export tasks for temporal processing.

    Extracted from UnifiedTemporalProcessor to reduce monolith size and
    improve separation of concerns.
    """

    def __init__(self, unified_config, export_dir: Path, logger_instance=None):
        """
        Initialize the temporal data exporter.

        Args:
            unified_config: Unified configuration object
            export_dir: Base export directory
            logger_instance: Optional logger instance
        """
        self.unified_config = unified_config
        self.export_dir = Path(export_dir)
        self.logger = logger_instance or logger

    def export_temporal_sample_from_structured(
        self,
        structured_time_series: Dict[str, np.ndarray],
        temporal_feature_names: List[str],
        export_dir: Path,
        n_sample_series: int = 10,
        export_full_dataset: bool = False,
        include_scm_metadata: Optional[List[Dict]] = None,
    ) -> str:
        """
        Export a sample of complete time series to Excel using structured data.
        Optionally export the sensor metadata and SCM metadata as JSON.

        Args:
            structured_time_series: Dict mapping sensor names to arrays of shape (n_samples, sequence_length)
            temporal_feature_names: List of feature names for column ordering
            export_dir: Directory to export to
            n_sample_series: Number of complete series to sample (default 10)
            export_full_dataset: If True, export the sensor metadata as JSON (default False)
            include_scm_metadata: Optional SCM sample metadata to export as JSON (default None)

        Returns:
            Path to exported Excel file
        """
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"📊 Exporting sample with {n_sample_series} complete time series from structured data..."
        )

        # Get dimensions from first sensor
        first_sensor = list(structured_time_series.keys())[0]
        n_samples, sequence_length = structured_time_series[first_sensor].shape

        self.logger.info(
            f"   Data dimensions: {n_samples} samples × {sequence_length} timesteps"
        )
        self.logger.info(f"   Available sensors: {list(structured_time_series.keys())}")

        # === EXPORT FULL DATASET (if requested) ===
        if export_full_dataset:
            self.logger.info("💾 Exporting complete temporal dataset...")
            self.logger.info(f"   Shape: {n_samples} samples × {sequence_length} timesteps")
            self.logger.info(f"   Sensors: {list(structured_time_series.keys())}")

            # Export sensor metadata
            sensor_metadata = {
                "n_samples": int(n_samples),
                "sequence_length": int(sequence_length),
                "sensors": list(structured_time_series.keys()),
                "temporal_feature_names": temporal_feature_names,
            }
            metadata_path = export_dir / "sensor_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(sensor_metadata, f, indent=2)
            self.logger.info(f"✅ Sensor metadata exported: {metadata_path}")

        # === EXPORT SCM METADATA (if provided) ===
        if include_scm_metadata is not None:
            self.logger.info("💾 Exporting SCM node values...")
            scm_metadata_path = export_dir / "scm_node_values.json"
            with open(scm_metadata_path, "w") as f:
                json.dump(include_scm_metadata, f, indent=2)
            self.logger.info(f"✅ SCM metadata exported: {scm_metadata_path}")
            self.logger.info(f"   Exported {len(include_scm_metadata)} sample records")

        # Limit sample series to available data
        n_sample_series = min(n_sample_series, n_samples)
        sample_indices = list(range(n_sample_series))

        # Create export data
        export_data = []

        for sample_idx in sample_indices:
            for timestep in range(sequence_length):
                row = {
                    "sample_index": sample_idx,
                    "timestep": timestep,
                    "timestamp": pd.Timestamp("2024-01-01")
                    + pd.Timedelta(minutes=timestep * 5),
                }

                # Add sensor values for this timestep
                for sensor_name in temporal_feature_names:
                    if sensor_name in structured_time_series:
                        row[sensor_name] = structured_time_series[sensor_name][
                            sample_idx, timestep
                        ]
                    else:
                        row[sensor_name] = np.nan

                export_data.append(row)

        # Create DataFrame
        export_df = pd.DataFrame(export_data)

        # Reorder columns: sample_index, timestep, timestamp, then sensors in config order
        base_columns = ["sample_index", "timestep", "timestamp"]
        sensor_columns = [
            col for col in temporal_feature_names if col in export_df.columns
        ]
        final_columns = base_columns + sensor_columns
        export_df = export_df[final_columns]

        # Sort for readability
        export_df = export_df.sort_values(["sample_index", "timestep"])

        self.logger.info(f"   Sensors exported: {sensor_columns}")

        available_sensors = list(structured_time_series.keys())
        self._check_missing_sensors(
            available_sensors=available_sensors,
            requested_sensors=temporal_feature_names,
            exported_sensors=sensor_columns,
        )

        export_path = export_dir / "temporal_sequences_sample.xlsx"
        export_df.to_excel(export_path, index=False)

        self.logger.info(f"✅ Exported: {export_path}")

        return str(export_path)

    def _check_missing_sensors(
        self,
        available_sensors: List[str],
        requested_sensors: List[str],
        exported_sensors: List[str],
    ) -> None:
        """
        Check for missing sensors and log warnings.

        Args:
            available_sensors: List of all available sensor names
            requested_sensors: List of requested sensor names from config
            exported_sensors: List of actually exported sensor names
        """
        # Check if any available sensors were not requested
        missing_from_config = set(available_sensors) - set(requested_sensors)
        if missing_from_config:
            self.logger.warning(
                f"⚠️  Sensors available but not in config: {sorted(missing_from_config)}"
            )

        # Check if any requested sensors were not exported
        missing_from_export = set(requested_sensors) - set(exported_sensors)
        if missing_from_export:
            self.logger.warning(
                f"⚠️  Sensors in config but not exported: {sorted(missing_from_export)}"
            )

    def export_timeseries(
        self,
        structured_time_series: Dict[str, np.ndarray],
        tabular_df: pd.DataFrame,
        label_columns: List[str],
        samples_per_context: int = 10,
    ) -> Path:
        """
        Materialize a long-format timeseries.

        Combines the per-sample raw time series with the per-sample defect/label
        values, broadcasting scalar labels across each sample's sequence_length
        timesteps. Adds a synthetic 1-second-spaced Timestamp column so
        downsampling (binned by resolution_sec=1) works without
        further config. Written to ``<export_dir.parent>/full_timeseries.parquet``.

        Args:
            structured_time_series: sensor name -> (n_samples, sequence_length) array
            tabular_df: Sample-level dataframe containing label columns
            label_columns: Names of label columns to broadcast across timesteps
            samples_per_context: Number of consecutive samples to group into
                one context (via a ``context_index`` column the
                loader splits on). K=10 with N=100 yields 10 contexts of
                1280 timesteps each — multiple pattern events per context for
                GP-MDL scoring while keeping contexts independent.

        Returns:
            Path to the written timeseries file.
        """
        if not structured_time_series:
            raise ValueError("structured_time_series is empty; cannot export")

        sensor_names = list(structured_time_series.keys())
        first = structured_time_series[sensor_names[0]]
        n_samples, seq_len = first.shape
        total_rows = n_samples * seq_len

        if len(tabular_df) != n_samples:
            raise ValueError(
                f"tabular_df has {len(tabular_df)} rows but structured_time_series "
                f"has {n_samples} samples"
            )

        # Flatten sensor arrays sample-by-sample (sample 0 timesteps 0..L, then sample 1, ...)
        df = pd.DataFrame(
            {sensor: structured_time_series[sensor].reshape(-1) for sensor in sensor_names}
        )
        # 1-second spacing aligns with the downsample resolution_sec=1 default,
        # so max_samples can clamp via the time-binned downsampler.
        timestamps = pd.date_range(
            start="2024-01-01", periods=total_rows, freq="1s"
        )
        df.insert(0, "Timestamp", timestamps)

        # Group K consecutive samples into one context. The loader
        # (the loader) splits the DataFrame on this column
        # so each context = K samples concatenated → multiple pattern events
        # per context (good for GP-MDL) while contexts stay independent.
        context_idx = np.repeat(np.arange(n_samples), seq_len) // max(
            1, int(samples_per_context)
        )
        df.insert(1, "context_index", context_idx.astype(int))

        # Broadcast each sample's defect/label scalar across its seq_len timesteps
        for col in label_columns:
            if col in tabular_df.columns:
                df[col] = np.repeat(tabular_df[col].values, seq_len)

        # Also broadcast every other sample-level numeric scalar from scm_df
        # (temperature, machine_nb, GP parameter nodes, etc.). This is required
        # so the synthetic pipeline's round-trip through SensorDataProcessor
        # preserves static SCM scalars: SensorDataProcessor.load_sensor_data()
        # reads them into sensor_df, and Step 6b
        # (unified_temporal_processor.py:2185-2272) extracts the per-window
        # median for every classification_info.semi_static_features column.
        # Constant within a sample (one SCM sample = one analysis unit), so
        # median = the constant value. No NaN, no special-case handling.
        already_in_df = set(df.columns)
        scalar_cols = [
            c for c in tabular_df.columns
            if c not in already_in_df
            and pd.api.types.is_numeric_dtype(tabular_df[c])
        ]
        for col in scalar_cols:
            df[col] = np.repeat(tabular_df[col].values, seq_len)
        if scalar_cols:
            self.logger.info(
                f"   📊 Broadcast {len(scalar_cols)} sample-level scalar(s) "
                f"per timestep: {scalar_cols[:5]}{'...' if len(scalar_cols) > 5 else ''}"
            )

        # Alongside the parent of self.export_dir (i.e. the realization root)
        output_path = timeseries_write_path(self.export_dir.parent)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Stored in single precision (7 significant digits; halves the file).
        write_timeseries(df, output_path, float32=True)
        self.logger.info(
            f"   ✅ long-format timeseries: {output_path} "
            f"({total_rows} rows, {len(sensor_names)} sensors, {len(label_columns)} labels)"
        )
        return output_path
