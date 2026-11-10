"""
Batch Dataset Generator — multi-graph, multi-realization SCM generation.

Realizations are generated one after the other. Parallelism lives one level
down, inside each realization's per-sensor generation loop — see
``gp_parameter_layer.sensor_n_jobs`` in the YAML.

Usage:
    python -m advanced_scm.examples.generate_batch_datasets \
        --config configs/reference.yaml
"""

import argparse
import copy
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from advanced_scm.tabular_dataset_generator import TabularDatasetGenerator
from advanced_scm.unified_config import UnifiedSCMConfig
from logger_config import get_logger, setup_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate batch of independent graph datasets"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = UnifiedSCMConfig.from_yaml(args.config)

    batch_config = config.batch_generation or {}
    n_graphs = batch_config.get("n_graphs", 1)
    # instances first_graph .. first_graph + n_graphs - 1: the seeds depend only on the instance number, so a
    # cell can be extended with more instances that are the same as if it had been generated in one go
    first_graph = int(batch_config.get("first_graph", 0) or 0)
    # `graphs: [2, 3, 5]` generates exactly these instances instead of first_graph .. first_graph + n_graphs - 1
    graph_ids = [int(g) for g in batch_config["graphs"]] if batch_config.get("graphs") else list(
        range(first_graph, first_graph + n_graphs))
    n_graphs = len(graph_ids)
    n_realizations_per_graph = batch_config.get("n_realizations_per_graph", 1)

    output_base_dir = Path(config.output_directory)

    logger.info("=" * 80)
    logger.info("🚀 BATCH DATASET GENERATION")
    logger.info("=" * 80)
    logger.info(f"   n_graphs: {n_graphs}")
    logger.info(f"   n_realizations_per_graph: {n_realizations_per_graph}")
    logger.info(f"   Total datasets: {n_graphs * n_realizations_per_graph}")
    logger.info(f"   Base seed: {config.seed}")
    logger.info(f"   Output directory: {output_base_dir}")
    logger.info("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = output_base_dir / f"experiment_{timestamp}"
    experiment_root.mkdir(parents=True, exist_ok=True)

    setup_logger(
        log_to_file=True,
        calculations_dir=str(experiment_root),
        log_file_prefix="scm_batch_generation",
    )

    logger.info(f"📁 Experiment root: {experiment_root}")
    logger.info(
        f"📝 Log: {experiment_root}/logs/scm_batch_generation_{timestamp}.log"
    )

    sensor_n_jobs = getattr(config, "sensor_n_jobs", 1)
    logger.info(f"⚙️  sensor_n_jobs (per-realization parallelism): {sensor_n_jobs}")

    # TabularDatasetGenerator auto-creates a timestamped base; we override
    # output_dir per realization, so the temp dir is essentially vestigial.
    temp_generator_dir = experiment_root / "_temp_generator"
    temp_generator_dir.mkdir(parents=True, exist_ok=True)
    generator = TabularDatasetGenerator(output_dir=str(temp_generator_dir))

    experiment_index: list = []
    total_datasets = n_graphs * n_realizations_per_graph
    current_dataset = 0

    for graph_id in graph_ids:
        for real_id in range(n_realizations_per_graph):
            current_dataset += 1
            batch_id = graph_id // 100
            output_path = (
                experiment_root
                / f"batch_{batch_id:03d}"
                / f"graph_{graph_id:04d}"
                / f"realization_{real_id}"
            )

            logger.info("")
            logger.info(
                f"[{current_dataset}/{total_datasets}] "
                f"batch_{batch_id:03d}/graph_{graph_id:04d}/"
                f"realization_{real_id}"
            )

            seed = config.seed + graph_id * 10_000 + real_id
            config_copy = copy.deepcopy(config)
            config_copy.seed = seed
            # The graph is shared by every realization of this graph_id; only
            # the data seed varies across them.
            config_copy.graph_seed = config.seed + graph_id * 10_000
            logger.info(f"   ▶ seed={seed}")

            try:
                generator.generate_dataset(
                    unified_config=config_copy,
                    variant_id=0,
                    output_dir=output_path,
                )
            except Exception as e:
                logger.error(f"   ❌ Generation failed: {e}")
                raise

            metadata_path = output_path / "datasets" / "dataset_metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            graph_info = metadata.get("graph_info", {})
            experiment_index.append(
                {
                    "graph_id": graph_id,
                    "realization_id": real_id,
                    "batch_id": batch_id,
                    "path": output_path.relative_to(experiment_root).as_posix(),
                    "n_nodes": graph_info.get("nodes", 0),
                    "n_edges": graph_info.get("edges", 0),
                    "seed": seed,
                    "n_samples": config.n_samples,
                }
            )
            logger.info(
                f"   ✅ Slot complete. Nodes: {graph_info.get('nodes', 0)}, "
                f"Edges: {graph_info.get('edges', 0)}"
            )

    # Clean up the vestigial temp generator directory.
    if temp_generator_dir.exists():
        try:
            shutil.rmtree(temp_generator_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"⚠️ Could not remove temp directory: {e}")

    config_snapshot = experiment_root / "experiment_config.yaml"
    shutil.copy(args.config, config_snapshot)
    logger.info(f"📄 Config snapshot saved: {config_snapshot}")

    index_path = experiment_root / "experiment_index.json"
    with open(index_path, "w") as f:
        json.dump(experiment_index, f, indent=2)

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ BATCH GENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"   Total datasets generated: {len(experiment_index)}")
    logger.info(f"   Experiment root: {experiment_root}")
    logger.info(f"   Index saved: {index_path}")
    logger.info("=" * 80)

    if experiment_index:
        df = pd.DataFrame(experiment_index)
        logger.info("")
        logger.info("📊 Summary Statistics:")
        logger.info(f"   Mean nodes: {df['n_nodes'].mean():.1f}")
        logger.info(f"   Mean edges: {df['n_edges'].mean():.1f}")
        logger.info(f"   Node range: {df['n_nodes'].min()}-{df['n_nodes'].max()}")
        logger.info(f"   Edge range: {df['n_edges'].min()}-{df['n_edges'].max()}")


if __name__ == "__main__":
    main()
