"""Audit the ground truth of every realization in a generated experiment.

    python -m advanced_scm.examples.audit_ground_truth --config configs/audit.yaml

Writes datasets/ground_truth_audit.json per realization and
ground_truth_audit_summary.json at the experiment root. Exits 1 if any
realization fails.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from advanced_scm.ground_truth_audit import audit_realization


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(parser.parse_args().config).read_text(encoding="utf-8"))
    logging.disable(logging.WARNING)  # replays repeat the generator's logging

    root = Path(cfg["experiment_dir"])
    index = json.loads((root / "experiment_index.json").read_text(encoding="utf-8"))
    summary = []
    for entry in index:
        report = audit_realization(
            root / entry["path"],
            min_effect=float(cfg.get("min_effect", 0.01)),
            sensor_n_jobs=int(cfg.get("sensor_n_jobs", -1)),
        )
        summary.append(
            {
                "path": entry["path"],
                "passed": report["passed"],
                "replay_matches_export": report["replay_matches_export"],
                "failed": report["failed"],
                "weak_direct_causes": report["weak_direct_causes"],
                "weak_ancestors": report["weak_ancestors"],
            }
        )
        print(
            f"{entry['path']}: {'PASS' if report['passed'] else 'FAIL'}"
            f"  failed={report['failed']}  weak={report['weak_direct_causes'] + report['weak_ancestors']}"
        )
    (root / "ground_truth_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0 if all(s["passed"] for s in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
