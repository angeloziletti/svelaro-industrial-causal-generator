"""The difficulty map: its data files cover every published configuration, and the sample plants reproduce the
paper's point for the reference configuration."""
import csv
import json
from pathlib import Path

import numpy as np

from checks import difficulty_map as dmap

ROOT = Path(__file__).resolve().parents[1]


def _published():
    return {str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "configs").rglob("*.yaml")
            if "checks" not in p.parts and p.name != "audit.yaml"}


def test_resolution_and_paper_points_cover_every_configuration():
    with open(ROOT / "checks" / "difficulty_map_resolution.csv", encoding="utf-8") as f:
        res = {r["configuration"] for r in csv.DictReader(f)}
    paper = json.loads((ROOT / "checks" / "difficulty_map_paper.json").read_text(encoding="utf-8"))["points"]
    assert res == _published()
    assert {p["configuration"] for p in paper if "configuration" in p} == _published()


def test_sample_plants_reproduce_the_reference_point():
    with open(ROOT / "checks" / "difficulty_map_resolution.csv", encoding="utf-8") as f:
        factor, n_lags = next((int(r["factor"]), int(r["n_lags"])) for r in csv.DictReader(f)
                              if r["configuration"] == "configs/reference.yaml")
    per = []
    for plant in sorted((ROOT / "samples" / "batch_000").glob("graph_*/realization_0")):
        df, roles = dmap.plant_problem(plant)
        per.append(dmap.measure(df, dmap.TARGET, roles, factor, n_lags, dmap.BATCH))
    paper = next(p for p in json.loads((ROOT / "checks" / "difficulty_map_paper.json").read_text(encoding="utf-8"))
                 ["points"] if p.get("configuration") == "configs/reference.yaml")
    assert len(per) == 10
    assert abs(np.median([m["test_strength"] for m in per]) - paper["strength"]) < 1e-6 * paper["strength"]
    assert abs(np.mean([m["lookalike_share"] for m in per]) - paper["lookalike_share"]) < 1e-9
