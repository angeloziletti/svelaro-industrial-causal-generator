"""The difficulty map of the accompanying paper (Appendix "Difficulty measured on the data").

    python -m checks.difficulty_map --config configs/checks/difficulty_map.yaml

Two measures of how hard it is to find the direct causes of the target, computed on the data at the resolution the
methods see (block means of `factor` steps within each batch, `n_lags` lags; checks/estimator.py):

  test strength      for each direct cause, its partial R^2 given the target's own past and the other direct causes
                     (out of sample, folds split by batch), as the effect size f^2 = R^2 / (1 - R^2), times the number
                     of rows: the expected statistic of a test for that cause. A plant's value is the median over its
                     direct causes; a configuration's value is the median over its plants.
  look-alike share   the share of wrong candidates whose past (lags 1..L) carries more k-nearest-neighbour mutual
                     information about the target than the past of the median direct cause. It needs only the direct
                     causes, so it applies to public datasets whose other roles are unknown. A configuration's value
                     is the mean over its plants.

Our plants are read from each configuration's latest generated experiment (generate them first), at the factor and
n_lags the paper used for that configuration (checks/difficulty_map_resolution.csv). The public problems are optional:
set their paths in the configuration (see the README for where to obtain the files); each is measured at the paper's
resolution rule applied to its target. The output lists every point; with `compare_with`, each is printed beside the
paper's value.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_regression

from advanced_scm import shares
from checks.estimator import BATCH, TARGET, cv_r2, episode_blocks, lagged

CAUSE = "direct_parent"
SIBLING = "sibling"


def measure(df: pd.DataFrame, target: str, roles: Dict[str, str], factor: int, n_lags: int, episode_col: str,
            folds: int = 5, alpha: float = 1e-3, mi_rows: int = 4000) -> Dict:
    """Test strength and look-alike share of one problem. `roles`: candidate -> role (CAUSE for a direct cause)."""
    cands = list(roles)
    col = {c: i for i, c in enumerate(cands + [target])}
    X, y, g = lagged(episode_blocks(df, cands + [target], episode_col, factor), n_lags, len(cands))
    own = X[:, col[target], 1:]

    def r2(names):  # out-of-sample R^2 of the target from its own past and the named candidates, floored at 0
        return max(0.0, cv_r2(np.concatenate([X[:, col[c], :] for c in names] + [own], axis=1), y, g, folds, alpha))

    causes = [c for c in cands if roles[c] == CAUSE]
    wrong = [c for c in cands if roles[c] != CAUSE]
    full = r2(causes)
    partial = {}
    for c in causes:
        drop = r2([d for d in causes if d != c])
        partial[c] = max(0.0, (full - drop) / (1 - drop)) if drop < 1 else 0.0
    lam = [len(y) * (p / (1 - p) if p < 1 else 1e9) for p in partial.values()]
    idx = np.random.default_rng(0).choice(len(y), min(len(y), mi_rows), replace=False)
    mi = {c: float(mutual_info_regression(X[idx, col[c], 1:], y[idx], n_neighbors=5, random_state=0).max())
          if X[:, col[c], 1:].std() > 0 else 0.0 for c in cands}
    med = float(np.median([mi[c] for c in causes]))
    return {"test_strength": float(np.median(lam)), "lookalike_share": float(np.mean([mi[c] > med for c in wrong])),
            "partial_cause": partial, "rows": int(len(y)), "factor": factor, "n_lags": n_lags}


def plant_problem(plant: Path):
    """The plant's series and the role of every recorded variable, from its ground truth."""
    meta = json.loads((plant / "datasets" / "dataset_metadata.json").read_text(encoding="utf-8"))
    labels = meta["target_driver_labels"]["labels"]
    roles = {s: (SIBLING if v["label"] == "non_cause" and v.get("confounded_with_target") else v["label"])
             for s, v in labels.items() if v.get("candidate", True) and v.get("kind", "sensor") == "sensor"}
    df = pd.read_parquet(plant / "full_timeseries.parquet", columns=list(roles) + [TARGET, BATCH])
    return df, roles


def matched_resolution(episodes: List[np.ndarray], lo: int = 4, hi: int = 64) -> tuple:
    """The paper's rule for one problem: tau = the target's within-episode, noise-robust decorrelation time (episodes
    cut to the shortest); factor = the largest power of two <= tau, clamped to [lo, hi]; n_lags = ceil(2 tau / f)."""
    T = min(len(e) for e in episodes)
    tau = shares.process_tau(np.stack([np.asarray(e[:T], float) for e in episodes]))
    f = int(min(hi, max(lo, 2 ** int(np.floor(np.log2(max(tau, 1)))))))
    return f, int(max(1, math.ceil(2 * tau / f)))


# ---- public problems, from the files their authors publish ----

def te_problems(c: Dict):
    """Tennessee Eastman: the 22 classic test runs d00_te.dat ... d21_te.dat (one episode each; XMEAS 1-22 and XMV 1-11
    as X1-X33) and the CIPCaD-Bench graph (rows = parent, columns = child). Every variable with exactly one parent is a
    target; its parent is its direct cause and every other variable counts as wrong."""
    cols = list(range(22)) + list(range(41, 52))
    names = [f"X{i + 1}" for i in range(33)]
    runs = [np.loadtxt(Path(c["runs_dir"]) / f"d{i:02d}_te.dat")[:, cols] for i in range(22)]
    df = pd.concat([pd.DataFrame(r, columns=names).assign(episode=i) for i, r in enumerate(runs)], ignore_index=True)
    adj = np.loadtxt(c["graph"])
    for j in np.nonzero(adj.sum(axis=0) == 1)[0]:
        target, parent = names[j], names[int(np.argmax(adj[:, j]))]
        yield f"te/{target}", df, target, {v: CAUSE if v == parent else "other" for v in names if v != target}


def indpensim_problems(c: Dict):
    """IndPenSim: the 100-batch release (one table; batches recovered from the time column resetting). Target the
    production rate dP (penicillin concentration differenced within each batch); one problem per control regime.
    The roles of the online variables are given in the configuration."""
    rename = c["columns"]
    path = Path(c["data"])
    usecols = [c["time_column"]] + list(rename)
    raw = (pd.read_parquet(path, columns=usecols) if path.suffix == ".parquet"
           else pd.read_csv(path, usecols=usecols, low_memory=False))
    t = pd.to_numeric(raw[c["time_column"]], errors="coerce").to_numpy(float)
    raw["batch"] = np.cumsum(np.concatenate([[True], np.diff(t) < 0]))
    raw = raw.rename(columns=rename)
    raw["dP"] = raw.groupby("batch")["P"].diff()
    raw = raw.dropna(subset=["dP"])
    r = c["roles"]
    roles = {**{v: CAUSE for v in r["causes"]}, **{v: "ancestor" for v in r["ancestors"]},
             **{v: "descendant" for v in r["descendants"]}}
    for name, (b0, b1) in c["regimes"].items():
        yield f"indpensim/{name}", raw[raw["batch"].between(b0, b1)], "dP", dict(roles)


def uf_problems(c: Dict):
    """The food-processing line of CIPCaD-Bench (UF.csv), a real plant: one run at a fixed sampling interval, cut into
    episodes wherever a sample is missing; episodes shorter than min_rows are dropped."""
    u = pd.read_csv(c["csv"], parse_dates=["DateTime"])
    u["episode"] = u["DateTime"].diff().dt.total_seconds().ne(c["sampling_seconds"]).cumsum()
    u = u[u.groupby("episode")["episode"].transform("size") >= c["min_rows"]]
    yield "uf/" + c["target"], u, c["target"], {v: CAUSE if v in c["causes"] else "ancestor" for v in c["candidates"]}


PUBLIC = {"te": te_problems, "indpensim": indpensim_problems, "uf": uf_problems}


def latest_experiment(config: Path) -> Path:
    out = Path(yaml.safe_load(config.read_text(encoding="utf-8"))["output"]["directory"])
    exps = sorted(out.glob("experiment_*"))
    if not exps:
        raise SystemExit(f"{config}: nothing generated in {out}; generate it first")
    return exps[-1]


def point_set(config: str, reference: str) -> str:
    return "reference" if config == reference else "combined" if "combo" in Path(config).stem else "single"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text(encoding="utf-8"))
    kw = {"folds": cfg["folds"], "alpha": float(cfg["ridge_alpha"]), "mi_rows": cfg["mi_rows"]}
    with open(cfg["resolution"], encoding="utf-8") as f:
        resolution = {r["configuration"]: (int(r["factor"]), int(r["n_lags"])) for r in csv.DictReader(f)}
    configs = cfg.get("configurations") or sorted(resolution)
    points = []
    for config in configs:
        exp = Path(cfg["experiment_dirs"][config]) if config in (cfg.get("experiment_dirs") or {}) \
            else latest_experiment(Path(config))
        factor, n_lags = resolution[config]
        per = []
        for plant in sorted((exp / "batch_000").glob("graph_*/realization_0")):
            df, roles = plant_problem(plant)
            per.append(measure(df, TARGET, roles, factor, n_lags, BATCH, **kw))
        points.append({"set": point_set(config, cfg["reference"]), "configuration": config, "plants": len(per),
                       "strength": float(np.median([m["test_strength"] for m in per])),
                       "lookalike_share": float(np.mean([m["lookalike_share"] for m in per]))})
        print(f"{config}: strength {points[-1]['strength']:.1f}, look-alike share "
              f"{points[-1]['lookalike_share']:.3f} ({len(per)} plants)", flush=True)
    for source, c in (cfg.get("public") or {}).items():
        if not c or not all(c.get(k) for k in PUBLIC_REQUIRED[source]):
            continue
        for name, df, target, roles in PUBLIC[source](c):
            ep = "episode" if "episode" in df else "batch"
            factor, n_lags = matched_resolution([q[target].to_numpy(float) for _, q in df.groupby(ep)])
            m = measure(df, target, roles, factor, n_lags, ep, **kw)
            points.append({"set": source, "problem": name, "strength": m["test_strength"],
                           "lookalike_share": m["lookalike_share"]})
            print(f"{name}: strength {m['test_strength']:.1f}, look-alike share {m['lookalike_share']:.3f}", flush=True)
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"points": points}, indent=1), encoding="utf-8")
    print(f"wrote {out}: {len(points)} points")
    if cfg.get("compare_with"):
        paper = {p.get("configuration") or p["problem"]: p
                 for p in json.loads(Path(cfg["compare_with"]).read_text(encoding="utf-8"))["points"]}
        worst = 0.0
        for p in points:
            q = paper.get(p.get("configuration") or p["problem"])
            if q:
                d = max(abs(p["strength"] - q["strength"]) / max(1.0, abs(q["strength"])),
                        abs(p["lookalike_share"] - q["lookalike_share"]))
                worst = max(worst, d)
        print(f"largest difference from the paper's points: {worst:.2e} (strength relative, share absolute)")


PUBLIC_REQUIRED = {"te": ["runs_dir", "graph"], "indpensim": ["data"], "uf": ["csv"]}

if __name__ == "__main__":
    main()
