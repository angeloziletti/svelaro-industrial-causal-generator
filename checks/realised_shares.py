"""Configured against realised shares: every share of the generator measured on the plants it generated.

    python -m checks.realised_shares --config configs/checks/realised_shares.yaml

Generate the configurations named in the YAML first (python -m advanced_scm.examples.generate_batch_datasets --config
<each>); the check reads each configuration's latest experiment under its output.directory. The setting of each
configuration is read from the configuration itself (vector_scm.shares).

All quantities are out-of-sample R^2 at the resolution the methods see (checks/estimator.py), one number per plant,
then the median and interquartile range over the plants of each setting, with all else at the reference:
  signal        R^2 of the target from its direct causes: their lags, the squares of the lags and the products of
                pairs (so causes with an optimum or a joint effect count too)
  upstream      R^2 of a variable from the variables upstream of it (its time-step links), median over the variables
  downstream    R^2 of a downstream measurement from the target, median over the measurements
  interaction   share of the causes' R^2 carried only by the pair products: (R^2 with - R^2 without) / R^2 with
  confounder    the confounder acts in two stages: once per batch on the profile parameters of a cause and its
                siblings, and within the batch as a slow component Z_k(t) added to their trajectories, whose
                strength is multiplied by g(s) = sqrt(s/(1-s)) / sqrt(r/(1-r)) (r = reference share). Measured per
                cause and sibling as q = 1 - corr^2 between the variable with and without Z_k(t) (the same plant,
                generated twice, configs/checks/*_no_within.yaml), and reported as the odds ratio
                [q_s/(1-q_s)] / [q_r/(1-q_r)] against the designed g^2.
  sibling floor the R^2 a sibling shares with its cause (larger of both directions, median over the pairs) with both
                stages of the confounder removed (configs/checks/reference_no_confounder.yaml), beside the same on
                the reference plants.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from checks.estimator import cv_r2, design, plants, resolution, series_file

SHARE_KEY = {"signal": "defect_timeseries", "upstream": "echo", "downstream": "consequence",
             "interaction": "interaction", "confounder": "confounder"}


class Experiments:
    """Config file -> its generated experiment (latest experiment_* under output.directory, or an override)."""

    def __init__(self, overrides):
        self.overrides = {str(Path(k)): Path(v) for k, v in (overrides or {}).items()}
        self._res = {}

    def config(self, path):
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    def experiment(self, path):
        if str(Path(path)) in self.overrides:
            return self.overrides[str(Path(path))]
        out = Path(self.config(path)["output"]["directory"])
        exps = sorted(out.glob("experiment_*"))
        if not exps:
            raise FileNotFoundError(f"{path}: nothing generated under {out}; generate it first")
        return exps[-1]

    def setting(self, path, axis):
        return float(self.config(path)["vector_scm"]["shares"][SHARE_KEY[axis]])

    def resolution(self, path):
        if str(path) not in self._res:
            self._res[str(path)] = resolution(self.experiment(path))
        r = self._res[str(path)]
        return r["factor"], r["n_lags"]


def labels(plant):
    return json.loads((plant / "datasets/dataset_metadata.json").read_text(encoding="utf-8"))


def series_share(axis, plant, f, L, folds, alpha):
    meta = labels(plant)
    lab = meta["target_driver_labels"]["labels"]
    if axis in ("signal", "interaction"):
        causes = [s for s, v in lab.items() if v["label"] == "direct_parent"]
        X, g = design(plant, causes + ["defect_ts"], f, L)
        Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
        y, n = X[:, -1, 0], len(causes)
        lin = [X[:, i, :] for i in range(n)]
        sq = [Z[:, i, :] ** 2 for i in range(n)]
        prod = [Z[:, i, :] * Z[:, j, :] for i in range(n) for j in range(i + 1, n)]
        full = cv_r2(np.concatenate(lin + sq + prod, axis=1), y, g, folds, alpha)
        if axis == "signal":
            return full
        return 0.0 if not prod else (full - cv_r2(np.concatenate(lin + sq, axis=1), y, g, folds, alpha)) / full
    if axis == "upstream":
        edges = {}
        for e in meta["sensor_graph"].get("dynamic_edges") or []:
            edges.setdefault(e["target"], []).append(e["source"])
        sens = sorted(set(edges) | {s for v in edges.values() for s in v})
        X, g = design(plant, sens + [sens[0]], f, L)
        col = {s: i for i, s in enumerate(sens)}
        return float(np.median([cv_r2(np.concatenate([X[:, col[s], :] for s in src], axis=1), X[:, col[t], 0], g,
                                      folds, alpha) for t, src in edges.items()]))
    if axis == "downstream":
        cons = [c["sensor"] for c in meta["vector_scm_audit"].get("consequences") or []]
        X, g = design(plant, cons + ["defect_ts"], f, L)
        return float(np.median([cv_r2(X[:, -1, :], X[:, i, 0], g, folds, alpha) for i in range(len(cons))]))
    raise ValueError(axis)


def summary(values):
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "n": len(values)}


def within_shares(with_plant, without_plant):
    """q per cause and sibling: 1 - corr^2 between the variable with and without Z_k(t)."""
    a = pd.read_parquet(series_file(with_plant))
    b = pd.read_parquet(series_file(without_plant))
    lab = labels(with_plant)["target_driver_labels"]["labels"]
    group = {s for s, v in lab.items() if v.get("shadow_of")} | {v["shadow_of"] for v in lab.values() if v.get("shadow_of")}
    return {x: 1 - np.corrcoef(a[x], b[x])[0, 1] ** 2 for x in sorted(group)}


def sibling_link(plant, f, L, folds, alpha):
    lab = labels(plant)["target_driver_labels"]["labels"]
    pairs = [(v["shadow_of"], s) for s, v in lab.items() if v.get("shadow_of") and v.get("candidate", True)]
    sens = sorted({x for pr in pairs for x in pr})
    X, g = design(plant, sens + [sens[0]], f, L)
    col = {s: i for i, s in enumerate(sens)}
    return float(np.median([max(cv_r2(X[:, col[c], :], X[:, col[s], 0], g, folds, alpha),
                                cv_r2(X[:, col[s], :], X[:, col[c], 0], g, folds, alpha)) for c, s in pairs]))


def by_graph(experiment):
    return {p.parent.name: p for p in plants(experiment)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text(encoding="utf-8"))
    E = Experiments(cfg.get("experiments"))
    folds, alpha, ref = int(cfg["folds"]), float(cfg["ridge_alpha"]), cfg["reference"]
    out = {"shares": {}, "confounder": [], "sibling_floor": None, "sibling_reference": None}

    for axis, configs in cfg["sweeps"].items():
        rows = []
        for c in configs:
            c = ref if c == "reference" else c
            f, L = E.resolution(c)
            v = [series_share(axis, p, f, L, folds, alpha) for p in plants(E.experiment(c))]
            rows.append({"setting": E.setting(c, axis), "config": c, "factor": f, "n_lags": L, **summary(v)})
            print(f"{axis:12s} setting {rows[-1]['setting']:<5} measured {rows[-1]['median']:.3f}", flush=True)
        out["shares"][axis] = sorted(rows, key=lambda r: r["setting"])

    conf = cfg["confounder"]
    r = E.setting(ref, "confounder")
    odds = lambda q: q / (1 - q)  # noqa: E731
    ref_with, ref_without = by_graph(E.experiment(ref)), by_graph(E.experiment(conf["reference_without"]))
    ref_q = {gname: within_shares(ref_with[gname], ref_without[gname]) for gname in ref_with if gname in ref_without}
    out["confounder"].append({"setting": r, "g": 1.0, "expected": 1.0, "median": 1.0, "q1": 1.0, "q3": 1.0,
                              "q_median": float(np.median([q for d in ref_q.values() for q in d.values()])),
                              "plants": sorted(ref_q)})
    for c in conf["settings"]:
        s = E.setting(c["config"], "confounder")
        w, wo = by_graph(E.experiment(c["config"])), by_graph(E.experiment(c["without"]))
        ratios, qs, used = [], [], []
        for gname, rq in ref_q.items():
            if gname not in w or gname not in wo:
                continue
            q = within_shares(w[gname], wo[gname])
            used.append(gname)
            qs += list(q.values())
            ratios += [odds(q[x]) / odds(rq[x]) for x in rq if x in q and 1e-6 < rq[x] < 1 and q[x] < 1]
        g = float(np.sqrt(odds(s)) / np.sqrt(odds(r)))
        out["confounder"].append({"setting": s, "g": g, "expected": g * g, **summary(ratios),
                                  "q_median": float(np.median(qs)), "plants": used})
        print(f"confounder   setting {s:<5} odds ratio expected {g * g:.3f} measured {out['confounder'][-1]['median']:.3f}"
              f" on {len(used)} plants", flush=True)

    f, L = E.resolution(ref)
    neither = by_graph(E.experiment(cfg["sibling_floor"]))
    floor = [sibling_link(neither[gname], f, L, folds, alpha) for gname in sorted(neither) if gname in ref_with]
    full = [sibling_link(ref_with[gname], f, L, folds, alpha) for gname in sorted(neither) if gname in ref_with]
    out["sibling_floor"], out["sibling_reference"] = summary(floor), summary(full)
    print(f"sibling link without the confounder {out['sibling_floor']['median']:.3f}, "
          f"at the reference {out['sibling_reference']['median']:.3f}", flush=True)

    dest = Path(cfg["output"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written", dest)


if __name__ == "__main__":
    main()
