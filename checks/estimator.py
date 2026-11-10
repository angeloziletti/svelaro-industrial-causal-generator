"""The estimator behind the checks: out-of-sample R^2 at the resolution the methods see.

A plant is split into its batches. Each batch is reduced to block means of `factor` steps; a regression uses the
predictors at lags 0..L and the target's own past at lags 1..L. Every R^2 is out of sample: a ridge fit on
standardised columns, with folds split by batch, so no batch is both fitted and scored.

The resolution follows the rule of the accompanying paper, from the target alone and without ground truth:
  tau        the within-batch, noise-robust decorrelation time of the target (advanced_scm.shares.process_tau)
  factor     the largest power of two <= the smallest tau over the plants of a configuration, clamped to [4, 64]
  n_lags     ceil(2 * largest tau / factor), so every plant is seen over about two of its own decorrelation times
"""
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from advanced_scm import shares

BATCH, TARGET = "context_index", "defect_ts"


def series_file(plant: Path) -> Path:
    """The plant's time series."""
    return Path(plant) / "full_timeseries.parquet"


def plants(experiment: Path) -> List[Path]:
    """The plant folders of one generated experiment, in graph order."""
    return sorted(p for p in (Path(experiment) / "batch_000").glob("graph_*/realization_0") if p.is_dir())


def target_tau(plant: Path) -> int:
    df = pd.read_parquet(series_file(plant), columns=[BATCH, TARGET])
    return shares.process_tau(df[TARGET].to_numpy(dtype=float).reshape(df[BATCH].nunique(), -1))


def resolution(experiment: Path, lo: int = 4, hi: int = 64) -> dict:
    """factor and n_lags of one configuration (all its plants), by the rule in the module doc."""
    taus = [target_tau(p) for p in plants(experiment)]
    factor = int(min(hi, max(lo, 2 ** int(np.floor(np.log2(max(min(taus), 1)))))))
    return {"factor": factor, "n_lags": int(max(1, math.ceil(2 * max(taus) / factor))), "tau_per_plant": taus}


def episode_blocks(df: pd.DataFrame, cols: List[str], episode_col: Optional[str], factor: int,
                   block_rows: Optional[int] = None) -> List[np.ndarray]:
    """One array (steps, cols) per episode, as block means of `factor` steps. Without an episode column the run is cut
    into contiguous episodes of `block_rows` rows."""
    if episode_col:
        parts = [g[cols].to_numpy(float) for _, g in df.groupby(episode_col, sort=True)]
    else:
        x = df[cols].to_numpy(float)
        parts = [x[i:i + block_rows] for i in range(0, len(x) - block_rows + 1, block_rows)]
    out = []
    for p in parts:
        n = len(p) // factor
        if n:
            out.append(p[: n * factor].reshape(n, factor, len(cols)).mean(axis=1))
    return out


def lagged(blocks: List[np.ndarray], n_lags: int, target_idx: int):
    """Design per column (rows, cols, lags), the target, and the episode of each row. Lags 0..L for every column; the
    target column's lag 0 is the value predicted, so callers use only its lags 1..L as predictors."""
    X, y, groups = [], [], []
    for k, b in enumerate(blocks):
        T = len(b)
        if T <= n_lags:
            continue
        rows = np.arange(n_lags, T)
        X.append(np.stack([b[rows - lag] for lag in range(n_lags + 1)], axis=2))
        y.append(b[rows, target_idx])
        groups.append(np.full(len(rows), k))
    return np.concatenate(X), np.concatenate(y), np.concatenate(groups)


def cv_r2(design: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5, alpha: float = 1e-3) -> float:
    """Out-of-sample R^2 of a ridge fit on standardised columns, folds split by episode."""
    if design.shape[1] == 0:
        return 0.0
    ids = np.unique(groups)
    fold_of = {g: i % folds for i, g in enumerate(ids)}
    f = np.array([fold_of[g] for g in groups])
    pred = np.empty_like(y)
    for k in range(folds):
        tr, te = f != k, f == k
        if not te.any():
            continue
        mu, sd = design[tr].mean(0), design[tr].std(0)
        sd[sd == 0] = 1.0
        A = (design[tr] - mu) / sd
        ym = y[tr].mean()
        b = np.linalg.solve(A.T @ A + alpha * len(A) * np.eye(A.shape[1]), A.T @ (y[tr] - ym))
        pred[te] = ym + ((design[te] - mu) / sd) @ b
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def design(plant: Path, cols: List[str], factor: int, n_lags: int):
    """Lagged design of the given columns of a plant at the resolution; the last column is the regression target."""
    df = pd.read_parquet(series_file(plant), columns=list(dict.fromkeys(cols)) + [BATCH])
    X, _, g = lagged(episode_blocks(df, cols, BATCH, factor), n_lags, len(cols) - 1)
    return X, g
