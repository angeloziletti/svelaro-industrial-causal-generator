"""The estimator of the checks recovers known shares on small simulated problems."""
import numpy as np
import pandas as pd

from checks.estimator import cv_r2, episode_blocks, lagged


def _problem(seed=0, n_ep=40, T=200, strong=1.0, weak=0.3):
    """y is driven one step later by a (strong) and b (weak); noise is unrelated."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_ep):
        a, b, n = rng.normal(size=T), rng.normal(size=T), rng.normal(size=T)
        y = np.zeros(T)
        y[1:] = strong * a[:-1] + weak * b[:-1] + 0.5 * rng.normal(size=T - 1)
        rows.append(pd.DataFrame({"ep": e, "a": a, "b": b, "noise": n, "y": y}))
    return pd.concat(rows, ignore_index=True)


def _r2(df, cols, factor=1, n_lags=2):
    X, y, g = lagged(episode_blocks(df, cols + ["y"], "ep", factor), n_lags, len(cols))
    return cv_r2(np.concatenate([X[:, i, :] for i in range(len(cols))], axis=1), y, g)


def test_recovers_the_explained_share():
    df = _problem()
    assert abs(_r2(df, ["a", "b"]) - 1.09 / 1.34) < 0.03  # (1 + 0.09) / (1 + 0.09 + 0.25)
    assert abs(_r2(df, ["b"]) - 0.09 / 1.34) < 0.03


def test_out_of_sample_r2_is_near_zero_without_signal():
    assert abs(_r2(_problem(strong=0.0, weak=0.0), ["a", "b", "noise"], n_lags=3)) < 0.02
    assert abs(_r2(_problem(), ["noise"])) < 0.01


def test_block_means_and_continuous_runs():
    df = _problem()
    blocks = episode_blocks(df.drop(columns="ep"), ["a", "y"], None, factor=2, block_rows=400)
    assert len(blocks) == len(df) // 400 and all(b.shape == (200, 2) for b in blocks)
    by_ep = episode_blocks(df, ["a", "y"], "ep", factor=2)
    first = df[df.ep == 0][["a", "y"]].to_numpy()
    assert np.allclose(by_ep[0], first.reshape(100, 2, 2).mean(axis=1))
