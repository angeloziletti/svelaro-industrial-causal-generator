"""Reproducible Gaussian sampling for the pattern generator.

np.random.multivariate_normal factors the covariance with an SVD. The kernel
covariances used here have many (near-)equal eigenvalues, so the eigenvectors
are not unique: the factor -- and with it the sample drawn from the same
standard normals -- depends on the BLAS build and thread count. Sensors
generated in parallel workers (fewer BLAS threads) came out different from the
same sensors generated sequentially.

The Cholesky factor is unique, so `mvn_sample` gives the same series for the
same seed everywhere. Factors are cached by covariance content: a pattern's
kernel matrix is identical for every sample, so the O(T^3) factorisation runs
once per pattern instead of once per (sample, sensor).
"""
import hashlib
from typing import Dict

import numpy as np

_CACHE: Dict[bytes, np.ndarray] = {}
_CACHE_MAX = 256


def _cholesky(cov: np.ndarray) -> np.ndarray:
    key = hashlib.sha1(np.ascontiguousarray(cov).tobytes()).digest() + bytes(
        str(cov.shape), "ascii"
    )
    L = _CACHE.get(key)
    if L is None:
        jitter = 0.0
        scale = float(np.mean(np.diag(cov))) or 1.0
        while True:
            try:
                L = np.linalg.cholesky(cov + jitter * np.eye(len(cov)))
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 * scale if jitter == 0.0 else jitter * 10
                if jitter > 1e-2 * scale:
                    raise
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = L
    return L


