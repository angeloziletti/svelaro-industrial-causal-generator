"""
Shared I/O utilities for the entire project.
"""

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

# Bounded retry for the final rename. See `atomic_path`.
_ATOMIC_REPLACE_RETRIES = 4
_ATOMIC_REPLACE_BACKOFF_SEC = 0.05


# ---------------------------------------------------------------------------
# Atomic writes
#
# Write to a temporary file, then `os.replace`, which is atomic on Windows and POSIX
# for a same-directory rename: an interrupted write never leaves a partial file.
# ---------------------------------------------------------------------------


@contextmanager
def atomic_path(path: Union[str, Path], suffix: str = ".tmp") -> Iterator[Path]:
    """Yield a temp path to write to; rename it onto `path` only on clean exit.

    The temp name carries the writing process's PID, so two processes racing on
    the same target cannot corrupt each other's partial file, and a leftover
    `.tmp` from a dead run is never mistaken for the real artifact.

    On any exception the temp file is removed and the exception propagates --
    the original `path`, if it existed, is left exactly as it was.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}{suffix}")

    try:
        yield tmp
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt mid-write is exactly
        # the case this exists for.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise

    if not tmp.exists():
        raise FileNotFoundError(
            f"atomic_path({path}): nothing was written to {tmp.name}. The body "
            "must write to the yielded path, not to the final one."
        )

    # os.replace, NOT Path.rename: on Windows rename() raises FileExistsError
    # when the destination exists, which is the normal case here.
    #
    # The retry is for synced folders. Several output trees live under a synced folder
    # and the sync agent can hold a transient handle on the destination. Bounded,
    # then raise -- a write that cannot land must not be reported as landed.
    for attempt in range(_ATOMIC_REPLACE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _ATOMIC_REPLACE_RETRIES - 1:
                raise
            time.sleep(_ATOMIC_REPLACE_BACKOFF_SEC * (2**attempt))


# ---------------------------------------------------------------------------
# The combined time series of a plant
# ---------------------------------------------------------------------------

TIMESERIES_STEM = "full_timeseries"


def timeseries_write_path(directory: Union[str, Path]) -> Path:
    """Where a new combined timeseries goes."""
    return Path(directory) / f"{TIMESERIES_STEM}.parquet"


def find_timeseries(directory: Union[str, Path]) -> Optional[Path]:
    """Locate an existing combined timeseries; None when there is none."""
    candidate = timeseries_write_path(directory)
    return candidate if candidate.exists() else None


def write_timeseries(df: Any, path: Union[str, Path], float32: bool = False) -> Path:
    """Write the combined timeseries atomically as parquet. float32=True stores
    float64 columns in single precision."""
    path = Path(path)
    if float32:
        cols = [c for c in df.columns if str(df[c].dtype) == "float64"]
        if cols:
            df = df.astype({c: "float32" for c in cols})
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported timeseries container {path.name!r}: expected .parquet")
    with atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False)
    return path
