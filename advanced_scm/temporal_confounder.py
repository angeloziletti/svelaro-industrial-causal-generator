"""
Temporal confounder Z_k(t) generation.

A scalar Z_k per truth sensor that modulates GP parameters
of both truth and shadow red herrings, producing SAMPLE-LEVEL confounding.
Here Z_k becomes a smooth TIME SERIES Z_k(t) generated per
(truth sensor, sample) — added as an additive contribution to BOTH truth's
and shadow's pattern-injected time series. This propagates Z_k's signal
through the within-sample temporal structure, so long-format Pearson
between shadow_ts and defect_ts captures shared upstream temporal cause.

The vector mechanism (which only operates on truth time series) receives
truth's full signal (pattern + Z_k(t) contribution), so defect_ts(t)
inherits the Z_k(t) structure. Shadow_ts(t) shares Z_k(t) directly. Both
end up correlated within sample via the shared Z_k(t).

Implementation: smooth Z_k(t) is generated as Gaussian-filtered white
noise — fast, no scipy.gp dependency, gives RBF-kernel-like smoothness
with controllable correlation length.
"""

from typing import Optional

import numpy as np


def sample_temporal_confounder(
    n_samples: int,
    sequence_length: int,
    correlation_length: int,
    rng: np.random.Generator,
    normalise: bool = True,
) -> np.ndarray:
    """Generate Z_k(t) shape (n_samples, sequence_length).

    Low-pass-filtered Gaussian noise approximating a Gaussian-process
    sample with RBF kernel of the given correlation length. Each row is
    an independent realisation (one per sample); columns are timesteps.

    Args:
        n_samples: number of independent realisations to generate (one
            per synthetic sample).
        sequence_length: length of each realisation in timesteps.
        correlation_length: Gaussian filter sigma in timesteps; controls
            smoothness. Smaller → choppier; larger → smoother. Must be
            ≥ vector mechanism `lag_max` for Z_k(t) correlations to survive
            the lag.
        rng: numpy RNG. Caller is responsible for seeding deterministically.
        normalise: if True, z-score each row to unit variance/zero mean.
            Makes downstream alpha coefficients interpretable as
            variance-fraction contributions.

    Returns:
        Array shape (n_samples, sequence_length). When `normalise=True`,
        each row has mean ≈ 0 and std ≈ 1.

    Notes:
        - Uses `scipy.ndimage.gaussian_filter1d` with mode="wrap" to
          avoid edge artifacts; sequence is effectively circular which
          is fine for synthetic patterns.
        - Each row is INDEPENDENTLY sampled — there's no across-sample
          structure here; that's intentional. Z_k(t) varies per sample
          (different "ambient temperature trajectory" each batch) but
          its within-sample shape is what carries the confounding signal.
    """
    if correlation_length < 1:
        raise ValueError(
            f"correlation_length must be >= 1, got {correlation_length}"
        )
    if sequence_length < 2:
        raise ValueError(
            f"sequence_length must be >= 2, got {sequence_length}"
        )

    # Lazy import keeps test-time overhead low for callers that don't use
    # temporal confounders (scipy is a heavy dep at import time).
    from scipy.ndimage import gaussian_filter1d

    raw = rng.standard_normal((int(n_samples), int(sequence_length)))
    smoothed = gaussian_filter1d(
        raw, sigma=float(correlation_length), axis=1, mode="wrap"
    )

    if normalise:
        # Per-sample z-score so α coefficients are variance-interpretable
        # downstream. tiny epsilon guards against degenerate rows.
        mean = smoothed.mean(axis=1, keepdims=True)
        std = smoothed.std(axis=1, keepdims=True)
        smoothed = (smoothed - mean) / (std + 1e-9)

    return smoothed


def share_gain(cfg, confounder_shares: Optional[dict], truth_sensor: str) -> float:
    """g(s) for T_k's within-batch confounder (see apply_temporal_confounders_to_timeseries); 1.0 when the
    within-batch route is not tied to the share or T_k's confounder has no share."""
    r = getattr(cfg, "share_reference", None)
    if r is None:
        return 1.0
    z = f"shadow_confounder_z_{truth_sensor}->"
    s = [v["share"] for k, v in (confounder_shares or {}).items() if k.startswith(z)]
    if not s:
        return 1.0
    s, r = float(np.mean(s)), float(r)
    if not (0.0 < s < 1.0 and 0.0 < r < 1.0):
        raise ValueError(f"confounder share {s} and share_reference {r} must lie in (0, 1)")
    return float(np.sqrt(s / (1.0 - s)) / np.sqrt(r / (1.0 - r)))


def apply_temporal_confounders_to_timeseries(
    structured_time_series: dict,
    shadow_groups: dict,
    cfg,
    seed: int,
    logger=None,
    structure_seed: Optional[int] = None,
    confounder_shares: Optional[dict] = None,
) -> dict:
    """Apply Z_{k,s}(t) additively to truth + shadow time series in place.

    With `cfg.share_reference` set, the confounder share also sets the within-batch route: every strength of
    T_k's group (alpha_truth and each alpha_shadow, drawn as before) is multiplied by
        g(s) = sqrt(s / (1 - s)) / sqrt(r / (1 - r)),
    the ratio of the confounder's weight to the rest's in the share mix (shares.mix), relative to the reference
    share r. s is the mean share of Z_k's parameter edges (`confounder_shares`, from
    shares.calibrate_confounder_shares). At s = r the factor is exactly 1 and the data are unchanged; the draws
    are made before scaling, so they never change.

    For each truth sensor T_k with non-empty shadow group:
      1. Sample Z_{k,s}(t) shape (n_samples, T) using a seed derived from
         (seed + cfg.seed_offset + hash(T_k)).
      2. Add `cfg.alpha_truth * Z_k(t)` to `structured_time_series[T_k]`.
      3. For each shadow R in T_k's group: sample alpha_shadow
         ~ uniform(*cfg.alpha_shadow_range), add to
         `structured_time_series[R]`.

    Args:
        structured_time_series: dict mapping sensor_name → array of shape
            (n_samples, sequence_length). Modified IN PLACE.
        shadow_groups: dict mapping truth_sensor_name → list of shadow
            sensor names. From `shadow_red_herring_audit["shadow_groups"]`.
        cfg: TemporalConfounderConfig instance.
        seed: data seed: drives the Z_k(t) paths. Combined with cfg.seed_offset for
            reproducibility independent of other RNG streams.
        structure_seed: drives the shadow strengths alpha_shadow (a mechanism
            parameter, shared by realizations of one graph). None = `seed`.
        logger: optional logger.

    Returns:
        Audit dict with per-truth Z_k seed used + per-shadow alpha values.
    """
    audit = {
        "enabled": True,
        "correlation_length": int(cfg.correlation_length),
        "alpha_truth": cfg.alpha_truth,
        "alpha_shadow_range": list(cfg.alpha_shadow_range),
        "normalise_to_unit_variance": bool(cfg.normalise_to_unit_variance),
        "seed_offset": int(cfg.seed_offset),
        "rng_seed_base": int(seed) + int(cfg.seed_offset),
        "per_truth": {},          # truth_name → {z_k_seed, n_shadows}
        "per_shadow": {},         # shadow_name → {shadow_of, alpha_shadow}
    }

    # Determine n_samples / sequence_length from any sensor's array.
    if not shadow_groups:
        return audit
    sample_truth = next(iter(shadow_groups))
    if sample_truth not in structured_time_series:
        if logger is not None:
            logger.warning(
                f"   ⚠️ Temporal confounder: truth sensor '{sample_truth}' "
                f"missing from structured_time_series; skipping."
            )
        return audit
    n_samples, sequence_length = structured_time_series[sample_truth].shape

    n_truth_applied = 0
    n_shadow_applied = 0
    for truth_sensor, shadows in shadow_groups.items():
        if truth_sensor not in structured_time_series:
            if logger is not None:
                logger.warning(
                    f"   ⚠️ Temporal confounder: truth sensor "
                    f"'{truth_sensor}' missing; skipping its group."
                )
            continue
        if not shadows:
            # No shadows for this truth — Z_k(t) wouldn't have anywhere
            # to act, so skip. (Truth would still get its own Z_k(t) but
            # the confounding test requires at least one shadow to share.)
            continue

        # Per-truth deterministic seed. hash() is salted in modern Python
        # so we use a stable hash on the truth name's chars.
        truth_hash = sum(ord(c) for c in truth_sensor)
        z_seed = int(seed) + int(cfg.seed_offset) + truth_hash
        rng = np.random.default_rng(z_seed)
        alpha_rng = np.random.default_rng(
            int(seed if structure_seed is None else structure_seed)
            + int(cfg.seed_offset)
            + truth_hash
            + 1
        )

        Z_k = sample_temporal_confounder(
            n_samples=n_samples,
            sequence_length=sequence_length,
            correlation_length=int(cfg.correlation_length),
            rng=rng,
            normalise=bool(cfg.normalise_to_unit_variance),
        )

        # Apply to truth_ts. A [lo, hi] range draws alpha_truth like the shadow
        # strengths: a fixed, larger alpha_truth gave the truth sensors visibly
        # more variance than their shadows, betraying which sensors are causes.
        if isinstance(cfg.alpha_truth, (list, tuple)):
            alpha_t = float(alpha_rng.uniform(*map(float, cfg.alpha_truth)))
        else:
            alpha_t = float(cfg.alpha_truth)
        gain = share_gain(cfg, confounder_shares, truth_sensor)
        alpha_t = alpha_t * gain
        structured_time_series[truth_sensor] = (
            structured_time_series[truth_sensor] + alpha_t * Z_k
        )
        n_truth_applied += 1

        # Apply to each shadow with its own α_shadow.
        a_lo, a_hi = (
            float(cfg.alpha_shadow_range[0]),
            float(cfg.alpha_shadow_range[1]),
        )
        per_shadow = {}
        for r in shadows:
            if r not in structured_time_series:
                if logger is not None:
                    logger.warning(
                        f"   ⚠️ Temporal confounder: shadow '{r}' missing "
                        f"from structured_time_series; skipping."
                    )
                continue
            alpha_s = float(alpha_rng.uniform(a_lo, a_hi)) * gain
            structured_time_series[r] = (
                structured_time_series[r] + alpha_s * Z_k
            )
            per_shadow[r] = round(alpha_s, 4)
            audit["per_shadow"][r] = {
                "shadow_of": truth_sensor,
                "alpha_shadow": round(alpha_s, 4),
            }
            n_shadow_applied += 1

        audit["per_truth"][truth_sensor] = {
            "z_k_seed": z_seed,
            "n_shadows": len(per_shadow),
            "alpha_truth": alpha_t,
        }
        if getattr(cfg, "share_reference", None) is not None:
            audit["per_truth"][truth_sensor]["share_gain"] = gain

    if logger is not None:
        logger.info(
            f"   🌊 Temporal confounder Z_k(t) applied to "
            f"{n_truth_applied} truth + {n_shadow_applied} shadow sensors "
            f"(corr_length={cfg.correlation_length}, α_truth={cfg.alpha_truth}, "
            f"α_shadow_range={cfg.alpha_shadow_range})"
        )

    return audit
