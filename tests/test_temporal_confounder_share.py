"""The confounder share sets the within-batch confounder when share_reference is given; at the reference share,
and without share_reference, the data are bit-identical to before."""
import numpy as np

from advanced_scm.temporal_confounder import apply_temporal_confounders_to_timeseries, share_gain
from advanced_scm.unified_config import TemporalConfounderConfig

GROUPS = {"temp_1": ["temp_2", "temp_3"], "temp_4": ["temp_5"]}


def _series(seed=0):
    rng = np.random.default_rng(seed)
    return {s: rng.normal(size=(20, 300)) for s in ["temp_1", "temp_2", "temp_3", "temp_4", "temp_5"]}


def _shares(s):
    return {f"shadow_confounder_z_{t}->{p}_s0_amplitude": {"share": s}
            for t, ps in GROUPS.items() for p in [t] + ps}


def _apply(cfg, shares):
    ts = _series()
    apply_temporal_confounders_to_timeseries(ts, GROUPS, cfg, seed=11, structure_seed=7, confounder_shares=shares)
    return ts


def test_gain_values():
    cfg = TemporalConfounderConfig(enabled=True, share_reference=0.3)
    assert share_gain(cfg, _shares(0.3), "temp_1") == 1.0
    assert abs(share_gain(cfg, _shares(0.8), "temp_1") - np.sqrt(4.0) / np.sqrt(0.3 / 0.7)) < 1e-12
    assert abs(share_gain(cfg, _shares(0.1), "temp_1") - np.sqrt(1 / 9) / np.sqrt(0.3 / 0.7)) < 1e-12
    assert share_gain(TemporalConfounderConfig(enabled=True), _shares(0.8), "temp_1") == 1.0  # not tied
    assert share_gain(cfg, {}, "temp_1") == 1.0  # no share recorded for this confounder


def test_reference_share_is_bit_identical():
    old = _apply(TemporalConfounderConfig(enabled=True, alpha_truth=[0.3, 1.0], alpha_shadow_range=[0.3, 1.0]),
                 _shares(0.3))
    new = _apply(TemporalConfounderConfig(enabled=True, alpha_truth=[0.3, 1.0], alpha_shadow_range=[0.3, 1.0],
                                          share_reference=0.3), _shares(0.3))
    for s in old:
        assert np.array_equal(old[s], new[s])


def test_share_scales_the_within_batch_confounder():
    base = _series()
    cfg = TemporalConfounderConfig(enabled=True, alpha_truth=[0.3, 1.0], alpha_shadow_range=[0.3, 1.0],
                                   share_reference=0.3)
    ref, strong = _apply(cfg, _shares(0.3)), _apply(cfg, _shares(0.8))
    g = share_gain(cfg, _shares(0.8), "temp_1")
    for s in ["temp_1", "temp_2", "temp_3"]:  # the added confounder is the same path, scaled by g
        assert np.allclose(strong[s] - base[s], g * (ref[s] - base[s]))
