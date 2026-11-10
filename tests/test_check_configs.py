"""Each check configuration is its published configuration with only the intended change, so a check measures the
published plants and nothing else."""
from pathlib import Path

import yaml

C = Path(__file__).resolve().parents[1] / "configs"
TWINS = {"reference_no_within": "reference.yaml", "confound_0.1_no_within": "shares/confound_0.1.yaml",
         "confound_0.6_no_within": "shares/confound_0.6.yaml",
         "confound_0.8_no_within": "shares_extreme/confound_0.8.yaml", "reference_no_confounder": "reference.yaml"}


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{p}{k}.") if isinstance(v, dict) else {p + k: v})
    return out


def test_check_configs_change_only_the_confounder():
    for name, src in TWINS.items():
        a = _flat(yaml.safe_load((C / src).read_text(encoding="utf-8")))
        b = _flat(yaml.safe_load((C / "checks" / f"{name}.yaml").read_text(encoding="utf-8")))
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        allowed = {"metadata.description", "output.directory", "shadow_red_herrings.temporal_confounder.enabled",
                   "shadow_red_herrings.temporal_confounder.share_reference"}
        if name == "reference_no_confounder":
            allowed.add("vector_scm.shares.confounder")
            assert b["vector_scm.shares.confounder"] == 0.0
        assert diff <= allowed, (name, diff - allowed)
        assert b["shadow_red_herrings.temporal_confounder.enabled"] is False


def test_every_published_confounder_setting_ties_the_within_batch_strength():
    """Every published configuration whose confounder share differs from the reference 0.3 scales the within-batch
    confounder by g(s); the reference-share configurations need no tie (g = 1). The checks switch it off."""
    for p in C.rglob("*.yaml"):
        if p.parent.name == "checks":
            continue
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        share = (d.get("vector_scm") or {}).get("shares", {}).get("confounder", 0.3)
        tie = d.get("shadow_red_herrings", {}).get("temporal_confounder", {}).get("share_reference")
        assert tie == (0.3 if share != 0.3 else None), (p, share, tie)
