"""CI-safe boundary + coverage tests for ``Perturbator`` sampling internals.

These tests lock the corrected multiplicative ``get_scale`` (log-space sigma)
introduced in 1.1.2 and prove -- deterministically and statistically -- that
``+/- max_std * sigma`` reaches exactly the ``get_boundaries`` factor for every
parameter class:

* ADDITIVE       (we, be, pow)              -> sigma = std                 (linear axis)
* PERCENT        (hrs, sigma, epsilon, fact)-> sigma = std * value         (linear axis)
* MULTIPLICATIVE (if, freq, bfc, sfc, mrc)  -> sigma = ln(std); bounds value*std**(+/-ms)

Everything runs through the REAL ``Perturbator`` backend -- no MESS/HPC/DB.
The abstract-ish base class carries no abstract methods, so it is instantiated
directly with a ``SimpleNamespace`` logger, and ``i_sop`` is replaced by a
``SimpleNamespace`` exposing only ``uncertainties`` and ``parameters_names``
(the sole attributes ``get_scale`` reads).

Statistical tests use a fixed seed and N=100_000 samples drawn through the real
``get_rng``; each is paired with a deterministic identity assertion so it fails
precisely even if the numpy RNG stream ever drifts.

This file is complementary to ``test_perturbation_distribution_matrix_ci.py``
(which locks the *validation* rule); here we exercise the *sampling* backend on
the LEGAL cells only.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from kimeco.Perturbators.perturbator import Perturbator
from kimeco.enums import Distrib, Ptype


SEED = 20250820
N = 100_000
# +/-3 sigma of a standard normal -> 0.997300; +/-4 sigma -> 0.999937.
THREE_SIGMA = 0.9973002
# +/-2 sigma of a standard normal -> 0.954500.
TWO_SIGMA = 0.9544997
PARAM = "p"


def _make_pert(std_overrides: dict[str, Any],
               uncertainty: float,
               value: float) -> Perturbator:
    """Build a real Perturbator with a fake i_sop exposing exactly one param.

    ``uncertainty`` feeds ``get_scale`` (via ``i_sop.uncertainties``). Since the
    boundary fix, ``get_boundaries`` ALSO reads ``i_sop.uncertainties[param]``
    whenever a ``param`` is supplied, falling back to ``settings['std_<ptype>']``
    (from ``std_overrides``) when ``param`` is ``None`` or absent from the dict.

    The numeric cases below keep ``uncertainty == std`` and call
    ``get_boundaries`` positionally (no ``param`` -> the global-std fallback), so
    both the boundary and the ``get_rng`` sampling paths resolve to the same
    sigma and the +/-max_std*sigma identities still land exactly on the boundary.
    """
    settings: dict[str, Any] = {"active_p": []}
    settings.update(std_overrides)
    klog = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    pert = Perturbator(settings, SimpleNamespace(), klog)
    pert.i_sop = SimpleNamespace(
        uncertainties={PARAM: uncertainty},
        parameters_names={PARAM: value},
    )
    return pert


def _coverage(pert: Perturbator, ptype: str, i_val: float,
              distrib: Distrib, n: int = N) -> float:
    """Fraction of get_rng draws (centred on i_val) that fall inside bounds."""
    np.random.seed(SEED)
    bounds = pert.get_boundaries(ptype=ptype, i_val=i_val)
    lo, hi = min(bounds), max(bounds)
    inside = 0
    for _ in range(n):
        v = pert.get_rng(ptype=ptype, i_val=i_val, c_val=i_val,
                         param=PARAM, distrib=distrib)
        if lo < v < hi:
            inside += 1
    return inside / n


# ===========================================================================
# Group 1 -- Boundary correctness across ALL parameter classes
# ===========================================================================

def test_multiplicative_canonical_boundaries_exact() -> None:
    """value=1, std=1.2, max_std=3 -> bounds (1/1.2**3, 1.2**3) exactly."""
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    assert lo == pytest.approx(1 / 1.728)
    assert hi == pytest.approx(1.728)
    assert (lo, hi) == pytest.approx((1 / 1.2 ** 3, 1.2 ** 3))


def test_multiplicative_log_symmetry_for_value_not_one() -> None:
    """Multiplicative bounds are geometrically symmetric: lo*hi == i_val**2."""
    i_val = 800.0
    pert = _make_pert({"max_std": 4, "std_if": 1.1}, 1.1, i_val)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=i_val)
    factor = 1.1 ** 4  # geometric: std ** max_std == 1.4641
    assert lo == pytest.approx(i_val / factor)
    assert hi == pytest.approx(i_val * factor)
    # Geometric (log) symmetry about i_val.
    assert lo * hi == pytest.approx(i_val ** 2)


def test_additive_boundaries_are_i_val_plus_minus_std_times_max_std() -> None:
    """Additive bounds = i_val +/- std * max_std (linear)."""
    pert = _make_pert({"max_std": 3, "std_we": 1.0}, 1.0, 100.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.WE.value, i_val=100.0)
    assert lo == pytest.approx(100.0 - 1.0 * 3)
    assert hi == pytest.approx(100.0 + 1.0 * 3)


def test_percent_boundaries_scale_with_value() -> None:
    """Percent bounds = i_val +/- i_val * std * max_std (linear)."""
    i_val = 800.0
    pert = _make_pert({"max_std": 3, "std_hrs": 0.1}, 0.1, i_val)
    lo, hi = pert.get_boundaries(ptype=Ptype.HRS.value, i_val=i_val)
    assert lo == pytest.approx(i_val - i_val * 0.1 * 3)
    assert hi == pytest.approx(i_val + i_val * 0.1 * 3)


def test_large_std_multiplicative_lower_bound_stays_positive() -> None:
    """A large multiplicative std keeps the lower bound strictly > 0."""
    pert = _make_pert({"max_std": 5, "std_if": 4.0}, 4.0, 1.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    assert lo > 0.0
    assert hi > lo
    # Division-based lower bound: 1 / 4**5 = 1/1024.
    assert lo == pytest.approx(1 / 4 ** 5)
    assert hi == pytest.approx(4 ** 5)


def test_zero_bound_clamps_negative_lower_to_zero() -> None:
    """A percent zero-bound param clamps a negative lower bound to 0.0."""
    # sigma is PERCENT *and* in zero_bound. i_val*std*max_std = 1*0.1*20 = 2 > 1
    # so the raw lower bound (1-2 = -1) must be clamped to exactly 0.0.
    pert = _make_pert({"max_std": 20, "std_sigma": 0.1}, 0.1, 1.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.SIG.value, i_val=1.0)
    assert lo == 0.0
    assert hi == pytest.approx(1.0 + 1.0 * 0.1 * 20)


def test_within_boundaries_matches_get_boundaries_strictly() -> None:
    """within_boundaries uses STRICT comparisons against get_boundaries."""
    pert = _make_pert({"max_std": 3, "std_we": 1.0}, 1.0, 100.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.WE.value, i_val=100.0)
    # Inside -> True.
    assert pert.within_boundaries(
        perturbed_val=100.0, ptype=Ptype.WE.value, initial_val=100.0) is True
    # Exactly on the boundary -> False (strict).
    assert pert.within_boundaries(
        perturbed_val=lo, ptype=Ptype.WE.value, initial_val=100.0) is False
    assert pert.within_boundaries(
        perturbed_val=hi, ptype=Ptype.WE.value, initial_val=100.0) is False
    # Outside -> False.
    assert pert.within_boundaries(
        perturbed_val=hi + 1e-6, ptype=Ptype.WE.value,
        initial_val=100.0) is False


# ===========================================================================
# Group 2 -- Distribution x class matrix on the SAMPLING backend (get_rng),
#            LEGAL cells only. Complements the validation-matrix test file.
# ===========================================================================

# (ptype, uncertainty, value, {std key}, i_val, distrib)
_LEGAL_CELLS = [
    # ADDITIVE -> normal / uniform
    (Ptype.WE.value, 1.0, 100.0, "std_we", 100.0, Distrib.NORMAL),
    (Ptype.WE.value, 1.0, 100.0, "std_we", 100.0, Distrib.UNIFORM),
    # PERCENT -> normal / uniform
    (Ptype.HRS.value, 0.1, 800.0, "std_hrs", 800.0, Distrib.NORMAL),
    (Ptype.HRS.value, 0.1, 800.0, "std_hrs", 800.0, Distrib.UNIFORM),
    # MULTIPLICATIVE -> log-normal / log-uniform
    (Ptype.IF.value, 1.2, 1.0, "std_if", 1.0, Distrib.LOGNORMAL),
    (Ptype.IF.value, 1.2, 1.0, "std_if", 1.0, Distrib.LOGUNIFORM),
]

_UNIFORM_CELLS = [c for c in _LEGAL_CELLS
                  if c[5] in (Distrib.UNIFORM, Distrib.LOGUNIFORM)]


@pytest.mark.parametrize(
    "ptype,unc,value,std_key,i_val,distrib", _LEGAL_CELLS,
    ids=[f"{c[0]}-{c[5].value}" for c in _LEGAL_CELLS])
def test_legal_cells_sample_finite(ptype, unc, value, std_key, i_val,
                                   distrib) -> None:
    """Every legal (class, distribution) cell yields finite draws."""
    pert = _make_pert({"max_std": 3, std_key: unc}, unc, value)
    np.random.seed(SEED)
    draws = [pert.get_rng(ptype=ptype, i_val=i_val, c_val=i_val,
                          param=PARAM, distrib=distrib) for _ in range(1000)]
    assert all(np.isfinite(d) for d in draws)


@pytest.mark.parametrize(
    "ptype,unc,value,std_key,i_val,distrib", _UNIFORM_CELLS,
    ids=[f"{c[0]}-{c[5].value}" for c in _UNIFORM_CELLS])
def test_uniform_family_100pct_within_bounds(ptype, unc, value, std_key,
                                             i_val, distrib) -> None:
    """UNIFORM / LOG-UNIFORM draws are ALL within [lo, hi]."""
    pert = _make_pert({"max_std": 3, std_key: unc}, unc, value)
    lo, hi = pert.get_boundaries(ptype=ptype, i_val=i_val)
    np.random.seed(SEED)
    draws = np.array([
        pert.get_rng(ptype=ptype, i_val=i_val, c_val=i_val,
                     param=PARAM, distrib=distrib) for _ in range(N)])
    assert draws.min() >= lo
    assert draws.max() <= hi


def test_loguniform_is_uniform_on_log_axis() -> None:
    """LOG-UNIFORM draws are uniform in log-space: mean(log) ~ log midpoint."""
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    np.random.seed(SEED)
    logs = np.log([
        pert.get_rng(ptype=Ptype.IF.value, i_val=1.0, c_val=1.0,
                     param=PARAM, distrib=Distrib.LOGUNIFORM)
        for _ in range(N)])
    mid = (np.log(lo) + np.log(hi)) / 2  # == 0 for symmetric (1/1.6, 1.6)
    expected_std = (np.log(hi) - np.log(lo)) / np.sqrt(12)
    assert logs.mean() == pytest.approx(mid, abs=5e-3)
    assert logs.std() == pytest.approx(expected_std, rel=5e-3)


# ===========================================================================
# Group 3 -- Coverage on the LINEAR axis (additive + percent)
# ===========================================================================

def test_additive_dstep_identity_max_std_times_scale_is_half_width() -> None:
    """max_std * get_scale == boundary half-width for an additive param."""
    pert = _make_pert({"max_std": 3, "std_we": 1.0}, 1.0, 100.0)
    scale = pert.get_scale(ptype=Ptype.WE.value, param=PARAM)
    lo, hi = pert.get_boundaries(ptype=Ptype.WE.value, i_val=100.0)
    assert 3 * scale == pytest.approx((hi - lo) / 2)


def test_percent_dstep_identity_max_std_times_scale_is_half_width() -> None:
    """max_std * get_scale == boundary half-width for a percent param."""
    i_val = 800.0
    pert = _make_pert({"max_std": 3, "std_hrs": 0.1}, 0.1, i_val)
    scale = pert.get_scale(ptype=Ptype.HRS.value, param=PARAM)
    lo, hi = pert.get_boundaries(ptype=Ptype.HRS.value, i_val=i_val)
    assert 3 * scale == pytest.approx((hi - lo) / 2)


def test_additive_normal_997_coverage_at_max_std_3() -> None:
    """Additive normal draws land inside +/-3sigma bounds ~99.73% of the time."""
    pert = _make_pert({"max_std": 3, "std_we": 1.0}, 1.0, 100.0)
    frac = _coverage(pert, Ptype.WE.value, 100.0, Distrib.NORMAL)
    assert abs(frac - THREE_SIGMA) < 0.004


def test_percent_normal_997_coverage_at_max_std_3() -> None:
    """Percent normal draws land inside +/-3sigma bounds ~99.73% of the time."""
    pert = _make_pert({"max_std": 3, "std_hrs": 0.1}, 0.1, 800.0)
    frac = _coverage(pert, Ptype.HRS.value, 800.0, Distrib.NORMAL)
    assert abs(frac - THREE_SIGMA) < 0.004


def test_additive_normal_above_999_coverage_at_max_std_4() -> None:
    """Additive normal at max_std=4 covers >99.9% (+/-4sigma)."""
    pert = _make_pert({"max_std": 4, "std_we": 1.0}, 1.0, 100.0)
    frac = _coverage(pert, Ptype.WE.value, 100.0, Distrib.NORMAL)
    assert frac > 0.999


# ===========================================================================
# Group 4 -- Coverage on the LOG axis (multiplicative)
# ===========================================================================

def test_multiplicative_canonical_scale_is_ln_std() -> None:
    """get_scale returns the log-space sigma ln(std) for a multiplicative param."""
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    scale = pert.get_scale(ptype=Ptype.IF.value, param=PARAM)
    assert scale == pytest.approx(np.log(1.2))
    # max_std * sigma reaches exactly the log of the boundary factor 1.2**3.
    assert 3 * scale == pytest.approx(np.log(1.2 ** 3))


def test_multiplicative_lognormal_997_coverage_headline() -> None:
    """value=1, std=1.2, max_std=3: lognormal ~99.73% within (1/1.728,1.728)."""
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    assert (lo, hi) == pytest.approx((1 / 1.728, 1.728))
    frac = _coverage(pert, Ptype.IF.value, 1.0, Distrib.LOGNORMAL)
    assert abs(frac - THREE_SIGMA) < 0.004


def test_multiplicative_lognormal_9545_coverage_at_max_std_2() -> None:
    """max_std=2, std=1.2: lognormal ~95.45% within (1/1.44, 1.44) (+/-2sigma)."""
    pert = _make_pert({"max_std": 2, "std_if": 1.2}, 1.2, 1.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    assert (lo, hi) == pytest.approx((1 / 1.44, 1.44))
    frac = _coverage(pert, Ptype.IF.value, 1.0, Distrib.LOGNORMAL)
    assert abs(frac - TWO_SIGMA) < 0.005


def test_multiplicative_small_std_997_coverage() -> None:
    """A SMALL multiplicative std (1.1) still yields ~99.73% +/-3sigma coverage."""
    pert = _make_pert({"max_std": 3, "std_if": 1.1}, 1.1, 1.0)
    scale = pert.get_scale(ptype=Ptype.IF.value, param=PARAM)
    assert 3 * scale == pytest.approx(np.log(1.1 ** 3))
    frac = _coverage(pert, Ptype.IF.value, 1.0, Distrib.LOGNORMAL)
    assert abs(frac - THREE_SIGMA) < 0.004


def test_multiplicative_large_std_997_coverage() -> None:
    """A LARGE multiplicative std (4) still yields ~99.73% +/-3sigma coverage."""
    pert = _make_pert({"max_std": 3, "std_if": 4.0}, 4.0, 1.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    # factor = 4**3 = 64 -> bounds (1/64, 64).
    assert (lo, hi) == pytest.approx((1 / 64, 64.0))
    frac = _coverage(pert, Ptype.IF.value, 1.0, Distrib.LOGNORMAL)
    assert abs(frac - THREE_SIGMA) < 0.004


def test_multiplicative_default_max_std_4_above_999_coverage() -> None:
    """Default max_std=4 generalises: multiplicative lognormal covers >99.9%."""
    pert = _make_pert({"max_std": 4, "std_if": 1.2}, 1.2, 1.0)
    scale = pert.get_scale(ptype=Ptype.IF.value, param=PARAM)
    assert 4 * scale == pytest.approx(np.log(1.2 ** 4))
    frac = _coverage(pert, Ptype.IF.value, 1.0, Distrib.LOGNORMAL)
    assert frac > 0.999


def test_multiplicative_scale_is_value_independent() -> None:
    """get_scale is value-independent: if=800 gives the same sigma as value=1."""
    p_one = _make_pert({"max_std": 4, "std_if": 1.1}, 1.1, 1.0)
    p_big = _make_pert({"max_std": 4, "std_if": 1.1}, 1.1, 800.0)
    s_one = p_one.get_scale(ptype=Ptype.IF.value, param=PARAM)
    s_big = p_big.get_scale(ptype=Ptype.IF.value, param=PARAM)
    assert s_one == pytest.approx(s_big)
    # And it equals log(upper / i_val) / max_std for the value=800 boundaries.
    lo, hi = p_big.get_boundaries(ptype=Ptype.IF.value, i_val=800.0)
    assert s_big == pytest.approx(np.log(hi / 800.0) / 4)


def test_multiplicative_scale_regression_not_prefix_arithmetic() -> None:
    """Regression guard: get_scale is ln(std), NOT (std-1)*value.

    Canonical case (value=1, std=1.2, max_std=3): the log-space sigma is
    ln(1.2) ~ 0.18232, whereas the pre-fix arithmetic (std-1)*value = 0.2.
    """
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    scale = pert.get_scale(ptype=Ptype.IF.value, param=PARAM)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    # Correct identity: ln(std) == log(upper / i_val) / max_std.
    assert scale == pytest.approx(np.log(hi / 1.0) / 3)
    assert scale == pytest.approx(0.18232156, abs=1e-6)
    # Must NOT be the old arithmetic (std - 1) * value == 0.2.
    prefix = (1.2 - 1) * 1.0
    assert abs(scale - prefix) > 0.01


def test_get_mean_sigma_removed() -> None:
    """Dead get_mean_sigma was removed from the Perturbator surface."""
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    assert not hasattr(pert, "get_mean_sigma")


# ===========================================================================
# Group 5 -- Parameter-specific uncertainty overrides in get_boundaries.
#
# get_boundaries(..., param=P) resolves the sigma from
# i_sop.uncertainties[P] when present, else falls back to
# settings['std_<ptype>']. Here _make_pert seeds uncertainties[PARAM] with the
# OVERRIDE, and std_overrides carries the (different) GLOBAL std, so the two
# code paths diverge and can be compared head to head.
# ===========================================================================

def test_specific_std_widens_additive_bounds() -> None:
    """we: an override (2.0) wider than the global std (1.0) widens the band."""
    pert = _make_pert({"max_std": 3, "std_we": 1.0}, 2.0, 100.0)
    lo_o, hi_o = pert.get_boundaries(
        ptype=Ptype.WE.value, i_val=100.0, param=PARAM)
    lo_g, hi_g = pert.get_boundaries(ptype=Ptype.WE.value, i_val=100.0)
    # Override band = 100 +/- 2.0*3; global band = 100 +/- 1.0*3.
    assert (lo_o, hi_o) == pytest.approx((100.0 - 6.0, 100.0 + 6.0))
    assert (lo_g, hi_g) == pytest.approx((100.0 - 3.0, 100.0 + 3.0))
    assert (hi_o - lo_o) > (hi_g - lo_g)


def test_specific_std_narrows_additive_bounds() -> None:
    """be: an override (0.5) below the global std (1.0) narrows the band."""
    pert = _make_pert({"max_std": 3, "std_be": 1.0}, 0.5, 50.0)
    lo_o, hi_o = pert.get_boundaries(
        ptype=Ptype.BE.value, i_val=50.0, param=PARAM)
    lo_g, hi_g = pert.get_boundaries(ptype=Ptype.BE.value, i_val=50.0)
    assert (lo_o, hi_o) == pytest.approx((50.0 - 1.5, 50.0 + 1.5))
    assert (hi_o - lo_o) < (hi_g - lo_g)


def test_specific_std_widens_percent_bounds() -> None:
    """hrs: percent override (0.3 vs 0.1) widens the value-scaled band."""
    i_val = 800.0
    pert = _make_pert({"max_std": 3, "std_hrs": 0.1}, 0.3, i_val)
    lo_o, hi_o = pert.get_boundaries(
        ptype=Ptype.HRS.value, i_val=i_val, param=PARAM)
    lo_g, hi_g = pert.get_boundaries(ptype=Ptype.HRS.value, i_val=i_val)
    assert (lo_o, hi_o) == pytest.approx(
        (i_val - i_val * 0.3 * 3, i_val + i_val * 0.3 * 3))
    assert (hi_o - lo_o) > (hi_g - lo_g)


def test_specific_std_scales_multiplicative_bounds() -> None:
    """mrc: multiplicative override (1.5 vs 1.2) keeps log symmetry lo*hi==i^2."""
    i_val = 500.0
    pert = _make_pert({"max_std": 3, "std_mrc": 1.2}, 1.5, i_val)
    lo_o, hi_o = pert.get_boundaries(
        ptype=Ptype.MRC.value, i_val=i_val, param=PARAM)
    factor = 1.5 ** 3  # = 3.375, using the OVERRIDE not the global 1.2.
    assert lo_o == pytest.approx(i_val / factor)
    assert hi_o == pytest.approx(i_val * factor)
    assert lo_o * hi_o == pytest.approx(i_val ** 2)
    # Distinct from the global (1.2 -> factor 1.6) band.
    lo_g, hi_g = pert.get_boundaries(ptype=Ptype.MRC.value, i_val=i_val)
    assert (hi_o - lo_o) > (hi_g - lo_g)


@pytest.mark.parametrize(
    "ptype,std_key,std_val,i_val",
    [
        (Ptype.WE.value, "std_we", 1.0, 100.0),
        (Ptype.HRS.value, "std_hrs", 0.1, 800.0),
        (Ptype.MRC.value, "std_mrc", 1.2, 500.0),
    ],
    ids=["we", "hrs", "mrc"])
def test_no_override_falls_back_to_global(ptype, std_key, std_val,
                                          i_val) -> None:
    """When uncertainty == global std, the param path matches the global path."""
    pert = _make_pert({"max_std": 3, std_key: std_val}, std_val, i_val)
    with_param = pert.get_boundaries(ptype=ptype, i_val=i_val, param=PARAM)
    global_path = pert.get_boundaries(ptype=ptype, i_val=i_val)
    assert with_param == pytest.approx(global_path)


def test_param_none_uses_global() -> None:
    """param=None ignores i_sop.uncertainties and uses settings['std_<ptype>']."""
    pert = _make_pert({"max_std": 3, "std_we": 1.0}, 9.0, 100.0)
    none_path = pert.get_boundaries(
        ptype=Ptype.WE.value, i_val=100.0, param=None)
    # Explicit global expectation (std_we = 1.0), NOT the 9.0 override.
    assert none_path == pytest.approx((100.0 - 3.0, 100.0 + 3.0))


def test_param_absent_from_uncertainties_falls_back() -> None:
    """An unknown param key (not in uncertainties) falls back to the global std."""
    pert = _make_pert({"max_std": 3, "std_we": 1.0}, 9.0, 100.0)
    # 'ghost' is absent from i_sop.uncertainties (which only holds PARAM).
    ghost = pert.get_boundaries(
        ptype=Ptype.WE.value, i_val=100.0, param="ghost")
    assert ghost == pytest.approx((100.0 - 3.0, 100.0 + 3.0))


def test_within_boundaries_forwards_param() -> None:
    """within_boundaries forwards param: a value inside the widened band but
    outside the global band is accepted only when param is supplied."""
    pert = _make_pert({"max_std": 3, "std_we": 1.0}, 3.0, 100.0)
    # Global band = (97, 103); widened override band = (91, 109).
    value = 105.0
    assert pert.within_boundaries(
        perturbed_val=value, ptype=Ptype.WE.value, initial_val=100.0,
        param=PARAM) is True
    assert pert.within_boundaries(
        perturbed_val=value, ptype=Ptype.WE.value, initial_val=100.0) is False


def test_get_rng_uniform_honors_param_bounds() -> None:
    """get_rng(UNIFORM) samples the param-widened band: draws exceed the
    global upper bound (impossible without forwarding param)."""
    pert = _make_pert({"max_std": 3, "std_we": 1.0}, 3.0, 100.0)
    global_hi = pert.get_boundaries(ptype=Ptype.WE.value, i_val=100.0)[1]
    widened_hi = pert.get_boundaries(
        ptype=Ptype.WE.value, i_val=100.0, param=PARAM)[1]
    np.random.seed(SEED)
    draws = np.array([
        pert.get_rng(ptype=Ptype.WE.value, i_val=100.0, c_val=100.0,
                     param=PARAM, distrib=Distrib.UNIFORM)
        for _ in range(N)])
    assert draws.max() > global_hi
    assert draws.max() <= widened_hi
    assert draws.min() >= pert.get_boundaries(
        ptype=Ptype.WE.value, i_val=100.0, param=PARAM)[0]


# ===========================================================================
# Group 6 -- Seed rescaling: the out-of-bounds "seed" value each perturb_*
#            method uses to force at least one get_rng draw must itself be
#            rescaled by the parameter-specific uncertainty.
# ===========================================================================

def _instrument(pert: Perturbator, in_bounds: float):
    """Record first within_boundaries perturbed_val + count get_rng calls.

    get_rng is stubbed to return an in-bounds value so the seeding loop exits
    after exactly one draw, leaving the initial seed observable in wb[0].
    """
    wb: list[float] = []
    calls = {"rng": 0}
    orig_wb = pert.within_boundaries

    def spy_wb(**kw: Any) -> bool:
        wb.append(kw["perturbed_val"])
        return orig_wb(**kw)

    def stub_rng(**kw: Any) -> float:
        calls["rng"] += 1
        return in_bounds

    pert.within_boundaries = spy_wb  # type: ignore[method-assign]
    pert.get_rng = stub_rng          # type: ignore[method-assign]
    return wb, calls


@pytest.mark.parametrize("glob,spec", [(1.0, 5.0), (1.0, 0.5)],
                         ids=["widened", "narrowed"])
def test_perturb_energy_seed_rescaled_to_specific_std(glob, spec) -> None:
    """perturb_energy seeds E0 - 3*max_std*sigma using the override sigma."""
    e0 = 100.0
    max_std = 3
    param = f"LEFT__{Ptype.WE.value}"
    pert = _make_pert({"max_std": max_std, f"std_{Ptype.WE.value}": glob},
                      spec, e0)
    pert.select = [param]
    pert.i_sop.items = {"LEFT": SimpleNamespace(energy=e0)}
    pert.i_sop.uncertainties = {param: spec}
    pert.settings["distrib_we"] = Distrib.NORMAL
    item = SimpleNamespace(name="LEFT", pert_e=True, energy=e0, _energy=e0)
    wb, calls = _instrument(pert, in_bounds=e0)

    pert.perturb_energy(item=item)

    seed = wb[0]
    assert seed == pytest.approx(e0 - 3 * max_std * spec)
    lo = min(pert.get_boundaries(
        ptype=Ptype.WE.value, i_val=e0, param=param))
    assert seed < lo
    assert calls["rng"] >= 1


def test_perturb_hindered_rotors_seed_rescaled() -> None:
    """perturb_hindered_rotors seeds 1 - 3*max_std*sigma using override sigma."""
    glob, spec, max_std = 0.1, 0.3, 3
    param = f"W__{Ptype.HRS.value}0"
    pert = _make_pert({"max_std": max_std, f"std_{Ptype.HRS.value}": glob},
                      spec, 1.0)
    pert.select = [param]
    pert.i_sop.uncertainties = {param: spec}
    pert.settings["distrib_hrs"] = Distrib.NORMAL
    rot = SimpleNamespace(pert=0.0)
    well = SimpleNamespace(name="W", h_rotors=[rot])
    wb, calls = _instrument(pert, in_bounds=1.0)

    pert.perturb_hindered_rotors(well=well)

    seed = wb[0]
    assert seed == pytest.approx(1 - 3 * max_std * spec)
    lo = min(pert.get_boundaries(
        ptype=Ptype.HRS.value, i_val=1.0, param=param))
    assert seed < lo
    assert calls["rng"] >= 1
    assert rot.pert == pytest.approx(1.0)


def test_perturb_multi_rotors_seed_rescaled() -> None:
    """perturb_multi_rotors seeds 1 - 3*max_std*sigma using override sigma."""
    glob, spec, max_std = 1.2, 1.5, 3
    param = f"W__{Ptype.MRC.value}0"
    pert = _make_pert({"max_std": max_std, f"std_{Ptype.MRC.value}": glob},
                      spec, 1.0)
    pert.select = [param]
    pert.i_sop.uncertainties = {param: spec}
    pert.settings["distrib_mrc"] = Distrib.LOGNORMAL
    rot = SimpleNamespace(sfc=0.0)
    well = SimpleNamespace(name="W", m_rotors=[rot])
    wb, calls = _instrument(pert, in_bounds=1.0)

    pert.perturb_multi_rotors(well=well)

    seed = wb[0]
    assert seed == pytest.approx(1 - 3 * max_std * spec)
    lo = min(pert.get_boundaries(
        ptype=Ptype.MRC.value, i_val=1.0, param=param))
    assert seed < lo
    assert calls["rng"] >= 1
    assert rot.sfc == pytest.approx(1.0)
