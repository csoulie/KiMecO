"""CI-safe boundary + coverage tests for ``Perturbator`` sampling internals.

These tests lock the corrected multiplicative ``get_scale`` (log-space sigma)
introduced in 1.1.2 and prove -- deterministically and statistically -- that
``+/- max_std * sigma`` reaches exactly the ``get_boundaries`` factor for every
parameter class:

* ADDITIVE       (we, be, pow)              -> sigma = std                 (linear axis)
* PERCENT        (hrs, sigma, epsilon, fact)-> sigma = std * value         (linear axis)
* MULTIPLICATIVE (if, freq, bfc, sfc, mrc)  -> sigma = log(1+(std-1)*ms)/ms (log axis)

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
PARAM = "p"


def _make_pert(std_overrides: dict[str, Any],
               uncertainty: float,
               value: float) -> Perturbator:
    """Build a real Perturbator with a fake i_sop exposing exactly one param.

    ``uncertainty`` feeds ``get_scale`` (via ``i_sop.uncertainties``); the
    matching ``std_<ptype>`` in ``std_overrides`` feeds ``get_boundaries``.
    Keeping them equal is what makes +/-max_std*sigma land on the boundary.
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
    """value=1, std=1.2, max_std=3 -> bounds (1/1.6, 1.6) exactly."""
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    assert lo == pytest.approx(1 / 1.6)
    assert hi == pytest.approx(1.6)


def test_multiplicative_log_symmetry_for_value_not_one() -> None:
    """Multiplicative bounds are geometrically symmetric: lo*hi == i_val**2."""
    i_val = 800.0
    pert = _make_pert({"max_std": 4, "std_if": 1.1}, 1.1, i_val)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=i_val)
    factor = 1 + (1.1 - 1) * 4
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
    # Division-based lower bound: 1 / (1 + (4-1)*5) = 1/16.
    assert lo == pytest.approx(1 / 16)


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

def test_multiplicative_canonical_scale_is_log_1_6_over_3() -> None:
    """max_std * get_scale == log(1.6) for the canonical multiplicative case."""
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    scale = pert.get_scale(ptype=Ptype.IF.value, param=PARAM)
    assert 3 * scale == pytest.approx(np.log(1.6))


def test_multiplicative_lognormal_997_coverage_headline() -> None:
    """value=1, std=1.2, max_std=3: lognormal draws ~99.73% within (1/1.6,1.6)."""
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    assert (lo, hi) == pytest.approx((1 / 1.6, 1.6))
    frac = _coverage(pert, Ptype.IF.value, 1.0, Distrib.LOGNORMAL)
    assert abs(frac - THREE_SIGMA) < 0.004


def test_multiplicative_small_std_997_coverage() -> None:
    """A SMALL multiplicative std (1.1) still yields ~99.73% +/-3sigma coverage."""
    pert = _make_pert({"max_std": 3, "std_if": 1.1}, 1.1, 1.0)
    scale = pert.get_scale(ptype=Ptype.IF.value, param=PARAM)
    assert 3 * scale == pytest.approx(np.log(1 + (1.1 - 1) * 3))
    frac = _coverage(pert, Ptype.IF.value, 1.0, Distrib.LOGNORMAL)
    assert abs(frac - THREE_SIGMA) < 0.004


def test_multiplicative_large_std_997_coverage() -> None:
    """A LARGE multiplicative std (4) still yields ~99.73% +/-3sigma coverage."""
    pert = _make_pert({"max_std": 3, "std_if": 4.0}, 4.0, 1.0)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    # factor = 1 + (4-1)*3 = 10 -> bounds (0.1, 10).
    assert (lo, hi) == pytest.approx((0.1, 10.0))
    frac = _coverage(pert, Ptype.IF.value, 1.0, Distrib.LOGNORMAL)
    assert abs(frac - THREE_SIGMA) < 0.004


def test_multiplicative_default_max_std_4_above_999_coverage() -> None:
    """Default max_std=4 generalises: multiplicative lognormal covers >99.9%."""
    pert = _make_pert({"max_std": 4, "std_if": 1.2}, 1.2, 1.0)
    scale = pert.get_scale(ptype=Ptype.IF.value, param=PARAM)
    assert 4 * scale == pytest.approx(np.log(1 + (1.2 - 1) * 4))
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
    """Regression guard: get_scale is log((std-1)*ms+1)/ms, NOT (std-1)*value.

    Canonical case (value=1, std=1.2, max_std=3): the corrected log-space sigma
    is log(1.6)/3 ~ 0.1567, whereas the pre-fix arithmetic (std-1)*value = 0.2.
    """
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    scale = pert.get_scale(ptype=Ptype.IF.value, param=PARAM)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=1.0)
    # Correct identity: log(upper / i_val) / max_std.
    assert scale == pytest.approx(np.log(hi / 1.0) / 3)
    assert scale == pytest.approx(0.1566678764, abs=1e-6)
    # Must NOT be the old arithmetic (std - 1) * value == 0.2.
    prefix = (1.2 - 1) * 1.0
    assert abs(scale - prefix) > 0.01


def test_get_mean_sigma_removed() -> None:
    """Dead get_mean_sigma was removed from the Perturbator surface."""
    pert = _make_pert({"max_std": 3, "std_if": 1.2}, 1.2, 1.0)
    assert not hasattr(pert, "get_mean_sigma")
