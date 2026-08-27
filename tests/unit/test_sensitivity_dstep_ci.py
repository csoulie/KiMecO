"""CI-safe tests for the derivative-step perturbation used by the Linear
sensitivity analysis (``Linear.calculate_dstep``) and by the Nelder-Mead
optimizer (``NelderMead.calculate_dstep`` + ``get_initial_simplex``).

Both ``calculate_dstep`` implementations now RETURN the perturbed value and
dispatch on the parameter class via ``get_parameter_type``:

* MULTIPLICATIVE (if, freq, bfc, sfc, mrc): factor = uc**step;
  side=+1 -> value*factor, side=-1 -> value/factor  (geometric/log symmetric).
* ADDITIVE (we, be, pow) / PERCENT (hrs, sigma, epsilon, fact):
  value + get_scale(ptype, param) * step * side          (linear symmetric).

``step`` is ``lin_fact`` (== settings['sensi_d']) for the SA and
``settings['nm_dstep']`` for Nelder-Mead.

Everything runs through the REAL ``Perturbator.get_scale`` and the REAL enum
dispatch -- no MESS/HPC/DB. ``Linear`` and ``NelderMead`` are built with
``__new__`` and have exactly the attributes their ``calculate_dstep`` reads
attached by hand; the perturbator carries a ``SimpleNamespace`` i_sop.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from kimeco.Perturbators.perturbator import Perturbator
from kimeco.sensitivity.linear import Linear
from kimeco.optimizers.NelderMead.nelder_mead import NelderMead
from kimeco.enums import Ptype
import kimeco.sensitivity.linear as linear_mod


# Fully-qualified parameter names (dbs == '__'); the suffix drives dispatch.
IF_P = "R__if"      # multiplicative
MRC_P = "R__mrc0"   # multiplicative
WE_P = "W__we"      # additive
HRS_P = "W__hrs0"   # percent


def _make_pert(uncertainties: dict[str, float],
               values: dict[str, float],
               std_overrides: dict[str, Any] | None = None) -> Perturbator:
    """Build a real Perturbator with a fake i_sop exposing the given params."""
    settings: dict[str, Any] = {"active_p": [], "max_std": 3}
    if std_overrides:
        settings.update(std_overrides)
    klog = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    pert = Perturbator(settings, SimpleNamespace(), klog)
    pert.i_sop = SimpleNamespace(
        uncertainties=dict(uncertainties),
        parameters_names=dict(values),
    )
    return pert


def _make_sa(pert: Perturbator, lin_fact: float) -> Linear:
    """Bare Linear carrying only what ``calculate_dstep`` touches."""
    sa = Linear.__new__(Linear)
    sa.pert = pert
    sa.lin_fact = lin_fact
    return sa


def _make_nm(pert: Perturbator, nm_dstep: float) -> NelderMead:
    """Bare NelderMead carrying only what ``calculate_dstep`` touches."""
    nm = NelderMead.__new__(NelderMead)
    nm.pert = pert
    nm.settings = {"nm_dstep": nm_dstep}
    return nm


# ===========================================================================
# Linear SA -- MULTIPLICATIVE
# ===========================================================================

def test_sa_multiplicative_worked_example_exact() -> None:
    """uc=1.1, sensi_d=0.1, value=1.0, 'if' -> up/down == [1.1**0.1, 1/1.1**0.1]."""
    pert = _make_pert({IF_P: 1.1}, {IF_P: 1.0})
    sa = _make_sa(pert, lin_fact=0.1)
    up = sa.calculate_dstep(uc=1.1, param=IF_P, side=1, value=1.0)
    down = sa.calculate_dstep(uc=1.1, param=IF_P, side=-1, value=1.0)
    assert up == pytest.approx(1.1 ** 0.1)
    assert down == pytest.approx(1 / 1.1 ** 0.1)


def test_sa_multiplicative_magnitudes_for_nonunit_value() -> None:
    """Non-unit value scales by the same factor on each side."""
    value = 800.0
    pert = _make_pert({IF_P: 1.2}, {IF_P: value})
    sa = _make_sa(pert, lin_fact=0.1)
    factor = 1.2 ** 0.1
    up = sa.calculate_dstep(uc=1.2, param=IF_P, side=1, value=value)
    down = sa.calculate_dstep(uc=1.2, param=IF_P, side=-1, value=value)
    assert up == pytest.approx(value * factor)
    assert down == pytest.approx(value / factor)


def test_sa_multiplicative_log_symmetry() -> None:
    """up/value == value/down == f and up*down == value**2 (geometric)."""
    value = 500.0
    pert = _make_pert({MRC_P: 1.3}, {MRC_P: value})
    sa = _make_sa(pert, lin_fact=0.2)
    f = 1.3 ** 0.2
    up = sa.calculate_dstep(uc=1.3, param=MRC_P, side=1, value=value)
    down = sa.calculate_dstep(uc=1.3, param=MRC_P, side=-1, value=value)
    assert up / value == pytest.approx(f)
    assert value / down == pytest.approx(f)
    assert up * down == pytest.approx(value ** 2)


def test_sa_multiplicative_no_double_scaling() -> None:
    """Exactly [1/f, f]*value with f = uc**lin_fact, nothing extra."""
    value = 3.0
    uc, lin = 1.5, 0.1
    pert = _make_pert({IF_P: uc}, {IF_P: value})
    sa = _make_sa(pert, lin_fact=lin)
    f = uc ** lin
    assert sa.calculate_dstep(uc=uc, param=IF_P, side=1,
                              value=value) == pytest.approx(value * f)
    assert sa.calculate_dstep(uc=uc, param=IF_P, side=-1,
                              value=value) == pytest.approx(value / f)


def test_sa_multiplicative_regression_not_linear_logsigma() -> None:
    """Regression: multiplicative up is value*uc**lin, NOT value + ln(uc)*lin.

    The pre-fix bug added the log-space sigma linearly. With uc=2.0, lin=0.5 the
    geometric step is value*2**0.5 == 1.41421..., clearly separated from the
    buggy linear-log-sigma value 1 + ln(2)*0.5 == 1.34657.
    """
    value = 1.0
    uc, lin = 2.0, 0.5
    pert = _make_pert({IF_P: uc}, {IF_P: value},
                      std_overrides={"max_std": 3})
    sa = _make_sa(pert, lin_fact=lin)
    up = sa.calculate_dstep(uc=uc, param=IF_P, side=1, value=value)
    f = uc ** lin
    assert up == pytest.approx(value * f)
    assert up == pytest.approx(1.41421356)
    # The (buggy) linear-in-log-sigma value must differ.
    scale = pert.get_scale(ptype=Ptype.IF.value, param=IF_P)
    buggy = value + scale * lin
    assert buggy == pytest.approx(1.34657359)
    assert abs(up - buggy) > 1e-4


def test_sa_multiplicative_uc_one_is_identity() -> None:
    """uc==1 -> factor 1 -> up == down == value (no perturbation)."""
    value = 42.0
    pert = _make_pert({IF_P: 1.0}, {IF_P: value})
    sa = _make_sa(pert, lin_fact=0.1)
    up = sa.calculate_dstep(uc=1.0, param=IF_P, side=1, value=value)
    down = sa.calculate_dstep(uc=1.0, param=IF_P, side=-1, value=value)
    assert up == pytest.approx(value)
    assert down == pytest.approx(value)


def test_sa_multiplicative_large_uc_finite_positive_symmetric() -> None:
    """A very large uc stays finite, strictly positive, still log-symmetric."""
    value = 2.0
    uc, lin = 1000.0, 0.1
    pert = _make_pert({IF_P: uc}, {IF_P: value})
    sa = _make_sa(pert, lin_fact=lin)
    up = sa.calculate_dstep(uc=uc, param=IF_P, side=1, value=value)
    down = sa.calculate_dstep(uc=uc, param=IF_P, side=-1, value=value)
    assert np.isfinite(up) and np.isfinite(down)
    assert up > 0.0 and down > 0.0
    assert up * down == pytest.approx(value ** 2)


# ===========================================================================
# Linear SA -- ADDITIVE / PERCENT (linear axis)
# ===========================================================================

def test_sa_additive_linear_symmetry_and_step() -> None:
    """we: up-value == value-down == get_scale*lin_fact; scale == uncertainty."""
    value, unc, lin = 100.0, 2.0, 0.1
    pert = _make_pert({WE_P: unc}, {WE_P: value})
    sa = _make_sa(pert, lin_fact=lin)
    scale = pert.get_scale(ptype=Ptype.WE.value, param=WE_P)
    assert scale == pytest.approx(unc)          # additive scale == uncertainty
    step = scale * lin
    up = sa.calculate_dstep(uc=unc, param=WE_P, side=1, value=value)
    down = sa.calculate_dstep(uc=unc, param=WE_P, side=-1, value=value)
    assert up == pytest.approx(value + step)
    assert down == pytest.approx(value - step)
    assert (up - value) == pytest.approx(value - down)


def test_sa_additive_magnitude_is_scale_times_linfact() -> None:
    """The step magnitude equals get_scale * lin_fact exactly."""
    value, unc, lin = 100.0, 1.5, 0.2
    pert = _make_pert({WE_P: unc}, {WE_P: value})
    sa = _make_sa(pert, lin_fact=lin)
    scale = pert.get_scale(ptype=Ptype.WE.value, param=WE_P)
    up = sa.calculate_dstep(uc=unc, param=WE_P, side=1, value=value)
    assert (up - value) == pytest.approx(scale * lin)


def test_sa_percent_linear_symmetry() -> None:
    """hrs: percent scale == uncertainty*value; up-value == value-down."""
    value, unc, lin = 800.0, 0.1, 0.1
    pert = _make_pert({HRS_P: unc}, {HRS_P: value})
    sa = _make_sa(pert, lin_fact=lin)
    scale = pert.get_scale(ptype=Ptype.HRS.value, param=HRS_P)
    assert scale == pytest.approx(unc * value)   # percent scale == unc*value
    step = scale * lin
    up = sa.calculate_dstep(uc=unc, param=HRS_P, side=1, value=value)
    down = sa.calculate_dstep(uc=unc, param=HRS_P, side=-1, value=value)
    assert up == pytest.approx(value + step)
    assert down == pytest.approx(value - step)
    assert (up - value) == pytest.approx(value - down)


def test_sa_dispatch_uses_get_parameter_type(monkeypatch) -> None:
    """calculate_dstep routes through get_parameter_type: force ADDITIVE on an
    'if' param and confirm the additive (linear) branch runs, not geometric."""
    value, unc, lin = 4.0, 1.5, 0.1
    pert = _make_pert({IF_P: unc}, {IF_P: value})
    sa = _make_sa(pert, lin_fact=lin)
    # Patch the symbol the module actually calls.
    monkeypatch.setattr(linear_mod, "get_parameter_type", lambda p: Ptype.WE)
    out = sa.calculate_dstep(uc=unc, param=IF_P, side=1, value=value)
    scale = pert.get_scale(ptype=Ptype.WE.value, param=IF_P)
    assert out == pytest.approx(value + scale * lin)   # linear, not value*f


def test_sa_raises_without_perturbator() -> None:
    """A None perturbator makes calculate_dstep raise for the linear branch."""
    sa = Linear.__new__(Linear)
    sa.pert = None
    sa.lin_fact = 0.1
    with pytest.raises(RuntimeError):
        sa.calculate_dstep(uc=1.0, param=WE_P, side=1, value=1.0)


# ===========================================================================
# Nelder-Mead -- calculate_dstep uses settings['nm_dstep']
# ===========================================================================

def test_nm_multiplicative_uses_nm_dstep_not_sensi_d() -> None:
    """NM factor uses nm_dstep; side=+1 -> value*f."""
    value, uc, step = 2.0, 1.2, 0.05
    pert = _make_pert({IF_P: uc}, {IF_P: value})
    nm = _make_nm(pert, nm_dstep=step)
    f = uc ** step
    assert nm.calculate_dstep(uc=uc, param=IF_P, side=1,
                              value=value) == pytest.approx(value * f)


def test_nm_multiplicative_side_symmetry() -> None:
    """side=-1 -> value/f, side=+1 -> value*f, up*down == value**2."""
    value, uc, step = 7.0, 1.4, 0.1
    pert = _make_pert({MRC_P: uc}, {MRC_P: value})
    nm = _make_nm(pert, nm_dstep=step)
    f = uc ** step
    up = nm.calculate_dstep(uc=uc, param=MRC_P, side=1, value=value)
    down = nm.calculate_dstep(uc=uc, param=MRC_P, side=-1, value=value)
    assert up == pytest.approx(value * f)
    assert down == pytest.approx(value / f)
    assert up * down == pytest.approx(value ** 2)


def test_nm_additive_percent_symmetric_with_nm_dstep() -> None:
    """NM linear branch: value +/- get_scale*nm_dstep for we and hrs."""
    step = 0.1
    # additive
    pert_a = _make_pert({WE_P: 2.0}, {WE_P: 100.0})
    nm_a = _make_nm(pert_a, nm_dstep=step)
    scale_a = pert_a.get_scale(ptype=Ptype.WE.value, param=WE_P)
    up_a = nm_a.calculate_dstep(uc=2.0, param=WE_P, side=1, value=100.0)
    down_a = nm_a.calculate_dstep(uc=2.0, param=WE_P, side=-1, value=100.0)
    assert up_a == pytest.approx(100.0 + scale_a * step)
    assert down_a == pytest.approx(100.0 - scale_a * step)
    # percent
    pert_p = _make_pert({HRS_P: 0.1}, {HRS_P: 800.0})
    nm_p = _make_nm(pert_p, nm_dstep=step)
    scale_p = pert_p.get_scale(ptype=Ptype.HRS.value, param=HRS_P)
    up_p = nm_p.calculate_dstep(uc=0.1, param=HRS_P, side=1, value=800.0)
    assert up_p == pytest.approx(800.0 + scale_p * step)
    assert (up_p - 800.0) == pytest.approx(scale_p * step)


def test_nm_initial_simplex_multiplicative_vertex_roundtrips() -> None:
    """get_initial_simplex integration: the perturbed multiplicative vertex,
    back-transformed through get_absolute, equals value*f (== calculate_dstep).
    """
    value, uc, step = 2.0, 1.1, 0.1
    pert = _make_pert(
        {IF_P: uc}, {IF_P: value},
        std_overrides={"max_std": 3, f"std_{Ptype.IF.value}": uc})
    nm = _make_nm(pert, nm_dstep=step)
    sop = SimpleNamespace(
        uncertainties={IF_P: uc},
        parameters_names={IF_P: value},
    )
    nm.f_mdl = SimpleNamespace(sop=sop)
    nm.last_vertice = sop
    nm.current_dimensions = [IF_P]

    simplex = nm.get_initial_simplex()
    assert simplex.shape == (2, 1)
    # Row 0 is the (normalized) initial vertex -> 0 in log space.
    assert simplex[0][0] == pytest.approx(0.0)
    # Row 1 is the perturbed vertex; back-transform must equal value*f.
    f = uc ** step
    recovered = nm.get_absolute(param=IF_P, value=simplex[1][0])
    assert recovered == pytest.approx(value * f)
