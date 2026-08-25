"""CI-safe tests for ``GeneticAlgorithm.actualize_conv`` convergence logic.

The convergence test is now parameter-class aware (dispatched via
``get_parameter_type``):

* ADDITIVE (we, be, pow): absolute change (mean/std difference) vs
  settings['conv_<ptype>'].
* MULTIPLICATIVE (if, freq, bfc, sfc, mrc): LOG-space change
  |ln(old/new)| vs settings['param_conv'], with a zero-guard
  (old_mean == 0 or old_std == 0 -> not converged, no exception).
* PERCENT (hrs, sigma, epsilon, fact): RELATIVE change |old-new|/old vs
  settings['param_conv'], with the same zero-guard.

``actualize_conv`` only reads ``means/old_means/stds/old_stds/settings`` and
the name-mangled ``_GeneticAlgorithm__converged`` dict, and calls the module
``get_parameter_type``. We invoke it unbound against a ``SimpleNamespace`` so no
GA/DB/MESS machinery is constructed.
"""

from types import SimpleNamespace

import numpy as np

from kimeco.optimizers.GeneticAlgo.ga import GeneticAlgorithm


IF_P = "R__if"      # multiplicative
HRS_P = "W__hrs0"   # percent
WE_P = "W__we"      # additive

MANGLED = "_GeneticAlgorithm__converged"


def _fake_ga(*, means, old_means, stds, old_stds, settings) -> SimpleNamespace:
    """SimpleNamespace carrying exactly the attributes actualize_conv reads."""
    fake = SimpleNamespace(
        means=means,
        old_means=old_means,
        stds=stds,
        old_stds=old_stds,
        settings=settings,
    )
    setattr(fake, MANGLED, {})
    return fake


def _run(fake: SimpleNamespace) -> dict:
    GeneticAlgorithm.actualize_conv(fake)
    return getattr(fake, MANGLED)


# ===========================================================================
# MULTIPLICATIVE -- log-space change
# ===========================================================================

def test_multiplicative_uses_log_space_not_relative() -> None:
    """100->50 with param_conv=0.6: relative change (0.5) would converge, but
    log change |ln(100/50)| = 0.693 > 0.6 -> NOT converged."""
    fake = _fake_ga(
        means={IF_P: 50.0},
        old_means={IF_P: 100.0},
        stds={IF_P: 10.0},
        old_stds={IF_P: 10.0},   # std unchanged -> isolates the mean
        settings={"param_conv": 0.6},
    )
    conv = _run(fake)
    # log(2) == 0.693 exceeds 0.6.
    assert abs(np.log(100.0 / 50.0)) > 0.6
    assert conv[IF_P] is False


def test_multiplicative_converges_for_small_log_change() -> None:
    """100->95: |ln(100/95)| ~ 0.051 < 0.6 (both mean and std) -> converged."""
    fake = _fake_ga(
        means={IF_P: 95.0},
        old_means={IF_P: 100.0},
        stds={IF_P: 95.0},
        old_stds={IF_P: 100.0},
        settings={"param_conv": 0.6},
    )
    conv = _run(fake)
    assert abs(np.log(100.0 / 95.0)) < 0.6
    assert conv[IF_P] is True


def test_multiplicative_zero_guard_no_exception() -> None:
    """old_mean == 0 for a multiplicative param -> not converged, no divide."""
    fake = _fake_ga(
        means={IF_P: 50.0},
        old_means={IF_P: 0.0},
        stds={IF_P: 10.0},
        old_stds={IF_P: 10.0},
        settings={"param_conv": 0.6},
    )
    conv = _run(fake)   # must not raise
    assert conv[IF_P] is False


# ===========================================================================
# PERCENT -- relative change (regression: NOT log-space)
# ===========================================================================

def test_percent_keeps_relative_change() -> None:
    """Regression: 100->50 with param_conv=0.6 stays RELATIVE (0.5 < 0.6) ->
    converged, proving percent did NOT switch to the log rule."""
    fake = _fake_ga(
        means={HRS_P: 50.0},
        old_means={HRS_P: 100.0},
        stds={HRS_P: 50.0},
        old_stds={HRS_P: 100.0},
        settings={"param_conv": 0.6},
    )
    conv = _run(fake)
    assert abs(100.0 - 50.0) / 100.0 < 0.6   # relative rule converges
    assert conv[HRS_P] is True


def test_percent_zero_guard_no_exception() -> None:
    """old_mean == 0 for a percent param -> not converged, no divide-by-zero."""
    fake = _fake_ga(
        means={HRS_P: 50.0},
        old_means={HRS_P: 0.0},
        stds={HRS_P: 10.0},
        old_stds={HRS_P: 10.0},
        settings={"param_conv": 0.6},
    )
    conv = _run(fake)   # must not raise
    assert conv[HRS_P] is False


# ===========================================================================
# ADDITIVE -- absolute change vs conv_<ptype>
# ===========================================================================

def test_additive_uses_absolute_change_vs_conv_ptype() -> None:
    """we: absolute diff within conv_we converges even at large relative %."""
    fake = _fake_ga(
        means={WE_P: 1.4},
        old_means={WE_P: 1.0},   # absolute diff 0.4 < 1.0 (but 40% relative)
        stds={WE_P: 1.2},
        old_stds={WE_P: 1.0},    # absolute diff 0.2 < 1.0
        settings={"conv_we": 1.0},
    )
    conv = _run(fake)
    assert conv[WE_P] is True


def test_additive_not_converged_when_absolute_change_exceeds_threshold() -> None:
    """we: an absolute mean move beyond conv_we is not converged."""
    fake = _fake_ga(
        means={WE_P: 103.0},
        old_means={WE_P: 100.0},   # absolute diff 3.0 > 1.0
        stds={WE_P: 1.0},
        old_stds={WE_P: 1.0},
        settings={"conv_we": 1.0},
    )
    conv = _run(fake)
    assert conv[WE_P] is False
