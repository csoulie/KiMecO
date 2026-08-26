"""CI-safe tests for ``GeneticAlgorithm.get_stats`` (and a companion set for
``actualize_conv``) covering per-parameter-class statistics.

``get_stats`` is now parameter-class aware: it dispatches via
``get_parameter_type``/``Pclass`` so that

* ADDITIVE (we, be, pow) and PERCENT (hrs, sigma, epsilon, fact) parameters use
  the arithmetic mean and population std (``np.average`` / ``np.std`` ddof=0);
* MULTIPLICATIVE (if, freq, bfc, sfc, mrc) parameters use the geometric mean and
  geometric std via ``geometric_mean_and_std`` (log-space).

Both ``get_stats`` and ``actualize_conv`` only touch ``settings`` plus a handful
of plain attributes, so we invoke them unbound against a ``SimpleNamespace`` --
no GA/DB/MESS machinery is constructed. ``geometric_mean_and_std`` is a
``@staticmethod``; attaching the raw function to the fake namespace makes the
``self.geometric_mean_and_std(...)`` call inside ``get_stats`` resolve correctly.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from kimeco.optimizers.GeneticAlgo.ga import GeneticAlgorithm


IF_P = "R__if"      # multiplicative -> geometric mean/std
HRS_P = "W__hrs0"   # percent        -> arithmetic mean/std
WE_P = "W__we"      # additive       -> arithmetic mean/std

MANGLED = "_GeneticAlgorithm__converged"


# ===========================================================================
# get_stats -- mean/std per parameter class
# ===========================================================================

def _model(params: dict[str, float]) -> SimpleNamespace:
    """One fake model exposing ``mdl.sop.parameters_names``."""
    return SimpleNamespace(sop=SimpleNamespace(parameters_names=params))


def _stats_fake(active_p: list[str]) -> SimpleNamespace:
    """SimpleNamespace carrying exactly what get_stats reads/calls."""
    fake = SimpleNamespace(settings={"active_p": list(active_p)})
    # @staticmethod -> raw function; self.geometric_mean_and_std(vals) resolves.
    fake.geometric_mean_and_std = GeneticAlgorithm.geometric_mean_and_std
    return fake


def test_get_stats_mean_std_per_parameter_type() -> None:
    """Feed the same [2.0, 8.0] sample to one key of each parameter class in a
    single get_stats call and assert the class-correct mean/std.

    * ADDITIVE / PERCENT -> arithmetic: mean 5.0, population std 3.0.
    * MULTIPLICATIVE     -> geometric:  mean exp((ln2+ln8)/2)=4.0,
                                        std  exp((ln8-ln2)/2)=2.0.
    """
    fake = _stats_fake([WE_P, IF_P, HRS_P])
    models = [
        _model({WE_P: 2.0, IF_P: 2.0, HRS_P: 2.0}),
        _model({WE_P: 8.0, IF_P: 8.0, HRS_P: 8.0}),
    ]

    means, stds = GeneticAlgorithm.get_stats(fake, models)

    # ADDITIVE -- arithmetic mean / population std.
    assert means[WE_P] == pytest.approx(5.0)
    assert stds[WE_P] == pytest.approx(3.0)

    # PERCENT -- arithmetic (regression: NOT geometric / log-space).
    assert means[HRS_P] == pytest.approx(5.0)
    assert stds[HRS_P] == pytest.approx(3.0)

    # MULTIPLICATIVE -- geometric mean / geometric std.
    assert means[IF_P] == pytest.approx(4.0)
    assert stds[IF_P] == pytest.approx(2.0)


def test_get_stats_multiplicative_all_equal_gives_unit_gstd() -> None:
    """Constant multiplicative sample -> geometric mean is the value and the
    geometric std collapses to exp(0) = 1.0 (no spread)."""
    fake = _stats_fake([IF_P])
    models = [_model({IF_P: 5.0}) for _ in range(3)]

    means, stds = GeneticAlgorithm.get_stats(fake, models)

    assert means[IF_P] == pytest.approx(5.0)
    assert stds[IF_P] == pytest.approx(1.0)


# ===========================================================================
# actualize_conv -- convergence rule per parameter class, no cross-contamination
# ===========================================================================

def _conv_fake(*, means, old_means, stds, old_stds, settings) -> SimpleNamespace:
    fake = SimpleNamespace(
        means=means,
        old_means=old_means,
        stds=stds,
        old_stds=old_stds,
        settings=settings,
    )
    setattr(fake, MANGLED, {})
    return fake


def _run_conv(fake: SimpleNamespace) -> dict:
    GeneticAlgorithm.actualize_conv(fake)
    return getattr(fake, MANGLED)


def test_additive_converges_within_conv_ptype() -> None:
    """ADDITIVE we: absolute Δmean 0.4 and Δstd 0.2 both < conv_we 0.5 -> True."""
    fake = _conv_fake(
        means={WE_P: 5.4},
        old_means={WE_P: 5.0},
        stds={WE_P: 3.2},
        old_stds={WE_P: 3.0},
        settings={"conv_we": 0.5},
    )
    conv = _run_conv(fake)
    assert conv[WE_P] is True


def test_additive_not_converged_when_absolute_change_exceeds_threshold() -> None:
    """ADDITIVE we: absolute Δmean 3.0 > conv_we 1.0 -> False."""
    fake = _conv_fake(
        means={WE_P: 103.0},
        old_means={WE_P: 100.0},
        stds={WE_P: 1.0},
        old_stds={WE_P: 1.0},
        settings={"conv_we": 1.0},
    )
    conv = _run_conv(fake)
    assert conv[WE_P] is False


def test_multiplicative_uses_log_space_not_relative() -> None:
    """MULTIPLICATIVE if: 100->50 with param_conv 0.6. The log change
    |ln(100/50)| = 0.693 > 0.6 -> False (relative 0.5 would have converged)."""
    fake = _conv_fake(
        means={IF_P: 50.0},
        old_means={IF_P: 100.0},
        stds={IF_P: 10.0},
        old_stds={IF_P: 10.0},
        settings={"param_conv": 0.6},
    )
    conv = _run_conv(fake)
    assert abs(np.log(100.0 / 50.0)) > 0.6
    assert conv[IF_P] is False


def test_multiplicative_converges_for_small_log_change() -> None:
    """MULTIPLICATIVE if: 4.0->4.2 -> |ln(4/4.2)| = 0.0488 < 0.6 -> True."""
    fake = _conv_fake(
        means={IF_P: 4.2},
        old_means={IF_P: 4.0},
        stds={IF_P: 4.2},
        old_stds={IF_P: 4.0},
        settings={"param_conv": 0.6},
    )
    conv = _run_conv(fake)
    assert abs(np.log(4.0 / 4.2)) < 0.6
    assert conv[IF_P] is True


def test_percent_keeps_relative_change() -> None:
    """PERCENT hrs: 100->50 with param_conv 0.6 stays RELATIVE
    (|100-50|/100 = 0.5 < 0.6) -> True, proving no switch to the log rule."""
    fake = _conv_fake(
        means={HRS_P: 50.0},
        old_means={HRS_P: 100.0},
        stds={HRS_P: 50.0},
        old_stds={HRS_P: 100.0},
        settings={"param_conv": 0.6},
    )
    conv = _run_conv(fake)
    assert abs(100.0 - 50.0) / 100.0 < 0.6
    assert conv[HRS_P] is True
