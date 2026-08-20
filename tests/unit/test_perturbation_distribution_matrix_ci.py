"""CI-safe tests for the perturbation-distribution validation matrix.

These tests lock the three-way, Pclass-dependent distribution rule enforced
by ``KMOInput.set_default_values`` for every ``distrib_<ptype>`` keyword:

* ADDITIVE  parameters (we, be, pow)          -> uniform / normal only
* PERCENT   parameters (hrs, sigma, epsilon,  -> uniform / normal only
            fact)
* MULTIPLICATIVE parameters (if, freq, bfc,   -> log-uniform / log-normal only
            sfc, mrc)

Everything is exercised through the backend directly (no MESS/HPC), using a
lightweight fake logger so no real logging side effects occur.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from kimeco.enums import Distrib, Pclass, Ptype
from kimeco.user_input import KMOInput


# All perturbation parameter types that carry a distribution keyword, i.e.
# every Ptype that belongs to one of the three Pclass categories (SCORE has
# no distribution and is therefore excluded).
_CATEGORY_OF: dict[str, Pclass] = {}
for _pclass in Pclass:
    for _pvalue in _pclass.value:
        _CATEGORY_OF[_pvalue] = _pclass

DISTRIB_PTYPES = [p for p in Ptype if p.value in _CATEGORY_OF]
ALL_DISTRIBS = list(Distrib)

LOG_DISTRIBS = {Distrib.LOGNORMAL, Distrib.LOGUNIFORM}


def _is_allowed(ptype: Ptype, distrib: Distrib) -> bool:
    """Expected acceptance, derived from Pclass + Distrib (not hardcoded)."""
    category = _CATEGORY_OF[ptype.value]
    if category is Pclass.MULTIPLICATIVE:
        return distrib in LOG_DISTRIBS
    # ADDITIVE and PERCENT both allow only the non-log distributions.
    return distrib not in LOG_DISTRIBS


def _make_input(json_file: dict[str, Any]) -> KMOInput:
    """Build a KMOInput without running __init__ (no file/logger needed)."""
    ui = object.__new__(KMOInput)
    ui.cancel_run = False
    ui.json_file = json_file
    ui.klog = SimpleNamespace(
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    return ui


# --- Full 12 ptypes x 4 distributions matrix (48 combos) -------------------

@pytest.mark.parametrize("ptype", DISTRIB_PTYPES, ids=lambda p: p.value)
@pytest.mark.parametrize("distrib", ALL_DISTRIBS, ids=lambda d: d.value)
def test_backend_distribution_matrix(ptype: Ptype, distrib: Distrib) -> None:
    """Every (ptype, distrib) combo is accepted iff the Pclass rule allows it."""
    key = f"distrib_{ptype.value}"
    ui = _make_input({key: distrib.value})
    ui.set_default_values()

    allowed = _is_allowed(ptype, distrib)
    assert ui.cancel_run is (not allowed), (
        f"{key}={distrib.value} expected "
        f"{'allowed' if allowed else 'rejected'}"
    )
    if allowed:
        # Accepted values are coerced to the Distrib enum in place.
        assert ui.json_file[key] == distrib


def test_matrix_covers_48_combinations() -> None:
    """Guard the intended coverage: 12 ptypes x 4 distributions."""
    assert len(DISTRIB_PTYPES) == 12
    assert len(ALL_DISTRIBS) == 4


# --- Standard (happy-path) cases -------------------------------------------

@pytest.mark.parametrize("ptype", [Ptype.WE, Ptype.BE, Ptype.ETP],
                         ids=lambda p: p.value)
@pytest.mark.parametrize("distrib", [Distrib.UNIFORM, Distrib.NORMAL],
                         ids=lambda d: d.value)
def test_additive_accepts_uniform_and_normal(ptype: Ptype,
                                              distrib: Distrib) -> None:
    ui = _make_input({f"distrib_{ptype.value}": distrib.value})
    ui.set_default_values()
    assert ui.cancel_run is False
    assert ui.json_file[f"distrib_{ptype.value}"] == distrib


@pytest.mark.parametrize("ptype",
                         [Ptype.IF, Ptype.IFC, Ptype.BFC, Ptype.SFC,
                          Ptype.MRC],
                         ids=lambda p: p.value)
@pytest.mark.parametrize("distrib", [Distrib.LOGNORMAL, Distrib.LOGUNIFORM],
                         ids=lambda d: d.value)
def test_multiplicative_accepts_log_distributions(ptype: Ptype,
                                                   distrib: Distrib) -> None:
    ui = _make_input({f"distrib_{ptype.value}": distrib.value})
    ui.set_default_values()
    assert ui.cancel_run is False
    assert ui.json_file[f"distrib_{ptype.value}"] == distrib


@pytest.mark.parametrize("ptype",
                         [Ptype.HRS, Ptype.SIG, Ptype.EPSI, Ptype.ETF],
                         ids=lambda p: p.value)
@pytest.mark.parametrize("distrib", [Distrib.UNIFORM, Distrib.NORMAL],
                         ids=lambda d: d.value)
def test_percent_accepts_uniform_and_normal(ptype: Ptype,
                                             distrib: Distrib) -> None:
    ui = _make_input({f"distrib_{ptype.value}": distrib.value})
    ui.set_default_values()
    assert ui.cancel_run is False
    assert ui.json_file[f"distrib_{ptype.value}"] == distrib


def test_defaults_leave_cancel_run_false() -> None:
    """Empty json_file: defaults are filled and nothing is rejected."""
    ui = _make_input({})
    ui.set_default_values()
    assert ui.cancel_run is False
    # Defaults are coerced to Distrib enums for every distribution keyword.
    for ptype in DISTRIB_PTYPES:
        assert isinstance(ui.json_file[f"distrib_{ptype.value}"], Distrib)


# --- Edge cases ------------------------------------------------------------

@pytest.mark.parametrize("ptype", [Ptype.WE, Ptype.IF, Ptype.HRS],
                         ids=lambda p: p.value)
def test_unknown_distribution_cancels_run(ptype: Ptype) -> None:
    """An unrecognised distribution string cancels the run, one per category."""
    ui = _make_input({f"distrib_{ptype.value}": "triangular"})
    ui.set_default_values()
    assert ui.cancel_run is True


@pytest.mark.parametrize("ptype,raw,expected", [
    (Ptype.WE, "Normal", Distrib.NORMAL),
    (Ptype.BE, "UNIFORM", Distrib.UNIFORM),
    (Ptype.IF, "LOG-NORMAL", Distrib.LOGNORMAL),
    (Ptype.MRC, "Log-Uniform", Distrib.LOGUNIFORM),
    (Ptype.HRS, "NoRmAl", Distrib.NORMAL),
])
def test_case_insensitive_values_are_coerced(ptype: Ptype, raw: str,
                                             expected: Distrib) -> None:
    """Mixed-case values are casefolded, validated and coerced consistently."""
    ui = _make_input({f"distrib_{ptype.value}": raw})
    ui.set_default_values()
    assert ui.cancel_run is False
    assert ui.json_file[f"distrib_{ptype.value}"] == expected


def test_case_insensitive_rejected_value_still_rejected() -> None:
    """A disallowed distribution is rejected regardless of casing."""
    ui = _make_input({f"distrib_{Ptype.WE.value}": "LOG-NORMAL"})
    ui.set_default_values()
    assert ui.cancel_run is True


def test_warning_invoked_on_rejected_combo() -> None:
    """The fake klog.warning is called when a combo is rejected."""
    calls: list[str] = []
    ui = _make_input({f"distrib_{Ptype.WE.value}": Distrib.LOGNORMAL.value})
    ui.klog = SimpleNamespace(
        warning=lambda msg, *a, **k: calls.append(msg),
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    ui.set_default_values()
    assert ui.cancel_run is True
    assert calls, "expected klog.warning to be invoked on rejection"


def test_multiplicative_uniform_is_rejected() -> None:
    """Regression lock: multiplicative parameter rejects a non-log distrib."""
    ui = _make_input({f"distrib_{Ptype.IF.value}": Distrib.UNIFORM.value})
    ui.set_default_values()
    assert ui.cancel_run is True
