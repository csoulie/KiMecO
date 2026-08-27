"""CI-safe tests for ``Linear.average`` (kimeco/sensitivity/linear.py).

``average`` collapses a list of SOP objects into a single template SOP by
averaging each parameter with the mean appropriate to its parameter class,
dispatched via ``get_parameter_type`` / ``Pclass``:

* MULTIPLICATIVE (if, freq, bfc, sfc, mrc): geometric mean
  ``exp(mean(log(values)))`` -- with a guard that raises ``ValueError`` on empty
  or non-positive samples (a geometric mean of those is undefined).
* everything else -- ADDITIVE (we, be, pow) and PERCENT (hrs, sigma, ...):
  plain arithmetic mean ``sum(values) / count``.

Kept MESS/DB/HPC-free: ``average`` only reads ``sop.parameters_names`` and calls
``SOP.from_db_row``. We invoke it unbound on a bare ``Linear.__new__(Linear)``,
feed ``SimpleNamespace`` SOPs, and monkeypatch ``linear_mod.SOP.from_db_row`` to
rebuild ``parameters_names`` from the template keys + the averaged row (returning
a ``SimpleNamespace``) so no real SOP/DB machinery is constructed. The real
``get_parameter_type`` / ``Pclass`` drive the class dispatch.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from kimeco.sensitivity.linear import Linear
from kimeco.enums import Ptype
import kimeco.sensitivity.linear as linear_mod


# Fully-qualified parameter names (dbs == '__'); the suffix drives dispatch.
# Verified against the real get_parameter_type:
#   R__if  -> Ptype.IF  (MULTIPLICATIVE)
#   W__we  -> Ptype.WE  (ADDITIVE, non-multiplicative)
#   W__hrs0 -> Ptype.HRS (PERCENT, non-multiplicative)
IF_P = "R__if"      # multiplicative -> geometric mean
WE_P = "W__we"      # additive       -> arithmetic mean
HRS_P = "W__hrs0"   # percent        -> arithmetic mean


def _sop(params: dict[str, float]) -> SimpleNamespace:
    """One fake SOP exposing ``sop.parameters_names``."""
    return SimpleNamespace(parameters_names=dict(params))


def _fake_from_db_row(sop_tpl, row):
    """Stand-in for SOP.from_db_row: rebuild parameters_names from the template
    key order + the averaged ``row`` values, as a SimpleNamespace."""
    keys = list(sop_tpl.parameters_names.keys())
    return SimpleNamespace(parameters_names=dict(zip(keys, row)))


@pytest.fixture()
def sa() -> Linear:
    """Bare Linear -- ``average`` touches no instance attributes."""
    return Linear.__new__(Linear)


@pytest.fixture(autouse=True)
def patch_from_db_row(monkeypatch):
    """Route SOP.from_db_row through the SimpleNamespace rebuilder for every
    test so ``average`` never constructs a real SOP."""
    monkeypatch.setattr(linear_mod.SOP, "from_db_row",
                        staticmethod(_fake_from_db_row))


# ===========================================================================
# MULTIPLICATIVE -> geometric mean
# ===========================================================================

def test_average_multiplicative_geometric_mean_numerical(sa) -> None:
    """if [1, 4, 16] -> geometric mean exp((ln1+ln4+ln16)/3) = 4.0, which is
    distinct from the arithmetic mean 7.0."""
    sops = [_sop({IF_P: 1.0}), _sop({IF_P: 4.0}), _sop({IF_P: 16.0})]
    out = sa.average(sops)
    assert out.parameters_names[IF_P] == pytest.approx(4.0)
    assert out.parameters_names[IF_P] != pytest.approx(7.0)


def test_average_two_sop_geometric_known_value(sa) -> None:
    """if [2, 8] -> sqrt(2*8) = 4.0 (arithmetic mean would be 5.0)."""
    sops = [_sop({IF_P: 2.0}), _sop({IF_P: 8.0})]
    out = sa.average(sops)
    assert out.parameters_names[IF_P] == pytest.approx(4.0)
    assert out.parameters_names[IF_P] != pytest.approx(5.0)


def test_average_multiplicative_all_equal_returns_value(sa) -> None:
    """A constant multiplicative sample -> geometric mean is the value itself."""
    sops = [_sop({IF_P: 5.0}) for _ in range(3)]
    out = sa.average(sops)
    assert out.parameters_names[IF_P] == pytest.approx(5.0)


# ===========================================================================
# non-MULTIPLICATIVE -> arithmetic mean
# ===========================================================================

def test_average_non_multiplicative_uses_arithmetic_mean(sa) -> None:
    """Mixed template: additive we and percent hrs use the arithmetic mean while
    the multiplicative if uses the geometric mean, all in one average() call."""
    sops = [
        _sop({WE_P: 3.0, HRS_P: 10.0, IF_P: 1.0}),
        _sop({WE_P: 6.0, HRS_P: 20.0, IF_P: 4.0}),
        _sop({WE_P: 9.0, HRS_P: 30.0, IF_P: 16.0}),
    ]
    out = sa.average(sops)
    assert out.parameters_names[WE_P] == pytest.approx(6.0)    # arithmetic
    assert out.parameters_names[HRS_P] == pytest.approx(20.0)  # arithmetic
    assert out.parameters_names[IF_P] == pytest.approx(4.0)    # geometric


def test_average_dispatch_uses_get_parameter_type(sa, monkeypatch) -> None:
    """average routes class selection through get_parameter_type: force a
    non-multiplicative type on an 'if' param and confirm the arithmetic branch
    (mean 7.0) runs instead of the geometric branch (4.0)."""
    monkeypatch.setattr(linear_mod, "get_parameter_type", lambda p: Ptype.WE)
    sops = [_sop({IF_P: 1.0}), _sop({IF_P: 4.0}), _sop({IF_P: 16.0})]
    out = sa.average(sops)
    assert out.parameters_names[IF_P] == pytest.approx(7.0)
    assert out.parameters_names[IF_P] != pytest.approx(4.0)


def test_average_single_sop_returns_values_unchanged(sa) -> None:
    """count == 1 -> every class collapses to the single sample value."""
    sops = [_sop({WE_P: 12.0, HRS_P: 33.0, IF_P: 7.0})]
    out = sa.average(sops)
    assert out.parameters_names[WE_P] == pytest.approx(12.0)
    assert out.parameters_names[HRS_P] == pytest.approx(33.0)
    assert out.parameters_names[IF_P] == pytest.approx(7.0)


# ===========================================================================
# error / edge paths
# ===========================================================================

def test_average_multiplicative_nonpositive_raises(sa) -> None:
    """Geometric mean is undefined for zero/negative multiplicative values ->
    ValueError, for both a zero and a negative sample."""
    with pytest.raises(ValueError):
        sa.average([_sop({IF_P: 4.0}), _sop({IF_P: 0.0})])
    with pytest.raises(ValueError):
        sa.average([_sop({IF_P: 4.0}), _sop({IF_P: -2.0})])


def test_average_empty_sop_list_raises(sa) -> None:
    """average([]) indexes sop_list[0] before any per-parameter guard runs, so
    the real failure is an IndexError (reconciled to the implementation)."""
    with pytest.raises(IndexError):
        sa.average([])
