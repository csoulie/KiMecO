"""CI-safe theory-score anchoring for the geometric multiplicative fix.

These lock the round-trip invariant that ties the corrected multiplicative
perturbation geometry to the log-space theory scoring:

* ``get_boundaries(if, i_val=ref)`` == ``(ref / std**max_std, ref * std**max_std)``
* ``score_theory(v)`` for a multiplicative param == ``(ln(v/ref) / ln(std))**2``

Feeding a boundary value back into ``score_theory`` must therefore yield exactly
``max_std**2`` (the +/-max_std sigma edge of the log-normal), and feeding an
SA/NM derivative step ``ref*std**step`` must yield exactly ``step**2``. This ties
the Perturbator geometry, the SA/NM ``calculate_dstep`` and the Scoring log-space
sigma together into one deterministic, MESS/HPC/DB-free identity.

``Perturbator`` is instantiated directly with a ``SimpleNamespace`` logger and a
``SimpleNamespace`` i_sop; ``Linear`` and ``NelderMead`` are built with
``__new__`` carrying only the attributes ``calculate_dstep`` reads; ``Scoring``
is built with a ``SimpleNamespace`` reference SOP.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from kimeco.Perturbators.perturbator import Perturbator
from kimeco.sensitivity.linear import Linear
from kimeco.optimizers.NelderMead.nelder_mead import NelderMead
from kimeco.scoring_f.scoring import Scoring
from kimeco.enums import Ptype


IF_P = "R__if"  # multiplicative parameter (suffix drives type dispatch)


def _make_pert(std_if: float, ref: float, max_std: int) -> Perturbator:
    """Real Perturbator whose i_sop exposes exactly one multiplicative param."""
    settings: dict[str, Any] = {
        "active_p": [], "max_std": max_std, "std_if": std_if,
    }
    klog = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    pert = Perturbator(settings, SimpleNamespace(), klog)
    pert.i_sop = SimpleNamespace(
        uncertainties={IF_P: std_if},
        parameters_names={IF_P: ref},
    )
    return pert


def _ref_scoring(ref: float, unc: float) -> Scoring:
    """Scoring anchored on a one-parameter reference SOP (ref, uncertainty)."""
    reference = SimpleNamespace(
        parameters_names={IF_P: ref},
        uncertainties={IF_P: unc},
    )
    return Scoring(
        settings={"active_p": [IF_P], "fix_theory_divider": False},
        initial_SOP=reference,
    )


def _candidate(value: float) -> Any:
    """A one-parameter SOP carrying only the perturbed multiplicative value."""
    return SimpleNamespace(parameters_names={IF_P: value})


def _make_sa(pert: Perturbator, lin_fact: float) -> Linear:
    sa = Linear.__new__(Linear)
    sa.pert = pert
    sa.lin_fact = lin_fact
    return sa


def _make_nm(pert: Perturbator, nm_dstep: float) -> NelderMead:
    nm = NelderMead.__new__(NelderMead)
    nm.pert = pert
    nm.settings = {"nm_dstep": nm_dstep}
    return nm


@pytest.mark.parametrize(
    "u,m,expected",
    [(1.2, 3, 9.0), (2.0, 4, 16.0)],
    ids=["u1.2-m3", "u2.0-m4"])
def test_theory_score_equals_max_std_squared_at_boundary(u, m, expected) -> None:
    """Both multiplicative boundaries score exactly max_std**2 in log space.

    hi == ref*u**m and lo == ref/u**m, and score_theory(hi) == score_theory(lo)
    == (ln(u**m)/ln(u))**2 == m**2  (u=1.2,m=3 -> 9.0; u=2,m=4 -> 16.0).
    """
    ref = 100.0
    pert = _make_pert(std_if=u, ref=ref, max_std=m)
    lo, hi = pert.get_boundaries(ptype=Ptype.IF.value, i_val=ref, param=IF_P)
    assert hi == pytest.approx(ref * u ** m)
    assert lo == pytest.approx(ref / u ** m)

    sf = _ref_scoring(ref=ref, unc=u)
    assert sf.score_theory(_candidate(hi)) == pytest.approx(expected)
    assert sf.score_theory(_candidate(lo)) == pytest.approx(expected)


def test_sa_nm_dstep_theory_score_equals_step_squared() -> None:
    """SA and NM multiplicative dstep ref*u**step scores exactly step**2.

    The derivative step at the reference value is ref*u**step; fed back into the
    log-space theory score it yields (ln(u**step)/ln(u))**2 == step**2
    (u=1.2, step=0.1 -> 0.01) for both the SA and the NM implementations.
    """
    ref, u, step = 100.0, 1.2, 0.1
    pert = _make_pert(std_if=u, ref=ref, max_std=3)
    sf = _ref_scoring(ref=ref, unc=u)

    sa = _make_sa(pert, lin_fact=step)
    sa_up = sa.calculate_dstep(uc=u, param=IF_P, side=1, value=ref)
    assert sa_up == pytest.approx(ref * u ** step)
    assert sf.score_theory(_candidate(sa_up)) == pytest.approx(step ** 2)

    nm = _make_nm(pert, nm_dstep=step)
    nm_up = nm.calculate_dstep(uc=u, param=IF_P, side=1, value=ref)
    assert nm_up == pytest.approx(ref * u ** step)
    assert sf.score_theory(_candidate(nm_up)) == pytest.approx(step ** 2)
