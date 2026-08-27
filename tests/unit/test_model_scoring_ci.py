from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from kimeco.model import Model
from kimeco.scoring_f.scoring import Scoring, get_parameter_uncertainty_scale


def _sop(
    parameters: dict[str, float],
    uncertainties: dict[str, float],
    scores: dict[str, float],
):
    return SimpleNamespace(
        pres=[1.0],
        temp=[300.0],
        parameters_names=parameters,
        uncertainties=uncertainties,
        scores=scores,
    )


def _exp(
    name: str,
    data: np.ndarray,
    error: np.ndarray,
    weight: float = 1.0,
    sp_weights: np.ndarray | None = None,
):
    return SimpleNamespace(
        name=name,
        data=data,
        error=error,
        weight=weight,
        sp_weights=sp_weights,
    )


def test_scoring_sets_model_score_as_mutable_attribute() -> None:
    reference = _sop(
        parameters={"A__we": 10.0},
        uncertainties={"A__we": 2.0},
        scores={"exp_0": float("inf")},
    )
    settings = {
        "active_p": ["A__we"],
        "weight_theory": 1.0,
        "weight_experiments": 3.0,
        "fix_theory_divider": False,
    }
    exp = _exp(
        name="exp_0",
        data=np.array([[0.0, 1.0], [2.0, 2.0]], dtype=float),
        error=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float),
    )
    settings["experiments"] = [exp]
    sf = Scoring(settings=settings, initial_SOP=reference)

    mdl = Model(
        sop=cast(Any, _sop(
            parameters={"A__we": 12.0},
            uncertainties={"A__we": 2.0},
            scores={"exp_0": float("inf")},
        )),
        id=0,
    )
    mdl.sim = cast(Any, SimpleNamespace(
        profiles=[np.array([[0.0, 1.0], [2.0, 2.0]], dtype=float)]
    ))

    sf.score(mdl=mdl)

    assert mdl.theory_score == pytest.approx(1.0)
    assert mdl.experiment_score == pytest.approx(0.0)
    assert mdl.score == pytest.approx(0.25)
    assert mdl.sop.scores["exp_0"] == pytest.approx(0.0)


def test_scoring_uses_equal_split_when_global_weights_are_zero() -> None:
    reference = _sop(
        parameters={"A__we": 10.0},
        uncertainties={"A__we": 2.0},
        scores={"exp_0": float("inf")},
    )
    settings = {
        "active_p": ["A__we"],
        "weight_theory": 0.0,
        "weight_experiments": 0.0,
        "fix_theory_divider": False,
    }
    exp = _exp(
        name="exp_0",
        data=np.array([[0.0, 1.0], [np.sqrt(3.0), np.sqrt(3.0)]], dtype=float),
        error=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float),
    )
    settings["experiments"] = [exp]
    sf = Scoring(settings=settings, initial_SOP=reference)

    mdl = Model(
        sop=cast(Any, _sop(
            parameters={"A__we": 12.0},
            uncertainties={"A__we": 2.0},
            scores={"exp_0": float("inf")},
        )),
        id=1,
    )
    mdl.sim = cast(Any, SimpleNamespace(
        profiles=[np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)]
    ))

    sf.score(mdl=mdl)

    assert mdl.theory_score == pytest.approx(1.0)
    assert mdl.experiment_score == pytest.approx(3.0)
    assert mdl.score == pytest.approx(2.0)


def test_uncertainty_scale_scales_percent_and_multiplicative_types() -> None:
    scale_sigma = get_parameter_uncertainty_scale(
        reference_values={"__sigma0": 10.0},
        reference_uncertainties={"__sigma0": 0.1},
        param="__sigma0",
    )
    scale_if = get_parameter_uncertainty_scale(
        reference_values={"TS__if": 100.0},
        reference_uncertainties={"TS__if": 1.2},
        param="TS__if",
    )

    assert scale_sigma == pytest.approx(1.0)
    assert scale_if == pytest.approx(np.log(1.2))

    # Multiplicative scale depends only on the uncertainty (ln), not on the
    # reference value: a different reference value gives the same scale.
    scale_if_other_ref = get_parameter_uncertainty_scale(
        reference_values={"TS__if": 5.0},
        reference_uncertainties={"TS__if": 1.2},
        param="TS__if",
    )
    assert scale_if_other_ref == pytest.approx(np.log(1.2))


def test_scoring_keeps_finite_total_when_active_p_is_empty() -> None:
    reference = _sop(
        parameters={"A__we": 10.0},
        uncertainties={"A__we": 2.0},
        scores={"exp_0": float("inf")},
    )
    settings = {
        "active_p": [],
        "weight_theory": 1.0,
        "weight_experiments": 1.0,
        "fix_theory_divider": False,
    }
    exp = _exp(
        name="exp_0",
        data=np.array([[0.0, 1.0], [2.0, 2.0]], dtype=float),
        error=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float),
    )
    settings["experiments"] = [exp]
    sf = Scoring(settings=settings, initial_SOP=reference)

    mdl = Model(
        sop=cast(Any, _sop(
            parameters={"A__we": 12.0},
            uncertainties={"A__we": 2.0},
            scores={"exp_0": float("inf")},
        )),
        id=2,
    )
    mdl.sim = cast(Any, SimpleNamespace(
        profiles=[np.array([[0.0, 1.0], [2.0, 2.0]], dtype=float)]
    ))

    sf.score(mdl=mdl)

    assert mdl.theory_score == pytest.approx(1.0)
    assert np.isfinite(mdl.score)
    assert mdl.score == pytest.approx(0.5)


def test_multiplicative_theory_score_is_factor_symmetric() -> None:
    reference = _sop(
        parameters={"TS__if": 100.0},
        uncertainties={"TS__if": 1.2},
        scores={},
    )
    settings = {"active_p": ["TS__if"], "fix_theory_divider": False}
    sf = Scoring(settings=settings, initial_SOP=reference)

    up_sop = _sop(
        parameters={"TS__if": 100.0 * 1.5},
        uncertainties={"TS__if": 1.2},
        scores={},
    )
    down_sop = _sop(
        parameters={"TS__if": 100.0 / 1.5},
        uncertainties={"TS__if": 1.2},
        scores={},
    )

    up_score = sf.score_theory(cast(Any, up_sop))
    down_score = sf.score_theory(cast(Any, down_sop))

    expected = (np.log(1.5) / np.log(1.2)) ** 2
    assert up_score == pytest.approx(down_score)
    assert up_score == pytest.approx(expected)


def test_multiplicative_theory_score_unit_case() -> None:
    reference = _sop(
        parameters={"TS__if": 1.0},
        uncertainties={"TS__if": 2.0},
        scores={},
    )
    settings = {"active_p": ["TS__if"], "fix_theory_divider": False}
    sf = Scoring(settings=settings, initial_SOP=reference)

    candidate = _sop(
        parameters={"TS__if": 2.0},
        uncertainties={"TS__if": 2.0},
        scores={},
    )

    assert sf.score_theory(cast(Any, candidate)) == pytest.approx(1.0)


def test_multiplicative_uncertainty_one_raises() -> None:
    with pytest.raises(ValueError):
        get_parameter_uncertainty_scale(
            reference_values={"TS__if": 100.0},
            reference_uncertainties={"TS__if": 1.0},
            param="TS__if",
        )


def test_fix_theory_divider_false_uses_current_active_count() -> None:
    """Regression guard: with the flag off, score_theory normalizes by the
    number of parameters that actually differ from the reference."""
    reference = _sop(
        parameters={"A__we": 10.0, "B__we": 20.0},
        uncertainties={"A__we": 2.0, "B__we": 2.0},
        scores={},
    )
    settings = {
        "active_p": ["A__we", "B__we"],
        "fix_theory_divider": False,
    }
    sf = Scoring(settings=settings, initial_SOP=reference)

    candidate = _sop(
        parameters={"A__we": 12.0, "B__we": 24.0},
        uncertainties={"A__we": 2.0, "B__we": 2.0},
        scores={},
    )

    # Two parameters differ -> divide by 2.
    # ((12-10)/2)**2 / 2 + ((24-20)/2)**2 / 2 = 0.5 + 2.0 = 2.5
    assert sf.score_theory(cast(Any, candidate)) == pytest.approx(2.5)


def test_fix_theory_divider_true_locks_first_nonempty_length() -> None:
    """Core behavior: with the flag on, the divider is locked to the first
    non-empty active_p length and a later shorter active_p does not shrink it.
    """
    reference = _sop(
        parameters={"A__we": 10.0, "B__we": 20.0},
        uncertainties={"A__we": 2.0, "B__we": 2.0},
        scores={},
    )
    settings = {
        "active_p": ["A__we", "B__we"],
        "fix_theory_divider": True,
    }
    sf = Scoring(settings=settings, initial_SOP=reference)
    assert sf.t_div == 2

    # Later, active_p is reduced. The locked divider must NOT change.
    sf.set_active_p(["A__we"])
    assert sf.t_div == 2

    # Only one parameter actually differs, but the divider stays 2.
    candidate = _sop(
        parameters={"A__we": 12.0, "B__we": 20.0},
        uncertainties={"A__we": 2.0, "B__we": 2.0},
        scores={},
    )
    # ((12-10)/2)**2 / 2 = 0.5  (would be 1.0 if divider tracked the count)
    assert sf.score_theory(cast(Any, candidate)) == pytest.approx(0.5)


def test_fix_theory_divider_init_seeds_t_div_from_active_p() -> None:
    reference = _sop(
        parameters={"A__we": 10.0, "B__we": 20.0},
        uncertainties={"A__we": 2.0, "B__we": 2.0},
        scores={},
    )
    settings = {
        "active_p": ["A__we", "B__we"],
        "fix_theory_divider": True,
    }
    sf = Scoring(settings=settings, initial_SOP=reference)
    assert sf.t_div == 2


def test_fix_theory_divider_true_empty_active_p_gives_neutral_theory() -> None:
    """Edge: flag on but active_p empty -> t_div stays 0 -> neutral 0.0."""
    reference = _sop(
        parameters={"A__we": 10.0},
        uncertainties={"A__we": 2.0},
        scores={},
    )
    settings = {"active_p": [], "fix_theory_divider": True}
    sf = Scoring(settings=settings, initial_SOP=reference)
    assert sf.t_div == 0

    candidate = _sop(
        parameters={"A__we": 12.0},
        uncertainties={"A__we": 2.0},
        scores={},
    )
    assert sf.score_theory(cast(Any, candidate)) == pytest.approx(0.0)


def test_fix_theory_divider_first_nonempty_wins() -> None:
    """Edge: the FIRST non-empty active_p locks the divider; empties before it
    leave it at 0, and shorter lists after it do not change it."""
    reference = _sop(
        parameters={"A__we": 10.0},
        uncertainties={"A__we": 2.0},
        scores={},
    )
    settings = {"active_p": [], "fix_theory_divider": True}
    sf = Scoring(settings=settings, initial_SOP=reference)
    assert sf.t_div == 0

    # An empty update keeps it at 0.
    sf.set_active_p([])
    assert sf.t_div == 0

    # The first non-empty update locks the divider to 2.
    sf.set_active_p(["A__we", "B__we"])
    assert sf.t_div == 2

    # A later shorter update does not change the locked divider.
    sf.set_active_p(["A__we"])
    assert sf.t_div == 2


def test_additive_and_percent_scales_unchanged() -> None:
    scale_additive = get_parameter_uncertainty_scale(
        reference_values={"A__we": 10.0},
        reference_uncertainties={"A__we": 2.0},
        param="A__we",
    )
    scale_percent = get_parameter_uncertainty_scale(
        reference_values={"__sigma0": 10.0},
        reference_uncertainties={"__sigma0": 0.1},
        param="__sigma0",
    )

    assert scale_additive == pytest.approx(2.0)
    assert scale_percent == pytest.approx(1.0)
