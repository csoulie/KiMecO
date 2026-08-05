from __future__ import annotations

from kimeco.enums import ModelStatus
from kimeco.model import Model
from kimeco.parameters import SOP


def _make_sop(factor: float = 1.0,
              power: float = 2.0,
              n_exp: int = 1) -> SOP:
    """Build a minimal SOP whose ``parameters_names`` is well defined."""
    sop = SOP(n_exp=n_exp)
    sop.factor = factor
    sop.power = power
    sop.pres = [1.0]
    sop.temp = [300.0]
    return sop


# ---------------------------------------------------------------------------
# SOP identity
# ---------------------------------------------------------------------------
def test_sop_eq_and_hash_true_when_parameters_names_equal() -> None:
    a = _make_sop()
    b = _make_sop()

    assert a == b
    assert hash(a) == hash(b)


def test_sop_eq_false_when_score_differs() -> None:
    a = _make_sop()
    b = _make_sop()
    # Scores participate in parameters_names, so mutating one breaks equality.
    key = next(iter(b.scores))
    b.scores[key] = 0.5

    assert a != b
    assert hash(a) != hash(b)


def test_sop_eq_notimplemented_for_none_and_non_sop() -> None:
    a = _make_sop()

    assert a.__eq__(None) is NotImplemented
    assert a.__eq__(object()) is NotImplemented
    # Python falls back to identity, so no exception is raised.
    assert (a == None) is False  # noqa: E711
    assert a != None  # noqa: E711


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
def test_model_eq_true_when_all_match() -> None:
    m1 = Model(sop=_make_sop(), id=0, gen=0)
    m2 = Model(sop=_make_sop(), id=0, gen=0)

    assert m1 == m2
    assert hash(m1) == hash(m2)


def test_model_eq_false_on_status_but_hash_excludes_status() -> None:
    m1 = Model(sop=_make_sop(), id=0, gen=0)
    m2 = Model(sop=_make_sop(), id=0, gen=0,
               status=ModelStatus.KIN.value)

    # __eq__ includes status ...
    assert m1 != m2
    # ... but __hash__ deliberately excludes it.
    assert hash(m1) == hash(m2)


def test_model_eq_false_on_id_or_gen() -> None:
    base = Model(sop=_make_sop(), id=0, gen=0)

    assert base != Model(sop=_make_sop(), id=1, gen=0)
    assert base != Model(sop=_make_sop(), id=0, gen=1)


def test_model_eq_notimplemented_for_none_and_non_model() -> None:
    m1 = Model(sop=_make_sop(), id=0, gen=0)

    assert m1.__eq__(None) is NotImplemented
    assert m1.__eq__(5) is NotImplemented
    assert (m1 == None) is False  # noqa: E711


def test_set_collapses_identical_models() -> None:
    m1 = Model(sop=_make_sop(), id=0, gen=0)
    m2 = Model(sop=_make_sop(), id=0, gen=0)

    assert len({m1, m2}) == 1


def test_set_preserves_status_differing_models() -> None:
    m1 = Model(sop=_make_sop(), id=0, gen=0)
    m2 = Model(sop=_make_sop(), id=0, gen=0,
               status=ModelStatus.KIN.value)

    # Same hash bucket, but __eq__ keeps them distinct.
    assert len({m1, m2}) == 2


def test_set_preserves_sop_differing_models() -> None:
    m1 = Model(sop=_make_sop(factor=1.0), id=0, gen=0)
    m2 = Model(sop=_make_sop(factor=9.0), id=0, gen=0)

    assert len({m1, m2}) == 2


def test_set_dedup_ignores_origin_prefix() -> None:
    m1 = Model(sop=_make_sop(), id=0, gen=0)
    m2 = Model(sop=_make_sop(), id=0, gen=0)
    m1.origin_prefix = 'G'
    m2.origin_prefix = 'NM'

    # origin_prefix participates in neither __eq__ nor __hash__.
    assert m1 == m2
    assert len({m1, m2}) == 1
