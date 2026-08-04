from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from kimeco.model import Model
from kimeco.parameters import SOP
from kimeco.well import Well
from kimeco.bimolecular import Bimolecular
from kimeco.barrier import Barrier
from kimeco.database.sop_db import SOP_DB
from kimeco.goat import GOATs
from kimeco.scoring_f.scoring import Scoring


ROWS: dict[str, dict[int, dict[str, float]]] = {
    'G0000': {
        0: {'fact': 1.1, 'pw': 2.1, 's0': 0.5, 's1': 0.6},
        1: {'fact': 1.2, 'pw': 2.2, 's0': 0.7, 's1': 0.8},
        2: {'fact': 1.3, 'pw': 2.3, 's0': 0.11, 's1': 0.12},
    },
    'G0001': {
        5: {'fact': 1.5, 'pw': 2.5, 's0': 0.9, 's1': 0.10},
        6: {'fact': 1.6, 'pw': 2.6, 's0': 0.13, 's1': 0.14},
    },
}

GEN0: list[tuple[int, int]] = [(0, 0), (0, 1), (0, 2)]
GEN1: list[tuple[int, int]] = [(0, 1), (1, 5), (0, 0), (1, 6)]


def _build_sop(n_exp: int = 2) -> SOP:
    sop = SOP(n_exp=n_exp)
    sop.factor = 1.0
    sop.power = 1.0
    sop.pres_unit = 'atm'
    sop.pres = [1.0]
    sop.temp = [300.0]
    sop.add_new_well(name='LEFT', pes_id=0)
    sop.add_new_well(name='RIGHT', pes_id=1)
    sop.add_new_bimol(name='LEFT+H', pes_id=0)
    left_h = cast(Bimolecular, sop.items['LEFT+H'])
    left_h.add_new_frag(name='LEFT_FRAG')
    sop.items['LEFT_FRAG'] = left_h.fragments[-1]
    sop.add_new_barrier(
        name='TS_LEFT_RIGHT', lside='LEFT', rside='RIGHT', pes_id=1)
    for well_name in ('LEFT', 'RIGHT', 'LEFT_FRAG'):
        cast(Well, sop.items[well_name])._freq = np.array([])
    ts = cast(Barrier, sop.items['TS_LEFT_RIGHT'])
    ts._freq = np.array([])
    ts._energy = 0.0
    ts.ifreq = 0.0
    return sop


def _row_values(sop: SOP, spec: dict[str, float]) -> dict[str, Any]:
    values = dict(sop.parameters_names)
    values['__fact'] = spec['fact']
    values['__pow'] = spec['pw']
    values['exp_000__score'] = spec['s0']
    values['exp_001__score'] = spec['s1']
    return values


def _make_db(tmp_path: Path, sop: SOP) -> SOP_DB:
    db = SOP_DB(sop=sop, name='TEST_SOP', path=str(tmp_path))
    for table, rows in ROWS.items():
        db.create_new_table(name=table)
        for row_id, spec in rows.items():
            db.prepare_batch_upsert(
                table=table, id=row_id, values=_row_values(sop, spec))
    db.batch_upsert()
    return db


def _make_goats(tmp_path: Path,
                db: SOP_DB,
                generations: list[list[tuple[int, int]]],
                sf: Any) -> GOATs:
    goats = GOATs(
        sop_db=db,
        kin_db=cast(Any, SimpleNamespace()),
        sim_db=cast(Any, SimpleNamespace()),
        sf=cast(Any, sf),
        wdir=str(tmp_path),
        overwrite=False,
    )
    goats.generations = generations
    goats.prefix = 'G'
    return goats


def _expected(tokens: list[tuple[int, int]], key: str) -> np.ndarray:
    return np.array([ROWS[f'G{g:04d}'][m][key] for (g, m) in tokens])


def _install_fscore_spy(monkeypatch: pytest.MonkeyPatch) -> list[Model]:
    """Replace Scoring.fscore with a spy that deterministically scores."""
    seen: list[Model] = []

    def _spy(self: Scoring, mdl: Model) -> None:
        mdl.score = 1000 + mdl.id
        seen.append(mdl)

    monkeypatch.setattr(Scoring, 'fscore', _spy)
    return seen


def _scoring(sop: SOP) -> Scoring:
    return Scoring(settings={}, initial_SOP=sop)


def test_get_goat_for_gen_returns_models_with_ids_gen_and_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    seen = _install_fscore_spy(monkeypatch)
    goats = _make_goats(tmp_path, db, [GEN1], _scoring(sop))

    models = goats.get_goat_for_gen(0)

    assert isinstance(models, list)
    assert all(isinstance(mdl, Model) for mdl in models)
    assert [(mdl.gen, mdl.id) for mdl in models] == GEN1
    assert [mdl.score for mdl in models] == [1000 + m for (_, m) in GEN1]
    # fscore called exactly once per returned model.
    assert len(seen) == len(GEN1)


def test_get_goat_for_gen_minus_one_returns_last_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    _install_fscore_spy(monkeypatch)
    goats = _make_goats(tmp_path, db, [GEN0, GEN1], _scoring(sop))

    last = goats.get_goat_for_gen(-1)
    explicit = goats.get_goat_for_gen(1)

    assert [(m.gen, m.id) for m in last] == [(m.gen, m.id) for m in explicit]
    assert [(m.gen, m.id) for m in last] == GEN1


def test_get_goat_for_gen_out_of_range_raises_indexerror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    _install_fscore_spy(monkeypatch)
    goats = _make_goats(tmp_path, db, [GEN0], _scoring(sop))

    with pytest.raises(IndexError, match='Generation number out of range'):
        goats.get_goat_for_gen(5)


def test_get_goat_for_gen_missing_id_raises_valueerror_exact_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    _install_fscore_spy(monkeypatch)
    # Token references id 99 that does not exist in G0000.
    goats = _make_goats(tmp_path, db, [[(0, 99)]], _scoring(sop))

    expected = (
        'Expected exactly one row for id 99 in table G0000, found 0'
    )
    with pytest.raises(ValueError) as exc:
        goats.get_goat_for_gen(0)
    assert str(exc.value) == expected


def test_get_goat_for_gen_golden_parity_multi_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    _install_fscore_spy(monkeypatch)
    goats = _make_goats(tmp_path, db, [GEN1], _scoring(sop))

    models = goats.get_goat_for_gen(0)

    reconstructed = np.array(
        [mdl.sop.parameters_names['__fact'] for mdl in models])
    assert np.array_equal(reconstructed, _expected(GEN1, 'fact'))
