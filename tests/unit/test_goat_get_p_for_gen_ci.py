from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from kimeco.parameters import SOP
from kimeco.well import Well
from kimeco.bimolecular import Bimolecular
from kimeco.barrier import Barrier
from kimeco.database.sop_db import SOP_DB
from kimeco.goat import GOATs


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
                generations: list[list[tuple[int, int]]]) -> GOATs:
    goats = GOATs(
        sop_db=db,
        kin_db=cast(Any, SimpleNamespace()),
        sim_db=cast(Any, SimpleNamespace()),
        sf=cast(Any, SimpleNamespace()),
        wdir=str(tmp_path),
        overwrite=False,
    )
    goats.generations = generations
    goats.prefix = 'G'
    return goats


def _expected(tokens: list[tuple[int, int]], key: str) -> list[float]:
    return [ROWS[f'G{g:04d}'][m][key] for (g, m) in tokens]


def test_get_p_for_gen_shape_values_and_token_order(tmp_path: Path) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    goats = _make_goats(tmp_path, db, [GEN1])

    params = ['__fact', '__pow']
    result = goats.get_p_for_gen(params, 0)

    assert result.shape == (len(GEN1), len(params))
    assert np.array_equal(result[:, 0], np.array(_expected(GEN1, 'fact')))
    assert np.array_equal(result[:, 1], np.array(_expected(GEN1, 'pw')))


def test_get_p_for_gen_single_param(tmp_path: Path) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    goats = _make_goats(tmp_path, db, [GEN0])

    result = goats.get_p_for_gen(['__fact'], 0)

    assert result.shape == (len(GEN0), 1)
    assert np.array_equal(result[:, 0], np.array(_expected(GEN0, 'fact')))


def test_get_p_for_gen_multi_origin_token_order(tmp_path: Path) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    goats = _make_goats(tmp_path, db, [GEN1])

    result = goats.get_p_for_gen(['exp_000__score'], 0)

    assert np.array_equal(result[:, 0], np.array(_expected(GEN1, 's0')))


def test_batch_select_cols_is_id_keyed_and_clears_select(
    tmp_path: Path,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)

    db.prepare_batch_select(table='G0000', row_id=0)
    db.prepare_batch_select(table='G0000', row_id=1)
    db.prepare_batch_select(table='G0001', row_id=5)

    raw = db.batch_select_cols(cols=['__fact', 'exp_000__score'])

    assert set(raw.keys()) == {'G0000', 'G0001'}
    assert set(raw['G0000'].keys()) == {0, 1}
    assert set(raw['G0001'].keys()) == {5}
    # Each tuple carries only the requested columns in request order; the id
    # is carried as the dict key, not inside the tuple.
    assert raw['G0000'][0] == (1.1, 0.5)
    assert raw['G0000'][1] == (1.2, 0.7)
    assert raw['G0001'][5] == (1.5, 0.9)
    assert all(len(vals) == 2 for table in raw.values()
               for vals in table.values())
    # _select is cleared after execution.
    assert db._select == {}


def test_get_p_for_gen_io_parity_golden(tmp_path: Path) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    goats = _make_goats(tmp_path, db, [GEN1])

    params = ['__fact', '__pow', 'exp_000__score']
    result = goats.get_p_for_gen(params, 0)

    golden = np.array([
        [ROWS[f'G{g:04d}'][m]['fact'],
         ROWS[f'G{g:04d}'][m]['pw'],
         ROWS[f'G{g:04d}'][m]['s0']]
        for (g, m) in GEN1
    ])
    assert np.array_equal(result, golden)
