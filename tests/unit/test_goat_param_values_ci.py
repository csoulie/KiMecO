from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from kimeco.parameters import SOP
from kimeco.well import Well
from kimeco.bimolecular import Bimolecular
from kimeco.barrier import Barrier
from kimeco.database.sop_db import SOP_DB
from kimeco.goat import GOATs


# Rows keyed by (table, row_id). Only __fact/__pow/score columns are varied
# because those reconstruct cleanly through SOP.from_db_row (used by the
# legacy get_goat_for_gen we compare against).
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

# Single-origin generation (all tokens from G0000).
GEN0: list[tuple[int, int]] = [(0, 0), (0, 1), (0, 2)]
# Multi-origin generation with a deliberate token ordering that interleaves
# both source tables.
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


class _SpyScoring:
    """Minimal stand-in for Scoring that counts fscore invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def fscore(self, mdl: Any) -> None:
        self.calls += 1
        mdl.score = 1000 + mdl.id


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
    return np.array([
        ROWS[f'G{g:04d}'][m][key] for (g, m) in tokens
    ])


def test_param_values_match_legacy_single_and_multiple_cols(
    tmp_path: Path,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    sf = _SpyScoring()
    goats = _make_goats(tmp_path, db, [GEN0], sf)

    single = goats.get_goat_param_values(0, ['__fact'])
    legacy = goats.get_goat_for_gen(0)
    assert np.array_equal(
        single['__fact'],
        np.array([mdl.sop.parameters_names['__fact'] for mdl in legacy]),
    )

    multi = goats.get_goat_param_values(0, ['__fact', '__pow'])
    legacy = goats.get_goat_for_gen(0)
    for col in ('__fact', '__pow'):
        assert np.array_equal(
            multi[col],
            np.array([mdl.sop.parameters_names[col] for mdl in legacy]),
        )


def test_param_values_multi_origin_preserves_token_order(
    tmp_path: Path,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    goats = _make_goats(tmp_path, db, [GEN1], _SpyScoring())

    values = goats.get_goat_param_values(0, ['__fact', '__pow'])

    assert np.array_equal(values['__fact'], _expected(GEN1, 'fact'))
    assert np.array_equal(values['__pow'], _expected(GEN1, 'pw'))


def test_param_values_score_column_parity(tmp_path: Path) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    goats = _make_goats(tmp_path, db, [GEN1], _SpyScoring())

    values = goats.get_goat_param_values(0, ['exp_000__score'])
    legacy = goats.get_goat_for_gen(0)

    assert np.array_equal(values['exp_000__score'], _expected(GEN1, 's0'))
    assert np.array_equal(
        values['exp_000__score'],
        np.array(
            [mdl.sop.parameters_names['exp_000__score'] for mdl in legacy]),
    )


def test_param_values_gen_minus_one_returns_last_generation(
    tmp_path: Path,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    goats = _make_goats(tmp_path, db, [GEN0, GEN1], _SpyScoring())

    last = goats.get_goat_param_values(-1, ['__fact'])
    explicit = goats.get_goat_param_values(1, ['__fact'])

    assert np.array_equal(last['__fact'], explicit['__fact'])
    assert np.array_equal(last['__fact'], _expected(GEN1, 'fact'))


def test_param_values_returns_float_ndarray_with_expected_length(
    tmp_path: Path,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    goats = _make_goats(tmp_path, db, [GEN1], _SpyScoring())

    values = goats.get_goat_param_values(0, ['__fact'])

    arr = values['__fact']
    assert isinstance(arr, np.ndarray)
    assert np.issubdtype(arr.dtype, np.floating)
    assert len(arr) == len(GEN1)


def test_param_values_never_calls_fscore(tmp_path: Path) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    sf = _SpyScoring()
    goats = _make_goats(tmp_path, db, [GEN1], sf)

    goats.get_goat_param_values(0, ['__fact', '__pow'])

    assert sf.calls == 0


def test_param_values_bypasses_sop_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    goats = _make_goats(tmp_path, db, [GEN1], _SpyScoring())

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('from_db_row must not be called')

    monkeypatch.setattr(SOP, 'from_db_row', staticmethod(_boom))

    values = goats.get_goat_param_values(0, ['__fact'])

    assert np.array_equal(values['__fact'], _expected(GEN1, 'fact'))
