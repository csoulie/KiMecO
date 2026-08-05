from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from kimeco.database.sim_db import SIM_DB
from kimeco.model import Model


def _exp(species=('A',)):
    return SimpleNamespace(
        species=list(species),
        data=np.array([[0.0, 1.0]], dtype=float),  # data[0] == time grid
    )


def _model(sim_offset: int, n_exp: int = 1, id: int = 7) -> Model:
    mdl = Model.__new__(Model)
    mdl.id = id
    mdl._sim_offset = sim_offset
    profiles = [np.array([[float(i + 1)] * 2], dtype=float)
                for i in range(n_exp)]
    mdl.sim = cast(Any, SimpleNamespace(
        settings={'experiments': [_exp() for _ in range(n_exp)]},
        profiles=profiles,
    ))
    return mdl


def _keys(sim_db: SIM_DB, table: str) -> set:
    return set(sim_db._upsert.get(table, {}).keys())


# ---------------------------------------------------------------------------
# save_sim experiment_id mapping
# ---------------------------------------------------------------------------
def test_save_sim_default_offset_parity(tmp_path: Path) -> None:
    sim_db = SIM_DB(name='TEST_OFF_DEFAULT', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')
    mdl = _model(sim_offset=0, n_exp=3)

    mdl.save_sim(db=sim_db, table='G0000', sim_num=0)
    mdl.save_sim(db=sim_db, table='G0000', sim_num=2)

    # No offset -> experiment_id equals sim_num.
    assert _keys(sim_db, 'G0000') == {(7, 0), (7, 2)}


def test_save_sim_applies_sim_offset(tmp_path: Path) -> None:
    sim_db = SIM_DB(name='TEST_OFF_APPLY', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')
    mdl = _model(sim_offset=5, n_exp=1)

    mdl.save_sim(db=sim_db, table='G0000', sim_num=0)

    assert _keys(sim_db, 'G0000') == {(7, 5)}


def test_save_sim_offset_formula_n_run_plus_band(tmp_path: Path) -> None:
    # Ensemble e=1 with n_run=3, n_pp=2 -> offset = 3 + 1*2 = 5.
    n_run, n_pp, e = 3, 2, 1
    sim_db = SIM_DB(name='TEST_OFF_FORMULA', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')
    mdl = _model(sim_offset=n_run + e * n_pp, n_exp=2)

    mdl.save_sim(db=sim_db, table='G0000', sim_num=0)
    mdl.save_sim(db=sim_db, table='G0000', sim_num=1)

    assert _keys(sim_db, 'G0000') == {(7, 5), (7, 6)}


def test_band_writes_leave_originals_untouched(tmp_path: Path) -> None:
    sim_db = SIM_DB(name='TEST_OFF_ORIG', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')

    original = _model(sim_offset=0, n_exp=1)
    original.save_sim(db=sim_db, table='G0000', sim_num=0)
    original_blob = sim_db._upsert['G0000'][(7, 0)]

    banded = _model(sim_offset=3, n_exp=1)
    banded.save_sim(db=sim_db, table='G0000', sim_num=0)

    # Original row is still present and unchanged; band lands on a new key.
    assert (7, 0) in sim_db._upsert['G0000']
    assert (7, 3) in sim_db._upsert['G0000']
    assert sim_db._upsert['G0000'][(7, 0)] == original_blob


def test_bands_are_disjoint(tmp_path: Path) -> None:
    n_run, n_pp = 3, 2
    sim_db = SIM_DB(name='TEST_OFF_DISJOINT', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')

    band0 = _model(sim_offset=n_run + 0 * n_pp, n_exp=2)  # -> 3, 4
    band1 = _model(sim_offset=n_run + 1 * n_pp, n_exp=2)  # -> 5, 6
    for sim_num in range(2):
        band0.save_sim(db=sim_db, table='G0000', sim_num=sim_num)
        band1.save_sim(db=sim_db, table='G0000', sim_num=sim_num)

    assert _keys(sim_db, 'G0000') == {(7, 3), (7, 4), (7, 5), (7, 6)}


def test_two_ensembles_same_table_no_collision(tmp_path: Path) -> None:
    sim_db = SIM_DB(name='TEST_OFF_NOCOLL', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')

    ens_a = _model(sim_offset=3, n_exp=1, id=7)
    ens_b = _model(sim_offset=5, n_exp=1, id=7)
    ens_a.save_sim(db=sim_db, table='G0000', sim_num=0)
    ens_b.save_sim(db=sim_db, table='G0000', sim_num=0)

    # Same table + same model id, different offset -> distinct rows.
    assert _keys(sim_db, 'G0000') == {(7, 3), (7, 5)}


# ---------------------------------------------------------------------------
# request_sim_profiles + round trip
# ---------------------------------------------------------------------------
def test_request_sim_profiles_applies_offset(tmp_path: Path) -> None:
    sim_db = SIM_DB(name='TEST_OFF_REQ', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')
    mdl = _model(sim_offset=5, n_exp=2)
    mdl.sim.profiles = [None, None]

    mdl.request_sim_profiles(sim_db=sim_db, table='G0000')

    assert sim_db._select['G0000'] == {(7, 5), (7, 6)}


def test_write_read_round_trip_with_offset(tmp_path: Path) -> None:
    sim_db = SIM_DB(name='TEST_OFF_RT', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')
    mdl = _model(sim_offset=5, n_exp=1)
    mdl.sim.profiles = [np.array([[1.0, 0.25]], dtype=float)]

    mdl.save_sim(db=sim_db, table='G0000', sim_num=0)
    sim_db.batch_upsert()
    sim_db.prepare_batch_select(table='G0000', mdl_id=7, experiment_id=5)
    out = sim_db.batch_select()

    # Decoded columns are [mdl_id, time, species...]; the offset key resolves.
    arr = out['G0000'][7][5]
    assert arr[:, 1].tolist() == [0.0, 1.0]
    assert arr[:, 2].tolist() == [1.0, 0.25]
