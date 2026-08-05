from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from kimeco.database.sim_db import SIM_DB
from kimeco.gui.simsection import SIMSection


def _blob_from_dict(data: dict[str, list[float]]) -> bytes:
    table = pa.table(data)
    sink = pa.BufferOutputStream()
    feather.write_feather(table, sink)
    return sink.getvalue().to_pybytes()


def _pp_exp(pres=101325.0, temp=300.0, species=('A',)):
    return SimpleNamespace(
        exp_type='PP', P=pres, T=temp, species=list(species))


def test_make_figure_uses_nested_sim_db_results_for_species_trace() -> None:
    section = SIMSection.__new__(SIMSection)
    exp = _pp_exp(species=['A'])
    # Banded id -> show_exp_profile is False, so only the sim trace is drawn.
    section.experiments = [exp]
    section.pp_experiments = [exp]
    section.n_run = 1
    section.n_pp = 1
    section.settings = {
        'pres_unit': 'bar',
        'n_run_exp': 1,
        'pp_ensembles': ['G0000'],
        'experiments': [exp],
    }
    section.sim_db = cast(Any, SimpleNamespace(sv_species=['A']))

    rows = np.array([[7.0, 0.0, 1.0], [7.0, 1.0, 0.25]], dtype=float)
    rendered = section.make_figure(
        gen_name='G0000',
        TPGenSP={'G0000': {7: {1: rows}}},
        experiment_id=1,
        sp='A',
    )

    fig = cast(Any, rendered[-1]).figure

    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [0.0, 1.0]
    assert list(fig.data[0].y) == [1.0, 0.25]


def test_get_regular_condition_profiles_filters_by_experiment_id(
    tmp_path: Path,
) -> None:
    sim_db = SIM_DB(name='TEST_REG_SIM_DB', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')
    sim_db.prepare_batch_upsert(
        table='G0000', mdl_id=5, experiment_id=0,
        result=_blob_from_dict({'time': [0.0], 'A': [1.0]}))
    sim_db.prepare_batch_upsert(
        table='G0000', mdl_id=5, experiment_id=1,
        result=_blob_from_dict({'time': [0.0], 'A': [2.0]}))
    sim_db.batch_upsert()

    section = SIMSection.__new__(SIMSection)
    section.sim_db = sim_db
    section.gapp = cast(
        Any,
        SimpleNamespace(
            goats=SimpleNamespace(generations={0: [(0, 5)]}),
        ),
    )

    out = section.get_regular_condition_profiles(
        selected_gen=[0],
        experiment_id=1,
    )

    assert len(out) == 1
    # Only the requested experiment_id blob is returned.
    assert sorted(out[0]['G0000'][5]) == [1]
    assert out[0]['G0000'][5][1][0, 2] == 2.0


def test_get_regular_condition_profiles_missing_experiment_id(
    tmp_path: Path,
) -> None:
    sim_db = SIM_DB(name='TEST_REG_SIM_DB_MISS', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')
    sim_db.prepare_batch_upsert(
        table='G0000', mdl_id=5, experiment_id=0,
        result=_blob_from_dict({'time': [0.0], 'A': [1.0]}))
    sim_db.batch_upsert()

    section = SIMSection.__new__(SIMSection)
    section.sim_db = sim_db
    section.gapp = cast(
        Any,
        SimpleNamespace(
            goats=SimpleNamespace(generations={0: [(0, 5)]}),
        ),
    )

    # No rows for experiment_id=9 -> no raise, empty per-gen mapping.
    out = section.get_regular_condition_profiles(
        selected_gen=[0],
        experiment_id=9,
    )

    assert len(out) == 1
    assert out[0].get('G0000', {}) == {}


def test_get_regular_condition_profiles_with_offset_banded_id(
    tmp_path: Path,
) -> None:
    """A banded experiment_id selects the offset row written by that band."""
    sim_db = SIM_DB(name='TEST_REG_SIM_DB_OFF', path=str(tmp_path), threads=1)
    sim_db.create_new_table(name='G0000')
    sim_db.prepare_batch_upsert(
        table='G0000', mdl_id=5, experiment_id=0,
        result=_blob_from_dict({'time': [0.0], 'A': [1.0]}))
    sim_db.prepare_batch_upsert(
        table='G0000', mdl_id=5, experiment_id=4,
        result=_blob_from_dict({'time': [0.0], 'A': [9.0]}))
    sim_db.batch_upsert()

    section = SIMSection.__new__(SIMSection)
    section.sim_db = sim_db
    section.gapp = cast(
        Any,
        SimpleNamespace(
            goats=SimpleNamespace(generations={0: [(0, 5)]}),
        ),
    )

    out = section.get_regular_condition_profiles(
        selected_gen=[0],
        experiment_id=4,
    )

    assert sorted(out[0]['G0000'][5]) == [4]
    assert out[0]['G0000'][5][4][0, 2] == 9.0
