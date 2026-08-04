from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import kimeco.gui.sopsection as sopsection_mod
from kimeco.gui.sopsection import SOPSection
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


class FakeHistogram:
    """Records constructor args so the plot path can be inspected."""

    instances: list["FakeHistogram"] = []

    def __init__(self,
                 data: dict[int, Any],
                 settings: dict[str, Any],
                 histfunc: str = "count") -> None:
        self.data = data
        self.settings = settings
        self.vlines: list[dict[str, Any]] = []
        FakeHistogram.instances.append(self)

    def add_vline(self, **kwargs: Any) -> None:
        self.vlines.append(kwargs)

    def layout(self) -> list["FakeHistogram"]:
        return [self]


@pytest.fixture(autouse=True)
def _reset_fake_histogram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeHistogram.instances = []
    monkeypatch.setattr(sopsection_mod, 'Histogram', FakeHistogram)


class FakeApp:
    """Captures Dash callbacks by function name at registration."""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}

    def callback(self, *_args: Any, **_kwargs: Any) -> Any:
        def _decorator(fn: Any) -> Any:
            self.captured[fn.__name__] = fn
            return fn
        return _decorator


class SpyGoats:
    def __init__(self, per_gen: dict[int, dict[str, np.ndarray]]) -> None:
        self._per_gen = per_gen
        self.param_calls: list[tuple[int, list[str]]] = []
        self.goat_for_gen_calls = 0
        self.fscore_calls = 0

    def get_goat_param_values(
        self, gen: int, cols: list[str],
    ) -> dict[str, np.ndarray]:
        self.param_calls.append((gen, list(cols)))
        return {col: self._per_gen[gen][col] for col in cols}

    def get_goat_for_gen(self, gen: int) -> Any:
        self.goat_for_gen_calls += 1
        raise AssertionError('get_goat_for_gen must not run on the plot path')

    def fscore(self, mdl: Any) -> None:
        self.fscore_calls += 1
        raise AssertionError('fscore must not run on the plot path')


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


def _make_real_goats(tmp_path: Path,
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


def _make_section(columns: list[str], goats: Any) -> SOPSection:
    """Build a SOPSection without running __init__ and inject collaborators."""
    sec = SOPSection.__new__(SOPSection)
    sec.app = cast(Any, FakeApp())
    sec.sop_db = cast(Any, SimpleNamespace(columns=list(columns)))
    # init_vals is 1-indexed against columns (index 0 is reserved for id).
    init_vals = [0.0] + [0.0] * len(columns)
    sec.gapp = cast(Any, SimpleNamespace(goats=goats, init_vals=init_vals))
    # Boundaries and plot options are unchanged by the feature; stub them so
    # the test focuses on the data-fetch/plot wiring.
    sec.get_boundaries = cast(Any, lambda col: [0.0, 0.0])
    sec.get_plot_options = cast(Any, lambda col, ptypes: {
        'title': col, 'tickformat': '.2f', 'unit': ''})
    return sec


def _capture_update_sop_figure(sec: SOPSection) -> Any:
    sec.register_callbacks()
    return cast(Any, sec.app).captured['update_sop_figure']


def test_make_figure_passes_gen_keyed_dict_to_histogram() -> None:
    columns = ['__fact', '__pow', 'exp_000__score']
    sec = _make_section(columns, goats=SimpleNamespace())

    per_gen_values = {
        0: {'__fact': np.array([1.1, 1.2])},
        1: {'__fact': np.array([1.5, 1.6])},
    }
    children = sec.make_figure(
        boundaries=[0.0, 0.0],
        plot_settings={'title': '__fact', 'tickformat': '.2f', 'unit': ''},
        col='__fact',
        per_gen_values=per_gen_values,
    )

    assert len(FakeHistogram.instances) == 1
    hist = FakeHistogram.instances[0]
    assert set(hist.data.keys()) == {0, 1}
    assert all(isinstance(k, int) for k in hist.data)
    assert np.array_equal(hist.data[0], np.array([1.1, 1.2]))
    assert np.array_equal(hist.data[1], np.array([1.5, 1.6]))
    # make_figure returns the histogram layout list.
    assert children == [hist]


def test_update_sop_figure_fetches_once_per_generation() -> None:
    columns = ['__fact', '__pow', 'exp_000__score']
    per_gen = {
        0: {c: np.array([1.0, 2.0]) for c in columns},
        1: {c: np.array([3.0, 4.0]) for c in columns},
    }
    goats = SpyGoats(per_gen)
    sec = _make_section(columns, goats=goats)
    update_sop_figure = _capture_update_sop_figure(sec)

    style, children = update_sop_figure(1, ['fact'], columns, [0, 1])

    assert style == {'display': 'block'}
    # One fetch per generation with the full column list (not per column).
    assert goats.param_calls == [(0, columns), (1, columns)]
    # One figure produced per requested column.
    assert len(FakeHistogram.instances) == len(columns)
    assert len(children) == len(columns)


def test_update_sop_figure_single_column_path() -> None:
    columns = ['__fact', '__pow']
    per_gen = {0: {c: np.array([1.0, 2.0]) for c in columns}}
    goats = SpyGoats(per_gen)
    sec = _make_section(columns, goats=goats)
    update_sop_figure = _capture_update_sop_figure(sec)

    _style, children = update_sop_figure(1, ['fact'], ['__fact'], [0])

    assert goats.param_calls == [(0, ['__fact'])]
    assert len(FakeHistogram.instances) == 1
    assert len(children) == 1


def test_plot_path_does_not_call_get_goat_for_gen_or_fscore() -> None:
    columns = ['__fact']
    per_gen = {0: {'__fact': np.array([1.0, 2.0])}}
    goats = SpyGoats(per_gen)
    sec = _make_section(columns, goats=goats)
    update_sop_figure = _capture_update_sop_figure(sec)

    update_sop_figure(1, ['fact'], ['__fact'], [0])

    assert goats.goat_for_gen_calls == 0
    assert goats.fscore_calls == 0


def test_histogram_data_keys_are_int_and_values_are_sequences() -> None:
    columns = ['__fact']
    per_gen = {
        0: {'__fact': np.array([1.0, 2.0, 3.0])},
        2: {'__fact': np.array([4.0, 5.0])},
    }
    goats = SpyGoats(per_gen)
    sec = _make_section(columns, goats=goats)
    update_sop_figure = _capture_update_sop_figure(sec)

    update_sop_figure(1, ['fact'], ['__fact'], [0, 2])

    hist = FakeHistogram.instances[0]
    assert set(hist.data.keys()) == {0, 2}
    assert all(isinstance(k, int) for k in hist.data)
    for values in hist.data.values():
        assert len(values) >= 1


def test_score_values_are_byte_identical_to_persisted_scores(
    tmp_path: Path,
) -> None:
    sop = _build_sop()
    db = _make_db(tmp_path, sop)
    goats = _make_real_goats(tmp_path, db, [GEN1])

    sec = _make_section(db.columns, goats=goats)
    update_sop_figure = _capture_update_sop_figure(sec)

    col = 'exp_000__score'
    update_sop_figure(1, ['score'], [col], [0])

    expected = np.array([ROWS[f'G{g:04d}'][m]['s0'] for (g, m) in GEN1])
    hist = FakeHistogram.instances[0]
    plotted = hist.data[0]
    assert np.array_equal(plotted, expected)
    assert plotted.tobytes() == expected.tobytes()
