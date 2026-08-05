from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from kimeco.core import CoreRun


class _FakeDB:
    """Minimal stand-in exposing the tables API used by create_tables."""

    def __init__(self) -> None:
        self.tables: dict[str, object] = {}
        self.created: list[str] = []

    def create_new_table(self, name: str) -> None:
        self.tables[name] = object()
        self.created.append(name)


def _core(prefix: str = 'G', models: list | None = None) -> CoreRun:
    core = CoreRun.__new__(CoreRun)
    core.prefix = prefix
    core.models = cast(Any, models or [])
    core.sop_db = cast(Any, _FakeDB())
    core.kin_db = cast(Any, _FakeDB())
    core.sim_db = cast(Any, _FakeDB())
    return core


def _mdl(gen: int, id: int = 0, origin_prefix=None):
    return SimpleNamespace(gen=gen, id=id, origin_prefix=origin_prefix)


# ---------------------------------------------------------------------------
# get_table_name
# ---------------------------------------------------------------------------
def test_get_table_name_normal_run_parity() -> None:
    core = _core(prefix='G')

    assert core.get_table_name(_mdl(gen=0)) == 'G0000'
    assert core.get_table_name(_mdl(gen=2)) == 'G0002'


def test_get_table_name_uses_origin_prefix_when_set() -> None:
    core = _core(prefix='G')

    assert core.get_table_name(
        _mdl(gen=3, origin_prefix='NM')) == 'NM0003'
    # origin_prefix='G' is the normal-run value and still routes to G tables.
    assert core.get_table_name(
        _mdl(gen=1, origin_prefix='G')) == 'G0001'


def test_get_table_name_never_x_or_gt() -> None:
    core = _core(prefix='G')

    for mdl in (_mdl(gen=0), _mdl(gen=1, origin_prefix='NM')):
        name = core.get_table_name(mdl)
        assert not name.startswith('X')
        assert not name.startswith('GT')


# ---------------------------------------------------------------------------
# create_tables
# ---------------------------------------------------------------------------
def test_create_tables_mixed_prefix_creates_distinct_tables() -> None:
    models = [
        _mdl(gen=0, id=0, origin_prefix=None),   # G0000
        _mdl(gen=0, id=1, origin_prefix='NM'),   # NM0000
        _mdl(gen=1, id=2, origin_prefix=None),   # G0001
    ]
    core = _core(prefix='G', models=models)

    core.create_tables()

    expected = {'G0000', 'NM0000', 'G0001'}
    assert set(core.sop_db.created) == expected
    assert set(core.kin_db.created) == expected
    assert set(core.sim_db.created) == expected


def test_create_tables_normal_run_parity_no_x_or_gt() -> None:
    models = [_mdl(gen=0, id=0), _mdl(gen=1, id=1)]
    core = _core(prefix='G', models=models)

    core.create_tables()

    assert set(core.sop_db.created) == {'G0000', 'G0001'}
    assert not any(
        name.startswith(('X', 'GT')) for name in core.sop_db.created)
