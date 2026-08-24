from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import kimeco.core as core_mod
from kimeco.core import CoreRun
from kimeco.enums import ModelStatus
from kimeco.q_sys import JobStatus


class _FakeRateCo:
    """Configurable RateCo stand-in installed via monkeypatch."""

    next_missing_grid: list = []
    next_status: JobStatus = JobStatus.NOT_IN_QUEUE
    last: Any = None

    def __init__(self, **kwargs) -> None:
        self.missing_grid = list(_FakeRateCo.next_missing_grid)
        self.status = _FakeRateCo.next_status
        self.q_up_called = False
        self.recover_called = False
        _FakeRateCo.last = self

    def set_status(self, table: str) -> None:
        # Real set_status populates missing_grid via is_in_db; the fake keeps
        # the configured value so the reuse branch can be exercised directly.
        return None

    def q_up(self) -> None:
        self.q_up_called = True

    def recover_rslts(self) -> None:
        # Reuse branch recovers persisted results instead of running MESS.
        self.recover_called = True


def _core(postprocess: bool) -> CoreRun:
    core = CoreRun.__new__(CoreRun)
    core.prefix = 'G'
    core.loc = '/tmp'
    core.base_dir = 'base'
    core.settings = {'postprocess': postprocess}
    core.rc_tpls = cast(Any, [])
    core.kin_db = cast(Any, object())
    core.qs = cast(Any, object())
    core.klog = cast(Any, SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        error=lambda *a, **k: None))
    return core


def _mdl(status=ModelStatus.SOP):
    return SimpleNamespace(
        id=0, gen=0, name='E0000', sop=object(),
        status=status, origin_prefix=None)


def test_pp_full_reuse_sets_kin_and_skips_mess(monkeypatch) -> None:
    monkeypatch.setattr(core_mod, 'RateCo', _FakeRateCo)
    _FakeRateCo.next_missing_grid = []
    _FakeRateCo.next_status = JobStatus.NOT_IN_QUEUE
    core = _core(postprocess=True)
    mdl = _mdl()

    core.calculate_rate_coefficients(cast(Any, mdl))

    # Full grid already persisted -> reuse and skip the MESS submission.
    # The reuse branch no longer re-parses on-disk MESS .out files
    # (recover_rslts); rates are rebuilt from KIN_DB at simulation time.
    assert mdl.status == ModelStatus.KIN
    assert _FakeRateCo.last.recover_called is False
    assert _FakeRateCo.last.q_up_called is False


def test_pp_partial_submits_missing_sub_grid(monkeypatch) -> None:
    monkeypatch.setattr(core_mod, 'RateCo', _FakeRateCo)
    _FakeRateCo.next_missing_grid = [(1.0, 400.0)]
    _FakeRateCo.next_status = JobStatus.NOT_IN_QUEUE
    core = _core(postprocess=True)
    mdl = _mdl()

    core.calculate_rate_coefficients(cast(Any, mdl))

    # Missing cells remain -> the reuse shortcut is skipped and MESS runs.
    assert mdl.status == ModelStatus.SOP
    assert _FakeRateCo.last.q_up_called is True


def test_normal_full_grid_finished_parity(monkeypatch) -> None:
    monkeypatch.setattr(core_mod, 'RateCo', _FakeRateCo)
    _FakeRateCo.next_missing_grid = []
    _FakeRateCo.next_status = JobStatus.FINISHED
    core = _core(postprocess=False)

    saved: list[str] = []
    mdl = _mdl()
    mdl.save_kin = lambda db, table: saved.append(table)

    core.calculate_rate_coefficients(cast(Any, mdl))

    # Normal run: no reuse shortcut, FINISHED persists rates via save_kin.
    assert saved == ['G0000']
    assert _FakeRateCo.last.q_up_called is False
