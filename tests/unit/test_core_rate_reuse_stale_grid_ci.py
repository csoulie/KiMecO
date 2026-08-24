from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import kimeco.core as core_mod
from kimeco.core import CoreRun
from kimeco.enums import ModelStatus
from kimeco.q_sys import JobStatus


class _FakeRateCo:
    """Configurable RateCo stand-in installed via monkeypatch.

    ``recover_raises`` emulates a reused GA project whose on-disk MESS ``.out``
    files sit on a *different* pressure grid: parsing them raises the stale
    "0.5 is not in list" ValueError. The fixed reuse branch must never call
    ``recover_rslts`` so this raise must never surface.
    """

    next_missing_grid: list = []
    next_status: JobStatus = JobStatus.NOT_IN_QUEUE
    recover_raises: bool = False
    last: Any = None

    def __init__(self, **kwargs) -> None:
        self.missing_grid = list(_FakeRateCo.next_missing_grid)
        self.status = _FakeRateCo.next_status
        self.q_up_called = False
        self.recover_called = False
        self.file_read = False
        _FakeRateCo.last = self

    def set_status(self, table: str) -> None:
        return None

    def q_up(self) -> None:
        self.q_up_called = True

    def recover_rslts(self) -> None:
        self.recover_called = True
        self.file_read = True
        if _FakeRateCo.recover_raises:
            raise ValueError('0.5 is not in list')


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


def _reset_fake():
    _FakeRateCo.next_missing_grid = []
    _FakeRateCo.next_status = JobStatus.NOT_IN_QUEUE
    _FakeRateCo.recover_raises = False
    _FakeRateCo.last = None


def test_reuse_branch_does_not_touch_stale_mess_out(monkeypatch) -> None:
    monkeypatch.setattr(core_mod, 'RateCo', _FakeRateCo)
    _reset_fake()
    _FakeRateCo.recover_raises = True  # stale .out grid would raise if parsed
    core = _core(postprocess=True)
    mdl = _mdl()

    # Pre-fix code called recover_rslts() and this would raise ValueError.
    core.calculate_rate_coefficients(cast(Any, mdl))

    assert mdl.status == ModelStatus.KIN
    assert _FakeRateCo.last.recover_called is False
    assert _FakeRateCo.last.q_up_called is False


def test_reuse_branch_ignores_grid_mismatch_no_file_reads(monkeypatch) -> None:
    monkeypatch.setattr(core_mod, 'RateCo', _FakeRateCo)
    _reset_fake()
    # Branch decides purely on empty missing_grid, independent of grid values
    # or disk state. recover_raises stays True to prove no file access happens.
    _FakeRateCo.recover_raises = True
    core = _core(postprocess=True)
    mdl = _mdl()

    core.calculate_rate_coefficients(cast(Any, mdl))

    assert mdl.status == ModelStatus.KIN
    assert _FakeRateCo.last.recover_called is False
    assert _FakeRateCo.last.file_read is False
    assert _FakeRateCo.last.q_up_called is False


def test_reuse_path_status_sop_to_kin(monkeypatch) -> None:
    monkeypatch.setattr(core_mod, 'RateCo', _FakeRateCo)
    _reset_fake()
    core = _core(postprocess=True)
    mdl = _mdl(status=ModelStatus.SOP)

    assert mdl.status == ModelStatus.SOP
    core.calculate_rate_coefficients(cast(Any, mdl))

    assert mdl.status == ModelStatus.KIN
    assert _FakeRateCo.last.q_up_called is False


def test_partial_grid_stays_sop_and_submits_mess(monkeypatch) -> None:
    monkeypatch.setattr(core_mod, 'RateCo', _FakeRateCo)
    _reset_fake()
    _FakeRateCo.next_missing_grid = [(1.0, 400.0)]
    _FakeRateCo.next_status = JobStatus.NOT_IN_QUEUE
    core = _core(postprocess=True)
    mdl = _mdl(status=ModelStatus.SOP)

    core.calculate_rate_coefficients(cast(Any, mdl))

    # Missing cells remain -> reuse shortcut skipped, MESS submitted.
    assert mdl.status == ModelStatus.SOP
    assert _FakeRateCo.last.q_up_called is True
    assert _FakeRateCo.last.recover_called is False


def test_run_simulation_transitions_kin_to_sim(monkeypatch) -> None:
    calls: dict[str, Any] = {'load': None, 'q_up': 0}

    class _StubSim:
        def __init__(self, **kwargs) -> None:
            pass

        def q_up(self) -> None:
            calls['q_up'] += 1

    monkeypatch.setattr(core_mod, 'SIM', _StubSim)

    core = _core(postprocess=True)
    core.sim_db = cast(Any, object())

    def _fake_gen_folder(mdl):
        return '/tmp'

    core.get_gen_folder = _fake_gen_folder  # type: ignore[assignment]

    def _load(table: str) -> None:
        calls['load'] = table

    rate_coef = SimpleNamespace(load_rates_from_db=_load)
    mdl = _mdl(status=ModelStatus.KIN)
    mdl.rateCoef = rate_coef

    core.run_simulation(cast(Any, mdl))

    assert calls['load'] == 'G0000'
    assert calls['q_up'] == 1
    assert mdl.status == ModelStatus.SIM
