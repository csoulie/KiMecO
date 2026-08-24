"""CI-safe tests for the automech branch of ``RateCo.create_input``.

``AutomechKinWriter`` is replaced with a capture double so no ``mess_io`` /
automech dependency is needed. Covers:
  * use_automech=True -> one ``{name}P{slot:02d}.py`` per PES slot, no ``.inp``;
  * postprocess=True + missing_grid -> the writer receives the sub-grid;
  * postprocess=False -> full grid (sub_p/sub_t None);
  * use_automech=False -> real ``MessWriter`` still writes ``{name}P00.inp``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kimeco.logger_config import KMOLogger
from kimeco.q_sys import JobStatus
from kimeco.rate_coef import RateCo


class _FakeSOP:
    def __init__(self, pes_ids=(0, 1)) -> None:
        self.pes_ids = list(pes_ids)
        # A real placeholder-free SOP surface for the MessWriter fallback.
        self.parameters_names: dict[str, str] = {}
        self.items: dict[str, Any] = {}

    def reaction_iterator(self):
        return iter([])


class _NoQueue:
    def status(self, id: int, jtype: str) -> JobStatus:
        return JobStatus.NOT_IN_QUEUE


class _FakeKinDB:
    def get_rates_for_kin_id(self, table, kin_id, pes_id=None):
        return []


def _settings(postprocess: bool, use_automech: bool | None) -> dict[str, Any]:
    s: dict[str, Any] = {
        'rc_software': 'mess',
        'postprocess': postprocess,
        'rc_pres': [1.0, 2.0],
        'rc_temp': [300.0, 400.0],
        'pp_pres': [1.0, 2.0],
        'pp_temp': [300.0, 400.0],
        'cpu_kin': 2,
        'mem_kin': 500,
    }
    if use_automech is not None:
        s['use_automech'] = use_automech
    return s


def _rateco(tmp_path: Path, settings: dict[str, Any], n_pes: int = 2,
            pes_ids=(0, 1)) -> RateCo:
    rc = RateCo(
        sop=cast(Any, _FakeSOP(pes_ids=pes_ids)),
        settings=settings,
        software_tpls=[['static line\n'] for _ in range(n_pes)],
        id=0,
        q_idx=1,
        name='G0000E0001',
        loc=str(tmp_path),
        q_sys=cast(Any, _NoQueue()),
        db=cast(Any, _FakeKinDB()),
        klog=KMOLogger(filename=str(tmp_path / 'rc.log')),
    )
    Path(rc.loc).mkdir(parents=True, exist_ok=True)
    return rc


class _CaptureAutomech:
    """Records constructor args and emitted filenames; writes a stub file."""

    def __init__(self, records: list[dict], writes: list[str]):
        self.records = records
        self.writes = writes

    def __call__(self, sop, pes_id, sub_p=None, sub_t=None, settings=None):
        self.records.append(
            {'pes_id': pes_id, 'sub_p': sub_p, 'sub_t': sub_t})
        parent = self

        def _write(loc, filename):
            parent.writes.append(filename)
            Path(loc, filename).write_text('# stub driver\n')

        return SimpleNamespace(write=_write)


def _patch_automech(monkeypatch) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    writes: list[str] = []
    monkeypatch.setattr(
        'kimeco.rate_coef.AutomechKinWriter',
        _CaptureAutomech(records, writes))
    return records, writes


def _forbid_messwriter(monkeypatch) -> None:
    def _boom(*a, **k):
        raise AssertionError('MessWriter must not be used with use_automech')
    monkeypatch.setattr('kimeco.rate_coef.MessWriter', _boom)


# ---------------------------------------------------------------------------
# use_automech=True emission
# ---------------------------------------------------------------------------
def test_automech_emits_one_py_per_slot_no_inp(
        tmp_path: Path, monkeypatch) -> None:
    records, writes = _patch_automech(monkeypatch)
    _forbid_messwriter(monkeypatch)
    rc = _rateco(tmp_path, _settings(postprocess=False, use_automech=True))

    rc.create_input()

    assert writes == ['G0000E0001P00.py', 'G0000E0001P01.py']
    assert [r['pes_id'] for r in records] == [0, 1]
    # No MESS .inp files are produced on the automech path.
    assert list(Path(rc.loc).glob('*.inp')) == []


def test_automech_normal_run_passes_full_grid(
        tmp_path: Path, monkeypatch) -> None:
    records, _ = _patch_automech(monkeypatch)
    _forbid_messwriter(monkeypatch)
    rc = _rateco(tmp_path, _settings(postprocess=False, use_automech=True))

    rc.create_input()

    # No override in a normal run: the writer uses the SOP grid.
    assert all(r['sub_p'] is None and r['sub_t'] is None for r in records)


def test_automech_postprocess_passes_sub_grid(
        tmp_path: Path, monkeypatch) -> None:
    records, _ = _patch_automech(monkeypatch)
    _forbid_messwriter(monkeypatch)
    rc = _rateco(tmp_path, _settings(postprocess=True, use_automech=True))
    # A single missing (P,T) cell -> strict sub-grid.
    rc.missing_grid = [(2.0, 300.0)]

    rc.create_input()

    assert all(r['sub_p'] == [2.0] and r['sub_t'] == [300.0]
               for r in records)


# ---------------------------------------------------------------------------
# use_automech=False fallback still uses the real MessWriter
# ---------------------------------------------------------------------------
def test_use_automech_false_writes_inp_via_real_messwriter(
        tmp_path: Path, monkeypatch) -> None:
    # AutomechKinWriter must never be constructed on the classic path.
    def _boom(*a, **k):
        raise AssertionError('AutomechKinWriter used with use_automech=False')
    monkeypatch.setattr('kimeco.rate_coef.AutomechKinWriter', _boom)

    rc = _rateco(
        tmp_path,
        _settings(postprocess=False, use_automech=False),
        n_pes=1, pes_ids=(0,))

    rc.create_input()

    inp = Path(rc.loc) / 'G0000E0001P00.inp'
    assert inp.exists()
    assert 'static line' in inp.read_text()
    assert list(Path(rc.loc).glob('*.py')) == []
