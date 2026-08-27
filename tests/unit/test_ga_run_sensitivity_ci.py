from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from kimeco.optimizers.GeneticAlgo.ga import GeneticAlgorithm
from kimeco.goat import GOATs


class _KlogSpy:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: str) -> None:
        self.messages.append(msg)


class _SfSpy:
    def __init__(self) -> None:
        self.set_active_p_calls: list[list[str]] = []

    def set_active_p(self, active_p: list[str]) -> None:
        self.set_active_p_calls.append(list(active_p))


class _GoatsSpy:
    def __init__(self) -> None:
        self.reset_calls: int = 0

    def reset(self) -> None:
        self.reset_calls += 1


def _fake_self(active_p: list[str], sa_selection: list[str]):
    """Build a fake GA self that uses the SA_restart branch (no Linear)."""
    gen_id = 3
    return SimpleNamespace(
        new_gen_has_been_created=True,
        settings={
            "active_p": active_p,
            "SA_restart": {str(gen_id): sa_selection},
        },
        sf=_SfSpy(),
        goats=_GoatsSpy(),
        klog=_KlogSpy(),
        # Only touched on the Linear branch, kept for attribute symmetry.
        goat=[],
        input_tpls=[["dummy"]],
        pert=SimpleNamespace(),
    ), gen_id


def test_run_sensitivity_appends_only_new_params() -> None:
    fs, gen_id = _fake_self(
        active_p=["A__we"],
        sa_selection=["A__we", "B__we"],
    )

    GeneticAlgorithm.run_sensitivity(cast(Any, fs), gen_id=gen_id)

    # Only the genuinely-new B__we is appended; A__we is not duplicated.
    assert fs.settings["active_p"] == ["A__we", "B__we"]


def test_run_sensitivity_new_param_sets_active_p_and_resets_goats() -> None:
    fs, gen_id = _fake_self(
        active_p=["A__we"],
        sa_selection=["A__we", "B__we"],
    )

    GeneticAlgorithm.run_sensitivity(cast(Any, fs), gen_id=gen_id)

    # set_active_p called once with the extended list.
    assert fs.sf.set_active_p_calls == [["A__we", "B__we"]]
    # goats.reset() called exactly once.
    assert fs.goats.reset_calls == 1


def test_run_sensitivity_no_new_param_is_noop() -> None:
    fs, gen_id = _fake_self(
        active_p=["A__we"],
        sa_selection=["A__we"],
    )

    GeneticAlgorithm.run_sensitivity(cast(Any, fs), gen_id=gen_id)

    # No new parameter -> no set_active_p, no reset, active_p unchanged.
    assert fs.sf.set_active_p_calls == []
    assert fs.goats.reset_calls == 0
    assert fs.settings["active_p"] == ["A__we"]


def test_goats_reset_clears_all_seen(tmp_path) -> None:
    goats = GOATs(
        sop_db=cast(Any, SimpleNamespace()),
        kin_db=cast(Any, SimpleNamespace()),
        sim_db=cast(Any, SimpleNamespace()),
        sf=cast(Any, SimpleNamespace()),
        wdir=str(tmp_path),
        overwrite=False,
    )
    goats.all_seen = {(0, 1): object(), (0, 2): object()}
    assert goats.all_seen

    goats.reset()

    assert goats.all_seen == {}
