from __future__ import annotations

from typing import Any

import pytest

from kimeco.default_settings import default_settings
from kimeco.kinmec import KiMec


class _FakeSolution:
    """Stand-in for ``cantera.Solution`` so ``KiMec.__init__`` can run without
    a real mechanism file. ``__init__`` reads the settings grid keys (the code
    under test) before touching the Solution, and only calls ``.species()`` and
    ``.reactions()`` on it, both of which we return empty.
    """

    def species(self) -> list[Any]:
        return []

    def reactions(self) -> list[Any]:
        return []


@pytest.fixture(autouse=True)
def _patch_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``file`` is a dummy string that is never opened because Solution is faked.
    monkeypatch.setattr(
        "kimeco.kinmec.ct.Solution",
        lambda file: _FakeSolution(),
    )


def test_init_without_rc_grid_does_not_raise_keyerror() -> None:
    # Regression: postprocess=False with no rc_pres/rc_temp used to raise
    # KeyError; the fix uses settings.get(..., []) so the grid defaults empty.
    settings = {**default_settings, "postprocess": False}
    settings.pop("rc_pres", None)
    settings.pop("rc_temp", None)

    kinmec = KiMec(file="mech.yaml", settings=settings)

    assert kinmec.pres == []
    assert kinmec.temp == []
    assert kinmec.rc_tpl == ""


def test_init_postprocess_false_preserves_provided_rc_grid() -> None:
    settings = {
        **default_settings,
        "postprocess": False,
        "rc_pres": [1.0, 2.0],
        "rc_temp": [300.0, 400.0, 500.0],
    }

    kinmec = KiMec(file="mech.yaml", settings=settings)

    assert kinmec.pres == [1.0, 2.0]
    assert kinmec.temp == [300.0, 400.0, 500.0]
    # 2 pressures x 3 temperatures -> 6 substitution fields.
    assert kinmec.rc_tpl.count("rc_") == 6
    assert "rc_0_0: {rates[0][0]}" in kinmec.rc_tpl
    assert "rc_1_2: {rates[1][2]}" in kinmec.rc_tpl


def test_init_postprocess_true_reads_pp_grid_and_ignores_missing_rc() -> None:
    settings = {
        **default_settings,
        "postprocess": True,
        "pp_pres": [1.0],
        "pp_temp": [300.0, 400.0],
    }
    settings.pop("rc_pres", None)
    settings.pop("rc_temp", None)

    kinmec = KiMec(file="mech.yaml", settings=settings)

    assert kinmec.pres == [1.0]
    assert kinmec.temp == [300.0, 400.0]


def test_init_postprocess_true_with_default_empty_pp_grid() -> None:
    settings = {**default_settings, "postprocess": True}

    kinmec = KiMec(file="mech.yaml", settings=settings)

    assert kinmec.pres == []
    assert kinmec.temp == []
    assert kinmec.rc_tpl == ""


def test_mechanism_section_style_construction_succeeds() -> None:
    # Mirrors the GUI mechanism-section trigger: build from default settings
    # with postprocess disabled and confirm species/reactions are readable.
    settings = {**default_settings}
    settings["postprocess"] = False

    kinmec = KiMec(file="mech.yaml", settings=settings)

    assert kinmec.species == []
    assert kinmec.reactions == []
