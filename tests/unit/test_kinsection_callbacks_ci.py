from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from kimeco.gui import kinsection as kinsection_mod
from kimeco.gui.kinsection import KINSection


def _make_section() -> KINSection:
    section = KINSection.__new__(KINSection)
    section.settings = {
        "pres_unit": "atm",
        "rc_pres": [1.0],
        "rc_temp": [300.0],
    }
    section.init_SOP = cast(
        Any,
        SimpleNamespace(
            wells=[
                SimpleNamespace(name="A", pes_ids=[0, 1]),
                SimpleNamespace(name="B", pes_ids=[0]),
            ],
            bimolecular=[
                SimpleNamespace(name="A+H", pes_ids=[0]),
                SimpleNamespace(name="B+H", pes_ids=[1]),
            ],
            # Fragments may exist on SOP but are intentionally not
            # selectable in KIN section options.
            fragments=[SimpleNamespace(name="A_FRAG", pes_ids=[0])],
            pes_ids=[0, 1],
        ),
    )
    return section


def _capture_callbacks(monkeypatch, section: KINSection) -> dict[str, Any]:
    callbacks: dict[str, Any] = {}

    def _fake_callback(*_args, **_kwargs):
        def _decorator(func):
            callbacks[func.__name__] = func
            return func

        return _decorator

    monkeypatch.setattr(kinsection_mod, "callback", _fake_callback)
    section.register_callbacks()
    return callbacks


def _pes_ids(options: list[dict[str, str]], section: KINSection) -> set[int]:
    out: set[int] = set()
    for opt in options:
        parsed = section._decode_rc_option(opt["value"])
        if parsed is None:
            continue
        out.add(parsed[1])
    return out


def test_all_rc_options_excludes_fragments_from_source_model() -> None:
    section = _make_section()

    options = section._all_rc_options()

    labels = [opt["label"] for opt in options]
    values = [opt["value"] for opt in options]

    assert "A_FRAG [PES 00]" not in labels
    assert section._encode_rc_option("A_FRAG", 0) not in values


def test_label_format_includes_zero_padded_pes_id() -> None:
    assert KINSection._label_rc_option("WELL_A", 3) == "WELL_A [PES 03]"


def test_sync_options_filters_opposite_to_same_pes(monkeypatch) -> None:
    section = _make_section()
    callbacks = _capture_callbacks(monkeypatch, section)
    sync = callbacks["sync_rc_from_to_options"]

    rc_from = [section._encode_rc_option("A", 0)]
    rc_to: list[str] = []

    frm_options, frm_values, to_options, to_values = sync(rc_from, rc_to)

    assert frm_values == rc_from
    assert to_values == []
    assert _pes_ids(frm_options, section) == {0}
    assert _pes_ids(to_options, section) == {0}


def test_sync_options_auto_clears_cross_pes(monkeypatch) -> None:
    section = _make_section()
    callbacks = _capture_callbacks(monkeypatch, section)
    sync = callbacks["sync_rc_from_to_options"]

    rc_from = [section._encode_rc_option("A", 0)]
    rc_to = [section._encode_rc_option("B+H", 1)]

    frm_options, frm_values, to_options, to_values = sync(rc_from, rc_to)

    # Conflicting same-name selections across different PES IDs are
    # invalid after reciprocal filtering and must be cleared.
    assert frm_values == []
    assert to_values == []
    assert _pes_ids(frm_options, section) == {1}
    assert _pes_ids(to_options, section) == {0}


def test_sync_options_clears_stale_fragment_value(monkeypatch) -> None:
    section = _make_section()
    callbacks = _capture_callbacks(monkeypatch, section)
    sync = callbacks["sync_rc_from_to_options"]

    stale_fragment_value = section._encode_rc_option("A_FRAG", 0)

    _, frm_values, _, to_values = sync(
        [stale_fragment_value],
        [stale_fragment_value],
    )

    assert frm_values == []
    assert to_values == []
