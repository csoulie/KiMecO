from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest
from dash import html
from dash.exceptions import PreventUpdate

from kimeco.experiments.t_profile import TimeProfile
from kimeco.gui import simsection as simsection_mod
from kimeco.gui.simsection import SIMSection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_time_profile(temp, pres, species, data=None):
    """Build a minimal real TimeProfile (klog stored, never called)."""
    arr = np.zeros((len(species) + 1, 1), dtype=float) if data is None \
        else np.asarray(data, dtype=float)
    return TimeProfile(
        temp=temp,
        pres=pres,
        composition={sp: 1.0 for sp in species},
        data_file='data.csv',
        error_file='error.csv',
        sim_file='sim.inp',
        settings={},
        klog=None,          # type: ignore[arg-type]
        species=list(species),
        data=arr,
        error=np.zeros_like(arr),
    )


def _bare_section(**attrs) -> SIMSection:
    section = SIMSection.__new__(SIMSection)
    section.settings = {'pres_unit': 'bar'}
    section.experiments = []
    section.pp_experiments = []
    section.pp_tables = []
    for key, value in attrs.items():
        setattr(section, key, value)
    return section


def _capture_callbacks(monkeypatch, section: SIMSection) -> dict[str, Any]:
    callbacks: dict[str, Any] = {}

    def _fake_callback(*_args, **_kwargs):
        def _decorator(func):
            callbacks[func.__name__] = func
            return func

        return _decorator

    monkeypatch.setattr(simsection_mod, 'callback', _fake_callback)
    section.register_callbacks()
    return callbacks


# ---------------------------------------------------------------------------
# _experiment_label
# ---------------------------------------------------------------------------
def test_experiment_label_time_profile_contains_all_fields() -> None:
    section = _bare_section()
    exp = _make_time_profile(temp=300, pres=101325.0, species=['A', 'B'])

    label = section._experiment_label(exp, 2, section.settings)

    assert exp.exp_type in label
    assert '#2' in label
    assert '1.013 bar' in label
    assert '300 K' in label
    assert 'A' in label and 'B' in label
    assert label == (
        f'{exp.exp_type} #2 — 1.013 bar, 300 K — A, B'
    )


def test_experiment_label_time_profile_formatting_edges() -> None:
    section = _bare_section()
    exp = _make_time_profile(temp=1000.0, pres=101325.0, species=['ONLY'])

    label = section._experiment_label(exp, 0, section.settings)

    # :g must drop the trailing .0 on an integral temperature.
    assert '1000 K' in label
    assert '1000.0 K' not in label
    # Single species -> exactly that token after the em dash.
    assert label.endswith('— ONLY')


def test_experiment_label_generic_fallback() -> None:
    section = _bare_section()
    exp = SimpleNamespace(
        exp_type='Custom',
        P=555.0,
        T=777.0,
        species=['LEAK'],
    )

    label = section._experiment_label(exp, 4, section.settings)

    assert label == 'Custom #4'
    for leaked in ('555', '777', 'LEAK', 'bar', 'K'):
        assert leaked not in label


# ---------------------------------------------------------------------------
# _experiment_options
# ---------------------------------------------------------------------------
def test_experiment_options_regular_uses_experiments() -> None:
    exps = [
        SimpleNamespace(exp_type='E', P=1.0, T=1.0, species=['A']),
        SimpleNamespace(exp_type='E', P=2.0, T=2.0, species=['B']),
    ]
    section = _bare_section(experiments=exps)

    options = section._experiment_options('REG')

    assert [o['value'] for o in options] == [0, 1]
    for i, opt in enumerate(options):
        assert opt['label'] == section._experiment_label(
            exps[i], i, section.settings)


def test_experiment_options_pp_uses_pp_experiments() -> None:
    reg = [SimpleNamespace(exp_type='R', P=1.0, T=1.0, species=['A'])]
    pp = [
        SimpleNamespace(exp_type='P', P=1.0, T=1.0, species=['C']),
        SimpleNamespace(exp_type='P', P=2.0, T=2.0, species=['D']),
    ]
    section = _bare_section(experiments=reg, pp_experiments=pp)

    options = section._experiment_options('PP')

    assert [o['value'] for o in options] == [0, 1]
    assert options[0]['label'] == section._experiment_label(
        pp[0], 0, section.settings)


def test_experiment_options_empty() -> None:
    section = _bare_section(experiments=[], pp_experiments=[])
    assert section._experiment_options('REG') == []
    assert section._experiment_options('PP') == []


# ---------------------------------------------------------------------------
# update_sim_source callback
# ---------------------------------------------------------------------------
def test_update_sim_source_regular(monkeypatch) -> None:
    exps = [SimpleNamespace(exp_type='E', P=1.0, T=1.0, species=['A'])]
    section = _bare_section(experiments=exps, pp_tables=['G0000'])
    callbacks = _capture_callbacks(monkeypatch, section)

    options, style, pp_options = callbacks['update_sim_source']('REG')

    assert options == section._experiment_options('REG')
    assert style == {'display': 'none'}
    assert pp_options == section.pp_tables


def test_update_sim_source_postprocessing(monkeypatch) -> None:
    pp = [SimpleNamespace(exp_type='P', P=1.0, T=1.0, species=['C'])]
    section = _bare_section(pp_experiments=pp, pp_tables=['G0000', 'G0001'])
    callbacks = _capture_callbacks(monkeypatch, section)

    options, style, pp_options = callbacks['update_sim_source']('PP')

    assert options == section._experiment_options('PP')
    assert style == {'display': 'block'}
    assert pp_options == section.pp_tables


# ---------------------------------------------------------------------------
# show_sim_plot_button callback
# ---------------------------------------------------------------------------
def test_show_sim_plot_button_regular_selected(monkeypatch) -> None:
    section = _bare_section()
    callbacks = _capture_callbacks(monkeypatch, section)

    assert callbacks['show_sim_plot_button']('REG', [0], None) == {
        'display': 'block'
    }


@pytest.mark.parametrize('experiments', [None, []])
def test_show_sim_plot_button_empty_prevents(monkeypatch, experiments) -> None:
    section = _bare_section()
    callbacks = _capture_callbacks(monkeypatch, section)

    with pytest.raises(PreventUpdate):
        callbacks['show_sim_plot_button']('REG', experiments, None)


def test_show_sim_plot_button_pp_requires_tables(monkeypatch) -> None:
    section = _bare_section()
    callbacks = _capture_callbacks(monkeypatch, section)
    show = callbacks['show_sim_plot_button']

    with pytest.raises(PreventUpdate):
        show('PP', [0], None)

    assert show('PP', [0], ['G0000']) == {'display': 'block'}


# ---------------------------------------------------------------------------
# update_sim_figure callback
# ---------------------------------------------------------------------------
def test_update_sim_figure_regular_one_call_per_exp_species(
    monkeypatch,
) -> None:
    exps = [
        SimpleNamespace(exp_type='E', P=101325.0, T=300.0, species=['A']),
        SimpleNamespace(exp_type='E', P=101325.0, T=300.0, species=['B']),
    ]
    section = _bare_section(experiments=exps)
    section.sim_db = cast(Any, SimpleNamespace(sv_species=['A', 'B']))

    reg_calls: list[dict] = []
    fig_calls: list[dict] = []

    def _fake_regular(selected_gen, experiment_id):
        reg_calls.append(
            {'selected_gen': selected_gen, 'experiment_id': experiment_id})
        return [{'G0000': {}}]

    def _fake_make_figure(**kwargs):
        fig_calls.append(kwargs)
        return []

    monkeypatch.setattr(
        section, 'get_regular_condition_profiles', _fake_regular)
    monkeypatch.setattr(section, 'make_figure', _fake_make_figure)

    callbacks = _capture_callbacks(monkeypatch, section)
    style, _children = callbacks['update_sim_figure'](
        1, 'REG', [0, 1], None, [0])

    assert style == {'display': 'block'}
    # One make_figure per (experiment, measured species).
    assert len(fig_calls) == 2
    assert {(c['experiment_id'], c['sp']) for c in fig_calls} == {
        (0, 'A'), (1, 'B')}
    # experiment_id threaded through to the DB query.
    assert {c['experiment_id'] for c in reg_calls} == {0, 1}


def test_update_sim_figure_pp_pressure_and_temp_index(monkeypatch) -> None:
    pp = [SimpleNamespace(exp_type='P', P=101325.0, T=300.0, species=['A'])]
    section = _bare_section(pp_experiments=pp, pp_tables=['G0000'])
    section.pp_sim_db = cast(Any, SimpleNamespace(sv_species=['A']))
    section.settings.update({'pp_pres': [1.01325], 'pp_temp': [300.0]})

    pp_calls: list[dict] = []
    fig_calls: list[dict] = []

    def _fake_pp(tables, p_idx, t_idx):
        pp_calls.append(
            {'tables': tables, 'p_idx': p_idx, 't_idx': t_idx})
        return {'G0000': {}}

    monkeypatch.setattr(section, 'get_pp_condition_profiles', _fake_pp)
    monkeypatch.setattr(
        section, 'make_figure', lambda **kw: fig_calls.append(kw) or [])

    callbacks = _capture_callbacks(monkeypatch, section)
    callbacks['update_sim_figure'](1, 'PP', [0], ['G0000'], None)

    assert len(pp_calls) == 1
    # p_idx from settings['pp_pres'].index(P_bar); t_idx from pp_temp.index(T).
    assert pp_calls[0]['p_idx'] == 0
    assert pp_calls[0]['t_idx'] == 0
    assert len(fig_calls) == 1
    assert fig_calls[0]['experiment_id'] == 0


def test_update_sim_figure_empty_selection_prevents(monkeypatch) -> None:
    section = _bare_section(experiments=[])
    fig_calls: list[dict] = []
    monkeypatch.setattr(
        section, 'make_figure', lambda **kw: fig_calls.append(kw) or [])

    callbacks = _capture_callbacks(monkeypatch, section)
    with pytest.raises(PreventUpdate):
        callbacks['update_sim_figure'](1, 'REG', [], None, [0])

    assert fig_calls == []


# ---------------------------------------------------------------------------
# make_figure heading
# ---------------------------------------------------------------------------
def test_make_figure_heading_contains_species_and_experiment() -> None:
    exp = _make_time_profile(temp=300, pres=101325.0, species=['A'])
    section = _bare_section(pp_experiments=[exp])
    sim_db = cast(Any, SimpleNamespace(sv_species=['A']))

    rendered = section.make_figure(
        gen_name='G0000',
        TPGenSP={},
        experiment_id=0,
        sp='A',
        sim_db=sim_db,
        show_exp_profile=False,
    )

    heading = rendered[0]
    assert isinstance(heading, html.H3)
    exp_label = section._experiment_label(exp, 0, section.settings)
    assert heading.children == f'A — {exp_label} — G0000'


def test_make_figure_species_absent_returns_bare_figure() -> None:
    exp = _make_time_profile(temp=300, pres=101325.0, species=['B'])
    section = _bare_section(pp_experiments=[exp])
    sim_db = cast(Any, SimpleNamespace(sv_species=['B']))

    rendered = section.make_figure(
        gen_name='G0000',
        TPGenSP={},
        experiment_id=0,
        sp='A',
        sim_db=sim_db,
        show_exp_profile=False,
    )

    assert len(rendered) == 1
    assert isinstance(rendered[0], go.Figure)
