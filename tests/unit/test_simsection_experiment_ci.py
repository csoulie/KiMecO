from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest
from dash import html
from dash.exceptions import PreventUpdate

from kimeco.gui import simsection as simsection_mod
from kimeco.gui.simsection import SIMSection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _exp(exp_type='Time profile', pres=101325.0, temp=300.0, species=('A',)):
    n = len(species)
    return SimpleNamespace(
        exp_type=exp_type,
        P=pres,
        T=temp,
        species=list(species),
        data=np.array([[0.0, 1.0]] + [[0.1, 0.2]] * n),
        error=np.array([[0.0, 0.0]] + [[0.01, 0.01]] * n),
    )


def _bare_section(experiments=None,
                  pp_experiments=None,
                  n_run=None,
                  pp_ensembles=None,
                  **attrs) -> SIMSection:
    section = SIMSection.__new__(SIMSection)
    section.experiments = experiments if experiments is not None else []
    section.pp_experiments = (
        pp_experiments if pp_experiments is not None else [])
    section.n_run = n_run if n_run is not None else len(section.experiments)
    section.n_pp = len(section.pp_experiments)
    section.settings = {
        'pres_unit': 'bar',
        'n_run_exp': section.n_run,
        'pp_ensembles': pp_ensembles if pp_ensembles is not None else [],
    }
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
# _resolve_experiment
# ---------------------------------------------------------------------------
def test_resolve_experiment_regular_below_n_run() -> None:
    exps = [_exp(species=['A', 'B'])]
    section = _bare_section(experiments=exps, n_run=1)

    exp_obj, show, label = section._resolve_experiment(0)

    assert exp_obj is exps[0]
    assert show is True
    assert 'Time profile #0' in label
    assert '1.013 bar' in label
    assert '300 K' in label
    assert label.endswith('A, B')


def test_resolve_experiment_banded_extrapolated() -> None:
    exps = [_exp()]
    pp = [_exp(exp_type='PP', temp=500.0, species=['C'])]
    section = _bare_section(
        experiments=exps, pp_experiments=pp, n_run=1,
        pp_ensembles=['G0000'])

    # experiment_id = n_run + band*n_pp + local = 1 + 0*1 + 0.
    exp_obj, show, label = section._resolve_experiment(1)

    assert exp_obj is pp[0]
    assert show is False
    assert 'Extrapolated (band 0)' in label
    assert 'PP' in label
    assert '500 K' in label


def test_resolve_experiment_out_of_range_returns_none() -> None:
    exps = [_exp()]
    section = _bare_section(experiments=exps, n_run=1, pp_ensembles=[])

    exp_obj, show, label = section._resolve_experiment(5)

    assert exp_obj is None
    assert show is False
    assert label == 'Simulation #5'


# ---------------------------------------------------------------------------
# _experiment_options
# ---------------------------------------------------------------------------
def test_experiment_options_lists_regular_and_banded() -> None:
    exps = [_exp()]
    pp = [_exp(exp_type='PP', species=['C'])]
    section = _bare_section(
        experiments=exps, pp_experiments=pp, n_run=1,
        pp_ensembles=['G0000', 'G0001'])

    options = section._experiment_options()

    # value 0 (regular) + one banded id per (band, local): 1 and 2.
    assert [o['value'] for o in options] == [0, 1, 2]
    assert options[0]['label'] == section._resolve_experiment(0)[2]
    assert options[1]['label'] == section._resolve_experiment(1)[2]
    assert options[2]['label'] == section._resolve_experiment(2)[2]


def test_experiment_options_empty() -> None:
    section = _bare_section(
        experiments=[], pp_experiments=[], n_run=0, pp_ensembles=[])

    assert section._experiment_options() == []


# ---------------------------------------------------------------------------
# show_sim_plot_button callback
# ---------------------------------------------------------------------------
def test_show_sim_plot_button_selected(monkeypatch) -> None:
    section = _bare_section()
    callbacks = _capture_callbacks(monkeypatch, section)

    assert callbacks['show_sim_plot_button']([0]) == {'display': 'block'}


@pytest.mark.parametrize('experiments', [None, []])
def test_show_sim_plot_button_empty_prevents(monkeypatch, experiments) -> None:
    section = _bare_section()
    callbacks = _capture_callbacks(monkeypatch, section)

    with pytest.raises(PreventUpdate):
        callbacks['show_sim_plot_button'](experiments)


# ---------------------------------------------------------------------------
# update_sim_figure callback
# ---------------------------------------------------------------------------
def test_update_sim_figure_regular_one_call_per_species(monkeypatch) -> None:
    exps = [_exp(species=['A'])]
    section = _bare_section(experiments=exps, n_run=1)
    section.sim_db = cast(Any, SimpleNamespace(sv_species=['A']))

    reg_calls: list[dict] = []
    fig_calls: list[dict] = []

    def _fake_regular(selected_gen, experiment_id):
        reg_calls.append(
            {'selected_gen': selected_gen, 'experiment_id': experiment_id})
        return [{'G0000': {}}]

    monkeypatch.setattr(
        section, 'get_regular_condition_profiles', _fake_regular)
    monkeypatch.setattr(
        section, 'make_figure', lambda **kw: fig_calls.append(kw) or [])

    callbacks = _capture_callbacks(monkeypatch, section)
    style, _children = callbacks['update_sim_figure'](1, [0], [0])

    assert style == {'display': 'block'}
    assert len(fig_calls) == 1
    assert fig_calls[0]['experiment_id'] == 0
    assert fig_calls[0]['sp'] == 'A'
    assert reg_calls == [{'selected_gen': [0], 'experiment_id': 0}]


def test_update_sim_figure_banded_routes_experiment_id(monkeypatch) -> None:
    exps = [_exp(species=['A'])]
    pp = [_exp(exp_type='PP', species=['C'])]
    section = _bare_section(
        experiments=exps, pp_experiments=pp, n_run=1,
        pp_ensembles=['G0000'])
    section.sim_db = cast(Any, SimpleNamespace(sv_species=['C']))

    fig_calls: list[dict] = []
    monkeypatch.setattr(
        section, 'get_regular_condition_profiles',
        lambda selected_gen, experiment_id: [{'G0000': {}}])
    monkeypatch.setattr(
        section, 'make_figure', lambda **kw: fig_calls.append(kw) or [])

    callbacks = _capture_callbacks(monkeypatch, section)
    callbacks['update_sim_figure'](1, [1], [0])

    # The banded id resolves to the PP experiment's measured species.
    assert len(fig_calls) == 1
    assert fig_calls[0]['experiment_id'] == 1
    assert fig_calls[0]['sp'] == 'C'


@pytest.mark.parametrize(
    'clic,experiments', [(1, []), (None, [0])])
def test_update_sim_figure_prevents(monkeypatch, clic, experiments) -> None:
    section = _bare_section(experiments=[_exp()], n_run=1)
    fig_calls: list[dict] = []
    monkeypatch.setattr(
        section, 'make_figure', lambda **kw: fig_calls.append(kw) or [])

    callbacks = _capture_callbacks(monkeypatch, section)
    with pytest.raises(PreventUpdate):
        callbacks['update_sim_figure'](clic, experiments, [0])

    assert fig_calls == []


# ---------------------------------------------------------------------------
# make_figure heading
# ---------------------------------------------------------------------------
def test_make_figure_heading_contains_species_and_experiment() -> None:
    exps = [_exp(species=['A'])]
    section = _bare_section(experiments=exps, n_run=1)
    section.sim_db = cast(Any, SimpleNamespace(sv_species=['A']))

    rendered = section.make_figure(
        gen_name='G0000', TPGenSP={}, experiment_id=0, sp='A')

    heading = rendered[0]
    assert isinstance(heading, html.H3)
    exp_label = section._resolve_experiment(0)[2]
    assert heading.children == f'A — {exp_label} — G0000'


def test_make_figure_species_absent_returns_bare_figure() -> None:
    exps = [_exp(species=['A'])]
    section = _bare_section(experiments=exps, n_run=1)
    section.sim_db = cast(Any, SimpleNamespace(sv_species=['B']))

    rendered = section.make_figure(
        gen_name='G0000', TPGenSP={}, experiment_id=0, sp='A')

    assert len(rendered) == 1
    assert isinstance(rendered[0], go.Figure)


def test_make_figure_unresolved_experiment_returns_empty() -> None:
    section = _bare_section(experiments=[], n_run=0, pp_ensembles=[])
    section.sim_db = cast(Any, SimpleNamespace(sv_species=['A']))

    rendered = section.make_figure(
        gen_name='G0000', TPGenSP={}, experiment_id=9, sp='A')

    assert rendered == []
