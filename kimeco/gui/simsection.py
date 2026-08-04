from typing import Any, cast

import cantera.with_units as ctu
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
from kimeco.database.sim_db import SIM_DB
from kimeco.experiments.t_profile import TimeProfile
from kimeco.gui.section import Section
from kimeco.gui.sim_plot import apply_profile_layout

ureg = ctu.cantera_units_registry
Q_ = ureg.Quantity


class SIMSection(Section):
    def __init__(self, gapp) -> None:
        super().__init__(gapp)
        self.species: list[str] = self.sim_db.sv_species
        self.pp_species: list[str] = []
        self.pp_tables: list[str] = []
        self.experiments: list = self.settings['experiments']
        self.pp_experiments: list = self.settings.get('pp_experiments', [])
        if self.pp_sim_db is not None:
            self.pp_species = self.pp_sim_db.sv_species
            self.pp_tables = sorted(self.pp_sim_db.tables.keys())

    def _experiment_label(self,
                          exp,
                          experiment_id: int,
                          settings: dict[str, Any]) -> str:
        if isinstance(exp, TimeProfile):
            P_bar = Q_(exp.P, 'Pa').to('bar').magnitude
            return (
                f"{exp.exp_type} #{experiment_id} — "
                f"{P_bar:.4g} {settings['pres_unit']}, {exp.T:g} K — "
                f"{', '.join(exp.species)}"
            )
        return f'{exp.exp_type} #{experiment_id}'

    def _experiment_options(self, source: str) -> list[dict]:
        exps = self.pp_experiments if source == 'PP' else self.experiments
        return [
            {'label': self._experiment_label(exp, i, self.settings),
             'value': i}
            for i, exp in enumerate(exps)
        ]

    @property
    def layout(self) -> html.Div:
        return html.Div(
            id='sim',
            style={'display': 'block'},
            children=[
                html.H4('Simulation source'),
                dcc.RadioItems(
                    options=cast(Any, (
                        [{'label': 'Optimization', 'value': 'REG'}]
                        + ([{'label': 'Postprocessing', 'value': 'PP'}]
                           if self.pp_sim_db is not None else [])
                    )),
                    value='REG',
                    inline=True,
                    id='sim_source',
                ),
                html.Div(
                    id='pp_sim_tables_container',
                    style={'display': 'none'},
                    children=[
                        html.H4('Postprocessing tables'),
                        dcc.Dropdown(
                            options=self.pp_tables,
                            multi=True,
                            id='pp_sim_tables',
                        ),
                    ],
                ),
                html.H4('Experiments to visualize'),
                dcc.Dropdown(
                    options=cast(Any, self._experiment_options('REG')),
                    multi=True,
                    id='experiment',
                ),
                html.Div(
                    id='show_sim_plot_button',
                    style={'display': 'none'},
                    children=[
                        html.Button(
                            id='sim_plot_button',
                            children='Plot',
                        )
                    ],
                ),
                html.Div(className='row', id='sim_plot'),
            ],
        )

    def register_callbacks(self):
        @callback(
            Output('experiment', 'options'),
            Output('pp_sim_tables_container', 'style'),
            Output('pp_sim_tables', 'options'),
            Input('sim_source', 'value'),
        )
        def update_sim_source(source: str):
            if source == 'PP':
                return (
                    cast(Any, self._experiment_options('PP')),
                    {'display': 'block'},
                    self.pp_tables,
                )

            return (
                cast(Any, self._experiment_options('REG')),
                {'display': 'none'},
                self.pp_tables,
            )

        @callback(
            Output('show_sim_plot_button', 'style'),
            Input('sim_source', 'value'),
            Input('experiment', 'value'),
            Input('pp_sim_tables', 'value'),
        )
        def show_sim_plot_button(source: str,
                                 experiments: list[int] | int,
                                 pp_tables: list[str]):
            if not experiments:
                raise PreventUpdate
            if source == 'PP' and not pp_tables:
                raise PreventUpdate
            return {'display': 'block'}

        @callback(
            Output('sim_plot', 'style'),
            Output('sim_plot', 'children'),
            Input('show_sim_plot_button', 'n_clicks'),
            State('sim_source', 'value'),
            State('experiment', 'value'),
            State('pp_sim_tables', 'value'),
            State('gen range slider', 'value'),
            prevent_initial_call=True,
            running=[
                (Output('sim_plot_button', 'disabled'), True, False),
                (Output('sim_plot_button', 'children'), 'Updating', 'Plot'),
            ],
        )
        def update_sim_figure(clic,
                              source: str,
                              experiments: list[int] | int,
                              pp_tables: list[str],
                              selected_gen: list[int]):
            if clic is None or not experiments:
                raise PreventUpdate

            if not isinstance(experiments, (list, tuple, np.ndarray)):
                experiments = [experiments]

            sim_plot_children = []
            for exp_id in experiments:
                if source == 'PP':
                    exp = self.pp_experiments[exp_id]
                else:
                    exp = self.experiments[exp_id]
                measured = exp.species
                P_bar = round(Q_(exp.P, 'Pa').to('bar').magnitude, 5)
                T = exp.T

                if source == 'PP':
                    p_idx = self._get_pressure_index(
                        source='PP', pressure=P_bar)
                    t_idx = self.settings['pp_temp'].index(T)
                    all_table_sims = self.get_pp_condition_profiles(
                        tables=pp_tables or [],
                        p_idx=p_idx,
                        t_idx=t_idx,
                    )
                    for table_name in pp_tables or []:
                        table_results: dict[
                            str, dict[int, dict[int, NDArray]]] = {}
                        if table_name in all_table_sims:
                            table_results[table_name] = (
                                all_table_sims[table_name]
                            )
                        for sp in measured:
                            sim_plot_children.extend(
                                self.make_figure(
                                    gen_name=table_name,
                                    TPGenSP=table_results,
                                    experiment_id=exp_id,
                                    sp=sp,
                                    sim_db=self.pp_sim_db,
                                    show_exp_profile=False,
                                )
                            )
                else:
                    all_gen_sims = self.get_regular_condition_profiles(
                        selected_gen=selected_gen,
                        experiment_id=exp_id,
                    )
                    for table_results, gen_i in zip(
                            all_gen_sims, selected_gen):
                        for sp in measured:
                            sim_plot_children.extend(
                                self.make_figure(
                                    gen_name=f'G{gen_i:04d}',
                                    TPGenSP=table_results,
                                    experiment_id=exp_id,
                                    sp=sp,
                                    sim_db=self.sim_db,
                                    show_exp_profile=True,
                                )
                            )

            return {'display': 'block'}, sim_plot_children

    def _get_pressure_index(self,
                            source: str,
                            pressure: float) -> int:
        if source == 'PP':
            return self.settings.get('pp_pres', []).index(pressure)
        return self.settings['rc_pres'].index(pressure)

    def get_regular_condition_profiles(
            self,
            selected_gen: list[int],
            experiment_id: int) -> list[
                dict[str, dict[int, dict[int, NDArray]]]]:
        all_gen_sims: list[dict[str, dict[int, dict[int, NDArray]]]] = []
        models_per_gen: dict[int, list] = {}
        for gen_i in selected_gen:
            gens = self.gapp.goats.generations
            if isinstance(gens, dict):
                tokens = gens.get(gen_i, [])
            else:
                try:
                    tokens = gens[gen_i]
                except Exception:
                    tokens = []
            models_per_gen[gen_i] = tokens
            for (mdl_gen, mdl_id) in tokens:
                self.sim_db.prepare_batch_select(
                    table=f'G{mdl_gen:04d}',
                    mdl_id=mdl_id,
                    experiment_id=experiment_id,
                )

        all_results = self.sim_db.batch_select()
        for gen_i in selected_gen:
            tables = {f'G{tkn[0]:04d}' for tkn in models_per_gen[gen_i]}
            table_results: dict[str, dict[int, dict[int, NDArray]]] = {}
            for tbl in tables:
                if tbl in all_results:
                    table_results[tbl] = all_results[tbl]
            all_gen_sims.append(table_results)
        return all_gen_sims

    def get_pp_condition_profiles(
            self,
            tables: list[str],
            p_idx: int,
            t_idx: int) -> dict[str, dict[int, dict[int, NDArray]]]:
        if self.pp_sim_db is None:
            return {}

        sim_idx = p_idx * len(self.settings['pp_temp']) + t_idx
        for table_name in tables:
            table_rows = self.pp_sim_db.get_table(table=table_name)
            mdl_ids = sorted(
                {
                    int(row[0]) for row in table_rows
                    if int(row[1]) == sim_idx
                }
            )
            for mdl_id in mdl_ids:
                self.pp_sim_db.prepare_batch_select(
                    table=table_name,
                    mdl_id=mdl_id,
                    experiment_id=sim_idx,
                )
        all_results = self.pp_sim_db.batch_select()
        return all_results

    def make_figure(self,
                    gen_name: str,
                    TPGenSP: dict[str, dict[int, dict[int, NDArray]]],
                    experiment_id: int,
                    sp: str,
                    sim_db: SIM_DB | None,
                    show_exp_profile: bool):
        if sim_db is None:
            return []

        if show_exp_profile:
            exp_obj = self.experiments[experiment_id]
        else:
            exp_obj = self.pp_experiments[experiment_id]

        fig = go.Figure()
        nel = 0
        if sp not in sim_db.sv_species:
            return [fig]
        sp_idx = sim_db.sv_species.index(sp) + 2
        traces: list[go.Scatter] = []
        for origin, model_dict in TPGenSP.items():
            name = f'Origin: {origin}'
            for exp_dict in model_dict.values():
                for arr in exp_dict.values():
                    nel += 1
                    traces.append(
                        go.Scatter(
                            x=arr[:, 1],
                            y=arr[:, sp_idx].T,
                            mode='lines',
                            name=name,
                            showlegend=False,
                            opacity=0.25,
                            line=dict(color='#1E90FF'),
                        )
                    )
        fig.add_traces(traces)

        if show_exp_profile:
            eidx = experiment_id
            exp_species = self.settings['experiments'][eidx].species
            if sp in exp_species:
                exp_p = self.settings['experiments'][eidx].data
                exp_sp_idx = exp_species.index(sp) + 1
                fig.add_trace(
                    go.Scatter(
                        x=exp_p[0],
                        y=exp_p[exp_sp_idx],
                        error_y={
                            'array': self.settings['experiments'][
                                eidx].error[exp_sp_idx]
                        },
                        mode='lines',
                        name='Exp. profile',
                        line=dict(color='black'),
                    )
                )

        apply_profile_layout(fig)
        P_bar = round(Q_(exp_obj.P, 'Pa').to('bar').magnitude, 5)
        exp_label = self._experiment_label(
            exp_obj, experiment_id, self.settings)
        return [
            html.H3(f'{sp} — {exp_label} — {gen_name}'),
            html.H4(f'T (K): {exp_obj.T}'),
            html.H4(f"P ({self.settings['pres_unit']}): {P_bar}"),
            html.H5(f'Number of models: {nel}'),
            dcc.Graph(figure=fig),
        ]
