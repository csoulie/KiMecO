from typing import Any, cast

import cantera.with_units as ctu
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
from kimeco.gui.section import Section
from kimeco.gui.sim_plot import apply_profile_layout

ureg = ctu.cantera_units_registry
Q_ = ureg.Quantity


class SIMSection(Section):
    def __init__(self, gapp) -> None:
        super().__init__(gapp)
        self.species: list[str] = self.sim_db.sv_species
        self.experiments: list = self.settings['experiments']
        self.pp_experiments: list = self.settings.get('pp_experiments', [])
        self.n_run: int = self.settings['n_run_exp']
        self.n_pp: int = len(self.pp_experiments)

    def _resolve_experiment(self,
                            experiment_id: int) -> tuple[Any, bool, str]:
        """Resolve a possibly-banded ``experiment_id``.

        Returns ``(exp_obj, show_exp_profile, label)``. Ids below ``n_run``
        are regular optimization experiments; ids at or above ``n_run`` are
        extrapolated postprocessing profiles decoded via ``divmod``.
        """
        if (0 <= experiment_id < self.n_run
                and experiment_id < len(self.experiments)):
            exp = self.experiments[experiment_id]
            P_bar = Q_(exp.P, 'Pa').to('bar').magnitude
            label = (
                f"{exp.exp_type} #{experiment_id} — "
                f"{P_bar:.4g} {self.settings['pres_unit']}, {exp.T:g} K — "
                f"{', '.join(exp.species)}"
            )
            return exp, True, label
        if self.n_pp > 0:
            band, local = divmod(experiment_id - self.n_run, self.n_pp)
            if 0 <= local < len(self.pp_experiments):
                exp = self.pp_experiments[local]
                P_bar = Q_(exp.P, 'Pa').to('bar').magnitude
                label = (
                    f'Extrapolated (band {band}) — {exp.exp_type} '
                    f"{P_bar:.4g} {self.settings['pres_unit']}, {exp.T:g} K"
                )
                return exp, False, label
        return None, False, f'Simulation #{experiment_id}'

    def _experiment_options(self) -> list[dict]:
        options: list[dict] = [
            {'label': self._resolve_experiment(i)[2], 'value': i}
            for i in range(len(self.experiments))
        ]
        n_bands: int = len(self.settings.get('pp_ensembles', []))
        for band in range(n_bands):
            for local in range(self.n_pp):
                exp_id = self.n_run + band * self.n_pp + local
                options.append(
                    {'label': self._resolve_experiment(exp_id)[2],
                     'value': exp_id})
        return options

    @property
    def layout(self) -> html.Div:
        return html.Div(
            id='sim',
            style={'display': 'block'},
            children=[
                html.H4('Experiments to visualize'),
                dcc.Dropdown(
                    options=cast(Any, self._experiment_options()),
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
            Output('show_sim_plot_button', 'style'),
            Input('experiment', 'value'),
        )
        def show_sim_plot_button(experiments: list[int] | int):
            if not experiments:
                raise PreventUpdate
            return {'display': 'block'}

        @callback(
            Output('sim_plot', 'style'),
            Output('sim_plot', 'children'),
            Input('show_sim_plot_button', 'n_clicks'),
            State('experiment', 'value'),
            State('gen range slider', 'value'),
            prevent_initial_call=True,
            running=[
                (Output('sim_plot_button', 'disabled'), True, False),
                (Output('sim_plot_button', 'children'), 'Updating', 'Plot'),
            ],
        )
        def update_sim_figure(clic,
                              experiments: list[int] | int,
                              selected_gen: list[int]):
            if clic is None or not experiments:
                raise PreventUpdate

            if not isinstance(experiments, (list, tuple, np.ndarray)):
                experiments = [experiments]

            sim_plot_children = []
            for exp_id in experiments:
                exp_obj, _, _ = self._resolve_experiment(exp_id)
                if exp_obj is None:
                    continue
                measured = exp_obj.species

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
                            )
                        )

            return {'display': 'block'}, sim_plot_children

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

    def make_figure(self,
                    gen_name: str,
                    TPGenSP: dict[str, dict[int, dict[int, NDArray]]],
                    experiment_id: int,
                    sp: str):
        sim_db = self.sim_db
        if sim_db is None:
            return []

        exp_obj, show_exp_profile, exp_label = self._resolve_experiment(
            experiment_id)
        if exp_obj is None:
            return []

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
            exp_species = exp_obj.species
            if sp in exp_species:
                exp_p = exp_obj.data
                exp_sp_idx = exp_species.index(sp) + 1
                fig.add_trace(
                    go.Scatter(
                        x=exp_p[0],
                        y=exp_p[exp_sp_idx],
                        error_y={
                            'array': exp_obj.error[exp_sp_idx]
                        },
                        mode='lines',
                        name='Exp. profile',
                        line=dict(color='black'),
                    )
                )

        apply_profile_layout(fig)
        P_bar = round(Q_(exp_obj.P, 'Pa').to('bar').magnitude, 5)
        return [
            html.H3(f'{sp} — {exp_label} — {gen_name}'),
            html.H4(f'T (K): {exp_obj.T}'),
            html.H4(f"P ({self.settings['pres_unit']}): {P_bar}"),
            html.H5(f'Number of models: {nel}'),
            dcc.Graph(figure=fig),
        ]
