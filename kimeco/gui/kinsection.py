from dash import html, dcc, callback, Output, Input, State
import numpy as np
from typing import Any, Optional, Tuple
from kimeco.gui.section import Section
# Figure import not required for current return types; keep types minimal
from dash.exceptions import PreventUpdate
from kimeco.gui.histogram import Histogram
from kimeco.goat import RateResult


class KINSection(Section):

    @property
    def layout(self) -> html.Div:
        return html.Div(
            id='kin',
            style={'display': 'block'},
            children=[
                html.H3('Select which rate coefficients to plot:'),
                html.H4('From:'),
                dcc.Dropdown(options=[
                    opt for opt in self._all_rc_options()],
                    multi=True,
                    id='rc_from'),
                html.H4('To:'),
                dcc.Dropdown(options=[
                    opt for opt in self._all_rc_options()],
                    multi=True,
                    id='rc_to'),
                html.H4(f'Pressure ({self.settings["pres_unit"]}):'),
                dcc.Dropdown(options=[
                    p for p in self.settings['rc_pres']],
                    multi=True,
                    id='rc_P'),
                html.H4('Temperature (K):'),
                dcc.Dropdown(options=[
                    t for t in self.settings['rc_temp']],
                    multi=True,
                    id='rc_T'),
                html.H4('PES IDs (optional):'),
                dcc.Dropdown(options=[
                    pid for pid in self.init_SOP.pes_ids],
                    multi=True,
                    id='rc_pes'),
                # Plot BUTTON
                html.Div(
                    id='kin_plot_b',
                    style={'display': 'none'},
                    children=[
                        html.Button(id='kin_plot_button',
                                    children='Plot',
                                    )]),
                html.Div(
                    className='row',
                    id='kin_plot')])

    @staticmethod
    def _encode_rc_option(name: str,
                          pes_id: int) -> str:
        return f"{name}||{int(pes_id)}"

    @staticmethod
    def _decode_rc_option(value: Any) -> Optional[Tuple[str, int]]:
        if not isinstance(value, str) or '||' not in value:
            return None
        name, pes_raw = value.rsplit('||', 1)
        if len(name) == 0:
            return None
        try:
            pes_id = int(pes_raw)
        except (TypeError, ValueError):
            return None
        return name, pes_id

    @staticmethod
    def _label_rc_option(name: str,
                         pes_id: int) -> str:
        return f"{name} [PES {int(pes_id):02d}]"

    def _all_rc_options(self) -> list[dict[str, str]]:
        """Return wells+bimolecular options disambiguated by PES."""
        options_by_value: dict[str, dict[str, str]] = {}

        for well in self.init_SOP.wells:
            for pes_id in sorted({int(pid) for pid in well.pes_ids}):
                value = self._encode_rc_option(well.name, pes_id)
                options_by_value[value] = {
                    'label': self._label_rc_option(well.name, pes_id),
                    'value': value,
                }

        for bimol in self.init_SOP.bimolecular:
            for pes_id in sorted({int(pid) for pid in bimol.pes_ids}):
                value = self._encode_rc_option(bimol.name, pes_id)
                options_by_value[value] = {
                    'label': self._label_rc_option(bimol.name, pes_id),
                    'value': value,
                }

        def sort_key(item: dict[str, str]) -> tuple[int, str]:
            decoded = self._decode_rc_option(item['value'])
            if decoded is None:
                return (10**9, item['label'])
            name, pes_id = decoded
            return (pes_id, name)

        return sorted(options_by_value.values(), key=sort_key)

    def _rc_options_for_pes(self,
                            pes_ids: set[int]) -> list[dict[str, str]]:
        if len(pes_ids) == 0:
            return self._all_rc_options()
        options: list[dict[str, str]] = []
        for opt in self._all_rc_options():
            decoded = self._decode_rc_option(opt['value'])
            if decoded is None:
                continue
            _, pes_id = decoded
            if pes_id in pes_ids:
                options.append(opt)
        return options

    def register_callbacks(self):
        @callback(
            Output('rc_from', 'options'),
            Output('rc_from', 'value'),
            Output('rc_to', 'options'),
            Output('rc_to', 'value'),
            Input('rc_from', 'value'),
            Input('rc_to', 'value')
        )
        def sync_rc_from_to_options(
            rc_from: Optional[list[str]],
            rc_to: Optional[list[str]],
        ) -> tuple[
            list[dict[str, str]],
            list[str],
            list[dict[str, str]],
            list[str],
        ]:
            all_values = {
                opt['value'] for opt in self._all_rc_options()
                if 'value' in opt
            }
            frm_values = [
                val for val in (rc_from or [])
                if isinstance(val, str) and val in all_values
            ]
            to_values = [
                val for val in (rc_to or [])
                if isinstance(val, str) and val in all_values
            ]

            def pes_set(values: list[str]) -> set[int]:
                out: set[int] = set()
                for value in values:
                    parsed = self._decode_rc_option(value)
                    if parsed is None:
                        continue
                    _, pes_id = parsed
                    out.add(pes_id)
                return out

            frm_pes = pes_set(frm_values)
            to_pes = pes_set(to_values)

            frm_allowed_pes = to_pes if len(to_pes) > 0 else frm_pes
            to_allowed_pes = frm_pes if len(frm_pes) > 0 else to_pes

            frm_options = self._rc_options_for_pes(frm_allowed_pes)
            to_options = self._rc_options_for_pes(to_allowed_pes)

            frm_allowed_values = {opt['value'] for opt in frm_options}
            to_allowed_values = {opt['value'] for opt in to_options}

            filtered_frm_values = [
                val for val in frm_values if val in frm_allowed_values
            ]
            filtered_to_values = [
                val for val in to_values if val in to_allowed_values
            ]

            return (
                frm_options,
                filtered_frm_values,
                to_options,
                filtered_to_values,
            )

        @callback(
            Output('kin_plot_b', 'style'),
            Input('rc_from', 'value'),
            Input('rc_to', 'value'),
            Input('rc_P', 'value'),
            Input('rc_T', 'value')
        )
        def show_kin_plot_button(rc_from: list[str],
                                 rc_to: list[str],
                                 rc_P: list[float],
                                 rc_T: list[float]):
            if not rc_from or not rc_to or not rc_P or not rc_T:
                return {'display': 'none'}
            return {'display': 'block'}

        # Plot and print the distribution of the requested rate coefficients
        @callback(
            Output('kin_plot', 'style'),
            Output('kin_plot', 'children'),
            Input('kin_plot_button', 'n_clicks'),
            State('rc_from', 'value'),
            State('rc_to', 'value'),
            State('rc_P', 'value'),
            State('rc_T', 'value'),
            State('rc_pes', 'value'),
            State('gen range slider', 'value'),
            prevent_initial_call=True,
            running=[
                (Output("kin_plot_button", "disabled"), True, False),
                (Output("kin_plot_button", "children"), 'Updating', 'Plot')]
        )
        def update_kin_figure(
            clic,
            From: list[str],
            To: list[str],
            pres: list[float],
            temp: list[float],
            pes_ids: Optional[list[int]],
            selected_gen: list[int],
        ) -> tuple[dict[str, str], list[Any]]:
            if clic is None:
                raise PreventUpdate

            # Build requested P/T conditions
            req_conditions = [(float(p), float(t)) for p in pres for t in temp]

            parsed_from: list[tuple[str, int, str]] = []
            for raw_value in (From or []):
                parsed = self._decode_rc_option(raw_value)
                if parsed is None:
                    continue
                from_name, from_pes = parsed
                parsed_from.append((
                    from_name,
                    from_pes,
                    self._label_rc_option(from_name, from_pes),
                ))

            parsed_to: list[tuple[str, int, str]] = []
            for raw_value in (To or []):
                parsed = self._decode_rc_option(raw_value)
                if parsed is None:
                    continue
                to_name, to_pes = parsed
                parsed_to.append((
                    to_name,
                    to_pes,
                    self._label_rc_option(to_name, to_pes),
                ))

            requested_pes = {
                int(pid) for pid in (pes_ids or [])
            }
            has_pes_filter = len(requested_pes) > 0

            pair_requests: list[
                tuple[str, str, str, str, int]
            ] = []
            pairs_by_pes: dict[int, list[tuple[str, str]]] = {}
            for to_name, to_pes, to_label in parsed_to:
                for frm_name, frm_pes, frm_label in parsed_from:
                    if frm_pes != to_pes:
                        continue
                    if has_pes_filter and frm_pes not in requested_pes:
                        continue
                    pair_requests.append(
                        (frm_name, to_name, frm_label, to_label, frm_pes)
                    )
                    pairs_by_pes.setdefault(frm_pes, []).append(
                        (frm_name, to_name)
                    )

            if len(pair_requests) == 0:
                msg = html.Div([
                    html.P(
                        'No valid same-PES From/To pairs are selected.'
                    ),
                    html.P(
                        'Select From and To entries that share the same '
                        'PES ID.'
                    ),
                ])
                return {'display': 'block'}, [msg]

            all_rates_by_pes: dict[int, RateResult] = {}
            for pes_id, pairs in pairs_by_pes.items():
                unique_pairs = list(dict.fromkeys(pairs))
                all_rates_by_pes[pes_id] = (
                    self.gapp.goats.get_rate_coefficients(
                        req_conditions,
                        selected_gen,
                        unique_pairs,
                        pes_ids=[pes_id],
                    )
                )

            figs: list[Any] = []
            for p in pres:
                for t in temp:
                    for (
                        frm_name,
                        to_name,
                        frm_label,
                        to_label,
                        pes_id,
                    ) in pair_requests:
                        db_pair = (frm_name, to_name)
                        pes_rates = all_rates_by_pes.get(pes_id, {})
                        result = self.make_figure(
                            generations=selected_gen,
                            p=float(p),
                            t=float(t),
                            To=to_label,
                            From=frm_label,
                            db_pair=db_pair,
                            rates=pes_rates,
                        )
                        # make_figure may return a single component or a
                        # list
                        if isinstance(result, list):
                            figs.extend(result)
                        else:
                            figs.append(result)

            if not figs:
                # No figures produced — show a helpful message in the UI
                msg = html.Div([
                    html.P(
                        'No data available for the selected '
                        'species/conditions.'
                    ),
                    html.P(
                        'Check that you selected valid From/To species,'
                        ' pressures, temperatures, and generations.'
                    ),
                ])
                return {'display': 'block'}, [msg]

            return {'display': 'block'}, figs

    def make_figure(
        self,
        generations: list[int],
        p: float,
        t: float,
        To: str,
        From: str,
        db_pair: Optional[Tuple[Optional[str], Optional[str]]],
        rates: RateResult,
    ) -> Any:
        plot_settings: dict[str, Any] = {
            'title': '',
            'tickformat': '.2e',
            'unit': ''
        }

        frm_db, to_db = (None, None) if db_pair is None else db_pair

        # Determine unit and display names (keep display names From/To)
        if frm_db in self.init_SOP.wells_names:
            unit = 's<sup>-1</sup> '
            plot_settings['unit'] = u's\u207B\u00B9'
        else:
            unit = 'cm<sup>3</sup> molecule<sup>-1</sup> s<sup>-1</sup>'
            plot_settings['unit'] = (
                u'cm\u00B3 molecule\u207B\u00B9 s\u207B\u00B9'
            )

        all_gen_rows: dict[int, np.ndarray] = {}
        for gen_i in generations:
            tokens = []
            try:
                tokens = self.gapp.goats.generations[gen_i]
            except Exception:
                tokens = []
            all_gen_rows[gen_i] = np.empty(len(tokens))
            for idx, (mdl_gen, mdl_id) in enumerate(tokens):
                val = None
                if (gen_i in rates
                        and frm_db is not None
                        and to_db is not None):
                    conds = rates[gen_i]
                    key_cond = (float(p), float(t))
                    pair = conds.get(key_cond, {})
                    pair = pair.get((frm_db, to_db), {})
                    val = pair.get((mdl_gen, mdl_id), None)
                all_gen_rows[gen_i][idx] = (
                    val if val is not None else np.nan
                )

        plot_settings['title'] = (
            f"Rate coefficients ({unit}) from {From} to {To} at"
            f" {p} {self.settings['pres_unit']}/{t} K"
        )
        # Histogram expects dict[int, list[float]]; convert arrays to lists

        def coerce_list(arr):
            if hasattr(arr, 'tolist'):
                raw = arr.tolist()
            else:
                raw = list(arr)
            coerced: list[float] = []
            for v in raw:
                try:
                    coerced.append(float(v))
                except Exception:
                    coerced.append(float('nan'))
            return coerced

        hist_data = {g: coerce_list(arr) for g, arr in all_gen_rows.items()}
        hist = Histogram(
            data=hist_data,
            settings=plot_settings,
        )
        return hist.layout()
