import sys
import os
import json
import numpy as np
import cantera.with_units as ctu
import traceback
from kimeco._kimeco import KiMecO, optimizer_prefix
from kimeco.model import Model
from kimeco.enums import ModelStatus
from kimeco.goat import GOATs
from kimeco.parameters import SOP
from kimeco.optimizers.NelderMead.nelder_mead_swarm import NelderMeadSwarm
from kimeco.postprocessing.extrapolate import Extrapolate

ureg = ctu.cantera_units_registry
Q_ = ureg.Quantity


class PostProcess(KiMecO):
    def __init__(self,
                 input_file: str,
                 init_loc: str = os.getcwd(),
                 name: str = 'PostProcess') -> None:
        super().__init__(
            input_file=input_file,
            init_loc=init_loc,
            name=name)
        self.settings['postprocess'] = True
        self.prepare_postprocess_settings()

    def prepare_postprocess_settings(self) -> None:
        """Route the postprocessing experiments through the standard
        simulation pipeline.

        pp_experiments are validated and built as TimeProfile objects during
        input parsing. Here they replace the regular experiment list so the
        queueing system, SIM and profile recovery all operate on the pp
        conditions (one array task per unique cantera template).
        """
        pp_experiments = self.settings.get('pp_experiments', [])
        if not pp_experiments:
            raise ValueError(
                'pp_experiments should define at least one experiment '
                'for postprocessing.'
            )

        self.settings['experiments'] = pp_experiments
        self.settings['n_exp'] = len(pp_experiments)

    def copy_necessary_files(self) -> None:
        """Copy run files and emit a postprocess-flagged input file.

        The per-experiment simulation subprocesses rebuild a plain ``KiMecO``
        from the input file. Pointing them at a flagged copy makes them run in
        postprocess mode (pp_experiments conditions and the pp_* rate grid)
        without requiring any change to the user's cantera template.
        """
        super().copy_necessary_files()
        self._write_postprocess_input()

    def _write_postprocess_input(self) -> None:
        """Write a copy of the input JSON with ``postprocess`` enabled and
        redirect the simulation scripts to it.
        """
        input_file: str = self.settings['input_file']
        if os.path.isabs(input_file):
            original_input: str = input_file
        else:
            original_input = self.settings['init_loc'] + input_file
        with open(original_input, 'r') as f:
            raw_input = json.load(f)
        raw_input['postprocess'] = True
        flagged_input: str = os.path.join(
            self.settings['workdir'], '_kmopp_input.json')
        with open(flagged_input, 'w') as f:
            json.dump(raw_input, f)
        self.settings['input_file'] = flagged_input

    def set_initial_sop(self,
                        postprocess=True) -> None:
        """Overide parent method to link postprocess conditions
        to the SOP
        """
        super().set_initial_sop(
            postprocess=postprocess
        )

    def load_goats(self) -> None:
        """Load the goat file from the run to create a
        GOATs object for postprocessing.
        """
        goat_file: str = f"{self.settings['workdir']}/goats.txt"
        # Always construct GOATs from the same goat.txt used previously
        self.goats: GOATs = GOATs.from_file(
            filename=goat_file,
            sop_db=self.sop_db,
            kin_db=self.kin_db,
            sim_db=self.sim_db,
            sf=self.sf,
        )
        # Tables use the optimizer's prefix (e.g. 'G' for the GA).
        self.goats.prefix = optimizer_prefix(self.settings)

    def set_postprocessing(self) -> None:
        """Set parameters for postprocessing"""
        self.klog.info(f"{'Postprocessing parameters:':<65}")
        # Per-experiment metadata (replaces the flat T/P grid dump).
        pu: str = f'{self.settings["pres_unit"]}'
        for idx, exp in enumerate(self.settings['pp_experiments']):
            try:
                p_disp: float = Q_(exp.P, 'Pa').to(
                    self.settings['pres_unit']).magnitude
            except Exception:
                p_disp = exp.P
            comp: str = ', '.join(f'{k}={v}' for k, v in exp.X.items())
            self.klog.info(
                f"  pp_exp #{idx} [{exp.exp_type}] "
                f"T={exp.T:g} K, P={p_disp:g} {pu}, "
                f"species={exp.species}, X={{{comp}}}"
            )

        n_run: int = self.settings['n_run_exp']
        n_pp: int = self.settings['n_exp']
        # Collect models requested by every ensemble token, tag them with
        # their origin table and simulation band, then replay them together.
        ensembles: list[str] = self.settings['pp_ensembles']
        all_models: list[Model] = []

        for band, token in enumerate(ensembles):
            name: str = token
            models: list[Model] = []

            # Precompute token type flags
            cond_g: bool = token.startswith('G') and len(token) == 5 and \
                token[1:].isdigit()
            cond_nms: bool = token == 'NMS'
            cond_gt: bool = token.startswith('GT')
            cond_nm: bool = token.startswith('NM')

            # Generation table: G####
            if cond_g:
                models = self.get_generation(token)
            # GOATs generation: GT####
            elif cond_gt:
                gen_id = int(token[2:])
                try:
                    goats_models = self.goats.get_goat_for_gen(gen_id)
                except Exception as e:
                    self.klog.warning(f"Could not load GOATs for {token}: {e}")
                    traceback.print_exc()
                    raise ValueError('An unexpected error has occured')
                for mdl in goats_models:
                    new_mdl = Model(
                        sop=mdl.sop,
                        id=mdl.id,
                        gen=mdl.gen,
                        status=ModelStatus.SOP.value
                    )
                    new_mdl.origin_prefix = self.goats.prefix
                    models.append(new_mdl)

            # NMSG Nelder-Mead Swarm Generation: NMSG####
            elif cond_nms:
                # Find the size of the ensemble
                nms_cond_g = self.settings['NMS_start'].startswith('G') and \
                    self.settings['NMS_start'][1:].isdigit()
                nms_cond_gt = self.settings['NMS_start'].startswith('GT') and \
                    self.settings['NMS_start'][2:].lstrip('-').isdigit()
                if nms_cond_g:
                    tot_mdl: int = self.settings['n_mdl']
                elif nms_cond_gt:
                    tot_mdl = self.settings['goat_length']
                else:
                    raise NotImplementedError(
                        "NMS_start not recognized for NMSG postprocessing.")
                # Find all NMSG tables in the SOP DB
                NMS_gens: list[int] = [
                    int(tbl_name.split('NMSG')[-1])
                    for tbl_name in self.sop_db.tables
                    if tbl_name.startswith('NMSG')]
                if not NMS_gens:
                    raise ValueError(
                        "No NMSG tables found in SOP DB for postprocessing.")
                max_NMS_gen: int = max(NMS_gens)
                # Load models from NMSG tables, starting from the
                # highest generation until all models are found
                els2load: list[float] = [i for i in range(tot_mdl)]
                for gen in range(max_NMS_gen, -1, -1):
                    table_name: str = f'NMSG{gen:04d}'
                    try:
                        rows = self.sop_db.get_table(table_name)
                    except Exception as e:
                        raise ValueError(
                            f"Could not read table {table_name}: {e}")
                    for row in rows:
                        mdl_id = int(row[0])
                        if mdl_id in els2load:
                            new_mdl = Model(
                                sop=SOP.from_db_row(
                                    sop_tpl=self.init_SOP,
                                    row=np.asarray(row[1:]).tolist()
                                ),
                                id=mdl_id,
                                gen=gen
                            )
                            new_mdl.origin_prefix = NelderMeadSwarm.prefix
                            models.append(new_mdl)
                            els2load.pop(els2load.index(mdl_id))
                    if not els2load:
                        break
            # Nelder-Mead Generation: NM####
            elif cond_nm:
                gen_id = int(token[2:])
                # Find all NM tables in the SOP DB
                NM_gens: list[int] = [
                    int(tbl_name.split('NM')[-1])
                    for tbl_name in self.sop_db.tables
                    if (tbl_name.startswith('NM') and
                        not tbl_name.startswith('NMSG'))]
                if gen_id >= 0:
                    name = token
                else:
                    name = f"NM{len(NM_gens) - 1:04d}"
                if not NM_gens:
                    raise ValueError(
                        "No NM tables found in SOP DB for postprocessing.")
                elif name not in self.sop_db.tables:
                    raise ValueError(
                        f"Table {name} not found in SOP DB for "
                        "postprocessing."
                    )
                nm_rows = self.sop_db.get_table(name)
                new_mdl = Model(
                    sop=SOP.from_db_row(
                        sop_tpl=self.init_SOP,
                        row=nm_rows[0][1:]),
                    id=int(nm_rows[0][0]),
                    gen=int(name[2:])
                )
                new_mdl.origin_prefix = 'NM'
                models.append(new_mdl)
            # Unknown token
            else:
                self.klog.warning(
                    f"Unknown pp_ensemble token '{token}', skipping.")
                continue

            if not models:
                self.klog.warning(f"No models found for {name}, skipping.")
                continue
            for mdl in models:
                mdl.pp_band = band
            all_models.extend(models)

        if not all_models:
            self.klog.warning("No postprocessing models found.")
            return

        # De-duplicate models shared across ensembles (same sop/gen/id),
        # keeping the first band, then assign unique threads and the
        # simulation offset that bands each ensemble's extrapolated profiles.
        merged: list[Model] = list(dict.fromkeys(all_models))
        for i, mdl in enumerate(merged):
            mdl.thread_id = i
            mdl._sim_offset = n_run + mdl.pp_band * n_pp

        Extrapolate(
            models=merged,
            settings=self.settings,
            rc_tpls=self.input_tpls,
            sop_db=self.sop_db,
            kin_db=self.kin_db,
            sim_db=self.sim_db,
            sf=self.sf,
            pert=self.pert,
            klog=self.klog,
            prefix='PP'
        ).run()

    def get_generation(self,
                       token: str) -> list[Model]:
        """Retrieve all models from a generation table in the SOP DB."""
        models: list[Model] = []
        try:
            rows = self.sop_db.get_table(token)
        except Exception as e:
            raise ValueError(
                f"Could not read table {token}: {e}")
        for row in rows:
            mdl = Model(
                sop=SOP.from_db_row(
                    sop_tpl=self.init_SOP,
                    row=np.asarray(row[1:]).tolist()
                ),
                id=int(row[0]),
                gen=int(token[1:])
            )
            mdl.origin_prefix = 'G'
            models.append(mdl)
        return models


def main() -> None:

    def _print_help() -> None:
        print(
            """
KiMecO PostProcess (kmopp)
Run postprocessing and extrapolation on a completed KiMecO run.

This command reads a JSON input file, loads existing run databases/GOAT
ensembles, and executes postprocessing ensembles configured with pp_* keys
(for example pp_ensembles, pp_temp, pp_pres).

Usage:
  kmopp INPUT_JSON

Arguments:
  INPUT_JSON    Path to the KiMecO JSON configuration file.

Options:
  -h, --help    Show this help message and exit.
""".strip()
        )

    if len(sys.argv) == 2 and sys.argv[1] in {'-h', '--help'}:
        _print_help()
        sys.exit(0)

    if len(sys.argv) != 2:
        _print_help()
        sys.exit(1)

    try:
        pp = PostProcess(
            input_file=sys.argv[1])
    except IndexError as e:
        print(e)
        print('To use KiMecO PostProcess, supply the input file as argument.')
        sys.exit(-1)

    pp.initialize_workdir()
    pp.copy_necessary_files()
    pp.initialize_databases()
    pp.set_perturbator()
    pp.load_goats()
    pp.set_postprocessing()
    # Extrapolations are executed inside set_postprocessing()
