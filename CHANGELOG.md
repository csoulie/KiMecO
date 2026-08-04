# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- TimeProfile data/error CSVs now accept an optional bracketed time unit on the first-column header (e.g. `time[s]`, `TIME [ms]`, `time[1e-3s]`, `time[1e-3]`). The `time` token is case-insensitive and whitespace tolerant; Cantera time units plus `ms`/`millisecond(s)` aliases and numeric-factor forms are supported, with seconds assumed when no bracket is given. A new `TimeProfile.time` property exposes the seconds-normalized time grid.
- New public accessor `GOATs.get_goat_param_values(gen, cols)` returning `dict[str, np.ndarray]` of the requested SOP columns for a generation, in GOAT token order, without reconstructing models or running scoring (`gen == -1` selects the last generation; out-of-range raises `IndexError`).

### Changed
- The SOP GUI plotting subsection ("Type of parameter to plot" → Plot) now fetches each selected generation's data once for all selected columns instead of reconstructing full SOP/Model objects and running scoring per parameter, making parameter plotting much faster. Plotted values (including the Score parameter) are byte-identical to before and the UX is unchanged (one overlaid-histogram figure per selected column).
- Internal `GOATs.get_goat_for_gen` row matching reduced from O(n²) to O(n) via an id→row map, and `GOATs.get_p_for_gen` optimized in place; observable behavior is unchanged (rows now returned in deterministic GOAT token order, identical shapes/dtypes and error contracts).
- API note: `database.sop_db.batch_select_cols` now returns an id-keyed `dict` of the form `{table: {row_id: (col_values...)}}` (the row id is included in each entry) and no longer emits an empty `.where()` clause.
- TimeProfile time grids are normalized to seconds on read (species columns untouched), and data/error files may declare different time units as long as their converted-seconds grids match (compared with a numerical tolerance).
- In the GUI KIN section, reaction pair selection now uses only wells and bimolecular species (fragments excluded), labels entries as `NAME [PES XX]`, enforces same-PES `From`/`To` pairing with reciprocal filtering and auto-clear of invalid selections, and blocks invalid cross-PES plotting with an explanatory message.

### Fixed
- SOP parameter plotting in the analysis GUI no longer risks crashing from memory exhaustion, since it no longer redundantly rebuilds models and rescores once per selected parameter before plotting.

## [1.0.4] - 2026-07-23

### Added
- Analysis notebook for the ethyl oxidation example included in the `example` folder.
- QoL improvements to the experiment class, allowing easy plotting of TimeProfile type experiments in a jupyter notebook.

### Fixed
- Bug in the scoring module that caused the count of active parameters to be incorrectly computed. The issue has been resolved, and the count of active parameters is now independent from the active parameter list used by the perturbation and updated by the sensitivity analysis.

## [1.0.3] - 2026-07-21

### Added
- Sensitivity analysis can restart with frozen parameters.

### Fixed
- Two-sided derivatives properly skipped for frozen parameters in the linear sensitivity analysis.
- Minor bug fix in the scoring module to correctly compute the experimental score when species weights are applied.

## [1.0.2] - 2026-07-20

### Added
- Frozen parameters can now be specified in the input JSON file using the `fixed_params` key. This allows users to exclude certain parameters from being perturbed during optimization.
- Working example for ethyl oxidation with frozen parameters included in the `example` folder.

## [1.0.1] - 2026-07-14

### Added
- Visualization and export of KMO databases in the database tab of the GUI.
- Improved score printing for clearer run output.

### Fixed
- Minor print formatting issue.

### Changed
- Unified the package version across `pyproject.toml`, `setup.py`, and `meta.yaml`.

## [1.0.0] - 2024

### Added
- Initial public release of KiMecO (Kinetic Mechanism Optimizer).

[1.0.4]: https://github.com/sandialabs/KiMecO/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/sandialabs/KiMecO/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/sandialabs/KiMecO/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/sandialabs/KiMecO/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/sandialabs/KiMecO/releases/tag/v1.0.0
[Unreleased]: https://github.com/sandialabs/KiMecO/compare/v1.0.4...HEAD
