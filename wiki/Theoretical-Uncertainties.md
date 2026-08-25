# 4) Theoretical Uncertainties

This section controls perturbation model, uncertainty magnitudes, distributions, and convergence thresholds for parameter classes.

## 4.1 Global perturbation and score-balance keys

| Keyword | Default value | Description |
|---|---|---|
| max_std | 4 | Maximum deviation from initial model. |
| freq_mode | "batch" | Frequency perturbation mode. Possible values: "batch" or "individual". Individual mode has not been thoroughly tested. |
| weight_theory | 1.0 | Raw theory contribution weight; normalized at runtime with weight_experiments. |
| weight_experiments | 1.0 | Raw experiment contribution weight; normalized at runtime with weight_theory. |
| specific_std | {} | Per-parameter override map for standard deviations. An entry overrides the global `std_<ptype>` for that parameter and governs the sampling scale, the scoring weight, **and** the perturbation boundaries (trusted range): a larger value widens the accepted bounds, a smaller one tightens them. Parameters without an entry fall back to the global `std_<ptype>`. |

## 4.2 Standard deviations (std_*)

| Keyword | Default value | Description |
|---|---|---|
| std_we | 1.0 | Well and bimolecular fragments energy uncertainty (kcal/mol). |
| std_be | 1.5 | Barrier energy uncertainty (kcal/mol). |
| std_ifc | 1.1 | Individual vibrational frequency multiplicative uncertainty. Multiplicative factor. |
| std_bfc | 1.05 | Batch vibrational frequency multiplicative uncertainty. Multiplicative factor. |
| std_hrs | 0.1 | Hindered rotor uncertainty. Percentage. |
| std_if | 1.1 | Imaginary frequency multiplicative uncertainty. Multiplicative factor. |
| std_fact | 0.25 | Energy transfer factor uncertainty. Percentage. |
| std_pow | 0.075 | Energy transfer power uncertainty. Percentage. |
| std_epsilon | 0.1 | Lennard-Jones epsilon uncertainty. Percentage. |
| std_sig | 0.1 | Lennard-Jones sigma uncertainty. Percentage. |
| std_sfc | 2.0 | Symmetry-factor uncertainty for barrierless reactions. While the symmetry has no uncertainty, this parameter allows to scale the state density. Multiplicative factor. |
| std_mrc | 1.5 | Multi-dimensional rotor symmetry uncertainty. While the symmetry has no uncertainty, this parameter allows to scale the state density. Multiplicative factor. |

**Distribution restrictions.** Each parameter belongs to a category that constrains which sampling distribution (`distrib_*`) it may use:

- Multiplicative parameters (`if`, `sfc`, `mrc`, `bfc`, and the individual/batch frequencies) accept only `log-normal` or `log-uniform`.
- Additive (`we`, `be`, `pow`) and percentage (`hrs`, `sigma`, `epsilon`, `fact`) parameters accept only `uniform` or `normal`.

Any other combination cancels the run. In the GUI, only the valid distributions for a parameter's category are selectable.

For multiplicative parameters the theory-score distance is measured in log space as `ln(value / reference) / ln(uncertainty)`, so the penalty is symmetric under a factor and its inverse and consistent with the log-normal (log-space) perturbation of these parameters. Additive and percentage parameters keep their linear distance.

<!-- ### 4.3 Distributions (distrib_*)
|---|---|---|
| distrib_we | "normal" | Distribution for well energies. Additive class: no log distributions allowed. |
| distrib_be | "normal" | Distribution for barrier energies. Additive class: no log distributions allowed. |
| distrib_ifc | "log-normal" | Distribution for individual frequencies. |
| distrib_bfc | "log-normal" | Distribution for batch frequencies. |
| distrib_hrs | "normal" | Distribution for hindered rotors. |
| distrib_if | "log-normal" | Distribution for imaginary frequencies. |
| distrib_fact | "normal" | Distribution for energy transfer factor. |
| distrib_pow | "normal" | Distribution for energy transfer power. Additive class restrictions apply. |
| distrib_epsilon | "normal" | Distribution for Lennard-Jones epsilon. |
| distrib_sig | "normal" | Distribution for Lennard-Jones sigma. |
| distrib_sfc | "log-normal" | Distribution for barrierless symmetry factor. |
| distrib_mrc | "log-normal" | Distribution for multi-dimensional rotor symmetry factor. | -->

## 4.3 Convergence thresholds (conv_*)

| Keyword | Default value | Description |
|---|---|---|
| max_score | 4.0 | Convergence threshold. Maximum score for the best-model ensemble. |
| score_conv | 2 | Convergence threshold. Average score for best-model ensemble. |
| param_conv | 0.01 | Parameter-space convergence threshold used for both parameters values and standard deviations. Percentage. |
| conv_we | 0.1 | Convergence threshold for well energies.(kcal/mol) |
| conv_be | 0.1 | Convergence threshold for barrier energies.(kcal/mol) |
| conv_pow | 0.01 | Convergence threshold for energy transfer power. |
