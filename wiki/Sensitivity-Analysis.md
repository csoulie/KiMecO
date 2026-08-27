# 2) Sensitivity Analysis

This section controls both static sensitivity-based parameter selection and on-the-fly periodic sensitivity analysis during optimization.

| Keyword | Default value | Description |
|---|---|---|
| sensi_d | 0.1 | Derivative step multiplier applied to parameter uncertainty in sensitivity analysis. |
| cumul_sensi | 0.95 | Cumulative sensitivity threshold (0 to 1) used to select active parameters. |
| active_p | [] | Explicit list of parameters to perturb. If set, it skips only the **initial** sensitivity analysis; the listed parameters are preserved, and the on-the-fly sensitivity analysis during the GA still runs and augments this list. Leave empty to run the initial sensitivity analysis. |
| SA_start | 1 | Generation index to start on-the-fly sensitivity analysis. |
| SA_end | 80 | Generation index to stop on-the-fly sensitivity analysis. |
| SA_freq | 20 | Frequency (in generations) for running on-the-fly sensitivity updates. |
| fix_theory_divider | false | When `true`, the first theory divider used to average the active parameters' scores is fixed for the whole run, so models with different numbers of active parameters are never compared under different dividers. |

The derivative step is direction-dependent and set by the parameter's **class**. For multiplicative parameters (`if`, `sfc`, `mrc`, `bfc`, frequencies) the step is multiplicative and computed in log space: with factor `f = uc**sensi_d` (where `uc` is the parameter uncertainty), the up step is `value * f` and the down step is `value / f` (e.g. `uc = 1.1`, `sensi_d = 0.1` → perturbed values `[value / 1.1**0.1, value * 1.1**0.1]`). Additive and percentage parameters use a symmetric step, `value + scale * sensi_d * side`.

The averaged central model that anchors the finite differences is computed per parameter class too: multiplicative parameters (`if`, `freq`, `sfc`, `bfc`, `mrc`) are averaged in log space as a geometric mean (`exp(mean(log(values)))`), while additive and percentage parameters use the arithmetic mean, keeping the sensitivity analysis consistent with the log-space treatment used elsewhere in the pipeline (GA, perturbator, scoring, Nelder-Mead).
