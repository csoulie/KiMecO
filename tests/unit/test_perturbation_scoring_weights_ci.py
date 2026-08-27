import kimeco.gui.input_sections.perturbation_section as pert_section
from kimeco.enums import Distrib
from kimeco.gui.input_sections.perturbation_section import (
    _distrib_options_for_additive,
    _distrib_options_for_multiplicative,
    _distrib_options_for_percent,
    update_perturbation_config,
)


def _values(options: list) -> set:
    return {opt["value"] for opt in options}


def test_additive_options_are_exactly_uniform_and_normal() -> None:
    values = _values(_distrib_options_for_additive())
    assert values == {Distrib.UNIFORM.value, Distrib.NORMAL.value}
    assert Distrib.LOGNORMAL.value not in values
    assert Distrib.LOGUNIFORM.value not in values


def test_multiplicative_options_are_exactly_log_distributions() -> None:
    values = _values(_distrib_options_for_multiplicative())
    assert values == {Distrib.LOGNORMAL.value, Distrib.LOGUNIFORM.value}
    assert Distrib.UNIFORM.value not in values
    assert Distrib.NORMAL.value not in values


def test_percent_options_are_exactly_uniform_and_normal() -> None:
    values = _values(_distrib_options_for_percent())
    assert values == {Distrib.UNIFORM.value, Distrib.NORMAL.value}
    assert Distrib.LOGNORMAL.value not in values
    assert Distrib.LOGUNIFORM.value not in values


def test_option_helpers_use_label_value_shape() -> None:
    for helper in (_distrib_options_for_additive,
                   _distrib_options_for_multiplicative,
                   _distrib_options_for_percent):
        options = helper()
        assert len(options) == 2
        for opt in options:
            assert set(opt.keys()) == {"label", "value"}
            assert opt["label"] == opt["value"]


def test_legacy_generic_distrib_options_removed() -> None:
    assert not hasattr(pert_section, "_distrib_options")


def test_update_perturbation_config_preserves_zero_weights() -> None:
    config, valid, message, style = update_perturbation_config(
        max_std=4,
        weight_theory=0.0,
        weight_experiments=0.0,
        std_we=1.0,
        std_be=1.5,
        std_bfc=1.05,
        std_hrs=0.1,
        std_if=1.1,
        std_etf=0.25,
        std_etp=0.075,
        std_epsi=0.1,
        std_sigma=0.1,
        std_sfc=2.0,
        std_mrc=1.5,
        distrib_we="normal",
        distrib_be="normal",
        distrib_freq="log-normal",
        distrib_bfc="log-normal",
        distrib_hrs="normal",
        distrib_if="log-normal",
        distrib_etf="normal",
        distrib_etp="normal",
        distrib_epsi="normal",
        distrib_sigma="normal",
        distrib_sfc="log-normal",
        distrib_mrc="log-normal",
        conv_we=0.1,
        conv_be=0.1,
        conv_etp=0.01,
        specific_std_rows=[],
    )

    assert valid is True
    assert config["weight_theory"] == 0.0
    assert config["weight_experiments"] == 0.0
    assert style["display"] == "block"
    assert message is not None
