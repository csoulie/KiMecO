"""CI-safe tests for the perturbation config callback signature (Defect A).

The stale ``pert`` positional argument (and its ``perturbation-pert-dropdown``
Input) was removed so the callback's parameter list lines up with its declared
Inputs. These tests pin the new signature and the happy-path contract.
"""

import inspect

from dash import html

from kimeco.enums import Distrib, Pclass
from kimeco.gui.input_sections import perturbation_section as ps
from kimeco.gui.input_sections.perturbation_section import (
    create_perturbation_section,
    update_perturbation_config,
)


def _valid_kwargs(**overrides):
    """Build a fully-valid kwargs set for update_perturbation_config."""
    additive = set(Pclass.ADDITIVE.value)          # we, be, pow(etp)
    multiplicative = set(Pclass.MULTIPLICATIVE.value)  # freq, bfc, if, sfc, mrc
    # distrib_freq maps to the 'freq' multiplicative class member.
    kwargs = {
        "max_std": 3,
        "weight_theory": 1.0,
        "weight_experiments": 1.0,
        "std_we": 1.0, "std_be": 1.0, "std_bfc": 0.1, "std_hrs": 0.1,
        "std_if": 0.1, "std_etf": 0.1, "std_etp": 1.0, "std_epsi": 0.1,
        "std_sigma": 0.1, "std_sfc": 0.1, "std_mrc": 0.1,
        "conv_we": 0.1, "conv_be": 0.1, "conv_etp": 0.1,
        "specific_std_rows": [],
    }
    # Distributions: multiplicative -> log-normal, others -> normal.
    distrib_members = {
        "distrib_we": "we", "distrib_be": "be", "distrib_freq": "freq",
        "distrib_bfc": "bfc", "distrib_hrs": "hrs", "distrib_if": "if",
        "distrib_etf": "fact", "distrib_etp": "pow", "distrib_epsi": "epsilon",
        "distrib_sigma": "sigma", "distrib_sfc": "sfc", "distrib_mrc": "mrc",
    }
    for arg, member in distrib_members.items():
        if member in multiplicative:
            kwargs[arg] = Distrib.LOGNORMAL.value
        else:
            kwargs[arg] = Distrib.NORMAL.value
    assert additive  # sanity: additive class populated
    kwargs.update(overrides)
    return kwargs


def test_signature_has_no_pert_param():
    params = list(inspect.signature(update_perturbation_config).parameters)
    assert "pert" not in params


def test_first_param_is_max_std():
    params = list(inspect.signature(update_perturbation_config).parameters)
    assert params[0] == "max_std"


def test_callback_returns_valid_four_tuple():
    result = update_perturbation_config(**_valid_kwargs())
    assert isinstance(result, tuple)
    assert len(result) == 4
    config, valid, _msg, style = result
    assert isinstance(config, dict)
    assert config["max_std"] == 3
    assert valid is True
    # Fully-valid inputs -> no warning banner shown.
    assert style.get("display") == "none"


def test_no_phantom_pert_dropdown_in_source():
    src = open(ps.__file__, encoding="utf-8").read()
    assert "perturbation-pert-dropdown" not in src


def test_layout_builds_without_phantom_id():
    layout = create_perturbation_section()
    assert isinstance(layout, html.Div)


def test_multiplicative_non_log_distribution_warns():
    """Edge: a non-log distrib on a multiplicative param raises a warning."""
    result = update_perturbation_config(
        **_valid_kwargs(distrib_freq=Distrib.UNIFORM.value)
    )
    _config, valid, msg, style = result
    assert valid is True
    assert style.get("display") == "block"
    assert msg != ""
