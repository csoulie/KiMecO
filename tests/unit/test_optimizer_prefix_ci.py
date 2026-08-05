from __future__ import annotations

import pytest

from kimeco._kimeco import optimizer_prefix
from kimeco.enums import Optimizers
from kimeco.optimizers.GeneticAlgo.exponential import Exponential
from kimeco.optimizers.GeneticAlgo.ga import GeneticAlgorithm
from kimeco.optimizers.GeneticAlgo.tournament import Tournament
from kimeco.optimizers.NelderMead.nelder_mead import NelderMead
from kimeco.optimizers.NelderMead.nelder_mead_swarm import NelderMeadSwarm


def test_prefix_readable_on_classes_without_instantiation() -> None:
    # All prefixes are class attributes: no instance is constructed.
    assert GeneticAlgorithm.prefix == 'G'
    assert Exponential.prefix == 'G'
    assert Tournament.prefix == 'G'
    assert NelderMead.prefix == 'NM'
    assert NelderMeadSwarm.prefix == 'NMSG'


def test_optimizer_prefix_ga_ignores_ga_type() -> None:
    assert optimizer_prefix({'optimizer': Optimizers.GA}) == 'G'
    assert optimizer_prefix(
        {'optimizer': Optimizers.GA, 'ga_type': 'tournament'}) == 'G'
    assert optimizer_prefix(
        {'optimizer': Optimizers.GA, 'ga_type': 'exponential'}) == 'G'


def test_optimizer_prefix_nm() -> None:
    assert optimizer_prefix({'optimizer': Optimizers.NM}) == 'NM'


def test_optimizer_prefix_never_returns_legacy_tokens() -> None:
    for opt in (Optimizers.GA, Optimizers.NM):
        result = optimizer_prefix({'optimizer': opt})
        assert result != 'GT'
        assert not result.startswith('X')


def test_optimizer_prefix_unknown_raises() -> None:
    with pytest.raises(NotImplementedError):
        optimizer_prefix({'optimizer': 'bmcmc'})
