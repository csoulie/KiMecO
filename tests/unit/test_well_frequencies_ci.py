from __future__ import annotations

import numpy as np
import pytest

from kimeco.well import Well
from kimeco.enums import FreqMode


def _well(freqs: np.ndarray, bfc: float = 1.0) -> Well:
    w = Well(name="W", pes_ids=[0], freq_mode=FreqMode.BATCH)
    w._freq = freqs
    w.bfc = bfc
    return w


def test_batch_frequencies_match_formula() -> None:
    freqs = np.array([100.0, 500.0, 1500.0])
    bfc = 1.05
    w = _well(freqs, bfc)

    expected = freqs * bfc ** (100.0 / freqs)
    assert np.allclose(w.frequencies, expected)


def test_batch_low_frequency_perturbed_more_than_high() -> None:
    freqs = np.array([100.0, 1500.0])
    bfc = 1.1
    w = _well(freqs, bfc)

    factors = w.frequencies / freqs
    # With bfc > 1 the exponent 100/freq is larger for low frequencies,
    # so low frequencies are scaled more than high ones.
    assert factors[0] > factors[1]


def test_batch_bfc_one_is_noop() -> None:
    freqs = np.array([100.0, 500.0, 1500.0])
    w = _well(freqs, bfc=1.0)

    assert np.allclose(w.frequencies, freqs)


def test_individual_mode_uses_ifc_unchanged() -> None:
    freqs = np.array([100.0, 500.0])
    w = Well(name="W", pes_ids=[0], freq_mode=FreqMode.INDIVIDUAL)
    w._freq = freqs
    w.ifc = [1.2, 0.8]

    assert np.allclose(w.frequencies, freqs * np.array([1.2, 0.8]))
