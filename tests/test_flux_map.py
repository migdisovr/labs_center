import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparam_fit.flux import (
    analyze_flux_map,
    find_flux_features,
    make_dispersive_flux_map,
    orient_flux_map,
    resonator_frequency_vs_bias,
)


# Qubit entirely above the readout.  The closer approach is the lower
# sweet spot, and level repulsion pulls the resonator down there.
ABOVE = dict(
    fr_bare_hz=6.64e9,
    g_hz=90e6,
    fq_max_hz=8.0e9,
    fq_min_hz=7.15e9,
    period=3.0e-3,
    bias_at_upper=0.0,
)


def _bias(n=121):
    return np.linspace(-2.0e-3, 2.0e-3, n)


def test_orient_sorts_and_swaps_frequency_off_the_bias_axis():
    freq = np.linspace(6.63e9, 6.65e9, 12)
    bias = np.linspace(1e-3, -1e-3, 7)  # descending
    data = np.arange(12 * 7, dtype=float).reshape(7, 12)  # (n_bias, n_freq) after a swap
    # Caller passed current as "freq" and frequency as "bias", with data
    # shaped to those swapped vectors.
    f, b, s, swapped = orient_flux_map(bias, freq, data.T)
    assert swapped
    assert f[0] < f[-1] and b[0] < b[-1]
    assert s.shape == (freq.size, bias.size)
    # Descending bias was sorted; the first output column is the lowest bias,
    # which was the last row of the original (n_bias, n_freq) array.
    assert s[0, 0] == data[-1, 0]


def test_cosine_extrema_are_sweet_spots_and_not_the_inflections():
    bias = _bias(201)
    period = 3.0e-3
    fr = 6.64e9 + 2.0e6 * np.cos(2 * np.pi * bias / period)
    features = find_flux_features(bias, fr)
    spots = [f for f in features if f.kind == "sweet_spot"]
    inflections = [f for f in features if f.kind == "inflection"]
    maxima = sorted(f.bias for f in spots if f.extremum == "fr_maximum")
    minima = sorted(f.bias for f in spots if f.extremum == "fr_minimum")
    assert maxima == pytest.approx([0.0], abs=2e-5)
    assert minima == pytest.approx([-1.5e-3, 1.5e-3], abs=2e-5)
    assert inflections
    assert all(f.kind == "inflection" for f in inflections)
    # Steepest slope sits halfway between the extrema, not on them.
    assert min(abs(f.bias) for f in inflections) == pytest.approx(0.75e-3, abs=5e-5)


def test_qubit_above_labels_the_sharp_minimum_as_the_lower_sweet_spot():
    bias = _bias()
    fr, fq = resonator_frequency_vs_bias(bias, **ABOVE)
    assert fq.max() == pytest.approx(ABOVE["fq_max_hz"])
    assert fr.min() < ABOVE["fr_bare_hz"]
    features = find_flux_features(bias, fr)
    from sparam_fit.flux import assign_qubit_sweet_spots

    side, _ = assign_qubit_sweet_spots(features)
    assert side == "above"
    lowers = [f for f in features if f.qubit_role == "lower"]
    uppers = [f for f in features if f.qubit_role == "upper"]
    assert lowers and uppers
    assert all(f.extremum == "fr_minimum" and f.closest_approach for f in lowers)
    assert all(f.extremum == "fr_maximum" and not f.closest_approach for f in uppers)
    assert np.median([f.fr_hz for f in lowers]) < np.median([f.fr_hz for f in uppers])
    assert max(f.sharpness for f in lowers) > max(f.sharpness for f in uppers)


def test_qubit_below_labels_the_sharp_maximum_as_the_upper_sweet_spot():
    bias = _bias()
    fr, _ = resonator_frequency_vs_bias(
        bias,
        fr_bare_hz=6.64e9,
        g_hz=80e6,
        fq_max_hz=5.9e9,
        fq_min_hz=4.8e9,
        period=3.0e-3,
        bias_at_upper=0.0,
    )
    features = find_flux_features(bias, fr)
    from sparam_fit.flux import assign_qubit_sweet_spots

    side, _ = assign_qubit_sweet_spots(features)
    assert side == "below"
    uppers = [f for f in features if f.qubit_role == "upper"]
    assert uppers
    assert all(f.extremum == "fr_maximum" and f.closest_approach for f in uppers)
    assert max(f.fr_hz for f in uppers) > 6.64e9
    assert max(f.fr_hz for f in uppers) > max(
        f.fr_hz for f in features if f.qubit_role == "lower"
    )


def test_monotonic_and_endpoint_minimum_are_not_sweet_spots():
    bias = _bias(60)
    monotonic = find_flux_features(bias, np.linspace(6.63e9, 6.65e9, bias.size))
    assert [f for f in monotonic if f.kind == "sweet_spot"] == []
    # Minimum sits on the first sample.  The slope never changes sign.
    endpoint = 6.64e9 + 2e6 * (bias - bias[0]) ** 2
    found = find_flux_features(bias, endpoint)
    assert [f for f in found if f.kind == "sweet_spot"] == []


def test_a_branch_hop_does_not_become_a_sweet_spot():
    bias = _bias(201)
    period = 3.0e-3
    fr = 6.64e9 + 2.0e6 * np.cos(2 * np.pi * bias / period)
    fr = fr.copy()
    fr[120:] += 40e6
    features = find_flux_features(bias, fr)
    spots = [f for f in features if f.kind == "sweet_spot"]
    # The hop is at bias[120] ≈ 0.4 mA, between the maximum and the right minimum.
    # Those cosine extrema survive.  The discontinuity itself is not an extremum.
    hop = float(bias[120])
    assert spots
    assert all(abs(f.bias - hop) > 2e-4 for f in spots)
    minima = sorted(f.bias for f in spots if f.extremum == "fr_minimum")
    assert minima == pytest.approx([-1.5e-3, 1.5e-3], abs=3e-5)


def test_analyze_flux_map_fits_hanger_s21_at_both_sweet_spots():
    bias = _bias(81)
    freq, bias, s, fr_true, _ = make_dispersive_flux_map(
        bias,
        **ABOVE,
        Ql=1800.0,
        absQc=2800.0,
        phi=0.1,
        a=0.02,
        alpha=0.4,
        tau=40e-9,
        n_freq=251,
        span_bw=10.0,
    )
    analysis = analyze_flux_map(freq, bias, s, layout="hanger", s_param="S21", sigma_bias=0.0)
    assert analysis.qubit_side == "above"
    assert analysis.primary is not None
    assert analysis.primary.feature.qubit_role == "upper"
    assert analysis.primary.fit.params.formula == "notch"
    for cut in analysis.cuts:
        assert cut.fit.success
        true = float(np.interp(cut.feature.bias, bias, fr_true))
        assert cut.fit.params.fr == pytest.approx(true, rel=5e-5)
        assert cut.fit.params.Ql == pytest.approx(1800.0, rel=0.08)
        assert cut.fit.params.tau == pytest.approx(40e-9, rel=0.2)
    # |S| minimum tracks the ridge.  A nonzero Fano phi shifts the dip
    # from the circle-fit frequency by much less than a linewidth.
    assert analysis.track.fr_hz == pytest.approx(fr_true, abs=5e5)


def test_real_magnitude_map_does_not_pretend_to_be_a_circle_fit():
    bias = _bias(61)
    freq, bias, s, fr_true, _ = make_dispersive_flux_map(
        bias,
        **ABOVE,
        Ql=1600.0,
        absQc=2500.0,
        a=0.05,
        n_freq=201,
        span_bw=8.0,
    )
    analysis = analyze_flux_map(
        freq, bias, np.abs(s), which="upper", layout="hanger", s_param="S21", sigma_bias=0.0
    )
    assert analysis.cuts
    assert analysis.primary.fit.mode == "magnitude"
    true = float(np.interp(analysis.primary.feature.bias, bias, fr_true))
    assert analysis.primary.fit.params.fr == pytest.approx(true, rel=8e-5)
    assert any("magnitude" in note for note in analysis.notes)


def test_dispersive_helper_rejects_an_avoided_crossing():
    bias = _bias(21)
    with pytest.raises(ValueError):
        resonator_frequency_vs_bias(
            bias,
            fr_bare_hz=6.64e9,
            g_hz=80e6,
            fq_max_hz=7.2e9,
            fq_min_hz=6.5e9,
            period=3e-3,
            bias_at_upper=0.0,
        )
