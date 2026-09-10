import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparam_fit.models import ResonatorParams, model_s, quality_from_linewidth
from sparam_fit.synthetic import make_trace
from sparam_fit.fit import fit_amp_phase, fit_circle, fit_hybrid, fit_magnitude_only
from sparam_fit.circle import fit_circle_algebraic


TRUE = ResonatorParams(
    fr=6.642e9,
    Ql=1200.0,
    absQc=1800.0,
    phi=0.12,
    a=0.004,
    alpha=0.7,
    tau=45e-9,
    geometry="notch",
)


def test_linewidth_convention():
    assert quality_from_linewidth(6e9, 3e6) == pytest.approx(2000.0)


def test_algebraic_circle_recovers_radius():
    rng = np.random.default_rng(1)
    th = np.linspace(0, 2 * np.pi, 80)
    z = 0.3 + 0.15j + 0.2 * np.exp(1j * th) + 0.002 * (rng.normal(size=80) + 1j * rng.normal(size=80))
    xc, yc, r0 = fit_circle_algebraic(z)
    assert xc == pytest.approx(0.3, abs=0.02)
    assert yc == pytest.approx(0.15, abs=0.02)
    assert r0 == pytest.approx(0.2, abs=0.02)


def test_circle_fit_recovers_truth():
    f, s = make_trace(TRUE, n=501, span_bw=10.0, snr=80, seed=2)
    r = fit_circle(f, s, geometry="notch")
    p = r.params
    assert p.fr == pytest.approx(TRUE.fr, rel=2e-5)
    assert p.Ql == pytest.approx(TRUE.Ql, rel=0.05)
    assert p.Qi_dia_corr == pytest.approx(TRUE.Qi_dia_corr, rel=0.12)
    assert p.tau == pytest.approx(TRUE.tau, rel=0.15)


def test_amp_phase_recovers_truth():
    f, s = make_trace(TRUE, n=501, span_bw=10.0, snr=80, seed=3)
    r = fit_amp_phase(f, s, geometry="notch")
    p = r.params
    assert p.fr == pytest.approx(TRUE.fr, rel=2e-5)
    assert p.Ql == pytest.approx(TRUE.Ql, rel=0.05)
    assert p.a == pytest.approx(TRUE.a, rel=0.08)
    assert p.Qi_dia_corr == pytest.approx(TRUE.Qi_dia_corr, rel=0.12)


def test_magnitude_only_gets_fr_ql():
    f, s = make_trace(TRUE, n=501, span_bw=10.0, snr=200, seed=4)
    r = fit_magnitude_only(f, np.abs(s), geometry="notch")
    assert r.params.fr == pytest.approx(TRUE.fr, rel=5e-5)
    assert r.params.Ql == pytest.approx(TRUE.Ql, rel=0.08)
    assert r.params.tau == 0.0


def test_hybrid_consistency():
    f, s = make_trace(TRUE, n=401, span_bw=8.0, snr=120, seed=5)
    best, ap, circ = fit_hybrid(f, s)
    assert abs(ap.params.fr - circ.params.fr) / TRUE.fr < 5e-5
    assert abs(ap.params.Ql - circ.params.Ql) / TRUE.Ql < 0.08
    m = model_s(f, best.params)
    assert np.sqrt(np.mean(np.abs(m - s) ** 2)) < 0.15 * TRUE.a


def test_wrong_q_factor_of_two_is_not_used():
    """User notebook used Q = f / (2 kappa).  FWHM convention is Q = f / kappa."""
    fr, fwhm = 5.7349e9, 2.0815e6
    q_correct = fr / fwhm
    q_wrong = fr / (2 * fwhm)
    assert q_correct == pytest.approx(2755.2, rel=1e-3)
    assert q_wrong < 0.6 * q_correct
