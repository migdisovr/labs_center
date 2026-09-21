import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparam_fit.models import (
    ResonatorParams,
    absQc_from_canonical_radius,
    coupling_regime,
    formula_name,
    model_s,
    quality_from_linewidth,
    validate_layout_sparam,
)
from sparam_fit.synthetic import make_trace
from sparam_fit.fit import (
    fit_amp_phase,
    fit_circle,
    fit_group_delay,
    fit_group_delay_vs_power,
    fit_hybrid,
    fit_magnitude_only,
)
from sparam_fit.circle import fit_circle_algebraic


# Case A — hanger on a matched feedline, complex S21 (Probst notch).
HANGER_S21 = ResonatorParams(
    fr=6.642e9,
    Ql=1200.0,
    absQc=1800.0,
    phi=0.12,
    a=0.004,
    alpha=0.7,
    tau=45e-9,
    layout="hanger",
    s_param="S21",
)

# Case B — same hangers, far end short/open, complex S11 (one-port formula).
HANGER_S11 = ResonatorParams(
    fr=6.957e9,
    Ql=4000.0,
    absQc=20000.0,
    phi=0.05,
    a=1.0,
    alpha=-0.3,
    tau=95e-9,
    layout="hanger",
    s_param="S11",
)

# Case C — in-line through resonator, S21 peak.
THROUGH_S21 = ResonatorParams(
    fr=6.858e9,
    Ql=7000.0,
    absQc=7000.0,
    phi=0.0,
    a=1.0,
    alpha=0.0,
    tau=70e-9,
    layout="through",
    s_param="S21",
)


def test_linewidth_convention():
    assert quality_from_linewidth(6e9, 3e6) == pytest.approx(2000.0)


def test_formula_table_rows():
    assert formula_name("hanger", "S21") == "notch"
    assert formula_name("through", "S21") == "through"
    assert formula_name("hanger", "S11") == "reflection"
    assert formula_name("direct", "S11") == "reflection"


def test_invalid_layout_sparam_combos():
    with pytest.raises(ValueError, match="not a physical combo"):
        validate_layout_sparam("through", "S11")
    with pytest.raises(ValueError, match="not a physical combo"):
        validate_layout_sparam("direct", "S21")
    with pytest.raises(ValueError, match="not a physical combo"):
        ResonatorParams(fr=1e9, Ql=100, absQc=200, layout="through", s_param="S11")


def test_legacy_geometry_maps_with_warning():
    with pytest.warns(DeprecationWarning, match="geometry="):
        p = ResonatorParams(fr=1e9, Ql=100, absQc=200, geometry="notch")
    assert p.layout == "hanger" and p.s_param == "S21"
    with pytest.warns(DeprecationWarning, match="geometry="):
        p = ResonatorParams(fr=1e9, Ql=100, absQc=200, geometry="reflection")
    assert p.layout == "hanger" and p.s_param == "S11"
    with pytest.warns(DeprecationWarning, match="geometry="):
        p = ResonatorParams(fr=1e9, Ql=100, absQc=200, geometry="transmission")
    assert p.layout == "through" and p.s_param == "S21"


def test_algebraic_circle_recovers_radius():
    rng = np.random.default_rng(1)
    th = np.linspace(0, 2 * np.pi, 80)
    z = 0.3 + 0.15j + 0.2 * np.exp(1j * th) + 0.002 * (
        rng.normal(size=80) + 1j * rng.normal(size=80)
    )
    xc, yc, r0 = fit_circle_algebraic(z)
    assert xc == pytest.approx(0.3, abs=0.02)
    assert yc == pytest.approx(0.15, abs=0.02)
    assert r0 == pytest.approx(0.2, abs=0.02)


def test_absQc_from_radius_differs_for_s11_vs_s21():
    Ql, r0 = 4000.0, 0.2
    s21 = absQc_from_canonical_radius(Ql, r0, "hanger", "S21")
    s11 = absQc_from_canonical_radius(Ql, r0, "hanger", "S11")
    assert s21 == pytest.approx(Ql / (2.0 * r0))
    assert s11 == pytest.approx(Ql / r0)
    assert s11 == pytest.approx(2.0 * s21)


def test_case_a_hanger_s21_circle_fit():
    f, s = make_trace(HANGER_S21, n=501, span_bw=10.0, snr=80, seed=2)
    r = fit_circle(f, s, layout="hanger", s_param="S21")
    p = r.params
    assert p.layout == "hanger" and p.s_param == "S21"
    assert p.formula == "notch"
    assert p.fr == pytest.approx(HANGER_S21.fr, rel=2e-5)
    assert p.Ql == pytest.approx(HANGER_S21.Ql, rel=0.05)
    assert p.Qi_dia_corr == pytest.approx(HANGER_S21.Qi_dia_corr, rel=0.12)
    assert p.tau == pytest.approx(HANGER_S21.tau, rel=0.15)


def test_case_a_hanger_s21_amp_phase():
    f, s = make_trace(HANGER_S21, n=501, span_bw=10.0, snr=80, seed=3)
    r = fit_amp_phase(f, s, layout="hanger", s_param="S21")
    p = r.params
    assert p.fr == pytest.approx(HANGER_S21.fr, rel=2e-5)
    assert p.Ql == pytest.approx(HANGER_S21.Ql, rel=0.05)
    assert p.a == pytest.approx(HANGER_S21.a, rel=0.08)
    assert p.Qi_dia_corr == pytest.approx(HANGER_S21.Qi_dia_corr, rel=0.12)


def test_case_a_hanger_s21_magnitude_only():
    f, s = make_trace(HANGER_S21, n=501, span_bw=10.0, snr=200, seed=4)
    r = fit_magnitude_only(f, np.abs(s), layout="hanger", s_param="S21")
    assert r.params.fr == pytest.approx(HANGER_S21.fr, rel=5e-5)
    assert r.params.Ql == pytest.approx(HANGER_S21.Ql, rel=0.08)
    assert r.params.tau == 0.0


def test_case_a_hanger_s21_hybrid():
    f, s = make_trace(HANGER_S21, n=401, span_bw=8.0, snr=120, seed=5)
    best, ap, circ = fit_hybrid(f, s, layout="hanger", s_param="S21")
    assert abs(ap.params.fr - circ.params.fr) / HANGER_S21.fr < 5e-5
    assert abs(ap.params.Ql - circ.params.Ql) / HANGER_S21.Ql < 0.08
    m = model_s(f, best.params)
    assert np.sqrt(np.mean(np.abs(m - s) ** 2)) < 0.15 * HANGER_S21.a


def test_case_b_hanger_s11_circle_fit():
    f, s = make_trace(HANGER_S11, n=501, span_bw=10.0, snr=80, seed=8)
    r = fit_circle(f, s, layout="hanger", s_param="S11")
    p = r.params
    assert p.layout == "hanger" and p.s_param == "S11"
    assert p.formula == "reflection"
    assert p.fr == pytest.approx(HANGER_S11.fr, rel=3e-5)
    assert p.Ql == pytest.approx(HANGER_S11.Ql, rel=0.08)
    assert p.Qi_dia_corr == pytest.approx(HANGER_S11.Qi_dia_corr, rel=0.2)
    assert p.tau == pytest.approx(HANGER_S11.tau, rel=0.2)


def test_case_b_overcoupled_s11_delay_is_a_peak():
    """Hanger×S11 delay is a peak when overcoupled (53672-like).  Lorentzian
    recovers fr and τ; Ql from delay width is not the S-model Ql, and Qi
    is not identified (PHYSICS_AND_API.md §4)."""
    from sparam_fit.group_delay import group_delay_model
    from sparam_fit.fit import fit_group_delay as fit_gd

    p = ResonatorParams(
        fr=6.957e9,
        Ql=4000.0,
        absQc=5000.0,
        phi=0.0,
        a=1.0,
        tau=95e-9,
        layout="hanger",
        s_param="S11",
    )
    f, s = make_trace(p, n=1601, span_bw=16.0, snr=None)
    tg = group_delay_model(f, p)
    assert np.nanmax(tg) - p.tau > p.tau - np.nanmin(tg)
    r = fit_gd(f, tg)
    assert r.diagnostics["model"] == "lorentzian"
    assert r.diagnostics["peak_is_positive"]
    assert r.params.fr == pytest.approx(p.fr, rel=3e-5)
    assert r.params.tau == pytest.approx(p.tau, rel=0.15)
    assert r.params.Ql > 0
    assert "Qi" in r.diagnostics["not_identifiable"]
    assert not np.isfinite(r.params.Qi_dia_corr)


def test_case_c_through_s21_recovers_fr_ql_not_qi():
    f, s = make_trace(THROUGH_S21, n=501, span_bw=10.0, snr=None)
    r = fit_magnitude_only(f, np.abs(s), layout="through", s_param="S21")
    assert r.params.layout == "through" and r.params.s_param == "S21"
    assert r.params.fr == pytest.approx(THROUGH_S21.fr, rel=5e-5)
    assert r.params.Ql == pytest.approx(THROUGH_S21.Ql, rel=0.1)
    assert "Qi" in r.diagnostics["not_identifiable"]


def test_case_c_through_delay_peak_is_ql_over_pi_fr():
    from sparam_fit.group_delay import group_delay_model, group_delay_peak_scale

    f = np.linspace(
        THROUGH_S21.fr - 5 * THROUGH_S21.fr / THROUGH_S21.Ql,
        THROUGH_S21.fr + 5 * THROUGH_S21.fr / THROUGH_S21.Ql,
        2001,
    )
    tg = group_delay_model(f, THROUGH_S21)
    extra = tg[len(f) // 2] - THROUGH_S21.tau
    assert extra == pytest.approx(
        group_delay_peak_scale(THROUGH_S21.fr, THROUGH_S21.Ql), rel=0.02
    )


def test_wrong_q_factor_of_two_is_not_used():
    """User notebook used Q = f / (2 kappa).  FWHM convention is Q = f / kappa."""
    fr, fwhm = 5.7349e9, 2.0815e6
    q_correct = fr / fwhm
    q_wrong = fr / (2 * fwhm)
    assert q_correct == pytest.approx(2755.2, rel=1e-3)
    assert q_wrong < 0.6 * q_correct


def test_group_delay_from_s_matches_model():
    from sparam_fit.group_delay import group_delay_from_s, group_delay_model

    f, s = make_trace(HANGER_S21, n=801, span_bw=12.0, snr=None)
    tg_num = group_delay_from_s(f, s)
    tg_an = group_delay_model(f, HANGER_S21)
    err = np.nanmedian(np.abs(tg_num - tg_an))
    assert err < 0.05 * abs(HANGER_S21.tau)


def test_hanger_s21_delay_is_a_dip_lorentzian_amp_negative():
    """Matched hanger S21 has anomalous (negative) extra delay.  The
    Lorentzian still finds fr; it does not switch to a through/S11 model
    and it does not invent Qi."""
    from sparam_fit.group_delay import group_delay_model
    from sparam_fit.fit import fit_group_delay as fit_gd

    f, s = make_trace(HANGER_S21, n=801, span_bw=12.0, snr=None)
    tg = group_delay_model(f, HANGER_S21)
    r = fit_gd(f, tg)
    assert r.diagnostics["model"] == "lorentzian"
    assert r.diagnostics["amp_s"] < 0
    assert r.params.fr == pytest.approx(HANGER_S21.fr, rel=5e-5)
    assert r.params.tau == pytest.approx(HANGER_S21.tau, rel=0.2)
    assert not np.isfinite(r.params.Qi_dia_corr)


def test_delay_fit_ignores_geometry_and_does_not_auto_switch():
    from sparam_fit.group_delay import delay_lorentzian

    fr, Ql, tau = 6.95713e9, 6000.0, 95e-9
    f = np.linspace(6.948e9, 6.967e9, 401)
    tg = delay_lorentzian(f, fr, Ql, tau, slope=0.0, amp=Ql / (np.pi * fr))
    with pytest.warns(DeprecationWarning, match="Lorentzian"):
        r = fit_group_delay(f, tg, geometry="notch", model="sparam")
    assert r.diagnostics["model"] == "lorentzian"
    assert r.params.fr == pytest.approx(fr, rel=2e-5)


def test_orient_delay_sweep_swapped_axes():
    from sparam_fit.group_delay import orient_delay_sweep

    f = np.linspace(6.85e9, 6.86e9, 40)
    p = np.linspace(-50, -10, 5)
    z = np.arange(5 * 40, dtype=float).reshape(5, 40)
    fo, po, zo = orient_delay_sweep(f, p, z)
    assert np.allclose(fo, f) and np.allclose(po, p) and zo.shape == (5, 40)
    fo, po, zo = orient_delay_sweep(p, f, z.T)
    assert np.allclose(fo, f) and np.allclose(po, p) and zo.shape == (5, 40)


def test_delay_power_map_shape():
    from sparam_fit.group_delay import delay_lorentzian

    f = np.linspace(6.85e9, 6.866e9, 301)
    powers = np.array([-50.0, -30.0, -10.0])
    traces = []
    for Ql in (5000, 6000, 7000):
        traces.append(
            delay_lorentzian(f, 6.858e9, Ql, 60e-9, amp=Ql / (np.pi * 6.858e9))
        )
    results, arr = fit_group_delay_vs_power(f, powers, np.vstack(traces))
    assert len(results) == 3
    assert arr["Ql"][0] < arr["Ql"][-1]
    assert arr["fr"][0] == pytest.approx(6.858e9, rel=1e-4)
    assert not np.isfinite(arr["Qi"]).any()


def test_lorentzian_recovers_peak_on_cable_baseline():
    """Mimic experimental 53672: Lorentzian peak ~370 ns on ~95 ns shelf."""
    from sparam_fit.group_delay import delay_lorentzian

    fr, Ql, tau = 6.95713e9, 6000.0, 95e-9
    f = np.linspace(6.948e9, 6.967e9, 401)
    tg = delay_lorentzian(f, fr, Ql, tau, slope=0.0, amp=Ql / (np.pi * fr))
    rng = np.random.default_rng(0)
    tg = tg + 3e-9 * rng.normal(size=f.size)
    r = fit_group_delay(f, tg)
    assert r.diagnostics["model"] == "lorentzian"
    assert r.params.fr == pytest.approx(fr, rel=2e-5)
    assert r.params.Ql == pytest.approx(Ql, rel=0.05)
    assert r.params.tau == pytest.approx(tau, rel=0.08)
    assert r.diagnostics["rms_delay_s"] < 6e-9
    assert not np.isfinite(r.params.Qi_dia_corr)


def test_coupling_regime_s11_critical():
    assert coupling_regime(5000, 10000, layout="hanger", s_param="S11") == "critical"
    assert (
        coupling_regime(4000, 20000, layout="hanger", s_param="S11") == "undercoupled"
    )
    assert coupling_regime(4000, 5000, layout="hanger", s_param="S11") == "overcoupled"


def test_magnitude_does_not_silently_switch_layout():
    """A through peak must not be re-labelled hanger unless the caller says so."""
    f, s = make_trace(THROUGH_S21, n=401, span_bw=8.0, snr=None)
    r = fit_magnitude_only(f, np.abs(s), layout="hanger", s_param="S21")
    assert r.params.layout == "hanger"
    assert r.params.s_param == "S21"


def test_legacy_fit_geometry_kwarg_warns():
    f, s = make_trace(HANGER_S21, n=201, span_bw=8.0, snr=None)
    with pytest.warns(DeprecationWarning, match="geometry="):
        r = fit_circle(f, s, geometry="notch", refine_model=False)
    assert r.params.layout == "hanger" and r.params.s_param == "S21"
