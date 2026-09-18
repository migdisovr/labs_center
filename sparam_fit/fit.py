from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy.optimize import least_squares

from .circle import circle_chi2, fit_circle_algebraic
from .guess import (
    guess_delay_linear,
    guess_from_magnitude,
    refine_delay_circle,
    remove_delay,
)
from .models import ResonatorParams, model_s, wrap_phase
from .group_delay import (
    delay_lorentzian,
    group_delay_model,
    guess_from_group_delay,
)

Mode = Literal["circle", "amp_phase", "magnitude", "hybrid", "group_delay"]


@dataclass
class FitResult:
    params: ResonatorParams
    mode: Mode
    success: bool
    message: str = ""
    cov: Optional[np.ndarray] = None
    errors: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    s_delay_removed: Optional[np.ndarray] = None
    s_canonical: Optional[np.ndarray] = None
    circle: dict = field(default_factory=dict)

    def summary(self) -> str:
        p = self.params
        e = self.errors
        lines = [
            f"mode          : {self.mode}  ({self.message})",
            f"geometry      : {p.geometry}",
            f"f_r           : {p.fr:.6e} Hz" + (f"  ± {e['fr']:.3e}" if "fr" in e else ""),
            f"Q_l           : {p.Ql:.6e}" + (f"  ± {e['Ql']:.3e}" if "Ql" in e else ""),
        ]
        if self.diagnostics.get("model") == "lorentzian":
            lines += [
                f"tau (cable)   : {p.tau:.6e} s",
                f"amp (peak)    : {self.diagnostics.get('amp_s', float('nan')):.6e} s",
                f"slope         : {self.diagnostics.get('slope_s', 0.0):.6e} s/Hz",
                f"kappa_l / 2pi : {p.kappa_l_hz:.6e} Hz  (FWHM)",
            ]
        else:
            lines += [
                f"|Q_c|         : {p.absQc:.6e}",
                f"Q_c (DCM)     : {p.Qc_dia_corr:.6e}",
                f"Q_i (DCM)     : {p.Qi_dia_corr:.6e}"
                + (f"  ± {e.get('Qi_dia_corr', float('nan')):.3e}" if "Qi_dia_corr" in e else ""),
                f"Q_i (no corr) : {p.Qi_no_corr:.6e}",
                f"phi           : {p.phi:.6f} rad",
                f"a, alpha, tau : {p.a:.6e},  {p.alpha:.6f} rad,  {p.tau:.6e} s",
                f"kappa_l / 2pi : {p.kappa_l_hz:.6e} Hz  (FWHM)",
                f"kappa_i / 2pi : {p.kappa_i_hz:.6e} Hz",
                f"kappa_c / 2pi : {p.kappa_c_hz:.6e} Hz",
            ]
        return "\n".join(lines)


def _pack_full(p: ResonatorParams):
    return np.array([p.fr, p.Ql, p.absQc, p.phi, p.a, p.alpha, p.tau], dtype=float)


def _unpack_full(x, geometry) -> ResonatorParams:
    return ResonatorParams(
        fr=float(x[0]),
        Ql=float(x[1]),
        absQc=float(x[2]),
        phi=float(x[3]),
        a=float(x[4]),
        alpha=float(x[5]),
        tau=float(x[6]),
        geometry=geometry,
    )


def _scales(p0: ResonatorParams, f):
    fr_s = float(np.mean(f))
    return np.array(
        [
            fr_s,
            max(p0.Ql, 1.0),
            max(p0.absQc, 1.0),
            1.0,
            max(p0.a, 1e-12),
            1.0,
            max(abs(p0.tau), 1.0 / fr_s),
        ]
    )


def _complex_resid(xn, scales, f, s, geometry):
    p = _unpack_full(xn * scales, geometry)
    m = model_s(f, p)
    return np.concatenate([(m - s).real, (m - s).imag])


def _phase_resid(model, data):
    return wrap_phase(np.angle(model) - np.angle(data))


def _cov_from_jac(res, nparams):
    n = res.fun.size
    dof = max(n - nparams, 1)
    chi2 = float(np.sum(res.fun**2) / dof)
    try:
        j = res.jac
        cov_n = chi2 * np.linalg.pinv(j.T @ j)
        return cov_n, chi2
    except Exception:
        return None, chi2


def _errors_from_cov(cov, scales, names):
    if cov is None:
        return {}
    cov_p = cov * np.outer(scales, scales)
    err = np.sqrt(np.maximum(np.diag(cov_p), 0.0))
    return {n: float(e) for n, e in zip(names, err)}


def _initial_params(f, s, geometry, tau=None) -> ResonatorParams:
    mag = np.abs(s)
    g = guess_from_magnitude(f, mag)
    if tau is None:
        tau, _, _ = guess_delay_linear(f, s)
        tau = refine_delay_circle(f, s, tau)
    a = g["a"]
    # diameter estimate: for notch, d ≈ 1 - |S|_min / a
    if g["is_notch"] or geometry == "notch":
        d = max(1.0 - g["depth"] / max(a, 1e-30), 0.05)
        d = min(d, 0.995)
    else:
        d = min(max(g["depth"] / max(a, 1e-30), 0.05), 5.0)
    absQc = g["Ql"] / d if d > 0 else 2.0 * g["Ql"]
    z = remove_delay(f, s, tau)
    # environment phase from off-resonance mean
    n_edge = max(len(f) // 20, 5)
    alpha = float(np.angle(np.mean(np.concatenate([z[:n_edge], z[-n_edge:]]))))
    return ResonatorParams(
        fr=g["fr"],
        Ql=g["Ql"],
        absQc=float(absQc),
        phi=0.0,
        a=a,
        alpha=alpha,
        tau=float(tau),
        geometry=geometry,
    )


def fit_delay(f, s, tau0=None):
    f = np.asarray(f, dtype=float)
    s = np.asarray(s, dtype=np.complex128)
    if tau0 is None:
        tau0, _, _ = guess_delay_linear(f, s)
    tau = refine_delay_circle(f, s, tau0)
    return tau, remove_delay(f, s, tau)


def fit_circle(f, s, geometry: str = "notch", tau: float | None = None, refine_model: bool = True):
    """Probst/Ustinov circle-fit pipeline on complex S-data.

    Steps (paper sec. III–IV):
      1. Estimate / refine electrical delay so the locus is circular.
      2. Algebraic circle fit (Chernov–Lesort) → (xc, yc, r0).
      3. Translate to origin, fit θ(f) = θ0 + 2 arctan[2 Ql (1 - f/fr)].
      4. Off-resonant point P → a, alpha.
      5. Diameter-corrected Qc, Qi (Khalil DCM).
      6. Optional nonlinear polish of the full complex model.
    """
    f = np.asarray(f, dtype=float)
    s = np.asarray(s, dtype=np.complex128)
    tau, z = fit_delay(f, s, tau)
    xc, yc, r0 = fit_circle_algebraic(z, refine=True)
    zc = xc + 1j * yc

    g = guess_from_magnitude(f, np.abs(z))
    theta_data = np.angle(z - zc)

    def phase_model(xn, scales):
        theta0, Ql, fr = xn * scales
        return theta0 + 2.0 * np.arctan(2.0 * Ql * (1.0 - f / fr))

    th_s = np.array([1.0, max(g["Ql"], 1.0), float(np.mean(f))])
    p0 = np.array([0.0, g["Ql"], g["fr"]]) / th_s

    def resid_th(xn):
        return wrap_phase(phase_model(xn, th_s) - theta_data)

    th = least_squares(resid_th, p0, xtol=1e-12, ftol=1e-12)
    theta0, Ql, fr = th.x * th_s
    Ql = abs(float(Ql))
    fr = float(fr)
    theta0 = float(theta0)

    beta = wrap_phase(theta0 + np.pi)
    p_off = (xc + r0 * np.cos(beta)) + 1j * (yc + r0 * np.sin(beta))
    a = float(np.abs(p_off))
    alpha = float(np.angle(p_off))

    # Canonical circle (environment stripped)
    z_can = z / (a * np.exp(1j * alpha)) if a > 0 else z
    xc2, yc2, r02 = fit_circle_algebraic(z_can, refine=True)
    # After canonicalization P → 1, φ = -arcsin(yc/r)
    r02 = max(r02, 1e-12)
    arg = np.clip(yc2 / r02, -1.0, 1.0)
    phi = float(-np.arcsin(arg))
    absQc = float(Ql / (2.0 * r02))

    params = ResonatorParams(
        fr=fr, Ql=Ql, absQc=absQc, phi=phi, a=a, alpha=alpha, tau=tau, geometry=geometry
    )
    if refine_model:
        params = _polish_complex(f, s, params)

    circ = {
        "xc": xc,
        "yc": yc,
        "r0": r0,
        "xc_canonical": xc2,
        "yc_canonical": yc2,
        "r0_canonical": r02,
        "theta0": theta0,
        "offres": p_off,
        "chi2_circle": circle_chi2(z, xc, yc, r0),
    }
    m = model_s(f, params)
    rms = float(np.sqrt(np.mean(np.abs(m - s) ** 2)))
    return FitResult(
        params=params,
        mode="circle",
        success=True,
        message="circle fit + optional complex polish",
        diagnostics={"rms_complex": rms, "snr_est": _snr_est(z, xc, yc, r0)},
        s_delay_removed=z,
        s_canonical=z_can,
        circle=circ,
    )


def _snr_est(z, xc, yc, r0):
    r = np.sqrt((z.real - xc) ** 2 + (z.imag - yc) ** 2)
    sig = np.std(r - r0)
    if sig <= 0:
        return np.inf
    return float(r0 / sig)


def _polish_complex(f, s, p0: ResonatorParams) -> ResonatorParams:
    scales = _scales(p0, f)
    x0 = _pack_full(p0) / scales
    lo = np.array([0.5, 0.05, 0.05, -np.pi, 0.05, -np.pi, -50.0])
    hi = np.array([1.5, 50.0, 50.0, np.pi, 50.0, np.pi, 50.0])
    # delay / a relative bounds around current guess
    res = least_squares(
        lambda xn: _complex_resid(xn, scales, f, s, p0.geometry),
        x0,
        bounds=(lo, hi),
        xtol=1e-12,
        ftol=1e-12,
        max_nfev=400,
    )
    return _unpack_full(res.x * scales, p0.geometry)


def fit_amp_phase(
    f,
    s,
    geometry: str = "notch",
    tau: float | None = None,
    freeze_env_in_amp: bool = True,
):
    """Sequential amplitude → phase → joint complex fit.

    Why this order (and why a naïve 'fit phase then plug into |S|' fails):
      * |S| does **not** depend on alpha or tau (real delay).  Fitting them
        from amplitude is unidentified and poisons Q.
      * Raw phase is dominated by -2 pi f tau.  You must remove delay first,
        then the resonator arctan is visible.
      * phi (Fano) *does* enter |S| as line-shape asymmetry, so it belongs
        in the amplitude step.
      * Compare residuals with circular phase distance, never unwrap(model)
        minus unwrap(data) (independent 2pi jumps).
    """
    f = np.asarray(f, dtype=float)
    s = np.asarray(s, dtype=np.complex128)
    p0 = _initial_params(f, s, geometry, tau=tau)

    # --- 1. amplitude: a, fr, Ql, absQc, phi ---
    mag = np.abs(s)
    amp_names = ["fr", "Ql", "absQc", "phi", "a"]
    scales_a = np.array(
        [float(np.mean(f)), max(p0.Ql, 1.0), max(p0.absQc, 1.0), 1.0, max(p0.a, 1e-12)]
    )
    x0a = np.array([p0.fr, p0.Ql, p0.absQc, p0.phi, p0.a]) / scales_a

    def resid_amp(xn):
        fr, Ql, absQc, phi, a = xn * scales_a
        m = model_s(
            f,
            ResonatorParams(fr, abs(Ql), abs(absQc), phi, abs(a), p0.alpha, p0.tau, geometry),
        )
        return np.abs(m) - mag

    ra = least_squares(
        resid_amp,
        x0a,
        bounds=(
            [0.5, 0.05, 0.05, -np.pi, 0.05],
            [1.5, 50.0, 50.0, np.pi, 50.0],
        ),
        xtol=1e-12,
        ftol=1e-12,
    )
    fr, Ql, absQc, phi, a = ra.x * scales_a
    p_amp = ResonatorParams(float(fr), abs(float(Ql)), abs(float(absQc)), float(phi), abs(float(a)), p0.alpha, p0.tau, geometry)

    # --- 2. phase with resonator params held, then release Ql, fr, phi ---
    z = remove_delay(f, s, p_amp.tau)
    # alpha from edges after delay removal
    n_edge = max(len(f) // 20, 5)
    p_amp.alpha = float(np.angle(np.mean(np.concatenate([z[:n_edge], z[-n_edge:]]))))

    def resid_ph_env(xn):
        alpha, tau = xn
        m = model_s(
            f,
            ResonatorParams(p_amp.fr, p_amp.Ql, p_amp.absQc, p_amp.phi, p_amp.a, alpha, tau, geometry),
        )
        return _phase_resid(m, s)

    rp = least_squares(resid_ph_env, [p_amp.alpha, p_amp.tau], xtol=1e-12, ftol=1e-12, max_nfev=200)
    p_amp.alpha, p_amp.tau = float(rp.x[0]), float(rp.x[1])

    def resid_ph_all(xn):
        fr, Ql, absQc, phi, alpha, tau = xn
        m = model_s(
            f,
            ResonatorParams(fr, abs(Ql), abs(absQc), phi, p_amp.a, alpha, tau, geometry),
        )
        return _phase_resid(m, s)

    x0p = [p_amp.fr, p_amp.Ql, p_amp.absQc, p_amp.phi, p_amp.alpha, p_amp.tau]
    rp2 = least_squares(resid_ph_all, x0p, xtol=1e-12, ftol=1e-12, max_nfev=300)
    fr, Ql, absQc, phi, alpha, tau = rp2.x
    p_ph = ResonatorParams(
        float(fr), abs(float(Ql)), abs(float(absQc)), float(phi), p_amp.a, float(alpha), float(tau), geometry
    )

    # --- 3. joint complex polish (hybrid) ---
    p_fin = _polish_complex(f, s, p_ph)
    m = model_s(f, p_fin)
    z = remove_delay(f, s, p_fin.tau)
    xc, yc, r0 = fit_circle_algebraic(z)
    return FitResult(
        params=p_fin,
        mode="amp_phase",
        success=True,
        message="amplitude → phase → complex polish",
        diagnostics={
            "rms_complex": float(np.sqrt(np.mean(np.abs(m - s) ** 2))),
            "rms_amp": float(np.sqrt(np.mean((np.abs(m) - np.abs(s)) ** 2))),
            "rms_phase": float(np.sqrt(np.mean(_phase_resid(m, s) ** 2))),
            "amp_cost": float(ra.cost),
        },
        s_delay_removed=z,
        circle={"xc": xc, "yc": yc, "r0": r0},
    )


def fit_magnitude_only(f, mag, geometry: str = "notch", mag_is_db: bool = False):
    """Fit when only |S21| (or dB) is available.

    Identifiable: a, fr, Ql, |Qc|, phi (Fano skew).
    Not identifiable: tau, alpha.
    Transmission geometry: a and |Qc| remain degenerate (no off-resonant
    baseline at 1); report the product a*Ql/|Qc| as peak amplitude.
    """
    f = np.asarray(f, dtype=float)
    mag = np.asarray(mag, dtype=float)
    if mag_is_db:
        mag = 10 ** (mag / 20.0)
    g = guess_from_magnitude(f, mag)
    if geometry == "transmission" or not g["is_notch"]:
        geometry = "transmission" if not g["is_notch"] else geometry
    a = g["a"] if geometry == "notch" else max(g["depth"], 1e-12)
    d = max(abs(1.0 - g["depth"] / max(g["a"], 1e-30)), 0.05) if geometry == "notch" else 1.0
    absQc = g["Ql"] / d
    p0 = ResonatorParams(g["fr"], g["Ql"], absQc, 0.0, a, 0.0, 0.0, geometry)
    scales = np.array([float(np.mean(f)), max(p0.Ql, 1.0), max(p0.absQc, 1.0), 1.0, max(p0.a, 1e-12)])
    x0 = np.array([p0.fr, p0.Ql, p0.absQc, p0.phi, p0.a]) / scales

    def resid(xn):
        fr, Ql, absQc, phi, a = xn * scales
        m = model_s(
            f,
            ResonatorParams(fr, abs(Ql), abs(absQc), phi, abs(a), 0.0, 0.0, geometry),
        )
        return np.abs(m) - mag

    res = least_squares(
        resid,
        x0,
        bounds=([0.5, 0.05, 0.05, -np.pi, 0.05], [1.5, 50.0, 50.0, np.pi, 50.0]),
        xtol=1e-12,
        ftol=1e-12,
    )
    fr, Ql, absQc, phi, a = res.x * scales
    params = ResonatorParams(
        float(fr), abs(float(Ql)), abs(float(absQc)), float(phi), abs(float(a)), 0.0, 0.0, geometry
    )
    m = model_s(f, params)
    notes = []
    if geometry == "transmission":
        notes.append(
            "transmission |S|: a and |Qc| are degenerate; Qi from DCM is not trustworthy"
        )
    notes.append("tau and alpha cannot be recovered from magnitude alone")
    return FitResult(
        params=params,
        mode="magnitude",
        success=bool(res.success),
        message="; ".join(notes),
        diagnostics={
            "rms_amp": float(np.sqrt(np.mean((np.abs(m) - mag) ** 2))),
            "identifiable": ["fr", "Ql", "phi", "a (notch baseline)", "|Qc| (notch)"],
            "not_identifiable": ["tau", "alpha"]
            + (["Qi"] if geometry == "transmission" else []),
        },
    )


def fit_hybrid(f, s, geometry: str = "notch"):
    """Run amp+phase and circle fits; polish on the better start; compare."""
    f = np.asarray(f, dtype=float)
    s = np.asarray(s, dtype=np.complex128)
    r_ap = fit_amp_phase(f, s, geometry=geometry)
    r_c = fit_circle(f, s, geometry=geometry, tau=r_ap.params.tau, refine_model=True)
    rms_ap = r_ap.diagnostics.get("rms_complex", np.inf)
    rms_c = r_c.diagnostics.get("rms_complex", np.inf)
    best = r_ap if rms_ap <= rms_c else r_c
    best.mode = "hybrid"
    best.message = (
        f"selected {'amp_phase' if best is r_ap else 'circle'} "
        f"(rms_ap={rms_ap:.3e}, rms_circle={rms_c:.3e})"
    )
    best.diagnostics = {
        **best.diagnostics,
        "rms_amp_phase": rms_ap,
        "rms_circle": rms_c,
        "delta_fr_hz": abs(r_ap.params.fr - r_c.params.fr),
        "delta_Ql": abs(r_ap.params.Ql - r_c.params.Ql),
        "delta_Qi": abs(r_ap.params.Qi_dia_corr - r_c.params.Qi_dia_corr),
        "amp_phase_params": r_ap.params.as_dict(),
        "circle_params": r_c.params.as_dict(),
    }
    return best, r_ap, r_c


def _peak_is_positive(tau_g):
    tau_g = np.asarray(tau_g, dtype=float)
    n_edge = max(len(tau_g) // 15, 8)
    baseline = float(np.median(np.concatenate([tau_g[:n_edge], tau_g[-n_edge:]])))
    detr = tau_g - baseline
    return abs(float(detr.max())) >= abs(float(detr.min()))


def _resolve_delay_sign(tau_g, geometry, sign):
    if sign in (-1, 1, -1.0, 1.0):
        return float(np.sign(sign) or 1.0)
    peak_pos = _peak_is_positive(tau_g)
    # IEEE notch S21 has a delay *dip*; many VNA/*** traces are stored
    # with the opposite sign and look like a peak.  Flip to match the data.
    if geometry == "notch" and peak_pos:
        return -1.0
    if geometry in ("transmission", "reflection") and not peak_pos:
        return -1.0
    return 1.0


def fit_group_delay(
    f,
    tau_g,
    geometry: str = "auto",
    sign="auto",
    model: str = "auto",
):
    """Fit f_r, Q_l from a group-delay trace (seconds, Hz).

    ``model``
      * ``lorentzian`` — tau + slope*(f-fr) + A / (1+4 Q_l^2 (f/fr-1)^2).
        Use this for a VNA **peak** (your 53672 traces).  Does not invent Q_c.
      * ``sparam`` — derivative of the Probst S11/S21 model.  Needed for a
        hanger **dip**.  Reflection near critical coupling is ill-posed
        (S11→0, delay diverges) and is what produced the raised baseline.
      * ``auto`` — peak → lorentzian, dip → sparam notch.

    ``geometry`` is used only for the sparam model: notch / transmission /
    reflection.  ``auto`` picks notch for a dip and skips S-geometry for a peak.
    """
    f = np.asarray(f, dtype=float)
    tau_g = np.asarray(tau_g, dtype=float)
    g = guess_from_group_delay(f, tau_g)
    if model == "auto":
        model = "lorentzian" if g["peak_is_positive"] else "sparam"
    if geometry == "auto":
        geometry = "notch" if not g["peak_is_positive"] else "transmission"
    if model == "lorentzian":
        return _fit_delay_lorentzian(f, tau_g, g)
    if geometry not in ("notch", "transmission", "reflection"):
        raise ValueError(geometry)
    return _fit_delay_sparam(f, tau_g, g, geometry, sign)


def _fit_delay_lorentzian(f, tau_g, g):
    """Symmetric Lorentzian + linear baseline.  Five parameters, all seen in delay."""
    fr0 = g["fr"]
    Ql0 = g["Ql"]
    tau0 = g["baseline"]
    amp0 = g["amp"]
    n_edge = max(len(f) // 15, 8)
    slope0 = 0.0
    if n_edge > 2:
        slope0 = float(
            np.polyfit(
                np.concatenate([f[:n_edge], f[-n_edge:]]),
                np.concatenate([tau_g[:n_edge], tau_g[-n_edge:]]),
                1,
            )[0]
        )
    fr_s = float(np.mean(f))
    tau_s = max(abs(tau0), np.std(tau_g), 1e-12)
    amp_s = max(abs(amp0), 1e-12)
    slope_s = max(abs(slope0), amp_s / max(f[-1] - f[0], 1.0))
    delay_s = max(float(np.std(tau_g)), 1e-12)
    scales = np.array([fr_s, max(Ql0, 1.0), tau_s, amp_s, slope_s], dtype=float)
    x0 = np.array([fr0, Ql0, tau0, amp0, slope0]) / scales

    def unpack(xn):
        fr, Ql, tau, amp, slope = xn * scales
        return float(fr), abs(float(Ql)), float(tau), float(amp), float(slope)

    def resid(xn):
        fr, Ql, tau, amp, slope = unpack(xn)
        return (delay_lorentzian(f, fr, Ql, tau, slope=slope, amp=amp) - tau_g) / delay_s

    res = least_squares(
        resid,
        x0,
        bounds=(
            [0.5, 0.05, -50.0, -50.0, -50.0],
            [1.5, 50.0, 50.0, 50.0, 50.0],
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=800,
    )
    fr, Ql, tau, amp, slope = unpack(res.x)
    model = delay_lorentzian(f, fr, Ql, tau, slope=slope, amp=amp)
    amp_tied = Ql / (np.pi * fr)
    params = ResonatorParams(
        fr=fr,
        Ql=Ql,
        absQc=1e12,
        phi=0.0,
        a=1.0,
        alpha=0.0,
        tau=tau,
        geometry="transmission",
    )
    rms = float(np.sqrt(np.mean((model - tau_g) ** 2)))
    return FitResult(
        params=params,
        mode="group_delay",
        success=bool(res.success),
        message=(
            "lorentzian delay peak; Q_c/Q_i are not identified; "
            f"amp/amp_through={amp / amp_tied:.3f} (1 ≈ transmission S21)"
        ),
        diagnostics={
            "rms_delay_s": rms,
            "sign": 1.0,
            "model": "lorentzian",
            "amp_s": amp,
            "slope_s": slope,
            "amp_through_s": amp_tied,
            "Ql_from_height": float(np.pi * fr * abs(amp)),
            "baseline_s": g["baseline"],
            "peak_is_positive": g["peak_is_positive"],
            "tau_g_model": model,
            "identifiable": ["fr", "Ql", "tau", "slope", "amp"],
            "not_identifiable": ["Qc", "Qi", "a", "alpha"],
        },
    )


def _fit_delay_sparam(f, tau_g, g, geometry, sign):
    sgn = _resolve_delay_sign(tau_g, geometry, sign)
    tau0 = -g["tau"] if sgn < 0 else g["tau"]
    # Keep reflection away from S11=0 (Ql = |Qc|/2), which diverges delay.
    absQc0 = g["absQc"]
    if geometry == "reflection" and abs(g["Ql"] / max(absQc0, 1e-12) - 0.5) < 0.05:
        absQc0 = 2.5 * g["Ql"]
    p0 = ResonatorParams(
        fr=g["fr"],
        Ql=g["Ql"],
        absQc=absQc0,
        phi=0.0,
        a=1.0,
        alpha=0.0,
        tau=tau0,
        geometry=geometry,
    )
    fr_s = float(np.mean(f))
    tau_s = max(abs(p0.tau), np.std(tau_g), 1e-12)
    delay_s = max(float(np.std(tau_g)), 1e-12)
    # transmission delay is independent of Qc and phi — do not fit them
    if geometry == "transmission":
        scales = np.array([fr_s, max(p0.Ql, 1.0), tau_s], dtype=float)
        x0 = np.array([p0.fr, p0.Ql, p0.tau]) / scales

        def unpack(xn):
            fr, Ql, tau = xn * scales
            return ResonatorParams(
                float(fr), abs(float(Ql)), 1e12, 0.0, 1.0, 0.0, float(tau), geometry
            )

        bounds = ([0.5, 0.05, -50.0], [1.5, 50.0, 50.0])
    else:
        scales = np.array(
            [fr_s, max(p0.Ql, 1.0), max(p0.absQc, 1.0), 1.0, tau_s], dtype=float
        )
        x0 = np.array([p0.fr, p0.Ql, p0.absQc, p0.phi, p0.tau]) / scales

        def unpack(xn):
            fr, Ql, absQc, phi, tau = xn * scales
            return ResonatorParams(
                float(fr),
                abs(float(Ql)),
                abs(float(absQc)),
                float(phi),
                1.0,
                0.0,
                float(tau),
                geometry,
            )

        bounds = (
            [0.5, 0.05, 0.05, -np.pi, -50.0],
            [1.5, 50.0, 50.0, np.pi, 50.0],
        )

    def resid(xn):
        return (sgn * group_delay_model(f, unpack(xn)) - tau_g) / delay_s

    res = least_squares(
        resid,
        x0,
        bounds=bounds,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=800,
    )
    params = unpack(res.x)
    model = sgn * group_delay_model(f, params)
    rms = float(np.sqrt(np.mean((model - tau_g) ** 2)))
    notes = [f"sparam {geometry}, IEEE sign={sgn:g}"]
    if geometry == "transmission":
        notes.append("through-line delay does not determine Q_c or Q_i")
    if geometry == "reflection":
        from .models import coupling_regime

        notes.append("regime " + coupling_regime(geometry, params.Ql, params.absQc, params.phi))
        notes.append("delay-only Q_c is weak; prefer lorentzian if the overlay is poor")
    notes.append("a and alpha are invisible in group delay")
    return FitResult(
        params=params,
        mode="group_delay",
        success=bool(res.success),
        message="; ".join(notes),
        diagnostics={
            "rms_delay_s": rms,
            "sign": sgn,
            "model": "sparam",
            "baseline_s": g["baseline"],
            "peak_is_positive": g["peak_is_positive"],
            "tau_g_model": model,
            "identifiable": ["fr", "Ql", "tau"]
            + (["absQc", "phi", "Qi"] if geometry == "notch" else []),
        },
    )


def fit_group_delay_vs_power(
    f, power, delay_2d, geometry: str = "auto", sign="auto", model: str = "auto"
):
    """Fit each power slice of a (n_power, n_freq) delay map.

    Returns a list of FitResult (one per power) and a dict of arrays
    ``fr, Ql, Qi, tau, power`` for plotting vs drive.
    """
    f = np.asarray(f, dtype=float)
    power = np.asarray(power, dtype=float)
    delay_2d = np.asarray(delay_2d, dtype=float)
    if delay_2d.shape != (len(power), len(f)):
        if delay_2d.shape == (len(f), len(power)):
            delay_2d = delay_2d.T
        else:
            raise ValueError(
                f"delay shape {delay_2d.shape} does not match "
                f"(n_power={len(power)}, n_freq={len(f)})"
            )
    results = []
    for i, pwr in enumerate(power):
        r = fit_group_delay(
            f, delay_2d[i], geometry=geometry, sign=sign, model=model
        )
        r.diagnostics["power_dBm"] = float(pwr)
        results.append(r)
    out = {
        "power": power,
        "fr": np.array([r.params.fr for r in results]),
        "Ql": np.array([r.params.Ql for r in results]),
        "Qi": np.array([r.params.Qi_dia_corr for r in results]),
        "absQc": np.array([r.params.absQc for r in results]),
        "tau": np.array([r.params.tau for r in results]),
        "phi": np.array([r.params.phi for r in results]),
        "rms_delay_s": np.array([r.diagnostics["rms_delay_s"] for r in results]),
        "amp_s": np.array([r.diagnostics.get("amp_s", np.nan) for r in results]),
    }
    return results, out

