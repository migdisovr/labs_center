from __future__ import annotations

import numpy as np
from scipy.stats import linregress

from .circle import fit_circle_algebraic
from .models import wrap_phase


def guess_delay_linear(f, s, exclude_bw_mult: float = 3.0):
    """Rough cable delay from off-resonance phase slope.

    Model phase ~ alpha - 2*pi*f*tau, so tau = -slope / (2 pi).
    The resonance arctan jump biases a naive polyfit of the whole trace;
    points within ``exclude_bw_mult`` FWHM of the dip/peak are dropped.
    """
    f = np.asarray(f, dtype=float)
    s = np.asarray(s, dtype=np.complex128)
    mag2 = np.abs(s) ** 2
    # Notch: min |S|; transmission: max |S|
    span = mag2.max() - mag2.min()
    is_notch = (mag2[0] + mag2[-1]) / 2 > mag2.min() + 0.25 * span
    i0 = int(np.argmin(mag2) if is_notch else np.argmax(mag2))
    fr = f[i0]
    mid = 0.5 * (mag2[i0] + 0.5 * (mag2[0] + mag2[-1]))
    if is_notch:
        mask_res = mag2 < mid
    else:
        mask_res = mag2 > mid
    if np.any(mask_res):
        fwhm = f[mask_res].max() - f[mask_res].min()
    else:
        fwhm = 0.05 * (f[-1] - f[0])
    off = np.abs(f - fr) > exclude_bw_mult * max(fwhm, f[1] - f[0])
    if off.sum() < 8:
        off = np.ones_like(f, dtype=bool)
    phase = np.unwrap(np.angle(s))
    slope = linregress(f[off], phase[off]).slope
    return float(-slope / (2.0 * np.pi)), float(fr), float(max(fr / fwhm, 10.0))


def refine_delay_circle(f, s, tau0: float, max_nfev: int = 80):
    """Probst: vary tau until delay-corrected data is as circular as possible."""
    from scipy.optimize import least_squares

    f = np.asarray(f, dtype=float)
    s = np.asarray(s, dtype=np.complex128)
    scale = max(abs(tau0), 1.0 / np.mean(f))

    def resid(p):
        tau = p[0] * scale
        z = s * np.exp(2j * np.pi * f * tau)
        xc, yc, r0 = fit_circle_algebraic(z)
        return np.sqrt((z.real - xc) ** 2 + (z.imag - yc) ** 2) - r0

    res = least_squares(resid, [tau0 / scale], max_nfev=max_nfev, xtol=1e-12, ftol=1e-12)
    return float(res.x[0] * scale)


def remove_delay(f, s, tau):
    """Undo exp(-2 pi i f tau) from the Probst environment factor."""
    return np.asarray(s, dtype=np.complex128) * np.exp(
        2j * np.pi * np.asarray(f, dtype=float) * tau
    )


def guess_from_magnitude(f, mag):
    """Skewed-Lorentzian-style start values from |S| or |S|^2.

    Returns dict with fr, Ql, a, depth.  Works for both notch dips and
    transmission peaks.
    """
    f = np.asarray(f, dtype=float)
    mag = np.asarray(mag, dtype=float)
    mag2 = mag**2
    baseline = 0.5 * (mag2[0] + mag2[-1])
    i_min = int(np.argmin(mag2))
    i_max = int(np.argmax(mag2))
    is_notch = baseline > mag2[i_min] + 0.2 * (mag2[i_max] - mag2[i_min] + 1e-30)
    if is_notch:
        i0 = i_min
        mid = 0.5 * (baseline + mag2[i0])
        inside = mag2 <= mid
    else:
        i0 = i_max
        mid = 0.5 * (baseline + mag2[i0])
        inside = mag2 >= mid
    fr = float(f[i0])
    if inside.any():
        fwhm = float(f[inside].max() - f[inside].min())
    else:
        fwhm = float(0.1 * (f[-1] - f[0]))
    fwhm = max(fwhm, float(f[1] - f[0]))
    Ql = float(fr / fwhm)
    a = float(np.sqrt(max(baseline, 1e-30)))
    depth = float(np.sqrt(max(mag2[i0], 0.0)))
    return {"fr": fr, "Ql": Ql, "a": a, "depth": depth, "is_notch": is_notch, "fwhm": fwhm}


def phase_of_centered(z, xc, yc):
    return wrap_phase(np.angle(np.asarray(z) - (xc + 1j * yc)))
