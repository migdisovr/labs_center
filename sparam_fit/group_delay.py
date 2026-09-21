"""Group delay versus electrical (cable) delay.

Two different times are both called "delay" in the lab:

Electrical delay ``tau`` (Probst environment)
    A nearly constant travel time through cables/amps.
    It contributes ``exp(-2 pi i f tau)`` to S and a *flat* offset
    ``tau`` to the group-delay trace.  This is what circle-fit removes.

Group delay ``tau_g(f)`` (VNA format "Delay", qsweepy ``delay``)
    The frequency derivative of the measured phase::

        tau_g(f) = - d arg(S) / d omega = - (1 / 2 pi) d arg(S) / df

    with ``arg`` in radians and ``f`` in hertz.  Near a resonance it has a
    Lorentzian peak (S11 / through S21) or dip (hanger S21).  Peak width
    encodes ``Q_l``, peak position encodes ``f_r``, baseline encodes the
    electrical delay.

Amplitude cannot be reconstructed from ``tau_g`` alone.  Phase can::

        arg S(f) = alpha - 2 pi * cumtrapz(tau_g, f)
"""

from __future__ import annotations

import numpy as np

from .models import ResonatorParams


def group_delay_from_s(f, s, method: str = "analytic_diff"):
    """IEEE / VNA group delay from a complex S trace, in seconds.

    ``method='unwrap'`` uses ``gradient(unwrap(angle(S)))`` and is what most
    measurement software does.  ``analytic_diff`` uses
    ``-Im((dS/df)/S)/(2 pi)``, which is the same quantity but survives a
    2 pi wrap; it still blows up if S passes through the origin (deep notch
    or critical S11).
    """
    f = np.asarray(f, dtype=float)
    s = np.asarray(s, dtype=np.complex128)
    if method == "unwrap":
        phase = np.unwrap(np.angle(s))
        return -np.gradient(phase, f) / (2.0 * np.pi)
    ds_df = np.gradient(s, f)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ds_df / s
    return -np.imag(ratio) / (2.0 * np.pi)


def phase_from_group_delay(f, tau_g, alpha=0.0):
    """Integrate tau_g to a phase.  Inverse of group_delay_from_s up to ``alpha``.

    arg S(f) = alpha - 2 pi * integral_{f0}^f tau_g(f') df'
    """
    f = np.asarray(f, dtype=float)
    tau_g = np.asarray(tau_g, dtype=float)
    integ = np.concatenate(
        [[0.0], np.cumsum(0.5 * (tau_g[1:] + tau_g[:-1]) * np.diff(f))]
    )
    return alpha - 2.0 * np.pi * integ


def _dS_df(f, p: ResonatorParams):
    """Analytic dS/df of the layout × s_param formula table."""
    f = np.asarray(f, dtype=float)
    env = p.a * np.exp(1j * p.alpha) * np.exp(-2j * np.pi * f * p.tau)
    denv_df = env * (-2j * np.pi * p.tau)
    D = 1.0 + 2j * p.Ql * (f / p.fr - 1.0)
    dD_df = 2j * p.Ql / p.fr
    beta = (p.Ql / np.abs(p.absQc)) * np.exp(1j * p.phi)
    kind = p.formula
    if kind == "through":
        R = beta / D
        dR_df = -beta * dD_df / D**2
    elif kind == "notch":
        R = 1.0 - beta / D
        dR_df = beta * dD_df / D**2
    elif kind == "reflection":
        R = 1.0 - 2.0 * beta / D
        dR_df = 2.0 * beta * dD_df / D**2
    else:
        raise ValueError(kind)
    return denv_df * R + env * dR_df, env * R


def group_delay_model(f, p: ResonatorParams):
    """tau_g(f) in seconds for a ResonatorParams instance (IEEE sign)."""
    dS_df, S = _dS_df(f, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.imag(dS_df / S) / (2.0 * np.pi)


def delay_lorentzian(f, fr, Ql, tau, slope=0.0, amp=None):
    """Phenomenological group delay of a symmetric resonance (seconds).

    tau_g(f) = tau + slope*(f - fr) + amp / (1 + 4 Q_l^2 (f/fr - 1)^2)

    If ``amp`` is None it is tied to the through-line identity
    amp = Q_l / (pi f_r).  A free ``amp`` is the right model for a
    VNA delay peak when S11/S21 is unknown.  A hanger S21 dip has
    negative ``amp``.
    """
    f = np.asarray(f, dtype=float)
    if amp is None:
        amp = Ql / (np.pi * fr)
    delta = f / fr - 1.0
    return tau + slope * (f - fr) + amp / (1.0 + 4.0 * Ql**2 * delta**2)


def group_delay_peak_scale(fr, Ql):
    """Through-resonator peak height above a zero baseline: Q_l / (pi f_r).

    Energy lifetime is t_ph = Q_l / omega = Q_l / (2 pi f_r).
    The Lorentzian group-delay peak is 2 t_ph = Q_l / (pi f_r).
    """
    return Ql / (np.pi * fr)


def guess_from_group_delay(f, tau_g):
    """Start values from a 1D delay trace.  Detects peak vs dip vs baseline."""
    f = np.asarray(f, dtype=float)
    tau_g = np.asarray(tau_g, dtype=float)
    n_edge = max(len(f) // 15, 8)
    baseline = float(np.median(np.concatenate([tau_g[:n_edge], tau_g[-n_edge:]])))
    detr = tau_g - baseline
    i_max = int(np.argmax(detr))
    i_min = int(np.argmin(detr))
    peak_is_positive = abs(detr[i_max]) >= abs(detr[i_min])
    i0 = i_max if peak_is_positive else i_min
    fr = float(f[i0])
    height = float(detr[i0])
    half = 0.5 * height
    if peak_is_positive:
        inside = detr >= half
    else:
        inside = detr <= half
    if inside.any():
        fwhm = float(f[inside].max() - f[inside].min())
    else:
        fwhm = float(0.1 * (f[-1] - f[0]))
    fwhm = max(fwhm, float(np.min(np.diff(f))))
    Ql = float(fr / fwhm)
    Ql_from_height = float(np.pi * fr * abs(height)) if height != 0 else Ql
    return {
        "fr": fr,
        "Ql": Ql,
        "tau": baseline,
        "height": height,
        "baseline": baseline,
        "peak_is_positive": peak_is_positive,
        "fwhm": fwhm,
        "Ql_from_height": Ql_from_height,
        "amp": height,
    }


def orient_delay_sweep(p0, p1, data):
    """Map exdir-style (parameter0, parameter1, array) onto (f, power, delay).

    Frequency is recognized as the axis with values around 1e9 Hz; power as
    values with |median| < 200 (dBm).  ``delay`` is returned as
    shape ``(n_power, n_freq)``.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    data = np.asarray(data, dtype=float)

    def is_freq(a):
        return float(np.median(np.abs(a))) > 1e7

    if is_freq(p0) and not is_freq(p1):
        f, power = p0, p1
    elif is_freq(p1) and not is_freq(p0):
        f, power = p1, p0
    else:
        f, power = (p0, p1) if p0.ptp() > p1.ptp() else (p1, p0)

    if data.shape == (len(power), len(f)):
        z = data
    elif data.shape == (len(f), len(power)):
        z = data.T
    else:
        raise ValueError(
            f"cannot match data shape {data.shape} to "
            f"n_power={len(power)}, n_freq={len(f)}"
        )
    return f, power, z
