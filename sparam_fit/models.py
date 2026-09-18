from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np

Geometry = Literal["notch", "transmission", "reflection"]


@dataclass
class ResonatorParams:
    """Physical + environment parameters of the Probst/Khalil resonator model.

    Notch (hanger) S21::

        S21(f) = a * exp(i*alpha) * exp(-2*pi*i*f*tau) *
                 [ 1 - (Ql/|Qc|) * exp(i*phi) / (1 + 2*i*Ql*(f/fr - 1)) ]

    Transmission (through-line resonator) S21::

        S21(f) = a * exp(i*alpha) * exp(-2*pi*i*f*tau) *
                 (Ql/|Qc|) * exp(i*phi) / (1 + 2*i*Ql*(f/fr - 1))

    ``Qc`` stored here is |Qc|.  Diameter-corrected internal Q uses
    ``1/Qi = 1/Ql - cos(phi)/|Qc|`` (Khalil DCM).
    """

    fr: float
    Ql: float
    absQc: float
    phi: float = 0.0
    a: float = 1.0
    alpha: float = 0.0
    tau: float = 0.0
    geometry: Geometry = "notch"

    @property
    def Qc_complex(self) -> complex:
        return self.absQc * np.exp(-1j * self.phi)

    @property
    def Qi_dia_corr(self) -> float:
        inv = 1.0 / self.Ql - np.cos(self.phi) / self.absQc
        return float(1.0 / inv) if inv != 0 else np.inf

    @property
    def Qi_no_corr(self) -> float:
        inv = 1.0 / self.Ql - 1.0 / self.absQc
        return float(1.0 / inv) if inv != 0 else np.inf

    @property
    def Qc_dia_corr(self) -> float:
        """1 / Re(1/Qc_complex) = |Qc| / cos(phi)."""
        c = np.cos(self.phi)
        return float(self.absQc / c) if c != 0 else np.inf

    @property
    def kappa_l_hz(self) -> float:
        """Loaded FWHM in Hz (energy decay rate / 2pi)."""
        return self.fr / self.Ql

    @property
    def kappa_i_hz(self) -> float:
        return self.fr / self.Qi_dia_corr

    @property
    def kappa_c_hz(self) -> float:
        return self.fr / self.Qc_dia_corr

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            {
                "Qi_dia_corr": self.Qi_dia_corr,
                "Qi_no_corr": self.Qi_no_corr,
                "Qc_dia_corr": self.Qc_dia_corr,
                "kappa_l_hz": self.kappa_l_hz,
                "kappa_i_hz": self.kappa_i_hz,
                "kappa_c_hz": self.kappa_c_hz,
            }
        )
        return d


def _env(f, a, alpha, tau):
    return a * np.exp(1j * alpha) * np.exp(-2j * np.pi * f * tau)


def _resonator_denom(f, fr, Ql):
    return 1.0 + 2j * Ql * (f / fr - 1.0)


def s21_notch(f, fr, Ql, absQc, phi=0.0, a=1.0, alpha=0.0, tau=0.0):
    """Notch / hanger S21 (Probst eq. 1). ``f`` in Hz."""
    f = np.asarray(f, dtype=float)
    num = (Ql / np.abs(absQc)) * np.exp(1j * phi)
    return _env(f, a, alpha, tau) * (1.0 - num / _resonator_denom(f, fr, Ql))


def s21_transmission(f, fr, Ql, absQc, phi=0.0, a=1.0, alpha=0.0, tau=0.0):
    """Through-transmission S21 (Probst eq. 2). Cannot split Qi/Qc without known a."""
    f = np.asarray(f, dtype=float)
    num = (Ql / np.abs(absQc)) * np.exp(1j * phi)
    return _env(f, a, alpha, tau) * (num / _resonator_denom(f, fr, Ql))


def s11_reflection(f, fr, Ql, absQc, phi=0.0, a=1.0, alpha=0.0, tau=0.0):
    """Single-port reflection, diameter-corrected form.

    Canonical S11 = 1 - 2 (Ql/|Qc|) e^{i phi} / (1 + 2 i Ql (f/fr-1)),
    i.e. a notch of diameter 2 Ql/|Qc| (critical coupling when Ql = |Qc|/2).
    """
    f = np.asarray(f, dtype=float)
    num = 2.0 * (Ql / np.abs(absQc)) * np.exp(1j * phi)
    return _env(f, a, alpha, tau) * (1.0 - num / _resonator_denom(f, fr, Ql))


def model_s(f, params: ResonatorParams):
    kw = dict(
        fr=params.fr,
        Ql=params.Ql,
        absQc=params.absQc,
        phi=params.phi,
        a=params.a,
        alpha=params.alpha,
        tau=params.tau,
    )
    if params.geometry == "notch":
        return s21_notch(f, **kw)
    if params.geometry == "transmission":
        return s21_transmission(f, **kw)
    if params.geometry == "reflection":
        return s11_reflection(f, **kw)
    raise ValueError(f"unknown geometry {params.geometry}")


def quality_from_linewidth(fr_hz: float, fwhm_hz: float) -> float:
    """Q = f_r / Delta f_FWHM.  Do **not** insert an extra factor of 2."""
    return float(fr_hz / fwhm_hz)


def linewidth_hz(fr_hz: float, Q: float) -> float:
    return float(fr_hz / Q)


def wrap_phase(phi):
    """Map angle(s) to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(phi, dtype=float)))


def coupling_regime(geometry: Geometry, Ql: float, absQc: float, phi: float = 0.0) -> str:
    """undercoupled / critical / overcoupled from loaded and coupling Q.

    Energy rates κ = ω/Q.  Overcoupled means the port takes more energy per
    cycle than internal loss: κ_c > κ_i, i.e. |Q_c| < Q_i.

    Reflection (one-port S11)
        Critical when Q_i = |Q_c|, hence Q_l = |Q_c|/2.  Then S11(f_r) = 0
        and the resonance circle passes through the origin.
    Notch / hanger (S21)
        Same comparison of κ_c and κ_i after diameter correction.
        Overcoupled: deep dip (diameter d = Q_l/|Q_c| closer to 1).
    Transmission (two-port S21 peak)
        Each port has its own Q_c; a single |Q_c| in this model is lumped.
        Delay alone cannot decide over vs under.
    """
    cos = float(np.cos(phi))
    inv_qc = cos / absQc if absQc else 0.0
    inv_ql = 1.0 / Ql if Ql else 0.0
    inv_qi = inv_ql - inv_qc
    if inv_qi <= 0:
        return "overcoupled"
    qi = 1.0 / inv_qi
    ratio = qi / absQc if absQc else np.inf
    if abs(ratio - 1.0) < 0.08:
        return "critical"
    if geometry == "reflection":
        return "overcoupled" if ratio > 1.0 else "undercoupled"
    return "overcoupled" if ratio > 1.0 else "undercoupled"
