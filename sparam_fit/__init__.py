"""Robust fitting of superconducting microwave resonator S-parameters.

Implements the Probst / Ustinov circle-fit pipeline (RSI 2015, arXiv:1410.3365)
together with sequential amplitude+phase and magnitude-only fits.

Frequencies are always in Hz.  Quality factors are dimensionless.
Linewidths are reported as FWHM in Hz:  kappa_Hz = f_r / Q.
"""

from .models import (
    ResonatorParams,
    s21_notch,
    s21_transmission,
    s11_reflection,
    quality_from_linewidth,
    linewidth_hz,
)
from .fit import (
    FitResult,
    fit_circle,
    fit_amp_phase,
    fit_magnitude_only,
    fit_hybrid,
    fit_group_delay,
    fit_group_delay_vs_power,
)
from .group_delay import (
    group_delay_from_s,
    group_delay_model,
    phase_from_group_delay,
)

__all__ = [
    "ResonatorParams",
    "s21_notch",
    "s21_transmission",
    "s11_reflection",
    "quality_from_linewidth",
    "linewidth_hz",
    "FitResult",
    "fit_circle",
    "fit_amp_phase",
    "fit_magnitude_only",
    "fit_hybrid",
    "fit_group_delay",
    "fit_group_delay_vs_power",
    "group_delay_from_s",
    "group_delay_model",
    "phase_from_group_delay",
]
