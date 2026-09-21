"""Synthetic traces and a few IO helpers for S21 tables."""

from __future__ import annotations

import numpy as np

from .models import ResonatorParams, model_s


def make_trace(params: ResonatorParams, n=401, span_bw=8.0, snr=None, seed=0):
    """Evenly spaced frequencies over ``span_bw`` loaded bandwidths.

    If ``snr`` is set, complex Gaussian noise with rms = r0/SNR is added,
    matching the Probst definition at the circle radius.
    """
    bw = params.fr / params.Ql
    f = np.linspace(params.fr - 0.5 * span_bw * bw, params.fr + 0.5 * span_bw * bw, n)
    s = model_s(f, params)
    if snr is not None:
        rng = np.random.default_rng(seed)
        # Hanger/through S21: circle radius a * Ql/(2 |Qc|).
        # S11: diameter 2 Ql/|Qc|, radius a * Ql/|Qc|.
        if params.formula == "reflection":
            r0 = params.a * params.Ql / max(params.absQc, 1e-30)
        else:
            r0 = params.a * params.Ql / (2.0 * max(params.absQc, 1e-30))
        sigma = r0 / float(snr)
        s = s + sigma * (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    return f, s


def load_two_column_complex(path, freq_unit="Hz"):
    """Load freq, Re, Im or freq, mag, phase(deg) CSV/txt (auto)."""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] < 3:
        raise ValueError("need at least 3 columns")
    f = data[:, 0]
    if freq_unit.lower() in ("ghz", "g"):
        f = f * 1e9
    elif freq_unit.lower() in ("mhz", "m"):
        f = f * 1e6
    a, b = data[:, 1], data[:, 2]
    # heuristic: if |a| looks like dB (mostly negative, < 5) treat as dB+deg
    if np.nanmax(np.abs(a)) < 80 and np.nanmean(a) < 5:
        mag = 10 ** (a / 20.0)
        s = mag * np.exp(1j * np.deg2rad(b))
    else:
        s = a + 1j * b
    return f, s
