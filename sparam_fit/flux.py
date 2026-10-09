"""Resonator frequency versus external flux, and fits at the sweet spots.

One-tone map: complex S (usually hanger S21) on a grid of frequency and a
bias that sets the external flux (coil current or flux-line voltage).
The ridge is the *readout resonator*, pulled by a flux-tunable qubit.
Sweet spots are extrema of that ridge, df_r/dbias = 0.  They are not the
inflection points.

See docs/PHYSICS_AND_API.md section 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from scipy.signal import savgol_filter

from .fit import fit_amp_phase, fit_circle, fit_hybrid, fit_magnitude_only
from .models import ResonatorParams, model_s


@dataclass
class ResonanceTrack:
    """Resonator frequency read off each bias column of a map."""

    bias: np.ndarray
    fr_hz: np.ndarray
    depth: np.ndarray
    fwhm_hz: np.ndarray
    freq_index: np.ndarray
    segments: list


@dataclass
class FluxFeature:
    """One extremum or inflection of f_r(bias)."""

    kind: str
    extremum: str
    bias: float
    fr_hz: float
    dfr_dbias: float
    d2fr_dbias2: float
    sharpness: float
    prominence_hz: float
    qubit_role: str = "unassigned"
    closest_approach: bool = False
    segment: int = 0

    @property
    def curvature(self) -> float:
        """Geometric curvature |f''| / (1 + f'^2)^(3/2). Equals sharpness at a sweet spot."""
        return float(abs(self.d2fr_dbias2) / (1.0 + self.dfr_dbias**2) ** 1.5)


@dataclass
class FluxCutFit:
    """Complex (or magnitude) cut at one sweet spot, plus the library fit."""

    feature: FluxFeature
    freq: np.ndarray
    s: np.ndarray
    bias: float
    interpolated: bool
    fit: object = None
    fit_amp_phase: object = None
    fit_circle: object = None


@dataclass
class FluxAnalysis:
    """Full reduction of one flux map."""

    freq: np.ndarray
    bias: np.ndarray
    s: np.ndarray
    track: ResonanceTrack
    features: list
    cuts: list
    layout: str
    s_param: str
    feature: str
    qubit_side: str
    primary_index: int | None
    notes: list = field(default_factory=list)
    swapped_axes: bool = False

    @property
    def sweet_spots(self) -> list:
        return [f for f in self.features if f.kind == "sweet_spot"]

    @property
    def inflections(self) -> list:
        return [f for f in self.features if f.kind == "inflection"]

    @property
    def primary(self) -> FluxCutFit | None:
        if self.primary_index is None or not self.cuts:
            return None
        return self.cuts[self.primary_index]

    def summary(self) -> str:
        lines = [
            f"flux map      : {self.freq.size} freq × {self.bias.size} bias",
            f"layout/S      : {self.layout} × {self.s_param}  (track feature: {self.feature})",
            f"qubit side    : {self.qubit_side}",
            "sweet spot    : df_r/dbias = 0.  An inflection (d²f_r = 0) is the steepest slope, not a sweet spot.",
        ]
        if self.swapped_axes:
            lines.append("axes          : frequency and bias arguments were swapped")
        for note in self.notes:
            lines.append(f"note          : {note}")
        if not self.sweet_spots:
            lines.append("sweet spots   : none inside the scan")
        else:
            lines.append(f"sweet spots   : {len(self.sweet_spots)}")
            for spot in self.sweet_spots:
                lines.append(
                    f"  {spot.qubit_role:12s} {spot.extremum:12s}  "
                    f"bias={spot.bias:.6e}  f_r={spot.fr_hz:.6e} Hz  "
                    f"sharpness={spot.sharpness:.3e}"
                    + ("  closest" if spot.closest_approach else "")
                )
        if self.inflections:
            lines.append(f"inflections   : {len(self.inflections)} (not fitted)")
        prim = self.primary
        if prim is not None and prim.fit is not None:
            lines.append("primary fit   :")
            lines.append(prim.fit.summary())
        elif prim is not None:
            lines.append(
                f"primary cut   : {prim.feature.qubit_role} at bias={prim.bias:.6e} (not fitted)"
            )
        return "\n".join(lines)


def orient_flux_map(freq, bias, data, auto_swap: bool = True):
    """Return sorted ``freq`` (n_f,), ``bias`` (n_b,), ``S`` (n_f, n_b).

    ``data`` may be complex S or real |S|, shaped ``(n_f, n_b)`` or
    ``(n_b, n_f)``.  If the array passed as frequency looks like a coil
    current and the other axis looks like a microwave frequency, the two
    are swapped.  Duplicate bias (or frequency) samples are averaged.
    """
    freq = np.asarray(freq, dtype=float).ravel()
    bias = np.asarray(bias, dtype=float).ravel()
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"S must be 2-D, got shape {data.shape}")
    swapped = False
    if (
        auto_swap
        and freq.size
        and bias.size
        and np.median(np.abs(freq)) < 1e5
        and np.median(np.abs(bias)) > 1e6
    ):
        freq, bias = bias, freq
        swapped = True
    nf, nb = freq.size, bias.size
    if data.shape == (nf, nb):
        s = np.array(data, copy=True)
    elif data.shape == (nb, nf):
        s = np.array(data, copy=True).T
    else:
        raise ValueError(
            f"S shape {data.shape} matches neither (n_freq={nf}, n_bias={nb}) "
            f"nor (n_bias, n_freq)"
        )
    if nf > 1 and np.any(np.diff(freq) < 0):
        order = np.argsort(freq, kind="mergesort")
        freq = freq[order]
        s = s[order, :]
    if nb > 1 and np.any(np.diff(bias) < 0):
        order = np.argsort(bias, kind="mergesort")
        bias = bias[order]
        s = s[:, order]
    freq, s = _average_duplicates(freq, s, axis=0)
    bias, s = _average_duplicates(bias, s, axis=1)
    return freq, bias, s, swapped


def resonator_frequency_vs_bias(
    bias,
    *,
    fr_bare_hz: float,
    g_hz: float,
    fq_max_hz: float,
    fq_min_hz: float,
    period: float,
    bias_at_upper: float,
):
    """Dispersive readout frequency of a flux-tunable transmon.

    Reduced flux is ``φ = (bias - bias_at_upper) / period`` (1 at one flux
    quantum).  Junction asymmetry is fixed by the two sweet-spot qubit
    frequencies, with ``f_q ∝ sqrt(E_J)``::

        E_J(φ) / E_J(0) = sqrt(cos²(πφ) + d² sin²(πφ))
        d = (f_q,min / f_q,max)²

    Qubit in |g⟩, two-level level repulsion (valid for |f_q - f_r| ≫ g)::

        f_r(φ) = f_r,bare − g² / (f_q(φ) − f_r,bare)

    φ = integer is the upper sweet spot (maximum f_q).  φ = half-integer
    is the lower sweet spot (minimum f_q).  Returns ``(f_r, f_q)`` in Hz.
    """
    bias = np.asarray(bias, dtype=float)
    if period == 0:
        raise ValueError("period must be nonzero")
    if not (0 < fq_min_hz < fq_max_hz):
        raise ValueError("need 0 < fq_min_hz < fq_max_hz")
    d = (fq_min_hz / fq_max_hz) ** 2
    phi = (bias - bias_at_upper) / period
    ej = np.sqrt(np.cos(np.pi * phi) ** 2 + (d * np.sin(np.pi * phi)) ** 2)
    fq = fq_max_hz * np.sqrt(ej)
    delta = fq - fr_bare_hz
    if np.any(np.abs(delta) < 5.0 * abs(g_hz)):
        raise ValueError(
            "qubit comes within 5g of the resonator; this helper is dispersive only"
        )
    fr = fr_bare_hz - (g_hz**2) / delta
    return fr, fq


def make_dispersive_flux_map(
    bias,
    *,
    fr_bare_hz: float,
    g_hz: float,
    fq_max_hz: float,
    fq_min_hz: float,
    period: float,
    bias_at_upper: float,
    Ql: float,
    absQc: float,
    phi: float = 0.0,
    a: float = 1.0,
    alpha: float = 0.0,
    tau: float = 0.0,
    layout: str = "hanger",
    s_param: str = "S21",
    n_freq: int = 401,
    span_bw: float = 10.0,
    snr: float | None = None,
    seed: int = 0,
):
    """Synthetic one-tone map.  Returns ``freq, bias, S, f_r, f_q``.

    ``S`` has shape ``(n_freq, n_bias)``.  ``span_bw`` is the half-window,
    in loaded linewidths, added beyond the min and max of f_r.
    """
    bias = np.asarray(bias, dtype=float)
    fr, fq = resonator_frequency_vs_bias(
        bias,
        fr_bare_hz=fr_bare_hz,
        g_hz=g_hz,
        fq_max_hz=fq_max_hz,
        fq_min_hz=fq_min_hz,
        period=period,
        bias_at_upper=bias_at_upper,
    )
    fwhm = float(np.median(fr) / Ql)
    freq = np.linspace(
        float(fr.min()) - 0.5 * span_bw * fwhm,
        float(fr.max()) + 0.5 * span_bw * fwhm,
        int(n_freq),
    )
    s = np.empty((freq.size, bias.size), dtype=np.complex128)
    for j, frj in enumerate(fr):
        params = ResonatorParams(
            fr=float(frj),
            Ql=Ql,
            absQc=absQc,
            phi=phi,
            a=a,
            alpha=alpha,
            tau=tau,
            layout=layout,
            s_param=s_param,
        )
        s[:, j] = model_s(freq, params)
    if snr is not None:
        rng = np.random.default_rng(seed)
        if layout == "hanger" and str(s_param).upper() == "S11":
            r0 = a * Ql / max(absQc, 1e-30)
        else:
            r0 = a * Ql / (2.0 * max(absQc, 1e-30))
        sigma = r0 / float(snr)
        noise = sigma * (
            rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape)
        ) / np.sqrt(2.0)
        s = s + noise
    return freq, bias, s, fr, fq


def track_resonance(
    freq,
    bias,
    s,
    *,
    feature: str = "dip",
    sigma_bias: float = 0.6,
):
    """Follow the resonator across bias.

    For each bias column, subtract a linear baseline fit to the frequency
    edges and take the deepest dip (``feature='dip'``, hanger |S21|) or
    the highest peak (``feature='peak'``, through |S21|).  The frequency
    is refined by a 3-point parabola.  Tracking uses a light smooth along
    bias; the returned map is not modified.
    """
    if feature not in ("dip", "peak"):
        raise ValueError("feature must be 'dip' or 'peak'")
    freq = np.asarray(freq, dtype=float)
    bias = np.asarray(bias, dtype=float)
    mag = np.abs(np.asarray(s))
    if mag.shape != (freq.size, bias.size):
        raise ValueError(
            f"|S| shape {mag.shape} != (n_freq={freq.size}, n_bias={bias.size})"
        )
    if sigma_bias and bias.size > 2:
        mag_t = gaussian_filter1d(mag, float(sigma_bias), axis=1, mode="nearest")
    else:
        mag_t = mag
    fr = np.empty(bias.size, dtype=float)
    depth = np.empty(bias.size, dtype=float)
    fwhm = np.empty(bias.size, dtype=float)
    index = np.empty(bias.size, dtype=int)
    for j in range(bias.size):
        score = _column_score(freq, mag_t[:, j], feature)
        i = int(np.argmax(score))
        fr[j], depth[j] = _parabola_vertex(freq, score, i)
        fwhm[j] = _score_fwhm(freq, score, i)
        index[j] = i
    segments = jump_segments(bias, fr, fwhm)
    return ResonanceTrack(
        bias=np.asarray(bias, dtype=float),
        fr_hz=fr,
        depth=depth,
        fwhm_hz=fwhm,
        freq_index=index,
        segments=segments,
    )


def find_flux_features(
    bias,
    fr,
    *,
    fwhm=None,
    smooth_window: int | None = None,
    segments: list | None = None,
):
    """Extrema (sweet spots) and inflections of a resonator track.

    Sweet spots are interior zeros of df_r/dbias.  A maximum or minimum
    that only occurs on the endpoint of the scan is not reported: the
    derivative has not changed sign inside the data.  Inflections are
    interior zeros of d²f_r/dbias² where the slope is still large.
    """
    bias = np.asarray(bias, dtype=float)
    fr = np.asarray(fr, dtype=float)
    if fwhm is None:
        fwhm = np.full(bias.shape, np.nan)
    else:
        fwhm = np.asarray(fwhm, dtype=float)
    if segments is None:
        segments = jump_segments(bias, fr, fwhm)
    features: list[FluxFeature] = []
    for seg_i, (i0, i1) in enumerate(segments):
        features.extend(
            _features_on_segment(bias[i0 : i1 + 1], fr[i0 : i1 + 1], smooth_window, seg_i)
        )
    return features


def assign_qubit_sweet_spots(features, qubit_side: str | None = None):
    """Label extrema as upper or lower qubit sweet spots.

    Level repulsion: the resonator is pulled *down* when the qubit is above
    it and pushed *up* when the qubit is below it.  The sharper extremum
    is the closer approach.

    * qubit above the resonator → sharp f_r minimum = lower sweet spot,
      flatter f_r maximum = upper sweet spot.
    * qubit below the resonator → sharp f_r maximum = upper sweet spot,
      flatter f_r minimum = lower sweet spot.

    ``qubit_side`` may be ``'above'``, ``'below'``, or ``None`` to infer it
    from which extremum is sharper.  Inference needs both a maximum and a
    minimum inside the scan.  Returns ``(qubit_side, notes)``.
    """
    spots = [f for f in features if f.kind == "sweet_spot"]
    maxima = [f for f in spots if f.extremum == "fr_maximum"]
    minima = [f for f in spots if f.extremum == "fr_minimum"]
    notes: list[str] = []
    side = "unknown"
    if qubit_side in ("above", "below"):
        side = qubit_side
        notes.append(f"qubit side taken from the caller ({side})")
    elif qubit_side is not None:
        raise ValueError("qubit_side must be 'above', 'below', or None")
    elif maxima and minima:
        sharp_max = float(np.median([f.sharpness for f in maxima]))
        sharp_min = float(np.median([f.sharpness for f in minima]))
        # A pure cosine has equal curvature at both extrema.  Require a
        # clear difference before calling one of them the closer approach.
        if sharp_max <= 0:
            ratio = np.inf
        else:
            ratio = sharp_min / sharp_max
        if ratio > 1.3:
            side = "above"
        elif ratio < 1.0 / 1.3:
            side = "below"
        else:
            notes.append(
                "the frequency maxima and minima have similar curvature, "
                "so upper versus lower is not assigned"
            )
        if side in ("above", "below"):
            notes.append(
                "qubit side from level repulsion: the sharper extremum is the "
                f"closer approach (sharpness of f_r maxima {sharp_max:.3e}, "
                f"of f_r minima {sharp_min:.3e})"
            )
    else:
        notes.append(
            "only one kind of extremum is inside the scan, so upper versus "
            "lower is not assigned"
        )
    for spot in spots:
        if side == "above":
            spot.qubit_role = "lower" if spot.extremum == "fr_minimum" else "upper"
            spot.closest_approach = spot.extremum == "fr_minimum"
        elif side == "below":
            spot.qubit_role = "upper" if spot.extremum == "fr_maximum" else "lower"
            spot.closest_approach = spot.extremum == "fr_maximum"
        else:
            spot.qubit_role = "unassigned"
            spot.closest_approach = False
    return side, notes


def cut_at_bias(freq, bias, s, bias_value: float, interpolate: bool = True):
    """Complex S(f) at one bias.  Returns ``freq, cut, bias_used, interpolated``."""
    freq = np.asarray(freq, dtype=float)
    bias = np.asarray(bias, dtype=float)
    s = np.asarray(s)
    if s.shape != (freq.size, bias.size):
        raise ValueError("S shape does not match freq and bias")
    if bias.size == 1 or not interpolate:
        j = int(np.argmin(np.abs(bias - bias_value)))
        return freq, np.array(s[:, j], copy=True), float(bias[j]), False
    if bias_value <= bias[0] or bias_value >= bias[-1]:
        j = 0 if bias_value <= bias[0] else bias.size - 1
        return freq, np.array(s[:, j], copy=True), float(bias[j]), False
    j = int(np.searchsorted(bias, bias_value))
    b1, b2 = float(bias[j - 1]), float(bias[j])
    if b2 == b1:
        return freq, np.array(s[:, j], copy=True), b2, False
    t = (bias_value - b1) / (b2 - b1)
    cut = (1.0 - t) * s[:, j - 1] + t * s[:, j]
    return freq, cut, float(bias_value), True


def analyze_flux_map(
    freq,
    bias,
    s,
    *,
    layout: str = "hanger",
    s_param: str = "S21",
    feature: str | None = None,
    fit: str | None = "hybrid",
    which: str = "sweet_spots",
    qubit_side: str | None = None,
    auto_swap: bool = True,
    sigma_bias: float = 0.6,
    smooth_window: int | None = None,
    interpolate_cut: bool = True,
    fit_half_width_fwhm: float = 8.0,
):
    """Track f_r(bias), find sweet spots, fit S at those biases.

    ``which`` selects the cuts that are fitted: ``'sweet_spots'`` (every
    interior extremum), ``'upper'``, ``'lower'``, ``'sharpest'``,
    ``'flattest'``, ``'fr_maximum'``, or ``'fr_minimum'``.

    ``fit`` is ``'hybrid'``, ``'circle'``, ``'amp_phase'``, ``'magnitude'``,
    or ``None``.  A real-valued map is fitted with the magnitude model even
    if ``fit='hybrid'`` was requested.  The highlighted cut is the upper
    sweet spot when that label exists, otherwise the sharper extremum.
    """
    if feature is None:
        feature = "peak" if layout == "through" else "dip"
    freq, bias, s, swapped = orient_flux_map(freq, bias, s, auto_swap=auto_swap)
    track = track_resonance(freq, bias, s, feature=feature, sigma_bias=sigma_bias)
    features = find_flux_features(
        track.bias,
        track.fr_hz,
        fwhm=track.fwhm_hz,
        smooth_window=smooth_window,
        segments=track.segments,
    )
    side, notes = assign_qubit_sweet_spots(features, qubit_side=qubit_side)
    if swapped:
        notes.append("frequency and bias axes were swapped to match the data shape and units")
    selected = _select_features(features, which)
    real_map = not np.iscomplexobj(s) or np.max(np.abs(np.asarray(s).imag)) == 0
    fit_method = fit
    if real_map and fit in ("hybrid", "circle", "amp_phase"):
        fit_method = "magnitude"
        notes.append("map is real-valued, so the cut is fitted with fit_magnitude_only")
    cuts: list[FluxCutFit] = []
    fwhm_med = float(np.nanmedian(track.fwhm_hz)) if track.fwhm_hz.size else float("nan")
    for spot in selected:
        f_cut, col, b_used, interp = cut_at_bias(
            freq, bias, s, spot.bias, interpolate=interpolate_cut
        )
        f_fit, s_fit = _crop_cut(f_cut, col, spot.fr_hz, fwhm_med, fit_half_width_fwhm)
        result, r_ap, r_c = _fit_cut(f_fit, s_fit, layout, s_param, fit_method)
        cuts.append(
            FluxCutFit(
                feature=spot,
                freq=f_fit,
                s=s_fit,
                bias=b_used,
                interpolated=interp,
                fit=result,
                fit_amp_phase=r_ap,
                fit_circle=r_c,
            )
        )
    primary = _primary_index(cuts)
    return FluxAnalysis(
        freq=freq,
        bias=bias,
        s=s,
        track=track,
        features=features,
        cuts=cuts,
        layout=layout,
        s_param=s_param,
        feature=feature,
        qubit_side=side,
        primary_index=primary,
        notes=notes,
        swapped_axes=swapped,
    )


def jump_segments(bias, fr, fwhm):
    """Contiguous index ranges, splitting where the track hops by many linewidths."""
    bias = np.asarray(bias, dtype=float)
    fr = np.asarray(fr, dtype=float)
    n = fr.size
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    fwhm = np.asarray(fwhm, dtype=float)
    step = np.abs(np.diff(fr))
    finite = np.isfinite(fr)
    med = float(np.median(step[np.isfinite(step)])) if np.any(np.isfinite(step)) else 0.0
    fwhm_med = float(np.nanmedian(fwhm)) if np.any(np.isfinite(fwhm)) else 0.0
    if not np.isfinite(fwhm_med):
        fwhm_med = 0.0
    thresh = max(6.0 * med, 3.0 * fwhm_med, 0.0)
    breaks = set()
    for i, hop in enumerate(step):
        neighbors = []
        if i > 0:
            neighbors.append(step[i - 1])
        if i + 1 < step.size:
            neighbors.append(step[i + 1])
        local = float(np.median(neighbors)) if neighbors else med
        if np.isfinite(hop) and hop > max(thresh, 4.0 * local) and hop > med + fwhm_med:
            breaks.add(i)
    for i in range(n):
        if not finite[i]:
            breaks.add(i)
            if i:
                breaks.add(i - 1)
    segments = []
    start = 0
    for i in range(n - 1):
        if i in breaks or not finite[i] or not finite[i + 1]:
            if finite[start:i + 1].sum() >= 5 and i >= start:
                segments.append((start, i))
            start = i + 1
    if finite[start:].sum() >= 5:
        segments.append((start, n - 1))
    if not segments and finite.sum() >= 5:
        idx = np.flatnonzero(finite)
        segments.append((int(idx[0]), int(idx[-1])))
    return segments


def _average_duplicates(x, s, axis: int):
    if x.size < 2 or np.all(np.diff(x) > 0):
        return x, s
    uniq, inverse = np.unique(x, return_inverse=True)
    if axis == 0:
        out = np.zeros((uniq.size, s.shape[1]), dtype=np.result_type(s, float))
        counts = np.zeros(uniq.size)
        for i, k in enumerate(inverse):
            out[k] += s[i]
            counts[k] += 1
        out /= counts[:, None]
        return uniq, out
    out = np.zeros((s.shape[0], uniq.size), dtype=np.result_type(s, float))
    counts = np.zeros(uniq.size)
    for i, k in enumerate(inverse):
        out[:, k] += s[:, i]
        counts[k] += 1
    out /= counts
    return uniq, out


def _column_score(freq, mag, feature: str):
    mag = np.asarray(mag, dtype=float)
    n = mag.size
    k = max(n // 12, 3)
    if n < 2 * k:
        base = np.full(n, np.median(mag))
    else:
        xf = np.concatenate([freq[:k], freq[-k:]])
        yf = np.concatenate([mag[:k], mag[-k:]])
        slope, intercept = np.polyfit(xf, yf, 1)
        base = slope * freq + intercept
    if feature == "dip":
        return base - mag
    return mag - base


def _parabola_vertex(x, y, i: int):
    """Vertex of the parabola through samples i-1, i, i+1.  ``y`` is maximized."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if i <= 0 or i >= x.size - 1:
        return float(x[i]), float(y[i])
    x0, x1, x2 = float(x[i - 1]), float(x[i]), float(x[i + 1])
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0.0:
        return x1, y1
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    c = (
        x1 * x2 * (x1 - x2) * y0
        + x2 * x0 * (x2 - x0) * y1
        + x0 * x1 * (x0 - x1) * y2
    ) / denom
    if a == 0.0:
        return x1, y1
    xv = -b / (2.0 * a)
    lo, hi = (x0, x2) if x0 < x2 else (x2, x0)
    if xv < lo or xv > hi:
        return x1, y1
    return float(xv), float(a * xv**2 + b * xv + c)


def _score_fwhm(freq, score, i: int):
    half = 0.5 * float(score[i])
    if half <= 0:
        return float("nan")
    left = i
    while left > 0 and score[left] > half:
        left -= 1
    right = i
    n = score.size
    while right < n - 1 and score[right] > half:
        right += 1
    if left == 0 and score[left] > half:
        return float("nan")
    if right == n - 1 and score[right] > half:
        return float("nan")

    def _cross(i_out, i_in):
        s0, s1 = float(score[i_out]), float(score[i_in])
        if s1 == s0:
            return float(freq[i_in])
        t = (half - s0) / (s1 - s0)
        return float(freq[i_out] + t * (freq[i_in] - freq[i_out]))

    return _cross(right, right - 1) - _cross(left, left + 1)


def _features_on_segment(bias, fr, smooth_window, segment: int):
    n = bias.size
    if n < 7 or np.any(np.diff(bias) <= 0):
        return []
    window = smooth_window if smooth_window is not None else _auto_window(n)
    if window is None or window >= n or window < 5:
        smoothed = np.asarray(fr, dtype=float)
    else:
        if window % 2 == 0:
            window += 1
        if window >= n:
            window = n - 1 if n % 2 == 0 else n
            if window % 2 == 0:
                window -= 1
        smoothed = savgol_filter(fr, window_length=window, polyorder=3)
    sigma = float(np.median(np.abs(fr - smoothed)))
    try:
        spl = UnivariateSpline(bias, smoothed, k=3, s=0)
    except Exception:
        return []
    d1 = spl.derivative(1)
    d2 = spl.derivative(2)
    n_dense = max(8 * n, 800)
    xd = np.linspace(bias[0], bias[-1], n_dense)
    yd = spl(xd)
    g1 = d1(xd)
    span = float(bias[-1] - bias[0])
    # Keep zeros that have data on both sides: drop the outer samples.
    lo = float(bias[1])
    hi = float(bias[-2])
    ptp = float(np.max(smoothed) - np.min(smoothed))
    min_prom = max(sigma, 0.02 * ptp)
    features = []
    sweet_bias = []
    for i in range(n_dense - 1):
        a, b = float(g1[i]), float(g1[i + 1])
        if a == 0.0 or a * b < 0.0:
            root = float(xd[i]) if a == 0.0 else float(brentq(d1, xd[i], xd[i + 1]))
        else:
            continue
        if root <= lo or root >= hi:
            continue
        curv = float(d2(root))
        if a > 0 and b < 0 and curv < 0:
            kind_ext = "fr_maximum"
        elif a < 0 and b > 0 and curv > 0:
            kind_ext = "fr_minimum"
        else:
            continue
        prom = _basin_prominence(xd, yd, root, kind_ext)
        if prom < min_prom:
            continue
        if any(abs(root - prev) < 0.01 * span for prev in sweet_bias):
            continue
        sweet_bias.append(root)
        features.append(
            FluxFeature(
                kind="sweet_spot",
                extremum=kind_ext,
                bias=root,
                fr_hz=float(spl(root)),
                dfr_dbias=float(d1(root)),
                d2fr_dbias2=curv,
                sharpness=abs(curv),
                prominence_hz=float(prom),
                segment=segment,
            )
        )
    g2 = d2(xd)
    max_slope = float(np.max(np.abs(g1))) if g1.size else 0.0
    # Inflections are where the slope is steepest.  A noisy second derivative
    # crosses zero many times; keep one candidate per neighborhood, and only
    # away from the segment ends where the spline rings.
    margin = 0.04 * span
    candidates = []
    for i in range(n_dense - 1):
        a, b = float(g2[i]), float(g2[i + 1])
        if not (a == 0.0 or a * b < 0.0):
            continue
        root = float(xd[i]) if a == 0.0 else float(brentq(d2, xd[i], xd[i + 1]))
        if root <= bias[0] + margin or root >= bias[-1] - margin:
            continue
        slope = float(d1(root))
        if abs(slope) < 0.5 * max_slope:
            continue
        if any(abs(root - prev) < 0.03 * span for prev in sweet_bias):
            continue
        candidates.append((abs(slope), root, slope))
    candidates.sort(reverse=True)
    kept = []
    for mag, root, slope in candidates:
        if any(abs(root - prev) < 0.08 * span for prev in kept):
            continue
        kept.append(root)
        features.append(
            FluxFeature(
                kind="inflection",
                extremum="inflection",
                bias=root,
                fr_hz=float(spl(root)),
                dfr_dbias=slope,
                d2fr_dbias2=float(d2(root)),
                sharpness=mag,
                prominence_hz=0.0,
                segment=segment,
            )
        )
    return features


def _basin_prominence(xd, yd, root: float, kind: str):
    """Height of an extremum above the higher shoulder of its basin.

    The dense-grid index nearest the root is not always the sample at the
    bottom of the valley.  Snap to that sample first, otherwise one side of
    the walk stops immediately and the prominence collapses to zero.
    """
    i = int(np.searchsorted(xd, root))
    i = int(np.clip(i, 1, len(yd) - 2))
    window = yd[i - 1 : i + 2]
    i = i - 1 + int(np.argmax(window) if kind == "fr_maximum" else np.argmin(window))
    height = float(np.interp(root, xd, yd))
    if kind == "fr_maximum":
        left = i
        while left > 0 and yd[left - 1] <= yd[left]:
            left -= 1
        right = i
        while right < len(yd) - 1 and yd[right + 1] <= yd[right]:
            right += 1
        return float(height - max(yd[left], yd[right]))
    left = i
    while left > 0 and yd[left - 1] >= yd[left]:
        left -= 1
    right = i
    while right < len(yd) - 1 and yd[right + 1] >= yd[right]:
        right += 1
    return float(min(yd[left], yd[right]) - height)


def _auto_window(n: int):
    if n < 9:
        return None
    window = int(round(0.06 * n))
    if window % 2 == 0:
        window += 1
    window = max(5, window)
    window = min(window, 21)
    if window >= n:
        return None
    return window


def _select_features(features, which: str):
    spots = [f for f in features if f.kind == "sweet_spot"]
    if which in ("sweet_spots", "both", "all"):
        return spots
    if which == "sharpest":
        return [max(spots, key=lambda f: f.sharpness)] if spots else []
    if which == "flattest":
        return [min(spots, key=lambda f: f.sharpness)] if spots else []
    if which == "upper":
        return [f for f in spots if f.qubit_role == "upper"]
    if which == "lower":
        return [f for f in spots if f.qubit_role == "lower"]
    if which in ("fr_maximum", "fr_minimum"):
        return [f for f in spots if f.extremum == which]
    raise ValueError(
        "which must be sweet_spots, upper, lower, sharpest, flattest, "
        "fr_maximum, or fr_minimum"
    )


def _primary_index(cuts: list[FluxCutFit]):
    if not cuts:
        return None
    uppers = [i for i, c in enumerate(cuts) if c.feature.qubit_role == "upper"]
    if uppers:
        return min(uppers, key=lambda i: -cuts[i].feature.prominence_hz)
    return max(range(len(cuts)), key=lambda i: cuts[i].feature.sharpness)


def _crop_cut(freq, col, fr, fwhm, half_width_fwhm: float):
    if not np.isfinite(fwhm) or fwhm <= 0 or half_width_fwhm <= 0:
        return freq, col
    half = float(half_width_fwhm) * float(fwhm)
    mask = (freq >= fr - half) & (freq <= fr + half)
    if int(mask.sum()) < 25:
        return freq, col
    return freq[mask], col[mask]


def _fit_cut(freq, col, layout, s_param, method):
    if method is None:
        return None, None, None
    if method == "magnitude":
        result = fit_magnitude_only(freq, np.abs(col), layout=layout, s_param=s_param)
        return result, None, None
    if method == "circle":
        result = fit_circle(freq, col, layout=layout, s_param=s_param)
        return result, None, None
    if method == "amp_phase":
        result = fit_amp_phase(freq, col, layout=layout, s_param=s_param)
        return result, None, None
    if method == "hybrid":
        best, r_ap, r_c = fit_hybrid(freq, col, layout=layout, s_param=s_param)
        return best, r_ap, r_c
    raise ValueError("fit must be hybrid, circle, amp_phase, magnitude, or None")
