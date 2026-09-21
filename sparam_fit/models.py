from __future__ import annotations

import warnings
from dataclasses import InitVar, asdict, dataclass
from typing import Literal

import numpy as np

Layout = Literal["hanger", "through", "direct"]
SParam = Literal["S21", "S11"]
Formula = Literal["notch", "through", "reflection"]

# Chip layout × measured S-parameter → which row of the S(f) table.
# Same table as docs/PHYSICS_AND_API.md §3.
_ALLOWED = {
    ("hanger", "S21"): "notch",  # env · (1 − β/D)
    ("through", "S21"): "through",  # env · (β/D)
    ("hanger", "S11"): "reflection",  # env · (1 − 2β/D), short/open far end
    ("direct", "S11"): "reflection",  # env · (1 − 2β/D), true one-port
}

# Deprecated geometry= names → (layout, s_param).
_GEOMETRY_MAP = {
    "notch": ("hanger", "S21"),
    "transmission": ("through", "S21"),
    "reflection": ("hanger", "S11"),
}


def formula_name(layout: str, s_param: str) -> Formula:
    """Return ``notch`` / ``through`` / ``reflection`` for this layout × S."""
    layout, s_param = validate_layout_sparam(layout, s_param)
    return _ALLOWED[(layout, s_param)]


def validate_layout_sparam(layout: str, s_param: str) -> tuple[str, str]:
    """Normalize and accept only the physical combinations in the formula table."""
    layout = str(layout)
    s_param = str(s_param).upper()
    if s_param in ("21", "S_21"):
        s_param = "S21"
    elif s_param in ("11", "S_11"):
        s_param = "S11"
    if (layout, s_param) not in _ALLOWED:
        raise ValueError(
            f"layout={layout!r} with s_param={s_param!r} is not a physical combo. "
            "Allowed: hanger+S21 (matched feedline), hanger+S11 (short/open far end), "
            "through+S21, direct+S11.  through+S11 and direct+S21 are invalid.  "
            "See docs/PHYSICS_AND_API.md."
        )
    return layout, s_param


def from_geometry(geometry: str) -> tuple[str, str]:
    """Map the old ``geometry=`` flag onto ``(layout, s_param)``."""
    if geometry not in _GEOMETRY_MAP:
        raise ValueError(
            f"unknown geometry {geometry!r}; use layout= and s_param= "
            "(hanger/through/direct × S21/S11)"
        )
    return _GEOMETRY_MAP[geometry]


def _warn_geometry(geometry: str, stacklevel: int = 3) -> tuple[str, str]:
    warnings.warn(
        "geometry= is deprecated; pass layout= and s_param= instead "
        f"(geometry={geometry!r} → layout={_GEOMETRY_MAP[geometry][0]!r}, "
        f"s_param={_GEOMETRY_MAP[geometry][1]!r}).  See docs/PHYSICS_AND_API.md.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return from_geometry(geometry)


def resolve_layout_sparam(layout="hanger", s_param="S21", geometry=None, stacklevel=3):
    """Resolve fit kwargs.  ``geometry`` is accepted only as a deprecated alias."""
    if geometry is not None:
        return _warn_geometry(geometry, stacklevel=stacklevel)
    if layout in _GEOMETRY_MAP:
        return _warn_geometry(layout, stacklevel=stacklevel)
    return validate_layout_sparam(layout, s_param)


@dataclass
class ResonatorParams:
    """Physical + environment parameters of one resonator pole.

    Environment (always the same)::

        env(f) = a * exp(i*alpha) * exp(-2*pi*i*f*tau)

    Resonator pole::

        D = 1 + 2*i*Ql*(f/fr - 1)
        β = (Ql / |Qc|) * exp(i*phi)

    The S-parameter is selected only by ``layout`` × ``s_param``
    (docs/PHYSICS_AND_API.md §3)::

        hanger  × S21  →  env * (1 − β/D)     # matched feedline, notch
        through × S21  →  env * (β/D)         # in-line resonator
        hanger  × S11  →  env * (1 − 2β/D)    # short/open far end
        direct  × S11  →  env * (1 − 2β/D)    # true one-port

    ``Qc`` stored here is |Qc|.  Diameter-corrected internal Q uses
    ``1/Qi = 1/Ql − cos(phi)/|Qc|`` (Khalil DCM).
    """

    fr: float
    Ql: float
    absQc: float
    phi: float = 0.0
    a: float = 1.0
    alpha: float = 0.0
    tau: float = 0.0
    layout: str = "hanger"
    s_param: str = "S21"
    geometry: InitVar[str | None] = None

    def __post_init__(self, geometry):
        if geometry is not None:
            self.layout, self.s_param = _warn_geometry(geometry, stacklevel=4)
        else:
            self.layout, self.s_param = resolve_layout_sparam(
                self.layout, self.s_param, stacklevel=4
            )

    @property
    def formula(self) -> Formula:
        return formula_name(self.layout, self.s_param)

    @property
    def Qc_complex(self) -> complex:
        return self.absQc * np.exp(-1j * self.phi)

    @property
    def Qi_dia_corr(self) -> float:
        if not np.isfinite(self.absQc) or self.absQc == 0 or not np.isfinite(self.Ql):
            return float("nan")
        inv = 1.0 / self.Ql - np.cos(self.phi) / self.absQc
        return float(1.0 / inv) if inv != 0 else np.inf

    @property
    def Qi_no_corr(self) -> float:
        if not np.isfinite(self.absQc) or self.absQc == 0 or not np.isfinite(self.Ql):
            return float("nan")
        inv = 1.0 / self.Ql - 1.0 / self.absQc
        return float(1.0 / inv) if inv != 0 else np.inf

    @property
    def Qc_dia_corr(self) -> float:
        """1 / Re(1/Qc_complex) = |Qc| / cos(phi)."""
        if not np.isfinite(self.absQc) or self.absQc == 0:
            return float("nan")
        c = np.cos(self.phi)
        return float(self.absQc / c) if c != 0 else np.inf

    @property
    def kappa_l_hz(self) -> float:
        """Loaded FWHM in Hz (energy decay rate / 2pi)."""
        return self.fr / self.Ql

    @property
    def kappa_i_hz(self) -> float:
        qi = self.Qi_dia_corr
        return self.fr / qi if np.isfinite(qi) and qi != 0 else float("nan")

    @property
    def kappa_c_hz(self) -> float:
        qc = self.Qc_dia_corr
        return self.fr / qc if np.isfinite(qc) and qc != 0 else float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            {
                "formula": self.formula,
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
    """Hanger × matched S21 (Probst eq. 1): env · (1 − β/D).  ``f`` in Hz."""
    f = np.asarray(f, dtype=float)
    num = (Ql / np.abs(absQc)) * np.exp(1j * phi)
    return _env(f, a, alpha, tau) * (1.0 - num / _resonator_denom(f, fr, Ql))


def s21_transmission(f, fr, Ql, absQc, phi=0.0, a=1.0, alpha=0.0, tau=0.0):
    """Through × S21 (Probst eq. 2): env · (β/D).  a and |Qc| stay degenerate."""
    f = np.asarray(f, dtype=float)
    num = (Ql / np.abs(absQc)) * np.exp(1j * phi)
    return _env(f, a, alpha, tau) * (num / _resonator_denom(f, fr, Ql))


def s11_reflection(f, fr, Ql, absQc, phi=0.0, a=1.0, alpha=0.0, tau=0.0):
    """Direct S11, or hanger × S11 with short/open far end, near one mode.

    env · (1 − 2β/D).  Circle diameter 2 Ql/|Qc|; critical coupling when
    Ql = |Qc|/2 (circle through the origin, group delay diverges).
    """
    f = np.asarray(f, dtype=float)
    num = 2.0 * (Ql / np.abs(absQc)) * np.exp(1j * phi)
    return _env(f, a, alpha, tau) * (1.0 - num / _resonator_denom(f, fr, Ql))


def model_s(f, params: ResonatorParams):
    """Evaluate the S-parameter selected by ``params.layout`` × ``params.s_param``."""
    kw = dict(
        fr=params.fr,
        Ql=params.Ql,
        absQc=params.absQc,
        phi=params.phi,
        a=params.a,
        alpha=params.alpha,
        tau=params.tau,
    )
    kind = params.formula
    if kind == "notch":
        return s21_notch(f, **kw)
    if kind == "through":
        return s21_transmission(f, **kw)
    if kind == "reflection":
        return s11_reflection(f, **kw)
    raise ValueError(f"unknown formula {kind}")


def absQc_from_canonical_radius(Ql: float, r0: float, layout: str, s_param: str) -> float:
    """|Qc| from the canonical-circle radius after environment stripping.

    Hanger S21 / through S21: diameter = Ql/|Qc|, so r0 = Ql/(2 |Qc|).
    S11 (hanger short/open or direct): diameter = 2 Ql/|Qc|, so r0 = Ql/|Qc|.
    """
    kind = formula_name(layout, s_param)
    r0 = max(float(r0), 1e-12)
    if kind == "reflection":
        return float(Ql / r0)
    return float(Ql / (2.0 * r0))


def quality_from_linewidth(fr_hz: float, fwhm_hz: float) -> float:
    """Q = f_r / Delta f_FWHM.  Do **not** insert an extra factor of 2."""
    return float(fr_hz / fwhm_hz)


def linewidth_hz(fr_hz: float, Q: float) -> float:
    return float(fr_hz / Q)


def wrap_phase(phi):
    """Map angle(s) to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(phi, dtype=float)))


def coupling_regime(
    Ql: float,
    absQc: float,
    phi: float = 0.0,
    *,
    layout: str = "hanger",
    s_param: str = "S21",
    geometry: str | None = None,
) -> str:
    """undercoupled / critical / overcoupled from loaded and coupling Q.

    Energy rates κ ∝ 1/Q.  Overcoupled means the port takes more energy per
    cycle than internal loss: κ_c > κ_i, i.e. |Q_c| < Q_i.

    This is a *regime of the same formula*, not a fourth model.  S11 at
    critical coupling has S11(f_r) = 0 and the group delay diverges.

    ``geometry`` is a deprecated alias for the old notch/transmission/reflection
    flag; it is mapped onto layout × s_param and otherwise ignored.
    """
    if geometry is not None:
        layout, s_param = _warn_geometry(geometry, stacklevel=3)
    else:
        validate_layout_sparam(layout, s_param)
    if not np.isfinite(absQc) or absQc == 0 or not np.isfinite(Ql) or Ql == 0:
        return "unknown"
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
    return "overcoupled" if ratio > 1.0 else "undercoupled"
