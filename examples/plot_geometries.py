"""Visual comparison of notch / transmission / reflection and coupling."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparam_fit.models import (
    ResonatorParams,
    coupling_regime,
    model_s,
)
from sparam_fit.group_delay import delay_lorentzian, group_delay_model
from sparam_fit.fit import fit_group_delay
from sparam_fit.plots import plot_group_delay_fit


def _params(geometry, fr, Ql, absQc, tau=0.0):
    return ResonatorParams(
        fr=fr, Ql=Ql, absQc=absQc, phi=0.0, a=1.0, tau=tau, geometry=geometry
    )


def plot_geometry_atlas(out: Path):
    """|S|, complex plane, group delay for three layouts × two couplings."""
    fr = 6.957e9
    f = np.linspace(fr - 12e6, fr + 12e6, 1201)
    # Same Ql.  Reflection: critical is absQc = 2 Ql.
    # Notch: d = Ql/|Qc|; under d=0.3, over d=0.85.
    cases = [
        ("notch", "undercoupled", 5000, 5000 / 0.30),
        ("notch", "overcoupled", 5000, 5000 / 0.85),
        ("transmission", "weak couple", 5000, 20000),
        ("transmission", "strong couple", 5000, 6000),
        ("reflection", "undercoupled", 4000, 20000),
        ("reflection", "overcoupled", 4000, 6000),
    ]
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 14))
    for i, (geom, label, Ql, Qc) in enumerate(cases):
        p = _params(geom, fr, Ql, Qc, tau=80e-9)
        s = model_s(f, p)
        tg = group_delay_model(f, p)
        regime = coupling_regime(geom, Ql, Qc)
        axes[i, 0].plot(f * 1e-9, 20 * np.log10(np.abs(s) + 1e-12))
        axes[i, 0].set_ylabel("|S| (dB)")
        axes[i, 0].set_title(f"{geom}  {label}  [{regime}]")
        axes[i, 1].plot(s.real, s.imag)
        axes[i, 1].plot(0, 0, "k+", ms=8)
        axes[i, 1].set_aspect("equal", adjustable="datalim")
        axes[i, 1].set_xlabel("Re S")
        axes[i, 1].set_ylabel("Im S")
        axes[i, 2].plot(f * 1e-9, tg * 1e9)
        axes[i, 2].set_ylabel("tau_g (ns)")
        if i == len(cases) - 1:
            axes[i, 0].set_xlabel("f (GHz)")
            axes[i, 2].set_xlabel("f (GHz)")
    fig.suptitle(
        "S11 = reflection;  S21 hanger = notch;  S21 peak = transmission\n"
        "overcoupled: circle encloses 0 (S11) or deep dip (notch).  "
        "Transmission delay is independent of Qc.",
        fontsize=11,
    )
    fig.tight_layout()
    path = out / "geometry_atlas_s_and_delay.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return path


def plot_why_sparam_fails_on_vna_peak(out: Path):
    """Reproduce the user's overlay: true Lorentzian vs reflection S-model."""
    fr, Ql, tau = 6.95713e9, 6000.0, 95e-9
    f = np.linspace(6.948e9, 6.967e9, 401)
    amp = Ql / (np.pi * fr)
    tg = delay_lorentzian(f, fr, Ql, tau, amp=amp)
    rng = np.random.default_rng(1)
    tg = tg + 4e-9 * rng.normal(size=f.size)

    r_lor = fit_group_delay(f, tg, model="lorentzian")
    r_ref = fit_group_delay(f, tg, geometry="reflection", model="sparam")
    r_not = fit_group_delay(f, tg, geometry="notch", model="sparam")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(f * 1e-9, tg * 1e9, ".", ms=4, label="data (Lorentzian+noise)")
    ax.plot(
        f * 1e-9,
        r_lor.diagnostics["tau_g_model"] * 1e9,
        "-",
        lw=2,
        label=f"lorentzian  Ql={r_lor.params.Ql:.0f}  tau={r_lor.params.tau*1e9:.1f} ns",
    )
    ax.plot(
        f * 1e-9,
        r_ref.diagnostics["tau_g_model"] * 1e9,
        "--",
        label=f"S11 reflection  Ql={r_ref.params.Ql:.0f}  tau={r_ref.params.tau*1e9:.1f} ns",
    )
    ax.plot(
        f * 1e-9,
        r_not.diagnostics["tau_g_model"] * 1e9,
        ":",
        label=f"S21 notch (sign auto)  Ql={r_not.params.Ql:.0f}",
    )
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel("group delay (ns)")
    ax.set_title("Why the S-parameter delay model misses a VNA peak")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "delay_lorentzian_vs_sparam.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plot_group_delay_fit(
        f, tg, r_lor, title="lorentzian (use this on 53672)", save_path=out / "delay_lorentzian_fit.png"
    )
    return path, r_lor, r_ref, r_not


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    plot_geometry_atlas(out)
    _, r_lor, r_ref, r_not = plot_why_sparam_fails_on_vna_peak(out)
    (out / "geometry_and_delay_notes.txt").write_text(
        "LORENTZIAN\n"
        + r_lor.summary()
        + "\n\nREFLECTION SPARAM\n"
        + r_ref.summary()
        + "\n\nNOTCH SPARAM\n"
        + r_not.summary()
        + "\n"
    )
    print(r_lor.summary())
    print("wrote", out)


if __name__ == "__main__":
    main(Path("examples/artifacts"))
