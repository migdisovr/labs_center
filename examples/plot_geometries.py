"""Visual comparison of the three formula-table rows and coupling regimes."""

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


def _params(layout, s_param, fr, Ql, absQc, tau=0.0):
    return ResonatorParams(
        fr=fr,
        Ql=Ql,
        absQc=absQc,
        phi=0.0,
        a=1.0,
        tau=tau,
        layout=layout,
        s_param=s_param,
    )


def plot_geometry_atlas(out: Path):
    """|S|, complex plane, group delay for the three table rows × coupling."""
    fr = 6.957e9
    f = np.linspace(fr - 12e6, fr + 12e6, 1201)
    cases = [
        ("hanger", "S21", "undercoupled", 5000, 5000 / 0.30),
        ("hanger", "S21", "overcoupled", 5000, 5000 / 0.85),
        ("through", "S21", "weak couple", 5000, 20000),
        ("through", "S21", "strong couple", 5000, 6000),
        ("hanger", "S11", "undercoupled", 4000, 20000),
        ("hanger", "S11", "overcoupled", 4000, 6000),
    ]
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 14))
    for i, (layout, s_param, label, Ql, Qc) in enumerate(cases):
        p = _params(layout, s_param, fr, Ql, Qc, tau=80e-9)
        s = model_s(f, p)
        tg = group_delay_model(f, p)
        regime = coupling_regime(Ql, Qc, layout=layout, s_param=s_param)
        axes[i, 0].plot(f * 1e-9, 20 * np.log10(np.abs(s) + 1e-12))
        axes[i, 0].set_ylabel("|S| (dB)")
        axes[i, 0].set_title(f"{layout}×{s_param}  {label}  [{regime}]")
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
        "hanger×S21 = notch;  through×S21 = peak;  hanger×S11 = reflection\n"
        "overcoupled: circle encloses 0 (S11) or deep dip (notch).  "
        "Through delay is independent of Qc.",
        fontsize=11,
    )
    fig.tight_layout()
    path = out / "geometry_atlas_s_and_delay.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return path


def plot_why_sparam_fails_on_vna_peak(out: Path):
    """A free-amp Lorentzian delay peak is not the S11 (or S21) derivative."""
    fr, Ql, tau = 6.95713e9, 6000.0, 95e-9
    f = np.linspace(6.948e9, 6.967e9, 401)
    amp = Ql / (np.pi * fr)
    tg = delay_lorentzian(f, fr, Ql, tau, amp=amp)
    rng = np.random.default_rng(1)
    tg = tg + 4e-9 * rng.normal(size=f.size)

    r_lor = fit_group_delay(f, tg)
    p_s11 = _params("hanger", "S11", fr, Ql, 2.5 * Ql, tau=tau)
    p_s21 = _params("hanger", "S21", fr, Ql, Ql / 0.5, tau=tau)
    tg_s11 = group_delay_model(f, p_s11)
    tg_s21 = group_delay_model(f, p_s21)

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
        tg_s11 * 1e9,
        "--",
        label="hanger×S11 derivative (same fr, Ql; not a delay fit)",
    )
    ax.plot(
        f * 1e-9,
        tg_s21 * 1e9,
        ":",
        label="hanger×S21 derivative (IEEE dip)",
    )
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel("group delay (ns)")
    ax.set_title("Delay-only data: fit the Lorentzian, not an S-derivative")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "delay_lorentzian_vs_sparam.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plot_group_delay_fit(
        f, tg, r_lor, title="lorentzian (use this on 53672)", save_path=out / "delay_lorentzian_fit.png"
    )
    return path, r_lor


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    plot_geometry_atlas(out)
    _, r_lor = plot_why_sparam_fails_on_vna_peak(out)
    (out / "geometry_and_delay_notes.txt").write_text(
        "LORENTZIAN (delay-only; Qi not identified)\n" + r_lor.summary() + "\n"
    )
    print(r_lor.summary())
    print("wrote", out)


if __name__ == "__main__":
    main(Path("examples/artifacts"))
