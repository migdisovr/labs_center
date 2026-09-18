from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .models import model_s


def plot_fit_report(f, s, result, title=None, save_path=None):
    """Amplitude, phase, and complex-plane overlay for a FitResult."""
    p = result.params
    m = model_s(f, p)
    s = np.asarray(s, dtype=np.complex128)
    z = result.s_delay_removed
    if z is None:
        z = s * np.exp(2j * np.pi * f * p.tau)
    mz = m * np.exp(2j * np.pi * f * p.tau)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax = axes[0, 0]
    ax.plot(f * 1e-9, 20 * np.log10(np.abs(s) + 1e-30), ".", ms=3, label="data")
    ax.plot(f * 1e-9, 20 * np.log10(np.abs(m) + 1e-30), "-", lw=1.5, label="fit")
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel("|S| (dB)")
    ax.legend()
    ax.set_title("amplitude")

    ax = axes[0, 1]
    ax.plot(f * 1e-9, np.unwrap(np.angle(s)), ".", ms=3, label="data")
    ax.plot(f * 1e-9, np.unwrap(np.angle(m)), "-", lw=1.5, label="fit")
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel("arg S (rad, unwrapped)")
    ax.legend()
    ax.set_title("phase")

    ax = axes[1, 0]
    ax.plot(z.real, z.imag, ".", ms=3, label="delay-removed data")
    ax.plot(mz.real, mz.imag, "-", lw=1.5, label="fit")
    circ = result.circle
    if circ and "r0" in circ:
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(
            circ["xc"] + circ["r0"] * np.cos(th),
            circ["yc"] + circ["r0"] * np.sin(th),
            "--",
            lw=1,
            label="algebraic circle",
        )
        ax.plot(circ["xc"], circ["yc"], "x")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Re S")
    ax.set_ylabel("Im S")
    ax.legend()
    ax.set_title("complex plane (delay removed)")

    ax = axes[1, 1]
    ax.axis("off")
    txt = result.summary()
    if title:
        txt = title + "\n\n" + txt
    ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_delay_before_after(f, s, tau, save_path=None):
    z = np.asarray(s) * np.exp(2j * np.pi * np.asarray(f) * tau)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(np.real(s), np.imag(s), ".", ms=3)
    ax[0].set_title("raw complex S")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[1].plot(z.real, z.imag, ".", ms=3)
    ax[1].set_title(r"after $\times\exp(2\pi i f \tau)$")
    ax[1].set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_group_delay_fit(f, tau_g, result, title=None, save_path=None):
    """Overlay measured group delay and the fitted model."""
    from .group_delay import group_delay_model

    f = np.asarray(f, dtype=float)
    tau_g = np.asarray(tau_g, dtype=float)
    sgn = result.diagnostics.get("sign", 1.0)
    model = sgn * group_delay_model(f, result.params)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(f * 1e-9, tau_g * 1e9, ".", ms=4, label="data")
    ax.plot(f * 1e-9, model * 1e9, "-", lw=1.5, label="fit")
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel("group delay (ns)")
    ax.legend()
    ax.set_title("delay vs frequency")
    ax = axes[1]
    ax.axis("off")
    txt = result.summary()
    if title:
        txt = title + "\n\n" + txt
    ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_delay_power_map(f, power, delay_2d, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(
        np.asarray(f) * 1e-9,
        np.asarray(power),
        np.asarray(delay_2d) * 1e9,
        shading="auto",
        cmap="inferno",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("group delay (ns)")
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel("power (dBm)")
    ax.set_title(title or "delay vs frequency and power")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig
