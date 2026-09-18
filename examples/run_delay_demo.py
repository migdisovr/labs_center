from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparam_fit.models import ResonatorParams
from sparam_fit.group_delay import group_delay_from_s, group_delay_model
from sparam_fit.synthetic import make_trace
from sparam_fit.fit import fit_group_delay, fit_group_delay_vs_power
from sparam_fit.plots import (
    plot_delay_before_after,
    plot_delay_power_map,
    plot_group_delay_fit,
)


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    true = ResonatorParams(
        fr=6.858e9,
        Ql=6500.0,
        absQc=11000.0,
        phi=0.08,
        a=0.02,
        alpha=-0.4,
        tau=70e-9,
        geometry="notch",
    )
    f, s = make_trace(true, n=801, span_bw=12.0, snr=120, seed=7)
    tg_ieee = group_delay_from_s(f, s)
    # **** / VNA delay map in the user's screenshot is a *peak* (~50–400 ns)
    tg_display = -tg_ieee

    r = fit_group_delay(f, tg_display, geometry="notch")
    plot_group_delay_fit(
        f, tg_display, r, title="group delay (display sign auto)", save_path=out / "group_delay_fit.png"
    )
    plot_delay_before_after(f, s, r.params.tau, save_path=out / "electrical_vs_group_delay.png")

    powers = np.linspace(-60, -10, 26)
    rows = []
    for i, pwr in enumerate(powers):
        # TLS-like: Qi (hence Ql) grows with power; slight Kerr shift
        Qi = 8e3 * (1.5 + np.tanh((pwr + 40) / 8.0))
        Qc = 1.1e4
        Ql = 1.0 / (1.0 / Qi + 1.0 / Qc)
        fr = true.fr - 80e3 * (10 ** (pwr / 10.0) / 10 ** (-20 / 10.0))
        p = ResonatorParams(
            fr=fr, Ql=Ql, absQc=Qc, phi=0.08, a=1.0, tau=70e-9, geometry="notch"
        )
        rows.append(-group_delay_model(f, p))
    delay_2d = np.vstack(rows)
    plot_delay_power_map(
        f, powers, delay_2d, title="synthetic delay map (like 53672)", save_path=out / "delay_power_map.png"
    )
    _, arr = fit_group_delay_vs_power(f, powers, delay_2d, geometry="notch")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(arr["power"], arr["fr"] * 1e-9, ".-")
    ax[0].set_xlabel("power (dBm)")
    ax[0].set_ylabel("f_r (GHz)")
    ax[1].plot(arr["power"], arr["Ql"], ".-", label="Q_l")
    ax[1].plot(arr["power"], arr["Qi"], ".-", label="Q_i (DCM)")
    ax[1].set_xlabel("power (dBm)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "delay_vs_power_params.png", dpi=140, bbox_inches="tight")

    (out / "delay_fit_summary.txt").write_text(r.summary() + "\n")
    print(r.summary())
    print("wrote", out)


if __name__ == "__main__":
    main(Path("examples/artifacts"))
