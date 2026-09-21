from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparam_fit.group_delay import delay_lorentzian, group_delay_from_s
from sparam_fit.models import ResonatorParams
from sparam_fit.synthetic import make_trace
from sparam_fit.fit import fit_group_delay, fit_group_delay_vs_power
from sparam_fit.plots import (
    plot_delay_before_after,
    plot_delay_power_map,
    plot_group_delay_fit,
)


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    # Case B: hanger S11 delay is a Lorentzian peak on a cable shelf.
    fr, Ql, tau = 6.858e9, 6500.0, 70e-9
    f = np.linspace(fr - 8 * fr / Ql, fr + 8 * fr / Ql, 801)
    tg = delay_lorentzian(f, fr, Ql, tau, amp=Ql / (np.pi * fr))
    rng = np.random.default_rng(7)
    tg = tg + 2e-9 * rng.normal(size=f.size)

    r = fit_group_delay(f, tg)
    plot_group_delay_fit(
        f, tg, r, title="group delay Lorentzian (case B)", save_path=out / "group_delay_fit.png"
    )

    # Electrical delay from complex S is a different object — show hanger S21.
    true_s21 = ResonatorParams(
        fr=fr,
        Ql=Ql,
        absQc=11000.0,
        phi=0.08,
        a=0.02,
        alpha=-0.4,
        tau=tau,
        layout="hanger",
        s_param="S21",
    )
    f_s, s = make_trace(true_s21, n=801, span_bw=12.0, snr=120, seed=7)
    plot_delay_before_after(f_s, s, tau, save_path=out / "electrical_vs_group_delay.png")
    np.savez(out / "synthetic_hanger_s21.npz", f=f_s, s=s, tau_g=group_delay_from_s(f_s, s))

    powers = np.linspace(-60, -10, 26)
    rows = []
    for pwr in powers:
        Ql_p = 5e3 * (1.5 + np.tanh((pwr + 40) / 8.0))
        fr_p = fr - 80e3 * (10 ** (pwr / 10.0) / 10 ** (-20 / 10.0))
        rows.append(delay_lorentzian(f, fr_p, Ql_p, tau, amp=Ql_p / (np.pi * fr_p)))
    delay_2d = np.vstack(rows)
    plot_delay_power_map(
        f, powers, delay_2d, title="synthetic delay map (like 53672)", save_path=out / "delay_power_map.png"
    )
    _, arr = fit_group_delay_vs_power(f, powers, delay_2d)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(arr["power"], arr["fr"] * 1e-9, ".-")
    ax[0].set_xlabel("power (dBm)")
    ax[0].set_ylabel("f_r (GHz)")
    ax[1].plot(arr["power"], arr["Ql"], ".-", label="Q_l")
    ax[1].set_xlabel("power (dBm)")
    ax[1].legend()
    ax[1].set_title("Qi is not identified from delay")
    fig.tight_layout()
    fig.savefig(out / "delay_vs_power_params.png", dpi=140, bbox_inches="tight")

    (out / "delay_fit_summary.txt").write_text(r.summary() + "\n")
    print(r.summary())
    print("wrote", out)


if __name__ == "__main__":
    main(Path("examples/artifacts"))
