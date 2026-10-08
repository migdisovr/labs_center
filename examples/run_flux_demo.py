"""Synthetic hanger S21 versus coil current, with sweet-spot fits.

The qubit sits above the resonator, so the sharp downward pull is the lower
sweet spot and the flatter extremum is the upper one.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparam_fit import analyze_flux_map, make_dispersive_flux_map, plot_flux_analysis


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    bias = np.linspace(-2.0e-3, 2.0e-3, 121)
    freq, bias, s, fr, fq = make_dispersive_flux_map(
        bias,
        fr_bare_hz=6.64e9,
        g_hz=90e6,
        fq_max_hz=8.0e9,
        fq_min_hz=7.15e9,
        period=3.0e-3,
        bias_at_upper=0.0,
        Ql=2000.0,
        absQc=3200.0,
        phi=0.08,
        a=0.03,
        alpha=0.35,
        tau=35e-9,
        n_freq=301,
        span_bw=8.0,
        snr=200,
        seed=1,
    )
    result = analyze_flux_map(freq, bias, s, layout="hanger", s_param="S21")
    text = result.summary()
    (out / "flux_sweet_spots.txt").write_text(text + "\n")
    print(text)
    plot_flux_analysis(
        result,
        title="synthetic hanger, qubit above resonator",
        save_path=out / "flux_sweet_spots.png",
        bias_scale=1e3,
        bias_unit="mA",
    )
    plt.close("all")
    # Sanity: the fit at the upper sweet spot lands on the model curve.
    primary = result.primary
    true = float(np.interp(primary.feature.bias, bias, fr))
    print(f"primary bias {primary.feature.bias:.6e} A, fit fr {primary.fit.params.fr:.6e}, model {true:.6e}")
    print(f"qubit at that bias {float(np.interp(primary.feature.bias, bias, fq)):.6e} Hz")


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "artifacts")
