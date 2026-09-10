from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparam_fit.models import ResonatorParams, model_s
from sparam_fit.synthetic import make_trace
from sparam_fit.fit import fit_amp_phase, fit_circle, fit_hybrid, fit_magnitude_only
from sparam_fit.plots import plot_delay_before_after, plot_fit_report
from sparam_fit.guess import guess_delay_linear, refine_delay_circle, remove_delay


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    true = ResonatorParams(
        fr=6.642e9,
        Ql=1160.0,
        absQc=1550.0,
        phi=0.18,
        a=0.0032,
        alpha=-0.9,
        tau=38e-9,
        geometry="notch",
    )
    f, s = make_trace(true, n=601, span_bw=10.0, snr=70, seed=11)
    np.savez(out / "synthetic_notch.npz", f=f, s=s)

    tau0, _, _ = guess_delay_linear(f, s)
    tau = refine_delay_circle(f, s, tau0)
    plot_delay_before_after(f, s, tau, save_path=out / "delay_unwrap_circle.png")

    r_c = fit_circle(f, s, geometry="notch")
    plot_fit_report(f, s, r_c, title="circle fit", save_path=out / "circle_fit_report.png")

    r_ap = fit_amp_phase(f, s, geometry="notch")
    plot_fit_report(f, s, r_ap, title="amp + phase", save_path=out / "amp_phase_report.png")

    r_m = fit_magnitude_only(f, np.abs(s), geometry="notch")
    # magnitude-only has no complex overlay; still plot |S|
    plot_fit_report(f, s, r_m, title="magnitude only vs complex data", save_path=out / "magnitude_only_report.png")

    best, _, _ = fit_hybrid(f, s)
    plot_fit_report(f, s, best, title="hybrid", save_path=out / "hybrid_report.png")

    report = out / "fit_summary.txt"
    report.write_text(
        "TRUE\n"
        + str(true.as_dict())
        + "\n\nCIRCLE\n"
        + r_c.summary()
        + "\n\nAMP+PHASE\n"
        + r_ap.summary()
        + "\n\nMAGNITUDE\n"
        + r_m.summary()
        + "\n\nHYBRID\n"
        + best.summary()
        + "\n"
    )
    print(report.read_text())
    print("wrote", out)


if __name__ == "__main__":
    main(Path("examples/artifacts"))
