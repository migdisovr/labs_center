# labs_center — S-parameter resonator fitting

Python toolkit to extract $f_r$, $Q_l$, $Q_c$, $Q_i$ from microwave $S_{21}$
(experiment or simulation). The reference algorithm is the Probst / Ustinov
circle fit (RSI 2015, arXiv:1410.3365) with Khalil diameter correction.

## Install

```bash
pip install -r requirements.txt
```

The package is the `sparam_fit` folder; add the repo root to `PYTHONPATH`
or run notebooks from that root.

## Two primary tools

```python
from sparam_fit import (
    fit_amp_phase, fit_circle, fit_hybrid, fit_magnitude_only,
    fit_group_delay, fit_group_delay_vs_power, group_delay_from_s,
)

# 1) amplitude then phase then complex polish
r_ap = fit_amp_phase(freq_hz, s21, geometry="notch")

# 2) algebraic circle fit (independent cross-check)
r_c = fit_circle(freq_hz, s21, geometry="notch")

# both, pick lower complex RMS
best, r_ap, r_c = fit_hybrid(freq_hz, s21)

# magnitude-only (Ansys dB, or VNA without phase)
r_m = fit_magnitude_only(freq_hz, mag, geometry="notch", mag_is_db=False)

# 3) group delay only (VNA "Delay" / delay map)
#    tau_g is in seconds; electrical delay is the baseline of that trace
r_d = fit_group_delay(freq_hz, tau_g_s, geometry="notch")
results, vs_p = fit_group_delay_vs_power(freq_hz, power_dBm, delay_2d, geometry="notch")
print(best.summary())
```

# 1) amplitude then phase then complex polish
r_ap = fit_amp_phase(freq_hz, s21, geometry="notch")

# 2) algebraic circle fit (independent cross-check)
r_c = fit_circle(freq_hz, s21, geometry="notch")

# both, pick lower complex RMS
best, r_ap, r_c = fit_hybrid(freq_hz, s21)

# magnitude-only (Ansys dB, or VNA without phase)
r_m = fit_magnitude_only(freq_hz, mag, geometry="notch", mag_is_db=False)
print(best.summary())
```

Frequencies are in **Hz**. Loaded linewidth is FWHM: `kappa_Hz = fr / Ql`
(no extra factor of 2). Notch/hanger traces use `geometry="notch"`;
a through-peak uses `"transmission"` (then $Q_i$ is not identifiable
from uncalibrated $S_{21}$).

Delay removal is $S \leftarrow S\exp(+2\pi i f\tau)$ when the model contains
$\exp(-2\pi i f\tau)$.

**Group delay** (what a VNA stores as format Delay) is a different object:
$\tau_g(f)=-\mathrm{d}\arg S/\mathrm{d}\omega$.  The cable time is the
*baseline* of $\tau_g(f)$; the resonator is the peak/dip on top.  You can
go $S\to\tau_g$ uniquely; $\tau_g\to S$ only recovers phase, not $|S|$.

## Notebook

`notebooks/Fit_FlipChip_Sparametr.ipynb` — original scratchpad plus Probst
pipeline.  `notebooks/Fit_FlipChip_Sparametr_Actual.ipynb` — the working
lab notebook, with a group-delay section at the bottom.

Synthetic demos: `python3 examples/run_synthetic_demo.py` and
`python3 examples/run_delay_demo.py`

Tests: `python3 -m pytest tests/test_sparam_fit.py -q`

## Notebook

`notebooks/Fit_FlipChip_Sparametr.ipynb` is the original fitting scratchpad
with a new section appended at the bottom (theory, failure analysis of the
old residuals, and calls into `sparam_fit`).

Synthetic demo: `python3 examples/run_synthetic_demo.py`

Tests: `python3 -m pytest tests/test_sparam_fit.py -q`
