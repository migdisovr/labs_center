# labs_center — S-parameter resonator fitting

Python toolkit to extract $f_r$, $Q_l$, $Q_c$, $Q_i$ from microwave
S-parameters (experiment or simulation). The reference algorithm is the
Probst / Ustinov circle fit (RSI 2015, arXiv:1410.3365) with Khalil
diameter correction.

The S-formula is selected by **chip layout** × **measured S-parameter**,
not by a single `geometry` flag.  See [`docs/PHYSICS_AND_API.md`](docs/PHYSICS_AND_API.md).

## Install

```bash
pip install -r requirements.txt
```

The package is the `sparam_fit` folder; add the repo root to `PYTHONPATH`
or run notebooks from that root.

## Two axes, three lab cases

| Case | Call | Formula |
|------|------|---------|
| A. hanger, matched S21 | `fit_hybrid(f, s21, layout="hanger", s_param="S21")` | $S={\rm env}\,(1-\beta/D)$ |
| B. hanger, S11 short/open (or VNA Delay) | `fit_circle(f, s11, layout="hanger", s_param="S11")` or `fit_group_delay(f, tau_g)` | $S={\rm env}\,(1-2\beta/D)$; delay = Lorentzian |
| C. through S21 (not hangers) | `fit_magnitude_only(f, mag, layout="through", s_param="S21")` | $S={\rm env}\,(\beta/D)$ |

```python
from sparam_fit import (
    fit_amp_phase, fit_circle, fit_hybrid, fit_magnitude_only,
    fit_group_delay, fit_group_delay_vs_power, group_delay_from_s,
)

# Case A — hanger S21: amplitude then phase then complex polish,
# and an independent algebraic circle fit
r_ap = fit_amp_phase(freq_hz, s21, layout="hanger", s_param="S21")
r_c = fit_circle(freq_hz, s21, layout="hanger", s_param="S21")
best, r_ap, r_c = fit_hybrid(freq_hz, s21, layout="hanger", s_param="S21")
r_m = fit_magnitude_only(freq_hz, mag, layout="hanger", s_param="S21", mag_is_db=False)

# Case B — delay-only (VNA Delay / qsweepy).  Always Lorentzian; no Qi.
r_d = fit_group_delay(freq_hz, tau_g_s)
results, vs_p = fit_group_delay_vs_power(freq_hz, power_dBm, delay_2d)
print(best.summary())
```

Frequencies are in **Hz**. Loaded linewidth is FWHM: `kappa_Hz = fr / Ql`
(no extra factor of 2).

Delay removal is $S \leftarrow S\exp(+2\pi i f\tau)$ when the model contains
$\exp(-2\pi i f\tau)$.

**Group delay** (what a VNA stores as format Delay) is a different object:
$\tau_g(f)=-\mathrm{d}\arg S/\mathrm{d}\omega$.  The cable time is the
*baseline* of $\tau_g(f)$; the resonator is the peak/dip on top.  You can
go $S\to\tau_g$ uniquely; $\tau_g\to S$ only recovers phase, not $|S|$.
Delay-only fits therefore report $f_r$, $Q_l$, $\tau$ and **not** $Q_i$.

`geometry="notch"|"transmission"|"reflection"` still works as a deprecated
alias (`notch` → hanger×S21, `transmission` → through×S21,
`reflection` → hanger×S11).

## Notebook

`notebooks/Fit_FlipChip_Sparametr.ipynb` — original scratchpad plus Probst
pipeline.  `notebooks/Fit_FlipChip_Sparametr_Actual.ipynb` — the working
lab notebook, with a group-delay section at the bottom.

Synthetic demos: `python3 examples/run_synthetic_demo.py`,
`python3 examples/run_delay_demo.py`, `python3 examples/plot_geometries.py`

Tests: `python3 -m pytest tests/test_sparam_fit.py -q`
