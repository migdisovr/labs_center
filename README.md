# labs_center — S-parameter resonator fitting

Extract $f_r$, $Q_l$, $Q_c$, $Q_i$ from microwave resonator data
(experiment or simulation). Circle fit: Probst / Ustinov (RSI 2015,
arXiv:1410.3365) with Khalil diameter correction.

**Read `docs/PHYSICS_AND_API.md` before adding flags.** Chip layout
(hanger vs through) and measured quantity (S21 vs S11 vs group delay)
are two different axes; the old `geometry` / `model` pair mixed them.

## Install

```bash
pip install -r requirements.txt
```

Add the repo root to `PYTHONPATH`.

## Everyday calls (this lab)

Hanger on a feedline, **S21**, both ports matched:

```python
from sparam_fit import fit_hybrid
best, r_ap, r_c = fit_hybrid(freq_hz, s21, geometry="notch")
print(best.summary())
```

Same hangers, **S11** (feedline shorted or open), VNA Delay trace:

```python
from sparam_fit import fit_group_delay
r = fit_group_delay(freq_hz, tau_g_s, model="lorentzian")
```

Frequencies in **Hz**. Loaded linewidth is FWHM: `kappa_Hz = fr / Ql`.

Do **not** use `geometry="transmission"` for hangers (that is an in-line
bandpass, not a side-coupled resonator).

## Tests and demos

```bash
python3 -m pytest tests/test_sparam_fit.py -q
python3 examples/run_synthetic_demo.py
python3 examples/plot_geometries.py
```
