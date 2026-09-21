# Physics and API contract for `sparam_fit`

This file is the source of truth. Code may lag; if code and this file disagree,
this file wins until we change it on purpose.

There are **two independent choices**. Mixing them into one flag
(`geometry` + `model`) is what made the library feel like a black box.

1. **Chip layout** — how the resonator is wired.
2. **Measured observable** — S21, S11, |S| only, or group delay.

---

## 1. What we want

From experiment or simulation, extract **resonance frequency** and
**quality factors** with a stated model, without silent auto-switching.

Typical lab cases in this group:

| Case | Layout | Far end of feedline | What you record | Model to use |
|------|--------|---------------------|-----------------|--------------|
| A (most common) | hanger (“вешалка”) on a feedline | matched 50 Ω (second port) | complex **S21** | Probst **notch** |
| B (your delay maps 53672) | same hangers on a feedline | **short or open** | **S11** (or VNA Delay of S11) | near resonance ≈ **one-port reflection**; delay ≈ Lorentzian |
| C (rare here) | resonator **in series** in the line | both ports | S21 peak | Probst **transmission** (through). Do not use for hangers. |

Over/under/critical coupling is **not** a fourth model. It is a regime of
the same formula (κ_c ≷ κ_i).

---

## 2. Chip layouts (physics)

### Hanger / notch (your default)

Feedline goes from one end to the other. Resonators couple **from the side**.
Most of the microwave power continues down the line; a fraction is tapped
into the resonator.

```
port 1 ----feedline---- port 2
                 |
              resonator
```

- Measure **S21** with both ports matched → dip in |S21|. This **is**
  `s21_notch` (Probst eq. 1).
- Measure **S11** with port 2 matched 50 Ω → off-resonance S11 ≈ 0
  (matched line). Different and rarely what you want.
- Measure **S11** with the far end **shorted or open** (one-port fridge
  wiring) → standing wave on the feedline + hanger. Near one resonance,
  after cable delay, S11 traces a circle like a **one-port resonator**.
  Group delay is a **positive peak**. This is your 53672 data.

### Through / in-line (not your hangers)

The resonator **is** the path. Off resonance S21 → 0, on resonance a peak.
Do not describe hangers with this formula.

### Direct one-port

The resonator is the load on the connector (no long feedline of hangers).
S11 = 1 − 2 (Ql/|Qc|) e^{iφ} / (1 + 2i Ql (f/fr−1)). Same functional
form we use as an **approximation** for hanger + short/open near resonance.

---

## 3. Formulas (environment always the same)

Cable / amps:

    env(f) = a · exp(i α) · exp(−2 π i f τ)
    τ  = electrical delay (constant, seconds)
    a, α = gain and constant phase

Resonator pole:

    D = 1 + 2 i Ql (f/fr − 1)
    β = (Ql / |Qc|) exp(i φ)

| Layout × observable | S(f) |
|---------------------|------|
| hanger × S21 (matched) | env · (1 − β / D) |
| through × S21 | env · (β / D) |
| one-port × S11, or hanger × S11 (short/open), near one mode | env · (1 − 2 β / D) |

Invalid combinations (raise `ValueError`): `through×S11`, `direct×S21`.

Diameter correction (Khalil), when the circle model applies:

    1/Qi = 1/Ql − cos(φ)/|Qc|

Canonical-circle radius after environment stripping:

    hanger/through S21:  r0 = Ql / (2 |Qc|)   ⇒  |Qc| = Ql / (2 r0)
    S11:                 r0 = Ql / |Qc|       ⇒  |Qc| = Ql / r0

Q = f / FWHM. No extra factor of 2. Linewidth in Hz: κ/2π = f/Q.

### Over / under / critical (same formulas)

κ_c = 2π f / Qc, κ_i = 2π f / Qi (if Q are defined that way; equivalently
rates in Hz: f/Qc vs f/Qi).

- **Undercoupled** κ_c < κ_i: weak port, shallow feature. S11 circle does
  **not** enclose 0.
- **Critical** κ_c = κ_i: S11(fr) = 0, circle through 0. **Group delay of
  S11 diverges.** Do not fit S11-delay with the full S11 model there.
- **Overcoupled** κ_c > κ_i: deep hanger dip; S11 circle encloses 0
  (phase winds ~2π).

You cannot read over/under from delay **height alone** without |S|.

---

## 4. Two different “delays”

| Name | Symbol | What it is |
|------|--------|------------|
| Electrical / cable | τ | Constant. Factor exp(−2π i f τ) in S. Baseline of a delay trace. Circle-fit removes it: S ← S exp(+2π i f τ). |
| Group delay | τ_g(f) | τ_g = − d arg(S) / dω = −(1/2π) dφ/df. VNA format Delay, qsweepy `delay`. Function of frequency. |

S → τ_g is unique. τ_g → S recovers **phase only**, not |S|.

### Lorentzian (not a layout)

For **one isolated high-Q pole**, arg(S) ~ arctan(2 Ql (f/fr − 1)), so

    τ_g(f) = τ + s (f − fr) + A / (1 + 4 Ql² (f/fr − 1)²)

- `s` = residual slope of the cable (optional).
- `A` = extra delay at fr. For through S21, A = Ql /(π fr) exactly.
  For S11, A depends on coupling (blows up at critical).
  For hanger S21, A is **negative** (a dip).

Use this when you **only have a delay trace** and want fr, Ql, τ.
Do **not** report Qi from it unless you also have |S| or a full circle.

---

## 5. Implemented API

Two axes, plus the observable implicit in which function you call:

- `layout`: `"hanger"` | `"through"` | `"direct"`
- `s_param`: `"S21"` | `"S11"`
- observable: `fit_circle` / `fit_amp_phase` / `fit_hybrid` (complex S),
  `fit_magnitude_only` (|S|), `fit_group_delay` (τ_g)

The formula table in §3 is the only switch.  There is **no** silent `auto`
that swaps notch vs reflection, or peak vs dip vs through.

```python
from sparam_fit import (
    fit_amp_phase, fit_circle, fit_hybrid, fit_magnitude_only,
    fit_group_delay, fit_group_delay_vs_power,
)

# Case A — hanger, matched S21 (Probst circle / amp-phase / hybrid)
best, r_ap, r_c = fit_hybrid(freq_hz, s21, layout="hanger", s_param="S21")

# Case B — hanger S11 (short/open), or VNA Delay of that S11
r_s11 = fit_circle(freq_hz, s11, layout="hanger", s_param="S11")
r_d = fit_group_delay(freq_hz, tau_g_s)          # always Lorentzian → fr, Ql, τ
results, vs_p = fit_group_delay_vs_power(freq_hz, power_dBm, delay_2d)

# Case C — through S21 (not hangers)
r_m = fit_magnitude_only(freq_hz, mag, layout="through", s_param="S21")
```

`ResonatorParams.layout` and `.s_param` store the two axes.
`.formula` is the table-row name (`notch` / `through` / `reflection`) used
internally to pick S(f).

Deprecated: `geometry="notch"|"transmission"|"reflection"` still maps onto
`(layout, s_param)` with a `DeprecationWarning`
(`notch` → hanger×S21, `transmission` → through×S21, `reflection` → hanger×S11).

Delay-only:

- `fit_group_delay` **always** uses the Lorentzian of §4.
- Qi / |Qc| / a / α are listed as not identifiable.
- `group_delay_model(f, params)` is the derivative of the S formula, for
  overlays when you already have a complex-S fit.  It is not a delay fitter.

Keep Probst circle / amp-phase / hybrid unchanged in spirit for hanger S21;
the S11 circle uses the S11 radius law |Qc| = Ql / r0.

---

## 6. How to call it on your chips

- hanger S21 complex → `fit_hybrid(..., layout="hanger", s_param="S21")`
- hanger S11 delay peak → `fit_group_delay(...)`  (Lorentzian; no Qi)
- hanger S11 complex → `fit_circle(..., layout="hanger", s_param="S11")`
- do not use `layout="through"` for hangers
