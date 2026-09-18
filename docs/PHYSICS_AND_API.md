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

Diameter correction (Khalil), when the circle model applies:

    1/Qi = 1/Ql − cos(φ)/|Qc|

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

## 5. What the library has today (honest map)

Complex S (your original tools), default `geometry="notch"` = hanger S21:

- `fit_amp_phase`, `fit_circle`, `fit_hybrid`, `fit_magnitude_only`

Electrical delay from complex S:

- `guess_delay_linear`, `refine_delay_circle`, `remove_delay`

Group delay:

- `group_delay_from_s` — S → τ_g
- `group_delay_model` — τ_g from a `ResonatorParams` (derivative of the S model)
- `delay_lorentzian` — the A, Ql, τ formula above
- `fit_group_delay(f, τ_g, geometry=..., model=...)` — **this is the confusing API**

Current flags (to be replaced later, do not add more):

- `geometry`: `"notch"` | `"transmission"` | `"reflection"` — really
  “which S formula”, mixing layout and S11/S21.
- `model`: `"sparam"` | `"lorentzian"` | `"auto"` — really “fit the S
  derivative vs fit the phenomenological delay peak”.

`auto` currently: delay **peak** → Lorentzian; **dip** → sparam notch.
That is a heuristic, not physics.

`s11_reflection` is the one-port formula. It is a **near-resonance
approximation** for hanger + short/open S11, not a second hanger layout.

---

## 6. What we will change (later, on purpose)

No new fit algorithms until this mapping is in the function signatures.

1. Replace the pair `(geometry, model)` by:
   - `layout`: `"hanger"` | `"through"` | `"direct"`
   - `s_param`: `"S21"` | `"S11"`
   - `observable`: `"complex"` | `"magnitude"` | `"group_delay"`
2. Formula table in §3 is the only switch.
3. Delay-only default: Lorentzian → fr, Ql, τ. Qi only if complex S is present.
4. No silent `auto` that swaps notch vs reflection.
5. Keep Probst circle / amp-phase / hybrid unchanged for hanger S21.
6. Tests: one synthetic trace per row of the table in §1.

Until that refactor, for your data:

- hanger S21 complex → `fit_hybrid(..., geometry="notch")`
- hanger S11 delay peak → `fit_group_delay(..., model="lorentzian")`
- do not use `geometry="transmission"` for hangers
