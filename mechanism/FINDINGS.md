# What the mechanism says, and where it goes quiet

## Engineering checks (16/16)

`test_mechanism.py`. Every case here has an answer we put in ourselves, so passing shows the code
does what it claims — not that the criterion tracks real theory choice.

- finds the unobservable direction in a description whose concepts are never separable
- quotients it away with predictions **unchanged to 1e-6** — a reinterpretation, not a better fit
- **declines to act** on an honest theory with nothing hidden
- **rejects** a patch whose new concept moves no prediction (the property that stops "add a
  parameter" from being free)
- accepts a new concept only when the residual really has structure the old language lacked

## Live controversy: NGC 3198 (SPARC)

No ground truth. That is what makes it free of the leakage that makes historical cases worthless
for evaluating a model that has read the histories. No accuracy figure is attached.

| description | concepts | chi2/N | coverage | unobservable at the error bars |
|---|---:|---:|---:|---:|
| LCDM: baryons + NFW halo | 3 | 1.22 | 93% | **0** |
| MOND: baryons, a0 universal | 1 | 9.21 | 60% | **0** |
| MOND: baryons, a0 free | 2 | 1.78 | 86% | **0** |

**The criterion is silent on this controversy, and that is the result.**

Neither description carries structure the measurements cannot see. LCDM's softest direction — the
disk–halo trade-off, `-0.72*M/L -0.11*V200 +0.68*c` — still moves the curve **6.4 sigma** under a
100% excursion. It is a real degeneracy and it is *measurable*, which is unsurprising: SPARC's
Spitzer 3.6 um photometry was assembled precisely to break it.

The deeper point is a **scope limit we did not anticipate**. The Lorentz/Einstein signature needs
two descriptions that *agree on the data* and differ in what they carry. These do not agree —
chi2/N of 1.22 against 9.21. When descriptions disagree, ordinary model comparison settles it and
no reinterpretation machinery is needed. The criterion applies to a narrower class of disputes
than we assumed.

Consistency check against the literature: our free-a0 fit gives a0 = 2053 (km/s)^2/kpc
= 0.67e-8 cm/s^2, against the canonical 1.22e-8. NGC 3198 is documented as the most relevant
borderline case for MOND, preferring a lower a0 and a shorter distance than Cepheids give. We
reproduce that tension rather than contradicting it.

Data: Lelli, McGaugh & Schombert (2016), SPARC, `MassModels_Lelli2016c.mrt`.

---

## Discovering an unknown symmetry (`discover_symmetry.py`)

`operational/gauge.py` classifies transformations it is *handed*. Here the mechanism is given
only a basis of eight candidate generators — most of which are not invariances — and has to find
for itself which directions the probes cannot see.

**Step 1–2, find them.** The null space of `d(probes)/dc` under uniform application:
rank 3 of 8, so **5 invariant directions found**. Rejected on their own: `link phase`,
`rescale`, `shift flavour 0`.

**Step 3, sort them.** A second null-space computation *inside* the first, now with the
coefficient allowed to vary from site to site:

| | | |
|---|---:|---|
| **1 redundancy** | local move `2.6e-05` | `+1.00 [gauge (compensated)]` → quotient **[L1]** |
| **4 symmetries** | local move `5.1e+01`–`6.6e+01` | keep, expect a charge **[L4]** |

The gauge direction is isolated cleanly, and it had to be: at *uniform* order the gauge
transformation and the global phase act identically on the state, so the first null space mixes
them. Only the local test breaks that degeneracy.

**Step 4, do the symmetry branches pay?** Every one yields a conserved charge:

| symmetry direction | charge | relative drift |
|---|---:|---:|
| `+0.34I -0.34sx +0.84sy -0.25sz` | 51.702175 | 1.8e-11 |
| `-0.63sx +0.77sz` | −25.036839 | 5.5e-11 |
| `+0.90I +0.23sx -0.18sy +0.32sz` | 120.197699 | 5.8e-11 |
| `+0.24I -0.66sx -0.52sy -0.49sz` | 16.365246 | 7.6e-10 |
| **redundancy**: would-be local charge | 6.09 → 11.34 | **8.6e-01** |

Energy drift on the same trajectory is 1.3e-10, so the symmetry charges are conserved at the
trajectory's own precision, and the redundancy's would-be charge is **nine orders of magnitude**
worse. That asymmetry is the evidence the sort is real rather than a relabelling.

**What it cost to get wrong.** Without step 3, all five directions are unobservable and all five
get quotiented — **destroying four conservation laws** to remove one redundancy. That is
`STANDARD_MODEL.md` D1, and `concept_space.quotient()` now refuses to run without a locality
test rather than guess (`21/21` checks, up from 16).

---

## The LLM proposes; the measurement disposes (`llm_propose.py`)

A null-space search finds only what its basis contains, so something must widen the basis. Our
own experiment says what a language model is for here: 87% correct when it could recognise the
episode, 48% — chance — when the names were stripped. It recalls; it does not judge. So it is
used **only to generate**, never to decide.

| world | residual | proposed | accepted |
|---|---|---:|---:|
| **A** | a real missing concept, `0.4*sin(3x)` | 8 | 2–5 |
| **B** | **control** — pure noise, nothing to find | 8 | **0** |

**Replicated across three independent runs** (`llm_propose_out.txt`, `llm_propose_run1.txt`,
`llm_propose_run2.txt`). World B accepted **0 of 8 in every run**. World A accepted 2–3, then 5,
then 4 — the variation is the temperature, and the higher figures sharpen the limitation below:
the acceptance rule lets over half the proposals through. The oracle check is identical in all
three: 97% for the truth against 68% for the near miss and 20% for the wrong ones.

The model proposes with equal fluency in both. Only the measurement tells them apart, and that
is the whole arrangement: a proposal that buys fit with structure no measurement can see is
rejected however confident the model sounds.

**Where the bottleneck actually is.** Offered the true concept directly, the filter is decisive:

| candidate | coverage |
|---|---:|
| `sin(3*x)` — **the truth** | **97%** |
| `sin(2.9*x)` — near miss | 68% |
| `sin(2*x)` — wrong | 20% |
| `x**3` — wrong | 20% |
| base model | 18% |

So the test separates the truth from near-misses by a wide margin, and **the proposer is the
limiting component, not the criterion** — across runs gpt-oss:20b offered `sin(x)`, `sin(2x)`,
`x**3` and `sin(pi*x)`, never `sin(3x)`, though the residual is a clean sine.

**A weakness in our own rule, stated plainly.** "Coverage strictly improves" is too permissive:
it admitted `sin(x)` and `x**3` at 20–22% against a base of 18%, and across three runs it let
through 2, 4 and 5 of 8 proposals. The rule *ranks* correctly but does not *discriminate*; it
needs a margin, not a strict inequality. Proposal counts vary with temperature 0.8 — the control
accepting zero did not, in any run.
