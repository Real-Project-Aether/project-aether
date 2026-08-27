# The Standard Model as a test of the framework

The Standard Model is the most precisely tested description ever built, and its concepts —
the neutrino, colour, quarks, the Higgs — were formed under exactly the conditions
`OPERATIONAL.md` claims to address: a description repeatedly found to be too coarse, too
fine, or inconsistent. So it is the right test, and it is a hard one.

**Verdict first: the framework as written accounts for two of the eight events below, and
could not have built the Standard Model.** Three of the four defects are fixable and the
fixes are given. The fourth is not, and it bounds what the framework is a theory of.

---

## 1. The eight events

| # | event | what triggered it | framework's operation | fits? |
|---|---|---|---|---|
| 1 | **neutrino** (Pauli, 1930) | β decay gives a *continuous* electron spectrum where a two-body decay demands a line — same initial and final nuclear states, different outcomes | **degeneracy** (§3.1): variance inside a fibre above the replicate floor → open a channel | **yes** |
| 2 | **colour** (Greenberg 1964; Han–Nambu 1965) | Δ⁺⁺ = uuu at J=3/2, L=0 is symmetric in space, spin and flavour — Fermi statistics *forbids* it | none — the description is not too coarse in variance, it is **contradictory** | no |
| 3 | **gauge redundancy** | `A_μ → A_μ + ∂_μλ(x)` changes no measurement | **superfluity** (§3.2): quotient by the group all probes are blind to | **partly** |
| 4 | **charge conservation** (Noether, 1918) | a *global* phase rotation also changes no measurement | §3.2 would quotient it — **and thereby delete the conservation law** | **no, and it is a bug** |
| 5 | **charm / GIM** (1970) | flavour-changing neutral currents *far more suppressed* than the description predicts | none — the description over-predicts; nothing is degenerate or superfluous | no |
| 6 | **anomaly cancellation** (Bouchiat–Iliopoulos–Meyer, 1972) | gauge anomalies must cancel or the theory is inconsistent — this requires complete quark/lepton generations | none — **no probe is involved at all** | no |
| 7 | **Higgs from unitarity** (Lee–Quigg–Thacker, 1977) | WW scattering violates unitarity above ~1 TeV without it; this told experimenters *where to look* | none — again theory-internal | no |
| 8 | **running couplings** (RG) | a "constant" turns out to be a function of scale | `RETYPE`, from the earlier design — but the *fibre itself* becomes scale-dependent, which §3.1 has no room for | no |

Two of eight. That is the finding, and it is not a detail.

---

## 2. Four defects

### D1 — §3.2 as written deletes conservation laws  ·  **fixable**

Gauge redundancy (#3) and a global symmetry (#4) both satisfy §3.2's test exactly: *every
probe is invariant under `g`*. §3.2 quotients both. But quotienting a global U(1) does not
discover charge conservation — it **erases** the quantity that is conserved, by declaring the
distinction it grades unphysical.

The two cases are separated by one property the framework never asked about: **locality.**

```
∀p ∈ P : outcome(p, s) = outcome(p, g·s)

    and g may be applied INDEPENDENTLY AT EACH POINT   ->  redundancy   -> quotient
    and g must be applied UNIFORMLY                     ->  symmetry     -> KEEP,
                                                            and read off the conserved
                                                            current (Noether)
```

This turns §3.2 from one operation into two, and the second one — *a distinction that no probe
can see, but only when moved everywhere at once, is a conservation law* — is the more
productive of the pair in the history above.

A second correction rides along. Even for genuine redundancy, physics **keeps** the redundant
gauge description, because dropping it costs manifest locality and Lorentz invariance; the
quotient is taken at the level of physical states, not of the description. So §3.2's
prescription must be *quotient the state space, and keep the description if the quotient costs
more structure than it removes* — which is a description-length judgement after all, and the
one place where the discarded MDL machinery still earns its keep.

**D1 is now tested — `gauge.py`.** A lattice U(1) gauge system with two flavours, probed only
by gauge-invariant, flavour-blind quantities (site density, hopping, plaquette).

| transformation | probes move when uniform | when local | verdict |
|---|---|---|---|
| gauge U(1) | 3.6e-15 | 3.6e-15 | REDUNDANCY |
| flavour SU(2) | 7.1e-15 | **9.77** | SYMMETRY |
| flavour phase | 3.6e-15 | **5.47** | SYMMETRY |
| rescale (negative control) | **7.95** | 7.26 | NEITHER |

Four for four, with the control rejected before it reaches the local test at all.

The second half matters more, because sorting is only worth doing if the two branches buy
different things. Evolved under `i dφ/dt = ∂H/∂φ*` (energy drift 1.0e-09, so the trajectory
is trustworthy):

| branch | predicted quantity | relative drift |
|---|---|---|
| symmetry — flavour SU(2) | Noether charge | 4.0e-09 |
| symmetry — flavour phase | Noether charge | 5.9e-10 |
| symmetry — **global part** of gauge U(1) | Noether charge | 2.1e-10 |
| redundancy — **strictly local** gauge | — | **7.9e-03** |

Seven orders of magnitude. The symmetry branch predicts a quantity that the dynamics really
does conserve; the redundancy branch predicts nothing.

**And running it corrected the fix.** The third row was not in the plan. The gauge group
*contains* a global subgroup, and that subgroup is a genuine symmetry with a conserved charge —
electric charge, in the real case. So §3.2's repaired rule is not "quotient the group every
probe is blind to". It is:

> **quotient the strictly local part; keep the global part and read its conserved charge.**

Quotienting the whole gauge group would have thrown away charge conservation, which is the
very failure D1 was raised to prevent, one level down.

### D2 — the replicate must be a re-*preparation*  ·  **fixable, and the fix is forced**

This is the sharpest. §3.1 fires when

```
Var(y | fibre)  ≫  Var(y | the same situation, measured again)
```

Quantum mechanics makes the second term the whole question. Prepare an identical state,
measure, and outcomes still differ — irreducibly. If "the same situation measured again"
means *re-reading one instance*, the noise floor is near zero, the detector fires on every
quantum measurement ever made, and it demands a hidden channel each time. That is the local
hidden-variable programme, and Bell excludes it.

The fix is a definition, and the Standard Model forces it:

> **A replicate is a re-run of the same preparation procedure, not a re-reading of one
> outcome.**

With that, quantum indeterminacy sits *inside* the measured noise floor, and the detector
correctly stays silent on a system that is simply not deterministic. Note what this costs:
the framework can no longer distinguish "irreducibly random" from "we cannot prepare finely
enough". It has to treat the preparation procedure as the unit of identity — which is exactly
the operational move the framework was named after, arrived at here by force rather than by
choice.

**D2 is now built and tested — `d2.py`.** The definition turns out to say something sharper
than it first appears: **the noise floor is set by the finest preparation you can actually
perform.** Prepare more finely and more becomes visible; cannot prepare more finely, and what
is left is noise by definition, because no experiment could isolate it.

A channel exists; the fraction of it the preparation controls is swept:

| preparation resolution | 0.00 | 0.05 | 0.20 | 0.50 | 1.00 |
|---|---|---|---|---|---|
| **B** — replicate = re-preparation | **0.00** | 0.50 | 1.00 | 1.00 | 1.00 |
| **A** — replicate = re-reading one outcome | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

B tracks the preparation rather than the channel, and its silence at resolution 0 is correct
behaviour rather than insensitivity: there, nothing about the channel is isolable by any
available act.

Randomness no preparation can remove — the Bell case, with no channel at all:

| σ | 0.5 | 1.0 | 2.0 |
|---|---|---|---|
| **B** | **0.00** | **0.00** | **0.00** |
| **A** | 1.00 | 1.00 | 1.00 |

And the row that condemns A outright — **no channel and no indeterminacy whatsoever**:

| | B | A |
|---|---|---|
| null world | **0.04** (α = 0.05) | **1.00** |

A fires on a world with nothing in it. Re-reading a stored outcome cannot recover that
measurement's own noise, so the floor is short by exactly the measurement noise and *everything*
reads as reducible excess. The wrong definition does not merely over-fire in quantum
settings — it never stops firing anywhere.

### D3 — the 2×2 has no box for "no channel could do this"  ·  **fixable**

§3's diagram offers two failures: the description is too coarse, or too fine. Bell
correlations are neither. They are structure that **no local channel could produce**, in a
description that is not at fault.

Without a third outcome the framework searches forever, and `acquire.py`'s one positive
result — 10× when the answer was "go and look" — becomes a trap wherever the answer is "there
is nothing there to look at".

```
before proposing a channel, ask whether ANY channel could generate the observed pattern
    it could      -> propose, rank by footprint (§7), acquire
    it could not  -> the description is not the problem; record and stop
```

In physics the test is a Bell-type bound. In general it is whatever bound a common cause
obeys in that domain. This is a genuine addition and not a physics curiosity: it is the only
guard the framework has against proposing an unfindable variable indefinitely.

**D3 is now built and tested — `d3.py`.** The general form of "could any channel have done
this" is the **causal compatibility** problem, and it is an existing field: Bell inequalities,
and the inflation technique of Wolfe, Spekkens and Fritz for general latent structures. None of
that is invented here; what is tested is only whether the brake works when wired into this
framework.

For two probes with two settings and two outcomes, the distributions a shared channel can
produce form a polytope whose vertices are the deterministic strategies, so membership is exact
and is a linear program — stronger than checking one inequality. The test has a known right
answer to be graded against: quantum correlations at visibility `v` are channel-explicable
exactly when `v ≤ 1/√2 = 0.7071`.

Exact distances, no sampling:

| visibility | 0.50 | 0.70 | **0.7071** | 0.72 | 0.85 | 1.00 |
|---|---|---|---|---|---|---|
| distance to explicable | 0 | 0 | **0** | 0.036 | 0.404 | **0.828** |

The boundary lands exactly where it must, and 0.828 at `v=1` is `2√2 − 2`, the CHSH violation
itself. Worlds generated from a real hidden channel all give exactly 0.

**The first finite-sample version failed its own control, and that is worth recording.**
Calibrating the null by resampling from the plug-in projection of the estimate gave a false
positive rate of **0.125 and 0.100** at n=5000 and n=20000 against a nominal α=0.05 — the
projection lands on a low-dimensional face, where resampled distances are smaller than on a
facet, so the threshold came out too low. A brake that says *"no channel exists, stop looking"*
one time in ten when a channel does exist is worse than no brake.

The fix is to calibrate at the **least favourable** point of a composite null — on the
boundary, reached by mixing toward uniform along the ray the data sits on. After it:

| | n=1000 | n=5000 | n=20000 | n=80000 |
|---|---|---|---|---|
| v = 0.60 | 0.00 | 0.00 | 0.00 | 0.00 |
| v = 0.70 | 0.00 | 0.00 | 0.00 | 0.00 |
| **v = 0.7071 (boundary)** | **0.00** | **0.00** | **0.00** | **0.00** |
| v = 0.75 | 0.00 | 0.75 | 1.00 | 1.00 |
| v = 0.85 | 1.00 | 1.00 | 1.00 | 1.00 |
| v = 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **control: real hidden channel** | **0/40** | **0/40** | **0/40** | **0/40** |

All three predictions recorded before running now hold: the crossover sits at 1/√2 and nowhere
else; the detector is conservative approaching it, needing more data to resolve a small
violation rather than guessing; and the false positive rate is at or below α at every sample
size. The cost of the fix is power — `v=0.75` now needs n≥5000 where the miscalibrated version
"detected" it at n=1000 — and for a brake that is the right direction to err.

### D4 — the Standard Model was built by internal consistency  ·  **not fixable**

Events 2, 5, 6 and 7 have no probe in them.

- Δ⁺⁺ violated Fermi statistics. Colour was invented so the state would be *allowed*.
- FCNC were too suppressed. Charm was invented so the excess would *cancel*.
- Gauge anomalies had to sum to zero. That required generations to be *complete*.
- WW scattering had to stay unitary. That put the Higgs *below about a TeV* — and told
  experiments the energy to build for.

Every one predicted a real particle. Every one was paid for by a **theorem first**, with the
experiment arriving years or decades later. The framework's founding commitment — *a concept
is paid for by an experiment* — has these exactly backwards.

**And this reopens something already tested here.** `discovery/pressure.py` proposed concepts
from structural defects in the description and was **refuted on three substrates**. So is
D4 a loophole being smuggled back in?

The honest difference is grade, not kind:

| `pressure.py` | the Standard Model |
|---|---|
| dangling structure, role collision, signature strain | anomaly coefficients, Fermi statistics, unitarity |
| **heuristics** — a defect is a hint | **theorems** — a state is forbidden or it is not |

A framework may admit theorem-grade internal constraints without readmitting heuristic ones.
But this is not a free repair, because it names a prerequisite the framework does not have:
**you can only have theorems if there is a formal theory to have them about.** The Standard
Model has one. A concept bank over a dataset does not.

---

## 3. What survives, and it is not nothing

Event 1 is a clean fit, and it is the event that founded the Standard Model's particle
content. It is also a direct case of `RESULTS.md`'s finding.

In 1930 the candidate explanations were three: energy conservation fails in nuclear
processes; an unseen particle carries the balance; the measurements are wrong. What
discriminated them was **the shape of the β spectrum** — *where* the missing energy sat, and
how it was distributed to the endpoint — not *how much* was missing on average. A pooled
statistic ("energy is missing") is consistent with all three. The distribution over outcomes
picks out the third body and even constrains its mass and spin.

That is the footprint argument (§7) in the case that started the whole subject, arrived at
independently.

---

## 4. The revised scope

The framework should stop claiming to be a theory of scientific concept formation. Against
the best-tested theory in physics it explains two events in eight, and misses the mechanism
that produced most of the rest.

> **It is a theory of concept formation for descriptions with no formal theory attached** —
> where there are no theorems to violate, and the only thing a proposed concept can be
> answerable to is a measurement.

That is a real class and it covers most of machine learning. It is a much smaller claim than
`OPERATIONAL.md` §1 makes, and the Standard Model is why.

## 5. Changes owed to `OPERATIONAL.md`

| defect | change |
|---|---|
| D1 | **tested, `gauge.py`.** Split §3.2 into *redundancy* (strictly local `g` → quotient) and *symmetry* (global `g` → keep, extract the conserved charge). Quotient the **strictly local part only** — the group's global subgroup is a real symmetry. Quotient the state space, not necessarily the description |
| D2 | **built and tested, `d2.py`.** A replicate is a re-run of the preparation procedure. The floor is set by the finest preparation available; the re-reading definition fires on an empty world |
| D3 | **built and tested, `d3.py`.** LP membership in the local polytope, calibrated at the least favourable point of the composite null. Boundary exact at 1/√2, false positives 0/40 at every n |
| D4 | restate §1's scope: descriptions without a formal theory attached |

D1, D2 and D3 are implemented and verified. D4 remains a subtraction rather than an addition,
and is not repairable within the framework's own commitments.
