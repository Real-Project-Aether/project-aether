# L2, coarse-graining: the guard that says "it closes" is nearly worthless

L2 is the largest rung — 501 events — and the first thing this build produced was a finding about
the rung itself.

## Our published definition of L2 covers 7% of L2

The site and paper say L2 is *"decide two things are the same, or one thing is really two."*
Against the events:

| what the 501 L2 events carry | count | share |
|---|---:|---:|
| coarse-graining — CHANGE SCALE, ABSTRACT/IDEALIZE | 170 | **34%** |
| find a symmetry | 102 | 20% |
| add/remove a latent object (L4-flavoured) | 49 | 10% |
| change what is measured | 43 | 9% |
| **merge or split categories — our stated definition** | 34 | **7%** |
| no operator recorded at all | 37 | 7% |

**L2 is a residual bucket, not a move** — where events land when they change the description but
are not cleanly L1, L3 or L4. The dominant real sub-move is coarse-graining, and that is what was
built. The published wording needs correcting; that is tracked separately.

## The operation

A reduction is accepted only if both hold:

1. **it closes** — the coarse variables predict their own future without the fine state
2. **it pays** — the coarse state still carries the fine observables

Guard 1 is tested twice, deliberately: a coarse law *learned* from coarse readings alone, and one
*derived* by pushing the known fine law through the reduction. If derived closes and learned does
not, the failure was our fitter rather than the physics. On the lattice they agree to three
decimals (1.626 vs 1.628), so the negative result there is the physics.

## Result 1 — it finds a coarse level when one exists

`l2_slowfast.py`. Four slow variables coupled to eight fast ones, with the separation set by one
knob. The true coarse description is the slow four.

| separation | closes | pays | verdict |
|---:|---:|---:|---|
| 62.9× | 0.062 | 0.95 | **accept** |
| 25.2× | 0.156 | 0.91 | **accept** |
| 12.6× | 0.319 | 0.85 | does not close |
| 6.3× | 0.675 | 0.76 | does not close |
| 1.1× | 1.966 | 0.78 | does not close |

**The sensitivity floor is between 12× and 25× of timescale separation.** Below that no coarse
description closes well enough to accept, which is the correct answer rather than a miss.

## Result 2 — the finding: guard 1 is nearly worthless alone

Three vacuity controls, at a separation where a coarse level certainly exists:

| control | closes | pays | refused by |
|---|---:|---:|---|
| constant reduction (map everything to zero) | **0.000** | −1.00 | guard 2 only |
| random subspace | 2.167 | −0.06 | guard 1 |
| the **fast** modes instead of the slow ones | **0.000** | −1.00 | guard 2 only |

**Two of the three close perfectly.** The constant map is trivially autonomous, and the fast
subspace closes because fast modes decay to zero — it is a description of nothing, evolving
correctly. Guard 1 accepts both. Only guard 2 refuses them.

This is the same lesson L3 taught, on a different rung: there the structure test alone accepted a
false transfer five times in six. **The "it looks right" guard is cheap and nearly uninformative;
the "it pays and can be refuted" guard does the work.** Two rungs, two mechanisms, the same
result — which suggests it is a property of this kind of discovery operation and not an accident
of either build.

## Result 3 — the lattice has no coarse level, and says so

`l2_coarse.py` on the 6×6 two-flavour gauge system, 72 complex amplitudes. The spectral cut keeps
6 modes at a 2.05× gap. Nothing is accepted at any level: closure error falls monotonically from
1.71 at keep=2 to 0.00 only at keep=72, which is the identity. Retention likewise needs ~55 of 72
modes to stay positive.

That is a real answer about this system, not a failure: a nonlinear field theory whose spectrum
spans 2× has no scale separation to exploit. Effective theories arise where separations are
large, and this lattice's is not. Consistent with the slow-fast result, which puts the floor at
12–25×.

**Charges, reported and not blocked** (the coarse analogue of D1): the reduction destroys both
flavour charges, drift 7.5e-01 and 1.9e+00. Since it fails both guards anyway this costs nothing,
but the reporting path is exercised and works.

## Four bugs found by invariants, worth recording

The pay guard was wrong three times, and each time an invariant caught it rather than a plausible
number:

1. **fitted, elementwise features** — the fine observables are quadratic in the field, and
   `|c_i|²` has no cross terms. Failed even when the reduction kept *everything*.
2. **fitted, quadratic features** — scored a perfect **in-sample** R² at keep=6 with 37 features
   on 62 samples, and −113 out of sample. Pure overfitting.
3. **fitted, more data** — dominated by the scale spread across random initial states rather than
   by information content.
4. **fit-free reconstruction, within-trajectory denominator** — the observables are nearly
   conserved along a trajectory, so the denominator was almost zero and every truncation scored
   minus infinity.

The version that works is fit-free and normalised against the **ensemble** spread, and it carries
a hard invariant: a reduction that keeps everything must score exactly 1.000. It does. Any guard
without such an invariant should be distrusted — three of these four produced confident,
plausible, wrong numbers.

## What this does not settle

- **The proposer is linear.** It takes eigen-directions of the linearised dynamics. Kuramoto's
  phase reduction, the canonical L2 event, is not a linear projection and would not be found.
- **One system with a coarse level, one without.** Both are constructed. Nothing here has been
  run against a real physical system that is known to have an effective description.
- **Coarse-graining is 34% of L2.** The other two thirds — retyping what is measured, merging
  categories — still have no operation.
