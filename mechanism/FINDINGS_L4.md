# L4, a new object: posit an entity to close a ledger that will not balance

181 corpus events. The shape is constant — something does not add up, and a new kind of thing is
introduced so that it does. Pauli's neutrino, colour charge, the charm quark.

This composes with the machinery that already works. The L1/L4 symmetry branch **finds**
conserved charges; this operation **uses** one: when a found charge fails to balance across an
observed process, the missing amount is a description of something not yet named.

`l4_posit.py`. Three species of charge +2, +1, +1; a parent of +2 decays into two products of +1
each, and species 2 is invisible to every probe. Charge is conserved by construction and the
construction asserts it, so any visible imbalance is the hidden product and nothing else.

| case | missing per event | significance | verdict |
|---|---:|---:|---|
| a product lands in the invisible species | +1.192 (14.8% of input) | 56σ | **posit a carrier** |
| the books balance | −0.011 (−0.1%) | −1.0σ | silent |
| imbalance within measurement noise | +0.029 (+0.4%) | 2.5σ | silent |

**The posited amount is checkable against the truth we hid**: 30% of two products going unseen,
with a mean of four parents, should leave 1.200 unaccounted. It recovered **1.192** — under 1%
error. Across the range it names the missing amount to within 0.01 absolute.

## What the sensitivity sweep actually showed

At 5% and 10% hidden the imbalance reaches 16σ and 30σ and the mechanism **still stays silent**.
That is not a miss. Significance is not the same as exceeding your systematic uncertainty: a
calibration error of a few percent produces exactly such an imbalance, and positing a new particle
on it is how false discoveries are made. The second guard requires the imbalance to be several
times the *noise scale*, not merely many standard errors from zero. It fires only above about 15%
of the input charge.

The case it is modelled on behaves the same way. Beta decay's energy deficit was not a 5% effect
— the electron spectrum ran continuously below a sharp endpoint, leaving a large fraction
unaccounted. Pauli was not reading a marginal excess.

## The honest limit

This assumes the conservation law holds and reads the residual off it. **The hard half of Pauli's
move was deciding to trust conservation over the measurements** — colleagues including Bohr were
prepared to abandon energy conservation instead. Nothing here makes that choice; it is handed the
law and does the inference. That inference is still the step that names a new object, but it is
the second step, not the first.

Also: what gets posited is a *quantity of charge*, not an entity with mass, spin and statistics.
Going from "something carries +1.19 and no probe sees it" to "a neutral fermion of spin 1/2" needs
the theorem-grade internal constraints the paper's boundary section says are out of reach.
