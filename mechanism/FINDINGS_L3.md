# L3, transfer: structural similarity is not evidence. Only the transferred claim is.

189 corpus events sit on this rung. The operation: find a map carrying one domain's states into
another's so that what the first domain knows becomes a claim about the second — and then let the
second domain's own dynamics rule on it.

`l3_transfer.py`. Domain **A** is a 2-component field with a conserved charge it already knows.
Domain **B** is a larger 3-component system; whether it contains A's structure is what has to be
discovered. Two guards, both required:

1. **structure** — a fitted linear map must make B's probes agree with A's, to within 5%
2. **it must pay** — A's charge, pushed through that map, must be conserved by B's own evolution

### Correction, 2026-08-31: the structure test was scored in-sample

The first version of this file reported that the control passed the structure test **5 times in
6**. That number was measured on the same 40 states the map was fitted to — `find_map` returned
the optimiser's own final residual and the guard was thresholded on it. The charge test already
used fresh states, so the two guards were never on equal footing, and a reviewer was right to ask.

`find_map` now also scores the residual on 40 **disjoint** states and the guard is thresholded on
that. Everything else is unchanged. The table below reports both columns; the shipped script runs
the six-seed sweep, so these figures can be reproduced rather than taken on trust.

Six seeds, each with a matched control whose symmetry is deliberately broken:

| | structure, in-sample | structure, **held out** (< 0.05) | charge test (< 1e-3) |
|---|---|---|---|
| real | median 0.0057 — 6/6 | median 0.0061, worst 0.0463 — **6/6 pass** | median 1.3e-11, worst 5.9e-11 — **6/6 pass** |
| control | median 0.0456 — 5/6 | median 0.0520, best 0.0405 — **2/6 pass** | median 5.2e0, best 2.8e0 — **0/6 pass** |

Combined: **real accepted 6/6, control accepted 0/6.**

## The finding is which guard did the work

**The structure test is much the weaker guard, but less catastrophically than we first reported.**
Held out, the control passes it in 2 seeds of 6 — 33%, not the 83% the in-sample score suggested.
Fit a linear map with enough freedom and B's probes can be made to resemble A's whether or not B
shares anything with A; a good part of that resemblance, but not all of it, is ordinary
overfitting.

**The count is threshold-sensitive and the spread is the real result.** The control's held-out
residuals span 0.0405–0.0588 against a 0.05 cut, so 2/6 is a fact about where that cut sits. The
real transfers sit an order of magnitude below it (median 0.0061). A single pass count should not
be quoted without the distribution.

**The charge test separates them by ten orders of magnitude** — 4.7e10 between the worst real case
and the best control, and unlike the structure test it needs no threshold tuning to do it. It works
because it is not a similarity score: it is a claim B did not have, which B's dynamics can refute.

This is a caution for any analogy-driven approach to discovery, which is a common shape in the
literature. Finding that two domains *look* alike is cheap; on held-out data it is still wrong a
third of the time. The transfer only means something when it hands the target a prediction the
target can then break.

## What this does not settle

- The structure test's failure rate depends on the 0.05 threshold, and the control residuals
  cluster just around it. We report the distribution for that reason.
- One structural family, tested against one kind of broken control. A control that breaks the
  symmetry more subtly may well survive the charge test too.
- The map searched is **linear**. Real transfers in the corpus — statistical mechanics into
  optimisation, quantum fluctuations onto cosmological scales — are not linear recodings.
- The transferred claim here is always a conserved charge, because that is what our L1/L4
  machinery produces. A general L3 would transfer other kinds of claim, and we have no operation
  that produces those.
