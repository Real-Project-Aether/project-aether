# L3, transfer: structural similarity is not evidence. Only the transferred claim is.

189 corpus events sit on this rung. The operation: find a map carrying one domain's states into
another's so that what the first domain knows becomes a claim about the second — and then let the
second domain's own dynamics rule on it.

`l3_transfer.py`. Domain **A** is a 2-component field with a conserved charge it already knows.
Domain **B** is a larger 3-component system; whether it contains A's structure is what has to be
discovered. Two guards, both required:

1. **structure** — a fitted linear map must make B's probes agree with A's, to within 5%
2. **it must pay** — A's charge, pushed through that map, must be conserved by B's own evolution

Six seeds, each with a matched control whose symmetry is deliberately broken:

| | structure test (< 0.05) | charge test (< 1e-3) |
|---|---|---|
| real | median 0.006, worst 0.041 — **6/6 pass** | median 1.1e-11, worst 8.4e-11 — **6/6 pass** |
| control | median 0.044, best 0.033 — **5/6 also pass** | median 6.4e0, best 1.3e0 — **0/6 pass** |

Combined: **real accepted 6/6, control accepted 0/6.**

## The finding is which guard did the work

**The structure test is nearly worthless.** The control passes it five times in six. Fit a linear
map with enough freedom and B's probes can be made to resemble A's whether or not B shares
anything with A. On its own this guard would have accepted a false transfer in 83% of trials.

**The charge test separates them by ten orders of magnitude** — 1.5e10 between the worst real case
and the best control. It works because it is not a similarity score: it is a claim B did not have,
which B's dynamics can refute.

This is a caution for any analogy-driven approach to discovery, which is a common shape in the
literature. Finding that two domains *look* alike is cheap and nearly uninformative. The transfer
only means something when it hands the target a prediction the target can then break.

## What this does not settle

- One structural family, tested against one kind of broken control. A control that breaks the
  symmetry more subtly may well survive the charge test too.
- The map searched is **linear**. Real transfers in the corpus — statistical mechanics into
  optimisation, quantum fluctuations onto cosmological scales — are not linear recodings.
- The transferred claim here is always a conserved charge, because that is what our L1/L4
  machinery produces. A general L3 would transfer other kinds of claim, and we have no operation
  that produces those.
