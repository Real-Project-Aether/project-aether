# Frozen analysis specification — X2 matched controls

**Written 2026-09-01, before the matched-control run.** Not a formal preregistration, but it is
timestamped in a public repository and it fixes the choices that would otherwise be made after
seeing results. Three times in this project a test was changed after a first run: X1's null suite
was enlarged, X2's feature-selection rule was rewritten, and the causal guard was re-centred. Each
change was defensible and each is disclosed in the paper, but the pattern is why this file exists.

If the run contradicts what is specified here, **the specification wins and the contradiction is
reported.**

---

## 1. What is being measured

CAR, the consequence agreement rate of Eq. 2 in the paper:

```
CAR = P[ H(c) = 1 | G(c) = 1, c not in V ]
```

with `G` the enrichment guard and `H` the intervention test. The open question the run settles is
whether CAR ≈ 0.62 for enrichment-selected features is **high or low**, which requires a matched
comparison. Without it the number is uninterpretable, which is the state the paper is in now.

## 2. Primary endpoint, fixed in advance

**ΔCAR = CAR(selected) − CAR(matched control)**, with a hierarchical bootstrap interval.

- If the interval excludes zero and ΔCAR > 0, enrichment carries causal information beyond what a
  comparable arbitrary feature carries.
- **If the interval includes zero, the paper says the enrichment guard is not distinguishable from
  a matched control on this test, and the X2 claim is withdrawn to that effect.** This outcome is
  live and is not a failure of the run.

Secondary: CAR as a function of enrichment (calibration), and stratification by firing frequency.

## 3. The matched control

For each selected feature `f` with concept token `t_f` and held-out firing rate `r_f`, draw a
control feature `g` from the same arm satisfying:

- firing rate within ±20% of `r_f`;
- `g`'s own top concept token ≠ `t_f`;
- `g` not itself in the selected set.

Then score `g`'s intervention **on `f`'s concept mask**. This asks the question that matters: does
ablating a comparably active but unrelated feature damage `f`'s concept as much as ablating `f`
does? If it does, the specificity we measured is generic damage.

Decoder-norm matching is **not** applied and the reason is stated here rather than discovered
later: our decoder columns are renormalised to unit norm every training step, so all features
already have identical decoder norm and the match is vacuous.

If no control satisfies the constraints, the feature is **excluded from the paired comparison** and
the number of exclusions is reported.

## 4. Feature selection — unchanged from the shipped run

Frozen as already implemented, so the pipeline is not tuned to the outcome:

| stage | rule |
|---|---|
| eligible firing rate | `2e-3 < freq < 5e-2` on held-out tokens |
| concept frequency | top token must occur ≥ 60 times in held-out text |
| minimum firings | ≥ 40 |
| ranking | by enrichment, descending |
| selected | top 150 per arm |

The flow counts `N0 → N1 → N2 → N3 → 150` are recorded per arm and reported, so that "all selected
features clear the enrichment threshold" is visibly a construction and not a finding.

## 5. Arms

Unchanged: Pythia-160m layers 4 and 8, Pythia-410m-deduped layers 8 and 16. Four arms, WikiText-2,
80/20 split, held-out scoring throughout.

**Four arms cannot support a claim about generalisation across models**, and the paper will not
make one.

## 6. Thresholds

- enrichment `> 4` for `G` — unchanged;
- intervention: a feature is supported when `ΔL_C > 0`, `ΔL_C − ΔL_¬C > 0`, and the bootstrap
  interval on the difference excludes zero.

The unstable ratio `ΔL_C / ΔL_¬C` used in the shipped run is **replaced** by the standardised
difference `S_f = (ΔL_C − ΔL_¬C) / SE[·]`, because the ratio explodes when the denominator is near
zero — a defect visible in the shipped results, where one feature scored 1644. The ratio is still
recorded for comparability.

## 7. Uncertainty

Hierarchical bootstrap, 1000 resamples:

1. resample the four arms with replacement;
2. within each, resample held-out sequences with replacement;
3. recompute intervention effects;
4. cluster features sharing a top token, since they are not independent.

The shipped CI of 59–66% treated 600 features as independent Bernoulli trials. **The new interval
will be wider and that is the correct direction.**

## 8. Exclusions

- features with fewer than 30 concept positions in the evaluation set;
- features with no admissible matched control (counted and reported);
- arms failing to train to ≥ 90% variance explained (none so far).

## 9. What is not being run

X1 across 20–40 configurations, nonlinear coarse-graining proposers, richer concept families than
token identity, and human validation of the taxonomy. These are named in the paper as future work
rather than omitted silently.
