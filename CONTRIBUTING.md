# Contributing

The single most useful contribution is **independent re-annotation**.

## Why

Every annotation in this corpus was produced by one annotator, and that annotator is a language
model. We cannot tell from the inside whether the scheme is reproducible or whether it records
one system's habits. Only someone else applying it can settle that.

## How to re-annotate

1. Take a sample from `data/classifications.csv` — a few hundred events is plenty. Sampling
   across prizes and eras is more informative than taking a contiguous block.
2. Read **only** `awards.csv.citation` for each event. That is all our annotator saw. Do not
   read our `mode`, `levels`, `ops` or `justification` first — that is the whole point.
3. Assign a mode, and a depth if a description changed. The scheme is defined in the README.
4. Send us your labels and we will publish the agreement statistic **whatever it says**,
   including if it is bad. Open an issue or a PR against `contrib/`.

Disagreement is the useful signal. If you think an event is miscoded, say so and say why — the
`justification` column exists so that a disagreement can be about something specific.

## Other things worth doing

- **Fill a coverage gap.** Biology and the earth sciences are thinner than physics and
  mathematics. Prizes we do not cover at all are listed in `data/prizes.csv` by omission.
- **Fix a record.** Wrong laureate, wrong year, a citation we mis-transcribed, a dead
  `source_url`. Small and concrete, and each one makes the corpus more trustworthy.
- **Attack the scheme.** The `RECORD`/`REPRESENT` boundary is the soft one, and the facts column
  has no depth ladder. If you can design a symmetric scheme that survives contact with 1,547
  real citations, that supersedes this one and we will say so.

## What we will not merge

Annotations produced by prompting a language model with our existing labels in context. That
reproduces our annotator rather than testing it. If you use a model, say which one and show the
prompt; a genuinely independent model run is interesting, a contaminated one is not.

## Provenance

Please keep `source` and `source_url` populated on anything you add, and note in your PR how you
established each fact. The corpus is only as good as its ability to be checked.
