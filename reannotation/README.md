# Re-annotation kit

Every count in the paper rests on **one annotator**. There is therefore no inter-annotator
agreement statistic for this corpus, and we do not report a proxy for one. This directory is the
tooling for someone else to produce that number.

It takes about an hour for 120 events, needs no physics background beyond reading a prize
citation, and does not require running any of our other code.

```bash
python3 make_sample.py                  # -> sample_blind.csv, 120 events, our labels withheld
#   ... fill in your_mode and your_levels ...
python3 score.py sample_filled.csv      # raw agreement and Cohen's kappa on three axes
```

The sample is a simple random draw with a fixed seed, deliberately **not** stratified: stratifying
would oversample the rare categories and inflate kappa against the true marginals.

## What you are deciding

**First, which of three things the contribution changed.** This is the `your_mode` column.

| mode | the contribution changed... | example |
|---|---|---|
| `REPRESENT` | the **description** — how the subject is written down | renormalisation group; gauge theory |
| `RECORD` | the **facts** in hand — a measurement, an observation | the top quark's mass; a new pulsar |
| `REACH` | the **instruments** — what can now be done at all | the scanning tunnelling microscope |
| `TRACTABLE` | what is **computable** in practice | structure prediction; fast algorithms |
| `ARTIFACT` | a **built thing** that is itself the contribution | a laser; a detector |
| `ORGANIZE` | how a **community or field** is arranged | founding a discipline |
| `OUT-OF-SCOPE` | nothing scientific (peace, literature, service) | |
| `UNRESOLVED` | the citation does not say enough to decide | |

The boundary that matters most, and the one we most want checked, is **`RECORD` against
`REPRESENT`**: measuring something new versus changing what the measurements are taken to be
about. When a citation supports both readings, prefer the one the citation's own wording leads
with.

**Second, if and only if the mode is `REPRESENT`, how deep the change went.** This is
`your_levels`. Leave it blank for every other mode.

| rung | the description changed by... |
|---|---|
| `L0` | solving inside the description that already existed — no change of language |
| `L1` | carrying the **same predictions with less structure** (Einstein dropping the aether) |
| `L2` | changing the **level or quantity** it is written in (coarse-graining, effective theories) |
| `L3` | **carrying another field's language in** (statistical mechanics into optimisation) |
| `L4` | adding a **kind of thing** the language did not have (the neutrino; colour charge) |
| `L5` | changing **what counts as a move** at all |

Write one rung. If two genuinely apply, separate with `|` — only the first is scored.

## What we will do with your numbers

Report them, whatever they are, in the next version of the paper, with your name if you want it
there. **A kappa below about 0.4 would mean the scheme is not reliably reproducible by a second
reader.** That is a finding about the scheme rather than about you, and it is worth more to us
than a confirmation: it would tell us the ladder needs re-cutting, which is a conclusion we
already suspect for `L2` (see the limitations section of the paper).

Send the filled CSV, or open a pull request, via the project site:
<https://real-project-aether.github.io/project-aether/>

## Licence

Our annotations are CC BY 4.0. The prize citation text in the sample belongs to the awarding
bodies and is reproduced here for research use.
