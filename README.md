# The Prize Corpus, and an operation for reinterpretation

**2,327 scientific prize records (1731–2026), 1,547 annotated by _what kind of thing the
contribution was_ — plus a mechanism for the one discovery move that annotation shows is
missing from every automated system we could find.**

[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-blue.svg)](LICENSE-DATA.md)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

Existing prize datasets are bibliometric — who won, what they published, how it was cited.
None of them records whether a piece of work *reframed* something, *measured* something new,
*extended reach*, or *built an instrument*. That distinction is what this corpus adds.

> **Read [Limitations](#limitations) before using the annotations. There is exactly one
> annotator and it is a language model.**

---

## The headline finding

Most awarded work is not a change of description at all.

| | events | share |
|---|---:|---:|
| changed the **description** (`REPRESENT`) | 698 | 45.1% |
| changed the **facts** (`RECORD`, `REACH`) | 665 | 43.0% |
| built something (`ARTIFACT`, `TRACTABLE`) | 134 | 8.7% |
| other / out of scope | 50 | 3.2% |

**820 of 1,547 events carry no depth grade**, because no description changed. A two-way split
between "fitting" and "reinterpreting" — the way machine learning usually frames discovery —
would miss more than half of what gets awarded.

And within the description column, the move machine learning knows how to make (**L0**, working
inside a fixed description) accounts for **131 of 1,041 depth labels**. Strict reinterpretation
— same predictions, less structure, Einstein against Lorentz — is **38 events, under 3%**. It is
the rarest thing in the corpus.

## The mechanism (`mechanism/`)

Among **physics** prizes, 65% changed a description (vs 45% across all science), and the rung
machine learning implements covers **42 of 542** physics depth labels — one in thirteen. Placing
fourteen automated-discovery systems on the same ladder, all sit on that rung or one above it,
and **none performs L1**: take a theory that already fits, return one that fits identically while
carrying less.

`mechanism/` implements that operation. Structure no measurement can see is the null space of the
prediction Jacobian. Removing it naively **deletes conservation laws** — a global symmetry is
invisible to every probe too — so the operation first asks whether the invariance survives being
applied *point by point*:

| | |
|---|---|
| invariant **point by point** | redundancy → quotient it → **L1** |
| invariant only **uniformly** | symmetry → keep it, read off a conserved charge → **L4** |

```bash
cd mechanism
python3 test_mechanism.py        # 21/21 engineering checks
python3 discover_symmetry.py     # finds 5 invariances in a lattice gauge system,
                                 # sorts 1 redundancy from 4 symmetries, emits 4 charges
python3 live_rotation_curve.py   # NGC 3198: where the criterion goes silent, and why
```

Those three need only NumPy and SciPy. One script does need more:

```bash
python3 llm_propose.py           # needs a chat model on an OpenAI-compatible endpoint
```

It expects one at `http://127.0.0.1:11434/v1` serving `gpt-oss:20b` — point it elsewhere with
`VLLM_MODEL_URL` and `VLLM_MODEL`. Without a server it will not run, but you do not need one to
check the result: `llm_propose_out.txt` is the recorded output of the run the findings describe.

On the lattice, handed eight candidate generators and told nothing about which are real, it finds
five invariant directions, isolates the gauge redundancy, and emits four conserved charges
accurate to 1e-10 — while the redundancy's would-be charge drifts by 0.86. Without the sort, all
four conservation laws would have been destroyed to remove one redundancy.

See [`mechanism/FINDINGS.md`](mechanism/FINDINGS.md) for full results, **including where it
fails**: on galaxy rotation curves the criterion is silent, because it needs two descriptions
that *agree* on the data. And [`mechanism/STANDARD_MODEL.md`](mechanism/STANDARD_MODEL.md) bounds
it: four of eight Standard Model concepts were bought by a **theorem** before any experiment, and
no probe-based method can reach those.

## Quick start

```bash
git clone https://github.com/Real-Project-Aether/project-aether.git && cd project-aether
python3 scripts/summary.py          # reproduces every number in the paper
python3 scripts/verify.py           # checks the repo holds together; non-zero if not
```

```python
import sqlite3, pandas as pd
db = sqlite3.connect("data/prizes.sqlite")

# every awarded contribution that introduced a kind of object its language lacked
pd.read_sql("""
    SELECT a.year, a.laureate, c.levels, c.ops, a.citation
      FROM awards a
      JOIN classifications c
        ON a.prize_key = c.prize_key AND a.year = c.year
     WHERE c.levels LIKE '%L4%'
     ORDER BY a.year DESC
""", db)
```

## Files

| file | rows | what it is |
|---|---:|---|
| `data/prizes.sqlite` | — | everything below, indexed, in one file |
| `data/awards.csv` | 2,327 | prize, year, laureate, official citation, institution, source |
| `data/classifications.csv` | 1,547 | mode, depth, operators, trigger, evidence grade, justification |
| `data/papers.csv` | 15,693 | laureate → paper links; 14,968 carry a DOI or arXiv ID |
| `data/documents_index.csv` | 4,044 | document type and **source URL** — the text itself is not shipped |
| `data/prizes.csv` | 23 | prize registry with official sites |
| `scripts/refetch_documents.py` | — | re-downloads the full text from `source_url` |
| `scripts/summary.py` | — | recomputes every statistic reported in the paper |

Prizes covered: the three science Nobels, Copley, Turing, Fields, Abel, Wolf (5 fields),
Crafoord, Kavli (3), Shaw (3), Breakthrough (3), Lasker, Templeton, Dirac, Boltzmann, Onsager
and others. **2,221 of 2,327** carry the awarding body's own citation; the remaining 106 never
had one published.

## The annotation scheme

Two axes. They stay separate because they are different kinds of measurement: **mode** is
nominal (no ordering holds between "measured something new" and "retyped what counts as the
same"), while **depth** is ordinal, and it is the ordering that carries the argument.

### Mode — what the contribution changed (every event gets exactly one)

| mode | n | what it means | example |
|---|---:|---|---|
| `REPRESENT` | 698 | the description used to write the phenomenon down | Prusiner 1997 — "a new biological principle of infection" |
| `RECORD` | 479 | facts nobody had | Pääbo 2022 — genomes of extinct hominins |
| `REACH` | 186 | what can be observed or reached at all | Karikó & Weissman 2023 — mRNA made usable as a platform |
| `ARTIFACT` | 102 | a thing that works | Goodenough et al. 2019 — the lithium-ion battery |
| `TRACTABLE` | 32 | what can be computed | Hassabis & Jumper 2024 — protein structure |
| `ORGANIZE` | 11 | a field's standing or reach | founding, legitimating, carrying a field outward |
| `OUT-OF-SCOPE` | 31 | not a scientific contribution | chiefly Templeton awards to religious figures |
| `UNRESOLVED` | 8 | citation too thin to place | early Copley Medals, 1803–1824 |

### Depth — how far the description moved (only where one did)

| level | n | what it means | example |
|---|---:|---|---|
| `L0` | 131 | solved inside the existing description | Duminil-Copin 2022 — phase transitions in 3D and 4D |
| `L1` | 38 | same predictions, **less** structure | Nambu 2008 — symmetry breaking carried into particle physics |
| `L2` | 501 | retype, merge or split what counts as the same | Cook 1982 — NP-completeness |
| `L3` | 189 | bridge one domain onto another | Bennett & Brassard 2025 — quantum communication |
| `L4` | 181 | introduce a kind of object the language lacked | Scholze 2018 — perfectoid spaces |
| `L5` | 1 | change the repertoire of moves itself | Cohen 1966 — forcing |

**`L0` does not mean easy.** It means no description changed. Duminil-Copin won a Fields Medal
at L0.

Depth is multi-label, so 1,041 labels fall on 727 events — the 698 `REPRESENT` events plus 29
whose citation rewards a second, description-changing move as well (Chadwick 1935 found the
neutron, `RECORD`, and the neutron is also a new kind of object, `L4`).

Events additionally carry labels from **39 operators**: `ADD/REMOVE LATENT OBJECT` (139),
`MAP/TRANSFER` (135), `FIND INVARIANT/SYMMETRY` (110), `CHANGE SCALE` (103), and others.

## Limitations

**One annotator, and it is a model.** Every count above is one annotator's judgement, and that
annotator is a large language model (Claude, Anthropic) reading each official citation under
human direction, in 13 batch files shipped with the release. These are **not** the judgements of
human domain experts. There is exactly one annotator, so no inter-annotator agreement statistic
exists and we do not report a proxy for one. Evidence grades — A (1,442), B (77), C (28) — are a
self-reported confidence signal, not a validation.

**The `RECORD`/`REPRESENT` boundary is the soft one.** A paper that reports a measurement often
reframes something too, and which half the citation rewards is a judgement call. Read the 820 as
a *lower bound* on how badly a two-way split fails, not as a constant.

**The scheme is not symmetric.** The facts column draws its own coarse distinction — facts newly
in hand (`RECORD`) versus reach newly extended (`REACH`) — but encodes it as two modes rather
than as a ladder. A scheme that graded every column would be cleaner. We did not build one.

**Hindsight and selection.** Prizes are awarded with decades of hindsight, so the corpus samples
work that turned out to matter, and citations justify a decision already taken. Physics and
mathematics are better represented than biology or the earth sciences.

## What would help most

**Independent re-annotation.** Even a few hundred events, by people who disagree with us, would
establish whether the scheme is reproducible or whether this is one system's habits written
down. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

See [CITATION.cff](CITATION.cff). The paper is at
[real-project-aether.github.io/project-aether/paper.pdf](https://real-project-aether.github.io/project-aether/paper.pdf);
the LaTeX source is not part of this repository.

## Licence

Annotations, compiled metadata and code: **CC BY 4.0** / **MIT**. The official citation text is
the awarding bodies' and is not ours to license — see [LICENSE-DATA.md](LICENSE-DATA.md).
