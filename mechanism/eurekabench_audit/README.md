# Auditing EurekaBench's insight judge with typed nulls (pilot, 2026-10-02)

EurekaBench (Geng et al., arXiv:2610.00492) scores a discovered mechanism by whether an LLM judge
can derive a list of scientific insights from it. That judge is an acceptance guard, so its null
acceptance can be measured the way the paper measures the others: hand it submissions that carry
none of the insights and count how many insight verdicts come back true. Every verdict on these
submissions should be false.

## Nulls (`submissions/`)

Two tasks: A = `fusion/nt-edge` (13 insights), B = `neuroscience/zigzag-foraging` (10 insights).

| type | A | B |
|---|---|---|
| 1. assertion only: the causal chain is stated, the log has no data | A1 | B1 |
| 2. prediction only: an empirical fit with no mechanism | A2 | B2 |
| 3. A: a fluent mechanism that contradicts the task premise; B: a mechanism for a different question (the cord wave) | A3 | B3 |

EurekaBench's judge prompt says an assertion in the mechanism is never by itself grounds for a true
verdict, and that an answer from the judge's own knowledge must be false.

## Judge

Each submission was judged twice, by independent opus subagents of Claude Code, each given
EurekaBench's `insights_strict.md` prompt verbatim (`build_prompts.py`) under a hashed directory.
This approximates the published judges (Claude Opus 4.8 and GPT 5.6 Sol) and departs from the
benchmark in one respect: the simulator was not available, so the judge could only read. One
sentence saying so was appended to the prompt.

## Result (`summary.json`, `judgements/`)

| null type | true verdicts | NAR |
|---|---|---|
| assertion only | 9 of 46 | 0.20 |
| prediction only | 0 of 46 | 0.00 |
| wrong or off-topic | 0 of 46 | 0.00 |
| all | 9 of 138 | 0.07 (AnyNullPass 1) |

The assertion-only fusion null (A1) received true verdicts on 4 of 13 insights in both replicates
(I1, I2, I5, and I3 or I4). The judges' reasons quote the submission's own chain ("the cap is set by
the geometry, not by the power") as the derivation, although the log gives no measurement. B1
received one true verdict in one replicate. The prediction-only and the wrong or off-topic nulls
received none.

## Limits

Two tasks, two replicates, an approximate judge, and no simulator. A judge with the simulator could
test the asserted chain, so this measures what the judge accepts from text, which its prompt rules
out. EurekaBench ships no readable reference mechanism, so PAR (acceptance of a known positive) was
not measured. Running the published judges on the same nulls with the simulator is the next
measurement.
