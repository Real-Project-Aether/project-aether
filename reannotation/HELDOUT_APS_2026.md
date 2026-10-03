# Held-out check: APS prize events not in the annotated corpus

`data/awards.csv` holds award records for the Heineman Prize (2024, 2025, 2026) and the Onsager
Prize (2025, 2026) that carry no citation and were never annotated. On 2026-10-02 the official
citations were taken from the APS prize pages and the five events were annotated blind, from the
citation and the scheme in `README.md` alone, by an annotator with no access to the corpus labels.
Results are in `data/heldout_aps_2026.csv`. They are kept out of `classifications.csv`, so the
counts the paper reports are unchanged.

What the corpus predicts for these two prizes, from their history: every earlier event is
REPRESENT (84 Heineman rows, 43 Onsager rows), and the first rung is L2 for 51 of 84 and 34 of 43.

| | blind label | matches the historical prediction |
|---|---|---|
| mode | REPRESENT for 5 of 5 | 5 of 5 |
| first rung | L0, L3, L4, L0, L2 | 1 of 5 (L2) |

The mode is reproduced. The rung is not: the blind annotator reads two citations as solving inside
an existing description (L0) where the corpus annotator chose L2 for most events of these prizes,
and two of the five rungs are marked low confidence. This is one small sample of five events and a
second annotator, so it bears on the reproducibility of the rung labels (the open question in
`README.md`), not on the paper's counts.

Not covered: the Heineman 1967 (Wick) and 1997 (Lehmann) records, for which the APS page gives no
citation, and the APS prizes for 2027, which had not been announced when this was written.
