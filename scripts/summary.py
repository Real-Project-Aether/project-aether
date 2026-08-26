#!/usr/bin/env python3
"""Recompute every statistic reported in the paper, straight from the shipped data.

Run from the repository root:  python3 scripts/summary.py
Exits non-zero if any published number fails to reproduce.
"""
import collections
import csv
import os
import re
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def levels(row):
    """Depth is multi-label within a single cell, pipe-separated."""
    return [t.strip() for t in re.split(r"[|,;]", row["levels"]) if t.strip()]


def main():
    db = sqlite3.connect(os.path.join(DATA, "prizes.sqlite"))
    one = lambda q: db.execute(q).fetchone()[0]
    with open(os.path.join(DATA, "classifications.csv")) as fh:
        rows = list(csv.DictReader(fh))

    n = len(rows)
    mode = collections.Counter(r["mode"] for r in rows)
    label = collections.Counter(l for r in rows for l in levels(r))
    grade = collections.Counter(r["evidence"].strip() for r in rows)
    ops = collections.Counter(
        o.strip() for r in rows for o in re.split(r"[|,;]", r["ops"]) if o.strip()
    )
    graded = sum(1 for r in rows if levels(r))

    # (label, computed, published)
    checks = [
        ("award records",            one("SELECT count(*) FROM awards"), 2327),
        ("prizes",                   one("SELECT count(*) FROM prizes"), 23),
        ("awards with a citation",   one("SELECT count(*) FROM awards "
                                         "WHERE citation IS NOT NULL AND trim(citation) <> ''"), 2221),
        ("official documents",       one("SELECT count(*) FROM documents"), 4044),
        ("laureate-to-paper links",  one("SELECT count(*) FROM papers"), 15693),
        ("laureates resolved",       one("SELECT count(*) FROM laureate_resolution"), 1969),
        ("annotated events",         n, 1547),
        ("mode REPRESENT",           mode["REPRESENT"], 698),
        ("mode RECORD",              mode["RECORD"], 479),
        ("mode REACH",               mode["REACH"], 186),
        ("mode ARTIFACT",            mode["ARTIFACT"], 102),
        ("mode TRACTABLE",           mode["TRACTABLE"], 32),
        ("mode ORGANIZE",            mode["ORGANIZE"], 11),
        ("mode OUT-OF-SCOPE",        mode["OUT-OF-SCOPE"], 31),
        ("mode UNRESOLVED",          mode["UNRESOLVED"], 8),
        ("events with a depth grade", graded, 727),
        ("events with NO depth grade", n - graded, 820),
        ("depth labels total",       sum(label.values()), 1041),
        ("L0", label["L0"], 131), ("L1", label["L1"], 38), ("L2", label["L2"], 501),
        ("L3", label["L3"], 189), ("L4", label["L4"], 181),
        ("evidence grade A",         grade["A"], 1442),
        ("evidence grade B",         grade["B"], 77),
        ("evidence grade C",         grade["C"], 28),
        ("distinct operators",       len(ops), 39),
        ("op ADD/REMOVE LATENT OBJECT", ops["ADD/REMOVE LATENT OBJECT"], 139),
        ("op MAP/TRANSFER",          ops["MAP/TRANSFER"], 135),
        ("op FIND INVARIANT/SYMMETRY", ops["FIND INVARIANT/SYMMETRY"], 110),
    ]

    width = max(len(c[0]) for c in checks)
    bad = 0
    print(f"{'statistic':<{width}}  {'data':>6}  {'paper':>6}")
    print("-" * (width + 16))
    for name, got, want in checks:
        ok = got == want
        bad += not ok
        print(f"{name:<{width}}  {got:>6}  {want:>6}  {'ok' if ok else '<<< MISMATCH'}")

    print("-" * (width + 16))
    print(f"{len(checks) - bad}/{len(checks)} reproduce")

    # The headline finding, derived rather than asserted.
    only_l0 = sum(1 for r in rows if set(levels(r)) == {"L0"})
    above = sum(1 for r in rows if any(l.rstrip("?") != "L0" for l in levels(r)))
    print(f"\nDoes every event fit 'regression' or 'reinterpretation'?")
    print(f"  regression only (L0 alone)      {only_l0:>5}  {100*only_l0/n:5.1f}%")
    print(f"  some move above L0              {above:>5}  {100*above/n:5.1f}%")
    print(f"  NEITHER (no description change) {n-graded:>5}  {100*(n-graded)/n:5.1f}%")
    print(f"  -> a two-way split misses {100*(n-graded)/n:.0f}% of awarded work.")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
