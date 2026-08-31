"""Score a filled re-annotation sheet against our labels.

Reports raw agreement and Cohen's kappa, on three things, in increasing order of how much the
paper depends on them:

  1. mode          the eight-way category (REPRESENT, RECORD, REACH, ...)
  2. description   the binary the headline numbers rest on: did this change a description at all?
  3. depth rung    L0..L5, scored only where BOTH annotators recorded a rung

Kappa, not raw agreement, is the number to quote: with 45% of events in one mode, two annotators
who both guess the majority label agree most of the time while sharing no judgement at all.

    python3 corpus/reannotation/score.py sample_filled.csv
"""
import argparse, csv, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DESCRIPTION_MODES = {"REPRESENT"}          # the modes that constitute a change of description


def kappa(pairs):
    """Cohen's kappa on a list of (a, b) label pairs."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in set(ca) | set(cb))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def report(name, pairs):
    if not pairs:
        print(f"  {name:<14} no overlapping labels")
        return
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    k = kappa(pairs)
    print(f"  {name:<14} n={n:<5} raw agreement {agree}/{n} = {100*agree/n:5.1f}%    "
          f"kappa = {k:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filled", help="the sheet from make_sample.py with your_mode filled in")
    a = ap.parse_args()

    ours = {}
    for r in csv.DictReader(open(DATA / "classifications.csv", encoding="utf-8")):
        ours[f"{r['prize_key']}|{r['year']}|{r['who']}"] = r

    theirs = list(csv.DictReader(open(a.filled, encoding="utf-8")))
    if "your_mode" not in (theirs[0] if theirs else {}):
        sys.exit("that file has no 'your_mode' column -- is it the sheet from make_sample.py?")

    mode_pairs, desc_pairs, depth_pairs, missing, unfilled = [], [], [], 0, 0
    for r in theirs:
        mine = ours.get(r["event_id"])
        if mine is None:
            missing += 1
            continue
        their_mode = (r.get("your_mode") or "").strip().upper()
        if not their_mode:
            unfilled += 1
            continue
        mode_pairs.append((mine["mode"].strip().upper(), their_mode))
        desc_pairs.append((mine["mode"].strip().upper() in DESCRIPTION_MODES,
                           their_mode in DESCRIPTION_MODES))
        my_lv = (mine.get("levels") or "").strip().upper()
        tl = (r.get("your_levels") or "").strip().upper()
        if my_lv and tl:
            depth_pairs.append((my_lv.split("|")[0], tl.split("|")[0]))

    print(f"\nre-annotation agreement against the shipped labels ({a.filled})\n")
    report("mode (8-way)", mode_pairs)
    report("description?", desc_pairs)
    report("depth rung", depth_pairs)
    if missing:
        print(f"\n  {missing} row(s) had an event_id not in the corpus and were ignored")
    if unfilled:
        print(f"  {unfilled} row(s) had no your_mode and were skipped")
    print("\n  Kappa below ~0.4 means the scheme is not reliably reproducible by a second reader,")
    print("  which would be a finding about the scheme and we would report it as one.\n")


if __name__ == "__main__":
    main()
