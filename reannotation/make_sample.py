"""Draw a blind re-annotation sample.

The corpus has ONE annotator, so no inter-annotator agreement statistic exists for it. This is
the single largest weakness in the paper that uses it, and it is one an outsider can fix without
running any of our code. This script produces the sheet to fill in; score.py grades it.

The sample is a simple random draw with a fixed seed, NOT stratified by mode. Stratifying would
oversample the rare categories and inflate kappa against the true marginals, which is exactly the
number a reader should not trust.

    python3 corpus/reannotation/make_sample.py            # 120 events -> sample_blind.csv
    python3 corpus/reannotation/make_sample.py --n 300 --seed 7
"""
import argparse, csv, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "sample_blind.csv"))
    a = ap.parse_args()

    cls = list(csv.DictReader(open(DATA / "classifications.csv", encoding="utf-8")))
    awards = {(r["prize_key"], r["year"], r["laureate"]): r
              for r in csv.DictReader(open(DATA / "awards.csv", encoding="utf-8"))}

    rows = []
    for r in cls:
        key = (r["prize_key"], r["year"], r["who"])
        aw = awards.get(key)
        if aw and aw.get("citation", "").strip():
            rows.append((key, aw, r))

    random.Random(a.seed).shuffle(rows)
    picked = rows[: a.n]

    out = Path(a.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event_id", "prize", "year", "laureate", "citation",
                    "your_mode", "your_levels", "your_notes"])
        for (pk, yr, who), aw, _ in picked:
            w.writerow([f"{pk}|{yr}|{who}", aw["prize_name"], yr, who,
                        aw["citation"].strip(), "", "", ""])

    print(f"wrote {out}  ({len(picked)} events, seed {a.seed})")
    print("Our labels are NOT in this file. Fill your_mode and your_levels, then run score.py.")


if __name__ == "__main__":
    main()
