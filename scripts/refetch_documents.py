"""Rebuild the full-text layer of the corpus from source.

The release ships metadata and short official citations, not the ~50M characters of laureate
lectures, prize essays and award-ceremony documents that the working corpus holds -- that text
belongs to the awarding bodies. `documents_index.csv` carries the source URL of all 4,044 of
them, so the full corpus is reconstructible by anyone who wants it.

    python3 refetch_documents.py --out docs/            # everything, politely paced
    python3 refetch_documents.py --out docs/ --prize nobel_physics
    python3 refetch_documents.py --out docs/ --limit 20 # try it before committing to hours

Written to be re-runnable: anything already on disk is skipped, so an interrupted run resumes.
Be considerate of the servers -- the default pace is deliberate and there is no reason to
lower it.
"""
from __future__ import annotations
import argparse, csv, os, subprocess, sys, time

PACE = 1.5           # seconds between requests
UA = "Mozilla/5.0 (compatible; prize-corpus-refetch/1.0; academic use)"


def fetch(url, dest):
    r = subprocess.run(["curl", "-sL", "--compressed", "-m", "90", "-A", UA, "-o", dest, url],
                       capture_output=True)
    if r.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) < 500:
        return False
    head = open(dest, encoding="utf-8", errors="replace").read(4000)
    # the Internet Archive serves its own outage notice with HTTP 200; a size check alone
    # accepts it, and 93 files in the original harvest were silently poisoned this way
    if "Temporarily Offline" in head:
        os.remove(dest)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="documents_index.csv")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--prize", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.index))
            if (r["source_url"] or "").startswith("http")
            and (a.prize is None or r["prize_key"] == a.prize)]
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(a.out, exist_ok=True)

    ok = skip = fail = 0
    for i, r in enumerate(rows, 1):
        dest = os.path.join(a.out, f"{r['doc_id']}_{r['prize_key']}_{r['year']}.html")
        if os.path.exists(dest) and os.path.getsize(dest) > 500:
            skip += 1
            continue
        if fetch(r["source_url"], dest):
            ok += 1
        else:
            fail += 1
            print(f"  failed: {r['source_url']}", file=sys.stderr)
        time.sleep(PACE)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  ok={ok} skipped={skip} failed={fail}", flush=True)

    print(f"\ndone: {ok} fetched, {skip} already present, {fail} failed, of {len(rows)}")
    if fail:
        print("Failures are expected: pages move and some archives rate-limit. Re-run to retry.")


if __name__ == "__main__":
    main()
