"""Build the redistributable release from the working database.

The working database carries 50.2M characters of citation and lecture text downloaded from
awarding bodies. That text belongs to them, so it is NOT redistributed here. What ships is the
metadata, our annotations, and the source URL of every document, plus a script that re-downloads
the bodies. Anyone can reconstruct the full corpus; nobody has to take our copy of someone
else's prose.
"""
from __future__ import annotations
import csv, json, os, sqlite3, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "prize_db", "prizes.sqlite")
CLS = os.path.join(ROOT, "validation", "classification_full.json")
OUT = os.path.join(HERE, "prizes.sqlite")

# every column of `documents` except the one that carries the text itself
DOC_COLS = ["doc_id", "prize_key", "year", "doc_type", "laureate", "field", "source_url",
            "rel_path", "chars"]


def build_sqlite():
    if os.path.exists(OUT):
        os.remove(OUT)
    src = sqlite3.connect(SRC)
    dst = sqlite3.connect(OUT)

    for table in ("prizes", "awards", "laureate_resolution", "papers"):
        ddl = src.execute(
            "select sql from sqlite_master where type='table' and name=?", (table,)
        ).fetchone()[0]
        dst.execute(ddl)
        rows = src.execute(f"select * from {table}").fetchall()
        n = len(rows[0]) if rows else 0
        dst.executemany(f"insert into {table} values ({','.join('?' * n)})", rows)
        print(f"  {table:22s} {len(rows):6d} rows")

    # documents WITHOUT body -- the whole point of the release build
    dst.execute("""CREATE TABLE documents(
        doc_id INTEGER PRIMARY KEY, prize_key TEXT, year INT, doc_type TEXT, laureate TEXT,
        field TEXT, source_url TEXT, rel_path TEXT, chars INT)""")
    rows = src.execute(f"select {','.join(DOC_COLS)} from documents").fetchall()
    dst.executemany(f"insert into documents values ({','.join('?' * len(DOC_COLS))})", rows)
    print(f"  {'documents (no body)':22s} {len(rows):6d} rows")

    # the annotations, as a first-class table
    dst.execute("""CREATE TABLE classifications(
        prize_key TEXT, year INT, who TEXT, mode TEXT, modes TEXT, levels TEXT, ops TEXT,
        trigger TEXT, confidence TEXT, evidence TEXT, grain TEXT, subdepth TEXT,
        batch TEXT, justification TEXT)""")
    cls = load_classifications()
    dst.executemany("insert into classifications values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [tuple(r[k] for k in CLS_COLS) for r in cls])
    print(f"  {'classifications':22s} {len(cls):6d} rows")

    for idx in ("CREATE INDEX ix_aw_year ON awards(year)",
                "CREATE INDEX ix_aw_pk ON awards(prize_key)",
                "CREATE INDEX ix_cls ON classifications(prize_key, year)"):
        dst.execute(idx)
    dst.commit()
    src.close(); dst.close()


CLS_COLS = ["prize_key", "year", "who", "mode", "modes", "levels", "ops", "trigger",
            "confidence", "evidence", "grain", "subdepth", "batch", "justification"]


def load_classifications():
    raw = json.load(open(CLS))
    rows = raw if isinstance(raw, list) else list(raw.values())
    out = []
    for r in rows:
        out.append({
            "prize_key": r.get("prize_key"), "year": r.get("year"), "who": r.get("who"),
            "mode": r.get("mode"),
            "modes": "|".join(r.get("modes") or []),
            "levels": "|".join(r.get("levels") or []),
            "ops": "|".join(r.get("ops") or []),
            "trigger": r.get("trigger"), "confidence": r.get("conf"),
            "evidence": r.get("evidence"), "grain": r.get("grain"),
            "subdepth": r.get("subdepth"), "batch": r.get("batch"),
            "justification": r.get("justification"),
        })
    return out


def dump_csv(name, header, rows):
    path = os.path.join(HERE, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name:26s} {len(rows):6d} rows")


def build_csvs():
    db = sqlite3.connect(OUT)

    cols = ["prize_key", "prize_name", "sub_prize", "year", "laureate", "full_name",
            "citation", "portion", "institution", "field", "official_url", "source"]
    dump_csv("awards.csv", cols,
             db.execute(f"select {','.join(cols)} from awards order by prize_key, year").fetchall())

    dump_csv("classifications.csv", CLS_COLS,
             db.execute(f"select {','.join(CLS_COLS)} from classifications "
                        "order by prize_key, year").fetchall())

    pcols = ["laureate", "title", "doi", "arxiv", "citations", "year", "journal",
             "n_authors", "source", "match", "url"]
    dump_csv("papers.csv", pcols,
             db.execute(f"select {','.join(pcols)} from papers order by laureate").fetchall())

    dump_csv("documents_index.csv", DOC_COLS,
             db.execute(f"select {','.join(DOC_COLS)} from documents order by doc_id").fetchall())

    prz = ["prize_key", "prize_name", "field", "n_awards", "first_year", "last_year",
           "wikipedia_page", "official_site"]
    dump_csv("prizes.csv", prz,
             db.execute(f"select {','.join(prz)} from prizes order by prize_key").fetchall())
    db.close()


if __name__ == "__main__":
    print("building release database (documents.body excluded)")
    build_sqlite()
    print("\nwriting CSVs")
    build_csvs()
    size = os.path.getsize(OUT) / 1e6
    print(f"\nrelease sqlite: {size:.1f} MB")
