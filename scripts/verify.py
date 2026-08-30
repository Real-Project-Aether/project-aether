#!/usr/bin/env python3
"""Check the repository holds together. Exits non-zero if anything fails.

`summary.py` checks the published NUMBERS reproduce. This checks the things that broke anyway:

  - every script runs FROM THE REPOSITORY ROOT, not only from beside itself. A relative data
    path in live_rotation_curve.py meant it worked for us and failed for anyone who cloned it.
  - the ladder is worded the same way everywhere. The site, the figure and the paper drifted
    apart once already, and nothing noticed.
  - no placeholder URLs survive.

Run from anywhere:  python3 scripts/verify.py
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


# --- every entry point must run from the repository root ------------------------------------
SCRIPTS = ["mechanism/test_mechanism.py", "mechanism/live_rotation_curve.py",
           "mechanism/discover_symmetry.py", "scripts/summary.py"]
for s in SCRIPTS:
    r = subprocess.run([sys.executable, s], cwd=ROOT, capture_output=True, timeout=3600)
    check(f"runs from repo root: {s}", r.returncode == 0,
          (r.stderr.decode()[-90:].strip() if r.returncode else ""))

# --- the ladder must be worded identically wherever it appears ------------------------------
site = (ROOT / "docs" / "index.html").read_text()
RUNGS = ["Regression", "Reinterpretation", "Retyping", "Transfer", "New object", "New move"]
for r in RUNGS:
    check(f"site names the rung: {r}", r.lower() in site.lower())

# the L2 definition drifted from its own evidence once; keep site and data in agreement
import csv
rows = list(csv.DictReader(open(ROOT / "data" / "classifications.csv")))
def lv(r): return [t.strip().rstrip("?") for t in re.split(r"[|,;]", r["levels"]) if t.strip()]
def ops(r): return {o.strip() for o in re.split(r"[|,;]", r["ops"]) if o.strip()}
l2 = [r for r in rows if "L2" in lv(r)]
coarse = sum(1 for r in l2 if ops(r) & {"CHANGE SCALE", "ABSTRACT/IDEALIZE"})
merge = sum(1 for r in l2 if ops(r) & {"CLASSIFY/ENUMERATE"})
check("L2 is defined by its dominant sub-move, not its rarest",
      coarse > merge and "level, or the quantity" in site,
      f"coarse-graining {coarse} vs merge/split {merge} of {len(l2)}")

# --- nothing may still point at a placeholder ------------------------------------------------
check("no placeholder URLs in the site", 'https://github.com/"' not in site)
readme = (ROOT / "README.md").read_text()
check("no placeholder URLs in the README", "<you>" not in readme)

# --- the site must be well formed -------------------------------------------------------------
bad = [t for t in ("html", "head", "body", "section", "div", "table", "tr", "td", "dl", "dd",
                   "ol", "li", "span", "footer", "script")
       if len(re.findall(rf"<{t}[ >]", site)) != len(re.findall(rf"</{t}>", site))]
check("site tags balance", not bad, ", ".join(bad))

# --- the documented requirement must actually be documented -----------------------------------
check("the model-server requirement is stated", "11434" in readme and "VLLM_MODEL" in readme)

width = max(len(n) for n, _, _ in CHECKS)
failed = 0
for name, ok, detail in CHECKS:
    failed += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<{width}}  {detail}")
print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks pass")
sys.exit(1 if failed else 0)
