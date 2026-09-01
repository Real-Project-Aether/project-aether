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
           "mechanism/discover_symmetry.py", "mechanism/l2_slowfast.py",
           "mechanism/l2_coarse.py", "mechanism/l3_transfer.py", "mechanism/l4_posit.py",
           "scripts/summary.py"]
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

# --- the paper names four operations; all four must actually ship ------------------------------
for f in ("l2_coarse.py", "l3_transfer.py", "l4_posit.py", "concept_space.py"):
    check(f"the operation ships: {f}", (ROOT / "mechanism" / f).exists())
for f in ("FINDINGS_L2.md", "FINDINGS_L3.md", "FINDINGS_L4.md", "FINDINGS_L5.md"):
    check(f"its findings ship: {f}", (ROOT / "mechanism" / f).exists())

# --- the external replication must ship, and its headline numbers must hold -------------------
# The sweep needs `pip install dysts` and ~15 minutes, so it is not in SCRIPTS. Its shipped
# results are checked here instead, because the paper quotes them.
import json
ext = ROOT / "mechanism" / "external_dysts.json"
check("the external replication ships", ext.exists() and
      (ROOT / "mechanism" / "external_dysts.py").exists())
if ext.exists():
    ex = json.loads(ext.read_text())
    const_close = sum(1 for r in ex if r["res"]["constant"]["closes"])
    const_pay = max(r["res"]["constant"]["r2"] for r in ex)
    both = sum(1 for r in ex if r["res"]["constant"]["closes"] and r["res"]["constant"]["pays"])
    check("external: guard 1 accepts the empty reduction on 124 of 129",
          len(ex) == 129 and const_close == 124, f"{const_close} of {len(ex)}")
    check("external: guard 2 refuses every empty reduction",
          both == 0 and const_pay < 0.50, f"max pay {const_pay:+.3f}, survivors {both}")

# --- the guards must be shown sensitive, not only specific -------------------------------------
pos = ROOT / "mechanism" / "external_positives.json"
check("the positive-control suite ships", pos.exists() and
      (ROOT / "mechanism" / "external_positives.py").exists())
if pos.exists():
    d = json.loads(pos.read_text())
    check("positives: precision 1.00, recall 0.71 across four dimensions",
          d["precision"] == 1.0 and abs(d["recall"] - 0.708) < 0.02 and d["n_systems"] == 24,
          f"P={d['precision']:.2f} R={d['recall']:.2f} over {d['n_systems']} systems")
    check("positives: closure alone still accepts most empty candidates",
          d["closure_only_accepts"] == 47 and d["fp"] == 0,
          f"{d['closure_only_accepts']} of {d['n_negatives']} close, {d['fp']} accepted by both")
    check("positives: the guard is sharp in every dimension",
          all(v == 0.05 for v in d["breakdown_theta"].values()),
          f"acceptance lost at {sorted(set(d['breakdown_theta'].values()))} rad")

# --- the constant-reduction result must not be an artefact of our coordinates -----------------
co = ROOT / "mechanism" / "external_coordinates.json"
check("the coordinate-invariance test ships", co.exists() and
      (ROOT / "mechanism" / "external_coordinates.py").exists())
if co.exists():
    import math
    d = json.loads(co.read_text())
    fin = [r for r in d["rows"] if math.isfinite(r["origin"]) and math.isfinite(r["translated"])]
    check("closure is unchanged when the origin moves off the attractor",
          all(abs(r["origin"] - r["translated"]) < 1e-9 for r in fin) and len(fin) > 30,
          f"identical on {len(fin)} systems")
    check("closure holds without lifting to any fine state",
          d["frac_intrinsic"] > 0.9 and d["frac_translated"] > 0.9,
          f"intrinsic {d['frac_intrinsic']:.2f}, translated {d['frac_translated']:.2f}")

# --- the two interpretability experiments must ship, with their headline numbers ---------------
for f in ("xai_cka.py", "xai_sae.py"):
    check(f"the interpretability experiment ships: {f}", (ROOT / "mechanism" / f).exists())
xk = ROOT / "mechanism" / "xai_cka.json"
if xk.exists():
    d = json.loads(xk.read_text())
    check("X1: in-sample on 8 stimuli accepts every vacuous candidate",
          d["alpha_structural_insample_small"] == 1.0, f"alpha={d['alpha_structural_insample_small']}")
    check("X1: both guards refuse every vacuous candidate",
          d["alpha_both"] == 0.0 and d["rows"]["real"]["accepted"],
          f"alpha={d['alpha_both']}, real accepted={d['rows']['real']['accepted']}")
    check("X1: the uncentred causal guard preferred a vacuous candidate",
          d["rows"]["randtok"]["causal_raw"] > d["rows"]["real"]["causal_raw"],
          f"randtok {d['rows']['randtok']['causal_raw']:.2f} vs real {d['rows']['real']['causal_raw']:.2f}")
xs = ROOT / "mechanism" / "xai_sae.json"
if xs.exists():
    d = json.loads(xs.read_text())
    check("X2: matched controls make CAR interpretable",
          abs(d["car_selected"] - 0.46) < 0.02 and abs(d["car_control"] - 0.07) < 0.02,
          f"CAR selected {d['car_selected']:.2f} vs matched control {d['car_control']:.2f}")
    check("X2: delta-CAR is positive in every arm",
          d["delta_car"] > 0.3 and all(
              sum(r["supported"] for r in d["rows"] if r["arm"] == a and r["control"] is not None) >
              sum(r["control_supported"] for r in d["rows"] if r["arm"] == a and r["control"] is not None)
              for a in {r["arm"] for r in d["rows"]}),
          f"delta-CAR = {d['delta_car']:+.2f}")
    check("X2: enrichment guard has low NAR yet poor calibration",
          d["nar"] < 0.10 and d["car_selected"] < 0.60,
          f"NAR {d['nar']:.2f}, CAR {d['car_selected']:.2f}")
    check("X2: selection flow recorded, so 150-of-N is visible",
          all(f["N3"] > f["selected"] for f in d["flow"]),
          "; ".join(f"{f['N3']}->{f['selected']}" for f in d["flow"]))

# --- the guards must be shown sensitive, not only specific -------------------------------------
pos = ROOT / "mechanism" / "external_positives.json"
check("the positive-control suite ships", pos.exists() and
      (ROOT / "mechanism" / "external_positives.py").exists())
if pos.exists():
    d = json.loads(pos.read_text())

# --- the two interpretability experiments must ship, with their headline numbers ---------------
for f in ("xai_cka.py", "xai_sae.py"):
    check(f"the interpretability experiment ships: {f}", (ROOT / "mechanism" / f).exists())
xk = ROOT / "mechanism" / "xai_cka.json"
if xk.exists():
    d = json.loads(xk.read_text())
    check("X1: in-sample on 8 stimuli accepts every vacuous candidate",
          d["alpha_structural_insample_small"] == 1.0, f"alpha={d['alpha_structural_insample_small']}")
    check("X1: both guards refuse every vacuous candidate",
          d["alpha_both"] == 0.0 and d["rows"]["real"]["accepted"],
          f"alpha={d['alpha_both']}, real accepted={d['rows']['real']['accepted']}")
    check("X1: the uncentred causal guard preferred a vacuous candidate",
          d["rows"]["randtok"]["causal_raw"] > d["rows"]["real"]["causal_raw"],
          f"randtok {d['rows']['randtok']['causal_raw']:.2f} vs real {d['rows']['real']['causal_raw']:.2f}")

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
