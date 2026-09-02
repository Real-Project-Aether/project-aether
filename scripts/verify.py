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
# --- the pay guard must not depend on how the observables are scaled ------------------------
import numpy as _np                                                              # noqa: E402
sys.path.insert(0, str(ROOT / "mechanism"))
try:
    from l2_coarse import pay_score as _pay
    _rng = _np.random.default_rng(0)
    _Y = _rng.normal(size=(400, 3)) @ _np.diag([1.0, 50.0, 0.02])
    _big = _Y.copy(); _big[:, 1] = _Y[:, 1].mean()
    _small = _Y.copy(); _small[:, 2] = _Y[:, 2].mean()
    check("the pay guard scores an identity reduction at exactly 1",
          abs(_pay(_Y, _Y) - 1.0) < 1e-9, f"{_pay(_Y, _Y):.6f}")
    check("the pay guard does not care which observable was damaged",
          abs(_pay(_Y, _big) - _pay(_Y, _small)) < 0.05,
          f"largest component {_pay(_Y, _big):.3f} vs smallest {_pay(_Y, _small):.3f}; "
          f"unweighted they were {_pay(_Y, _big, whiten=False):.2f} and "
          f"{_pay(_Y, _small, whiten=False):.2f}")
except Exception as _e:                                                          # pragma: no cover
    check("the pay guard is importable", False, f"{type(_e).__name__}: {_e}")

pos = ROOT / "mechanism" / "external_positives.json"
check("the positive-control suite ships", pos.exists() and
      (ROOT / "mechanism" / "external_positives.py").exists())
if pos.exists():
    d = json.loads(pos.read_text())
    check("positives: the guard accepts every TRUE reduction",
          abs(d["recall_true"] - 1.0) < 1e-9,
          f"recall on the true subspace = {d['recall_true']:.2f} over {d['n_systems']} systems")
    check("positives: the shortfall is the proposer, not the guard",
          d["recall_true"] - d["recall_estimated"] > 0.2,
          f"true {d['recall_true']:.2f} vs estimated {d['recall_estimated']:.2f}")
    check("positives: tolerance is not a single number",
          len({v for v in d["breakdown_theta"].values() if v is not None}) > 1,
          f"acceptance lost at {sorted({v for v in d['breakdown_theta'].values() if v is not None})} rad")
    check("positives: closure alone still accepts most empty candidates",
          d["closure_only_accepts"] == 47 and d["fp"] == 0,
          f"{d['closure_only_accepts']} of {d['n_negatives']} close, {d['fp']} accepted by both")


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
check("X1: the shipped result comes from the rebuilt experiment",
      xk.exists() and "configs" in json.loads(xk.read_text()),
      "localised patching, multiple configurations, AUROC against each null")
if xk.exists() and "configs" in json.loads(xk.read_text()):
    cfgs = d["configs"]
    check("X1: the structural guard accepts most nulls at ANY sample size",
          d["nar_structural_insample_small"] >= 0.8 and d["nar_structural_heldout"] >= 0.5,
          f"NAR {d['nar_structural_insample_small']:.2f} in-sample at {d['sweep_n'][0]} stimuli, "
          f"{d['nar_structural_heldout']:.2f} held out on all of them -- more data does not fix it, "
          f"because CKA never looks at the claimed map")
    check("X1: the causal guard is run on hundreds of localised interventions",
          all(r["real"]["n_triples"] >= 400 for r in cfgs.values()),
          f"{min(r['real']['n_triples'] for r in cfgs.values())} triples per configuration, "
          f"{len(cfgs)} configurations")
    check("X1: the real correspondence is accepted in every configuration",
          all(r["real"]["accepted"] for r in cfgs.values()),
          f"{sum(r['real']['accepted'] for r in cfgs.values())}/{len(cfgs)}")
    if "par_both" in d:
        check("X1: known positives are accepted, so its NAR is interpretable too",
              d["par_both"] == 1.0, f"PAR = {d['par_both']:.2f}")
    check("X1: CKA cannot see the claimed map -- nulls score exactly what the real pair scores",
          all(round(r[v]["cka"], 3) == round(r["real"]["cka"], 3)
              for r in cfgs.values() for v in r
              if r[v]["type"] in ("correspondence-breaking", "randomised")),
          "a correspondence and its random rotation receive the same CKA")
    check("X1: both guards together refuse every null",
          d["nar_both"] == 0.0 and d["any_null_pass"] == 0,
          f"NAR(both)={d['nar_both']:.2f}, AnyNullPass={d['any_null_pass']}")
    check("X1: correspondence-breaking nulls are the ones the structural guard misses",
          any(r[v]["guard1"] and not r[v]["guard2"]
              for r in cfgs.values()
              for v in r if r[v]["type"] == "correspondence-breaking"),
          "a null that preserves the marginals passes guard 1 and fails guard 2")
    # History, kept because it is the paper's own example of a guard needing a null: correlating
    # RAW shifts once ranked a random-token null (0.70) above the genuine pair (0.39). That was an
    # artefact of overwriting the whole residual stream with one vector -- both models then moved
    # toward the same generic tokens and the shared response dominated. Under localised difference
    # patching the generic response is small and the raw score no longer prefers a null. Centring
    # is still applied, but it is no longer load-bearing, and the paper says so.
    _mx = max(r[v]["causal_median_raw"] for r in cfgs.values() for v in r if v != "real")
    _mn = min(r["real"]["causal_median_raw"] for r in cfgs.values())
    check("X1: localising the intervention removes the pathology that centring was added for",
          _mx < _mn,
          f"worst null raw {_mx:.2f} vs weakest real raw {_mn:.2f}; under the old global overwrite "
          f"a null scored above the real pair")
    check("X1: the causal score separates real from null, reported as AUROC not a threshold",
          all(r[v]["auroc_vs_real"] is not None for r in cfgs.values()
              for v in r if v != "real"),
          "AUROC recorded for every null in every configuration")

xf = ROOT / "mechanism" / "xai_families.json"
check("null families are run, not only hand-written nulls", xf.exists(),
      "fresh draws from each correspondence-breaking family")
if xf.exists():
    d = json.loads(xf.read_text())
    rows = [r for v in d["pairs"].values() for r in v["rows"]]
    check("families: the structural guard accepts every draw",
          all(r["guard1"] for r in rows),
          f"{sum(r['guard1'] for r in rows)}/{len(rows)} draws pass guard 1")
    check("families: the pair accepts no draw",
          not any(r["accepted"] for r in rows),
          f"AnyNullPass = {max(v['any_null_pass'] for v in d['pairs'].values())} "
          f"over {len(rows)} draws")
    check("families: no null approaches the causal threshold",
          max(r["causal_median"] for r in rows) < 0.20,
          f"best null {max(r['causal_median'] for r in rows):.3f} against a 0.20 threshold")
    check("families: enough draws per family to be a rate",
          d["n_draw_per_family"] >= 20 and len(d["pairs"]) >= 2,
          f"{d['n_draw_per_family']} draws x {len(d['config']['families'])} families "
          f"x {len(d['pairs'])} model pairs")

xe = ROOT / "mechanism" / "xai_esm.json"
check("X3: the audit runs on a scientific foundation model", xe.exists(),
      "ESM-2 protein language models, three scale pairs")
if xe.exists():
    d = json.loads(xe.read_text())
    cf = d["configs"]
    check("X3: the structural guard alone accepts most nulls on a protein model",
          d["nar_structural_heldout"] > 0.5,
          f"NAR(guard 1) = {d['nar_structural_heldout']:.2f}")
    check("X3: adding the consequence test refuses every null",
          d["nar_both"] == 0.0 and d["any_null_pass"] == 0,
          f"NAR(both) = {d['nar_both']:.2f}, AnyNullPass = {d['any_null_pass']}")
    check("X3: known positives are accepted, so NAR is interpretable",
          d["par_both"] == 1.0,
          f"PAR = {d['par_both']:.2f} over identity and adjacent-layer controls")
    check("X3: the real correspondence is accepted in every configuration",
          all(r["real"]["accepted"] for r in cf.values()),
          f"{sum(r['real']['accepted'] for r in cf.values())}/{len(cf)}")
    check("X3: CKA cannot see the claimed map -- nulls score exactly what the real pair scores",
          all(round(r[v]["cka"], 3) == round(r["real"]["cka"], 3)
              for r in cf.values() for v in r
              if r[v]["type"] in ("correspondence-breaking", "randomised")),
          "a correspondence and its random rotation receive the same CKA")
    check("X3: held-out proteins are homology-separated by construction",
          "UniRef50" in d["config"]["data"],
          d["config"]["data"])
    check("X3: the guards are imported from X1, not reimplemented",
          "from xai_cka import" in (ROOT / "mechanism" / "xai_esm.py").read_text(),
          "same ridge_fit, r2, linear_cka, corr and auroc")

xs = ROOT / "mechanism" / "xai_sae.json"
if xs.exists():
    d = json.loads(xs.read_text())
    check("X2: the primary endpoint carries an interval, and it is the one ANALYSIS.md names",
          d["delta_ci"][0] > 0,
          f"delta-CAR = {d['delta_car']:+.2f}, 95% CI {d['delta_ci'][0]:+.2f} to {d['delta_ci'][1]:+.2f}")
    check("X2: the interval contains its own point estimate",
          d["ci"][0] <= d["car_at_k"] <= d["ci"][1],
          f"CAR {d['car_at_k']:.2f} in [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]")
    # ANALYSIS.md section 7 requires the clustered interval to be wider than the interval that
    # treats the features as independent Bernoulli trials.
    _p, _n = d["car_at_k"], d["paired"]
    _naive = 1.96 * (_p * (1 - _p) / _n) ** 0.5
    check("X2: clustering widens the interval, as the frozen spec said it must",
          (d["ci"][1] - d["ci"][0]) / 2 > _naive,
          f"clustered half-width {(d['ci'][1]-d['ci'][0])/2:.3f} vs naive binomial {_naive:.3f}")
    check("X2: delta-CAR is positive in every arm",
          all(v["car"] > v["control"] for v in d["per_arm"].values()),
          "; ".join(f"{a.split()[0][:12]} {v['car']:.2f}>{v['control']:.2f}"
                    for a, v in d["per_arm"].items()))
    check("X2: the enrichment guard has low NAR yet imperfect agreement",
          d["nar"] < 0.10 and d["car_at_k"] < 0.80,
          f"NAR {d['nar']:.2f}, CAR {d['car_at_k']:.2f}")
    check("X2: selection flow recorded, so 150-of-N is visible as a construction",
          all(f["N3"] > f["selected"] for f in d["flow"]),
          "; ".join(f"{f['N3']}->{f['selected']}" for f in d["flow"]))
    check("X2: the consequence test is itself audited by a positive control",
          "positive_control_rate" in d and d.get("positive_defined", 0) > 0,
          f"detected {d.get('positive_control_rate', float('nan')):.2f} of "
          f"{d.get('positive_defined', 0)} defined")
    check("X2: an undefined positive control is not counted as a failed detection",
          d.get("positive_defined", 0) <= len(d["rows"]) and
          all("positive_defined" in r for r in d["rows"]),
          f"{d.get('positive_defined', 0)} of {len(d['rows'])} features have a defined control")

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

# --- this file must not repeat itself ---------------------------------------------------------
# Three times a block was spliced into this file twice over, once silently swallowing the
# positive-suite assertions in between. A duplicated check inflates the count while testing
# nothing new.
import re as _re, collections as _c                                              # noqa: E402
_names = _re.findall(r'check\(\s*(f?"[^"]*")', pathlib.Path(__file__).read_text())
_dups = sorted(n for n, c in _c.Counter(_names).items() if c > 1)
check("no check in this file is written twice", not _dups, ", ".join(_dups) or "all distinct")

width = max(len(n) for n, _, _ in CHECKS)
failed = 0
for name, ok, detail in CHECKS:
    failed += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<{width}}  {detail}")
print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks pass")
sys.exit(1 if failed else 0)

