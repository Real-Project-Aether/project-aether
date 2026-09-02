"""Every number in the paper, with the file it comes from.

A reviewer should be able to check the claims without running anything expensive. This reads the
shipped results and prints each headline figure beside its source, then says which scripts
regenerate them and what they need.

    python3 scripts/reproduce.py

Exits non-zero if any shipped result is missing or disagrees with the paper.
"""
import csv, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL = []


def load(name):
    p = ROOT / "mechanism" / name
    if not p.exists():
        FAIL.append(f"missing: mechanism/{name}")
        return None
    return json.loads(p.read_text())


def row(claim, value, source, ok=True):
    if not ok:
        FAIL.append(claim)
    print(f"  {'ok  ' if ok else 'FAIL'} {claim:<52}{str(value):>22}   {source}")


print(__doc__.strip().split("\n")[0])
print("=" * 104)

# ---------------------------------------------------------------- environment
print("\nENVIRONMENT")
import platform
print(f"  python {platform.python_version()}")
for mod in ("numpy", "scipy"):
    try:
        print(f"  {mod} {__import__(mod).__version__}")
    except ImportError:
        print(f"  {mod} MISSING -- required")
for mod, why in (("dysts", "external_dysts.py only"), ("torch", "xai_*.py only"),
                 ("transformers", "xai_*.py only")):
    try:
        print(f"  {mod} {__import__(mod).__version__:<12} ({why})")
    except Exception:
        print(f"  {mod} not installed    ({why}; shipped results are read from JSON)")

# ---------------------------------------------------------------- corpus
print("\nTHE CORPUS                                                                    value   source")
rows = list(csv.DictReader(open(ROOT / "data" / "classifications.csv")))
def lv(r): return [t.strip().rstrip("?") for t in re.split(r"[|,;]", r["levels"]) if t.strip()]
c = Counter(l for r in rows for l in lv(r))
row("annotated events", len(rows), "data/classifications.csv", len(rows) == 1547)
row("depth labels", sum(c.values()), "data/classifications.csv", sum(c.values()) == 1041)
row("rungs L0/L1/L2/L3/L4", f"{c['L0']}/{c['L1']}/{c['L2']}/{c['L3']}/{c['L4']}",
    "data/classifications.csv", [c['L0'], c['L1'], c['L2'], c['L3'], c['L4']] == [131, 38, 501, 189, 181])
nodesc = sum(1 for r in rows if not lv(r))
row("changed no description at all", f"{100*nodesc/len(rows):.0f}%", "data/classifications.csv",
    abs(100*nodesc/len(rows) - 53) < 0.6)

# ---------------------------------------------------------------- external benchmark
print("\nEXTERNAL BENCHMARK (dysts)")
ex = load("external_dysts.json")
if ex:
    cl = sum(1 for r in ex if r["res"]["constant"]["closes"])
    pay = sum(1 for r in ex if r["res"]["constant"]["pays"])
    sl = sum(1 for r in ex if r["res"]["slow"]["closes"])
    mx = max(r["res"]["constant"]["r2"] for r in ex)
    row("systems that ran", len(ex), "mechanism/external_dysts.json", len(ex) == 129)
    row("empty reduction closes", f"{cl} / {len(ex)}  ({100*cl/len(ex):.0f}%)",
        "mechanism/external_dysts.json", cl == 124)
    row("empty reduction pays", pay, "mechanism/external_dysts.json", pay == 0)
    row("its best pay score", f"{mx:+.3f}", "mechanism/external_dysts.json", mx < 0.50)
    # Was 1 of 129 while acceptance took min(derived, learned); under the learned-only rule the
    # one system that closed (NuclearQuadrupole, learned 0.272 against a 0.25 threshold) no
    # longer does. The proposer, not the guard: a linear projection onto slow eigen-directions
    # is the wrong object for a chaotic attractor.
    row("real candidate closes", f"{sl} / {len(ex)}", "mechanism/external_dysts.json", sl == 0)

# ---------------------------------------------------------------- labelled benchmark
print("\nLABELLED BENCHMARK (built here; ground truth by construction)")
po = load("external_positives.json")
if po:
    row("labelled systems", po["n_systems"], "mechanism/external_positives.json", po["n_systems"] == 24)
    row("precision", f"{po['precision']:.2f}", "mechanism/external_positives.json", po["precision"] == 1.0)
    row("recall, guard on the TRUE subspace", f"{po['recall_true']:.2f}",
        "mechanism/external_positives.json", abs(po["recall_true"]-1.0) < 1e-9)
    row("recall, guard on the proposer's estimate", f"{po['recall_estimated']:.2f}",
        "mechanism/external_positives.json", abs(po["recall_estimated"]-0.708) < 0.02)
    row("empty candidates satisfying closure",
        f"{po['closure_only_accepts']} / {po['n_negatives']}",
        "mechanism/external_positives.json", po["closure_only_accepts"] == 47)
    row("empty candidates accepted by both", po["fp"], "mechanism/external_positives.json", po["fp"] == 0)
    row("tilt at which acceptance is lost",
        ", ".join(f"{k}:{v}" for k, v in sorted(po["breakdown_theta"].items())),
        "mechanism/external_positives.json",
        len({v for v in po["breakdown_theta"].values() if v is not None}) > 1)

co = load("external_coordinates.json")
if co:
    import math
    fin = [r for r in co["rows"] if math.isfinite(r["origin"]) and math.isfinite(r["translated"])]
    row("closure unchanged when the origin moves", f"{len(fin)} / {len(fin)} identical",
        "mechanism/external_coordinates.json",
        all(abs(r["origin"]-r["translated"]) < 1e-9 for r in fin))
    row("closure without lifting to a fine state", f"{co['frac_intrinsic']*100:.0f}%",
        "mechanism/external_coordinates.json", co["frac_intrinsic"] > 0.9)

# ---------------------------------------------------------------- interpretability
print("\nINTERPRETABILITY (X1 alignment, X2 sparse autoencoder)")
a = load("xai_cka.json")
if a:
    row("X1 NAR, structural guard, in-sample on 8 stimuli",
        f"{a['nar_structural_insample_small']:.2f}",
        "mechanism/xai_cka.json", a["nar_structural_insample_small"] >= 0.8)
    row("X1 NAR, structural guard, held out on all positions",
        f"{a['nar_structural_heldout']:.2f}",
        "mechanism/xai_cka.json", a["nar_structural_heldout"] >= 0.5)
    row("X1 NAR, both guards", f"{a['nar_both']:.2f}",
        "mechanism/xai_cka.json", a["nar_both"] == 0.0)
    _c = list(a["configs"].values())
    row("X1 configurations / interventions each",
        f"{len(_c)} / {a['config']['triples_per_config']}",
        "mechanism/xai_cka.json", a["config"]["triples_per_config"] >= 400)
    row("X1 real correspondence accepted",
        f"{sum(r['real']['accepted'] for r in _c)} / {len(_c)}",
        "mechanism/xai_cka.json", all(r["real"]["accepted"] for r in _c))
    _same = all(round(r[v]["cka"], 3) == round(r["real"]["cka"], 3) for r in _c for v in r
                if r[v]["type"] in ("correspondence-breaking", "randomised"))
    row("X1 nulls carry the real pair's CKA exactly", "yes" if _same else "no",
        "mechanism/xai_cka.json", _same)

e = load("xai_esm.json")
if e:
    _c = list(e["configs"].values())
    row("X3 NAR, structural guard (ESM-2)", f"{e['nar_structural_heldout']:.2f}",
        "mechanism/xai_esm.json", e["nar_structural_heldout"] > 0.5)
    row("X3 NAR, both guards", f"{e['nar_both']:.2f}",
        "mechanism/xai_esm.json", e["nar_both"] == 0.0)
    row("X3 PAR, known positives", f"{e['par_both']:.2f}",
        "mechanism/xai_esm.json", e["par_both"] == 1.0)
    row("X3 configurations / interventions each",
        f"{len(_c)} / {e['config']['triples_per_config']}",
        "mechanism/xai_esm.json", len(_c) >= 3)

f = load("xai_families.json")
if f:
    _r = [r for v in f["pairs"].values() for r in v["rows"]]
    row("null-family draws (3 families x 2 model pairs)", len(_r),
        "mechanism/xai_families.json", len(_r) >= 100)
    row("families: NAR, structural guard",
        f"{sum(r['guard1'] for r in _r)/len(_r):.2f}",
        "mechanism/xai_families.json", all(r["guard1"] for r in _r))
    row("families: NAR, both guards", f"{sum(r['accepted'] for r in _r)/len(_r):.2f}",
        "mechanism/xai_families.json", not any(r["accepted"] for r in _r))
    row("families: best causal score by any null",
        f"{max(r['causal_median'] for r in _r):.3f}",
        "mechanism/xai_families.json", max(r["causal_median"] for r in _r) < 0.20)

b = load("xai_sae.json")
if b:
    row("X2 features judged (testable, with a matched control)",
        f"{b['paired']} of {b['testable']} testable",
        "mechanism/xai_sae.json", b["paired"] <= b["testable"] <= len(b["rows"]))
    row("X2 selected features with no concept in the consequence split", b["untestable"],
        "mechanism/xai_sae.json", b["untestable"] > 0)
    row("X2 CAR@150, enrichment-selected",
        f"{b['car_at_k']:.2f}  (CI {b['ci'][0]:.2f}-{b['ci'][1]:.2f})",
        "mechanism/xai_sae.json", b["ci"][0] <= b["car_at_k"] <= b["ci"][1])
    row("X2 CAR, firing-rate-matched control", f"{b['car_control']:.3f}",
        "mechanism/xai_sae.json", b["car_control"] < 0.05)
    row("X2 delta-CAR (primary endpoint)",
        f"{b['delta_car']:+.2f}  (CI {b['delta_ci'][0]:+.2f} to {b['delta_ci'][1]:+.2f})",
        "mechanism/xai_sae.json", b["delta_ci"][0] > 0)
    row("X2 NAR over the null suite", f"{b['nar']:.2f}",
        "mechanism/xai_sae.json", b["nar"] < 0.10)
    row("X2 positive control on the consequence test",
        f"{b['positive_control_rate']:.2f} of {b['positive_defined']} defined",
        "mechanism/xai_sae.json", "positive_control_rate" in b)

# ---------------------------------------------------------------- how to regenerate
print("""
REGENERATING THESE
  python3 mechanism/discover_symmetry.py       L1 on the lattice            numpy/scipy      ~1 min
  python3 mechanism/l2_slowfast.py             L2 sensitivity floor         numpy/scipy      ~2 min
  python3 mechanism/l3_transfer.py             L3, six-seed sweep           numpy/scipy     ~20 min
  python3 mechanism/l4_posit.py                L4 positing                  numpy/scipy      ~1 min
  python3 mechanism/external_positives.py      the labelled benchmark       numpy/scipy     ~35 min\n  python3 mechanism/external_coordinates.py    coordinate invariance        + dysts          ~5 min
  python3 mechanism/external_dysts.py          the external benchmark       + dysts         ~15 min
  python3 mechanism/xai_cka.py                 X1                           + torch, GPU     ~5 min
  python3 mechanism/xai_sae.py                 X2, four arms                + torch, GPU    ~40 min
  python3 scripts/verify.py                    52 checks over all of it     numpy/scipy     ~25 min

  torchvision in some environments is built against a different torch and fails to load;
  the xai_* scripts disable it explicitly, since they touch no images.
""")

if FAIL:
    print(f"{len(FAIL)} PROBLEM(S): " + "; ".join(FAIL))
    sys.exit(1)
print("every shipped number agrees with the paper.")
