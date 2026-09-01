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
    row("real candidate closes", f"{sl} / {len(ex)}", "mechanism/external_dysts.json", sl == 1)

# ---------------------------------------------------------------- labelled benchmark
print("\nLABELLED BENCHMARK (built here; ground truth by construction)")
po = load("external_positives.json")
if po:
    row("labelled systems", po["n_systems"], "mechanism/external_positives.json", po["n_systems"] == 24)
    row("precision", f"{po['precision']:.2f}", "mechanism/external_positives.json", po["precision"] == 1.0)
    row("recall", f"{po['recall']:.2f}", "mechanism/external_positives.json", abs(po["recall"]-0.708) < 0.02)
    row("empty candidates satisfying closure",
        f"{po['closure_only_accepts']} / {po['n_negatives']}",
        "mechanism/external_positives.json", po["closure_only_accepts"] == 47)
    row("empty candidates accepted by both", po["fp"], "mechanism/external_positives.json", po["fp"] == 0)
    row("acceptance lost at perturbation",
        f"{sorted(set(po['breakdown_theta'].values()))[0]} rad",
        "mechanism/external_positives.json", set(po["breakdown_theta"].values()) == {0.05})

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
    row("X1 alpha, in-sample, 8 stimuli", f"{a['alpha_structural_insample_small']:.2f}",
        "mechanism/xai_cka.json", a["alpha_structural_insample_small"] == 1.0)
    row("X1 alpha, held out, 15360 positions", f"{a['alpha_structural_heldout']:.2f}",
        "mechanism/xai_cka.json", abs(a["alpha_structural_heldout"]-0.40) < 0.01)
    row("X1 alpha, both guards", f"{a['alpha_both']:.2f}", "mechanism/xai_cka.json", a["alpha_both"] == 0.0)
    row("uncentred guard ranked a null above the real pair",
        f"{a['rows']['randtok']['causal_raw']:.2f} vs {a['rows']['real']['causal_raw']:.2f}",
        "mechanism/xai_cka.json", a["rows"]["randtok"]["causal_raw"] > a["rows"]["real"]["causal_raw"])
b = load("xai_sae.json")
if b:
    row("X2 features paired with matched controls", b["paired"], "mechanism/xai_sae.json", b["paired"] == 600)
    row("X2 CAR, enrichment-selected", f"{b['car_selected']:.2f}  (CI {b['ci'][0]:.2f}-{b['ci'][1]:.2f})",
        "mechanism/xai_sae.json", abs(b["car_selected"]-0.46) < 0.02)
    row("X2 CAR, firing-rate-matched control", f"{b['car_control']:.2f}",
        "mechanism/xai_sae.json", abs(b["car_control"]-0.07) < 0.02)
    row("X2 delta-CAR (primary endpoint)", f"{b['delta_car']:+.2f}",
        "mechanism/xai_sae.json", b["delta_car"] > 0.3)
    row("X2 NAR over the null suite", f"{b['nar']:.2f}", "mechanism/xai_sae.json", b["nar"] < 0.10)

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
