"""Audit of SHAPE's semantic-space tracker with a typed null suite.

SHAPE (Song et al., 2026) labels a chain-of-thought trace with mathematical heuristics and tracks
which SEMANTIC SPACE the model is working in, then reports how reasoning effort is distributed over
those spaces. Its heuristic tagger is validated against an adjudicated reference set. Its
semantic-space tracker is not: the authors state that the spaces are trajectory-relative
interpretations for which no objective gold standard exists, so the tracker is calibrated by manual
review instead. By the distinction this paper draws, the tracker is a structural score. It is
computed on the trace it is describing and commits to nothing that could fail.

This runs the audit the tracker has not had. The guidebooks and the metric code are the authors'
own, vendored under shape_vendor/ with provenance.

The suite is typed, and includes a positive control, because a tracker that refuses everything
would score NAR 0 while being useless:

  real            a model's own solution to the problem it is shown
  POSITIVE        two independent solutions to the SAME problem, concatenated. There is a genuine
                  restart at the seam, so a tracker that reads interpretation structure should
                  register at least one more space here than in `real`.
  shuffled        NULL, degenerate: the same content units in random order. Every unit is real
                  mathematics; the sequence is not a solution to anything.
  mismatched      NULL, correspondence-breaking: a trace for a DIFFERENT problem, presented as a
                  solution to this one. Internally coherent, unrelated to the stated problem.
  spliced         NULL, randomised: content units drawn from several different traces. No single
                  interpretation of any problem is being pursued.

What the audit measures. The tracker cannot "reject" anything, so NAR is not the fraction it turns
away; it always emits a space sequence. The measurable question is whether its OUTPUT distinguishes
a real trace from one carrying no coherent interpretation. We report the AUROC of real against each
null using the tracker's own metrics. An AUROC near 0.5 means the metric cannot see the difference,
and a structural finding resting on it is resting on something that reads the same on a shuffled
trace. We report the positive control the same way: if real against POSITIVE is also near 0.5, the
annotator cannot do the task at this scale and the null result is uninformative rather than damning.

For `shuffled` and `spliced` the heuristic tags are carried over from the trace each unit came
from, unchanged. That isolates the tracker: the tags are identical, only their arrangement differs.
"""
import argparse, json, os, random, re, sys, warnings
from pathlib import Path
import numpy as np

VENDOR = Path(__file__).resolve().parent / "shape_vendor"
sys.path.insert(0, str(VENDOR))

import transformers.utils.import_utils as _iu          # torchvision shim, as elsewhere here
_iu.is_torchvision_available = lambda *a, **k: False
import transformers.utils as _tu
_tu.is_torchvision_available = _iu.is_torchvision_available

import torch                                                             # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM             # noqa: E402

warnings.filterwarnings("ignore")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ANNOTATOR = "Qwen/Qwen2.5-7B-Instruct"
N_PROBLEMS = 24
MAX_UNITS = 10                      # content units kept per trace, to bound the call budget
GEN_TOKENS = 480
ANN_TOKENS = 320

HEUR_GUIDE = (VENDOR / "heuristics_guide.md").read_text()
SPACE_GUIDE = (VENDOR / "semantic_space_guide.md").read_text()

# The repository guidebook names these as the necessary condition for a space transition.
TRIGGERS = ("H1", "H2", "H5", "H6", "H10", "H13")


# ------------------------------------------------------------------ model

class Chat:
    """Minimal stand-in for the OpenAI-shaped client their code expects."""

    def __init__(self, model_id):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=DEV).eval()

    @torch.no_grad()
    def __call__(self, prompt, max_new_tokens=ANN_TOKENS, temperature=0.0):
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt").to(DEV)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens,
                                  do_sample=temperature > 0, temperature=temperature or None,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def as_json(text, default):
    """Their annotators are asked for raw JSON; models wrap it. Recover the first object."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for cand in (t, *re.findall(r"\{.*?\}(?=\s*$)", t, re.S), *re.findall(r"\{.*\}", t, re.S)):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return default


# ------------------------------------------------------------------ stage 1: traces

def sentences(text):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()]
    return [x for x in s if len(x) > 3]


def content_units(text, max_units=MAX_UNITS):
    """Group sentences into units of two, then cap.

    Their pipeline segments with an LLM. We use a fixed rule instead and hold it constant across
    every condition, because the audit is of the TRACKER: any segmentation noise would otherwise
    differ between real and null traces for reasons unrelated to the tracker.
    """
    s = sentences(text)
    units = [" ".join(s[i:i+2]) for i in range(0, len(s), 2)]
    return units[:max_units]


# ------------------------------------------------------------------ stage 2: their prompts

def heuristic_prompt(problem, prev_units, unit):
    prev = "\n".join(f"Chunk {i}: {u[:200]}" for i, u in enumerate(prev_units[-4:])) \
        or "There are no previous chunks."
    return f"""In this project, we analyze the reasoning process of Large Reasoning Models (LRMs) by identifying heuristic strategies used during mathematical problem solving. Given one content unit from the model response, annotate it with heuristic or non-heuristic codes from the guidebook.

[Guidebook]
{HEUR_GUIDE}
[End of Guidebook]

[Math Problem]
{problem}
[End of Math Problem]

[Previous Context]
{prev}
[End of Previous Context]

[Current Chunk]
{unit}
[End of Current Chunk]

[Format]
Output JSON exactly in the following structure, and nothing else:
{{"annotations": [{{"code": "H4", "evidence": "quote", "reasoning": "why"}}]}}

Use sub-codes when they apply. Multi-tagging is allowed. Use N1-N4 only when no heuristic is present."""


def space_prompt(problem, memory, recent, unit, tags):
    mem = "\n".join(
        f"[ID: {m['id']}] Register: {m['register']} | Constraints: {m['constraints']} | "
        f"Core Tools: {m['core_tools']} | Summary: {m['summary']}" for m in memory)
    rec = "\n".join(f"Chunk {i}: {u[:200]}" for i, u in enumerate(recent[-10:])) or "(none)"
    return f"""You are an expert mathematical cognitive scientist analyzing the reasoning traces of a Large Reasoning Model. Your task is to track state changes in the Semantic Space.

[Guidebook]
{SPACE_GUIDE}
[End of Guidebook]

[Math Problem]
{problem}
[End of Math Problem]

[Memory of Past Spaces]
{mem}
[End of Memory of Past Spaces]

[Recent Context]
{rec}
[End of Recent Context]

[Current Chunk]
{unit}
Triggered Heuristics: {', '.join(tags)}
[End of Current Chunk]

[Format]
Reply with raw JSON only, no markdown fence.
For NEW: {{"decision": "NEW", "rationale": "...", "target_space_id": <int>, "new_space_definition": {{"register": "", "constraints": "", "core_tools": "", "summary": "", "anchor_text": ""}}}}
For RETURN: {{"decision": "RETURN", "rationale": "...", "target_space_id": <int>}}
For MAINTAIN: {{"decision": "MAINTAIN", "rationale": "...", "target_space_id": <int>}}"""


# ------------------------------------------------------------------ stage 3: their state machine

def track_spaces(chat, problem, units, tags_per_unit):
    """Algorithm 1 of the paper: a prompted state machine over a space memory buffer."""
    memory = [dict(id=0, register="Natural Language Context",
                   constraints="Original problem constraints", core_tools="None",
                   summary="Initial mathematical problem phrasing.")]
    cur, seq, decisions = 0, [], []
    for i, unit in enumerate(units):
        tags = tags_per_unit[i]
        trig = [t for t in tags if re.match(r"^H\d+", t)
                and re.match(r"^(H\d+)", t).group(1) in TRIGGERS]
        if trig:
            r = as_json(chat(space_prompt(problem, memory, units[:i], unit, trig)),
                        {"decision": "MAINTAIN"})
            d = str(r.get("decision", "MAINTAIN")).upper()
            if d == "NEW":
                nid = len(memory)
                nd = r.get("new_space_definition") or {}
                memory.append(dict(id=nid,
                                   register=str(nd.get("register", ""))[:80],
                                   constraints=str(nd.get("constraints", ""))[:80],
                                   core_tools=str(nd.get("core_tools", ""))[:80],
                                   summary=str(nd.get("summary", ""))[:120]))
                cur = nid
            elif d == "RETURN":
                try:
                    t = int(r.get("target_space_id", cur))
                    cur = t if 0 <= t < len(memory) else cur
                except Exception:
                    pass
            decisions.append(d)
        seq.append(cur)
    return seq, decisions


def shape_metrics(seq, tags_per_unit):
    """N_space_eff and N_trans_eff exactly as shape_vendor/shape_metrics.py defines them."""
    w = [sum(1 for t in tg if re.match(r"^H\d+", t)) for tg in tags_per_unit]
    tot = sum(w)
    if tot == 0:
        return dict(n_spaces=len(set(seq)), N_space_eff=float("nan"),
                    N_trans_eff=float("nan"), rho=float("nan"))
    q = {}
    for s, wi in zip(seq, w):
        q[s] = q.get(s, 0) + wi
    Ns = np.exp(-sum((v / tot) * np.log(v / tot) for v in q.values() if v))
    segs, prev = [], object()
    for s, wi in zip(seq, w):
        if s != prev:
            segs.append(0); prev = s
        segs[-1] += wi
    Nt = np.exp(-sum((v / tot) * np.log(v / tot) for v in segs if v)) - 1
    return dict(n_spaces=len(set(seq)), N_space_eff=float(Ns), N_trans_eff=float(Nt),
                rho=float(Nt / Ns) if Ns else float("nan"))


def auroc(pos, neg):
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    a = np.concatenate([p, n]); order = a.argsort()
    r = np.empty(len(a)); r[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


# ------------------------------------------------------------------ driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PROBLEMS)
    ap.add_argument("--annotator", default=ANNOTATOR)
    a = ap.parse_args()
    rng = random.Random(0)
    log = lambda m: (print(m, flush=True))

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    probs = [ds[i] for i in range(len(ds))][: a.n * 3]
    rng.shuffle(probs)
    probs = probs[: a.n]
    log(f"SHAPE tracker audit: {a.n} MATH-500 problems, annotator {a.annotator}")

    chat = Chat(a.annotator)

    # --- traces
    log("generating traces")
    traces, traces2 = [], []
    for k, p in enumerate(probs):
        q = (f"Solve this problem. Show your reasoning step by step.\n\n{p['problem']}")
        traces.append(chat(q, max_new_tokens=GEN_TOKENS, temperature=0.0))
        traces2.append(chat(q + "\n\nUse a different method than the most obvious one.",
                            max_new_tokens=GEN_TOKENS, temperature=0.8))
        if (k + 1) % 6 == 0:
            log(f"  {k+1}/{a.n}")

    units = [content_units(t) for t in traces]
    units2 = [content_units(t) for t in traces2]

    # --- heuristic tags on the real traces (reused by shuffled and spliced)
    log("tagging heuristics")
    tags = []
    for k, (p, us) in enumerate(zip(probs, units)):
        tg = []
        for i, u in enumerate(us):
            r = as_json(chat(heuristic_prompt(p["problem"], us[:i], u)), {"annotations": []})
            codes = [str(x.get("code", "")).strip() for x in r.get("annotations", [])
                     if isinstance(x, dict)]
            tg.append([c for c in codes if re.match(r"^[HN]\d+", c)] or ["N2"])
        tags.append(tg)
        if (k + 1) % 6 == 0:
            log(f"  {k+1}/{a.n}")

    tags2 = []
    for p, us in zip(probs, units2):
        tg = []
        for i, u in enumerate(us):
            r = as_json(chat(heuristic_prompt(p["problem"], us[:i], u)), {"annotations": []})
            codes = [str(x.get("code", "")).strip() for x in r.get("annotations", [])
                     if isinstance(x, dict)]
            tg.append([c for c in codes if re.match(r"^[HN]\d+", c)] or ["N2"])
        tags2.append(tg)

    # --- conditions
    log("building conditions and tracking spaces")
    rows = []
    for k, p in enumerate(probs):
        other = (k + 1) % a.n
        pool = [(u, t) for j in range(a.n) if j != k for u, t in zip(units[j], tags[j])]
        rng.shuffle(pool)
        sp = pool[: len(units[k])]
        idx = list(range(len(units[k]))); rng.shuffle(idx)

        conds = {
            "real":       (p["problem"], units[k], tags[k], "real"),
            "two_method": (p["problem"], units[k] + units2[k], tags[k] + tags2[k], "positive"),
            "shuffled":   (p["problem"], [units[k][i] for i in idx], [tags[k][i] for i in idx],
                           "degenerate"),
            "mismatched": (p["problem"], units[other], tags[other], "correspondence-breaking"),
            "spliced":    (p["problem"], [u for u, _ in sp], [t for _, t in sp], "randomised"),
        }
        for name, (prob, us, tg, typ) in conds.items():
            if not us:
                continue
            seq, dec = track_spaces(chat, prob, us, tg)
            m = shape_metrics(seq, tg)
            rows.append(dict(problem=k, condition=name, type=typ, n_units=len(us),
                             decisions=dec, seq=seq, **m))
        if (k + 1) % 4 == 0:
            log(f"  {k+1}/{a.n}")

    # --- report
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)
    real = by.get("real", [])
    print(f"\n{'condition':<14}{'type':<26}{'n':>4}{'spaces':>8}{'N_sp_eff':>10}"
          f"{'rho':>8}{'AUROC vs real':>15}")
    out = {}
    for name, rs in by.items():
        f = lambda k: [x[k] for x in rs if np.isfinite(x[k])]
        au = (float("nan") if name == "real"
              else auroc([x["N_space_eff"] for x in real], [x["N_space_eff"] for x in rs]))
        print(f"{name:<14}{rs[0]['type']:<26}{len(rs):>4}"
              f"{np.mean([x['n_spaces'] for x in rs]):>8.2f}"
              f"{np.mean(f('N_space_eff')):>10.2f}{np.mean(f('rho')):>8.2f}"
              f"{'  --' if name=='real' else f'{au:>15.3f}'}")
        out[name] = dict(type=rs[0]["type"], n=len(rs),
                         mean_n_spaces=float(np.mean([x["n_spaces"] for x in rs])),
                         mean_N_space_eff=float(np.mean(f("N_space_eff"))),
                         mean_rho=float(np.mean(f("rho"))), auroc_vs_real=au)

    pos = out.get("two_method", {}).get("auroc_vs_real", float("nan"))
    nulls = {k: v["auroc_vs_real"] for k, v in out.items()
             if v["type"] not in ("real", "positive")}
    print(f"\n  positive control (two_method vs real) AUROC = {pos:.3f}")
    print(f"  nulls vs real, AUROC: " + ", ".join(f"{k} {v:.3f}" for k, v in nulls.items()))
    valid = np.isfinite(pos) and abs(pos - 0.5) > 0.15
    print("  -> " + ("the annotator separates a construction with a known extra space from a real "
                     "trace, so a null result below is about the tracker" if valid else
                     "the annotator does NOT separate the positive control from a real trace, so "
                     "this audit is INCONCLUSIVE at this annotator scale, and the null results "
                     "below say nothing about the tracker"))

    Path(__file__).with_name("shape_audit.json").write_text(json.dumps(dict(
        rows=rows, summary=out, positive_auroc=pos, null_aurocs=nulls,
        conclusive=bool(valid),
        config=dict(annotator=a.annotator, n_problems=a.n, max_units=MAX_UNITS,
                    dataset="HuggingFaceH4/MATH-500", triggers=list(TRIGGERS),
                    guidebooks="vendored from holi-lab/SHAPE-of-CoT, see shape_vendor/PROVENANCE.md",
                    deviation="annotator is 7B via transformers, not Qwen3.5-27B via vLLM")),
        indent=1))
    print(f"\nwrote shape_audit.json")


if __name__ == "__main__":
    main()
