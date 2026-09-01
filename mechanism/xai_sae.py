"""X2 --- the guard thesis on a sparse autoencoder feature.

Review round 1, point 1, second half. X1 tested a correspondence between two models; this tests
the object mechanistic interpretability actually works with: a feature from an SAE trained on a
language model's residual stream, and the claim that it *means* something.

    guard 1, CORRELATIONAL -- the one in common use. A feature looks interpretable if its
             activation concentrates on an identifiable class of tokens. We score selectivity:
             how much of the feature's activation mass lands on its own top token type, against
             how often that type occurs at all.

    guard 2, INTERVENTIONAL -- ablate the feature and require the damage to be SPECIFIC. If the
             feature carries the concept, removing it should hurt prediction on that concept far
             more than it hurts prediction everywhere else. A direction that merely correlates
             has no reason to.

Vacuity class V, none of which carries a concept, all of which should be refused:
    random      a random unit direction in the same residual space
    scrambled   a real feature's decoder vector with its coordinates permuted
    surrogate   a real feature plus noise, retaining most of its correlational profile
    deadfeature an SAE feature that almost never fires

Everything is scored on held-out tokens.


Requires torch, transformers and datasets, and a GPU for comfort. That is why verify.py
does not run it; its results ship alongside as JSON.
"""
import json, warnings
from pathlib import Path
import numpy as np

import transformers.utils.import_utils as _iu
_iu.is_torchvision_available = lambda *a, **k: False
import transformers.utils as _tu
_tu.is_torchvision_available = _iu.is_torchvision_available

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM    # noqa: E402

warnings.filterwarnings("ignore")
torch.manual_seed(0)

DEV      = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = [("EleutherAI/pythia-160m",         4),
        ("EleutherAI/pythia-160m",         8),
        ("EleutherAI/pythia-410m-deduped", 8),
        ("EleutherAI/pythia-410m-deduped", 16)]
N_SEQ, SEQ_LEN = 1200, 128
EXPAND   = 8
TOPK     = 32          # L0, set directly
EPOCHS   = 8
N_FEAT   = 150         # real features examined per arm
N_EVAL   = 240
MIN_ON   = 30          # concept positions required before a feature is judged
MIN_CONCEPT = 60       # a concept must occur this often in held-out text to be testable         # held-out sequences used for the intervention (all of them)

SELECT_MIN   = 4.0     # guard 1: enrichment over base rate
SPECIFIC_MIN = 2.0     # guard 2: concept damage must be 2x the off-concept damage


# ------------------------------------------------------------------ data

def get_text(tok, n_seq, seq_len):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    buf, out = [], []
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200:
            continue
        buf.extend(tok(t).input_ids)
        while len(buf) >= seq_len and len(out) < n_seq:
            out.append(buf[:seq_len]); buf = buf[seq_len:]
        if len(out) >= n_seq:
            break
    return torch.tensor(out)


@torch.no_grad()
def resid(model, ids, layer, batch=32):
    out = []
    for i in range(0, len(ids), batch):
        h = model(ids[i:i+batch].to(DEV), output_hidden_states=True).hidden_states[layer]
        out.append(h.reshape(-1, h.shape[-1]).float())
    return torch.cat(out)


# ------------------------------------------------------------------ the SAE

class SAE(nn.Module):
    """TopK sparse autoencoder.

    An L1 penalty was tried first and would not sparsify: at the coefficient that left
    reconstruction usable the equilibrium sat at L0 ~ 565 of 6144, and the most frequently firing
    features were generic ones whose top token was simply the most common token in the corpus.
    TopK sets L0 by construction, which is what the experiment needs.
    """
    def __init__(self, d, m, k):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d, bias=False)
        with torch.no_grad():
            w = torch.randn(d, m); w /= w.norm(dim=0, keepdim=True)
            self.dec.weight.copy_(w); self.enc.weight.copy_(w.T)
        self.b_pre = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        pre = self.enc(x - self.b_pre)
        val, idx = torch.topk(pre, self.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, idx, torch.relu(val))
        return self.dec(z) + self.b_pre, z


def train_sae(X, m, k=TOPK, epochs=EPOCHS, bs=4096, lr=1e-3):
    d = X.shape[1]
    sae = SAE(d, m, k).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    var = float(((X - X.mean(0)) ** 2).sum(-1).mean())
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=X.device)
        tot, nb = 0.0, 0
        for i in range(0, len(X), bs):
            xb = X[perm[i:i+bs]]
            xh, _ = sae(xb)
            rec = ((xh - xb) ** 2).sum(-1).mean()
            opt.zero_grad(); rec.backward(); opt.step()
            with torch.no_grad():
                sae.dec.weight.div_(sae.dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))
            tot += float(rec); nb += 1
        print(f"    epoch {ep+1}/{epochs}  recon {tot/nb:8.3f}  "
              f"({100*(1-tot/nb/var):5.1f}% of variance)  L0 = {k}")
    return sae.eval()


# ------------------------------------------------------------------ guard 1: correlational

def selectivity(act, toks, min_concept=MIN_CONCEPT):
    """Enrichment of the feature's firings on its own top token type, as a likelihood ratio

        P(token = t | feature fires) / P(token = t).

    A caution learned the hard way. Ranking candidate tokens by this ratio alone is degenerate:
    when the feature captures every occurrence of t, the ratio collapses to N / (times fired),
    which is independent of how many occurrences there were. The twelve "most selective" features
    it first returned had top tokens occurring 3 to 29 times in the held-out set, too few to
    intervene on and too few to mean anything. The candidate concept must therefore be frequent
    enough to measure, which is what `min_concept` enforces.

    Returns (enrichment, concept token id, mask of concept positions).
    """
    fired = act > 0
    if fired.sum() < 40:
        return 0.0, None, None
    t = toks[fired]
    ids, counts = torch.unique(t, return_counts=True)
    order = torch.argsort(counts, descending=True)
    for j in order.tolist():
        cand = ids[j]
        mask = (toks == cand)
        if int(mask.sum()) < min_concept:
            continue                        # concept too rare to test; try the next token type
        share = float(counts[j]) / float(fired.sum())
        base = float(mask.float().mean())
        return share / max(base, 1e-9), int(cand), mask
    return 0.0, None, None


# ------------------------------------------------------------------ guard 2: interventional

class AblateDir:
    def __init__(self, model, layer, u):
        self.block = model.gpt_neox.layers[layer - 1]; self.u = u.to(DEV)
    def __enter__(self):
        u = self.u
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h @ u).unsqueeze(-1) * u
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        self.h = self.block.register_forward_hook(hook); return self
    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def clean_logprobs(model, ids, batch=48):
    """Log-probability the model assigns to each actual next token, with nothing ablated.

    Identical for every feature in an arm, so it is computed once rather than once per feature.
    That halves the forward passes, which is what makes hundreds of features affordable.
    """
    out = []
    for i in range(0, len(ids), batch):
        b = ids[i:i+batch].to(DEV)
        tgt = b[:, 1:]
        lp = torch.log_softmax(model(b).logits.float()[:, :-1], -1)
        out.append(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).reshape(-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def specificity(model, ids, layer, u, concept_mask, clean, batch=48):
    """Ablate u; contrast the loss increase on the concept's next tokens with everywhere else.

    Returns (S, dLC, dLnC, ratio). S is the standardised difference

        S = (dLC - dLnC) / SE[dLC - dLnC],   SE = sqrt(var_on/n_on + var_off/n_off),

    which ANALYSIS.md fixes as the primary statistic. The ratio dLC/dLnC used in an earlier run is
    still returned for comparability, but it is not thresholded on: it explodes when the
    denominator is near zero, and did -- one feature scored 1644.
    """
    abl = []
    with AblateDir(model, layer, u):
        for i in range(0, len(ids), batch):
            b = ids[i:i+batch].to(DEV)
            lp = torch.log_softmax(model(b).logits.float()[:, :-1], -1)
            abl.append(lp.gather(-1, b[:, 1:].unsqueeze(-1)).squeeze(-1).reshape(-1).cpu())
    d = clean - torch.cat(abl)
    m = concept_mask.reshape(-1, SEQ_LEN)[:, :-1].reshape(-1).cpu()
    on, off = d[m], d[~m]
    if len(on) < MIN_ON:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a, o = float(on.mean()), float(off.mean())
    se = float((on.var(unbiased=True)/len(on) + off.var(unbiased=True)/len(off)).sqrt())
    S = (a - o) / max(se, 1e-12)
    ratio = a / o if abs(o) > 1e-8 else float("nan")
    return S, a, o, ratio


def matched_control(f, freq, tops, selected, rng):
    """A control feature from the same arm, matched on firing rate, with a different concept.

    ANALYSIS.md: firing rate within +/-20%, top token different, not itself selected. Decoder-norm
    matching is deliberately not applied -- our decoder columns are renormalised to unit norm every
    step, so all features already share it and the match would be vacuous.
    """
    lo, hi = 0.8 * float(freq[f]), 1.2 * float(freq[f])
    pool = [g for g in range(len(freq))
            if lo <= float(freq[g]) <= hi and g not in selected and tops.get(g) not in (None, tops.get(f))]
    return int(rng.choice(pool)) if pool else None


# ------------------------------------------------------------------ main

def wilson(k, n, z=1.96):
    """Wilson score interval -- honest at the small n we would otherwise be quoting bare."""
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z*z/n
    c = (ph + z*z/(2*n)) / d
    h = z*((ph*(1-ph)/n + z*z/(4*n*n)) ** 0.5) / d
    return (max(0.0, c-h), min(1.0, c+h))


def run_arm(model_name, layer, rng):
    """Train an SAE on one (model, layer), score its most selective features and matched controls."""
    tok = AutoTokenizer.from_pretrained(model_name)
    ids = get_text(tok, N_SEQ, SEQ_LEN)
    n_fit = int(len(ids) * 0.8)
    ids_fit, ids_held = ids[:n_fit], ids[n_fit:]
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEV).eval()

    X = resid(model, ids_fit, layer)
    print(f"  training SAE on {tuple(X.shape)}")
    sae = train_sae(X, X.shape[1] * EXPAND)
    Xh = resid(model, ids_held, layer)
    toks_h = ids_held.reshape(-1).to(DEV)
    with torch.no_grad():
        _, Zh = sae(Xh)
    freq = (Zh > 0).float().mean(0)
    D = sae.dec.weight.detach()
    arm = f"{model_name.split('/')[-1]} L{layer}"

    # ---- selection flow, recorded so that "all selected features clear the threshold" is
    #      visibly a construction rather than a result (ANALYSIS.md section 4)
    N0 = D.shape[1]
    in_range = ((freq > 2e-3) & (freq < 5e-2)).nonzero().flatten().tolist()
    N1 = len(in_range)
    scored, tops = [], {}
    for i in in_range:
        e, t, _ = selectivity(Zh[:, i], toks_h)
        if t is not None:
            scored.append((e, int(i))); tops[int(i)] = t
    N2 = len(scored)
    scored = [(e, i) for e, i in scored if e > SELECT_MIN]
    N3 = len(scored)
    scored.sort(reverse=True)
    reals = [i for _, i in scored[:N_FEAT]]
    flow = dict(arm=arm, N0=N0, N1=N1, N2=N2, N3=N3, selected=len(reals))
    print(f"  flow: all {N0} -> firing range {N1} -> testable concept {N2} -> enrichment>4 {N3} "
          f"-> selected {len(reals)}")

    clean = clean_logprobs(model, ids_held[:N_EVAL])
    sel = set(reals)

    def score_on(u, mask):
        return specificity(model, ids_held[:N_EVAL], layer, u, mask[:N_EVAL * SEQ_LEN], clean)

    rows, n_nocontrol = [], 0
    for f in reals:
        enr, ctok, mask = selectivity(Zh[:, f], toks_h)
        if ctok is None:
            continue
        S, a, o, ratio = score_on(D[:, f] / D[:, f].norm(), mask)
        g = matched_control(f, freq, tops, sel, rng)
        if g is None:
            n_nocontrol += 1
            Sg = float("nan")
        else:
            Sg, _, _, _ = score_on(D[:, g] / D[:, g].norm(), mask)   # control, on f's concept
        rows.append(dict(arm=arm, feature=int(f), enrichment=float(enr), token=tok.decode([ctok]),
                         S=float(S), dLC=float(a), dLnC=float(o), ratio=float(ratio),
                         control=(None if g is None else int(g)), S_control=float(Sg),
                         supported=bool(S == S and S > 1.96 and a > 0),
                         control_supported=bool(Sg == Sg and Sg > 1.96)))
    if n_nocontrol:
        print(f"  {n_nocontrol} feature(s) had no admissible matched control and are excluded from the pair")

    # ---- the null suite, unchanged
    f0 = reals[0]; u0 = D[:, f0] / D[:, f0].norm()
    rnd = torch.tensor(rng.normal(size=D.shape[0]), dtype=torch.float32, device=DEV)
    perm = torch.tensor(rng.permutation(D.shape[0]), device=DEV); us = u0[perm]
    noise = torch.tensor(rng.normal(scale=0.6, size=D.shape[0]), dtype=torch.float32, device=DEV)
    usg = u0 + noise / noise.norm() * u0.norm() * 0.6
    nz = (freq > 0).nonzero().flatten()
    fd = int(nz[int(freq[nz].argmin())]) if len(nz) else 0
    nulls = []
    for nm, u, act, typ in (("random direction", rnd/rnd.norm(), (Xh @ (rnd/rnd.norm())).clamp_min(0), "randomised"),
                            ("scrambled feature", us/us.norm(), (Xh @ (us/us.norm())).clamp_min(0), "randomised"),
                            ("correlated surrogate", usg/usg.norm(), (Xh @ (usg/usg.norm())).clamp_min(0), "structure-preserving"),
                            ("near-dead feature", D[:, fd]/D[:, fd].norm(), Zh[:, fd], "degenerate")):
        enr, ctok, mask = selectivity(act, toks_h)
        if ctok is None:
            nulls.append(dict(arm=arm, name=nm, type=typ, enrichment=0.0, guard1=False)); continue
        nulls.append(dict(arm=arm, name=nm, type=typ, enrichment=float(enr), guard1=bool(enr > SELECT_MIN)))

    del model, sae, X, Xh, Zh
    torch.cuda.empty_cache()
    return rows, nulls, flow


def hierarchical_bootstrap(rows, n=1000, seed=0):
    """Resample arms, then features within arms, clustering features that share a top token."""
    rng = np.random.default_rng(seed)
    arms = sorted({r["arm"] for r in rows})
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in arms}
    out = []
    for _ in range(n):
        picked = rng.choice(arms, size=len(arms), replace=True)
        sup = tot = 0
        for a in picked:
            rs = by_arm[a]
            clusters = {}
            for r in rs:
                clusters.setdefault(r["token"], []).append(r)
            keys = list(clusters)
            for k in rng.choice(keys, size=len(keys), replace=True):
                for r in clusters[k]:
                    tot += 1; sup += r["supported"]
        if tot:
            out.append(sup / tot)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    rng = np.random.default_rng(0)
    print("X2 -- does an SAE feature's meaning survive an intervention a matched control does not?")
    print(f"{len(ARMS)} arms, up to {N_FEAT} features each; primary endpoint is CAR(selected) - "
          f"CAR(matched), per ANALYSIS.md\n")

    rows, nulls, flows = [], [], []
    for model_name, layer in ARMS:
        print(f"[{model_name} layer {layer}]")
        r, n, f = run_arm(model_name, layer, rng)
        rows += r; nulls += n; flows.append(f)
        print()

    print(f"{'stage':<34}" + "".join(f"{f['arm'][:16]:>18}" for f in flows))
    for k, lbl in (("N0", "all SAE features"), ("N1", "firing rate in range"),
                   ("N2", "testable concept"), ("N3", "enrichment > 4"),
                   ("selected", "selected for intervention")):
        print(f"  {lbl:<32}" + "".join(f"{f[k]:>18}" for f in flows))
    print("  (all selected features clear the enrichment threshold by construction; that is not a result)\n")

    paired = [r for r in rows if r["control"] is not None]
    car_sel = sum(r["supported"] for r in paired) / max(len(paired), 1)
    car_ctl = sum(r["control_supported"] for r in paired) / max(len(paired), 1)
    lo, hi = hierarchical_bootstrap(paired)

    print(f"{'arm':<26}{'paired':>8}{'CAR sel':>10}{'CAR ctl':>10}{'delta':>9}")
    print("-" * 63)
    for a in sorted({r["arm"] for r in paired}):
        p_ = [r for r in paired if r["arm"] == a]
        cs = sum(r["supported"] for r in p_) / len(p_)
        cc = sum(r["control_supported"] for r in p_) / len(p_)
        print(f"{a:<26}{len(p_):>8}{cs:>10.2f}{cc:>10.2f}{cs-cc:>9.2f}")
    print("-" * 63)
    print(f"{'pooled':<26}{len(paired):>8}{car_sel:>10.2f}{car_ctl:>10.2f}{car_sel-car_ctl:>9.2f}")
    print(f"\n  CAR(selected) = {car_sel:.2f}   hierarchical 95% CI {lo:.2f}-{hi:.2f}"
          f"   (clustered by arm and shared top token)")
    print(f"  CAR(matched control) = {car_ctl:.2f}")
    print(f"  PRIMARY ENDPOINT  delta-CAR = {car_sel-car_ctl:+.2f}")

    nm = list(dict.fromkeys(x["name"] for x in nulls))
    nar = sum(x["guard1"] for x in nulls) / max(len(nulls), 1)
    print(f"\n  null suite, per control:")
    for n_ in nm:
        c = [x for x in nulls if x["name"] == n_]
        print(f"    {n_:<24}{c[0]['type']:<22}{sum(x['guard1'] for x in c)}/{len(c)} accepted")
    print(f"  NAR = {nar:.2f}   AnyNullPass = {int(any(x['guard1'] for x in nulls))}")

    Path(__file__).with_name("xai_sae.json").write_text(json.dumps(dict(
        rows=rows, nulls=nulls, flow=flows, paired=len(paired),
        car_selected=car_sel, car_control=car_ctl, delta_car=car_sel-car_ctl, ci=[lo, hi],
        nar=nar, any_null_pass=int(any(x["guard1"] for x in nulls)),
        config=dict(arms=[list(a) for a in ARMS], expand=EXPAND, topk=TOPK, epochs=EPOCHS,
                    n_seq=N_SEQ, seq_len=SEQ_LEN, n_feat=N_FEAT, n_eval=N_EVAL,
                    select_min=SELECT_MIN, min_concept=MIN_CONCEPT, min_on=MIN_ON,
                    support_rule="S > 1.96 and dLC > 0")), indent=1))


if __name__ == "__main__":
    main()
