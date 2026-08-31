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
    """Ablate u. Compare the loss increase ON the concept's next tokens with everywhere else.

    A direction carrying the concept damages it selectively; one that merely correlates damages
    everything about equally, giving a ratio near 1.
    """
    abl = []
    with AblateDir(model, layer, u):
        for i in range(0, len(ids), batch):
            b = ids[i:i+batch].to(DEV)
            tgt = b[:, 1:]
            lp = torch.log_softmax(model(b).logits.float()[:, :-1], -1)
            abl.append(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).reshape(-1).cpu())
    d = clean - torch.cat(abl)

    m = concept_mask.reshape(-1, SEQ_LEN)[:, :-1].reshape(-1).cpu()
    on, off = d[m], d[~m]
    if len(on) < 30:
        return float("nan"), float("nan"), float("nan")
    a, o = float(on.mean()), float(off.mean())
    if abs(o) < 1e-8:
        return float("nan"), a, o
    return a / o, a, o


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
    """Train an SAE on one (model, layer) and score its most selective features."""
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

    ok = ((freq > 2e-3) & (freq < 5e-2)).nonzero().flatten().tolist()
    scored = []
    for i in ok:
        e, t, _ = selectivity(Zh[:, i], toks_h)
        if t is not None:
            scored.append((e, int(i)))
    scored.sort(reverse=True)
    reals = [i for _, i in scored[:N_FEAT]]
    print(f"  {len(ok)} features fire in range; scoring the {len(reals)} most selective")

    clean = clean_logprobs(model, ids_held[:N_EVAL])

    def one(name, u, act, is_real):
        enr, ctok, mask = selectivity(act, toks_h)
        if ctok is None:
            return None
        g1 = enr > SELECT_MIN
        ratio, on, off = specificity(model, ids_held[:N_EVAL], layer, u,
                                     mask[:N_EVAL * SEQ_LEN], clean)
        g2 = bool(ratio > SPECIFIC_MIN) if ratio == ratio else False
        return dict(arm=f"{model_name.split('/')[-1]} L{layer}", name=name,
                    enrichment=float(enr), guard1=bool(g1), ratio=float(ratio),
                    guard2=g2, accepted=bool(g1 and g2), real=is_real,
                    token=tok.decode([ctok]))

    rows = []
    for fi in reals:
        r = one(f"feature #{fi}", D[:, fi] / D[:, fi].norm(), Zh[:, fi], True)
        if r: rows.append(r)

    f0 = reals[0]; u0 = D[:, f0] / D[:, f0].norm()
    rnd = torch.tensor(rng.normal(size=D.shape[0]), dtype=torch.float32, device=DEV)
    perm = torch.tensor(rng.permutation(D.shape[0]), device=DEV)
    us = u0[perm]
    noise = torch.tensor(rng.normal(scale=0.6, size=D.shape[0]), dtype=torch.float32, device=DEV)
    usg = u0 + noise / noise.norm() * u0.norm() * 0.6
    nz = (freq > 0).nonzero().flatten()
    fd = int(nz[int(freq[nz].argmin())]) if len(nz) else 0
    for nm, u, act in (("CONTROL random direction", rnd/rnd.norm(), (Xh @ (rnd/rnd.norm())).clamp_min(0)),
                       ("CONTROL scrambled feature", us/us.norm(), (Xh @ (us/us.norm())).clamp_min(0)),
                       ("CONTROL correlated surrogate", usg/usg.norm(), (Xh @ (usg/usg.norm())).clamp_min(0)),
                       ("CONTROL near-dead feature", D[:, fd]/D[:, fd].norm(), Zh[:, fd])):
        r = one(nm, u, act, False)
        if r: rows.append(r)

    del model, sae, X, Xh, Zh
    torch.cuda.empty_cache()
    return rows


def main():
    rng = np.random.default_rng(0)
    print("X2 -- does an SAE feature's meaning survive an intervention?")
    print(f"{len(ARMS)} arms, up to {N_FEAT} features each, everything scored on held-out tokens\n")

    rows = []
    for model_name, layer in ARMS:
        print(f"[{model_name} layer {layer}]")
        rows.extend(run_arm(model_name, layer, rng))
        print()

    R = [x for x in rows if x["real"]]
    V = [x for x in rows if not x["real"]]
    g1 = sum(x["guard1"] for x in R); both = sum(x["accepted"] for x in R)
    lo, hi = wilson(both, g1)

    print(f"{'arm':<26}{'features':>10}{'guard 1':>10}{'both':>8}{'rate':>9}")
    print("-" * 64)
    for arm in dict.fromkeys(x["arm"] for x in R):
        a = [x for x in R if x["arm"] == arm]
        a1 = sum(x["guard1"] for x in a); ab = sum(x["accepted"] for x in a)
        print(f"{arm:<26}{len(a):>10}{a1:>10}{ab:>8}{(ab/max(a1,1)):>9.2f}")
    print("-" * 64)
    print(f"{'pooled':<26}{len(R):>10}{g1:>10}{both:>8}{(both/max(g1,1)):>9.2f}")
    print(f"\nOf the {g1} features the correlational guard accepts, the intervention confirms "
          f"{both} --- {100*both/max(g1,1):.0f}% (95% CI {100*lo:.0f}--{100*hi:.0f}%).")

    print("\nper-control alpha, so the reader can assemble any vacuity class they like")
    print(f"  {'control':<30}{'n':>5}{'guard 1 accepts':>18}{'alpha':>8}")
    for nm in dict.fromkeys(x["name"] for x in V):
        c = [x for x in V if x["name"] == nm]
        acc = sum(x["guard1"] for x in c)
        print(f"  {nm:<30}{len(c):>5}{acc:>18}{acc/max(len(c),1):>8.2f}")
    a_sup = max((sum(x["guard1"] for x in V if x["name"] == nm) / max(len([y for y in V if y["name"]==nm]),1))
                for nm in dict.fromkeys(x["name"] for x in V)) if V else 0.0
    print(f"  {'-> alpha = sup over the class':<30}{'':>5}{'':>18}{a_sup:>8.2f}")

    print("\nthreshold sensitivity of the confirmation rate")
    rr = np.array([x["ratio"] for x in R if x["guard1"]])
    rr = rr[~np.isnan(rr)]
    print(f"  {'SPECIFIC_MIN':>13}{'confirmed':>12}{'rate':>8}")
    sweep = {}
    for t in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        k = int((rr > t).sum()); sweep[t] = k
        print(f"  {t:>13.1f}{k:>12}{k/max(len(rr),1):>8.2f}")

    Path(__file__).with_name("xai_sae.json").write_text(json.dumps(dict(
        rows=rows, guard1=g1, both=both, rate=both/max(g1,1), ci=[lo, hi],
        alpha_per_control={nm: sum(x["guard1"] for x in V if x["name"]==nm) /
                               max(len([y for y in V if y["name"]==nm]),1)
                           for nm in dict.fromkeys(x["name"] for x in V)},
        alpha_sup=a_sup, threshold_sweep={str(k): v for k, v in sweep.items()},
        config=dict(arms=[list(a) for a in ARMS], expand=EXPAND, topk=TOPK, epochs=EPOCHS,
                    n_seq=N_SEQ, seq_len=SEQ_LEN, n_feat=N_FEAT, n_eval=N_EVAL,
                    select_min=SELECT_MIN, specific_min=SPECIFIC_MIN,
                    min_concept=MIN_CONCEPT)), indent=1))


if __name__ == "__main__":
    main()
