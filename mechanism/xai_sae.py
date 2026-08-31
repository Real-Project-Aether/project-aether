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
MODEL    = "EleutherAI/pythia-160m"
LAYER    = 8
N_SEQ, SEQ_LEN = 1200, 128
EXPAND   = 8
TOPK     = 32          # L0, set directly
EPOCHS   = 8
N_FEAT   = 12          # real features examined
DEBUG    = False
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
def specificity(model, ids, layer, u, concept_mask, batch=16):
    """Ablate u. Compare the loss increase ON the concept's next tokens with everywhere else.

    A direction that carries the concept damages it selectively. A direction that merely
    correlates with it damages everything about equally, giving a ratio near 1.
    """
    on, off = [], []
    flat = concept_mask.reshape(-1).cpu()
    pos = 0
    for i in range(0, len(ids), batch):
        b = ids[i:i+batch].to(DEV)
        tgt = b[:, 1:].reshape(-1)
        lp_clean = torch.log_softmax(model(b).logits.float()[:, :-1], -1)
        with AblateDir(model, layer, u):
            lp_abl = torch.log_softmax(model(b).logits.float()[:, :-1], -1)
        d = (lp_clean.gather(-1, tgt[:, None, None].reshape(lp_clean.shape[0], -1, 1)).squeeze(-1)
             - lp_abl.gather(-1, tgt[:, None, None].reshape(lp_abl.shape[0], -1, 1)).squeeze(-1))
        d = d.reshape(-1).cpu()
        n = b.shape[0] * SEQ_LEN
        m = flat[pos:pos + n].reshape(b.shape[0], SEQ_LEN)[:, :-1].reshape(-1)
        pos += n
        on.append(d[m]); off.append(d[~m])
    on = torch.cat(on); off = torch.cat(off)
    if len(on) < 30:
        return float("nan"), float("nan"), float("nan")     # too few concept positions to judge
    a, o = float(on.mean()), float(off.mean())
    if abs(o) < 1e-8:                    # ablating one of 6144 directions is a small effect;
        return float("nan"), a, o        # only a truly zero denominator is disqualifying
    return a / o, a, o


# ------------------------------------------------------------------ main

def main():
    rng = np.random.default_rng(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = get_text(tok, N_SEQ, SEQ_LEN)
    n_fit = int(len(ids) * 0.8)
    ids_fit, ids_held = ids[:n_fit], ids[n_fit:]
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEV).eval()

    print("X2 -- does an SAE feature's meaning survive an intervention?")
    print(f"{MODEL} layer {LAYER}; SAE {EXPAND}x; {len(ids)} x {SEQ_LEN} tokens, "
          f"{len(ids)-n_fit} sequences held out\n")

    X = resid(model, ids_fit, LAYER)
    print(f"  training SAE on {tuple(X.shape)} activations")
    sae = train_sae(X, X.shape[1] * EXPAND)

    Xh = resid(model, ids_held, LAYER)
    toks_h = ids_held.reshape(-1).to(DEV)
    with torch.no_grad():
        _, Zh = sae(Xh)
    freq = (Zh > 0).float().mean(0)
    D = sae.dec.weight.detach()                       # (d, m) unit columns

    # Rank by how selective a feature is, restricted to firing rates where a feature can be
    # about something. Ranking by raw frequency picks the generic always-on features, whose top
    # token is just the corpus's most common token; that was the first version's mistake.
    # The lower bound is set by what the INTERVENTION can measure, not by what looks
    # interpretable: a feature whose concept appears 30 times in the held-out set gives no
    # usable estimate of on-concept damage. 2e-3 of 30720 positions is ~60 occurrences.
    ok = ((freq > 2e-3) & (freq < 5e-2)).nonzero().flatten().tolist()
    scored = []
    for i in ok:
        e, t, _ = selectivity(Zh[:, i], toks_h)
        scored.append((e, int(i)))
    scored.sort(reverse=True)
    reals = [i for _, i in scored[:N_FEAT]]
    nz = (freq > 0).nonzero().flatten()
    dead = [int(nz[int(freq[nz].argmin())])] if len(nz) else [0]
    print(f"  {len(ok)} features fire between 0.05% and 5% of the time; "
          f"examining the {len(reals)} most selective")

    def run_one(name, u, act, is_real):
        enr, ctok, mask = selectivity(act, toks_h)
        if ctok is None:
            return dict(name=name, enrichment=0.0, guard1=False, ratio=0.0,
                        guard2=False, accepted=False, real=is_real, token=None)
        g1 = enr > SELECT_MIN
        m_eval = mask[:N_EVAL * SEQ_LEN].cpu()
        if DEBUG:
            print(f"    [dbg] {name}: fired={int((act>0).sum())} mask_sum={int(m_eval.sum())} "
                  f"u.shape={tuple(u.shape)} u.norm={float(u.norm()):.3f}")
        ratio, on, off = specificity(model, ids_held[:N_EVAL], LAYER, u, m_eval)
        g2 = bool(ratio > SPECIFIC_MIN) if ratio == ratio else False
        return dict(name=name, enrichment=float(enr), guard1=bool(g1), ratio=float(ratio),
                    on=on, off=off, guard2=bool(g2), accepted=bool(g1 and g2), real=is_real,
                    token=tok.decode([ctok]))

    rows = []
    print(f"\n  {'candidate':<26}{'enrichment':>11}{'guard 1':>9}"
          f"{'specificity':>13}{'guard 2':>9}   verdict   top token")
    print("  " + "-" * 92)

    for j, fi in enumerate(reals):
        rows.append(run_one(f"SAE feature #{fi}", D[:, fi] / D[:, fi].norm(), Zh[:, fi], True))

    # ---- vacuity class
    f0 = reals[0]
    u0, a0 = D[:, f0] / D[:, f0].norm(), Zh[:, f0]
    r = torch.tensor(rng.normal(size=D.shape[0]), dtype=torch.float32, device=DEV)
    rows.append(run_one("CONTROL random direction", r / r.norm(), (Xh @ (r / r.norm())).clamp_min(0), False))
    perm = torch.tensor(rng.permutation(D.shape[0]), device=DEV)
    us = u0[perm]
    rows.append(run_one("CONTROL scrambled feature", us / us.norm(), (Xh @ (us / us.norm())).clamp_min(0), False))
    noise = torch.tensor(rng.normal(scale=0.6, size=D.shape[0]), dtype=torch.float32, device=DEV)
    usg = u0 + noise / noise.norm() * u0.norm() * 0.6
    uu = usg / usg.norm()
    rows.append(run_one("CONTROL correlated surrogate", uu, (Xh @ uu).clamp_min(0), False))
    fd = dead[0]
    rows.append(run_one("CONTROL near-dead feature", D[:, fd] / D[:, fd].norm(), Zh[:, fd], False))

    for x in rows:
        rs = 'undefined' if x['ratio'] != x['ratio'] else f"{x['ratio']:.2f}"
        print(f"  {x['name']:<26}{x['enrichment']:>11.1f}{'PASS' if x['guard1'] else 'fail':>9}"
              f"{rs:>13}{'PASS' if x['guard2'] else 'fail':>9}   "
              f"{'ACCEPT' if x['accepted'] else 'reject':<9} {str(x['token'])[:14]!r:<16}"
              f"on={x.get('on', float('nan')):+.2e} off={x.get('off', float('nan')):+.2e}")

    V = [x for x in rows if not x["real"]]
    R = [x for x in rows if x["real"]]
    a1 = sum(x["guard1"] for x in V) / len(V)
    a2 = sum(x["accepted"] for x in V) / len(V)
    print(f"\n  vacuity class V = {len(V)} controls")
    print(f"  alpha(guard 1 alone) = {a1:.2f}   <- P[correlational guard accepts something empty]")
    print(f"  alpha(both guards)   = {a2:.2f}")
    print(f"  real SAE features accepted: {sum(x['accepted'] for x in R)}/{len(R)}"
          f"   (guard 1 alone would accept {sum(x['guard1'] for x in R)}/{len(R)})")

    Path(__file__).with_name("xai_sae.json").write_text(json.dumps(dict(
        rows=rows, alpha_guard1=a1, alpha_both=a2,
        config=dict(model=MODEL, layer=LAYER, expand=EXPAND, epochs=EPOCHS,
                    n_seq=N_SEQ, seq_len=SEQ_LEN, select_min=SELECT_MIN,
                    specific_min=SPECIFIC_MIN)), indent=1))


if __name__ == "__main__":
    main()
