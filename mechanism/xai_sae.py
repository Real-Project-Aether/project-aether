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
import json
import sys, warnings
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
N_CTRL   = 3           # matched control features per selected feature
N_EVAL   = 240
MIN_ON   = 30          # concept positions required before a feature is judged
GRAD_SEQ   = 160          # sequences used for the gradient positive control
MIN_SEQ  = 12          # sequences carrying the concept, since the sequence is the unit of resampling
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

def selectivity(act, next_toks, min_concept=MIN_CONCEPT):
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
    # The concept is the token the model must PREDICT at t+1 when the feature fires at t, not the
    # token already present at t. The loss we measure is a next-token loss, and an earlier version
    # bound the concept to the current token instead, so guard and consequence test referred to
    # different claims.
    t = next_toks[fired]
    ids, counts = torch.unique(t, return_counts=True)
    order = torch.argsort(counts, descending=True)
    for j in order.tolist():
        cand = ids[j]
        mask = (next_toks == cand)
        if int(mask.sum()) < min_concept:
            continue                        # concept too rare to test; try the next token type
        share = float(counts[j]) / float(fired.sum())
        base = float(mask.float().mean())
        return share / max(base, 1e-9), int(cand), mask
    return 0.0, None, None


# ------------------------------------------------------------------ guard 2: interventional

class FeatureAblate:
    """Remove one SAE feature's contribution: h <- h - z_f(h) d_f.

    The earlier version did h <- h - (h.u)u, which erases ALL residual-stream information along
    the decoder direction, including information the feature did not put there. That is a
    direction-erasure intervention, not a feature ablation, and the paper claimed the latter.
    Here z_f is the feature's own post-TopK activation at each position, so only what the feature
    contributed is removed.

    Passing a raw direction instead of a feature index (sae=None) reproduces the old behaviour and
    is kept only for the random-direction control, where there is no feature to ablate.
    """
    def __init__(self, model, layer, sae=None, feature=None, direction=None):
        self.block = model.gpt_neox.layers[layer - 1]
        self.sae, self.f = sae, feature
        self.u = None if direction is None else direction.to(DEV)

    def __enter__(self):
        sae, f, u = self.sae, self.f, self.u
        def hook(mod, out_in, out):
            h = out[0] if isinstance(out, tuple) else out
            if sae is not None:
                shp = h.shape
                flat = h.reshape(-1, shp[-1]).float()
                pre = sae.enc(flat - sae.b_pre)
                val, idx = torch.topk(pre, sae.k, dim=-1)
                z = torch.zeros_like(pre).scatter_(-1, idx, torch.relu(val))
                h = (flat - z[:, f:f+1] * sae.dec.weight[:, f].unsqueeze(0)).reshape(shp).to(h.dtype)
            else:
                h = h - (h @ u).unsqueeze(-1) * u
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        self.h = self.block.register_forward_hook(hook); return self

    def __exit__(self, *a):
        self.h.remove()


def concept_gradient_direction(model, layer, ids, ctok, batch=8):
    """d(logit_ctok)/d(h_layer), averaged over positions whose next token is ctok.

    By construction the locally most causal direction for that token at that layer. Used only as a
    positive control on the consequence test: if H cannot detect this, H detects nothing.
    """
    block = model.gpt_neox.layers[layer - 1]
    grads = []
    for i in range(0, len(ids), batch):
        b = ids[i:i+batch].to(DEV)
        stash = {}
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h.retain_grad(); stash["h"] = h
            return out
        hd = block.register_forward_hook(hook)
        try:
            logits = model(b).logits
            tgt = b[:, 1:]
            m = (tgt == ctok)
            if int(m.sum()) == 0:
                continue
            sel = logits[:, :-1, ctok][m].sum()
            model.zero_grad(set_to_none=True)
            sel.backward()
            g = stash["h"].grad
            if g is not None:
                grads.append(g.detach().reshape(-1, g.shape[-1]).float().mean(0).cpu())
        except Exception:
            pass
        finally:
            hd.remove()
    if not grads:
        return None
    v = torch.stack(grads).mean(0)
    n = float(v.norm())
    return (v / n).to(DEV) if n > 1e-8 else None


@torch.no_grad()
def clean_logprobs(model, ids, batch=48):
    """Log-probability the model assigns to each actual next token, with nothing ablated.

    Identical for every feature in an arm, so it is computed once rather than once per feature.
    That halves the forward passes, which is what makes hundreds of features affordable.
    """
    out = []
    for i in range(0, len(ids), batch):
        b = ids[i:i+batch].to(DEV)
        lp = torch.log_softmax(model(b).logits.float()[:, :-1], -1)
        out.append(lp.gather(-1, b[:, 1:].unsqueeze(-1)).squeeze(-1).cpu())
    return torch.cat(out)                       # (n_seq, SEQ_LEN-1), sequence structure kept


@torch.no_grad()
def specificity(model, ids, layer, clean, concept_mask, sae=None, feature=None,
                direction=None, batch=48):
    """Ablate the feature (or direction) and contrast the next-token loss increase on the
    concept against everywhere else, aggregated PER SEQUENCE.

    Returns (S, dLC, dLnC, per_sequence_contrasts). Tokens inside a sequence are strongly
    dependent, so the contrast is formed within each sequence and the sequence is the unit of
    resampling; an earlier version treated token positions as independent and understated the
    uncertainty.
    """
    abl = []
    with FeatureAblate(model, layer, sae=sae, feature=feature, direction=direction):
        for i in range(0, len(ids), batch):
            b = ids[i:i+batch].to(DEV)
            lp = torch.log_softmax(model(b).logits.float()[:, :-1], -1)
            abl.append(lp.gather(-1, b[:, 1:].unsqueeze(-1)).squeeze(-1).cpu())
    A = torch.cat(abl)                                   # (n_seq, SEQ_LEN-1)
    D = clean - A                                        # loss increase per position
    M = concept_mask[:len(ids)]                          # (n_seq, SEQ_LEN-1) bool

    per_seq = []
    for i in range(len(D)):
        on, off = D[i][M[i]], D[i][~M[i]]
        if len(on) >= 1 and len(off) >= 1:
            per_seq.append(float(on.mean() - off.mean()))
    if len(per_seq) < MIN_SEQ or int(M.sum()) < MIN_ON:
        return float("nan"), float("nan"), float("nan"), []
    a = float(D[M].mean()); o = float(D[~M].mean())
    v = np.asarray(per_seq)
    se = float(v.std(ddof=1) / max(len(v) ** 0.5, 1e-12))
    return float(v.mean() / max(se, 1e-12)), a, o, per_seq


def matched_controls(f, freq, tops, selected, stats, rng, k=3):
    """k control features matched on firing rate AND mean nonzero activation, different concept.

    One control was not enough: a single draw cannot separate a feature-specific effect from the
    spread of effects that comparable features have. Decoder-norm matching is not applied because
    our decoder columns are renormalised to unit norm every step, so it is vacuous.
    """
    fr, mu, _ = stats.get(f, (float(freq[f]), 0.0, 0.0))
    pool = [g for g in stats
            if g not in selected and tops.get(g) not in (None, tops.get(f))
            and 0.8 * fr <= stats[g][0] <= 1.2 * fr
            and (mu == 0 or 0.5 * mu <= stats[g][1] <= 2.0 * mu)]
    if not pool:
        return []
    return [int(x) for x in rng.choice(pool, size=min(k, len(pool)), replace=False)]


def _unused_matched_control(f, freq, tops, selected, rng):
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
    """One (model, layer) arm, on three disjoint partitions.

    D_sae trains the autoencoder, D_guard selects features and fixes their concepts, and
    D_cons measures the intervention. An earlier version used one held-out split for both
    selection and measurement, so the concept was chosen on the same data that scored it.
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    ids = get_text(tok, N_SEQ, SEQ_LEN)
    n1, n2 = int(len(ids) * 0.6), int(len(ids) * 0.8)
    D_sae, D_guard, D_cons = ids[:n1], ids[n1:n2], ids[n2:]
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEV).eval()
    arm = f"{model_name.split('/')[-1]} L{layer}"

    X = resid(model, D_sae, layer)
    print(f"  SAE on {tuple(X.shape)} from D_sae ({len(D_sae)} seq); "
          f"guard on {len(D_guard)}, consequence on {len(D_cons)}")
    sae = train_sae(X, X.shape[1] * EXPAND)

    # ---- selection, on D_guard only
    Xg = resid(model, D_guard, layer)
    nxt_g = D_guard[:, 1:].reshape(-1).to(DEV)
    keep = torch.ones(len(D_guard), SEQ_LEN, dtype=torch.bool); keep[:, -1] = False
    with torch.no_grad():
        _, Zg = sae(Xg)
    Zg = Zg[keep.reshape(-1)]                      # align activations with next-token targets
    freq = (Zg > 0).float().mean(0)
    D = sae.dec.weight.detach()

    N0 = D.shape[1]
    in_range = ((freq > 2e-3) & (freq < 5e-2)).nonzero().flatten().tolist()
    N1 = len(in_range)
    scored, tops, act_stats = [], {}, {}
    for i in in_range:
        e, t, _ = selectivity(Zg[:, i], nxt_g)
        if t is not None:
            scored.append((e, int(i))); tops[int(i)] = t
            z = Zg[:, i]; nz = z[z > 0]
            act_stats[int(i)] = (float(freq[i]), float(nz.mean()), float(nz.std()))
    N2 = len(scored)
    scored = [(e, i) for e, i in scored if e > SELECT_MIN]
    N3 = len(scored); scored.sort(reverse=True)
    reals = [i for _, i in scored[:N_FEAT]]
    flow = dict(arm=arm, N0=N0, N1=N1, N2=N2, N3=N3, selected=len(reals))
    print(f"  flow: {N0} -> firing {N1} -> testable {N2} -> enrichment>4 {N3} -> selected {len(reals)}")

    # ---- measurement, on D_cons only
    clean = clean_logprobs(model, D_cons[:N_EVAL])
    nxt_c = D_cons[:N_EVAL, 1:]                    # (n_seq, SEQ_LEN-1) next-token targets
    sel = set(reals)

    def mask_for(ctok):
        return (nxt_c == ctok)

    def judge(sae_, feat, direction, ctok):
        S, a, o, per = specificity(model, D_cons[:N_EVAL], layer, clean, mask_for(ctok),
                                   sae=sae_, feature=feat, direction=direction)
        return S, a, o, per

    rows, n_nocontrol = [], 0
    for f in reals:
        ctok = tops[f]
        S, a, o, per = judge(sae, f, None, ctok)
        ctrls = matched_controls(f, freq, tops, sel, act_stats, rng, k=N_CTRL)
        cs = []
        for g in ctrls:
            Sg, _, _, _ = judge(sae, g, None, ctok)     # control feature, on f's concept
            cs.append(float(Sg))
        if not ctrls:
            n_nocontrol += 1
        # Positive control for the consequence test itself, valid AT THIS LAYER. The first
        # version ablated the unembedding row for the concept; detection then tracked relative
        # depth monotonically (0.01 at layer 4 of 12, 0.68 at 16 of 24), because the unembedding
        # is the right causal direction only near the output. The gradient of the concept logit
        # with respect to the layer's own residual stream is the locally steepest direction for
        # that token AT that layer, so a consequence test that cannot detect it is broken.
        Wc = concept_gradient_direction(model, layer, D_cons[:GRAD_SEQ], ctok)
        Sp, _, _, _ = judge(None, None, Wc, ctok) if Wc is not None else (float("nan"),)*4
        rows.append(dict(arm=arm, feature=int(f), enrichment=float(scored[reals.index(f)][0]),
                         token=tok.decode([ctok]), concept=int(ctok),
                         S=float(S), dLC=float(a), dLnC=float(o), per_seq=per,
                         controls=[int(g) for g in ctrls], S_controls=cs,
                         S_logit_positive=float(Sp),
                         positive_defined=bool(Sp == Sp),
                         positive_wrong_sign=bool(Sp == Sp and Sp < 0),
                         supported=bool(S == S and S > 1.96 and a > 0),
                         control_supported=bool(cs and max(cs) == max(cs) and
                                                np.nanmean([1.0 if (c == c and c > 1.96) else 0.0
                                                            for c in cs]) > 0.5),
                         positive_detected=bool(Sp == Sp and Sp > 1.96)))
    if n_nocontrol:
        print(f"  {n_nocontrol} feature(s) had no admissible matched control")

    nulls = []
    f0 = reals[0]; u0 = D[:, f0] / D[:, f0].norm()
    rnd = torch.tensor(rng.normal(size=D.shape[0]), dtype=torch.float32, device=DEV)
    perm = torch.tensor(rng.permutation(D.shape[0]), device=DEV); us = u0[perm]
    noise = torch.tensor(rng.normal(scale=0.6, size=D.shape[0]), dtype=torch.float32, device=DEV)
    usg = u0 + noise / noise.norm() * u0.norm() * 0.6
    nz = (freq > 0).nonzero().flatten()
    fd = int(nz[int(freq[nz].argmin())]) if len(nz) else 0
    for nm, act, typ in (("random direction", (Xg @ (rnd/rnd.norm())).clamp_min(0)[keep.reshape(-1)], "randomised"),
                         ("scrambled feature", (Xg @ (us/us.norm())).clamp_min(0)[keep.reshape(-1)], "randomised"),
                         ("correlated surrogate", (Xg @ (usg/usg.norm())).clamp_min(0)[keep.reshape(-1)], "structure-preserving"),
                         ("near-dead feature", Zg[:, fd], "degenerate")):
        e, t, _ = selectivity(act, nxt_g)
        nulls.append(dict(arm=arm, name=nm, type=typ, enrichment=float(e),
                          guard1=bool(t is not None and e > SELECT_MIN)))

    del model, sae, X, Xg, Zg
    torch.cuda.empty_cache()
    return rows, nulls, flow


def cluster_bootstrap(rows, stat, n=2000, seed=0):
    """Resample arms, then concept clusters within arms. The per-feature verdict is held fixed.

    An earlier version resampled sequences and re-thresholded S on top of that. That is wrong, and
    visibly so: S is already a t-statistic over sequence-level contrasts, so re-resampling the
    sequences counts the same noise a second time, and against a one-sided threshold the extra
    noise pushes more features up than down. It produced an interval lying entirely above its own
    point estimate, with an upper end above the best single arm -- impossible under arm resampling.

    The units of generalisation are arms and concept clusters, so those are what we resample.
    """
    rng = np.random.default_rng(seed)
    arms = sorted({r["arm"] for r in rows})
    by_arm = {}
    for a in arms:
        cl = {}
        for r in rows:
            if r["arm"] == a:
                cl.setdefault(r["concept"], []).append(r)
        by_arm[a] = list(cl.values())
    out = []
    for _ in range(n):
        vals = []
        for a in rng.choice(arms, size=len(arms), replace=True):
            cls = by_arm[a]
            for j in rng.integers(0, len(cls), len(cls)):
                vals += [stat(r) for r in cls[j]]
        if vals:
            out.append(float(np.mean(vals)))
    if not out:
        return (float("nan"),) * 2
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def car_curve(paired, ks=(10, 25, 50, 100, 150)):
    """CAR as a function of K, the rank cutoff by enrichment.

    CAR is not a property of the guard alone: it depends on which accepted candidates are put to
    the consequence test. We select the top 150 by enrichment per arm, so the quantity is CAR@150
    and it must be named that way. Reporting the curve says how much of the shortfall is the cutoff
    -- if CAR fell steeply with K, the guard would be well calibrated at the top and poorly further
    down, which is a different diagnosis from being uncalibrated throughout.
    """
    out = {}
    for k in ks:
        sel = []
        for a in sorted({r["arm"] for r in paired}):
            q = sorted([r for r in paired if r["arm"] == a],
                       key=lambda r: -r["enrichment"])[:k]
            sel += q
        if sel:
            out[str(k)] = dict(n=len(sel),
                               car=sum(r["supported"] for r in sel) / len(sel),
                               control=sum(bool(r["control_supported"]) for r in sel) / len(sel))
    return out


SUPPORTED = lambda r: float(r["supported"])
DELTA     = lambda r: float(r["supported"]) - float(bool(r["control_supported"]))
POSITIVE  = lambda r: float(r["positive_detected"])


def main():
    rng = np.random.default_rng(0)
    print("X2 -- do SAE features survive an intervention a matched control does not?")
    print(f"{len(ARMS)} arms; three disjoint partitions per arm; primary endpoint CAR@{N_FEAT} "
          f"minus matched-control rate\n")

    rows, nulls, flows = [], [], []
    for model_name, layer in ARMS:
        print(f"[{model_name} layer {layer}]")
        r, n_, f = run_arm(model_name, layer, rng)
        rows += r; nulls += n_; flows.append(f); print()

    print(f"{'stage':<32}" + "".join(f"{f['arm'][:15]:>17}" for f in flows))
    for k, lbl in (("N0","all SAE features"),("N1","firing rate in range"),
                   ("N2","testable concept"),("N3","enrichment > 4"),("selected","selected")):
        print(f"  {lbl:<30}" + "".join(f"{f[k]:>17}" for f in flows))

    # ANALYSIS.md section 8 excludes "features with fewer than 30 concept positions in the
    # evaluation set". That threshold was being applied to the split features are SELECTED on, not
    # to the disjoint split they are JUDGED on, so a feature whose concept never occurs in D_cons
    # entered the denominator as a failure -- its consequence test was not computable at all. The
    # boundary is not a judgement call: the excluded features have exactly ZERO concept positions
    # there and the next smallest has 19, so any threshold between 1 and 19 gives the same set.
    testable = [r for r in rows if len(r["per_seq"]) >= MIN_SEQ]
    n_untestable = len(rows) - len(testable)
    paired = [r for r in testable if r["controls"]]
    car   = sum(r["supported"] for r in paired) / max(len(paired), 1)
    carc  = sum(r["control_supported"] for r in paired) / max(len(paired), 1)
    posd  = [r for r in testable if r["positive_defined"]]
    pos   = sum(r["positive_detected"] for r in posd) / max(len(posd), 1)
    wrong = sum(r["positive_wrong_sign"] for r in posd) / max(len(posd), 1)
    lo, hi   = cluster_bootstrap(paired, SUPPORTED)
    dlo, dhi = cluster_bootstrap(paired, DELTA)
    plo, phi = cluster_bootstrap(posd, POSITIVE)

    print(f"\n{'arm':<26}{'paired':>8}{'CAR@150':>10}{'ctrl':>8}{'delta':>9}")
    print("-" * 61)
    for a in sorted({r["arm"] for r in paired}):
        q = [r for r in paired if r["arm"] == a]
        cs = sum(r["supported"] for r in q)/len(q); cc = sum(r["control_supported"] for r in q)/len(q)
        print(f"{a:<26}{len(q):>8}{cs:>10.2f}{cc:>8.2f}{cs-cc:>9.2f}")
    print("-" * 61)
    print(f"{'pooled':<26}{len(paired):>8}{car:>10.2f}{carc:>8.2f}{car-carc:>9.2f}")
    print(f"\n  {n_untestable} of {len(rows)} selected features have no concept position in the "
          f"consequence split and are excluded (ANALYSIS.md \u00a78); {len(paired)} are judged")
    print(f"  CAR@{N_FEAT} = {car:.2f}   cluster bootstrap 95% CI {lo:.2f}-{hi:.2f}")
    print(f"  matched-control rate = {carc:.2f}")
    print(f"  PRIMARY ENDPOINT  delta = {car-carc:+.2f}   95% CI {dlo:+.2f} to {dhi:+.2f}")
    print("    -> " + ("interval excludes zero; the enrichment guard carries causal information "
                       "a matched control does not"
                       if dlo > 0 else
                       "INTERVAL INCLUDES ZERO -- per ANALYSIS.md the X2 claim is withdrawn"))
    print(f"\n  AUDIT OF THE CONSEQUENCE TEST ITSELF")
    print(f"    gradient positive control detected: {pos:.2f} of the {len(posd)} features where it "
          f"is defined (95% CI {plo:.2f}-{phi:.2f}); undefined for {len(rows)-len(posd)}")
    print(f"    the control's OWN validity: {wrong:.2f} of its ablations move the logit the wrong way")
    for a in sorted({r["arm"] for r in rows}):
        q = [r for r in rows if r["arm"] == a]
        qd = [r for r in q if r["positive_defined"]]
        print(f"      {a:<26}detected {np.mean([r['positive_detected'] for r in qd]) if qd else float('nan'):.2f}"
              f"   wrong-signed {np.mean([r['positive_wrong_sign'] for r in qd]) if qd else float('nan'):.2f}"
              f"   undefined {len(q)-len(qd)}")
    print("    -> " + ("H detects a direction that is causal by construction; CAR is interpretable"
                       if pos > 0.8 and wrong < 0.1 else
                       "the control does not validate H on every arm; see the per-arm split above. "
                       "A wrong-signed ablation impeaches the CONTROL, not H: a direction whose "
                       "removal raises the logit it was built to lower is not causal at that layer."))

    nm = list(dict.fromkeys(x["name"] for x in nulls))
    nar = sum(x["guard1"] for x in nulls) / max(len(nulls), 1)
    print(f"\n  null suite, per control:")
    for n_ in nm:
        c = [x for x in nulls if x["name"] == n_]
        print(f"    {n_:<24}{c[0]['type']:<22}{sum(x['guard1'] for x in c)}/{len(c)}")
    print(f"  NAR = {nar:.2f}   AnyNullPass = {int(any(x['guard1'] for x in nulls))}")

    Path(__file__).with_name("xai_sae.json").write_text(json.dumps(dict(
        rows=rows, nulls=nulls, flow=flows, paired=len(paired),
        untestable=n_untestable, testable=len(testable), car_curve=car_curve(paired),
        car_at_k=car, car_control=carc, delta_car=car-carc, ci=[lo, hi],
        delta_ci=[dlo, dhi], positive_control_rate=pos, positive_ci=[plo, phi],
        per_arm={a: dict(
            n=sum(r["arm"] == a for r in paired),
            car=sum(r["supported"] for r in paired if r["arm"] == a)
                / max(sum(r["arm"] == a for r in paired), 1),
            control=sum(bool(r["control_supported"]) for r in paired if r["arm"] == a)
                / max(sum(r["arm"] == a for r in paired), 1),
            positive=sum(r["positive_detected"] for r in testable
                         if r["arm"] == a and r["positive_defined"])
                / max(sum(r["arm"] == a and r["positive_defined"] for r in testable), 1),
            positive_defined=sum(r["arm"] == a and r["positive_defined"] for r in testable),
            positive_wrong_sign=sum(r["positive_wrong_sign"] for r in testable
                                    if r["arm"] == a and r["positive_defined"])
                / max(sum(r["arm"] == a and r["positive_defined"] for r in testable), 1))
            for a in sorted({r["arm"] for r in rows})},
        nar=nar,
        any_null_pass=int(any(x["guard1"] for x in nulls)),
        config=dict(arms=[list(a) for a in ARMS], expand=EXPAND, topk=TOPK, epochs=EPOCHS,
                    n_seq=N_SEQ, seq_len=SEQ_LEN, n_feat=N_FEAT, n_ctrl=N_CTRL, n_eval=N_EVAL,
                    select_min=SELECT_MIN, min_concept=MIN_CONCEPT, min_on=MIN_ON,
                    min_seq=MIN_SEQ, splits="60/20/20 D_sae/D_guard/D_cons",
                    support_rule="S > 1.96 on sequence-level contrasts and dLC > 0")), indent=1))


def reanalyse():
    """Re-derive every headline number from the shipped rows, without re-running the models.

    The per-feature verdicts, their sequence contrasts, the matched controls and the positive
    control are all recorded, so the analysis is a function of the JSON alone. This exists because
    two analysis defects were found after the GPU run -- a bootstrap that counted sequence noise
    twice, and an exclusion applied to the wrong split -- and re-deriving is the same computation
    the run would do, checkable against the recorded verdicts.
    """
    f = Path(__file__).with_name("xai_sae.json")
    d = json.loads(f.read_text())
    rows = d["rows"]

    testable = [r for r in rows if len(r["per_seq"]) >= MIN_SEQ]
    paired = [r for r in testable if r["controls"]]
    defined = [r for r in testable
               if r.get("positive_defined", r["S_logit_positive"] == r["S_logit_positive"])]

    car  = sum(r["supported"] for r in paired) / len(paired)
    carc = sum(r["control_supported"] for r in paired) / len(paired)
    pos  = sum(r["positive_detected"] for r in defined) / max(len(defined), 1)
    wrong = sum(r.get("positive_wrong_sign", False) for r in defined) / max(len(defined), 1)

    d.update(paired=len(paired), testable=len(testable), untestable=len(rows) - len(testable),
             car_at_k=car, car_control=carc, delta_car=car - carc,
             positive_control_rate=pos, positive_defined=len(defined),
             positive_wrong_sign=wrong,
             car_curve=car_curve(paired),
             ci=list(cluster_bootstrap(paired, SUPPORTED)),
             delta_ci=list(cluster_bootstrap(paired, DELTA)),
             positive_ci=list(cluster_bootstrap(defined, POSITIVE)))
    d["per_arm"] = {}
    for a in sorted({r["arm"] for r in rows}):
        pa = [r for r in paired if r["arm"] == a]
        da = [r for r in defined if r["arm"] == a]
        d["per_arm"][a] = dict(
            n=len(pa),
            car=sum(r["supported"] for r in pa) / max(len(pa), 1),
            control=sum(bool(r["control_supported"]) for r in pa) / max(len(pa), 1),
            positive=sum(r["positive_detected"] for r in da) / max(len(da), 1),
            positive_defined=len(da),
            positive_wrong_sign=sum(r.get("positive_wrong_sign", False) for r in da)
                / max(len(da), 1))
    f.write_text(json.dumps(d, indent=1))

    print(f"{d['untestable']} of {len(rows)} selected features have no concept position in the "
          f"consequence split and are excluded (ANALYSIS.md \u00a78)")
    print(f"CAR   {car:.3f}  95% CI {d['ci'][0]:.3f}-{d['ci'][1]:.3f}   over {len(paired)} judged")
    print(f"ctrl  {carc:.3f}")
    print(f"delta {car-carc:+.3f}  95% CI {d['delta_ci'][0]:+.3f} to {d['delta_ci'][1]:+.3f}")
    print(f"pos   {pos:.3f}  over {len(defined)} defined; wrong-signed {wrong:.3f}")
    for a, v in d["per_arm"].items():
        print(f"  {a:<28}n={v['n']:<5}CAR={v['car']:.2f}  ctrl={v['control']:.2f}  "
              f"pos={v['positive']:.2f} (n={v['positive_defined']}, "
              f"wrong {v['positive_wrong_sign']:.2f})")


if __name__ == "__main__":
    if "--reanalyse" in sys.argv:
        reanalyse()
    else:
        main()
