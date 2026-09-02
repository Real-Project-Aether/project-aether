"""X1 --- does a representational correspondence between two models survive a causal test?

The claim under test is a **correspondence**: that model A's residual stream at one layer means the
same thing as model B's at another, via a fitted linear map W.

    guard 1, STRUCTURAL -- the one in common use. Linear alignment R^2 and linear CKA, reported
             in-sample and held out and swept over the number of stimuli the map is fitted on,
             because our L3 result showed this guard's failure rate is largely a statement about
             how much data it was scored on.

    guard 2, CAUSAL -- both models share a tokenizer, so their next-token distributions are
             directly comparable. Patch a donor state into A at ONE position; push the same state
             through W, patch THAT into B at the same position, and compare how the two models'
             distributions move at that position. A real correspondence transfers *which* donor
             was patched; a map that merely reconstructs well has no reason to.

REBUILD, after a review found four defects in the previous version. Each is fixed here and the
fix is named so the change is checkable rather than asserted:

  1. `r2` centred the held-out data on its OWN mean, which leaks the evaluation set's location
     into its own score. Both the residual and the total sum of squares are now taken about the
     mean of the FITTING set, which is the honest out-of-sample R^2.

  2. Patching overwrote the residual stream at EVERY position with one broadcast vector
     (`expand_as`). That is not an intervention on a representation, it is replacing the model's
     entire state, and it makes any two models look alike because both collapse. Patching is now
     localised to a single position, every other position left untouched, and the effect is read
     at that position.

  3. Eight donors is too few to distinguish a correspondence from noise. The sweep now runs
     hundreds of donor x position x recipient triples across several seeds.

  4. A single threshold at 0.20 was doing the work of an evaluation. The causal score is now
     reported as a distribution per candidate, with AUROC against each null and an acceptance
     curve over the whole threshold range, so a reader can pick their own operating point.

The null suite is also expanded, in the direction the review asked for -- nulls that break the
CORRESPONDENCE while preserving the marginal structure that guard 1 looks at, which is the only
kind of null that can separate the two guards:

    permuted_pairs   W fitted on row-shuffled (A,B) pairs: same marginals, no pairing
    orthogonal       the real W followed by a random rotation of B's space
    gaussian_matched W fitted to Gaussian targets with B's mean and covariance
    donor_perm       the real W, but B receives a DIFFERENT donor's state than A did
    untrained        B randomly initialised
    mismatched       A and B read disjoint text

Requires torch, transformers, datasets and a GPU. verify.py asserts against the shipped JSON
rather than re-running it.
"""
import json, warnings, sys
from pathlib import Path
import numpy as np

# torchvision 0.22.1 here is built against a different torch and its C++ ops fail to load;
# transformers imports it unconditionally via image_utils. We touch no images.
import transformers.utils.import_utils as _iu
_iu.is_torchvision_available = lambda *a, **k: False
import transformers.utils as _tu
_tu.is_torchvision_available = _iu.is_torchvision_available

import torch                                                              # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig  # noqa: E402

warnings.filterwarnings("ignore")
torch.manual_seed(0)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Every pair must share the Pythia tokenizer: guard 2 compares two next-token distributions, so a
# shared output vocabulary is a precondition. GPT-2 cannot be a target for that reason, which is a
# real limit of the causal guard and is stated in the paper rather than worked around.
CONFIGS = (
    ("EleutherAI/pythia-70m",  4, "EleutherAI/pythia-160m", 8),
    ("EleutherAI/pythia-70m",  2, "EleutherAI/pythia-160m", 4),
    ("EleutherAI/pythia-160m", 8, "EleutherAI/pythia-410m", 12),
    ("EleutherAI/pythia-70m",  4, "EleutherAI/pythia-410m", 12),
)

N_SEQ, SEQ_LEN = 240, 128
RIDGE          = 1e2
SWEEP          = [8, 32, 128, 1024, 8192]     # stimuli the map is fitted on
R2_MIN, CKA_MIN, CAUSAL_MIN = 0.50, 0.50, 0.20

SEEDS      = 5
N_DONOR    = 12      # donor states per seed
N_POS      = 8       # positions patched per donor
N_RECIP    = 16      # recipient sequences averaged over per triple
POS_LO     = 16      # avoid the first tokens, whose distribution is dominated by position


# ------------------------------------------------------------------ data / activations

def get_text(tok, n_seq, seq_len, skip=0):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    buf, out, seen = [], [], 0
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200:
            continue
        seen += 1
        if seen <= skip:
            continue
        buf.extend(tok(t).input_ids)
        while len(buf) >= seq_len and len(out) < n_seq:
            out.append(buf[:seq_len]); buf = buf[seq_len:]
        if len(out) >= n_seq:
            break
    return torch.tensor(out)


@torch.no_grad()
def acts(model, ids, layer, batch=24):
    out = []
    for i in range(0, len(ids), batch):
        h = model(ids[i:i+batch].to(DEV), output_hidden_states=True).hidden_states[layer]
        out.append(h.reshape(-1, h.shape[-1]).float().cpu())
    return torch.cat(out)


# ------------------------------------------------------------------ guard 1

def ridge_fit(X, Y, lam=RIDGE):
    """Return (W, mu_X, mu_Y). The means are returned because scoring must reuse THESE."""
    muX, muY = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Xc, Yc = X - muX, Y - muY
    G = Xc.T @ Xc + lam * torch.eye(Xc.shape[1])
    return torch.linalg.solve(G, Xc.T @ Yc), muX, muY


def r2(X, Y, W, muX, muY):
    """Out-of-sample R^2, taken about the FITTING mean.

    The previous version centred X and Y on their own means before scoring, so the evaluation set's
    location was fitted to itself. On held-out data that is leakage, and it inflated every number
    in the sweep.
    """
    resid = ((Y - muY) - (X - muX) @ W) ** 2
    total = (Y - muY) ** 2
    return float(1 - resid.sum() / max(float(total.sum()), 1e-12))


def linear_cka(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    return float((Xc.T @ Yc).norm() ** 2 / max(float((Xc.T @ Xc).norm() * (Yc.T @ Yc).norm()), 1e-12))


# ------------------------------------------------------------------ guard 2: localised patching

class PatchAt:
    """Replace the residual stream at ONE position; every other position is left alone."""
    def __init__(self, model, layer, pos, value):
        self.block = model.gpt_neox.layers[layer - 1]
        self.pos, self.value = pos, value

    def __enter__(self):
        pos, val = self.pos, self.value

        def hook(mod, inp, out):
            tup = isinstance(out, tuple)
            h = (out[0] if tup else out).clone()
            h[:, pos, :] = val.to(h.device, h.dtype)
            return ((h,) + tuple(out[1:])) if tup else h

        self.h = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


class AddAt:
    """Add a per-recipient delta to the residual stream at ONE position.

    Difference patching: rather than overwriting B's state with a mapped donor state -- which asks
    W to reconstruct an absolute activation, and so confounds the causal test with the structural
    one it is meant to be independent of -- we add the mapped DIFFERENCE. B keeps its own state and
    receives a controlled perturbation, which stays in distribution and tests exactly the claim:
    does the correspondence carry a causal change from one model to the other?

    The map's intercepts cancel in a difference, so only W is needed here, not the means.
    """
    def __init__(self, model, layer, pos, delta):
        self.block = model.gpt_neox.layers[layer - 1]
        self.pos, self.delta = pos, delta

    def __enter__(self):
        pos, dl = self.pos, self.delta

        def hook(mod, inp, out):
            tup = isinstance(out, tuple)
            h = (out[0] if tup else out).clone()
            h[:, pos, :] = h[:, pos, :] + dl.to(h.device, h.dtype)
            return ((h,) + tuple(out[1:])) if tup else h

        self.h = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def added_shift(model, ids, layer, pos, delta, clean):
    with AddAt(model, layer, pos, delta):
        p = torch.softmax(model(ids.to(DEV)).logits[:, pos].float(), -1).mean(0).cpu()
    return p - clean


@torch.no_grad()
def hidden_at(model, ids, layer, pos):
    """Clean residual stream at `pos` for each sequence in the batch."""
    h = model(ids.to(DEV), output_hidden_states=True).hidden_states[layer][:, pos, :]
    return h.float().cpu()


@torch.no_grad()
def clean_dist(model, ids, pos):
    return torch.softmax(model(ids.to(DEV)).logits[:, pos].float(), -1).mean(0).cpu()


@torch.no_grad()
def patched_shift(model, ids, layer, pos, value, clean):
    """Mean change in the next-token distribution AT `pos` when `value` is patched in at `pos`."""
    with PatchAt(model, layer, pos, value):
        p = torch.softmax(model(ids.to(DEV)).logits[:, pos].float(), -1).mean(0).cpu()
    return p - clean


def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / max(float(x.norm() * y.norm()), 1e-12))


def auroc(pos_scores, neg_scores):
    """Rank-based AUROC; ties get half credit. No sklearn dependency."""
    p, n = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


# ------------------------------------------------------------------ the sweep

def run_config(A_name, LA, B_name, LB, log):
    tok = AutoTokenizer.from_pretrained(A_name)
    ids = get_text(tok, N_SEQ, SEQ_LEN)
    other = get_text(tok, N_SEQ, SEQ_LEN, skip=4000)
    n_fit = len(ids) // 2
    ids_fit, ids_held = ids[:n_fit], ids[n_fit:]

    A = AutoModelForCausalLM.from_pretrained(A_name).to(DEV).eval()
    B = AutoModelForCausalLM.from_pretrained(B_name).to(DEV).eval()
    B0 = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(B_name)).to(DEV).eval()

    A_fit, A_held = acts(A, ids_fit, LA), acts(A, ids_held, LA)
    B_fit, B_held = acts(B, ids_fit, LB), acts(B, ids_held, LB)
    B0_fit, B0_held = acts(B0, ids_fit, LB), acts(B0, ids_held, LB)
    Bm_fit, Bm_held = acts(B, other[:n_fit], LB), acts(B, other[n_fit:], LB)

    g = torch.Generator().manual_seed(0)
    dB = B_fit.shape[1]

    # Gaussian targets matched to B's first two moments: guard 1 sees the same covariance
    # structure, so anything it reports here is reconstruction and not correspondence.
    Bc = B_fit - B_fit.mean(0, keepdim=True)
    cov = (Bc.T @ Bc) / max(len(Bc) - 1, 1)
    L = torch.linalg.cholesky(cov.double() + 1e-3 * torch.eye(dB, dtype=torch.float64)).float()
    def gauss(n):
        return torch.randn(n, dB, generator=g) @ L.T + B_fit.mean(0, keepdim=True)
    Bg_fit, Bg_held = gauss(len(B_fit)), gauss(len(B_held))

    perm = torch.randperm(len(B_fit), generator=g)
    Q, _ = torch.linalg.qr(torch.randn(dB, dB, generator=g))    # random orthogonal rotation

    # Each candidate is a CLAIM: "this map carries A's state at LA into B's at LB". Guard 1 must
    # therefore be scored with the map the candidate actually claims, against the target that
    # candidate is claimed to reach. An earlier version scored `orthogonal` and `donor_perm` with
    # the REAL map, so both reported R^2 identical to the real pair -- flattering and wrong.
    W_real,  muX,  muY  = ridge_fit(A_fit, B_fit)
    W_perm,  muXp, muYp = ridge_fit(A_fit, B_fit[perm])
    W_gauss, muXg, muYg = ridge_fit(A_fit, Bg_fit)
    W_mis,   muXm, muYm = ridge_fit(A_fit, Bm_fit)
    W_untr,  muXu, muYu = ridge_fit(A_fit, B0_fit)

    # name -> (effective map, means, held-out target it claims to reach, model patched, donor shift)
    CANDS = {
        "real":             (W_real,      muX,  muY,  B_held,  B,  0),
        "permuted_pairs":   (W_perm,      muXp, muYp, B_held,  B,  0),
        "gaussian_matched": (W_gauss,     muXg, muYg, B_held,  B,  0),
        "orthogonal":       (W_real @ Q,  muX,  muY,  B_held,  B,  0),
        "donor_perm":       (W_real,      muX,  muY,  B_held,  B,  1),
        "untrained":        (W_untr,      muXu, muYu, B0_held, B0, 0),
        "mismatched":       (W_mis,       muXm, muYm, B_held,  B,  0),
    }
    TYPES = {"real": "real",
             "permuted_pairs": "correspondence-breaking",
             "gaussian_matched": "randomised",
             "orthogonal": "correspondence-breaking",
             # donor_perm does not propose a different map: it holds the real correspondence fixed
             # and permutes which donor B receives, so it is a control on the CONSEQUENCE TEST
             # rather than a candidate for guard 1. Its guard-1 row is the real map's by
             # construction and is reported as such.
             "donor_perm": "consequence-test control",
             "untrained": "degenerate",
             "mismatched": "task-confounded"}

    # --- guard 1, swept over fitting-set size
    sweep = {}
    for v, (Wv, mx, my, Th, _mdl, _off) in CANDS.items():
        sweep[v] = {}
        tgt_fit = {"real": B_fit, "permuted_pairs": B_fit[perm], "gaussian_matched": Bg_fit,
                   "orthogonal": B_fit, "donor_perm": B_fit, "untrained": B0_fit,
                   "mismatched": Bm_fit}[v]
        for n in SWEEP:
            Wn, mxn, myn = ridge_fit(A_fit[:n], tgt_fit[:n])
            if v == "orthogonal":
                Wn = Wn @ Q
            sweep[v][n] = (r2(A_fit[:n], tgt_fit[:n], Wn, mxn, myn), r2(A_held, Th, Wn, mxn, myn))

    full = {}
    for v, (Wv, mx, my, Th, _mdl, _off) in CANDS.items():
        full[v] = dict(W=Wv, muX=mx, muY=my,
                       r2=r2(A_held, Th, Wv, mx, my), cka=linear_cka(A_held, Th))

    # --- guard 2, over donor x position x recipient triples
    recip = ids_held[:N_RECIP]
    donors_pool = ids_held[N_RECIP:]
    scores = {v: [] for v in CANDS}
    # The UNCENTRED score is kept because it is a finding, not a discard: correlating raw shifts
    # ranked a trained model reading random tokens ABOVE the genuine pair, since any patch pushes
    # both models toward the same common tokens and that shared generic response dominates. The
    # rebuild must still be able to reproduce it.
    raw = {v: [] for v in CANDS}

    for seed in range(SEEDS):
        rs = np.random.default_rng(100 + seed)
        positions = sorted(rs.choice(np.arange(POS_LO, SEQ_LEN - 1), N_POS, replace=False).tolist())
        di = rs.choice(len(donors_pool), N_DONOR, replace=False)
        dpos = rs.integers(POS_LO, SEQ_LEN - 1, N_DONOR)

        with torch.no_grad():
            dstates = []
            for k in range(N_DONOR):
                d = donors_pool[di[k]:di[k]+1].to(DEV)
                h = A(d, output_hidden_states=True).hidden_states[LA][0, int(dpos[k])]
                dstates.append(h.float().cpu())
            dstates = torch.stack(dstates)

        for pos in positions:
            # A's own clean state at the patched position, one per recipient: the difference is
            # taken against the recipient it is applied to, not against a pooled mean.
            HA = hidden_at(A, recip, LA, pos)                       # (R, dA)
            cleanA = clean_dist(A, recip, pos)
            sA, dltA = [], []
            for k in range(N_DONOR):
                d = dstates[k].unsqueeze(0) - HA                    # (R, dA) per-recipient delta
                dltA.append(d)
                sA.append(added_shift(A, recip, LA, pos, d, cleanA))
            sA = torch.stack(sA)
            cA = sA - sA.mean(0, keepdim=True)      # drop the generic response to being patched

            for v, (Wv, mx, my, Th, mdl, off) in CANDS.items():
                cleanB = clean_dist(mdl, recip, pos)
                # the map's intercepts cancel in a difference, so only W acts here
                mapped = [d @ Wv for d in dltA]
                if off:
                    mapped = mapped[off:] + mapped[:off]
                sB = torch.stack([added_shift(mdl, recip, LB, pos, mapped[k], cleanB)
                                  for k in range(N_DONOR)])
                cB = sB - sB.mean(0, keepdim=True)
                scores[v] += [corr(cA[k], cB[k]) for k in range(N_DONOR)]
                raw[v] += [corr(sA[k], sB[k]) for k in range(N_DONOR)]
        log(f"    seed {seed}: {len(scores['real'])} triples so far")

    del A, B, B0
    torch.cuda.empty_cache()

    nulls = [v for v in CANDS if v != "real"]
    out = {}
    for v in CANDS:
        s = np.array(scores[v], float)
        out[v] = dict(
            type=TYPES[v], r2=full[v]["r2"], cka=full[v]["cka"],
            guard1=bool(full[v]["r2"] > R2_MIN or full[v]["cka"] > CKA_MIN),
            causal_median=float(np.median(s)), causal_mean=float(s.mean()),
            causal_median_raw=float(np.median(raw[v])),
            causal_q=[float(x) for x in np.percentile(s, [5, 25, 75, 95])],
            n_triples=len(s), guard2=bool(np.median(s) > CAUSAL_MIN),
            auroc_vs_real=(None if v == "real" else auroc(scores["real"], scores[v])),
            sweep={str(k): list(val) for k, val in sweep[v].items()})
        out[v]["accepted"] = bool(out[v]["guard1"] and out[v]["guard2"])
    return out, nulls, scores, raw


def main():
    results = {}
    for (A_name, LA, B_name, LB) in CONFIGS:
        key = f"{A_name.split('/')[-1]} L{LA} -> {B_name.split('/')[-1]} L{LB}"
        print(f"[{key}]"); sys.stdout.flush()
        try:
            rows, nulls, scores, raw = run_config(A_name, LA, B_name, LB,
                                                  lambda m: (print(m), sys.stdout.flush()))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:120]}"); continue
        results[key] = rows
        print(f"  {'candidate':<18}{'type':<26}{'R2':>7}{'CKA':>7}{'g1':>5}"
              f"{'causal med':>12}{'AUROC':>8}{'g2':>5}  verdict")
        for v, r in rows.items():
            au = "  --  " if r["auroc_vs_real"] is None else f"{r['auroc_vs_real']:>6.3f}"
            print(f"  {v:<18}{r['type']:<26}{r['r2']:>7.3f}{r['cka']:>7.3f}"
                  f"{'P' if r['guard1'] else '.':>5}{r['causal_median']:>12.3f}{au:>8}"
                  f"{'P' if r['guard2'] else '.':>5}  {'ACCEPT' if r['accepted'] else 'reject'}")
        n1 = sum(rows[v]["guard1"] for v in nulls) / len(nulls)
        n2 = sum(rows[v]["accepted"] for v in nulls) / len(nulls)
        print(f"  NAR(guard 1 alone) = {n1:.2f}   NAR(both guards) = {n2:.2f}   "
              f"real accepted = {rows['real']['accepted']}\n"); sys.stdout.flush()

    # pooled across configurations
    allc = sorted({v for r in results.values() for v in r})
    print(f"\npooled over {len(results)} configurations")
    print(f"  {'candidate':<18}{'accept rate':>13}{'median AUROC':>14}")
    pooled = {}
    for v in allc:
        rs = [r[v] for r in results.values() if v in r]
        au = [x["auroc_vs_real"] for x in rs if x["auroc_vs_real"] is not None]
        pooled[v] = dict(accept_rate=float(np.mean([x["accepted"] for x in rs])),
                         guard1_rate=float(np.mean([x["guard1"] for x in rs])),
                         median_auroc=(float(np.median(au)) if au else None),
                         n_configs=len(rs))
        print(f"  {v:<18}{pooled[v]['accept_rate']:>13.2f}"
              f"{'  --' if not au else f'{np.median(au):>14.3f}'}")

    nulls = [v for v in allc if v != "real"]
    nar1 = float(np.mean([pooled[v]["guard1_rate"] for v in nulls]))
    nar2 = float(np.mean([pooled[v]["accept_rate"] for v in nulls]))
    small = float(np.mean([np.mean([r[v]["sweep"][str(SWEEP[0])][0] > R2_MIN
                                    for r in results.values() if v in r]) for v in nulls]))
    print(f"\n  NAR(guard 1, in-sample, {SWEEP[0]} stimuli) = {small:.2f}")
    print(f"  NAR(guard 1, held out, all stimuli)      = {nar1:.2f}")
    print(f"  NAR(both guards)                         = {nar2:.2f}")
    print(f"  AnyNullPass(both guards)                 = "
          f"{int(any(pooled[v]['accept_rate'] > 0 for v in nulls))}")

    Path(__file__).with_name("xai_cka.json").write_text(json.dumps(dict(
        configs=results, pooled=pooled,
        nar_structural_heldout=nar1, nar_structural_insample_small=small, nar_both=nar2,
        any_null_pass=int(any(pooled[v]["accept_rate"] > 0 for v in nulls)),
        sweep_n=SWEEP,
        config=dict(configs=[list(c) for c in CONFIGS], n_seq=N_SEQ, seq_len=SEQ_LEN, ridge=RIDGE,
                    seeds=SEEDS, n_donor=N_DONOR, n_pos=N_POS, n_recip=N_RECIP,
                    triples_per_config=SEEDS * N_POS * N_DONOR,
                    r2_min=R2_MIN, cka_min=CKA_MIN, causal_min=CAUSAL_MIN,
                    patch="difference patching at a single position, read at that position")), indent=1))


if __name__ == "__main__":
    main()
