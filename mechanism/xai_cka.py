"""X1 --- the L3 finding on neural-network representations.

Review round 1, point 1: the guard thesis had only ever been shown on physics. This runs it on two
language models, in the vocabulary an interpretability reviewer uses.

The claim under test is a **correspondence**: that model A's residual stream at one layer means the
same thing as model B's at another, via a fitted linear map W.

    guard 1, STRUCTURAL -- the one in common use. Linear alignment quality (R^2) and linear CKA.
             Reported both in-sample and held out, and swept over the number of stimuli the map is
             fitted on, because our L3 result showed this guard's failure rate is mostly a
             statement about how much data it was scored on.

    guard 2, CAUSAL -- both models share a tokenizer, so their output distributions are directly
             comparable. Patch a donor activation into A and record how A's next-token
             distribution moves. Push the same donor activation through W, patch THAT into B, and
             record how B's distribution moves. A real correspondence moves both models the same
             way; a map that merely reconstructs well has no reason to.

Vacuity class V, all of which should be refused:
    untrained    B is a randomly initialised model of the same architecture
    mismatched   B is the trained model read on DIFFERENT text -- real structure, no correspondence
    noise        B replaced by Gaussian noise at B's scale

`real` is two trained Pythia models, which do share mechanisms and should be accepted.


Requires torch, transformers and datasets, and a GPU for comfort. That is why verify.py
does not run it; its results ship alongside as JSON.
"""
import json, warnings
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

DEV     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_A, MODEL_B = "EleutherAI/pythia-70m", "EleutherAI/pythia-160m"
LAYER_A, LAYER_B = 4, 8
N_SEQ, SEQ_LEN   = 240, 128
RIDGE            = 1e2
CHEM_MODEL       = "seyonec/ChemBERTa-zinc-base-v1"
SWEEP            = [8, 32, 128, 1024, 8192]      # stimuli the map is fitted on

R2_MIN, CKA_MIN  = 0.50, 0.50                    # guard 1 accepts

# Real molecules, for the different-domain control. ChemBERTa on SMILES is a trained model with
# rich structure that shares no mechanism whatever with an English language model.
SMILES = ("CC(=O)OC1=CC=CC=C1C(=O)O CN1C=NC2=C1C(=O)N(C)C(=O)N2C CC(C)CC1=CC=C(C=C1)C(C)C(=O)O "
          "CC(=O)NC1=CC=C(O)C=C1 CN1CCC[C@H]1C1=CN=CC=C1 OC(=O)C1=CC=CC=C1O "
          "C1=CC=C(C=C1)C=O CCO CC(=O)O C1CCCCC1 c1ccccc1 CCN(CC)CC "
          "NC(=O)c1ccccc1 CC(C)(C)NCC(O)c1ccc(O)c(CO)c1 CN(C)CCCN1c2ccccc2Sc2ccccc21 "
          "COc1ccc2cc(ccc2c1)C(C)C(=O)O CC1=C(C=C(C=C1)S(=O)(=O)N)Cl "
          "CC(C)NCC(COc1ccccc1CC=C)O ClC1=CC=C(C=C1)C(C1=CC=CC=C1)N1C=CN=C1 "
          "CC12CCC3C(CCc4cc(O)ccc34)C1CCC2O").split()
CAUSAL_MIN       = 0.20                          # guard 2: correlation of the two models' shifts


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
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    G = Xc.T @ Xc + lam * torch.eye(Xc.shape[1])
    return torch.linalg.solve(G, Xc.T @ Yc)


def r2(X, Y, W):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    return float(1 - ((Yc - Xc @ W) ** 2).sum() / max(float((Yc ** 2).sum()), 1e-12))


def linear_cka(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    return float((Xc.T @ Yc).norm() ** 2 / max(float((Xc.T @ Xc).norm() * (Yc.T @ Yc).norm()), 1e-12))


# ------------------------------------------------------------------ guard 2: patching

class Patch:
    """Replace the residual stream at `layer` with supplied activations."""
    def __init__(self, model, layer, value):
        self.block = model.gpt_neox.layers[layer - 1]
        self.value = value
    def __enter__(self):
        val = self.value
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = val.to(h.device, h.dtype).expand_as(h).clone()
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        self.h = self.block.register_forward_hook(hook); return self
    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def shift(model, ids, layer, value):
    """Mean change in the next-token distribution when `value` is patched in at `layer`."""
    b = ids.to(DEV)
    clean = torch.softmax(model(b).logits.float(), -1).mean((0, 1))
    with Patch(model, layer, value):
        pat = torch.softmax(model(b).logits.float(), -1).mean((0, 1))
    return (pat - clean).cpu()


@torch.no_grad()
def causal_agreement(model_A, model_B, ids, donor_ids, W, mu_A, mu_B, n_donor=8):
    """Patch a donor state into A; push the same state through W and patch it into B.

    Both models share a tokenizer, so the two next-token shifts live in the same vocabulary.

    The shifts must be CENTRED ACROSS DONORS before they are compared. An earlier version
    correlated the raw shifts and scored a trained model reading random tokens at 0.70 --- higher
    than the real pair --- because any patch at all pushes both models toward the same common
    tokens, and that shared generic response dominates the correlation. Centring removes it and
    asks the question that matters: does it transfer *which* donor was patched, not merely *that*
    something was.

    Our own second guard therefore needed a vacuity control of its own, which is the paper's
    thesis applied to the paper.

    Returns (centred median correlation, raw median correlation, per-donor centred values).
    """
    sA, sB = [], []
    for k in range(n_donor):
        d = donor_ids[k:k+1].to(DEV)
        aH = model_A(d, output_hidden_states=True).hidden_states[LAYER_A].mean(1, keepdim=True)
        sA.append(shift(model_A, ids, LAYER_A, aH))
        bH = ((aH.float().cpu() - mu_A) @ W + mu_B).to(DEV)
        sB.append(shift(model_B, ids, LAYER_B, bH))
    SA, SB = torch.stack(sA), torch.stack(sB)

    def corr(x, y):
        x, y = x - x.mean(), y - y.mean()
        return float((x @ y) / max(float(x.norm() * y.norm()), 1e-12))

    raw = float(np.median([corr(SA[i], SB[i]) for i in range(n_donor)]))
    CA, CB = SA - SA.mean(0, keepdim=True), SB - SB.mean(0, keepdim=True)   # drop generic response
    cent = [corr(CA[i], CB[i]) for i in range(n_donor)]
    return float(np.median(cent)), raw, cent


# ------------------------------------------------------------------ sweep

def main():
    rng = np.random.default_rng(0)
    tok = AutoTokenizer.from_pretrained(MODEL_A)
    ids = get_text(tok, N_SEQ, SEQ_LEN)
    other = get_text(tok, N_SEQ, SEQ_LEN, skip=4000)          # disjoint text for `mismatched`
    n_fit = len(ids) // 2
    ids_fit, ids_held = ids[:n_fit], ids[n_fit:]

    A = AutoModelForCausalLM.from_pretrained(MODEL_A).to(DEV).eval()
    B = AutoModelForCausalLM.from_pretrained(MODEL_B).to(DEV).eval()
    B0 = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL_B)).to(DEV).eval()

    A_fit, A_held = acts(A, ids_fit, LAYER_A), acts(A, ids_held, LAYER_A)
    B_fit, B_held = acts(B, ids_fit, LAYER_B), acts(B, ids_held, LAYER_B)

    # A trained model reading uniformly random token IDs: real computation, real trained weights,
    # no semantic correspondence with A. Shares A's tokenizer, so guard 2 still applies.
    rand_ids = torch.tensor(rng.integers(0, tok.vocab_size, size=(len(ids), SEQ_LEN)))

    variants = {
        "real":       (B_fit, B_held, B),
        "untrained":  (acts(B0, ids_fit, LAYER_B), acts(B0, ids_held, LAYER_B), B0),
        "mismatched": (acts(B, other[:n_fit], LAYER_B), acts(B, other[n_fit:], LAYER_B), B),
        "randtok":    (acts(B, rand_ids[:n_fit], LAYER_B), acts(B, rand_ids[n_fit:], LAYER_B), B),
        "noise":      (torch.tensor(rng.normal(scale=float(B_fit.std()), size=tuple(B_fit.shape)),
                                    dtype=torch.float32),
                       torch.tensor(rng.normal(scale=float(B_fit.std()), size=tuple(B_held.shape)),
                                    dtype=torch.float32), B),
    }

    # Different domain entirely. Guard 2 is UNDEFINED here and we say so rather than inventing a
    # number: the causal test compares two models' next-token distributions, and a chemistry model
    # has no vocabulary in common with an English one. That the refutable guard needs a shared
    # output space to compare in is a real limitation of it, not an oversight.
    try:
        from transformers import AutoModel
        ctok = AutoTokenizer.from_pretrained(CHEM_MODEL)
        cmod = AutoModel.from_pretrained(CHEM_MODEL).to(DEV).eval()
        buf = []
        for smi in SMILES * 400:
            buf.extend(ctok(smi, add_special_tokens=False).input_ids)
        n_need = (len(ids)) * SEQ_LEN
        buf = (buf * (n_need // max(len(buf), 1) + 2))[: len(ids) * SEQ_LEN]
        cids = torch.tensor(buf).reshape(len(ids), SEQ_LEN)
        with torch.no_grad():
            cf = torch.cat([cmod(cids[i:i+24].to(DEV), output_hidden_states=True)
                            .hidden_states[3].reshape(-1, 768).float().cpu()
                            for i in range(0, n_fit, 24)])
            ch = torch.cat([cmod(cids[i:i+24].to(DEV), output_hidden_states=True)
                            .hidden_states[3].reshape(-1, 768).float().cpu()
                            for i in range(n_fit, len(ids), 24)])
        variants["chembert"] = (cf, ch, None)
    except Exception as e:
        print(f"  (different-domain control unavailable: {type(e).__name__} {str(e)[:60]})")

    print("X1 -- does a representational correspondence survive a causal test?")
    print(f"A = {MODEL_A} L{LAYER_A} (d=512)   B = {MODEL_B} L{LAYER_B} (d=768)")
    print(f"{len(ids)} x {SEQ_LEN} tokens; map fitted on {n_fit} sequences, scored on the other "
          f"{len(ids)-n_fit}\n")

    # --- guard 1, swept over how many stimuli the map is fitted on
    print("guard 1 (structural): alignment R^2, in-sample vs held out, by number of fitting stimuli")
    print(f"  {'n stimuli':>10}" + "".join(f"{v:>22}" for v in variants))
    print("  " + " " * 10 + "".join(f"{'in-samp / held-out':>22}" for _ in variants))
    print("  " + "-" * (10 + 22 * len(variants)))
    sweep = {v: {} for v in variants}
    for n in SWEEP:
        cells = []
        for v, (Bf, Bh, _) in variants.items():
            W = ridge_fit(A_fit[:n], Bf[:n])
            ins, hel = r2(A_fit[:n], Bf[:n], W), r2(A_held, Bh, W)
            sweep[v][n] = (ins, hel)
            cells.append(f"{ins:>10.2f} /{hel:>9.2f}")
        print(f"  {n:>10}" + "".join(f"{c:>22}" for c in cells))

    # --- both guards at full data
    print(f"\nboth guards, map fitted on all {len(A_fit)} token positions")
    print(f"  {'candidate':<12}{'R2 held':>9}{'CKA':>8}{'guard 1':>9}"
          f"{'causal (raw)':>18}{'guard 2':>9}   verdict")
    print("  " + "-" * 72)
    rows = {}
    mu_A = A_fit.mean(0, keepdim=True)
    for v, (Bf, Bh, model_int) in variants.items():
        W = ridge_fit(A_fit, Bf)
        R, C = r2(A_held, Bh, W), linear_cka(A_held, Bh)
        g1 = (R > R2_MIN) or (C > CKA_MIN)
        if model_int is None:
            cc, raw, allc, g2 = None, None, [], None   # no shared output vocabulary to compare in
        else:
            cc, raw, allc = causal_agreement(A, model_int, ids_held[:8], ids_held[8:], W,
                                             mu_A, Bf.mean(0, keepdim=True))
            g2 = cc > CAUSAL_MIN
        acc = bool(g1 and g2) if g2 is not None else None
        rows[v] = dict(r2=R, cka=C, guard1=bool(g1), causal=cc,
                       causal_raw=(None if cc is None else raw), causal_all=allc,
                       guard2=(None if g2 is None else bool(g2)), accepted=acc,
                       sweep={str(k): val for k, val in sweep[v].items()})
        cs = "               n/a" if cc is None else f"{cc:>10.3f} ({raw:+.2f})"
        gs = "  n/a" if g2 is None else ("PASS" if g2 else "fail")
        vs = "guard 2 n/a" if acc is None else ("ACCEPT" if acc else "reject")
        print(f"  {v:<12}{R:>9.3f}{C:>8.3f}{'PASS' if g1 else 'fail':>9}{cs}{gs:>9}   {vs}")

    V = tuple(v for v in ("untrained", "mismatched", "randtok", "noise", "chembert") if v in rows)
    V2 = tuple(v for v in V if rows[v]["accepted"] is not None)
    a1 = sum(rows[v]["guard1"] for v in V) / len(V)
    a2 = sum(rows[v]["accepted"] for v in V2) / len(V2)
    a1_small = sum(sweep[v][SWEEP[0]][0] > R2_MIN for v in V) / len(V)   # in-sample, few stimuli
    print(f"\n  vacuity class V = {V}")
    print(f"  alpha(guard 1, in-sample, {SWEEP[0]} stimuli) = {a1_small:.2f}")
    print(f"  alpha(guard 1, held out, all stimuli)      = {a1:.2f}")
    print(f"  alpha(both guards, over {V2})   = {a2:.2f}")
    print(f"  real case accepted = {rows['real']['accepted']}")

    Path(__file__).with_name("xai_cka.json").write_text(json.dumps(dict(
        rows=rows, alpha_structural_heldout=a1, alpha_structural_insample_small=a1_small,
        alpha_both=a2, sweep_n=SWEEP,
        config=dict(model_a=MODEL_A, model_b=MODEL_B, layer_a=LAYER_A, layer_b=LAYER_B,
                    n_seq=N_SEQ, seq_len=SEQ_LEN, ridge=RIDGE,
                    r2_min=R2_MIN, cka_min=CKA_MIN, causal_min=CAUSAL_MIN)), indent=1))


if __name__ == "__main__":
    main()
