"""X3 --- the correspondence audit on a scientific foundation model (ESM-2 protein LMs).

The same question as X1, asked of models trained on protein sequences rather than English: does a
representational correspondence between two ESM-2 scales survive a causal test, and do the
structural guards in common use refuse candidates that carry no correspondence?

Three things make this a cleaner test than the language-model version:

  * every ESM-2 scale shares one 33-token amino-acid vocabulary, so two models' output
    distributions are directly comparable without any alignment step;
  * ESM-2 is a masked LM, so the prediction at a residue is a claim about THAT residue, and the
    consequence test reads the effect where the intervention was made;
  * UniRef50 entries are cluster representatives at 50% sequence identity, so held-out proteins
    are below 50% identity to the fitting proteins BY CONSTRUCTION -- the homology split a
    protein-model evaluation needs, without our having to build it.

Guards, both imported from `xai_cka` rather than reimplemented, so this is the same audit:

    guard 1, STRUCTURAL   linear alignment R^2 and linear CKA on residue representations.
    guard 2, CAUSAL       difference patching at one residue: perturb model A's residual stream at
                          position t, carry the same perturbation through W into model B, and
                          compare the induced changes in the two models' amino-acid logits at t.

Candidates are typed. Nulls must be refused; positives must be accepted, which is what makes NAR
interpretable -- a guard that refuses everything scores NAR 0 and is useless:

    identity          POSITIVE   the same model and layer, W = I. Cannot fail unless the
                                 harness is broken, which is what it is there to check.
    adjacent          POSITIVE   the same model, next layer. Must be accepted.
    real              the claim under test: a genuine correspondence between two scales
    permuted_pairs    NULL       W fitted on shuffled residue pairings: same marginals, no pairing
    residue_shuffle   NULL       residue positions permuted within each protein
    orthogonal        NULL       the real map followed by a random rotation of B's space
    gaussian_matched  NULL       W fitted to Gaussian targets carrying B's mean and covariance
    untrained         NULL       a randomly initialised target model
    donor_perm        CONTROL    the real map, but B receives a different donor's perturbation --
                                 a control on the consequence test, not a candidate for guard 1

Requires torch, transformers, datasets and a GPU. verify.py asserts against the shipped JSON.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np

# The guards themselves are imported, not rewritten. Importing this module also installs the
# torchvision shim this environment needs before transformers is loaded.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xai_cka import (ridge_fit, r2, linear_cka, corr, auroc,      # noqa: E402
                     R2_MIN, CKA_MIN, CAUSAL_MIN, RIDGE, SWEEP)

import torch                                                      # noqa: E402
from transformers import AutoTokenizer, EsmForMaskedLM, AutoConfig  # noqa: E402

warnings.filterwarnings("ignore")
torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

T6, T12, T30 = ("facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D",
                "facebook/esm2_t30_150M_UR50D")
CONFIGS = ((T6, 3, T12, 6), (T6, 4, T30, 15), (T12, 6, T30, 15))

N_PROT, SEQ_LEN = 320, 128      # residues per protein, after truncation
SEEDS, N_DONOR, N_POS, N_RECIP = 3, 12, 8, 16
POS_LO = 8                       # skip the first residues, whose context is one-sided


def get_proteins(tok, n, seq_len):
    """UniRef50 cluster representatives, truncated to a common length.

    Entries are representatives of clusters at 50% identity, so two different entries are below
    that threshold to each other. Splitting by entry therefore gives a homology-aware split.
    """
    from datasets import load_dataset
    ds = load_dataset("agemagician/uniref50", split="train", streaming=True)
    out = []
    for row in ds:
        s = row["text"].strip().upper()
        if len(s) < seq_len or any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in s[:seq_len]):
            continue
        out.append(s[:seq_len])
        if len(out) >= n:
            break
    enc = tok(out, return_tensors="pt", padding=False)
    return enc.input_ids            # (n, seq_len + 2), <cls> ... <eos>


@torch.no_grad()
def acts(model, ids, layer, batch=16):
    """Residue representations, special tokens dropped."""
    out = []
    for i in range(0, len(ids), batch):
        h = model(ids[i:i+batch].to(DEV), output_hidden_states=True).hidden_states[layer]
        out.append(h[:, 1:-1, :].reshape(-1, h.shape[-1]).float().cpu())
    return torch.cat(out)


class AddAt:
    """Add a per-recipient delta to the residual stream at ONE residue position."""
    def __init__(self, model, layer, pos, delta):
        self.block = model.esm.encoder.layer[layer - 1]
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
def clean_dist(model, ids, pos):
    return torch.softmax(model(ids.to(DEV)).logits[:, pos].float(), -1).mean(0).cpu()


@torch.no_grad()
def added_shift(model, ids, layer, pos, delta, clean):
    with AddAt(model, layer, pos, delta):
        p = torch.softmax(model(ids.to(DEV)).logits[:, pos].float(), -1).mean(0).cpu()
    return p - clean


@torch.no_grad()
def hidden_at(model, ids, layer, pos):
    h = model(ids.to(DEV), output_hidden_states=True).hidden_states[layer][:, pos, :]
    return h.float().cpu()


def run_config(A_name, LA, B_name, LB, log):
    tok = AutoTokenizer.from_pretrained(A_name)
    ids = get_proteins(tok, N_PROT, SEQ_LEN)
    n_fit = len(ids) // 2
    ids_fit, ids_held = ids[:n_fit], ids[n_fit:]
    log(f"    {len(ids)} UniRef50 proteins, {SEQ_LEN} residues each; "
        f"{n_fit} fit / {len(ids)-n_fit} held out")

    A = EsmForMaskedLM.from_pretrained(A_name).to(DEV).eval()
    B = EsmForMaskedLM.from_pretrained(B_name).to(DEV).eval()
    B0 = EsmForMaskedLM(AutoConfig.from_pretrained(B_name)).to(DEV).eval()

    A_fit, A_held = acts(A, ids_fit, LA), acts(A, ids_held, LA)
    B_fit, B_held = acts(B, ids_fit, LB), acts(B, ids_held, LB)
    B0_fit, B0_held = acts(B0, ids_fit, LB), acts(B0, ids_held, LB)
    Aadj_fit, Aadj_held = acts(A, ids_fit, LA + 1), acts(A, ids_held, LA + 1)

    g = torch.Generator().manual_seed(0)
    dB, dA = B_fit.shape[1], A_fit.shape[1]

    Bc = B_fit - B_fit.mean(0, keepdim=True)
    cov = (Bc.T @ Bc) / max(len(Bc) - 1, 1)
    L = torch.linalg.cholesky(cov.double() + 1e-3 * torch.eye(dB, dtype=torch.float64)).float()
    Bg_fit = torch.randn(len(B_fit), dB, generator=g) @ L.T + B_fit.mean(0, keepdim=True)
    Bg_held = torch.randn(len(B_held), dB, generator=g) @ L.T + B_fit.mean(0, keepdim=True)

    perm = torch.randperm(len(B_fit), generator=g)
    Q, _ = torch.linalg.qr(torch.randn(dB, dB, generator=g))

    # residues permuted WITHIN each protein: composition preserved, position correspondence gone
    R = SEQ_LEN
    def shuffle_within(X):
        Y = X.clone().reshape(-1, R, X.shape[1])
        for i in range(len(Y)):
            Y[i] = Y[i][torch.randperm(R, generator=g)]
        return Y.reshape(-1, X.shape[1])
    Bs_fit, Bs_held = shuffle_within(B_fit), shuffle_within(B_held)

    W_real, mx, my = ridge_fit(A_fit, B_fit)
    fits = {
        "identity":         (torch.eye(dA), A_fit.mean(0, keepdim=True), A_fit.mean(0, keepdim=True),
                             A_held, A, LA, 0, "positive"),
        "adjacent":         (*ridge_fit(A_fit, Aadj_fit), Aadj_held, A, LA + 1, 0, "positive"),
        "real":             (W_real, mx, my, B_held, B, LB, 0, "real"),
        "permuted_pairs":   (*ridge_fit(A_fit, B_fit[perm]), B_held, B, LB, 0,
                             "correspondence-breaking"),
        "residue_shuffle":  (*ridge_fit(A_fit, Bs_fit), B_held, B, LB, 0,
                             "correspondence-breaking"),
        "orthogonal":       (W_real @ Q, mx, my, B_held, B, LB, 0, "correspondence-breaking"),
        "gaussian_matched": (*ridge_fit(A_fit, Bg_fit), B_held, B, LB, 0, "randomised"),
        "untrained":        (*ridge_fit(A_fit, B0_fit), B0_held, B0, LB, 0, "degenerate"),
        "donor_perm":       (W_real, mx, my, B_held, B, LB, 1, "consequence-test control"),
    }

    tgt_fit = {"identity": A_fit, "adjacent": Aadj_fit, "real": B_fit,
               "permuted_pairs": B_fit[perm], "residue_shuffle": Bs_fit, "orthogonal": B_fit,
               "gaussian_matched": Bg_fit, "untrained": B0_fit, "donor_perm": B_fit}

    sweep, full = {}, {}
    for v, (Wv, mxv, myv, Th, mdl, lb, off, typ) in fits.items():
        sweep[v] = {}
        for n in SWEEP:
            if v == "identity":
                Wn, mxn, myn = torch.eye(dA), A_fit[:n].mean(0, keepdim=True), A_fit[:n].mean(0, keepdim=True)
            else:
                Wn, mxn, myn = ridge_fit(A_fit[:n], tgt_fit[v][:n])
                if v == "orthogonal":
                    Wn = Wn @ Q
            sweep[v][n] = (r2(A_fit[:n], tgt_fit[v][:n], Wn, mxn, myn),
                           r2(A_held, Th, Wn, mxn, myn))
        full[v] = dict(r2=r2(A_held, Th, Wv, mxv, myv), cka=linear_cka(A_held, Th))

    recip = ids_held[:N_RECIP]
    donors = ids_held[N_RECIP:]
    scores = {v: [] for v in fits}

    for seed in range(SEEDS):
        rs = np.random.default_rng(200 + seed)
        positions = sorted(rs.choice(np.arange(POS_LO, SEQ_LEN - 1), N_POS, replace=False).tolist())
        di = rs.choice(len(donors), N_DONOR, replace=False)
        dpos = rs.integers(POS_LO, SEQ_LEN - 1, N_DONOR)
        with torch.no_grad():
            dstates = torch.stack([
                A(donors[di[k]:di[k]+1].to(DEV), output_hidden_states=True)
                 .hidden_states[LA][0, int(dpos[k])].float().cpu() for k in range(N_DONOR)])

        for pos in positions:
            HA = hidden_at(A, recip, LA, pos)
            cleanA = clean_dist(A, recip, pos)
            dltA = [dstates[k].unsqueeze(0) - HA for k in range(N_DONOR)]
            sA = torch.stack([added_shift(A, recip, LA, pos, dltA[k], cleanA)
                              for k in range(N_DONOR)])
            cA = sA - sA.mean(0, keepdim=True)

            for v, (Wv, mxv, myv, Th, mdl, lb, off, typ) in fits.items():
                cleanB = clean_dist(mdl, recip, pos)
                mapped = [d @ Wv for d in dltA]
                if off:
                    mapped = mapped[off:] + mapped[:off]
                sB = torch.stack([added_shift(mdl, recip, lb, pos, mapped[k], cleanB)
                                  for k in range(N_DONOR)])
                cB = sB - sB.mean(0, keepdim=True)
                scores[v] += [corr(cA[k], cB[k]) for k in range(N_DONOR)]
        log(f"    seed {seed}: {len(scores['real'])} triples")

    del A, B, B0
    torch.cuda.empty_cache()

    out = {}
    for v, (Wv, mxv, myv, Th, mdl, lb, off, typ) in fits.items():
        s = np.array(scores[v], float)
        out[v] = dict(type=typ, r2=full[v]["r2"], cka=full[v]["cka"],
                      guard1=bool(full[v]["r2"] > R2_MIN or full[v]["cka"] > CKA_MIN),
                      causal_median=float(np.median(s)), n_triples=len(s),
                      guard2=bool(np.median(s) > CAUSAL_MIN),
                      auroc_vs_real=(None if v == "real" else auroc(scores["real"], scores[v])),
                      sweep={str(k): list(val) for k, val in sweep[v].items()})
        out[v]["accepted"] = bool(out[v]["guard1"] and out[v]["guard2"])
    return out


def main():
    print("X3 -- the correspondence audit on ESM-2 protein language models")
    print(f"{len(CONFIGS)} configurations; {SEEDS*N_POS*N_DONOR} interventions each; "
          f"UniRef50 held-out proteins are <50% identical to the fitting set by construction\n")
    results = {}
    for (A_name, LA, B_name, LB) in CONFIGS:
        key = f"{A_name.split('/')[-1]} L{LA} -> {B_name.split('/')[-1]} L{LB}"
        print(f"[{key}]"); sys.stdout.flush()
        try:
            rows = run_config(A_name, LA, B_name, LB,
                              lambda m: (print(m), sys.stdout.flush()))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {type(e).__name__}: {str(e)[:150]}"); continue
        results[key] = rows
        print(f"  {'candidate':<18}{'type':<26}{'R2':>7}{'CKA':>7}{'g1':>4}"
              f"{'causal':>9}{'AUROC':>8}{'g2':>4}  verdict")
        for v, r in rows.items():
            au = "   -- " if r["auroc_vs_real"] is None else f"{r['auroc_vs_real']:>6.3f}"
            print(f"  {v:<18}{r['type']:<26}{r['r2']:>7.3f}{r['cka']:>7.3f}"
                  f"{'P' if r['guard1'] else '.':>4}{r['causal_median']:>9.3f}{au:>8}"
                  f"{'P' if r['guard2'] else '.':>4}  {'ACCEPT' if r['accepted'] else 'reject'}")
        nulls = [v for v in rows if rows[v]["type"] not in ("real", "positive",
                                                           "consequence-test control")]
        pos = [v for v in rows if rows[v]["type"] == "positive"]
        print(f"  NAR(guard 1) = {sum(rows[v]['guard1'] for v in nulls)/len(nulls):.2f}   "
              f"NAR(both) = {sum(rows[v]['accepted'] for v in nulls)/len(nulls):.2f}   "
              f"PAR(both) = {sum(rows[v]['accepted'] for v in pos)/max(len(pos),1):.2f}   "
              f"real accepted = {rows['real']['accepted']}\n"); sys.stdout.flush()

    if not results:
        print("no configuration completed"); return
    allc = sorted({v for r in results.values() for v in r})
    nulls = [v for v in allc
             if next(r[v]["type"] for r in results.values() if v in r)
             not in ("real", "positive", "consequence-test control")]
    pos = [v for v in allc
           if next(r[v]["type"] for r in results.values() if v in r) == "positive"]
    pooled = {v: dict(
        guard1_rate=float(np.mean([r[v]["guard1"] for r in results.values() if v in r])),
        accept_rate=float(np.mean([r[v]["accepted"] for r in results.values() if v in r])),
        median_causal=float(np.median([r[v]["causal_median"] for r in results.values() if v in r])),
        median_auroc=(None if v == "real" else float(np.median(
            [r[v]["auroc_vs_real"] for r in results.values() if v in r]))))
        for v in allc}
    nar1 = float(np.mean([pooled[v]["guard1_rate"] for v in nulls]))
    nar2 = float(np.mean([pooled[v]["accept_rate"] for v in nulls]))
    par = float(np.mean([pooled[v]["accept_rate"] for v in pos]))
    small = float(np.mean([np.mean([r[v]["sweep"][str(SWEEP[0])][0] > R2_MIN
                                    for r in results.values() if v in r]) for v in nulls]))
    print(f"pooled over {len(results)} configurations")
    print(f"  NAR(guard 1, in-sample, {SWEEP[0]} stimuli) = {small:.2f}")
    print(f"  NAR(guard 1, held out)                   = {nar1:.2f}")
    print(f"  NAR(both guards)                         = {nar2:.2f}")
    print(f"  PAR(both guards, known positives)        = {par:.2f}")
    print(f"  real accepted in                         = "
          f"{sum(r['real']['accepted'] for r in results.values())}/{len(results)}")

    Path(__file__).with_name("xai_esm.json").write_text(json.dumps(dict(
        configs=results, pooled=pooled, nar_structural_heldout=nar1,
        nar_structural_insample_small=small, nar_both=nar2, par_both=par,
        any_null_pass=int(any(pooled[v]["accept_rate"] > 0 for v in nulls)),
        sweep_n=SWEEP,
        config=dict(configs=[list(c) for c in CONFIGS], n_prot=N_PROT, seq_len=SEQ_LEN,
                    ridge=RIDGE, seeds=SEEDS, n_donor=N_DONOR, n_pos=N_POS, n_recip=N_RECIP,
                    triples_per_config=SEEDS * N_POS * N_DONOR,
                    r2_min=R2_MIN, cka_min=CKA_MIN, causal_min=CAUSAL_MIN,
                    data="UniRef50 cluster representatives (<50% identity between entries)",
                    patch="difference patching at a single residue, read at that residue")),
        indent=1))


if __name__ == "__main__":
    main()
