"""Null FAMILIES, not isolated nulls.

A suite of four or six hand-written nulls answers a weak question: did these particular candidates
get through? The stronger question is what a whole family of correspondence-breaking candidates
does, so that NAR is a rate with an interval and AnyNullPass is a statement about many draws rather
than one. This runs three families over many draws, for one language-model pair and one protein
pair, using the guards imported from the X1 and X3 modules.

    permuted_pairs    a fresh permutation of the source/target pairing per draw
    orthogonal        a fresh random rotation of the target space per draw
    gaussian_matched  a fresh covariance-matched Gaussian target per draw

Guard 1 is scored on every draw, which is cheap. Guard 2 is scored on every draw too, at a reduced
number of interventions, because the question that matters -- whether ANY member of a family gets
past the pair -- cannot be answered by scoring one member.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xai_cka as X1                                                    # noqa: E402
import torch                                                            # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM            # noqa: E402

warnings.filterwarnings("ignore")
DEV = X1.DEV
N_DRAW = 24                 # draws per family
N_DONOR, N_POS, N_RECIP = 12, 8, 16


def families(A_fit, B_fit, B_held, g):
    """Yield (family, draw index, effective map, means) for every draw."""
    W_real, muX, muY = X1.ridge_fit(A_fit, B_fit)
    dB = B_fit.shape[1]
    Bc = B_fit - B_fit.mean(0, keepdim=True)
    cov = (Bc.T @ Bc) / max(len(Bc) - 1, 1)
    L = torch.linalg.cholesky(cov.double() + 1e-3 * torch.eye(dB, dtype=torch.float64)).float()
    for i in range(N_DRAW):
        perm = torch.randperm(len(B_fit), generator=g)
        W, mx, my = X1.ridge_fit(A_fit, B_fit[perm])
        yield "permuted_pairs", i, W, mx, my
    for i in range(N_DRAW):
        Q, _ = torch.linalg.qr(torch.randn(dB, dB, generator=g))
        yield "orthogonal", i, W_real @ Q, muX, muY
    for i in range(N_DRAW):
        Y = torch.randn(len(B_fit), dB, generator=g) @ L.T + B_fit.mean(0, keepdim=True)
        W, mx, my = X1.ridge_fit(A_fit, Y)
        yield "gaussian_matched", i, W, mx, my


def run(tag, A, B, LA, LB, ids_fit, ids_held, acts, hidden_at, clean_dist, added_shift, log):
    A_fit, A_held = acts(A, ids_fit, LA), acts(A, ids_held, LA)
    B_fit, B_held = acts(B, ids_fit, LB), acts(B, ids_held, LB)
    W_real, muX, muY = X1.ridge_fit(A_fit, B_fit)
    real = dict(r2=X1.r2(A_held, B_held, W_real, muX, muY), cka=X1.linear_cka(A_held, B_held))

    recip, donors = ids_held[:N_RECIP], ids_held[N_RECIP:]
    rs = np.random.default_rng(0)
    positions = sorted(rs.choice(np.arange(8, ids_held.shape[1] - 2), N_POS, replace=False).tolist())
    di = rs.choice(len(donors), N_DONOR, replace=False)
    dpos = rs.integers(8, ids_held.shape[1] - 2, N_DONOR)
    with torch.no_grad():
        dstates = torch.stack([
            A(donors[di[k]:di[k]+1].to(DEV), output_hidden_states=True)
             .hidden_states[LA][0, int(dpos[k])].float().cpu() for k in range(N_DONOR)])

    # A's side is identical for every candidate, so it is computed once
    per_pos = {}
    for pos in positions:
        HA = hidden_at(A, recip, LA, pos)
        cleanA = clean_dist(A, recip, pos)
        dlt = [dstates[k].unsqueeze(0) - HA for k in range(N_DONOR)]
        sA = torch.stack([added_shift(A, recip, LA, pos, dlt[k], cleanA) for k in range(N_DONOR)])
        per_pos[pos] = (dlt, sA - sA.mean(0, keepdim=True), clean_dist(B, recip, pos))

    g = torch.Generator().manual_seed(0)
    rows = []
    for fam, i, W, mx, my in families(A_fit, B_fit, B_held, g):
        sc = []
        for pos in positions:
            dlt, cA, cleanB = per_pos[pos]
            mapped = [d @ W for d in dlt]
            sB = torch.stack([added_shift(B, recip, LB, pos, mapped[k], cleanB)
                              for k in range(N_DONOR)])
            cB = sB - sB.mean(0, keepdim=True)
            sc += [X1.corr(cA[k], cB[k]) for k in range(N_DONOR)]
        g1 = bool(X1.r2(A_held, B_held, W, mx, my) > X1.R2_MIN
                  or X1.linear_cka(A_held, B_held) > X1.CKA_MIN)
        med = float(np.median(sc))
        rows.append(dict(family=fam, draw=i, r2=X1.r2(A_held, B_held, W, mx, my),
                         cka=X1.linear_cka(A_held, B_held), guard1=g1,
                         causal_median=med, guard2=bool(med > X1.CAUSAL_MIN),
                         accepted=bool(g1 and med > X1.CAUSAL_MIN), n=len(sc)))
        if (i + 1) % 8 == 0:
            log(f"    {fam}: {i+1}/{N_DRAW} draws")
    return real, rows


def summarise(tag, real, rows, out):
    fams = sorted({r["family"] for r in rows})
    print(f"\n[{tag}]  real pair: $R^2$ {real['r2']:.3f}, CKA {real['cka']:.3f}")
    print(f"  {'family':<20}{'draws':>7}{'NAR g1':>9}{'NAR both':>10}{'worst causal':>14}")
    for f in fams:
        q = [r for r in rows if r["family"] == f]
        n1 = sum(r["guard1"] for r in q) / len(q)
        n2 = sum(r["accepted"] for r in q) / len(q)
        print(f"  {f:<20}{len(q):>7}{n1:>9.2f}{n2:>10.2f}{max(r['causal_median'] for r in q):>14.3f}")
    n1 = sum(r["guard1"] for r in rows) / len(rows)
    n2 = sum(r["accepted"] for r in rows) / len(rows)
    print(f"  {'ALL':<20}{len(rows):>7}{n1:>9.2f}{n2:>10.2f}"
          f"{max(r['causal_median'] for r in rows):>14.3f}")
    print(f"  AnyNullPass over {len(rows)} draws = {int(any(r['accepted'] for r in rows))}")
    out[tag] = dict(real=real, rows=rows, n_draws=len(rows),
                    nar_guard1=n1, nar_both=n2,
                    any_null_pass=int(any(r["accepted"] for r in rows)),
                    worst_causal=max(r["causal_median"] for r in rows),
                    per_family={f: dict(
                        n=len([r for r in rows if r["family"] == f]),
                        nar_guard1=sum(r["guard1"] for r in rows if r["family"] == f)
                            / len([r for r in rows if r["family"] == f]),
                        nar_both=sum(r["accepted"] for r in rows if r["family"] == f)
                            / len([r for r in rows if r["family"] == f]),
                        worst_causal=max(r["causal_median"] for r in rows if r["family"] == f))
                        for f in fams})


def main():
    log = lambda m: (print(m), sys.stdout.flush())
    out = {}
    print(f"Null families: {N_DRAW} draws each of three correspondence-breaking families, "
          f"{N_POS*N_DONOR} interventions per draw\n")

    # --- language models, using X1's own helpers
    print("[pythia-70m L4 -> pythia-160m L8]"); sys.stdout.flush()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    ids = X1.get_text(tok, 240, 128)
    A = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m").to(DEV).eval()
    B = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m").to(DEV).eval()
    real, rows = run("pythia-70m L4 -> pythia-160m L8", A, B, 4, 8, ids[:120], ids[120:],
                     X1.acts, X1.hidden_at, X1.clean_dist, X1.added_shift, log)
    summarise("pythia-70m L4 -> pythia-160m L8", real, rows, out)
    del A, B; torch.cuda.empty_cache()

    # --- protein models, using X3's helpers
    import xai_esm as X3
    print("\n[esm2_t6_8M L3 -> esm2_t12_35M L6]"); sys.stdout.flush()
    ptok = AutoTokenizer.from_pretrained(X3.T6)
    pids = X3.get_proteins(ptok, 320, 128)
    from transformers import EsmForMaskedLM
    PA = EsmForMaskedLM.from_pretrained(X3.T6).to(DEV).eval()
    PB = EsmForMaskedLM.from_pretrained(X3.T12).to(DEV).eval()
    real, rows = run("esm2_t6 L3 -> esm2_t12 L6", PA, PB, 3, 6, pids[:160], pids[160:],
                     X3.acts, X3.hidden_at, X3.clean_dist, X3.added_shift, log)
    summarise("esm2_t6 L3 -> esm2_t12 L6", real, rows, out)

    if not out:
        print("\nNO PAIR COMPLETED -- leaving the existing result file untouched")
        sys.exit(1)
    Path(__file__).with_name("xai_families.json").write_text(json.dumps(dict(
        pairs=out, n_draw_per_family=N_DRAW,
        config=dict(families=["permuted_pairs", "orthogonal", "gaussian_matched"],
                    n_donor=N_DONOR, n_pos=N_POS, n_recip=N_RECIP,
                    interventions_per_draw=N_POS * N_DONOR)), indent=1))


if __name__ == "__main__":
    main()
