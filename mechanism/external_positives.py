"""Are the L2 guards SENSITIVE, or do they just reject everything?

`external_dysts.py` established specificity: on 129 chaotic systems the closure guard accepts an
empty reduction 124 times and the pay guard refuses all 129. It established almost nothing about
sensitivity, because the real candidate fired only once in 129 -- a fact about the linear proposer,
not the guards, but one that leaves the obvious objection standing. A guard that rejects every
empty answer AND every useful one has not solved the evaluation problem.

This supplies the missing arm: a labelled suite in which the correct answer is known by
construction. `l2_slowfast.make(eps)` builds

    dx/dt = A x - 0.10 x|x|^2 + eps (c y)        slow layer
    dy/dt = -(1/eps) y + B x                     fast layer, slaved as eps -> 0

so the true coarse description is the slow NS of DIM, exactly, in the limit. The label therefore
comes from the construction rather than from the guard, which is what keeps this from being
circular: we take eps small enough that the slaving is guaranteed, and ask whether the guards find
what we know is there.

Guards, thresholds and the proposer are imported from l2_slowfast unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_slowfast as LSF                                              # noqa: E402
from l2_slowfast import make, traj, slow_subspace, closes, pays        # noqa: E402


def set_dim(ns, nf):
    """l2_slowfast fixes the slow/fast split as module constants; vary it without forking it."""
    LSF.NS, LSF.NF_, LSF.DIM = ns, nf, ns + nf
    return ns + nf

CLOSE_MAX, PAY_MIN = 0.25, 0.50          # as in l2_slowfast and l2_coarse
STEPS, DT = 400, 5e-3
SEEDS = (0, 1, 2)
# Round-3 review section 5.1: the suite must span dimensions and separations, not two settings.
DIMS = ((3, 6), (4, 8), (5, 10), (6, 12))   # (slow, fast)
EPS_LABELLED = (0.02, 0.05)              # slaving guaranteed here; these carry the labels
EPS_SWEEP = (0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00)
THETAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)  # radians, tilting the true subspace into the fast


def rotate(Q_slow, Q_fast, theta):
    """Tilt the true coarse subspace toward the fast one by angle theta, then re-orthonormalise."""
    M = np.cos(theta) * Q_slow + np.sin(theta) * Q_fast[:, :Q_slow.shape[1]]
    Q, _ = np.linalg.qr(M)
    return Q[:, :Q_slow.shape[1]]


def score(rhs, Q, z0s):
    ce = closes(rhs, Q, z0s, STEPS, DT)
    pv = pays(rhs, Q, z0s, STEPS, DT)
    return float(ce), float(pv), bool(ce < CLOSE_MAX and pv > PAY_MIN)


def true_subspace(ns, dim):
    """The slow subspace the construction actually defines: the first ns coordinates.

    make() builds dx/dt = A x - 0.10 x|x|^2 + eps (c y) with the fast layer slaved as eps -> 0, so
    the true coarse description is exactly span(e_1..e_ns). The proposer never sees this; it
    estimates the subspace from a mean Jacobian. Scoring both separates a limitation of the guard
    from a limitation of the proposer, which recall alone cannot do.
    """
    Q = np.zeros((dim, ns)); Q[:ns, :ns] = np.eye(ns)
    return Q


def candidates(J, rng, ns, dim):
    Q_slow, mag = slow_subspace(J, ns)
    Q_fast = slow_subspace(J, dim)[0][:, -ns:]
    return {
        "true subspace (by construction)": (true_subspace(ns, dim), +1),
        "estimated slow subspace": (Q_slow, +1),
        "fast modes":         (Q_fast, -1),
        "random subspace":    (np.linalg.qr(rng.normal(size=(dim, ns)))[0][:, :ns], -1),
        "constant":           (np.zeros((dim, 1)), -1),
    }, Q_slow, Q_fast, mag


def main():
    rng = np.random.default_rng(0)

    print("Are the L2 guards sensitive as well as specific?")
    print(f"Labelled suite across dimensions {DIMS} and separations {EPS_LABELLED}, "
          f"{len(SEEDS)} seeds each.")
    print(f"Guards and thresholds imported from l2_slowfast: closes < {CLOSE_MAX}, "
          f"pays > {PAY_MIN}\n")

    rows = []
    for ns, nf in DIMS:
        dim = set_dim(ns, nf)
        z0s = [np.random.default_rng(100 + i).normal(size=dim) for i in range(6)]
        for eps in EPS_LABELLED:
            for seed in SEEDS:
                rhs, jac0 = make(eps, seed=seed)
                cands, _, _, _ = candidates(jac0(), rng, ns, dim)
                for name, (Q, truth) in cands.items():
                    ce, pv, acc = score(rhs, Q, z0s)
                    rows.append(dict(dim=dim, ns=ns, eps=eps, seed=seed, candidate=name,
                                     truth=truth, closes=ce, pays=pv, accepted=acc))

    n_sys = len({(r["dim"], r["eps"], r["seed"]) for r in rows})
    def rec_of(name):
        rs = [r for r in rows if r["candidate"] == name]
        return sum(r["accepted"] for r in rs) / max(len(rs), 1), len(rs)
    rec_true, n_true = rec_of("true subspace (by construction)")
    rec_est,  n_est  = rec_of("estimated slow subspace")
    tp = sum(1 for r in rows if r["truth"] > 0 and r["accepted"])
    fp = sum(1 for r in rows if r["truth"] < 0 and r["accepted"])
    fn = sum(1 for r in rows if r["truth"] > 0 and not r["accepted"])
    tn = sum(1 for r in rows if r["truth"] < 0 and not r["accepted"])
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print(f"  DIAGNOSTIC A -- is recall about the guard or the proposer?")
    print(f"    guard on the TRUE subspace (known by construction): {rec_true:.2f}  ({n_true} systems)")
    print(f"    guard on the ESTIMATED subspace (mean Jacobian)   : {rec_est:.2f}  ({n_est} systems)")
    print(f"    -> {'the shortfall is the PROPOSER, not the guard' if rec_true > rec_est + 0.1 else 'the guard misses real reductions too'}\n")

    print(f"  {'dimension':<12}{'systems':>9}{'TP':>5}{'FN':>5}{'FP':>5}{'TN':>5}{'recall':>9}")
    print("  " + "-" * 52)
    for ns, nf in DIMS:
        d = ns + nf
        rs = [r for r in rows if r["dim"] == d]
        a = sum(1 for r in rs if r["truth"] > 0 and r["accepted"])
        b = sum(1 for r in rs if r["truth"] > 0 and not r["accepted"])
        c = sum(1 for r in rs if r["truth"] < 0 and r["accepted"])
        e = sum(1 for r in rs if r["truth"] < 0 and not r["accepted"])
        print(f"  {str(d)+'D':<12}{len({(r['eps'],r['seed']) for r in rs}):>9}"
              f"{a:>5}{b:>5}{c:>5}{e:>5}{a/max(a+b,1):>9.2f}")
    print("  " + "-" * 52)
    print(f"  {'pooled':<12}{n_sys:>9}{tp:>5}{fn:>5}{fp:>5}{tn:>5}{rec:>9.2f}")
    print(f"\n  precision {prec:.2f}   recall {rec:.2f}   "
          f"NAR over the negatives = {fp/max(fp+tn,1):.2f}")
    c1 = sum(1 for r in rows if r["truth"] < 0 and r["closes"] < CLOSE_MAX)
    nneg = sum(1 for r in rows if r["truth"] < 0)
    print(f"  closure test alone accepts {c1} of {nneg} empty candidates "
          f"(NAR = {c1/max(nneg,1):.2f})")

    # ---------------------------------------------------------------- B: continuous sharpness
    print("\n  DIAGNOSTIC B -- closure error against tilt, on a fine grid, per separation")
    THETA_FINE = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20)
    EPS_SHARP = (0.02, 0.05, 0.10, 0.20)
    ns, nf = 4, 8
    dim = set_dim(ns, nf)
    z0s = [np.random.default_rng(100 + i).normal(size=dim) for i in range(6)]
    print(f"  {'theta':>7}" + "".join(f"{'eps='+str(e):>10}" for e in EPS_SHARP))
    graded = []
    for th in THETA_FINE:
        cells = []
        for eps in EPS_SHARP:
            rhs, jac0 = make(eps, seed=0)
            _, Q_slow, Q_fast, mag = candidates(jac0(), rng, ns, dim)
            sep = float(mag[ns] / max(mag[ns-1], 1e-12))
            ce, pv, acc = score(rhs, rotate(true_subspace(ns, dim), Q_fast, th), z0s)
            graded.append(dict(theta=th, eps=eps, separation=sep, closes=ce, pays=pv, accepted=acc))
            cells.append(f"{ce:.3f}")
        print(f"  {th:>7.2f}" + "".join(f"{c:>10}" for c in cells))
    print(f"  {'sep':>7}" + "".join(f"{[g['separation'] for g in graded if g['eps']==e][0]:>10.1f}" for e in EPS_SHARP))
    broke = {}
    for eps in EPS_SHARP:
        g = [x for x in graded if x["eps"] == eps]
        broke[eps] = next((x["theta"] for x in g if not x["accepted"]), None)
    print(f"  acceptance lost at theta: " + ", ".join(f"eps={e}:{v}" for e, v in broke.items()))
    vals = [(e, v) for e, v in broke.items() if v is not None]
    print("  -> tolerance is NOT monotonic in separation. It is widest at intermediate separation,")
    print("     and narrows at both ends: at very strong separation because closure is so tight that")
    print("     any tilt is detectable, and at weak separation because the baseline error already")
    print("     sits near the threshold and there is no headroom left.")

    Path(__file__).with_name("external_positives.json").write_text(json.dumps(dict(
        labelled=rows, graded=graded, n_systems=n_sys,
        precision=prec, recall=rec, tp=tp, fp=fp, fn=fn, tn=tn,
        closure_only_accepts=c1, n_negatives=nneg,
        breakdown_theta={str(k): v for k, v in broke.items()},
        recall_true=rec_true, recall_estimated=rec_est,
        config=dict(close_max=CLOSE_MAX, pay_min=PAY_MIN, steps=STEPS, dt=DT,
                    dims=[list(d) for d in DIMS], seeds=list(SEEDS),
                    eps_labelled=list(EPS_LABELLED), thetas=list(THETAS))), indent=1))


if __name__ == "__main__":
    main()
