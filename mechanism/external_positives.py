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
from l2_slowfast import (NS, DIM, make, traj, slow_subspace, closes, pays)   # noqa: E402

CLOSE_MAX, PAY_MIN = 0.25, 0.50          # as in l2_slowfast and l2_coarse
STEPS, DT = 400, 5e-3
SEEDS = (0, 1, 2, 3)
EPS_LABELLED = (0.02, 0.05)              # slaving guaranteed here; these carry the labels
EPS_SWEEP = (0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00)
THETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)  # radians, rotating the true subspace into the fast one


def rotate(Q_slow, Q_fast, theta):
    """Tilt the true coarse subspace toward the fast one by angle theta, then re-orthonormalise."""
    M = np.cos(theta) * Q_slow + np.sin(theta) * Q_fast[:, :Q_slow.shape[1]]
    Q, _ = np.linalg.qr(M)
    return Q[:, :Q_slow.shape[1]]


def score(rhs, Q, z0s):
    ce = closes(rhs, Q, z0s, STEPS, DT)
    pv = pays(rhs, Q, z0s, STEPS, DT)
    return float(ce), float(pv), bool(ce < CLOSE_MAX and pv > PAY_MIN)


def candidates(J, rng):
    Q_slow, mag = slow_subspace(J, NS)
    Q_fast = slow_subspace(J, DIM)[0][:, -NS:]
    return {
        "true slow subspace": (Q_slow, +1),
        "fast modes":         (Q_fast, -1),
        "random subspace":    (np.linalg.qr(rng.normal(size=(DIM, NS)))[0], -1),
        "constant":           (np.zeros((DIM, 1)), -1),
    }, Q_slow, Q_fast, mag


def main():
    rng = np.random.default_rng(0)
    z0s = [np.random.default_rng(100 + i).normal(size=DIM) for i in range(6)]

    print("Are the L2 guards sensitive as well as specific?")
    print(f"{NS} slow + {DIM-NS} fast variables; the true coarse description is the slow {NS}.")
    print(f"Guards and thresholds imported from l2_slowfast: closes < {CLOSE_MAX}, pays > {PAY_MIN}\n")

    # ---------------------------------------------------------------- labelled suite
    rows = []
    print("labelled suite -- eps where the construction guarantees a coarse level")
    print(f"  {'eps':>5}{'seed':>5}  {'candidate':<20}{'truth':>6}{'closes':>9}{'pays':>8}{'verdict':>9}")
    print("  " + "-" * 64)
    for eps in EPS_LABELLED:
        for seed in SEEDS:
            rhs, jac0 = make(eps, seed=seed)
            cands, _, _, _ = candidates(jac0(), rng)
            for name, (Q, truth) in cands.items():
                ce, pv, acc = score(rhs, Q, z0s)
                rows.append(dict(eps=eps, seed=seed, candidate=name, truth=truth,
                                 closes=ce, pays=pv, accepted=acc))
                print(f"  {eps:>5.2f}{seed:>5}  {name:<20}{'+' if truth>0 else '-':>6}"
                      f"{ce:>9.3f}{pv:>8.2f}{'ACCEPT' if acc else 'reject':>9}")

    tp = sum(1 for r in rows if r["truth"] > 0 and r["accepted"])
    fp = sum(1 for r in rows if r["truth"] < 0 and r["accepted"])
    fn = sum(1 for r in rows if r["truth"] > 0 and not r["accepted"])
    tn = sum(1 for r in rows if r["truth"] < 0 and not r["accepted"])
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"\n  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"  precision {prec:.2f}   recall {rec:.2f}   "
          f"(alpha over the negatives = {fp / max(fp + tn, 1):.2f})")

    # ---------------------------------------------------------------- graded degradation
    print("\nperturbing the true subspace toward the fast one (eps = 0.02, seed 0)")
    print(f"  {'theta':>7}{'closes':>10}{'pays':>8}{'verdict':>9}")
    print("  " + "-" * 36)
    rhs, jac0 = make(0.02, seed=0)
    _, Q_slow, Q_fast, _ = candidates(jac0(), rng)
    graded = []
    for th in THETAS:
        ce, pv, acc = score(rhs, rotate(Q_slow, Q_fast, th), z0s)
        graded.append(dict(theta=th, closes=ce, pays=pv, accepted=acc))
        print(f"  {th:>7.2f}{ce:>10.3f}{pv:>8.2f}{'ACCEPT' if acc else 'reject':>9}")
    broke = next((g["theta"] for g in graded if not g["accepted"]), None)
    print(f"  -> acceptance breaks down at theta = {broke}" if broke is not None
          else "  -> accepted at every perturbation tested, which would be a failure of the guard")

    # ---------------------------------------------------------------- separation sweep
    print("\nthe true subspace across the separation sweep (seed 0)")
    print(f"  {'eps':>6}{'separation':>12}{'closes':>9}{'pays':>8}{'verdict':>9}")
    print("  " + "-" * 46)
    sweep = []
    for eps in EPS_SWEEP:
        rhs, jac0 = make(eps, seed=0)
        Q, mag = slow_subspace(jac0(), NS)
        sep = float(mag[NS] / max(mag[NS - 1], 1e-12))
        ce, pv, acc = score(rhs, Q, z0s)
        sweep.append(dict(eps=eps, separation=sep, closes=ce, pays=pv, accepted=acc))
        print(f"  {eps:>6.2f}{sep:>12.1f}{ce:>9.3f}{pv:>8.2f}{'ACCEPT' if acc else 'reject':>9}")

    Path(__file__).with_name("external_positives.json").write_text(json.dumps(dict(
        labelled=rows, graded=graded, sweep=sweep,
        precision=prec, recall=rec, tp=tp, fp=fp, fn=fn, tn=tn,
        breakdown_theta=broke,
        config=dict(close_max=CLOSE_MAX, pay_min=PAY_MIN, steps=STEPS, dt=DT,
                    seeds=list(SEEDS), eps_labelled=list(EPS_LABELLED),
                    eps_sweep=list(EPS_SWEEP), thetas=list(THETAS))), indent=1))


if __name__ == "__main__":
    main()
