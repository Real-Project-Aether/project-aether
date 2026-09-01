"""The L2 guards on 137 dynamical systems we did not build.

Requires the benchmark: `pip install dysts`. This is the only script here with a dependency
outside numpy/scipy, which is why verify.py does not run it; its results ship as
external_dysts.json and external_dysts_out.txt.

`l2_coarse.py` found that guard 1 --- "the coarse variables close" --- accepts reductions that
describe nothing: on our lattice, two of three empty controls closed perfectly and only guard 2
refused them. Three controls is not evidence. This runs the same two guards over the `dysts`
benchmark (Gilpin 2021), ~137 chaotic systems with published equations, none of them ours.

For each system the fastest direction is dropped (d -> d-1, the mildest possible reduction) and
four candidates are scored. The cut keeps half the directions (k = d//2), the same kind of
aggressive compression the lattice used (6 of 72); keeping d-1 would leave the slow and fast
controls overlapping in the many three-dimensional systems here.


    slow      the d-1 slowest eigen-directions of the mean Jacobian   -- the real reduction
    fast      the d-1 fastest instead                                 -- control
    random    a random orthonormal subspace of the same size          -- control
    constant  the zero map                                            -- control

The three controls are reductions to nothing in particular. A guard worth having refuses them.

What is imported from l2_coarse and NOT restated here: `closure_error`, `pay_score`, and the two
thresholds `CLOSE_MAX` / `PAY_MIN`. That is the point of the exercise --- a guard that has to be
retuned per system is not a guard. Only the pieces that cannot be shared are written here: these
systems are real-valued and mostly quadratic, where the lattice is complex and cubic.
"""
import sys, os, warnings, contextlib, io, inspect
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))   # l2_coarse.py ships alongside
from l2_coarse import closure_error, pay_score, CLOSE_MAX, PAY_MIN   # noqa: E402

import dysts.flows as F                                              # noqa: E402

N_POINTS = 200        # trajectory length used for the closure test
N_ENSEMBLE = 6        # initial conditions pooled for the pay test's denominator

# How many directions the reduction keeps. Default is an aggressive halving, matching the
# lattice's 6-of-72 cut. K_MODE=mild keeps d-1, the gentlest possible reduction, and is used to
# check whether "the real reduction rarely closes" is a fact about these systems or about our cut.
K_MODE = os.environ.get("K_MODE", "half")
KEEP_RULE = (lambda d: max(1, d - 1)) if K_MODE == "mild" else (lambda d: max(1, d // 2))
SEED = 0


# ------------------------------------------------------------------ the system under test

def trajectories(sys_obj, n_ic, n_pts, rng):
    """Integrate n_ic trajectories from perturbed initial conditions. Returns (list, dt)."""
    ic0 = np.asarray(sys_obj.ic, float)
    out, dt = [], None
    for i in range(n_ic):
        sys_obj.ic = ic0 * (1.0 + 0.05 * rng.normal(size=ic0.shape)) if i else ic0
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sol = sys_obj.make_trajectory(n_pts, return_times=True)
        if sol is None:
            continue
        t, x = sol
        x = np.asarray(x, float)
        if x.ndim != 2 or len(x) < n_pts // 2 or not np.all(np.isfinite(x)):
            continue
        out.append(x)
        if dt is None:
            dt = float(np.median(np.diff(np.asarray(t, float).ravel())))
    sys_obj.ic = ic0
    return out, dt


def mean_jacobian(sys_obj, X):
    """Average the analytic Jacobian along a trajectory: the linear operator this system looks
    like on average. The lattice proposer used an exact linear operator; this is its analogue."""
    d = X.shape[1]

    def jac_at(x):
        try:
            J = np.asarray(sys_obj.jac(x, 0.0), float)
            if J.shape == (d, d) and np.all(np.isfinite(J)):
                return J
        except (NotImplementedError, Exception):
            pass
        # Not every system in the benchmark ships an analytic Jacobian. Finite differences on the
        # published rhs are the same operator, so those systems stay in the sweep.
        h = 1e-6 * (1.0 + np.abs(x))
        J = np.empty((d, d))
        f0 = np.asarray(sys_obj.rhs(x, 0.0), float)
        for j in range(d):
            xp = x.copy(); xp[j] += h[j]
            J[:, j] = (np.asarray(sys_obj.rhs(xp, 0.0), float) - f0) / h[j]
        return J if np.all(np.isfinite(J)) else None

    Js = [J for J in (jac_at(x) for x in X[:: max(1, len(X) // 40)]) if J is not None]
    return np.mean(Js, axis=0) if Js else None


def eig_subspace(J, k, slowest=True):
    """Orthonormal real basis for the k slowest (or fastest) eigen-directions of J.

    |lambda| is the rate, matching `slow_subspace` in l2_coarse. Eigenvectors of a real
    non-symmetric J come in complex conjugate pairs and are not orthogonal, so the selected
    directions are split into real and imaginary parts and orthonormalised --- QQ^T must be a
    genuine projector for the pay guard's reconstruction to mean anything.
    """
    w, V = np.linalg.eig(J)
    order = np.argsort(np.abs(w))
    idx = order[:k] if slowest else order[::-1][:k]
    cols = []
    for i in idx:
        cols.append(V[:, i].real)
        if abs(V[:, i].imag).max() > 1e-12:
            cols.append(V[:, i].imag)
    B = np.array(cols).T
    Q, _ = np.linalg.qr(B)
    return Q[:, :k]


# ------------------------------------------------------------------ the coarse law

def learn_law(C, dC):
    """Fit a coarse law from coarse readings alone. Linear plus quadratic monomials --- the
    real-valued analogue of l2_coarse's [c, |c|^2 c], chosen for the polynomial vector fields
    in this benchmark, not tuned against the results."""
    def feat(c):
        return np.concatenate([[1.0], c, np.outer(c, c)[np.triu_indices(len(c))]])
    X = np.array([feat(c) for c in C])
    A, *_ = np.linalg.lstsq(X, dC, rcond=None)
    return lambda c: A.T @ feat(c)


def probes(X):
    """Observables of a fine state: the coordinates and their pairwise products. No standardising
    --- pay_score's denominator is the ensemble spread, which does that job."""
    return np.array([np.concatenate([x, np.outer(x, x)[np.triu_indices(len(x))]]) for x in X])


def evaluate(sys_obj, Q, X_list, dt):
    """Both guards on one candidate reduction.

    Two corrections over the first version, both of which made the guard easier than it should be.
    The learned coarse law was fitted on every trajectory including the one it was then evaluated
    on; it is now fitted on all but the first and evaluated on the first, which no trajectory in
    the fit has seen. And acceptance took min(derived, learned), so the oracle-derived law could
    rescue a failed fit --- for an automated-discovery claim the LEARNED law must carry the guard,
    with the derived law kept only as a diagnostic separating a bad proposer from a bad fit.
    """
    rhs = lambda x: np.asarray(sys_obj.rhs(x, 0.0), float)

    C_all, dC_all = [], []
    for X in X_list:
        C = X @ Q                                     # coarse readings
        R = C @ Q.T                                   # the coarse observer's best guess at x
        dC = np.array([Q.T @ rhs(r) for r in R])
        C_all.append(C); dC_all.append(dC)
    C0 = C_all[0]                                     # held out from the fit
    if len(C_all) > 1:
        C_fit = np.vstack(C_all[1:]); dC_fit = np.vstack(dC_all[1:])
    else:
        C_fit, dC_fit = C_all[0], dC_all[0]           # degenerate case, reported as such

    derived = lambda c: Q.T @ rhs(Q @ c)
    try:
        e_derived = float(closure_error(derived, C0[0], C0, dt, 1))
    except Exception:
        e_derived = np.inf
    try:
        e_learned = float(closure_error(learn_law(C_fit, dC_fit), C0[0], C0, dt, 1))
    except Exception:
        e_learned = np.inf
    e_derived = e_derived if np.isfinite(e_derived) else np.inf
    e_learned = e_learned if np.isfinite(e_learned) else np.inf

    Xs = np.vstack(X_list)
    r2 = pay_score(probes(Xs), probes(Xs @ Q @ Q.T))

    closes = e_learned < CLOSE_MAX                    # the learned law carries the guard
    return closes, (r2 > PAY_MIN), e_derived, e_learned, r2


# ------------------------------------------------------------------ the sweep

def main():
    names = sorted(n for n, o in inspect.getmembers(F, inspect.isclass)
                   if o.__module__.startswith("dysts"))
    rng = np.random.default_rng(SEED)
    rows, skipped = [], []

    print("The L2 guards on systems we did not build")
    print(f"dysts benchmark, {len(names)} candidate systems; keep rule K_MODE={K_MODE}")
    print(f"thresholds imported unchanged from l2_coarse: closes < {CLOSE_MAX}, pays > {PAY_MIN}\n")

    for name in names:
        try:
            s = getattr(F, name)()
            X_list, dt = trajectories(s, N_ENSEMBLE, N_POINTS, rng)
            if len(X_list) < 2 or dt is None or not np.isfinite(dt) or dt <= 0:
                skipped.append((name, "integration failed")); continue
            d = X_list[0].shape[1]
            if d < 3:
                skipped.append((name, f"dim {d} too small")); continue
            J = mean_jacobian(s, X_list[0])
            if J is None or J.shape != (d, d):
                skipped.append((name, "no jacobian")); continue
            k = KEEP_RULE(d)        # a real compression, matching the lattice's aggressive cut
            cands = {
                "slow":     eig_subspace(J, k, slowest=True),
                "fast":     eig_subspace(J, k, slowest=False),
                "random":   np.linalg.qr(rng.normal(size=(d, k)))[0][:, :k],
                "constant": np.zeros((d, 1)),
            }
            res = {c: evaluate(s, Q, X_list, dt) for c, Q in cands.items()}
            rows.append((name, d, res))
            print(f"  {name:<26} d={d}  " + "  ".join(
                f"{c}:{'C' if res[c][0] else '-'}{'P' if res[c][1] else '-'}"
                for c in ("slow", "fast", "random", "constant")), flush=True)
        except Exception as e:
            skipped.append((name, type(e).__name__))

    # ---------------------------------------------------------------- report
    print(f"\nran on {len(rows)} systems; skipped {len(skipped)}\n")
    print(f"  {'candidate':<12}{'closes':>12}{'closes+pays':>14}   what guard 2 caught")
    print("  " + "-" * 70)
    summary = {}
    for c in ("slow", "fast", "random", "constant"):
        n_close = sum(1 for _, _, r in rows if r[c][0])
        n_both = sum(1 for _, _, r in rows if r[c][0] and r[c][1])
        summary[c] = (n_close, n_both, n_close - n_both)
        pct = 100.0 * n_close / max(len(rows), 1)
        print(f"  {c:<12}{n_close:>6} ({pct:3.0f}%){n_both:>14}   "
              f"{n_close - n_both} of {n_close} refused by the pay guard")

    n = len(rows)
    empt_close = sum(summary[c][0] for c in ("fast", "random", "constant"))
    empt_both = sum(summary[c][1] for c in ("fast", "random", "constant"))
    print(f"\n  empty reductions accepted by guard 1 alone: {empt_close} of {3*n}"
          f"  ({100.0*empt_close/max(3*n,1):.0f}%)")
    print(f"  empty reductions surviving both guards:     {empt_both} of {3*n}"
          f"  ({100.0*empt_both/max(3*n,1):.0f}%)")

    import json
    out = [{"system": nm, "dim": d,
            "res": {c: {"closes": bool(r[c][0]), "pays": bool(r[c][1]),
                        "derived": r[c][2], "learned": r[c][3], "r2": r[c][4]}
                    for c in r}} for nm, d, r in rows]
    (Path(__file__).parent / f"external_dysts_{K_MODE}.json").write_text(json.dumps(out, indent=1))
    if skipped:
        print("\n  skipped: " + ", ".join(f"{n_}({w})" for n_, w in skipped[:12])
              + (" ..." if len(skipped) > 12 else ""))


if __name__ == "__main__":
    main()
