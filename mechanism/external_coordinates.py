"""Is the constant reduction's closure a fact about closure, or about our coordinates?

Round-3 review, section 2.2. Our external result says the reduction that maps every state to zero
has closure error exactly 0 on 124 of 129 systems. If that depends on the fine-state origin being
special --- on Q^dagger q0 landing on an invariant state --- then the test is measuring projected
trajectory reconstruction, not closure, and it should be renamed.

The conceptual null is Q(x) = q0 for all x. Its coarse trajectory is constant and therefore
autonomous, and closure must not care whether x = 0 happens to be a fixed point. Three variants:

    origin      the shipped test: Q = 0, lifted back to the fine-state origin
    translated  the same after moving the coordinate origin to a random point far off the attractor
    intrinsic   a zero-dimensional reduction whose coarse dynamics is defined directly as dc/dt = 0,
                with no lift to any fine state at all

If all three close, the result is about closure. If only the first does, we rename the test.
"""
import json, sys, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from l2_coarse import closure_error, CLOSE_MAX          # noqa: E402
import dysts.flows as F                                  # noqa: E402
import inspect                                           # noqa: E402

N_SYS, N_POINTS = 40, 200
SEED = 0


def trajectory(sys_obj, n_pts):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sol = sys_obj.make_trajectory(n_pts, return_times=True)
    if sol is None:
        return None, None
    t, x = sol
    x = np.asarray(x, float)
    if x.ndim != 2 or len(x) < n_pts // 2 or not np.all(np.isfinite(x)):
        return None, None
    return x, float(np.median(np.diff(np.asarray(t, float).ravel())))


def closes_constant(rhs, X, dt, shift):
    """Closure of Q = 0 in coordinates y = x - shift, so the origin is no longer distinguished."""
    d = X.shape[1]
    Q = np.zeros((d, 1))
    Y = X - shift
    C = Y @ Q                                   # coarse readings: identically zero
    f = lambda c: Q.T @ rhs(Q @ c + shift)      # push the fine law through, lifting back properly
    return float(closure_error(f, C[0], C, dt, 1))


def main():
    rng = np.random.default_rng(SEED)
    names = sorted(n for n, o in inspect.getmembers(F, inspect.isclass)
                   if o.__module__.startswith("dysts"))
    rows = []
    print(__doc__.strip().split("\n")[0]); print("=" * 78)
    print(f"\n  {'system':<24}{'origin':>10}{'translated':>13}{'intrinsic':>11}")
    print("  " + "-" * 58)
    for name in names:
        if len(rows) >= N_SYS:
            break
        try:
            s = getattr(F, name)()
            X, dt = trajectory(s, N_POINTS)
            if X is None or dt is None or not np.isfinite(dt) or dt <= 0:
                continue
            rhs = lambda x: np.asarray(s.rhs(x, 0.0), float)
            scale = float(np.abs(X).max())
            e_org = closes_constant(rhs, X, dt, np.zeros(X.shape[1]))
            shift = rng.normal(scale=3.0 * max(scale, 1.0), size=X.shape[1])
            e_tra = closes_constant(rhs, X, dt, shift)
            # intrinsic: the coarse state is a constant and dc/dt = 0 by definition, no lift
            C = np.zeros((len(X), 1))
            e_int = float(closure_error(lambda c: np.zeros(1), C[0], C, dt, 1))
            rows.append(dict(system=name, origin=e_org, translated=e_tra, intrinsic=e_int))
            if len(rows) <= 12:
                print(f"  {name:<24}{e_org:>10.3f}{e_tra:>13.3f}{e_int:>11.3f}")
        except Exception:
            continue

    def frac(k): return sum(1 for r in rows if r[k] < CLOSE_MAX) / max(len(rows), 1)
    print(f"  {'...':<24}\n")
    print(f"  systems tested: {len(rows)}")
    for k, lbl in (("origin", "constant reduction, original coordinates"),
                   ("translated", "constant reduction, origin moved off the attractor"),
                   ("intrinsic", "zero-dimensional reduction, no lift at all")):
        n = sum(1 for r in rows if r[k] < CLOSE_MAX)
        print(f"    {lbl:<52}{n:>3}/{len(rows)}  ({frac(k)*100:.0f}%)")

    import math
    fin = [r for r in rows if math.isfinite(r["origin"]) and math.isfinite(r["translated"])]
    same = all(abs(r["origin"] - r["translated"]) < 1e-9 for r in fin)
    print(f"\n  closure identical under translation, where both are finite: {len(fin)}/{len(fin)}"
          if same else "\n  closure CHANGES under translation")
    ok = same and frac("translated") > 0.9 and frac("intrinsic") > 0.9
    print("  -> the test measures closure, not a property of our coordinates."
          if ok else
          "  -> closure DEPENDS on the coordinate origin; the test must be renamed.")
    Path(__file__).with_name("external_coordinates.json").write_text(json.dumps(dict(
        rows=rows, frac_origin=frac("origin"), frac_translated=frac("translated"),
        frac_intrinsic=frac("intrinsic"), translation_invariant=bool(same),
        config=dict(n_sys=len(rows), n_points=N_POINTS, close_max=CLOSE_MAX, seed=SEED)), indent=1))


if __name__ == "__main__":
    main()
