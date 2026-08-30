"""L2, coarse-graining: find the level at which a description closes.

The largest rung in the corpus, 501 events, and the dominant real move inside it is a change of
level: gravity as an effective field theory, oscillators reduced to phase variables, gauge theory
made renormalisable. A description at the coarse level is not a summary of the fine one --- it is
a theory in its own right, if it CLOSES.

    L1 asks what PARAMETER detail predictions do not need.
    This asks what STATE detail the future does not need.

Two guards, both required, because either alone is empty:

    1. IT CLOSES   the coarse variables predict their own future without the fine state
    2. IT PAYS     the coarse variables still forecast held-out FINE observables

Guard 2 exists because a constant reduction closes perfectly: map every state to zero and the
coarse dynamics is trivially autonomous. L3 taught this the hard way --- there the structure test
alone accepted a false transfer five times in six, and only the refutable claim separated them.

Charge loss is REPORTED, not blocked. Discarding state detail can destroy a conserved quantity
while still closing, which is the coarse-grained shadow of the D1 bug. Physics drops symmetries
it does not need on purpose, so refusing would be wrong --- but saying nothing would repeat D1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # gauge.py ships alongside
import gauge as G                                                    # noqa: E402

DIM = G.L * G.L * G.NF          # 72 complex amplitudes on a 6x6 lattice with 2 flavours


def flat(phi):
    return phi.reshape(-1)


def unflat(v):
    return v.reshape(G.L, G.L, G.NF)


# --------------------------------------------------------------- the proposer: slow modes

def linear_operator(U):
    """The linear part of i dphi/dt, as a matrix, built by applying it to basis vectors.

    Plane waves are NOT eigenmodes here: the gauge links are random, so momentum is not a good
    quantum number. Diagonalising the operator that the system actually has avoids assuming an
    answer the lattice does not have.
    """
    M = np.zeros((DIM, DIM), dtype=complex)
    for i in range(DIM):
        e = np.zeros(DIM, dtype=complex); e[i] = 1.0
        z = unflat(e)
        out = np.zeros_like(z)
        for mu, ax in ((0, 0), (1, 1)):
            out -= np.roll(z, -1, axis=ax) * U[:, :, mu, None]
            out -= np.roll(z * U[:, :, mu, None].conj(), 1, axis=ax)
        out += G.M2 * z                              # the |phi|^2 term is nonlinear; excluded here
        M[:, i] = flat(out)
    return M


def slow_subspace(U, keep=None, gap_min=1.5):
    """Eigen-decompose, sort by |frequency|, and cut where the spectrum leaves a real gap."""
    M = linear_operator(U)
    w, V = np.linalg.eig(M)
    order = np.argsort(np.abs(w))
    w, V = w[order], V[:, order]
    mag = np.abs(w)
    if keep is None:                                  # cut at the largest ratio in the low half
        best, keep = 0.0, DIM // 2
        for k in range(4, DIM):
            if mag[k - 1] > 1e-9:
                r = mag[k] / mag[k - 1]
                if r > best and r >= gap_min:
                    best, keep = r, k
    Q, _ = np.linalg.qr(V[:, :keep])                  # orthonormal basis for the slow subspace
    return Q, w, keep


# --------------------------------------------------------------- trajectories

def rhs(phi, U, lam=None):
    """i dphi/dt = dH/dphi*, the full nonlinear law. Same physics as gauge.py, exposed here so
    the nonlinearity can be dialled for the control sweep."""
    lam = G.LAM if lam is None else lam
    out = np.zeros_like(phi)
    for mu, ax in ((0, 0), (1, 1)):
        out -= np.roll(phi, -1, axis=ax) * U[:, :, mu, None]
        out -= np.roll(phi * U[:, :, mu, None].conj(), 1, axis=ax)
    n = np.einsum("xya,xya->xy", phi.conj(), phi).real
    return (out + G.M2 * phi + 2 * lam * n[:, :, None] * phi) / 1j


def trajectory(phi, U, steps, dt, lam=None, every=1):
    """RK4, recording the state every `every` steps."""
    out = [phi.copy()]
    for t in range(steps):
        k1 = rhs(phi, U, lam); k2 = rhs(phi + .5*dt*k1, U, lam)
        k3 = rhs(phi + .5*dt*k2, U, lam); k4 = rhs(phi + dt*k3, U, lam)
        phi = phi + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        if (t + 1) % every == 0:
            out.append(phi.copy())
    return np.array(out)


def project(traj, Q):
    return np.array([Q.conj().T @ flat(p) for p in traj])


# --------------------------------------------------------------- guard 1: does it close?

def derived_law(Q, U, lam):
    """Push the known fine law through the reduction: assume the fast part is absent."""
    def f(c):
        return Q.conj().T @ flat(rhs(unflat(Q @ c), U, lam))
    return f


def learn_law(C, dC):
    """Fit a coarse law from coarse readings alone, never seeing the fine state.

    Features are linear plus the cubic |c|^2 c that the fine law's shape suggests. If a closed
    coarse description exists this should find it; if the fit fails and the derived law succeeds,
    the failure was ours and not the physics.
    """
    def feat(c):
        n = np.abs(c) ** 2
        return np.concatenate([c, n * c])
    X = np.array([feat(c) for c in C])
    A, *_ = np.linalg.lstsq(X, dC, rcond=None)
    return lambda c: A.T @ feat(c)


def closure_error(f, c0, C_true, dt, every):
    """Integrate the coarse law alone and see how far it drifts from the truth."""
    c, pred = c0.copy(), [c0.copy()]
    h = dt * every
    for _ in range(len(C_true) - 1):
        k1 = f(c); k2 = f(c + .5*h*k1); k3 = f(c + .5*h*k2); k4 = f(c + h*k3)
        c = c + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        pred.append(c.copy())
    pred = np.array(pred)
    num = np.linalg.norm(pred - C_true)
    den = np.linalg.norm(C_true - C_true.mean(0))
    return float(num / max(den, 1e-12))


# --------------------------------------------------------------- guard 2: does it pay?

def pays(Q, U, lam, seeds_fit, seeds_test, steps, dt, every):
    """How much of the fine observables does the coarse state still carry?

    Reconstruct the fine state from the coarse one --- phi_hat = Q Q* phi --- and compare the
    observables. No fitting anywhere, deliberately: three fitted versions of this guard failed,
    and every failure was the fitter rather than the physics. One scored a perfect in-sample R2
    at keep=6 with 37 features on 62 samples; another was dominated by the scale spread between
    random initial states. A reconstruction test has no such freedom and carries a hard
    invariant: a reduction that keeps everything must score exactly 1.

        pays = 1 - || probes(Q Q* phi) - probes(phi) ||^2 / || probes(phi) - mean ||^2
    """
    # The denominator must be the spread of observable values across the ENSEMBLE of states.
    # Measured within one trajectory it is almost zero --- the density is conserved, so the
    # observables barely move --- and every truncation then scores minus infinity.
    Y, Yh = [], []
    for s in seeds_test:
        phi, _ = G.random_state(seed=s)
        tr = trajectory(phi, U, steps, dt, lam, every)
        Y.extend(G.probes((p, U)) for p in tr)
        Yh.extend(G.probes((unflat(Q @ (Q.conj().T @ flat(p))), U)) for p in tr)
    Y, Yh = np.array(Y), np.array(Yh)
    num = float(np.sum((Y - Yh) ** 2))
    den = float(np.sum((Y - Y.mean(0)) ** 2))
    return float(np.clip(1.0 - num / max(den, 1e-12), -1.0, 1.0))


# --------------------------------------------------------------- the operation

STEPS, DT, EVERY = 240, 2e-3, 8
CLOSE_MAX, PAY_MIN = 0.25, 0.50


def evaluate(Q, U, lam, label, n_traj=6):
    """Run both guards on one candidate reduction."""
    seeds = list(range(n_traj))
    C_all, dC_all, first = [], [], None
    for s in seeds:
        phi, _ = G.random_state(seed=s)
        tr = trajectory(phi, U, STEPS, DT, lam, EVERY)
        C = project(tr, Q)
        dC = np.array([Q.conj().T @ flat(rhs(unflat(Q @ c), U, lam)) for c in C])
        C_all.append(C); dC_all.append(dC)
        if first is None:
            first = C
    C_fit = np.vstack(C_all); dC_fit = np.vstack(dC_all)

    e_derived = closure_error(derived_law(Q, U, lam), first[0], first, DT, EVERY)
    e_learned = closure_error(learn_law(C_fit, dC_fit), first[0], first, DT, EVERY)
    r2 = pays(Q, U, lam, list(range(30, 42)), list(range(50, 56)), STEPS, DT, EVERY)

    closes = min(e_derived, e_learned) < CLOSE_MAX
    ok = closes and r2 > PAY_MIN
    print(f"  {label:<26}{Q.shape[1]:>4}{e_derived:>11.3f}{e_learned:>11.3f}{r2:>9.2f}   "
          f"{'ACCEPT' if ok else ('closes, does not pay' if closes else 'does not close')}")
    return ok, e_derived, e_learned, r2


def charges_kept(Q, U, lam, tol=1e-3):
    """Which of the fine system's conserved charges survive the reduction? Reported, not blocked."""
    T = {"flavour I": np.eye(2, dtype=complex),
         "flavour sz": np.array([[1, 0], [0, -1]], dtype=complex)}
    kept, lost = [], []
    for name, gen in T.items():
        drift = []
        for s in (0, 1):
            phi, _ = G.random_state(seed=s)
            tr = trajectory(phi, U, STEPS, DT, lam, EVERY)
            coarse = np.array([unflat(Q @ (Q.conj().T @ flat(p))) for p in tr])   # fine -> coarse -> fine
            q = [G.charge(p, gen) for p in coarse]
            drift.append(abs(q[-1] - q[0]) / max(abs(q[0]), 1e-12))
        (kept if np.median(drift) < tol else lost).append((name, float(np.median(drift))))
    return kept, lost


if __name__ == "__main__":
    phi0, U = G.random_state(seed=0)
    Q, w, keep = slow_subspace(U)
    mag = np.sort(np.abs(w))
    print("L2 -- is there a level at which this description closes?\n")
    print(f"lattice {G.L}x{G.L}, {G.NF} flavours -> {DIM} complex amplitudes")
    print(f"  spectral cut: keep {keep} of {DIM}  (compression {DIM/keep:.1f}x, "
          f"gap {mag[keep]/mag[keep-1]:.2f}x)\n")

    print(f"  {'reduction':<26}{'dim':>4}{'derived':>11}{'learned':>11}{'pays R2':>9}   verdict")
    print("  " + "-" * 78)
    evaluate(Q, U, G.LAM, "spectral, slow modes")

    # vacuity controls: both may close; only guard 2 can tell them apart
    Qc = np.zeros((DIM, 1), dtype=complex)
    evaluate(Qc, U, G.LAM, "CONTROL constant (all zero)")
    rng = np.random.default_rng(3)
    Qr, _ = np.linalg.qr(rng.normal(size=(DIM, keep)) + 1j*rng.normal(size=(DIM, keep)))
    evaluate(Qr, U, G.LAM, "CONTROL random subspace")

    print(f"\n  sensitivity: how far can the nonlinearity be pushed before closure fails?")
    print(f"  {'LAM':<26}{'dim':>4}{'derived':>11}{'learned':>11}{'pays R2':>9}   verdict")
    print("  " + "-" * 78)
    for lam in (0.0, 0.2, 0.5, 1.0, 2.0, 4.0):
        evaluate(Q, U, lam, f"nonlinearity {lam:.1f}")

    kept, lost = charges_kept(Q, U, G.LAM)
    print(f"\n  charges under the reduction (reported, not blocked):")
    for n, d in kept: print(f"    kept  {n:<12} drift {d:.1e}")
    for n, d in lost: print(f"    LOST  {n:<12} drift {d:.1e}")
