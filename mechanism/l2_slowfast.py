"""A system that DOES have a coarse level, so the mechanism can be shown finding one.

On the lattice (`l2_coarse.py`) both guards are satisfied only by the identity: that system has
no coarse description, which is a real answer but only shows the mechanism saying no. A test that
can only fail is not a test.

Here the separation is explicit and dialled by one knob. Slow variables x evolve on O(1) times;
fast variables y relax on O(eps) times and are driven by x:

    dx/dt = A x  -  x |x|^2  +  c y            the slow layer, weakly disturbed by the fast one
    dy/dt = -(1/eps) y  +  B x                 the fast layer, slaved to the slow one

As eps -> 0 the fast layer follows x instantaneously, y -> eps B x, and the slow layer closes on
its own. As eps -> 1 the two are comparable and no coarse description exists. Sweeping eps walks
the mechanism from a system that has a coarse level to one that does not, and the point at which
it stops accepting is the thing being measured.
"""
from __future__ import annotations

import numpy as np

NS, NF_ = 4, 8                     # slow variables, fast variables
DIM = NS + NF_


def make(eps, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=0.4, size=(NS, NS)); A = 0.5 * (A - A.T)      # antisymmetric: oscillates
    B = rng.normal(scale=0.6, size=(NF_, NS))
    c = rng.normal(scale=0.5, size=(NS, NF_))

    def rhs(z):
        x, y = z[:NS], z[NS:]
        dx = A @ x - 0.10 * x * (x @ x) + eps * (c @ y)
        dy = -(1.0 / eps) * y + B @ x
        return np.concatenate([dx, dy])

    def jac0():
        J = np.zeros((DIM, DIM))
        J[:NS, :NS] = A
        J[:NS, NS:] = eps * c
        J[NS:, :NS] = B
        J[NS:, NS:] = -(1.0 / eps) * np.eye(NF_)
        return J

    return rhs, jac0


def traj(rhs, z0, steps, dt):
    out = [z0.copy()]; z = z0.copy()
    for _ in range(steps):
        k1 = rhs(z); k2 = rhs(z + .5*dt*k1); k3 = rhs(z + .5*dt*k2); k4 = rhs(z + dt*k3)
        z = z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        out.append(z.copy())
    return np.array(out)


def slow_subspace(J, keep):
    """The proposer: eigen-directions of the linearisation with the smallest |eigenvalue|."""
    w, V = np.linalg.eig(J)
    o = np.argsort(np.abs(w))
    Q, _ = np.linalg.qr(np.real_if_close(V[:, o[:keep]]).astype(complex))
    return Q, np.abs(w)[o]


def closes(rhs, Q, z0s, steps, dt):
    """Guard 1 -- do the coarse variables predict their own future without the fine state?"""
    errs = []
    for z0 in z0s:
        T = traj(rhs, z0, steps, dt)
        C = (Q.conj().T @ T.T).T
        c, pred = C[0].copy(), [C[0].copy()]
        def f(cc):
            return Q.conj().T @ rhs(np.real(Q @ cc))
        for _ in range(len(C) - 1):
            k1=f(c); k2=f(c+.5*dt*k1); k3=f(c+.5*dt*k2); k4=f(c+dt*k3)
            c = c + dt/6*(k1+2*k2+2*k3+k4); pred.append(c.copy())
        pred = np.array(pred)
        errs.append(np.linalg.norm(pred-C) / max(np.linalg.norm(C - C.mean(0)), 1e-12))
    return float(np.median(errs))


def observables(z):
    """What an experiment reports: the slow layer, and the total energy."""
    return np.concatenate([z[:NS], [float(z @ z)]])


def pays(rhs, Q, z0s, steps, dt):
    """Guard 2 -- does the coarse state still carry the fine observables? Fit-free."""
    Y, Yh = [], []
    for z0 in z0s:
        T = traj(rhs, z0, steps, dt)
        for z in T:
            Y.append(observables(z))
            Yh.append(observables(np.real(Q @ (Q.conj().T @ z))))
    Y, Yh = np.array(Y), np.array(Yh)
    den = float(np.sum((Y - Y.mean(0))**2))
    return float(np.clip(1 - float(np.sum((Y-Yh)**2)) / max(den, 1e-12), -1, 1))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    z0s = [rng.normal(size=DIM) for _ in range(6)]
    STEPS, DT = 400, 5e-3
    CLOSE_MAX, PAY_MIN = 0.25, 0.50

    print(f"{NS} slow variables + {NF_} fast = {DIM}. The true coarse description is the slow {NS}.\n")
    print(f"  {'eps':>6}{'sep':>8}{'keep':>6}{'closes':>10}{'pays':>8}   verdict")
    print("  " + "-"*54)
    for eps in (0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00):
        rhs, jac0 = make(eps)
        J = jac0()
        Q, mag = slow_subspace(J, NS)
        sep = mag[NS] / max(mag[NS-1], 1e-12)
        ce = closes(rhs, Q, z0s, STEPS, DT)
        pv = pays(rhs, Q, z0s, STEPS, DT)
        ok = ce < CLOSE_MAX and pv > PAY_MIN
        print(f"  {eps:>6.2f}{sep:>8.1f}{NS:>6}{ce:>10.3f}{pv:>8.2f}   "
              f"{'ACCEPT' if ok else ('closes, does not pay' if ce<CLOSE_MAX else 'does not close')}")

    print("\n  vacuity controls at eps = 0.02, where a coarse level certainly exists")
    rhs, jac0 = make(0.02); J = jac0()
    for lab, Q in (("constant", np.zeros((DIM,1))),
                   ("random subspace", np.linalg.qr(np.random.default_rng(5).normal(size=(DIM,NS)))[0]),
                   ("FAST modes, not slow", slow_subspace(J, DIM)[0][:, -NS:])):
        Q = Q.astype(complex)
        ce = closes(rhs, Q, z0s, STEPS, DT); pv = pays(rhs, Q, z0s, STEPS, DT)
        ok = ce < CLOSE_MAX and pv > PAY_MIN
        print(f"  {lab:<22}{ce:>10.3f}{pv:>8.2f}   {'ACCEPT  <-- WRONG' if ok else 'refused'}")
