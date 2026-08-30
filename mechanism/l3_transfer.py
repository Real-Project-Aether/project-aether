"""L3, transfer: carry a solved structure from one domain into another.

189 events in the corpus sit on this rung. Nambu took spontaneous symmetry breaking from
superconductivity into particle physics; Mukhanov, Starobinsky and Sunyaev took quantum
fluctuations onto cosmological scales. In each case a domain that already had an answer lent its
language to one that did not.

Operationally, two systems each with a state space, probes and dynamics. If some map carries B's
states into A's while preserving what the probes see, then what A knows transfers to B --- and
the transfer is CHECKABLE, because B's own dynamics can refute it.

The danger is vacuity: between any two state spaces some map exists. Two constraints stop that,
and both must hold:

    1. STRUCTURE, not bijection.  The map must make B's probes agree with A's, to within the
       measurement noise. A map that merely relabels states fails this.
    2. IT MUST PAY.  What transfers has to be a claim B did not already have, and B's dynamics
       must then bear it out. Here that claim is a conserved quantity: A knows a charge, and the
       map hands B a quantity that B's evolution either conserves or does not.

The control is the point. When B is built WITHOUT A's structure, step 1 must fail --- and if it
somehow does not, step 2 must.
"""
from __future__ import annotations

import numpy as np

NA, NB = 2, 3          # A carries 2 components; B is larger and hides A inside it
SITES = 24


# --------------------------------------------------------------------------- domain A (solved)

def a_state(rng):
    return rng.normal(size=(SITES, NA)) + 1j * rng.normal(size=(SITES, NA))


def a_probes(s):
    """What an experiment on A reports: total density and neighbour overlap. Flavour-blind."""
    dens = np.einsum("xa,xa->x", s.conj(), s).real
    hop = np.einsum("xa,xa->x", s.conj(), np.roll(s, -1, axis=0)).real
    return np.concatenate([dens, hop])


def a_evolve(s, steps=600, dt=2e-3):
    """i ds/dt = dH/ds*, with H flavour-blind, so any U(2) generator gives a conserved charge."""
    for _ in range(steps):
        def g(z):
            out = -np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)
            n = np.einsum("xa,xa->x", z.conj(), z).real
            return (out + 0.5 * z + 0.2 * n[:, None] * z) / 1j
        k1 = g(s); k2 = g(s + .5*dt*k1); k3 = g(s + .5*dt*k2); k4 = g(s + dt*k3)
        s = s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return s


A_GENERATOR = np.array([[1, 0], [0, -1]], dtype=complex)      # the charge A already knows


def charge(s, T):
    return float(np.einsum("xa,ab,xb->", s.conj(), T, s).real)


# --------------------------------------------------------------------------- domain B (target)

def make_B(shared: bool, seed=0):
    """B looks nothing like A. Whether it CONTAINS A is the thing to be discovered.

    shared=True : A's two components are embedded by an unknown mixing, plus one inert extra.
    shared=False: the control -- the components are coupled so no flavour rotation survives.
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(NB, NA)) + 1j * rng.normal(size=(NB, NA))     # the unknown embedding
    M /= np.linalg.norm(M, axis=0, keepdims=True)

    def state(r):
        core = a_state(r)                                   # the shared structure, if any
        extra = 0.4 * (r.normal(size=(SITES, 1)) + 1j * r.normal(size=(SITES, 1)))
        if shared:
            return core @ M.T + extra * np.array([[0.3, -0.2, 0.9]])
        # control: entangle the components so no 2-dimensional flavour subspace is invariant
        z = core @ M.T + extra * np.array([[0.3, -0.2, 0.9]])
        return z + 0.6 * np.roll(z.conj(), 1, axis=1) * np.abs(z)

    def probes(s):
        dens = np.einsum("xa,xa->x", s.conj(), s).real
        hop = np.einsum("xa,xa->x", s.conj(), np.roll(s, -1, axis=0)).real
        return np.concatenate([dens, hop])

    def evolve(s, steps=600, dt=2e-3):
        for _ in range(steps):
            def g(z):
                out = -np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)
                n = np.einsum("xa,xa->x", z.conj(), z).real
                h = (out + 0.5 * z + 0.2 * n[:, None] * z)
                if not shared:                              # break the flavour symmetry
                    h = h + 0.35 * np.roll(z, 1, axis=1) * np.abs(z)
                return h / 1j
            k1 = g(s); k2 = g(s + .5*dt*k1); k3 = g(s + .5*dt*k2); k4 = g(s + dt*k3)
            s = s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return s

    return state, probes, evolve, M


# --------------------------------------------------------------------------- the operation

def find_map(b_state, b_probes, n_samples=40, seed=1, restarts=6):
    """Search for a linear P: B-components -> A-components that makes B's probes look like A's.

    Structure-preserving, not merely invertible: what is fitted is the requirement that the
    probes agree. B is larger than A, so P is a projection, and the extra components of B must
    fall out of it rather than be absorbed.
    """
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    states = [b_state(np.random.default_rng(100 + i)) for i in range(n_samples)]
    targets = np.concatenate([b_probes(s) for s in states])

    def resid(v):
        P = (v[:NB*NA] + 1j*v[NB*NA:]).reshape(NB, NA)
        out = []
        for s in states:
            out.append(a_probes(s @ P))
        return np.concatenate(out) - targets

    best, best_cost = None, np.inf
    for r in range(restarts):
        v0 = rng.normal(scale=0.6, size=2*NB*NA)
        try:
            sol = least_squares(resid, v0, max_nfev=4000)
        except Exception:
            continue
        if sol.cost < best_cost:
            best, best_cost = sol.x, sol.cost
    P = (best[:NB*NA] + 1j*best[NB*NA:]).reshape(NB, NA)
    rel = np.sqrt(2*best_cost / max(np.sum(targets**2), 1e-12))
    return P, rel


def transfer_and_test(P, b_state, b_evolve, seed=7):
    """A knows a charge. Push it through the map and ask B's dynamics whether it holds."""
    Pinv = np.linalg.pinv(P)                      # B-components <- A-components
    T_B = Pinv.conj().T @ A_GENERATOR @ Pinv      # A's generator, expressed in B's coordinates
    drifts = []
    for k in range(4):
        s0 = b_state(np.random.default_rng(seed + k))
        s1 = b_evolve(s0)
        q0, q1 = charge(s0, T_B), charge(s1, T_B)
        drifts.append(abs(q1 - q0) / max(abs(q0), 1e-12))
    return float(np.median(drifts)), T_B


def run(shared, label, seed=0):
    b_state, b_probes, b_evolve, M_true = make_B(shared, seed=seed)
    P, rel = find_map(b_state, b_probes)
    structure_ok = rel < 0.05                                 # probes agree to within 5%
    drift, _ = transfer_and_test(P, b_state, b_evolve)
    pays = drift < 1e-3                                       # the transferred charge holds
    print(f"\n{label}")
    print(f"  1. structure  probe mismatch after mapping : {rel:.3f}   "
          f"{'PASS' if structure_ok else 'FAIL'} (needs < 0.05)")
    print(f"  2. it pays    transferred charge drift     : {drift:.2e}   "
          f"{'PASS' if pays else 'FAIL'} (needs < 1e-3)")
    verdict = "TRANSFER ACCEPTED" if (structure_ok and pays) else "no transfer"
    print(f"     -> {verdict}")
    return structure_ok and pays


if __name__ == "__main__":
    print("L3 -- can a solved structure be carried into a different-looking domain?")
    print("A: a 2-component field with a known conserved charge.")
    print("B: a 3-component system. Whether it contains A is what has to be discovered.")
    good = run(True,  "B BUILT FROM A's STRUCTURE  (transfer should succeed)")
    ctrl = run(False, "CONTROL -- B's symmetry deliberately broken (must be refused)")
    print(f"\n{'PASS' if (good and not ctrl) else 'FAIL'}: "
          f"accepted the real transfer = {good}, refused the control = {not ctrl}")
