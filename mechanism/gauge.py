"""D1: does the local/uniform test separate a redundancy from a symmetry?

STANDARD_MODEL.md found that OPERATIONAL.md section 3.2 quotients any transformation every
probe is blind to -- which deletes conservation laws, because a global symmetry passes that
test too. The proposed fix was one extra question: can the transformation be applied
INDEPENDENTLY AT EACH POINT and still leave every probe invariant?

    invariant when applied locally   -> redundancy -> quotient, and it predicts nothing
    invariant only when applied      -> symmetry   -> keep, and it predicts a CONSERVED
    uniformly                                          QUANTITY, which is a probe to run

That asymmetry is the point. A redundancy costs the description something and buys no
prediction; a symmetry buys a prediction that can be tested against the dynamics. This file
tests whether the rule actually sorts them, on a lattice gauge system where the answer is
known independently.

State: complex matter field phi[x, a] with a flavour index, on a periodic 2-D lattice, and
U(1) link variables. Probes are the gauge-invariant quantities an experiment could report.
"""
from __future__ import annotations
import numpy as np

L, NF = 6, 2                      # lattice size, number of flavours
rng_global = np.random.default_rng(0)


def random_state(seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(L, L, NF)) + 1j * rng.normal(size=(L, L, NF))
    theta = rng.uniform(0, 2 * np.pi, size=(L, L, 2))
    return phi, np.exp(1j * theta)


def probes(state):
    """Everything an experiment can report. All gauge invariant, all flavour-blind."""
    phi, U = state
    out = [np.einsum("xya,xya->xy", phi.conj(), phi).real]                  # site density
    for mu, ax in ((0, 0), (1, 1)):
        shifted = np.roll(phi, -1, axis=ax)
        out.append(np.einsum("xya,xya->xy", phi.conj(), shifted * U[:, :, mu, None]).real)
    # plaquette
    W = U[:, :, 0] * np.roll(U[:, :, 1], -1, axis=0) \
        * np.roll(U[:, :, 0], -1, axis=1).conj() * U[:, :, 1].conj()
    out.append(W.real)
    return np.concatenate([o.ravel() for o in out])


# ---------------------------------------------------------------- transformations
def gauge(state, alpha):
    """phi -> e^{i alpha} phi,  U_{x,mu} -> e^{i alpha_x} U e^{-i alpha_{x+mu}}."""
    phi, U = state
    p = phi * np.exp(1j * alpha)[:, :, None]
    u = U.copy()
    for mu, ax in ((0, 0), (1, 1)):
        u[:, :, mu] = np.exp(1j * alpha) * U[:, :, mu] * np.exp(-1j * np.roll(alpha, -1, axis=ax))
    return p, u


def flavour_su2(state, alpha):
    """A rotation mixing the flavours. alpha scales the SAME generator at each site."""
    phi, U = state
    G = np.array([[0, 1], [1, 0]], dtype=complex)          # a Pauli-x generator
    out = np.empty_like(phi)
    for x in range(L):
        for y in range(L):
            V = _expm_herm(alpha[x, y] * G)
            out[x, y] = V @ phi[x, y]
    return out, U


def flavour_phase(state, alpha):
    """Opposite phases on the two flavours: the flavour-number U(1)."""
    phi, U = state
    p = phi.copy()
    p[:, :, 0] *= np.exp(1j * alpha)
    p[:, :, 1] *= np.exp(-1j * alpha)
    return p, U


def rescale(state, alpha):
    """NEGATIVE CONTROL: not a symmetry at all -- it changes the density."""
    phi, U = state
    p = phi.copy()
    p[:, :, 0] *= (1.0 + 0.3 * alpha)
    return p, U


def _expm_herm(A):
    w, V = np.linalg.eigh(A / 1j)          # A = i*H with H hermitian
    return V @ np.diag(np.exp(1j * w)) @ V.conj().T


# ---------------------------------------------------------------- the test
def classify(transform, n_state=8, tol=1e-9):
    """Uniform first: if the probes move, it is not a symmetry. Then local."""
    uni_err, loc_err = [], []
    for s in range(n_state):
        st = random_state(seed=s)
        base = probes(st)
        rng = np.random.default_rng(100 + s)

        a_uni = np.full((L, L), rng.uniform(0.2, 1.5))              # same everywhere
        uni_err.append(np.abs(probes(transform(st, a_uni)) - base).max())

        a_loc = rng.uniform(0.2, 1.5, size=(L, L))                  # independent per site
        loc_err.append(np.abs(probes(transform(st, a_loc)) - base).max())

    u, l = max(uni_err), max(loc_err)
    if u > tol:
        verdict = "NEITHER — not a symmetry"
    elif l <= tol:
        verdict = "REDUNDANCY — quotient it"
    else:
        verdict = "SYMMETRY — keep it, expect a conserved quantity"
    return u, l, verdict


if __name__ == "__main__":
    print("D1: local/uniform sorting of invariances\n")
    print(f"  {'transformation':>16s} {'uniform':>12s} {'local':>12s}   verdict")
    for name, f in [("gauge U(1)", gauge),
                    ("flavour SU(2)", flavour_su2),
                    ("flavour phase", flavour_phase),
                    ("rescale (control)", rescale)]:
        u, l, v = classify(f)
        print(f"  {name:>16s} {u:12.2e} {l:12.2e}   {v}")


# ---------------------------------------------------------------- the payoff
# Sorting invariances is only half of D1. The claim was that the two branches differ in what
# they BUY: a symmetry predicts a conserved quantity -- a probe that can be run against the
# dynamics -- and a redundancy predicts nothing. That is what makes the distinction worth
# drawing, so it has to be checked rather than asserted.

M2, LAM = 0.5, 0.2


def hamiltonian(state):
    phi, U = state
    h = 0.0
    for mu, ax in ((0, 0), (1, 1)):
        shifted = np.roll(phi, -1, axis=ax)
        h -= 2 * np.einsum("xya,xya->", phi.conj(), shifted * U[:, :, mu, None]).real
    n = np.einsum("xya,xya->xy", phi.conj(), phi).real
    return h + M2 * n.sum() + LAM * (n ** 2).sum()


def dphi_dt(phi, U):
    """i dphi/dt = dH/dphi*."""
    g = np.zeros_like(phi)
    for mu, ax in ((0, 0), (1, 1)):
        g -= np.roll(phi, -1, axis=ax) * U[:, :, mu, None]
        g -= np.roll(phi * U[:, :, mu, None].conj(), 1, axis=ax)
    n = np.einsum("xya,xya->xy", phi.conj(), phi).real
    g += M2 * phi + 2 * LAM * n[:, :, None] * phi
    return g / 1j


def evolve(state, steps=4000, dt=2e-3):
    phi, U = state
    for _ in range(steps):                       # RK4
        k1 = dphi_dt(phi, U)
        k2 = dphi_dt(phi + 0.5 * dt * k1, U)
        k3 = dphi_dt(phi + 0.5 * dt * k2, U)
        k4 = dphi_dt(phi + dt * k3, U)
        phi = phi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return phi, U


def charge(phi, T):
    """Noether charge of phi -> exp(i eps T) phi for a hermitian flavour generator T."""
    return np.einsum("xya,ab,xyb->", phi.conj(), T, phi).real


def local_charge(phi, alpha):
    """The would-be charge of a POSITION-DEPENDENT phase rotation."""
    n = np.einsum("xya,xya->xy", phi.conj(), phi).real
    return float((alpha * n).sum())


if __name__ == "__main__":
    print("\n\nwhat each branch buys: is the predicted quantity actually conserved?\n")
    st = random_state(seed=3)
    end = evolve(st)
    h0, h1 = hamiltonian(st), hamiltonian(end)
    print(f"  energy drift over the trajectory: {abs(h1 - h0) / abs(h0):.2e}  (integrator check)\n")

    I = np.eye(2, dtype=complex)
    SX = np.array([[0, 1], [1, 0]], dtype=complex)
    SZ = np.array([[1, 0], [0, -1]], dtype=complex)

    print(f"  {'branch':>34s} {'quantity':>18s} {'relative drift':>16s}")
    for label, T in [("SYMMETRY  flavour SU(2), sigma_x", SX),
                     ("SYMMETRY  flavour phase, sigma_z", SZ),
                     ("SYMMETRY  global part of gauge U(1)", I)]:
        q0, q1 = charge(st[0], T), charge(end[0], T)
        print(f"  {label:>34s} {'Noether charge':>18s} {abs(q1 - q0) / max(abs(q0), 1e-12):16.2e}")

    rng = np.random.default_rng(7)
    alpha = rng.uniform(0.2, 1.5, size=(L, L))
    q0, q1 = local_charge(st[0], alpha), local_charge(end[0], alpha)
    print(f"  {'REDUNDANCY  strictly local gauge':>34s} {'(no charge)':>18s} "
          f"{abs(q1 - q0) / abs(q0):16.2e}   <- NOT conserved")
    print("\n  A redundancy yields a conserved quantity only for its constant part, which is")
    print("  the symmetry sitting inside it. The strictly local part predicts nothing, and")
    print("  that is exactly the asymmetry D1 claimed.")
