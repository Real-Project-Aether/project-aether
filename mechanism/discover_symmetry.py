"""Find the invariances nobody named, then sort them into redundancies and conservation laws.

`operational/gauge.py` classifies transformations it is HANDED. That is verification, not
discovery: someone already knew to write down gauge(), flavour_su2(), flavour_phase(). Here the
mechanism is given only a basis of candidate generators -- most of which are not invariances at
all -- and has to find for itself which directions the probes cannot see.

    1. build  d(probes)/dc  over the generator basis, applied UNIFORMLY
    2. its null space IS the set of invariant directions        <- found, not supplied
    3. for each, ask whether it survives being applied INDEPENDENTLY AT EACH POINT
           survives -> REDUNDANCY -> quotient it                       (L1)
           does not -> SYMMETRY   -> keep it, emit the Noether charge  (L4)
    4. verify each emitted charge against the dynamics

Step 3 is the fix from STANDARD_MODEL.md D1. Without it step 2 would quotient a global phase
rotation and delete charge conservation along with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # gauge.py ships alongside
from gauge import L, NF, random_state, probes, charge, evolve, hamiltonian   # noqa: E402

TOL = 1e-9
PAULI = {
    "I":  np.eye(2, dtype=complex),
    "sx": np.array([[0, 1], [1, 0]], dtype=complex),
    "sy": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "sz": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _expm2(A):
    w, V = np.linalg.eigh(A)
    return (V * np.exp(1j * w)) @ V.conj().T


# ---------------------------------------------------------------- candidate generators
# Each takes a FIELD alpha of shape (L, L): constant for the uniform test, random per site for
# the local test. Most of these are not invariances; the mechanism has to work that out.

def flavour(T):
    def f(state, alpha):
        phi, U = state
        out = np.empty_like(phi)
        for i in range(L):
            for j in range(L):
                out[i, j] = _expm2(alpha[i, j] * T) @ phi[i, j]
        return out, U
    return f


def gauge_compensated(state, alpha):
    """phi picks up a phase and the links compensate it. The textbook redundancy."""
    phi, U = state
    p = phi * np.exp(1j * alpha)[:, :, None]
    u = U.copy()
    for mu, ax in ((0, 0), (1, 1)):
        u[:, :, mu] = np.exp(1j * alpha) * U[:, :, mu] * np.exp(-1j * np.roll(alpha, -1, axis=ax))
    return p, u


def link_phase(state, alpha):
    """Links rotate, matter does not. Distractor."""
    phi, U = state
    u = U * np.exp(1j * alpha)[:, :, None]
    return phi, u


def rescale(state, alpha):
    """Not unitary. Distractor, and the control from gauge.py."""
    phi, U = state
    return phi * (1 + 0.3 * alpha)[:, :, None], U


def shift_flavour0(state, alpha):
    """Add a constant to one flavour component. Distractor."""
    phi, U = state
    p = phi.copy()
    p[:, :, 0] = p[:, :, 0] + 0.1 * alpha
    return p, U


BASIS = [("flavour I", flavour(PAULI["I"]), PAULI["I"]),
         ("flavour sx", flavour(PAULI["sx"]), PAULI["sx"]),
         ("flavour sy", flavour(PAULI["sy"]), PAULI["sy"]),
         ("flavour sz", flavour(PAULI["sz"]), PAULI["sz"]),
         ("gauge (compensated)", gauge_compensated, None),
         ("link phase", link_phase, None),
         ("rescale", rescale, None),
         ("shift flavour 0", shift_flavour0, None)]


# ---------------------------------------------------------------- step 1-2: find invariances

def uniform_jacobian(states, eps=1e-6):
    """d(probes)/dc for each generator, applied with the SAME coefficient everywhere."""
    cols = []
    for _, f, _ in BASIS:
        col = []
        for st in states:
            base = probes(st)
            a = np.full((L, L), eps)
            col.append((probes(f(st, a)) - base) / eps)
        cols.append(np.concatenate(col))
    return np.array(cols).T


def local_jacobian(states, null, seed=7, eps=1e-6, n_profiles=3):
    """Response of the probes when a null direction is applied with a SITE-DEPENDENT coefficient.

    Restricted to directions already known to be invariant when applied uniformly, so what this
    measures is purely the local part. Its own null space is the redundancy subspace: the
    directions that survive being applied independently at each point. Everything else in `null`
    is a symmetry, and that is the distinction D1 says the framework must not skip.
    """
    cols = []
    for v in null:
        col = []
        for k, st in enumerate(states):
            for j in range(n_profiles):
                rng = np.random.default_rng(1000 * j + seed + k)
                w = rng.uniform(-1, 1, size=(L, L))
                w -= w.mean()                       # mean-zero: isolate the non-uniform part
                base = probes(st)
                cur = st
                for i, c in enumerate(v):           # apply the mixture, not one generator
                    if abs(c) > 1e-12:
                        cur = BASIS[i][1](cur, eps * c * w)
                col.append((probes(cur) - base) / eps)
        cols.append(np.concatenate(col))
    return np.array(cols).T


def mixture_generator(v):
    """The flavour generator of a null direction, as a combination. None if it touches links."""
    T = np.zeros((NF, NF), dtype=complex)
    touches_links = False
    for i, c in enumerate(v):
        if abs(c) < 1e-6:                     # numerical dust, not a real link component
            continue
        if BASIS[i][2] is None:
            touches_links = True
        else:
            T = T + c * BASIS[i][2]
    return T, touches_links


def main():
    states = [random_state(seed=s) for s in range(6)]

    print("Candidate generators supplied (nothing is said about which are real invariances):")
    for n, _, _ in BASIS:
        print(f"    {n}")

    # --- step 1-2: which directions can no probe see, applied uniformly?
    J = uniform_jacobian(states)
    s_u = np.linalg.svd(J / (np.linalg.norm(J) / np.sqrt(J.size)), compute_uv=False)
    Vt = np.linalg.svd(J / (np.linalg.norm(J) / np.sqrt(J.size)))[2]
    rank = int((s_u > 1e-6 * s_u[0]).sum())        # relative: the gap is orders wide
    null = Vt[rank:]
    print(f"\nStep 1-2  uniform d(probes)/dc singular values: " + "  ".join(f"{v:.1e}" for v in s_u))
    print(f"          rank {rank} of {len(BASIS)}  ->  {len(null)} invariant direction(s) FOUND")
    obs = [BASIS[i][0] for i in range(len(BASIS))
           if max(abs(Vt[r][i]) for r in range(rank)) > 0.5]
    print(f"          rejected as not invariant: {', '.join(obs)}")

    # --- step 3: inside that, which survive being applied point by point?
    Jl = local_jacobian(states, null)
    s_l = np.linalg.svd(Jl / (np.linalg.norm(Jl) / np.sqrt(Jl.size)), compute_uv=False)
    Vl = np.linalg.svd(Jl / (np.linalg.norm(Jl) / np.sqrt(Jl.size)))[2]
    rank_l = int((s_l > 1e-6 * s_l[0]).sum())      # relative, same reason
    red_dirs = [Vl[i] @ null for i in range(rank_l, len(null))]     # locally invariant
    sym_dirs = [Vl[i] @ null for i in range(rank_l)]                # locally NOT invariant
    print(f"\nStep 3    local d(probes)/dc within those directions: "
          + "  ".join(f"{v:.1e}" for v in s_l))
    print(f"          {len(red_dirs)} REDUNDANCY (survives point-by-point)  ->  quotient   [L1]")
    print(f"          {len(sym_dirs)} SYMMETRY   (needs to move everywhere at once) -> keep [L4]")
    for v in red_dirs:
        mix = " ".join(f"{c:+.2f}[{BASIS[i][0]}]" for i, c in enumerate(v) if abs(c) > 0.15)
        print(f"            redundancy = {mix}")

    # --- step 4: does each symmetry branch actually buy a conserved quantity?
    print("\nStep 4    do the symmetry branches buy a conserved quantity?")
    print(f"{'symmetry direction':<40}{'charge before':>15}{'after':>15}{'rel drift':>12}")
    print("-" * 82)
    st = random_state(seed=0)
    ev = evolve(st, steps=3000, dt=2e-3)
    drifts = []
    for v in sym_dirs:
        T, links = mixture_generator(v)
        if links or np.linalg.norm(T) < 1e-9:
            continue
        q0, q1 = charge(st[0], T), charge(ev[0], T)
        d = abs(q1 - q0) / max(abs(q0), 1e-12)
        drifts.append(d)
        mix = " ".join(f"{c:+.2f}{BASIS[i][0].replace('flavour ','')}"
                       for i, c in enumerate(v) if abs(c) > 0.15)
        print(f"{mix[:38]:<40}{q0:15.6f}{q1:15.6f}{d:12.1e}")

    # the redundancy branch must NOT buy one -- that asymmetry is the whole point
    for v in red_dirs:
        n = np.einsum("xya,xya->xy", st[0].conj(), st[0]).real
        rng = np.random.default_rng(3)
        w = rng.uniform(-1, 1, size=(L, L)); w -= w.mean()
        q0 = float((w * n).sum())
        n1 = np.einsum("xya,xya->xy", ev[0].conj(), ev[0]).real
        q1 = float((w * n1).sum())
        print(f"{'redundancy: would-be local charge':<40}{q0:15.6f}{q1:15.6f}"
              f"{abs(q1-q0)/max(abs(q0),1e-12):12.1e}")

    # --- the counterfactual: what the mechanism did before D1 was fixed
    print("\nWithout step 3 (quotient everything no probe can see -- the D1 bug):")
    print(f"          all {len(null)} invariant directions quotiented, including the {len(sym_dirs)} "
          f"that are symmetries")
    print(f"          -> {len(drifts)} conservation law(s) destroyed, each verified above to "
          f"{max(drifts):.0e}" if drifts else "")
    print("          the local/uniform test is what separates 'delete it' from 'it is a charge'")

    h0, h1 = hamiltonian(st), hamiltonian(ev)
    print(f"\n  energy drift on the same trajectory: {abs(h1-h0)/abs(h0):.1e}  (trajectory sound)")
    if drifts:
        print(f"  worst symmetry-charge drift:         {max(drifts):.1e}")


if __name__ == "__main__":
    main()
