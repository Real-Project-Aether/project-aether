"""A concept space that changes shape, driven by what no measurement can see.

The corpus says machine learning implements one rung of a six-rung ladder: L0, fitting inside a
fixed description. Everything above it -- reinterpreting, retyping, transferring, introducing a
kind of object -- has no operation in any discovery system we know of. This module is an attempt
at the missing one, L1.

The move it implements is Einstein's over Lorentz's. Both theories predicted identical numbers.
Lorentz's carried an aether velocity that no measurement could ever pin down; Einstein's did not.
Written as an operation on a parameter space, "carries structure no measurement can see" is the
null space of the prediction Jacobian, and "reinterpret" is quotienting that null space away.

That gives a concept space that is dynamic in a specific, checkable sense: concepts are not added
because they improve a fit -- that is L0 with extra steps -- but removed when the data says they
were never separable, and added only when a residual has structure the current concepts cannot
express. Acceptance is decided by measurement, not by the person writing the theory down:

    ACCEPT an operation iff  coverage does not fall
                        and  unobservable dimension strictly drops   (a reinterpretation)
                        or   unobservable dimension stays at zero    (an honest new concept)

The second clause is what stops "add a free parameter" from being free, and it is why fitting
better cannot win here: a patch that buys coverage with unobservable structure is rejected by
construction.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unobservable import unobservable_dim  # noqa: E402


# --------------------------------------------------------------------------- theories


@dataclass
class Theory:
    """A named concept list plus a rule that turns concept values into predictions."""

    name: str
    concepts: list[str]
    predict_fn: object                      # (x, params) -> predictions
    note: str = ""

    def predict(self, x, p):
        return self.predict_fn(x, np.asarray(p, float))

    @property
    def n_concepts(self):
        return len(self.concepts)


def fit(theory, x, y, sigma, p0=None, restarts=6, seed=0):
    """Least squares with restarts. Returns (params, residual sum of squares)."""
    from scipy.optimize import least_squares

    rng = np.random.default_rng(seed)
    n = theory.n_concepts
    best, best_cost = None, np.inf
    starts = [np.zeros(n) if p0 is None else np.asarray(p0, float)]
    starts += [rng.normal(0, 1, n) for _ in range(restarts - 1)]
    for s in starts:
        try:
            r = least_squares(lambda p: (theory.predict(x, p) - y) / sigma, s, method="lm",
                              max_nfev=20000)
        except Exception:
            continue
        if r.cost < best_cost:
            best, best_cost = r.x, r.cost
    if best is None:
        return np.zeros(n), np.inf
    return best, float(np.sum((theory.predict(x, best) - y) ** 2))


def coverage(theory, x, y, sigma, p):
    """Fraction of observations the fitted theory reproduces inside 2 sigma."""
    resid = np.abs(theory.predict(x, p) - y)
    return float(np.mean(resid <= 2 * sigma))


# --------------------------------------------------------------------------- the operations


def null_basis(theory, x, p, tol=1e-8):
    """The directions in concept space along which no prediction moves at all."""
    p = np.asarray(p, float)
    base = np.asarray(theory.predict(x, p), float)
    J = np.zeros((len(base), len(p)))
    for i in range(len(p)):
        step = 1e-4 * max(abs(p[i]), 1.0)
        q = p.copy()
        q[i] += step
        J[:, i] = (np.asarray(theory.predict(x, q), float) - base) / step
    scale = np.linalg.norm(J) / max(np.sqrt(J.size), 1)
    if scale <= 0:
        return np.eye(len(p)), np.zeros((len(p), 0))
    U, s, Vt = np.linalg.svd(J / scale)
    rank = int((s > tol).sum())
    return Vt[:rank].T, Vt[rank:].T          # (observable directions, unobservable directions)


REDUNDANCY, SYMMETRY, UNKNOWN = "REDUNDANCY", "SYMMETRY", "UNKNOWN"


def branch(direction, locality_test, tol=1e-9):
    """Is an unobservable direction a redundancy to delete, or a symmetry to keep?

    STANDARD_MODEL.md D1: a *global* phase rotation is invisible to every probe, exactly like a
    gauge transformation. Quotienting both -- which is what this module did before -- does not
    discover charge conservation, it ERASES the conserved quantity. One question separates them:

        invariant when applied INDEPENDENTLY AT EACH POINT  -> redundancy -> quotient   [L1]
        invariant only when applied UNIFORMLY               -> symmetry   -> keep it,
                                                                  and read off a charge [L4]

    `locality_test(direction)` returns how far the probes move under point-by-point application.
    Verified on a lattice gauge system in operational/gauge.py and again, without being told
    which transformations are invariances, in discover_symmetry.py.
    """
    if locality_test is None:
        return UNKNOWN
    return REDUNDANCY if locality_test(direction) <= tol else SYMMETRY


def quotient(theory, x, p, locality_test=None, allow_blind=False):
    """REINTERPRET: build the same theory with its *redundant* directions removed.

    Predictions are identical by construction -- that is what makes this a reinterpretation
    rather than a better fit. Directions that turn out to be symmetries are NOT removed; they
    are conserved quantities, and deleting them is the D1 bug.

    With no locality test available the substrate has no notion of "at each point", so the two
    cases cannot be told apart. The mechanism then refuses to act rather than guess: pass
    allow_blind=True only where you have established there is no symmetry to destroy.
    """
    obs, nul = null_basis(theory, x, p)
    if nul.shape[1] == 0:
        return None                          # nothing unobservable; nothing to reinterpret
    if locality_test is not None:
        keep = [i for i in range(nul.shape[1])
                if branch(nul[:, i], locality_test) == SYMMETRY]
        if keep:
            return None                      # these are charges, not redundancy -- do not delete
    elif not allow_blind:
        raise ValueError(
            "refusing to quotient without a locality test: a global symmetry is unobservable "
            "too, and removing it would delete a conservation law (STANDARD_MODEL.md D1). "
            "Supply locality_test, or pass allow_blind=True if the substrate has no symmetries.")
    p0 = np.asarray(p, float)
    k = obs.shape[1]

    def predict_fn(xx, q, _obs=obs, _p0=p0, _f=theory.predict_fn):
        return _f(xx, _p0 + _obs @ np.asarray(q, float)[:k])

    names = [f"c{i}" for i in range(k)]
    return Theory(
        name=f"{theory.name} / quotient",
        concepts=names,
        predict_fn=predict_fn,
        note=f"{nul.shape[1]} unobservable direction(s) removed from {theory.name}",
    )


def add_latent(theory, basis_fn, label):
    """NEW OBJECT: extend the concept list with a structure the old language could not express."""

    def predict_fn(xx, p, _f=theory.predict_fn, _n=theory.n_concepts, _b=basis_fn):
        p = np.asarray(p, float)
        return _f(xx, p[:_n]) + p[_n] * _b(xx)

    return Theory(
        name=f"{theory.name} + {label}",
        concepts=list(theory.concepts) + [label],
        predict_fn=predict_fn,
        note=f"latent concept '{label}' introduced",
    )


# --------------------------------------------------------------------------- the loop


@dataclass
class Step:
    op: str
    theory: str
    n_concepts: int
    unobs: int
    coverage: float
    accepted: bool
    why: str


@dataclass
class Result:
    final: Theory
    params: np.ndarray
    log: list = field(default_factory=list)

    def show(self):
        w = max(len(s.theory) for s in self.log) + 2
        print(f"{'op':<14}{'theory':<{w}}{'concepts':>9}{'unobs':>7}{'cover':>8}   verdict")
        print("-" * (44 + w))
        for s in self.log:
            v = "accept" if s.accepted else "reject"
            print(f"{s.op:<14}{s.theory:<{w}}{s.n_concepts:>9}{s.unobs:>7}{s.coverage:>7.0%}   {v} -- {s.why}")


def explore(theory, x, y, sigma, proposals=(), max_steps=8, tol=1e-9,
            locality_test=None, allow_blind=False):
    """Run the concept space until no operation is accepted.

    An operation is accepted only if coverage does not fall AND either the unobservable
    dimension strictly drops (a reinterpretation) or it stays at zero (an honest new concept).
    Nothing here rewards a better fit on its own.

    `locality_test` decides whether an unobservable direction is a redundancy to quotient or a
    symmetry to keep -- see branch(). These 1-D parametric substrates have no "each point" to
    apply a transformation at, so the tests pass allow_blind=True; the lattice, which does, is
    handled in discover_symmetry.py.
    """
    cur = theory
    p, _ = fit(cur, x, y, sigma)
    cov = coverage(cur, x, y, sigma, p)
    unobs, _ = unobservable_dim(cur, x, p)
    log = [Step("start", cur.name, cur.n_concepts, unobs, cov, True, "initial theory")]

    pending = list(proposals)
    for _ in range(max_steps):
        moved = False

        # L1 -- reinterpret: is any of this description invisible to every measurement?
        cand = quotient(cur, x, p, locality_test=locality_test,
                        allow_blind=allow_blind)
        if cand is not None:
            q, _ = fit(cand, x, y, sigma)
            c2 = coverage(cand, x, y, sigma, q)
            u2, _ = unobservable_dim(cand, x, q)
            ok = c2 >= cov - tol and u2 < unobs
            log.append(Step("reinterpret", cand.name, cand.n_concepts, u2, c2, ok,
                            f"unobservable {unobs}->{u2}, coverage {cov:.0%}->{c2:.0%}"))
            if ok:
                cur, p, cov, unobs = cand, q, c2, u2
                moved = True
                continue

        # L4 -- introduce a kind of object the language lacked
        if pending:
            basis_fn, label = pending.pop(0)
            cand = add_latent(cur, basis_fn, label)
            q, _ = fit(cand, x, y, sigma)
            c2 = coverage(cand, x, y, sigma, q)
            u2, _ = unobservable_dim(cand, x, q)
            ok = c2 > cov + tol and u2 == 0
            why = (f"coverage {cov:.0%}->{c2:.0%}, unobservable {u2}"
                   + ("" if u2 == 0 else " -- buys fit with structure nothing can see"))
            log.append(Step("new concept", cand.name, cand.n_concepts, u2, c2, ok, why))
            if ok:
                cur, p, cov, unobs = cand, q, c2, u2
                moved = True
                continue

        if not moved:
            break

    return Result(final=cur, params=p, log=log)


# --------------------------------------------------------------- unobservable AT THE NOISE LEVEL

def sloppy_spectrum(theory, x, p, sigma):
    """Singular values of the Jacobian in units of "sigmas moved per 100% parameter change".

    Exact algebraic degeneracy is the wrong test for real data. Lorentz's aether was not
    unobservable in principle -- it was unobservable to within the precision of the experiments
    of the day. The physical question is whether a direction can be moved by an order-unity
    amount without shifting any prediction by more than the error bars.

    Column i is scaled by |p_i| so the question asked of every direction is the same one, and by
    1/sigma so the answer is in units the measurement supplies. A singular value below 1 means a
    100% excursion along that direction hides inside the noise.
    """
    p = np.asarray(p, float)
    base = np.asarray(theory.predict(x, p), float)
    sig = np.broadcast_to(np.asarray(sigma, float), base.shape)
    J = np.zeros((len(base), len(p)))
    for i in range(len(p)):
        h = 1e-5 * max(abs(p[i]), 1e-8)
        q = p.copy(); q[i] += h
        J[:, i] = (np.asarray(theory.predict(x, q), float) - base) / h * abs(p[i]) / sig
    return np.linalg.svd(J, compute_uv=False), J


def unobservable_at_noise(theory, x, p, sigma, thresh=1.0):
    """How many directions can be moved by 100% without moving any prediction past the noise."""
    s, _ = sloppy_spectrum(theory, x, p, sigma)
    return int((s < thresh).sum()), s
