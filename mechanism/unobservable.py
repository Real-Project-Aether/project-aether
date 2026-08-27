"""A reconception removes description directions no probe can see. A patch adds them.

Clause 3 asked whether a theory forbids more, and it passed everything: any better-fitting
theory forbids more. It was measuring "is this a better theory", not "is this a reconception".

The five cases suggest a different signature. Lorentz's aether theory and special relativity
predict the SAME numbers; what differs is that Lorentz's description carries an aether velocity
that no measurement can pin down. Special relativity is that description with the unobservable
direction quotiented away.

    reconception  removes parameter directions that change nothing observable
    patch         adds them

That is D1 from operational/gauge.py -- verified there on a lattice gauge system -- pointed at
theory change instead of at a field. And it is measurable without any new machinery: the number
of directions in parameter space along which every prediction is unchanged is the null space of
the prediction Jacobian. Identifiability analysis, not an invention.
"""
from __future__ import annotations
import numpy as np


def unobservable_dim(theory, x, p, rel=1e-4, tol=1e-8):
    """Directions in parameter space that move no prediction at all."""
    p = np.asarray(p, float)
    if len(p) == 0:
        return 0, 0
    base = np.asarray(theory.predict(x, p), float)
    J = np.zeros((len(base), len(p)))
    for i in range(len(p)):
        step = rel * max(abs(p[i]), 1.0)
        q = p.copy(); q[i] += step
        J[:, i] = (np.asarray(theory.predict(x, q), float) - base) / step
    scale = np.linalg.norm(J) / max(np.sqrt(J.size), 1)
    if scale <= 0:
        return len(p), len(p)
    s = np.linalg.svd(J / scale, compute_uv=False)
    rank = int((s > tol).sum())
    return len(p) - rank, len(p)


def generalized_df(theory, x, y, sigma, n_boot=40, seed=0):
    """Effective degrees of freedom by perturbing the DATA, not the parameters.

    The Jacobian null space measures flexibility in the declared parameter space, so a theory
    that hides its flexibility in its code is invisible to it: a lookup table over 1048 points
    came out with an effective dimension of 1, because only its offset is a declared parameter,
    and the criterion duly preferred it to LCDM.

    This is Ye's generalised degrees of freedom -- df = sum_i d(yhat_i)/d(y_i), estimated by
    perturbing y and seeing how far the fit follows. A theory that tracks the data has df near n
    however its flexibility is written; one that commits to a shape has df near its parameter
    count. Standard statistics, and it cannot be hidden from.
    """
    from theory import fit
    rng = np.random.default_rng(seed)
    _, base = fit(theory, x, y, sigma)
    tot = 0.0
    for _ in range(n_boot):
        d = rng.normal(0, sigma)
        _, pert = fit(theory, x, y + d, sigma)
        tot += float(np.sum(d * (np.asarray(pert) - np.asarray(base)) / sigma ** 2))
    return tot / n_boot
