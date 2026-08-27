"""A theory as an object you can measure: coverage, length, and what it forbids.

AHA.md's three clauses need |T| to be computable, so a theory here is a PROGRAM -- a prediction
function plus its source and its free parameters. That makes:

    coverage    executable
    |T|         measurable
    prohibition computable, by asking what the fitted theory excludes OUTSIDE the fitted range

The length is two-part, the same shape as everywhere else in this repo: the code it takes to
state the theory, plus the parameters it needs fitted. A patch with a tunable knob must PAY for
that knob, which is exactly what stops "add a free parameter" from being free.

Encoding invariance is the thing that would make all of this meaningless -- if |T| measures my
writing style rather than the theory, the verdict is an artefact. `encoding_invariance` in
criterion.py tests that directly, and it is the first thing that has to pass.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import ast, zlib, inspect
import numpy as np
from scipy.optimize import least_squares


class _Canon(ast.NodeTransformer):
    """Alpha-rename every identifier to a canonical slot, so |T| measures structure only."""

    def __init__(self):
        self.names = {}

    def _slot(self, n):
        return self.names.setdefault(n, f"v{len(self.names)}")

    def visit_Name(self, node):
        node.id = self._slot(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self._slot(node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.name = self._slot(node.name)
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = self._slot(node.attr)
        return node


@dataclass
class Theory:
    name: str
    predict: object                 # (x, params) -> prediction
    n_params: int                   # free parameters the optimiser actually fits
    param_bounds: tuple             # (lo, hi) arrays
    source: str = ""                # the statement of the theory, for |T|
    n_effective: int = None         # parameters the theory COSTS, if different

    def __post_init__(self):
        # A lookup table's entries are not handed to the optimiser -- they are baked into the
        # prediction -- but they are still parameters and the two-part code has to charge for
        # them. Keeping the two counts separate is what lets a curve fit be cheap to FIT and
        # expensive to STATE, which is the whole point of the accounting.
        if self.n_effective is None:
            self.n_effective = self.n_params

    def code_bits(self):
        """Length of the theory's statement, normalised so writing style cannot buy anything.

        The first version dumped the AST and compressed it. That FAILED its own invariance test
        by a factor of 1.53: the dump still carries identifier names as strings, so a verbose
        encoding of the identical computation came out half as long again. Since the whole
        criterion turns on comparing |T'| against |T + patch|, a 53% swing available for free to
        whoever writes the code is exactly the leakage this test exists to catch.

        Fixed by alpha-renaming every identifier to a canonical slot before dumping, so only
        STRUCTURE survives. Temporaries are still counted -- inlining them is a compiler pass,
        not a normalisation -- so invariance is good but not perfect, and the residual is
        reported rather than assumed away.
        """
        tree = ast.parse(self.source)
        tree = _Canon().visit(tree)
        canon = ast.dump(tree, annotate_fields=False, include_attributes=False)
        return 8.0 * len(zlib.compress(canon.encode(), 9))

    def param_bits(self, n_data):
        """BIC-style: each parameter costs half a log of the sample size."""
        return 0.5 * self.n_effective * np.log(max(n_data, 2))

    def length(self, n_data):
        return self.code_bits() + self.param_bits(n_data)

    def cost_params_only(self, n_data):
        """Commitment measured by free parameters alone, with source length dropped.

        Source length turned out to measure how a theory is WRITTEN. On the quasicrystal case a
        one-parameter theory covering 100% came out more expensive than a six-parameter theory
        covering 79%, purely because its source has a nested comprehension. The
        encoding-invariance check missed this: it varied naming, not structure, so its 1.21
        residual badly understated the exposure -- and code length is exactly the channel
        through which whoever writes the encoding controls the verdict.
        """
        return self.param_bits(n_data) + 1.0

    def cost(self, n_data):
        """Length above the floor every program pays for merely being a program.

        Clause 2 first used absolute length, which is swamped by a ~1150-bit constant that any
        small function hits. Differencing against T_old fixed that and introduced a worse bug:
        a candidate SHORTER than T_old gets a negative cost, and a ratio with a negative
        denominator is meaningless -- which is how a bare curve fit came out at -376 bits and
        was declared the better theory. Subtracting a fixed trivial-program floor removes the
        constant and stays positive.
        """
        return max(self.length(n_data) - _FLOOR, 1.0)


_FLOOR = None


def fit(theory, x, y, sigma):
    """Least squares, with a grid fallback for objectives that have no gradient.

    A theory that predicts "the nearest allowed value" is piecewise constant, so its derivative
    is zero almost everywhere and least_squares never leaves its starting point -- which is how
    the quasicrystal case came out with 0.000 coverage for theories that fit it exactly at the
    right parameter. Detecting a flat gradient and switching to a coarse-to-fine grid costs
    little and removes a failure mode that looks exactly like a theory being wrong.
    """
    lo, hi = np.asarray(theory.param_bounds[0], float), np.asarray(theory.param_bounds[1], float)
    p0 = 0.5 * (lo + hi)
    if theory.n_params == 0:
        return np.array([]), theory.predict(x, np.array([]))

    def cost(p):
        return float((((theory.predict(x, p) - y) / sigma) ** 2).sum())

    # Multi-start, always, rather than only when the gradient tests exactly flat. The first
    # attempt gated a grid fallback on exact flatness and did not fire: a nearest-allowed-value
    # objective is not flat, it is riddled with local minima, and least_squares sat at its
    # starting point while the correct parameter one grid step away covered 98% of the data.
    # A coarse scan before refining costs almost nothing and removes the whole failure mode.
    if theory.n_params <= 3:
        grid = [np.linspace(lo[i], hi[i], 60) for i in range(theory.n_params)]
        mesh = np.array(np.meshgrid(*grid)).reshape(theory.n_params, -1).T
        costs = [cost(p) for p in mesh]
        p0 = mesh[int(np.argmin(costs))]
    else:
        # A full grid is hopeless past three dimensions, so scatter instead. Without this the
        # six-parameter twinning theory never moved off its starting point and scored 0.033,
        # which would have been read as the patch failing when it was the optimiser failing.
        g = np.random.default_rng(0)
        mesh = g.uniform(lo, hi, size=(4000, theory.n_params))
        costs = [cost(p) for p in mesh]
        p0 = mesh[int(np.argmin(costs))]

    r = least_squares(lambda p: (theory.predict(x, p) - y) / sigma, p0,
                      bounds=(lo, hi), max_nfev=4000)
    return r.x, theory.predict(x, r.x)


def coverage(theory, x, y, sigma, k=2.0):
    """Fraction of phenomena the theory reproduces within k sigma once fitted."""
    _, pred = fit(theory, x, y, sigma)
    return np.abs(pred - y) / sigma <= k


def prohibition(theory, x, y, sigma, x_out, n_boot=200, seed=0):
    """What the theory FORBIDS outside the range it was fitted on.

    Clause (3) needs a decidable psi that the new theory excludes and the old one permits. Two
    theories can agree perfectly where the data is and disagree sharply where it is not, and
    that disagreement is exactly the new prohibition. Returned as the predicted interval at
    x_out: a NARROW interval forbids a lot, a wide one forbids little.
    """
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n_boot):
        yb = y + rng.normal(0, sigma)
        try:
            p, _ = fit(theory, x, yb, sigma)
            preds.append(theory.predict(x_out, p))
        except Exception:
            continue
    P = np.array(preds)
    return np.percentile(P, 2.5, axis=0), np.percentile(P, 97.5, axis=0)


_TRIVIAL = Theory("trivial", lambda x, p: p[0], 0, ([0.0], [1.0]),
                  "def predict(x, p):\n    return p[0]\n")
_FLOOR = _TRIVIAL.code_bits()
