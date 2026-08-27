"""Engineering checks: the mechanism must accept the reinterpretations and reject the patches.

These are NOT a scientific validation. Every case here has an answer we put in ourselves, so
passing shows the code does what it claims, not that the criterion tracks real theory choice.
The live-controversy run is where it says something nobody knows, and there it gets no score.
"""
import numpy as np
from concept_space import (Theory, explore, fit, quotient, null_basis, unobservable_dim,
                           branch, REDUNDANCY, SYMMETRY, UNKNOWN)

rng = np.random.default_rng(0)
x = np.linspace(0.5, 5.0, 60)
SIG = 0.05

CHECKS = []


def _raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True


def check(name, got, want):
    ok = got == want
    CHECKS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} got={got!r:>12}  want={want!r}")


# 1. Gauge redundancy: a and b only ever appear as (a+b). One direction is invisible.
truth = 2.0 * x
y1 = truth + rng.normal(0, SIG, x.size)
redundant = Theory("split-coupling", ["a", "b"], lambda xx, p: (p[0] + p[1]) * xx)
p1, _ = fit(redundant, x, y1, SIG)
u1, _ = unobservable_dim(redundant, x, p1)
print("\n[1] a redundant description, where two concepts are never separable")
check("declared concepts", redundant.n_concepts, 2)
check("unobservable directions found", u1, 1)
q = quotient(redundant, x, p1, allow_blind=True)
check("reinterpreted concept count", q.n_concepts, 1)
pq, _ = fit(q, x, y1, SIG)
same = float(np.max(np.abs(q.predict(x, pq) - redundant.predict(x, p1))))
check("predictions unchanged (max abs diff < 1e-6)", same < 1e-6, True)
uq, _ = unobservable_dim(q, x, pq)
check("unobservable after reinterpretation", uq, 0)

# 2. A Lorentz-shaped theory: an extra concept that cancels out of every prediction.
def lorentz_like(xx, p):
    # p[2] is an 'aether velocity': it enters, and then exactly cancels.
    return p[0] * xx + p[1] + (p[2] * xx - p[2] * xx)

aether = Theory("carries-an-aether", ["slope", "offset", "aether_v"], lorentz_like)
y2 = 1.5 * x + 0.3 + rng.normal(0, SIG, x.size)
p2, _ = fit(aether, x, y2, SIG)
u2, _ = unobservable_dim(aether, x, p2)
print("\n[2] a theory carrying a velocity no measurement can reach")
check("unobservable directions found", u2, 1)
r2 = explore(aether, x, y2, SIG, allow_blind=True)
check("mechanism reinterpreted it", r2.final.n_concepts, 2)
check("final unobservable dimension", r2.log[-1].unobs, 0)

# 3. CONTROL -- an honest theory with nothing hidden. Must NOT be reinterpreted.
honest = Theory("honest-line", ["slope", "offset"], lambda xx, p: p[0] * xx + p[1])
p3, _ = fit(honest, x, y2, SIG)
u3, _ = unobservable_dim(honest, x, p3)
print("\n[3] CONTROL: an honest theory. Nothing to remove.")
check("unobservable directions", u3, 0)
check("quotient declines to act", quotient(honest, x, p3, allow_blind=True) is None, True)

# 4. CONTROL -- a patch that buys coverage with structure nothing can see. Must be REJECTED.
y4 = 1.5 * x + 0.3 + 0.4 * np.sin(3 * x) + rng.normal(0, SIG, x.size)
ghost = (lambda xx: np.zeros_like(xx), "ghost_term")     # improves nothing, adds a null direction
print("\n[4] CONTROL: a patch whose new concept moves no prediction")
r4 = explore(honest, x, y4, SIG, proposals=[ghost], allow_blind=True)
added = [s for s in r4.log if s.op == "new concept"]
check("patch was proposed", len(added) >= 1, True)
check("patch was rejected", any(not s.accepted for s in added), True)

# 5. An honest new concept: the residual really does have structure the language lacked.
real = (lambda xx: np.sin(3 * xx), "sin_mode")
print("\n[5] a real missing concept, recoverable from the residual")
r5 = explore(honest, x, y4, SIG, proposals=[real], allow_blind=True)
acc = [s for s in r5.log if s.op == "new concept" and s.accepted]
check("new concept accepted", len(acc) == 1, True)
check("final concept count", r5.final.n_concepts, 3)
check("final unobservable dimension", r5.log[-1].unobs, 0)
cov0 = r5.log[0].coverage
check("coverage improved", r5.log[-1].coverage > cov0, True)

print("\n[6] the D1 guard: an unobservable direction is not automatically deletable")
check("refuses to quotient with no locality test",
      _raises(lambda: quotient(redundant, x, p1)), True)
check("classifies a locally-invariant direction", branch(None, lambda v: 0.0), REDUNDANCY)
check("classifies a uniform-only direction", branch(None, lambda v: 9.9), SYMMETRY)
check("keeps a symmetry instead of deleting it",
      quotient(redundant, x, p1, locality_test=lambda v: 9.9) is None, True)
check("with no test available, says so", branch(None, None), UNKNOWN)

print(f"\n{sum(CHECKS)}/{len(CHECKS)} checks pass")
raise SystemExit(0 if all(CHECKS) else 1)
