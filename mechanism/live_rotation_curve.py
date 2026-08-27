"""A live controversy: does the dark-matter description of a rotation curve carry structure
no measurement can see?

NGC 3198, from SPARC (Lelli, McGaugh & Schombert 2016). Two descriptions fit the same curve:

    LCDM     baryons + an NFW halo, whose parameters are free per galaxy
    MOND     baryons only, with one universal acceleration scale

There is no ground truth here. That is the point -- it is why this test is free of the leakage
that makes historical cases worthless for evaluating a model that has read the histories. We
report what the criterion measures and attach no accuracy figure to it.
"""
import numpy as np
from concept_space import Theory, fit, coverage, unobservable_at_noise, quotient

A0 = 3700.0          # 1.2e-10 m/s^2 expressed in (km/s)^2 / kpc
H0 = 0.07            # 70 km/s/Mpc in km/s/kpc

d = np.loadtxt("data/NGC3198.txt")
R, VOBS, EV, VGAS, VDISK = d.T
EV = np.maximum(EV, 2.0)     # SPARC floors; keep errors honest and non-zero


def v_bary(p_ml):
    """Baryonic contribution with a stellar mass-to-light scaling."""
    return np.sign(VGAS) * VGAS**2 + p_ml * VDISK**2      # returns V^2


def nfw(r, v200, c):
    r200 = abs(v200) / (10 * H0)
    x = np.maximum(r / r200, 1e-9)
    cc = abs(c)
    m = np.log(1 + cc * x) - cc * x / (1 + cc * x)
    return v200**2 * m / (x * (np.log(1 + cc) - cc / (1 + cc)))


def lcdm_predict(r, p):
    ml, v200, c = p[0], p[1], p[2]
    return np.sqrt(np.maximum(v_bary(ml) + nfw(r, v200, c), 1e-9))


def mond_predict(r, p):
    ml = p[0]
    gb = np.maximum(v_bary(ml), 1e-9) / r
    g = gb / (1.0 - np.exp(-np.sqrt(gb / A0)))            # the RAR interpolating function
    return np.sqrt(g * r)


def mond_free_a0_predict(r, p):
    ml, a0 = p[0], abs(p[1])
    gb = np.maximum(v_bary(ml), 1e-9) / r
    g = gb / (1.0 - np.exp(-np.sqrt(gb / a0)))
    return np.sqrt(g * r)


MODELS = [
    (Theory("LCDM: baryons + NFW halo", ["M/L", "V200", "c"], lcdm_predict), [0.5, 150.0, 10.0]),
    (Theory("MOND: baryons, a0 universal", ["M/L"], mond_predict), [0.5]),
    (Theory("MOND: baryons, a0 free", ["M/L", "a0"], mond_free_a0_predict), [0.5, A0]),
]

print(f"NGC 3198 -- {len(R)} points, {R.min():.2f} to {R.max():.1f} kpc  (SPARC)\n")
print(f"{'description':<32}{'concepts':>9}{'chi2/N':>9}{'cover':>8}{'unobs':>7}   singular values")
print("-" * 104)
rows = []
for th, p0 in MODELS:
    p, _ = fit(th, R, VOBS, EV, p0=p0, restarts=14)
    pred = th.predict(R, p)
    chi2 = float(np.mean(((pred - VOBS) / EV) ** 2))
    cov = coverage(th, R, VOBS, EV, p)
    u, s = unobservable_at_noise(th, R, p, EV)
    spec = "  ".join(f"{v:8.2f}" for v in s)
    print(f"{th.name:<32}{th.n_concepts:>9}{chi2:>9.2f}{cov:>7.0%}{u:>7}   {spec}")
    rows.append((th, p, cov, u, s, chi2))

print("\nfitted values")
for th, p, *_ in rows:
    print("  " + th.name)
    for nm, v in zip(th.concepts, p):
        print(f"      {nm:<6} {v: .4g}")

print("\nwhat the mechanism does with the LCDM description")
th, p, cov, u, s, _ = rows[0]
q = quotient(th, R, p)
print(f"  exact algebraic null space : {'none' if q is None else str(q.n_concepts) + ' concepts left'}")
print(f"  at the measured error bars : {u} of {th.n_concepts} directions move no prediction past the noise")
print(f"  softest direction          : sigma_min = {s[-1]:.3f}  (a 100% excursion moves the curve "
      f"{s[-1]:.2f} sigma)")
_, J = __import__("concept_space").sloppy_spectrum(th, R, p, EV)
V = np.linalg.svd(J)[2]
soft = V[-1]
print("  it is a combination of     : " +
      ", ".join(f"{w:+.2f}*{n}" for w, n in zip(soft, th.concepts)))
