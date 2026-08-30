"""L4, a new object: posit an entity to make a violated bookkeeping constraint balance.

181 corpus events. The shape recurs exactly: something does not add up, and a new kind of thing
is introduced so that it does.

    beta decay's energy spectrum is continuous  ->  Pauli posits the neutrino
    Delta++ violates Fermi statistics           ->  colour charge is invented
    flavour-changing neutral currents too rare  ->  the charm quark is invented

This composes with what the L1/L4 symmetry branch already produces. That branch FINDS conserved
charges. This one USES a found charge: when it fails to balance across an observed process, the
missing amount is not noise but a description of something not yet named.

    charge in  -  charge out  =  0        within the noise  ->  nothing to posit
                              =  Q  !=  0 beyond the noise  ->  posit a carrier of charge Q

The honest limit, stated before the results: this assumes the conservation law holds and reads
the residual off it. The hard half of Pauli's move was DECIDING to trust conservation over the
measurements, and nothing here does that. What is mechanised is the inference once that choice
is made -- which is still the step that names a new object.
"""
from __future__ import annotations

import numpy as np

NF = 3                                   # three species, only two of which the probes can see
VISIBLE = [0, 1]
Q = np.array([2.0, 1.0, 1.0])            # charge per species; species 2 is invisible


def process(rng, n=400, hidden_share=0.0, noise=0.02):
    """One decay channel, measured. A parent of charge +2 breaks into two products of +1 each.

    Charge is conserved by construction, always. What varies is whether one product lands in the
    invisible species: with hidden_share = 0 both products are seen and the visible books close;
    with hidden_share = 1 half the charge leaves through a species no probe reports.
    """
    parent = rng.uniform(2.0, 6.0, size=n)          # how many parents decayed
    q_in = parent * Q[0]                            # +2 each

    hid = hidden_share * rng.uniform(0.85, 1.15, size=n)
    hid = np.clip(hid, 0.0, 1.0)
    out = np.zeros((n, NF))
    out[:, 1] = parent * (2.0 - hid)                # products landing in the visible species
    out[:, 2] = parent * hid                        # products landing in the invisible one

    # charge really is conserved: out @ Q == q_in, to machine precision
    assert np.allclose(out @ Q, q_in), "the construction must conserve charge"

    q_out_visible = out[:, VISIBLE] @ Q[VISIBLE]
    return (q_in * (1 + rng.normal(0, noise, n)),
            q_out_visible * (1 + rng.normal(0, noise, n)))


def posit(q_in, q_out, noise=0.02):
    """Does the ledger close? If not, describe what would close it."""
    resid = q_in - q_out
    scale = np.mean(np.abs(q_in))
    mean, sem = resid.mean(), resid.std(ddof=1) / np.sqrt(len(resid))
    z = mean / max(sem, 1e-12)
    # the imbalance must be real (many sigma) and not a rounding artefact (a real fraction of scale)
    real = abs(z) > 6 and abs(mean) / scale > 3 * noise
    return {
        "missing_per_event": mean,
        "as_fraction_of_input": mean / scale,
        "z": z,
        "posit": real,
    }


def run(label, hidden_share, expect):
    rng = np.random.default_rng(7)
    q_in, q_out = process(rng, hidden_share=hidden_share)
    r = posit(q_in, q_out)
    ok = r["posit"] == expect
    print(f"\n{label}")
    print(f"  charge missing per event : {r['missing_per_event']:+.3f}"
          f"   ({r['as_fraction_of_input']:+.1%} of input)")
    print(f"  significance             : {r['z']:.1f} sigma")
    print(f"  verdict                  : {'POSIT a carrier' if r['posit'] else 'nothing to posit'}"
          f"   {'PASS' if ok else 'FAIL'}")
    if r["posit"]:
        # what would have to be true of the thing being posited
        carried = r["missing_per_event"]
        print(f"  the posited object must   : carry {carried:+.3f} of charge per event,")
        print(f"                              and be invisible to every probe we have")
    return ok


if __name__ == "__main__":
    print("L4 -- posit an object to close a ledger that does not balance")
    print(f"charge per species: {dict(enumerate(Q))}, species 2 invisible to all probes")
    a = run("A DECAY WITH AN UNSEEN PRODUCT  (should posit)", 0.30, True)
    b = run("CONTROL -- the books balance     (must not posit)", 0.00, False)
    c = run("CONTROL -- imbalance within noise (must not posit)", 0.01, False)
    print(f"\n{'PASS' if (a and b and c) else 'FAIL'}: "
          f"posited when something was missing = {a}, stayed silent otherwise = {b and c}")
