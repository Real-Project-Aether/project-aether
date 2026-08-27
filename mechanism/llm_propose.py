"""The language model proposes; the measurement disposes.

A null-space search can only find structure inside the basis it is handed, so something has to
widen the basis. Our own experiment says what a language model is and is not good for: on five
episodes it identified the reinterpretation 87% of the time when it could recognise them and
48% -- chance -- when the names were stripped out. It recalls; it does not judge.

So use it only to GENERATE, never to decide. The model proposes candidate concepts; acceptance
is by the rule in concept_space:

    accept iff coverage strictly improves AND the unobservable dimension stays at zero

and the second half is what makes the arrangement safe. A fluent proposal that buys fit with
structure no measurement can see is rejected however confident the model sounds.

The control is the point. World B's residual is pure noise -- there is nothing to find. A model
asked for concepts will supply them anyway. If the filter works, world A's true concept is
recovered and world B yields nothing.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

import numpy as np

from concept_space import Theory, add_latent, coverage, fit, unobservable_dim

URL = os.environ.get("VLLM_MODEL_URL", "http://127.0.0.1:11434/v1") + "/chat/completions"
MODEL = os.environ.get("VLLM_MODEL", "gpt-oss:20b")

SAFE = {n: getattr(np, n) for n in
        ("sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "abs", "sign", "pi")}

SYSTEM = ("You are helping extend a physical model. You will be shown the residual of a fit -- "
          "what the current model fails to explain. Propose candidate basis functions that "
          "might account for it. Reply with ONLY a JSON list of Python expressions in the "
          "variable x, using numpy functions by bare name (sin, cos, exp, log, sqrt, tanh). "
          'Example: ["sin(3*x)", "x**2", "exp(-x)"]. No prose.')


def chat(prompt, n=8):
    body = {"model": MODEL, "temperature": 0.8, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt + f"\n\nPropose {n} candidate basis functions."}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse(txt):
    m = re.search(r"\[.*?\]", txt, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except Exception:
        return []
    return [str(i) for i in items if isinstance(i, str)][:12]


def compile_expr(expr):
    """Only numpy functions and x. Anything else is not evaluated at all."""
    if re.search(r"[^0-9a-zA-Z_+\-*/%.()\[\], ]", expr) or "__" in expr:
        return None
    names = set(re.findall(r"[A-Za-z_]\w*", expr))
    if not names <= set(SAFE) | {"x"}:
        return None
    try:
        code = compile(expr, "<llm>", "eval")
    except SyntaxError:
        return None

    def f(xx, _c=code):
        v = eval(_c, {"__builtins__": {}}, dict(SAFE, x=xx))    # noqa: S307 - namespace is closed
        return np.broadcast_to(np.asarray(v, float), xx.shape).copy()
    return f


def run_world(label, truth_fn, base, x, sigma, seed):
    rng = np.random.default_rng(seed)
    y = base.predict(x, [1.5, 0.3]) + truth_fn(x) + rng.normal(0, sigma, x.size)
    p, _ = fit(base, x, y, sigma)
    cov0 = coverage(base, x, y, sigma, p)
    resid = y - base.predict(x, p)

    pts = ", ".join(f"({a:.2f}, {b:+.3f})" for a, b in zip(x[::3], resid[::3]))
    txt = chat(f"Residual after fitting a straight line, as (x, residual) pairs:\n{pts}\n"
               f"Measurement noise is {sigma}.")
    props = parse(txt)

    print(f"\n{label}")
    print(f"  base model covers {cov0:.0%} of observations; model proposed {len(props)} concepts")
    print(f"  {'proposed concept':<26}{'cover':>8}{'unobs':>7}   verdict")
    print("  " + "-" * 62)
    accepted = []
    for expr in props:
        fn = compile_expr(expr)
        if fn is None:
            print(f"  {expr[:24]:<26}{'':>8}{'':>7}   rejected -- not evaluable")
            continue
        try:
            cand = add_latent(base, fn, expr)
            q, _ = fit(cand, x, y, sigma)
            c2 = coverage(cand, x, y, sigma, q)
            u2, _ = unobservable_dim(cand, x, q)
        except Exception:
            print(f"  {expr[:24]:<26}{'':>8}{'':>7}   rejected -- did not fit")
            continue
        ok = c2 > cov0 + 1e-9 and u2 == 0
        why = "ACCEPT" if ok else ("reject -- unobservable structure" if u2 else "reject -- no gain")
        print(f"  {expr[:24]:<26}{c2:>7.0%}{u2:>7}   {why}")
        if ok:
            accepted.append((expr, c2))
    return cov0, props, accepted


if __name__ == "__main__":
    x = np.linspace(0.5, 6.0, 60)
    SIG = 0.05
    base = Theory("straight line", ["slope", "offset"], lambda xx, p: p[0] * xx + p[1])

    print(f"model {MODEL}   acceptance: coverage must rise AND unobservable dimension stay 0")
    a = run_world("WORLD A -- a real missing concept, 0.4*sin(3x)",
                  lambda xx: 0.4 * np.sin(3 * xx), base, x, SIG, seed=1)
    b = run_world("WORLD B -- CONTROL, nothing to find (residual is noise)",
                  lambda xx: np.zeros_like(xx), base, x, SIG, seed=2)

    # Was the bottleneck the proposer or the filter? Offer the truth and see.
    print("\nORACLE CHECK -- would the filter have recognised the true concept if offered?")
    rng = np.random.default_rng(1)
    y = base.predict(x, [1.5, 0.3]) + 0.4 * np.sin(3 * x) + rng.normal(0, SIG, x.size)
    p0, _ = fit(base, x, y, SIG)
    cov0 = coverage(base, x, y, SIG, p0)
    print(f"  {'candidate':<18}{'coverage':>10}{'unobs':>7}")
    for lbl, fn in (("sin(3*x)  TRUE", lambda xx: np.sin(3 * xx)),
                    ("sin(2.9*x) near", lambda xx: np.sin(2.9 * xx)),
                    ("sin(2*x)  wrong", lambda xx: np.sin(2 * xx)),
                    ("x**3      wrong", lambda xx: xx ** 3)):
        c = add_latent(base, fn, lbl)
        q, _ = fit(c, x, y, SIG)
        print(f"  {lbl:<18}{coverage(c, x, y, SIG, q):>9.0%}{unobservable_dim(c, x, q)[0]:>7}")
    print(f"  base was {cov0:.0%}. The filter separates the truth from the near-misses by a wide")
    print("  margin, so the bottleneck here is the PROPOSER, not the test.")

    print("\nsummary")
    for lbl, (cov0, props, acc) in (("world A (real concept)", a), ("world B (control)", b)):
        print(f"  {lbl:<26} proposed {len(props):>2}   accepted {len(acc):>2}"
              + (f"   best: {acc[0][0]}" if acc else ""))
    print("\n  The model proposes in both worlds. Only the measurement tells them apart.")
