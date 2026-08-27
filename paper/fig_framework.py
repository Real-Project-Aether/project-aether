"""The conceptual framework: what a model has to be able to do, and how much of it is solved.

Two axes, both read off the corpus. A move either changes the FACTS you hold, the INSTRUMENTS
you have, or the DESCRIPTION you use; and a change to the description has a depth, from ordinary
parameter search up to a change in the repertoire of moves itself.

Ordinary machine learning implements exactly one rung. The point of drawing it this way is that
the rung it implements is 18% of the depth-graded events in the corpus.
"""
from __future__ import annotations
import collections, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.join(os.path.dirname(HERE), "release")

plt.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 200})
INK, GREY, ACCENT, PALE = "#1a1a1a", "#8a8a8a", "#b5504a", "#e8ded9"


def counts():
    rows = list(csv.DictReader(open(os.path.join(REL, "classifications.csv"))))
    modes = collections.Counter(r["mode"] for r in rows)
    lv = collections.Counter()
    for r in rows:
        for x in (r["levels"] or "").split("|"):
            if x:
                lv[x] += 1
    return modes, lv, len(rows)


# Names match the project site exactly, so a reader moving between the two never has to
# translate: the short name is what the rung is called, the gloss says what it means.
# Glosses are length-checked against the divider at SPLIT; see the assertion in main().
LADDER = [
    ("L0",  "Regression",      "solve inside the existing description"),
    ("L1",  "Reinterpretation", "same predictions, less structure"),
    ("L2",  "Retyping",        "two things are one, or one is two"),
    ("L3",  "Transfer",        "one field\u2019s language into another"),
    ("L4",  "New object",      "a kind of thing the language lacked"),
    ("L5?", "New move",        "change what counts as a move"),
]


def main():
    """Layout notes, both bugs found by looking at the render rather than the code.
    The ladder was drawn top-down, which put L0 -- the ground floor -- at the top and the
    "everything ML does" marker around L5. And the right-hand column started at x=0.655 while
    the ladder's descriptions ran past it, so the two collided."""
    modes, lv, n = counts()
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    SPLIT = 0.615
    ax.text(0.005, 0.965, "CHANGE THE DESCRIPTION", fontsize=7.5, color=INK, weight="bold")
    ax.text(0.335, 0.965, f"{modes['REPRESENT']} events", fontsize=6.8, color=GREY)

    y0, dy, maxn = 0.085, 0.152, max(lv[k] for k, _, _ in LADDER)
    rows = []                                 # ladder text, checked against the divider below
    for i, (lab, name, gloss) in enumerate(LADDER):          # L0 at the bottom, as a ladder
        y = y0 + i * dy
        c = lv[lab]
        w = 0.085 * (c / maxn) ** 0.5
        solved = lab == "L0"
        ax.add_patch(Rectangle((0.072, y), max(w, 0.006), 0.056,
                               fc=PALE if solved else ACCENT, ec="none"))
        ax.text(0.065, y + 0.028, lab, ha="right", va="center", fontsize=7.5,
                color=INK, weight="bold")
        ax.text(0.168, y + 0.052, name, va="center", fontsize=7.4, color=INK, weight="bold")
        rows.append(ax.text(0.168, y + 0.006, f"{c} event{'s' if c != 1 else ''}   — {gloss}",
                            va="center", fontsize=6.5, color=GREY))

    ax.add_patch(FancyBboxPatch((0.060, y0 - 0.022), 0.545, 0.098,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc="none", ec=GREY, lw=0.8, linestyle=(0, (3, 2))))
    ax.text(0.328, y0 - 0.072, "everything machine learning does today  —  1 rung of 6, 131 of 1,041 labels",
            ha="center", fontsize=6.6, color=GREY, style="italic")

    ax.plot([SPLIT, SPLIT], [0.02, 0.93], color="#cccccc", lw=0.7)
    for j, (title, k, gloss) in enumerate([
            ("CHANGE THE FACTS", "RECORD", "measure what was not\npreviously in hand"),
            ("CHANGE THE INSTRUMENTS", "REACH", "make observable what could\nnot be observed at all")]):
        y = 0.60 - j * 0.36
        ax.text(SPLIT + 0.03, y + 0.135, title, fontsize=7.5, color=INK, weight="bold")
        ax.add_patch(Rectangle((SPLIT + 0.03, y + 0.075), 0.30 * modes[k] / modes["RECORD"], 0.042,
                               fc=ACCENT, ec="none"))
        ax.text(SPLIT + 0.03, y + 0.042, f"{modes[k]} events", fontsize=6.8, color=GREY)
        ax.text(SPLIT + 0.03, y - 0.005, gloss, fontsize=6.9, color=INK, va="top")

    # The left column ran under the right one once already, because a gloss got longer than the
    # gap. Measure the rendered extents and fail loudly rather than shipping a collision.
    rend = fig.canvas.get_renderer()
    for txt in rows:
        x1 = txt.get_window_extent(renderer=rend).transformed(ax.transData.inverted()).x1
        assert x1 <= SPLIT - 0.005, (
            f"ladder text runs past the divider ({x1:.3f} > {SPLIT}): {txt.get_text()[:60]!r}")
    print(f"  widest ladder row ends at "
          f"{max(t.get_window_extent(renderer=rend).transformed(ax.transData.inverted()).x1 for t in rows):.3f}"
          f"  (divider {SPLIT})")

    fig.savefig(os.path.join(HERE, "fig_framework.pdf"), bbox_inches="tight")
    solved = lv["L0"]; above = sum(lv[k] for k, _, _ in LADDER) - solved
    print(f"  ladder: L0={solved}, above={above} ({100*above/(solved+above):.0f}% above ML's rung)")


if __name__ == "__main__":
    main()
