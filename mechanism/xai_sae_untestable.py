"""Why 191 of X2's 600 selected features have no computable consequence test.

`xai_sae.py` judges a feature on D_cons, the last 20% of the token stream, and needs at least 30
positions (and 12 sequences) whose next token is the feature's concept. A feature that fails this
is reported as untestable and left out of CAR. An earlier comment in that script said every such
feature has exactly zero concept positions in D_cons. This recounts them from the token stream
itself, which needs only the tokenizer and the text, not the models:

    python3 xai_sae_untestable.py        # writes xai_sae_untestable.json

The splits are contiguous blocks of WikiText-2, so the guard split and the consequence split hold
different articles. A concept that is frequent in one article can clear the selection threshold on
D_guard and be absent from D_cons.
"""
import collections
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

import xai_sae as X

HERE = Path(__file__).resolve().parent


def token_stream(name):
    tok = AutoTokenizer.from_pretrained(name)
    return tok, X.get_text(tok, X.N_SEQ, X.SEQ_LEN)


def main():
    res = json.loads((HERE / "xai_sae.json").read_text())
    rows = res["rows"]
    min_seq = res["config"]["min_seq"]

    # every arm uses a GPT-NeoX tokenizer; check that the token streams are identical
    streams = {}
    for name in sorted({m for m, _ in res["config"]["arms"]}):
        tok, ids = token_stream(name)
        streams[name] = ids
    first = next(iter(streams.values()))
    assert all(torch.equal(first, v) for v in streams.values()), "token streams differ by arm"

    n1, n2 = int(len(first) * 0.6), int(len(first) * 0.8)
    guard_next = first[n1:n2, 1:]
    cons_next = first[n2:][:X.N_EVAL, 1:]

    untestable = [r for r in rows if len(r["per_seq"]) < min_seq]
    testable = [r for r in rows if len(r["per_seq"]) >= min_seq]
    pos = lambda t, D: int((D == t).sum())
    seqs = lambda t, D: int((D == t).any(1).sum())

    u_pos = [pos(r["concept"], cons_next) for r in untestable]
    t_pos = [pos(r["concept"], cons_next) for r in testable]
    t_seq = [seqs(r["concept"], cons_next) for r in testable]
    top = collections.Counter(r["concept"] for r in untestable).most_common(1)[0]
    top_tok = top[0]

    out = dict(
        n_selected=len(rows),
        n_untestable=len(untestable),
        untestable_zero=sum(1 for p in u_pos if p == 0),
        untestable_max_positions=max(u_pos),
        testable_min_positions=min(t_pos),
        testable_min_sequences=min(t_seq),
        min_on=X.MIN_ON,
        top_concept=top_tok,
        top_concept_text=tok.decode([top_tok]),
        top_concept_features=top[1],
        top_concept_guard_positions=pos(top_tok, guard_next),
        top_concept_guard_sequences=seqs(top_tok, guard_next),
        top_concept_cons_positions=pos(top_tok, cons_next),
    )
    (HERE / "xai_sae_untestable.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"  {k:<30}{v}")


if __name__ == "__main__":
    main()
