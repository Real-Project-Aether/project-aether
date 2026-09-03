# Vendored from SHAPE-of-CoT

Files in this directory are copied unmodified from
https://github.com/holi-lab/SHAPE-of-CoT (Song et al., 2026, arXiv:2608.28600),
cloned 2026-09-02.

    heuristics_guide.md       the heuristic coding manual given to the annotator
    semantic_space_guide.md   the semantic-space tracking manual
    shape_metrics.py          their canonical metric code (N_space_eff, N_trans_eff)

They are vendored so that `shape_audit.py` uses the authors' own coding manuals
and metric definitions rather than our paraphrase of them. The audit is of the
instrument as its authors specify it.

Two deviations from their setup are forced by ours and are recorded in the audit
output:

  * their annotator is Qwen3.5-27B served with vLLM; vLLM is not installed here
    and a 27B model does not fit the available GPU, so the audit uses
    Qwen2.5-7B-Instruct through transformers. Annotator capacity is therefore
    lower than theirs. The audit compares conditions under the SAME annotator,
    and includes a positive control so that a null result can be told apart from
    an annotator that cannot do the task at all.
  * the repository guidebook uses a 13-heuristic taxonomy with trigger set
    {H1, H2, H5, H6, H10, H13}; the paper states 11 with {H1, H2, H3, H5, H8, H11}.
    We follow the repository, since that is the shipped code.
