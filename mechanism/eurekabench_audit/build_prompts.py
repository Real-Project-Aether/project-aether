"""Rebuild the judge prompts of the EurekaBench insight-judge audit.

The prompts are EurekaBench's own `tests/judges/insights_strict.md`, filled the way its verifier
fills them (criteria listed as "- I1: ..."), with one sentence appended because the simulator is
not available here. They are not shipped; this rebuilds them from a EurekaBench checkout:

    git clone https://github.com/EurekaBench/EurekaBench && git -C EurekaBench checkout 58956ee
    python3 build_prompts.py EurekaBench out/

Each submission is copied to out/<hash>/submission/ under a hashed directory name, so a judge
cannot read the null type from the path; out/key.json maps hashes back to submissions.
"""
import hashlib, json, shutil, sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TASK = {"A": "fusion/nt-edge", "B": "neuroscience/zigzag-foraging"}
NOTE = ("\n\nNote on this environment: the coupled model / simulator is not available here, so you cannot run "
        "experiments. Judge from the files alone, under the same rules.\n")


def criteria(section):
    cs = [c for c in section["criteria"] if not c.get("machine")] or section["criteria"]
    return "\n".join(f"- {c['id']}: {' '.join(str(c['criterion']).split())}" for c in cs)


def main(eb, out):
    eb, out = Path(eb) / "domains", Path(out)
    key = {}
    for sub in sorted(p.name for p in (HERE / "submissions").iterdir()):
        task = TASK[sub[0]]
        template = (eb / task / "tests/judges/insights_strict.md").read_text()
        rubric = yaml.safe_load((eb / task / "tests/rubric.yaml").read_text())
        for rep in (1, 2):
            h = hashlib.sha256(f"{sub}-{rep}-eb-audit-2026".encode()).hexdigest()[:10]
            ws = out / h
            (ws / "submission").mkdir(parents=True, exist_ok=True)
            (ws / "tmp").mkdir(exist_ok=True)
            for f in ("mechanism.md", "experiment.log"):
                shutil.copy(HERE / "submissions" / sub / f, ws / "submission" / f)
            prompt = template.format(
                mechanism_md=str(ws / "submission/mechanism.md"), experiment_log=str(ws / "submission/experiment.log"),
                submission_extras="none", output_dir=str(ws), tmp_dir=str(ws / "tmp"), judge_output="judgement.json",
                insights_criteria=criteria(rubric["insights"]), time_budget="2 hours") + NOTE
            (ws / "prompt.txt").write_text(prompt)
            key[h] = {"submission": sub, "replicate": rep, "task": task}
    (out / "key.json").write_text(json.dumps(key, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
