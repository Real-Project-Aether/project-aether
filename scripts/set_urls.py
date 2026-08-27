#!/usr/bin/env python3
"""Fill every placeholder URL once you have decided the GitHub owner and repository name.

    python3 scripts/set_urls.py <owner> <repo>          e.g.  quotient-group prize-corpus

Rewrites the site's links, the README clone line, the citation metadata and the paper's
availability sentence, so the published URL appears in exactly one place per artifact and
nowhere is left pointing at a placeholder. Safe to re-run: it also matches already-set URLs.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main(owner, repo):
    repo_url = f"https://github.com/{owner}/{repo}"
    pages_url = f"https://{owner}.github.io/{repo}/"
    changed = []

    # --- the website: bare placeholders, and any previously-set value
    site = ROOT / "docs" / "index.html"
    t = site.read_text()
    t = t.replace('href="https://github.com/">Open an issue', f'href="{repo_url}/issues">Open an issue')
    t = re.sub(r'href="https://github\.com/(?:[\w.-]+/[\w.-]+)?"(?=>Code)', f'href="{repo_url}"', t)
    t = t.replace('href="https://github.com/"', f'href="{repo_url}"')
    site.write_text(t); changed.append("docs/index.html")

    # --- README clone line
    rd = ROOT / "README.md"
    t = rd.read_text()
    t = re.sub(r"https://github\.com/\S+?/prize-corpus\.git", f"{repo_url}.git", t)
    t = re.sub(r"git clone \S+ && cd \S+", f"git clone {repo_url}.git && cd {repo}", t)
    rd.write_text(t); changed.append("README.md")

    # --- citation metadata
    cf = ROOT / "CITATION.cff"
    t = cf.read_text()
    if "repository-code:" in t:
        t = re.sub(r"repository-code: .*", f"repository-code: {repo_url}", t)
    else:
        t = t.replace("license: CC-BY-4.0",
                      f"license: CC-BY-4.0\nrepository-code: {repo_url}\nurl: {pages_url}")
    cf.write_text(t); changed.append("CITATION.cff")

    # --- the paper's availability sentence
    tex = ROOT.parent / "paper" / "main.tex"
    if tex.exists():
        t = tex.read_text()
        t = t.replace("PROJECT-URL-PLACEHOLDER", pages_url)
        t = re.sub(r"https://[\w.-]+\.github\.io/[\w.-]+/", pages_url, t)
        tex.write_text(t); changed.append(str(tex))

    print(f"repository : {repo_url}\nwebsite    : {pages_url}\n")
    for c in changed:
        print("  updated", c)
    left = sum(f.read_text().count('https://github.com/"') for f in [site])
    print(f"\nplaceholders remaining: {left}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
