# Licensing

## What we made — CC BY 4.0

The **annotations** (`classifications.csv`), the compiled metadata (prize registry, award
records excluding citation text, laureate resolution, paper links, document index), and all
code in this release are licensed **CC BY 4.0**. Use them, change them, redistribute them;
please attribute.

## What we did not make — not ours to license

The **official citation text** in `awards.csv` was written by the awarding bodies. It is
included as short quotation for research use: median 130 characters, mean 153, being the
factual statement of what each award was for. Every row carries `official_url` so the source
can be checked. These statements are not ours and are not covered by the licence above.

The **full text** of laureate lectures, prize essays and ceremony documents is **not
redistributed here at all**. `documents_index.csv` gives the source URL of each; use
`refetch_documents.py` to obtain them directly from the awarding bodies, under their terms.

## If you are an awarding body

If you would prefer your citation text not appear in this release, contact
tianyu16@illinois.edu and it will be removed — the corpus stays usable without it, since every
row keeps its `official_url`.
