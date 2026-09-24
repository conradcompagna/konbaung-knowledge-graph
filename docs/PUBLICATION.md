# Repository contents

Start with the [construction story and reconstruction checklist](BUILD_PROCESS.md)
for the selected artifacts and the process connecting them to the application.

## Included

**Runtime.** The reader and graph application, the public API, the frontend source, and
the deployment configuration, under `konbaung_reader_app/`.

**Pipeline.** OCR, corpus reconstruction, translation, claim extraction, audit,
embeddings, entity resolution and review, under `pipeline/`, with the prompts that
drive each stage and the gold standards they were scored against under `prompts/`.

**Research record.** Statistical analysis source, the findings it produced, figures,
the reproduction harness with per-stage logs and output checksums, methodology notes,
label-inventory distributions, and the superseded extraction approaches, under
`research/`.

## Separately provisioned resources

| Excluded | Reason |
|---|---|
| The extracted triples, RDF/N-Quads exports and the Oxigraph database | The research dataset. This is the product of the work and remains proprietary. |
| Source page scans and OCR page output | Third-party scans; 1,587 page images and their OCR responses. The OCR run logs and a sample output **are** published, under `konbaung-google-ocr/run_evidence/`. |
| Corpus exports and sentence corpora | Derived from the scans. |
| Embedding matrices and fastText vectors | Large binaries derived from the corpus. |
| Full analysis output | `research/results/` stays untracked. The curated summary statistics are published in `research/findings/`. |
| `*_anchors.csv` analysis tables | Small, but they quote triples verbatim with their source sentences. They are dataset, not finding. |
| Full entity and relation label inventories | The earlier research inventory has 5,667 and 13,727 labels; the served V3 graph has 23,890 and 11,886 respectively; the complete lists are substantially the extracted data. Distributions and the 150 most frequent of each are published in `research/datasets/`. |
| Manuscript drafts, editorial material, evidence packages | Unpublished historical argument, not software. |

No dataset licence is granted by publishing this software.

Live environment files, credentials, private keys, installed dependencies, local
environments, backup copies and generated frontend output are excluded. Configuration
examples require your own settings. Third-party notices are preserved in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The GitHub code is maintained separately from the deployed sites and original
development workspaces. See [setup](SETUP.md) for inputs, build steps, and validation
commands.
