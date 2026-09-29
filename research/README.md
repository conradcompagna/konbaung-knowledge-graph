# Research records

Records of how the graph and dataset were built.

| Path | Contents |
|---|---|
| [evaluation/](evaluation/) | Extraction evaluation: every judged triple and source relationship, with Burmese and English text |
| [notes/METHODOLOGY.md](notes/METHODOLOGY.md) | Textual units, translation, canonical claims, categories, embeddings and validation |
| [notes/ENTITY_RESOLUTION.md](notes/ENTITY_RESOLUTION.md) | Entity resolution: candidate retrieval, supervised ranking and manual review |
| [entity_resolution/](entity_resolution/) | Classifier results, cross-validation folds and review decisions |
| [data-release/](data-release/) | The published V3 dataset: contents, schema, checksums and citation |
| [experiments/](experiments/) | Earlier extraction schemas and repair strategies |
| [datasets/](datasets/) | Label counts from an earlier extraction snapshot |
| [reproduce/served_artifacts.json](reproduce/served_artifacts.json) | Hashes of the snapshots behind the deployed graph |
| [annotation_cost.json](annotation_cost.json) | Token usage and cost of one batch extraction pass |

The deployed snapshots are summarised in [`../docs/BUILD_PROCESS.md`](../docs/BUILD_PROCESS.md).
