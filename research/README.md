# Graph and dataset construction

The [construction guide](../docs/BUILD_PROCESS.md) follows the source pages through
OCR, sentence reconstruction, extraction, embeddings, and graph assembly.

| Record | Contents |
|---|---|
| [Methodology](notes/METHODOLOGY.md) | Textual units, translation, canonical claims, categories, embeddings, and validation |
| [Entity-resolution case study](notes/ENTITY_RESOLUTION.md) | Candidate retrieval, supervised ranking, and source-based review |
| [Entity-resolution evaluation](entity_resolution/) | Classifier results, cross-validation folds, and review decisions |
| [Extraction experiments](experiments/) | The development of extraction schemas and repair strategies |
| [Label inventories](datasets/) | Extraction-label counts and frequency summaries used in resolution development |
| [V3 data release](data-release/README.md) | Dataset contents, schema, checksums, and citation |
| [Served artifacts](reproduce/served_artifacts.json) | Verified identities and input hashes for the deployed graph and categories |
| [Annotation cost](annotation_cost.json) | Token usage and historical cost for a batch extraction pass |

The [pipeline guide](../pipeline/README.md) maps these stages to their source modules.
