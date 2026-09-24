# Label inventories

The extraction produced an open label inventory: entity types and relation predicates
were proposed by the model per page and then canonicalised, rather than being chosen
from a fixed schema in advance. Understanding that inventory's shape matters for
judging the extraction.

| File | Contents |
|---|---|
| `entity_labels_distribution.json` | 5,667 distinct entity labels over 46,826 occurrences, bucketed by frequency |
| `entity_labels_top150.tsv` | the 150 most frequent entity labels |
| `relation_labels_distribution.json` | 13,727 distinct relation labels over 23,413 occurrences, bucketed by frequency |
| `relation_labels_top150.tsv` | the 150 most frequent relation predicates |

The distributions are the point. Both inventories have a long tail of hapax labels —
2,685 entity labels and the great majority of relation labels occur exactly once. That
is what an open-vocabulary extraction produces, and it is the reason the pipeline has a
canonicalisation stage at all: `pipeline/resolution/` and `pipeline/embeddings/` exist
to collapse that tail.

These distributions describe the earlier research snapshot used in the resolution
work. The [construction guide](../../docs/BUILD_PROCESS.md) identifies the served V3
snapshot and its corresponding counts.
