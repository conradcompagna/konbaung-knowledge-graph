# Label inventories

Counts of the raw entity and relation labels produced by an earlier extraction
snapshot, before entity resolution. The extraction proposed labels freely per page
rather than choosing from a fixed list.

| File | Contents |
|---|---|
| `entity_labels_distribution.json` | 5,667 distinct entity labels over 46,826 occurrences, grouped by frequency |
| `entity_labels_top150.tsv` | The 150 most frequent entity labels |
| `relation_labels_distribution.json` | 13,727 distinct relation labels over 23,413 occurrences, grouped by frequency |
| `relation_labels_top150.tsv` | The 150 most frequent relation labels |

Both inventories have a long tail of labels that occur once (2,685 entity labels and
most relation labels), which is what the resolution code in `pipeline/resolution/`
reduces. The served V3 graph has 23,890 entity labels and 11,886 relation labels; see the
[root README](../../README.md).
