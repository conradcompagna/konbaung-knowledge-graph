# Konbaung Chronicle Knowledge Graph

**An end-to-end historical research system: Burmese chronicle scans → structured claims → an evidence-linked reader, knowledge graph, and public API.**

I built this project to investigate how power operated in the Konbaung dynasty through kingship, office, military command, religious patronage, tribute, and kinship. It turns a large primary source into inspectable research data while connecting interpretations to their source sentences and pages.

[Live reader](https://burmeseneuralreader.com/chronicles/vol1/47) · [API documentation](https://burmeseneuralreader.com/api/v1/docs) · [Setup](docs/SETUP.md) · [Portfolio](https://github.com/conradcompagna)

[![Checks](https://github.com/conradcompagna/konbaung-knowledge-graph/actions/workflows/checks.yml/badge.svg)](https://github.com/conradcompagna/konbaung-knowledge-graph/actions/workflows/checks.yml)

## From source pages to exploration

```mermaid
flowchart TB
    subgraph Build["Construct the research artifacts"]
        Scans["Page images"] --> OCR["OCR, restoration and sentence reconstruction"]
        OCR --> Extract["Translation and V3 claim extraction"]
        Extract --> Canonical["Select one annotation per sentence"]
        Canonical --> Embeddings["Eight-view embeddings and graph features"]
        Canonical --> Categories["52 entity / 81 relation categories"]
        Canonical --> Graph["Oxigraph RDF and source-linked claims"]
        Embeddings --> Graph
    end
    subgraph Use["Explore the evidence"]
        Graph --> Explorer["Graph explorer and versioned API"]
        Categories --> Explorer
        Explorer --> Reader["Claim, sentence and source-page reader"]
        Burmese["Burmese dictionary segmentation"] --> Reader
        Gloss["Optional contextual Gemini glosses"] --> Reader
    end
    Canonical -.-> Analysis["Research variants:<br/>resolution, review and statistics"]
    Embeddings -.-> Analysis
    Categories -.-> Analysis
```

Offline builds create the data used by the reader; opening the graph does not
rerun extraction or embedding generation. Research branches record their own
versioned input snapshots. The
[construction and version guide](docs/BUILD_PROCESS.md) connects the served
artifacts to their builders, hashes and research branches.

## Engineering highlights

- **Complete extraction infrastructure:** page-image OCR, sentence reconstruction, translation, structured LLM annotation, schema validation, targeted repair, and corpus compilation.
- **Traceable evidence:** canonical page text, source hashes, sentence identifiers, exact Unicode/UTF-16 span alignment, cross-page routing, and deduplication.
- **Substantial knowledge representation:** the deployed manifest records **27,129 canonical claims across 1,215 pages and three volumes**, with 23,890 entity labels and 11,886 relation labels.
- **Research access and analysis:** embedded Oxigraph/RDF storage, embedding-based exploration, entity-resolution workflows, a worker-based graph interface, and a versioned read-only API with OpenAPI documentation.

## Explore the code

| Stage | Starting point |
|---|---|
| OCR | [konbaung-google-ocr/src/](konbaung-google-ocr/src/) |
| Corpus reconstruction and compilation | [pipeline/corpus/](pipeline/corpus/) |
| Translation and claim extraction | [pipeline/translation/](pipeline/translation/), [pipeline/extraction/](pipeline/extraction/), [prompts/](prompts/) |
| Embeddings and entity resolution | [pipeline/embeddings/](pipeline/embeddings/), [pipeline/resolution/](pipeline/resolution/) |
| Graph storage and API | [graph_store.py](konbaung_reader_app/graph_store.py), [public_api.py](konbaung_reader_app/public_api.py) |
| Reader and graph interface | [app.py](konbaung_reader_app/app.py), [frontend/](konbaung_reader_app/frontend/) |
| Statistical analysis | [research/analysis/](research/analysis/) |
| Source alignment and integration checks | [tests/](tests/) |

The [pipeline guide](pipeline/README.md) connects these stages, from model-assisted
extraction to source inspection, entity review, and graph exploration. The
[graph interface guide](konbaung_reader_app/frontend/graph/README.md) maps the
frontend by navigation, rendering, filtering, layout, and evidence responsibilities.

---

## How it was built

The research package connects the system's construction to its historical purpose:
examining the relationships among office, command, patronage, tribute, and kinship.
It includes the methods, analysis code, selected findings, figures, and run records.

**[`research/`](research/)** — start at [research/README.md](research/README.md).

| | |
|---|---|
| [**Construction and selected artifacts**](docs/BUILD_PROCESS.md) | The source-to-product story, served V3 snapshots, final categories and reconstruction checklist. |
| [**Methodology**](research/notes/METHODOLOGY.md) | Corpus construction, extraction design, eight embedding views, and validation. |
| [**Pipeline guide**](pipeline/README.md) | Every stage from page image to canonical graph, and why extraction and resolution are shaped the way they are. |
| [**Reproduction record**](research/reproduce/) | The analysis runner, twelve per-stage logs, output checksums, input provenance, and data validation. |
| [`research/findings/`](research/findings/) | Model comparisons, relation-layer structure, stability estimates, and validation results. |
| [`research/figures/`](research/figures/) | The four figures, PNG and SVG. |
| [`research/experiments/`](research/experiments/) | Schema development across seven annotator generations and targeted-repair strategies. |
| [`research/datasets/`](research/datasets/) | Earlier research-snapshot label distributions; the served V3 graph counts are listed above. |
| [`research/notes/`](research/notes/) | Historical interpretation, statistical methods, and the evaluation record. |

For a short route through the findings, read the
[historical interpretation](research/notes/PRELIMINARY_HISTORICAL_INTERPRETATION.md)
and the [analysis results](research/notes/NEW_ANALYSES_AND_RESULTS.md). The latter
connects each result to its model, source context, and validation checks.

## Run the lightweight checks

```sh
python -m unittest discover -s tests/unit -v
cd konbaung_reader_app
npm ci
npm run check:types
npm run test:graph
npm run build:frontend
```

Eight Python unit tests cover source spans, UTF-16 offsets, dictionary display
fields, and cross-page projection using synthetic text. Eight graph regressions
cover controller state, navigation, filters, and layout-worker responses. The
frontend build runs these graph checks, TypeScript checking, and the 2,000-line
source limit before bundling.

The source release includes software, methods, and selected research outputs. Full
corpus and model assets are provisioned separately; see [setup](docs/SETUP.md) and
[publication contents](docs/PUBLICATION.md).
