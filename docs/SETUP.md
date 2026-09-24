# Setup and external resources

The source release supports frontend builds and model-free checks directly. To run
the corpus-backed reader, provision the graph and language resources described below;
the [live reader](https://burmeseneuralreader.com/chronicles/vol1/47) demonstrates the
configured application.

## Reader application

Use Python 3.12 and Node.js 22. Create and activate a virtual environment. Install the runtime requirements from `konbaung_reader_app/requirements.txt` and the Burmese reader's dependencies when using dictionary segmentation.

Clone `burmese-neural-reader` beside this repository, or set `BURMESE_READER_ROOT` to its root. The existing adapter imports `app.py` and initializes only its dictionary segmentation path. The Burmese dictionaries must be provisioned separately.

From `konbaung_reader_app`:

```sh
python -m pip install -r requirements.txt
npm ci
npm run build:frontend
```

The build writes ignored files under `static/build/`. The reader template loads `build/graph.js`; deploy these generated files with the application after building.

Before `python app.py`, provision the corpus, graph store, and embeddings expected by `graph_schema.py`, `corpus.py`, and `axial_store.py`. These files deliberately remain external:

- `konbaung_reader_app/static/data/konbaung/`: canonical page material.
- `konbaung_reader_app/data/konbaung_historiography_v3_canonical_20260724/`: canonical V3 annotations.
- `konbaung_reader_app/data/konbaung_knowledge_graph_v3/`: generated Oxigraph database, graph export, and overview artifacts.
- `konbaung_reader_app/data/konbaung_axial_categories_v2/`: final category assignments.
- The root embedding and occurrence directories referenced by `graph_schema.py`.

The [construction checklist](BUILD_PROCESS.md#reconstruction-checklist) and
[selected artifact record](../research/reproduce/served_artifacts.json) identify
the snapshots, models, counts and hashes behind these inputs.

The browser application listens on port 5077. The `Procfile` and `deploy/` directory preserve hosting infrastructure. Configure a separate instance and its own paths before deployment.

## Pipeline inputs and credentials

Place your own environment configuration in this repository's `.env` or export it to the process. Pipeline helpers and the reader no longer discover credentials in the separate Language Engine workspace. OCR uses its own `konbaung-google-ocr/.env` with `GOOGLE_API_KEY`.

Extraction, translation, embeddings, and classification issue paid cloud requests when explicitly run. Read each script's CLI and selected input/output paths before execution. `requirements-research.txt` lists the additional libraries imported by research and analysis scripts; it does not supply model weights or datasets.

## Checks

Run `python -m unittest discover -s tests/unit -v` from the repository root. These eight synthetic source-alignment tests require no private dataset. Install `ruff==0.16.8`, then run `ruff check .` and `ruff format --check .` for the Python checks used in CI.

The tests under `tests/integration/` and `konbaung_reader_app/tests/` require the omitted canonical dataset and a separately configured graph-backed reader. The frontend build and unit tests do not execute those integrations.

The database build refuses to overwrite an existing graph directory. Use a separate development graph store when building or testing. See the [pipeline guide](../pipeline/README.md) for organized entrypoints and resource requirements.
