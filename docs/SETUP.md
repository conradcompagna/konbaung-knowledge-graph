# Setup and external resources

The source tree preserves the pipeline's relative layout. A fresh clone contains neither the proprietary corpus nor a graph database, and therefore does not provide a working corpus-backed reader by itself.

## Reader application

Use Python 3.12 and Node.js. Create and activate a virtual environment. Install the runtime requirements from `konbaung_reader_app/requirements.txt` and the Burmese reader's dependencies when using dictionary segmentation.

Clone `burmese-neural-reader` beside this repository, or set `BURMESE_READER_ROOT` to its root. The existing adapter imports `newserverPDF21split.py` and initializes only its dictionary segmentation path. The Burmese dictionaries must be provisioned separately.

From `konbaung_reader_app`:

```sh
python -m pip install -r requirements.txt
npm ci
npm run build:frontend
```

Before `python app.py`, provision the corpus, graph store, and embeddings expected by `graph_schema.py`, `corpus.py`, and `axial_store.py`. These files deliberately remain external:

- `konbaung_reader_app/static/data/konbaung/`: canonical page material.
- `konbaung_reader_app/data/konbaung_historiography_v3_canonical_20260724/`: canonical V3 annotations.
- `konbaung_reader_app/data/konbaung_knowledge_graph_v3/`: generated Oxigraph database, graph export, and overview artifacts.
- The root embedding and occurrence directories referenced by `graph_schema.py`.

The browser application listens on port 5077. The `Procfile` and `deploy/` directory preserve hosting infrastructure. Configure a separate instance and its own paths before deployment.

## Pipeline inputs and credentials

Place your own environment configuration in this repository's `.env` or export it to the process. Pipeline helpers and the reader no longer discover credentials in the separate Language Engine workspace. OCR uses its own `konbaung-google-ocr/.env` with `GOOGLE_API_KEY`.

Extraction, translation, embeddings, and classification issue paid cloud requests when explicitly run. Read each script's CLI and selected input/output paths before execution. `requirements-research.txt` lists the additional libraries imported by research and analysis scripts; it does not supply model weights or datasets.

## Tests

Python tests under `tests/` and `konbaung_reader_app/tests/` and Playwright specifications describe corpus, API, and UI behavior. Most graph tests depend on the omitted canonical dataset. Do not interpret source syntax checks or a frontend build as a completed corpus-backed integration test.

The database build refuses to overwrite an existing graph directory. Use separately provisioned development resources, never the live website's storage, for testing or rebuilding.
