# Verification

## Model-free unit tests

```sh
python -m unittest discover -s tests/unit -v
```

These eight tests use synthetic text to check exact source spans, repeated matches, UTF-16 offsets, dictionary display fields, and cross-page evidence projection. They need only Python 3.12.

## Integration tests

`integration/` contains the corpus/graph checks and reader smoke scripts. Additional API and UI specifications live in `konbaung_reader_app/tests/`. These require the canonical dataset, graph store, and a separately configured local reader. They are not included in the dataset-free CI job.

CI runs the unit tests, Python formatting and undefined-name checks, authored JavaScript syntax checks, and the TypeScript/Vite frontend build. This verifies the software sources and build; it does not replace corpus-backed or browser integration testing.
