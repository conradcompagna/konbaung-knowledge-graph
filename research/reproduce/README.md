# Reproduction record

The analysis ran as eleven numbered stages plus figures, finalisation and an audit
catalogue. This directory holds the evidence that it ran, in what environment, over
what inputs, and with what output.

| File | What it records |
|---|---|
| `RUN_ANALYSES.ps1` | the runner, and therefore the stage order |
| `run_logs/NN_run.log` | stdout for each stage, twelve logs |
| `OUTPUT_CHECKSUMS.sha256` | SHA-256 of every output file the run produced |
| `input_provenance.json` | which input files the run consumed, and their identity |
| `data_validation.json` | the pre-run validation of those inputs |
| `original_source_verification.json` | verification against the original source scans |
| `source_recovery.json`, `recovery_log.txt` | recovery of source text for passages where the corpus and the scans disagreed |
| `pdf_inventory.json` | the source PDF inventory |
| `inventory_errors.json` | inventory problems found and how they were resolved |

`../findings/11_environment.json` records the Python and library versions.
`../findings/11_validation.json` is the post-run validation.

Stage order, from the runner:

```
01 contingency            07 brokerage, paths, hierarchy
02 graph layers           08 predictive check
03 tensor semantics       09 motif sensitivity and sources
04 paths, rules, motifs   10 audit catalogue
04b degree null           11 validate
05 network models         12 figures
06 robustness comparison  13 write reports · 14 finalise
```

The checksum file is the useful artefact here: it fixes exactly which output each
stage produced, so a re-run on the same inputs can be compared against it rather than
inspected by eye.
