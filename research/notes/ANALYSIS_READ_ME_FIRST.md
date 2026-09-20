# Konbaung analysis audit and new results

> Moved here from the September 2026 analysis run directory. Relative references
> below have been repointed at the published copies in `../findings/`,
> `../figures/` and `../reproduce/`. References marked *not published* are to
> full analysis outputs that stay local; see [../README.md](../README.md) for what
> is published and why.


The audit covers all 26 families in the pasted catalogue. The new analyses investigate how different actors, relations and recipients organize the chronicle's account of power. The strongest results concern differentiated roles and relation-layer structure; stronger causal claims fail several of the new controls.

- [Preliminary historical interpretation](PRELIMINARY_HISTORICAL_INTERPRETATION.md)
- [Audit of existing tests](AUDIT_OF_EXISTING_TESTS.md)
- [New analyses and results](NEW_ANALYSES_AND_RESULTS.md)
- Interactive report and searchable enumeration (`index.html`, not published)
- [26-family audit CSV](../findings/10_family_audit.csv)
- 380-item method checklist CSV (`results/10_requested_method_checklist.csv`, not published)
- Source passages in Burmese and English (`source_passages.html`, not published)

**Scope:** 67 named alternatives remain explicitly unrun, and historical event-time/spatial models require additional validated inputs. This is a completed audit and a substantial new test battery, not a claim that every algorithm in the catalogue has been executed.

**Source:** the intact `DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831.zip` in the original `dighumproject` directory. Its canonical data match the article archive. Originals were left unchanged. Reproducible code, raw result tables, source hashes, seeds and 126 passing checks are included.

Run [RUN_ANALYSES.ps1](../reproduce/RUN_ANALYSES.ps1) from this directory to reproduce the new analyses using the verified extracted inputs. The tensor package is in `vendor`; the installed scientific Python versions are recorded in `results/11_environment.json`. `scripts/inventory.py`, `prepare.py`, and `supplement_inventory.py` document the source inventory/extraction phase. The original archive is required to repeat source verification. No paid API calls are required.
