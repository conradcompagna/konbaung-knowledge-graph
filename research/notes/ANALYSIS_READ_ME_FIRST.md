# Analyzing the chronicle's account of power

The analysis combines contingency models, relation-layer comparisons, tensor
decomposition, network models, motif tests, and predictive baselines to examine
how different actors and relations organize the chronicle's account of power.
The clearest findings concern differentiated political roles and connections among
relation layers, including associations that remain after excluding the sovereign
category.

## Read the results

- [Historical interpretation](PRELIMINARY_HISTORICAL_INTERPRETATION.md): the research
  argument and the source passages that give the statistical patterns meaning.
- [Analysis results](NEW_ANALYSES_AND_RESULTS.md): model comparisons, four figures,
  and the controls used to interpret each result.
- [Evaluation record](AUDIT_OF_EXISTING_TESTS.md): coverage across 26 method families,
  with implemented analyses, prior studies, and requirements for further work.
- [Family summary](../findings/10_family_audit.csv): a compact machine-readable index.

## Inspect the run

The package retains source hashes, seeds, environment details, stage logs, and
**126 passing validation checks**. Start with the [reproduction guide](../reproduce/README.md)
and [runner](../reproduce/RUN_ANALYSES.ps1). The recorded environment is in
[11_environment.json](../findings/11_environment.json); input identity is documented
in [input_provenance.json](../reproduce/input_provenance.json).

Re-execution requires the separately maintained source archive and prepared inputs.
The statistical analysis itself requires no paid API calls. The full method checklist,
interactive report, and source-quotation tables belong to the original local analysis
package; the public [research guide](../README.md) maps the selected published outputs.
