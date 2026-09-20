$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$analysisScripts = @('01_contingency.py','02_graph_layers.py','03_tensor_semantics.py','04_paths_rules_motifs.py','04b_degree_null.py','05_network_models.py','06_robustness_comparison.py','07_brokerage_paths_hierarchy.py','08_predictive_check.py','09_motif_sensitivity_sources.py','10_audit_catalogue.py','11_validate.py','12_figures.py','13_write_reports.py','14_finalize.py')
foreach ($analysisScript in $analysisScripts) {
    Write-Host "Running $analysisScript"
    & python (Join-Path 'scripts' $analysisScript)
    if ($LASTEXITCODE -ne 0) { throw "Analysis failed: $analysisScript" }
}
