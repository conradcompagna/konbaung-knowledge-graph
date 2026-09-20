$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$previousWorkbook = $env:KONBAUNG_ENTITY_MANUAL_WORKBOOK
$previousOutput = $env:KONBAUNG_ENTITY_DISAMBIGUATION_OUTPUT

try {
    $env:KONBAUNG_ENTITY_MANUAL_WORKBOOK = Join-Path $project "konbaung_manual_entities_frequency_gt1.xlsx"
    $env:KONBAUNG_ENTITY_DISAMBIGUATION_OUTPUT = "konbaung_v3_final_entity_disambiguation_20260724"
    & python (Join-Path $project "konbaung_v3_node_disambiguation.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Entity disambiguation exited with code $LASTEXITCODE"
    }
}
finally {
    if ($null -eq $previousWorkbook) {
        Remove-Item Env:KONBAUNG_ENTITY_MANUAL_WORKBOOK -ErrorAction SilentlyContinue
    }
    else {
        $env:KONBAUNG_ENTITY_MANUAL_WORKBOOK = $previousWorkbook
    }
    if ($null -eq $previousOutput) {
        Remove-Item Env:KONBAUNG_ENTITY_DISAMBIGUATION_OUTPUT -ErrorAction SilentlyContinue
    }
    else {
        $env:KONBAUNG_ENTITY_DISAMBIGUATION_OUTPUT = $previousOutput
    }
}
