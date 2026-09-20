param(
    [string]$ClusterDirectory = (
        "C:\Users\conra\Desktop\dighumproject\" +
        "konbaung_v3_node_edge_clustering_first_pass_20260724"
    )
)

$ErrorActionPreference = "Stop"
$outputPath = Join-Path $ClusterDirectory "PARED_DOWN_METATAG_MAPPING.md"
$encoding = New-Object System.Text.UTF8Encoding($false)
$writer = New-Object System.IO.StreamWriter($outputPath, $false, $encoding)

function Clean-MarkdownCell {
    param([object]$Value)
    return ([string]$Value).Replace("`r", " ").Replace("`n", " ").Replace("|", "\|")
}

function Load-Assignments {
    param([string]$Path)
    $groups = @{}
    Get-Content -LiteralPath $Path | ForEach-Object {
        if (-not [string]::IsNullOrWhiteSpace($_)) {
            $row = $_ | ConvertFrom-Json
            if (-not $groups.ContainsKey($row.clusterId)) {
                $groups[$row.clusterId] = New-Object System.Collections.ArrayList
            }
            [void]$groups[$row.clusterId].Add($row)
        }
    }
    return $groups
}

function Write-MetaTag {
    param(
        [string]$MetaTag,
        [object]$AlternativeMetaTag,
        [string]$Description,
        [object[]]$Tags
    )

    $sortedTags = @(
        $Tags | Sort-Object `
            @{ Expression = { [int]$_.frequency }; Descending = $true }, `
            @{ Expression = { [string]$_.tag }; Ascending = $true }
    )
    [long]$totalCount = 0
    foreach ($tag in $sortedTags) {
        $totalCount += [int]$tag.frequency
    }
    $alternative = if ($null -eq $AlternativeMetaTag -or $AlternativeMetaTag -eq "") {
        "None"
    }
    else {
        [string]$AlternativeMetaTag
    }

    $writer.WriteLine("### $(Clean-MarkdownCell $MetaTag)")
    $writer.WriteLine("")
    $writer.WriteLine("- Meta-tag: $(Clean-MarkdownCell $MetaTag)")
    $writer.WriteLine("- Alternative meta-tag: $(Clean-MarkdownCell $alternative)")
    $writer.WriteLine("- Description: $(Clean-MarkdownCell $Description)")
    $writer.WriteLine("- Total count: $totalCount")
    $writer.WriteLine("")
    $writer.WriteLine("| Tag | Frequency |")
    $writer.WriteLine("|---|---:|")
    foreach ($tag in $sortedTags) {
        $writer.WriteLine(
            "| $(Clean-MarkdownCell $tag.tag) | $([int]$tag.frequency) |"
        )
    }
    $writer.WriteLine("")
}

function Write-Kind {
    param(
        [string]$Heading,
        [string]$ClusterPath,
        [string]$AssignmentPath
    )

    $clusters = Get-Content -Raw -LiteralPath $ClusterPath | ConvertFrom-Json
    $assignments = Load-Assignments -Path $AssignmentPath
    $unresolvedTags = New-Object System.Collections.ArrayList

    $writer.WriteLine("## $Heading")
    $writer.WriteLine("")

    foreach ($cluster in $clusters) {
        $clusterTags = @($assignments[$cluster.clusterId])
        if ($cluster.unresolved) {
            foreach ($tag in $clusterTags) {
                [void]$unresolvedTags.Add($tag)
            }
            continue
        }
        Write-MetaTag `
            -MetaTag $cluster.candidateMetaTag `
            -AlternativeMetaTag $cluster.alternativeMetaTag `
            -Description $cluster.description `
            -Tags $clusterTags
    }

    if ($unresolvedTags.Count -gt 0) {
        Write-MetaTag `
            -MetaTag "UNRESOLVED" `
            -AlternativeMetaTag $null `
            -Description "Tags not assigned to a substantive first-pass community." `
            -Tags @($unresolvedTags)
    }
}

try {
    $writer.WriteLine("# Pared-Down Meta-Tag Mapping")
    $writer.WriteLine("")
    Write-Kind `
        -Heading "Nodes" `
        -ClusterPath (Join-Path $ClusterDirectory "node_clusters_labeled.json") `
        -AssignmentPath (Join-Path $ClusterDirectory "node_assignments.jsonl")
    Write-Kind `
        -Heading "Edges" `
        -ClusterPath (Join-Path $ClusterDirectory "edge_clusters_labeled.json") `
        -AssignmentPath (Join-Path $ClusterDirectory "edge_assignments.jsonl")
}
finally {
    $writer.Dispose()
}

Write-Output $outputPath
