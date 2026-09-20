param(
    [string]$ModelDirectory = "C:\Users\conra\Desktop\dighumproject\models\fasttext"
)

$ErrorActionPreference = "Stop"

function Test-FastTextArchive {
    param(
        [string]$Language,
        [long]$ExpectedCompressedBytes
    )

    $archivePart = Join-Path $ModelDirectory "cc.$Language.300.bin.gz.part"
    $archive = Join-Path $ModelDirectory "cc.$Language.300.bin.gz"
    $compressedBytes = (Get-Item -LiteralPath $archivePart).Length
    if ($compressedBytes -ne $ExpectedCompressedBytes) {
        throw (
            "Unexpected size for {0}: expected {1}, got {2}" -f
            $archivePart,
            $ExpectedCompressedBytes,
            $compressedBytes
        )
    }

    $inputStream = [System.IO.File]::OpenRead($archivePart)
    $gzipStream = New-Object System.IO.Compression.GZipStream(
        $inputStream,
        [System.IO.Compression.CompressionMode]::Decompress
    )
    $buffer = New-Object byte[] (4MB)
    [long]$uncompressedBytes = 0

    try {
        while (($read = $gzipStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $uncompressedBytes += $read
        }
    }
    finally {
        $gzipStream.Dispose()
        $inputStream.Dispose()
    }

    Move-Item -LiteralPath $archivePart -Destination $archive
    Write-Output (
        "VERIFIED {0} compressed={1} uncompressed={2}" -f
        $Language,
        $compressedBytes,
        $uncompressedBytes
    )
}

Test-FastTextArchive -Language "my" -ExpectedCompressedBytes 2630512501
Test-FastTextArchive -Language "en" -ExpectedCompressedBytes 4503593528
