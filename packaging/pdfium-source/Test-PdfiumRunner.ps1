# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunnerPath,
    [Parameter(Mandatory = $true)][string]$EvidenceRoot,
    [string]$InputPdfPath,
    [string]$ExpectedInputSha256
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RunnerPath = [IO.Path]::GetFullPath($RunnerPath)
$EvidenceRoot = [IO.Path]::GetFullPath($EvidenceRoot)
if (-not (Test-Path -LiteralPath $RunnerPath -PathType Leaf)) {
    throw "PDFium runner is missing: $RunnerPath"
}
New-Item -ItemType Directory -Path $EvidenceRoot -Force | Out-Null

function Add-Ascii([IO.Stream]$Stream, [string]$Value) {
    $bytes = [Text.Encoding]::ASCII.GetBytes($Value)
    $Stream.Write($bytes, 0, $bytes.Length)
}

if ([string]::IsNullOrWhiteSpace($InputPdfPath) -xor [string]::IsNullOrWhiteSpace($ExpectedInputSha256)) {
    throw 'InputPdfPath and ExpectedInputSha256 must be supplied together.'
}
$usesExternalInput = -not [string]::IsNullOrWhiteSpace($InputPdfPath)
if ($usesExternalInput) {
    $pdfPath = [IO.Path]::GetFullPath($InputPdfPath)
    if (-not (Test-Path -LiteralPath $pdfPath -PathType Leaf)) { throw "PDFium smoke input is missing: $pdfPath" }
    $inputItem = Get-Item -LiteralPath $pdfPath -Force
    if (($inputItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'PDFium smoke input cannot be a reparse point.' }
    $inputHashBefore = (Get-FileHash -LiteralPath $pdfPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($inputHashBefore -ne $ExpectedInputSha256.ToLowerInvariant()) { throw 'PDFium smoke input SHA-256 does not match the pinned fixture.' }
    $pdfBytes = [IO.File]::ReadAllBytes($pdfPath)
}
else {
    $objects = @(
        '<< /Type /Catalog /Pages 2 0 R >>',
        '<< /Type /Pages /Count 1 /Kids [3 0 R] >>',
        '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] /Resources << >> >>'
    )
    $pdfStream = New-Object IO.MemoryStream
    try {
        Add-Ascii $pdfStream "%PDF-1.4`n% controlled public synthetic renderer smoke`n"
        $offsets = New-Object 'long[]' ($objects.Count + 1)
        for ($index = 0; $index -lt $objects.Count; $index++) {
            $offsets[$index + 1] = $pdfStream.Position
            Add-Ascii $pdfStream ("{0} 0 obj`n{1}`nendobj`n" -f ($index + 1), $objects[$index])
        }
        $xrefOffset = $pdfStream.Position
        Add-Ascii $pdfStream ("xref`n0 {0}`n" -f ($objects.Count + 1))
        Add-Ascii $pdfStream "0000000000 65535 f `n"
        for ($index = 1; $index -lt $offsets.Length; $index++) {
            Add-Ascii $pdfStream ("{0:D10} 00000 n `n" -f $offsets[$index])
        }
        Add-Ascii $pdfStream ("trailer`n<< /Size {0} /Root 1 0 R >>`nstartxref`n{1}`n%%EOF`n" -f ($objects.Count + 1), $xrefOffset)
        $pdfBytes = $pdfStream.ToArray()
    }
    finally {
        $pdfStream.Dispose()
    }
    $pdfPath = Join-Path $EvidenceRoot 'controlled-input.pdf'
    [IO.File]::WriteAllBytes($pdfPath, $pdfBytes)
    $inputHashBefore = (Get-FileHash -LiteralPath $pdfPath -Algorithm SHA256).Hash.ToLowerInvariant()
}

$rawPath = Join-Path $EvidenceRoot 'controlled-page.raw'
$startInfo = New-Object Diagnostics.ProcessStartInfo
$startInfo.FileName = $RunnerPath
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardInput = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
if ($null -ne $startInfo.PSObject.Properties['ArgumentList']) {
    [void]$startInfo.ArgumentList.Add('--output')
    [void]$startInfo.ArgumentList.Add($rawPath)
    [void]$startInfo.ArgumentList.Add('--page')
    [void]$startInfo.ArgumentList.Add('0')
    [void]$startInfo.ArgumentList.Add('--dpi')
    [void]$startInfo.ArgumentList.Add('72')
}
else {
    if ($rawPath.Contains('"')) {
        throw 'PDFium smoke output path cannot contain a quotation mark.'
    }
    $startInfo.Arguments = '--output "' + $rawPath + '" --page 0 --dpi 72'
}
$process = New-Object Diagnostics.Process
$process.StartInfo = $startInfo
try {
    if (-not $process.Start()) { throw 'PDFium runner did not start.' }
    $process.StandardInput.BaseStream.Write($pdfBytes, 0, $pdfBytes.Length)
    $process.StandardInput.Close()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        throw "PDFium runner exited $($process.ExitCode). stderr: $stderr stdout: $stdout"
    }
}
finally {
    $process.Dispose()
}

$raw = [IO.File]::ReadAllBytes($rawPath)
if ($raw.Length -lt 28) { throw 'PDFium raw output is truncated.' }
$magic = [Text.Encoding]::ASCII.GetString($raw, 0, 8)
if ($magic -ne "GRPDF01`0") { throw 'PDFium raw output magic is invalid.' }
$width = [BitConverter]::ToInt32($raw, 8)
$height = [BitConverter]::ToInt32($raw, 12)
$stride = [BitConverter]::ToInt32($raw, 16)
$payloadLength = [BitConverter]::ToInt64($raw, 20)
if ($width -lt 1 -or $height -lt 1 -or $width -gt 32768 -or $height -gt 32768 -or
    $stride -lt (4 * $width) -or $payloadLength -ne ([int64]$stride * $height) -or $raw.Length -ne 28 + $payloadLength) {
    throw "PDFium raw output contract mismatch: width=$width height=$height stride=$stride payload=$payloadLength bytes=$($raw.Length)"
}
if (-not $usesExternalInput -and ($width -ne 72 -or $height -ne 72)) {
    throw "Controlled PDFium smoke dimensions changed: width=$width height=$height"
}
$inputHashAfter = (Get-FileHash -LiteralPath $pdfPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($inputHashAfter -ne $inputHashBefore) { throw 'PDFium smoke input changed during rendering.' }

$result = [ordered]@{
    schemaVersion = 1
    runnerSha256 = (Get-FileHash -LiteralPath $RunnerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    inputKind = if ($usesExternalInput) { 'pinned-public-fixture' } else { 'controlled-synthetic-fixture' }
    inputPath = $pdfPath
    inputSha256 = $inputHashBefore
    inputUnchanged = $true
    rawSha256 = (Get-FileHash -LiteralPath $rawPath -Algorithm SHA256).Hash.ToLowerInvariant()
    width = $width
    height = $height
    stride = $stride
    payloadLength = $payloadLength
    stdout = $stdout.Trim()
    stderr = $stderr.Trim()
    verifiedUtc = [DateTime]::UtcNow.ToString('o')
}
$json = $result | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText((Join-Path $EvidenceRoot 'native-render-smoke.json'), $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
Write-Host "PDFium native render smoke: PASS $width x $height"
