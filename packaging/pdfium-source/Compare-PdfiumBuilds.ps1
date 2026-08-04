# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$FirstEvidenceRoot,
    [Parameter(Mandatory = $true)][string]$SecondEvidenceRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$firstRoot = [IO.Path]::GetFullPath($FirstEvidenceRoot)
$secondRoot = [IO.Path]::GetFullPath($SecondEvidenceRoot)

function Read-Build([string]$Root) {
    $manifestPath = Join-Path $Root 'build-manifest.json'
    $binaryPath = Join-Path $Root 'bin\graphreader_pdfium_renderer.exe'
    foreach ($path in @($manifestPath, $binaryPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "PDFium comparison input is missing: $path" }
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    return [ordered]@{
        Root = $Root
        Manifest = $manifest
        BinarySha256 = (Get-FileHash -LiteralPath $binaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$first = Read-Build $firstRoot
$second = Read-Build $secondRoot
foreach ($field in @('profileId', 'sourceRevision', 'sourceLockSha256', 'argsGnSha256', 'overlayBuildSha256', 'overlayRootTargetSha256', 'overlaySourceSha256', 'compatibilityPatchSha256', 'targetDependenciesSha256', 'peImportsSha256')) {
    if ([string]$first.Manifest.$field -ne [string]$second.Manifest.$field) {
        throw "PDFium builds differ at retained input '$field'."
    }
}
if ($first.BinarySha256 -ne $second.BinarySha256) {
    throw "PDFium builds are not byte reproducible: $($first.BinarySha256) != $($second.BinarySha256)"
}
Write-Host "PDFium reproducible binary: PASS $($first.BinarySha256)"
