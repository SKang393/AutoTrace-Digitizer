# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$FirstEvidenceRoot,
    [Parameter(Mandatory = $true)][string]$SecondEvidenceRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'OpenCvSourceAudit.Common.ps1')

$first = [IO.Path]::GetFullPath($FirstEvidenceRoot)
$second = [IO.Path]::GetFullPath($SecondEvidenceRoot)
$errors = @(Get-OpenCvReproducibilityErrors -FirstEvidenceRoot $first -SecondEvidenceRoot $second)
if ($errors.Count -gt 0) {
    Write-Error ("OpenCV source-build reproducibility: FAIL`n" + ($errors -join [Environment]::NewLine))
    exit 1
}

$dllHash = (Get-FileHash -LiteralPath (Join-Path $first 'bin\OpenCvSharpExtern.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
$mapHash = (Get-FileHash -LiteralPath (Join-Path $first 'build\extern\OpenCvSharpExtern.map') -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "OpenCV source-build reproducibility: PASS"
Write-Host "DLL SHA-256: $dllHash"
Write-Host "Linker map SHA-256: $mapHash"
