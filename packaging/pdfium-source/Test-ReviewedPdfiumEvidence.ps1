# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param([string]$EvidenceRoot)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) { $EvidenceRoot = Join-Path $projectRoot 'artifacts\pdfium-source\evidence' }
$EvidenceRoot = [IO.Path]::GetFullPath($EvidenceRoot)
$approvalPath = Join-Path $EvidenceRoot 'reviewed-approval.json'
if (-not (Test-Path -LiteralPath $approvalPath -PathType Leaf)) {
    throw "Reviewed PDFium approval is missing: $approvalPath"
}
$approval = Get-Content -LiteralPath $approvalPath -Raw | ConvertFrom-Json
foreach ($field in @('reviewApproved', 'redistributionApproved', 'bundlingApproved')) {
    if ($approval.$field -ne $true) { throw "PDFium approval field '$field' is not true." }
}
if ($approval.sourceRevision -ne '2870fa9244b0f0f69fb743fab1e08deefcb07b2b') {
    throw 'PDFium approval does not use the pinned revision.'
}
foreach ($entry in @(
    @{ Path = $approval.binaryPath; Hash = $approval.binarySha256 },
    @{ Path = $approval.sourceLockPath; Hash = $approval.sourceLockSha256 },
    @{ Path = $approval.buildManifestPath; Hash = $approval.buildManifestSha256 },
    @{ Path = $approval.noticePath; Hash = $approval.noticeSha256 }
)) {
    $path = [IO.Path]::GetFullPath((Join-Path $EvidenceRoot ([string]$entry.Path)))
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Approved PDFium input is missing: $path" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne ([string]$entry.Hash).ToLowerInvariant()) { throw "Approved PDFium input hash mismatch: $path" }
}
$noticePath = Join-Path $EvidenceRoot ([string]$approval.noticePath)
$noticePath = [IO.Path]::GetFullPath($noticePath)
$firstNoticeLine = Get-Content -LiteralPath $noticePath -TotalCount 1
if ($firstNoticeLine.Trim() -ne 'REVIEW STATUS: COMPLETE') { throw 'PDFium reviewed notice is not marked complete.' }
Write-Host 'Reviewed PDFium source-build evidence: PASS'
