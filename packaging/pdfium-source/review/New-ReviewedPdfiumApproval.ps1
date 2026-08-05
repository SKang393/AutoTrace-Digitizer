# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$EvidenceRoot,
    [string]$PolicyPath,
    [string]$MappedNoticePath,
    [switch]$ConfirmExactDependencyReview
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$profileRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $profileRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) {
    $EvidenceRoot = Join-Path $repositoryRoot 'artifacts\pdfium-source\evidence'
}
if ([string]::IsNullOrWhiteSpace($PolicyPath)) {
    $PolicyPath = Join-Path $PSScriptRoot 'dependency-review-policy.json'
}
if ([string]::IsNullOrWhiteSpace($MappedNoticePath)) {
    $MappedNoticePath = Join-Path $PSScriptRoot 'third-party-notices.dependency-mapped.txt'
}
if (-not $ConfirmExactDependencyReview.IsPresent) {
    throw 'Independent review confirmation is required. Re-run with -ConfirmExactDependencyReview only after reviewing the exact mapped dependency closure and notices.'
}

$EvidenceRoot = [IO.Path]::GetFullPath($EvidenceRoot)
$PolicyPath = [IO.Path]::GetFullPath($PolicyPath)
$MappedNoticePath = [IO.Path]::GetFullPath($MappedNoticePath)
foreach ($path in @($EvidenceRoot, $PolicyPath, $MappedNoticePath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "PDFium review input is missing: $path"
    }
}

& (Join-Path $PSScriptRoot 'Test-PdfiumDependencyReviewPolicy.ps1') `
    -EvidenceRoot $EvidenceRoot `
    -PolicyPath $PolicyPath `
    -NoticePath $MappedNoticePath

$policy = Get-Content -LiteralPath $PolicyPath -Raw | ConvertFrom-Json
if ([string]$policy.overallReviewStatus -ne 'dependency-mapped-not-approved' -or
    [string]$policy.noticeBundle.reviewStatus -ne 'dependency-mapped-not-approved') {
    throw 'The tracked dependency policy must remain fail closed; approval is emitted only as ignored local evidence.'
}

$mappedNotice = Get-Content -LiteralPath $MappedNoticePath -Raw
$mappedStatus = 'REVIEW STATUS: DEPENDENCY-MAPPED'
$mappedDisclaimer = 'This dependency mapping is not independent approval, clean-machine evidence, or release authorization.'
if (-not $mappedNotice.StartsWith($mappedStatus, [StringComparison]::Ordinal) -or
    $mappedNotice.IndexOf($mappedDisclaimer, [StringComparison]::Ordinal) -lt 0) {
    throw 'The mapped notice does not have the exact fail-closed review markers required for promotion.'
}

$reviewedNotice = $mappedNotice.Remove(0, $mappedStatus.Length).Insert(0, 'REVIEW STATUS: COMPLETE')
$reviewedNotice = $reviewedNotice.Replace(
    $mappedDisclaimer,
    'The exact dependency closure and notice bundle were reviewed for binary redistribution. Clean-machine evidence and public release authorization remain separate fail-closed gates.')

function Write-Utf8NoBomAtomic {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Content
    )

    $directory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $temporaryPath = Join-Path $directory ('.' + [IO.Path]::GetFileName($Path) + '.' + [Guid]::NewGuid().ToString('N') + '.tmp')
    try {
        [IO.File]::WriteAllText($temporaryPath, $Content, [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

$reviewedNoticePath = Join-Path $EvidenceRoot 'third-party-notices.reviewed.txt'
Write-Utf8NoBomAtomic -Path $reviewedNoticePath -Content $reviewedNotice
$noticeSha256 = (Get-FileHash -LiteralPath $reviewedNoticePath -Algorithm SHA256).Hash.ToLowerInvariant()

$candidatePath = Join-Path $EvidenceRoot 'reviewed-approval.candidate.json'
$candidate = Get-Content -LiteralPath $candidatePath -Raw | ConvertFrom-Json
$approval = [ordered]@{
    schemaVersion = 1
    rendererId = [string]$candidate.rendererId
    rendererVersion = [string]$candidate.rendererVersion
    binaryPath = [string]$candidate.binaryPath
    binarySha256 = [string]$candidate.binarySha256
    source = [string]$candidate.source
    sourceRevision = [string]$candidate.sourceRevision
    sourceLockPath = [string]$candidate.sourceLockPath
    sourceLockSha256 = [string]$candidate.sourceLockSha256
    buildManifestPath = [string]$candidate.buildManifestPath
    buildManifestSha256 = [string]$candidate.buildManifestSha256
    licenseSpdx = [string]$candidate.licenseSpdx
    noticePath = 'third-party-notices.reviewed.txt'
    noticeSha256 = $noticeSha256
    reviewApproved = $true
    redistributionApproved = $true
    bundlingApproved = $true
}
$approvalPath = Join-Path $EvidenceRoot 'reviewed-approval.json'
Write-Utf8NoBomAtomic `
    -Path $approvalPath `
    -Content (($approval | ConvertTo-Json -Depth 10) + [Environment]::NewLine)

& (Join-Path $profileRoot 'Test-ReviewedPdfiumEvidence.ps1') -EvidenceRoot $EvidenceRoot
Write-Host "Reviewed PDFium approval created as ignored local evidence: $approvalPath"
Write-Host 'Clean-machine and public release gates remain unchanged and blocked.'
