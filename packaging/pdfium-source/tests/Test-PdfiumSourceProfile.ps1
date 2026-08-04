# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$profileRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$lock = Get-Content -LiteralPath (Join-Path $profileRoot 'source-lock.json') -Raw | ConvertFrom-Json
if ($lock.schemaVersion -ne 1) { throw 'Unexpected PDFium source-lock schema.' }
if ($lock.profileId -ne 'graphreader-pdfium-minimal-win-x64') { throw 'Unexpected PDFium profile ID.' }
if ($lock.sources.pdfium.repository -ne 'https://pdfium.googlesource.com/pdfium') { throw 'PDFium source is not the official repository.' }
if ($lock.sources.pdfium.revision -ne '2870fa9244b0f0f69fb743fab1e08deefcb07b2b') { throw 'PDFium revision is not pinned.' }
if ([string]::IsNullOrWhiteSpace([string]$lock.sources.depotTools.revision)) { throw 'depot_tools is not pinned.' }
if ([int]$lock.target.maxParallelCompileJobs -ne 4) { throw 'PDFium build parallelism must be pinned to four jobs.' }
$patch = Join-Path $profileRoot 'native\pdfium-windows-hdc.patch'
$patchHash = (Get-FileHash -LiteralPath $patch -Algorithm SHA256).Hash.ToLowerInvariant()
if ($patchHash -ne [string]$lock.compatibilityPatchSha256) { throw 'PDFium compatibility patch checksum does not match the source lock.' }
foreach ($feature in @('v8', 'xfa', 'skia', 'fontations', 'partitionAlloc', 'icuDataFile')) {
    if ($lock.target.$feature -ne $false) { throw "PDFium feature '$feature' must be disabled." }
}

$args = Get-Content -LiteralPath (Join-Path $profileRoot 'args.gn') -Raw
foreach ($required in @(
    'is_component_build = false',
    'target_cpu = "x64"',
    'pdf_enable_v8 = false',
    'pdf_enable_xfa = false',
    'pdf_use_skia = false',
    'pdf_enable_fontations = false',
    'pdf_use_partition_alloc = false',
    'icu_use_data_file = false'
)) {
    if ($args.IndexOf($required, [StringComparison]::Ordinal) -lt 0) { throw "Missing GN policy: $required" }
}

$native = Get-Content -LiteralPath (Join-Path $profileRoot 'native\graphreader_pdfium_renderer.cc') -Raw
foreach ($required in @('FPDF_LoadMemDocument64', 'FPDF_RenderPageBitmap', '_setmode(_fileno(stdin), _O_BINARY)', 'kMaximumRawBytes')) {
    if ($native.IndexOf($required, [StringComparison]::Ordinal) -lt 0) { throw "Native runner contract is missing: $required" }
}

$validator = Get-Content -LiteralPath (Join-Path $profileRoot 'Test-ReviewedPdfiumEvidence.ps1') -Raw
foreach ($required in @('reviewed-approval.json', 'reviewApproved', 'redistributionApproved', 'bundlingApproved', 'REVIEW STATUS: COMPLETE')) {
    if ($validator.IndexOf($required, [StringComparison]::Ordinal) -lt 0) { throw "Evidence validator is missing fail-closed check: $required" }
}

$runnerSmoke = Get-Content -LiteralPath (Join-Path $profileRoot 'Test-PdfiumRunner.ps1') -Raw
foreach ($required in @("PSObject.Properties['ArgumentList']", '$startInfo.Arguments', 'cannot contain a quotation mark')) {
    if ($runnerSmoke.IndexOf($required, [StringComparison]::Ordinal) -lt 0) { throw "Runner smoke is missing cross-PowerShell argument handling: $required" }
}

$reviewRoot = Join-Path $profileRoot 'review'
$reviewPolicy = Get-Content -LiteralPath (Join-Path $reviewRoot 'dependency-review-policy.json') -Raw | ConvertFrom-Json
if ([string]$reviewPolicy.overallReviewStatus -ne 'dependency-mapped-not-approved' -or
    [string]$reviewPolicy.noticeBundle.reviewStatus -ne 'dependency-mapped-not-approved') {
    throw 'PDFium dependency policy must remain explicitly not approved.'
}
if (@($reviewPolicy.components).Count -ne 15 -or @($reviewPolicy.permittedPeImports).Count -ne 4) {
    throw 'PDFium dependency policy component or system-import count changed without review.'
}
foreach ($inventory in @(
    @{ Path = [string]$reviewPolicy.reviewInventory.targetDependenciesPath; Hash = [string]$reviewPolicy.reviewInventory.targetDependenciesSha256 },
    @{ Path = [string]$reviewPolicy.reviewInventory.peImportsPath; Hash = [string]$reviewPolicy.reviewInventory.peImportsSha256 },
    @{ Path = [string]$reviewPolicy.noticeBundle.path; Hash = [string]$reviewPolicy.noticeBundle.sha256 }
)) {
    $inventoryPath = Join-Path $reviewRoot $inventory.Path
    if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf) -or
        (Get-FileHash -LiteralPath $inventoryPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $inventory.Hash) {
        throw "PDFium tracked review inventory hash mismatch: $($inventory.Path)"
    }
}
$noticeFirstLine = Get-Content -LiteralPath (Join-Path $reviewRoot ([string]$reviewPolicy.noticeBundle.path)) -TotalCount 1
if ($noticeFirstLine.Trim() -ne 'REVIEW STATUS: DEPENDENCY-MAPPED') {
    throw 'PDFium mapped notice must not claim release approval.'
}

Write-Host 'PDFium source profile static policy: PASS'
