# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$TargetRoot,
    [string]$EvidenceRoot = (Join-Path $PSScriptRoot '..\..\artifacts\pdfium-source\evidence'),
    [string]$PolicyPath = (Join-Path $PSScriptRoot 'internal-portable-policy.json')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$expectedPolicySha256 = '8bea103823fbef6f91f12439f731cb78e98081e74f0f44b9d895e086eeb197a1'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$profileRoot = [IO.Path]::GetFullPath($PSScriptRoot)

function Assert-ContainedPath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$CandidatePath,
        [Parameter(Mandatory = $true)][string]$Label,
        [switch]$AllowBase
    )

    $fullBase = [IO.Path]::GetFullPath($BasePath)
    $fullCandidate = [IO.Path]::GetFullPath($CandidatePath)
    if ($AllowBase -and $fullCandidate.Equals($fullBase, [StringComparison]::OrdinalIgnoreCase)) {
        return $fullCandidate
    }

    $prefix = [IO.Path]::TrimEndingDirectorySeparator($fullBase) + [IO.Path]::DirectorySeparatorChar
    if (-not $fullCandidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must remain under $fullBase."
    }

    return $fullCandidate
}

function Assert-NoReparsePoint {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $current = [IO.Path]::GetFullPath($Path)
    while (-not (Test-Path -LiteralPath $current)) {
        $parent = [IO.Path]::GetDirectoryName($current)
        if ([string]::IsNullOrEmpty($parent) -or $parent -eq $current) {
            throw "$Label has no existing parent path."
        }
        $current = $parent
    }

    while ($true) {
        $attributes = [IO.File]::GetAttributes($current)
        if (($attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Label contains a reparse point: $current"
        }
        $parent = [IO.Path]::GetDirectoryName($current)
        if ([string]::IsNullOrEmpty($parent) -or $parent -eq $current) { break }
        $current = $parent
    }
}

function Assert-ExactObject {
    param(
        [Parameter(Mandatory = $true)][System.Text.Json.JsonElement]$Element,
        [Parameter(Mandatory = $true)][string[]]$ExpectedProperties,
        [Parameter(Mandatory = $true)][string]$Context
    )

    if ($Element.ValueKind -ne [System.Text.Json.JsonValueKind]::Object) {
        throw "$Context must be a JSON object."
    }
    $expected = [Collections.Generic.HashSet[string]]::new($ExpectedProperties, [StringComparer]::Ordinal)
    $observed = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($property in $Element.EnumerateObject()) {
        if (-not $observed.Add($property.Name)) { throw "$Context contains duplicate field '$($property.Name)'." }
        if (-not $expected.Contains($property.Name)) { throw "$Context contains unexpected field '$($property.Name)'." }
    }
    foreach ($propertyName in $ExpectedProperties) {
        if (-not $observed.Contains($propertyName)) { throw "$Context is missing field '$propertyName'." }
    }
}

function Get-VerifiedInput {
    param(
        [Parameter(Mandatory = $true)][string]$EvidenceBase,
        [Parameter(Mandatory = $true)]$Descriptor,
        [Parameter(Mandatory = $true)][string]$ExpectedRelativePath,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if ([string]$Descriptor.relativePath -ne $ExpectedRelativePath) {
        throw "$Label has an unexpected relative path."
    }
    if ([IO.Path]::IsPathRooted([string]$Descriptor.relativePath)) {
        throw "$Label path must be relative."
    }
    $path = Assert-ContainedPath -BasePath $EvidenceBase -CandidatePath (Join-Path $EvidenceBase ([string]$Descriptor.relativePath)) -Label $Label
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "$Label is missing: $path" }
    Assert-NoReparsePoint -Path $path -Label $Label
    $item = Get-Item -LiteralPath $path
    if ($item.Length -ne [long]$Descriptor.length) { throw "$Label length does not match policy." }
    $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne [string]$Descriptor.sha256) { throw "$Label SHA-256 does not match policy." }
    return $path
}

$fullPolicyPath = Assert-ContainedPath -BasePath $profileRoot -CandidatePath $PolicyPath -Label 'PDFium internal staging policy'
if (-not (Test-Path -LiteralPath $fullPolicyPath -PathType Leaf)) { throw 'PDFium internal staging policy is missing.' }
Assert-NoReparsePoint -Path $fullPolicyPath -Label 'PDFium internal staging policy'
$policyHash = (Get-FileHash -LiteralPath $fullPolicyPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($policyHash -ne $expectedPolicySha256) { throw 'PDFium internal staging policy SHA-256 is not the tracked value.' }

$policyText = [IO.File]::ReadAllText($fullPolicyPath, [Text.Encoding]::UTF8)
$policyDocument = [System.Text.Json.JsonDocument]::Parse($policyText)
try {
    $root = $policyDocument.RootElement
    Assert-ExactObject $root @('schemaVersion','policyId','stagingMode','source','inputs','allowedDynamicLibraries','stagedFiles','reviewApproved','cleanMachineEvidence','releaseApproved') 'PDFium internal staging policy'
    Assert-ExactObject ($root.GetProperty('source')) @('repository','revision') 'PDFium policy source'
    Assert-ExactObject ($root.GetProperty('inputs')) @('runner','sourceLock','buildManifest','dependencyGraph','peImports','dependencyReviewPolicy','deterministicNotice','candidateApproval') 'PDFium policy inputs'
    Assert-ExactObject ($root.GetProperty('inputs').GetProperty('runner')) @('relativePath','sha256','length') 'PDFium runner policy'
    Assert-ExactObject ($root.GetProperty('inputs').GetProperty('sourceLock')) @('relativePath','sha256','length') 'PDFium source-lock policy'
    Assert-ExactObject ($root.GetProperty('inputs').GetProperty('buildManifest')) @('relativePath','sha256','length') 'PDFium build-manifest policy'
    Assert-ExactObject ($root.GetProperty('inputs').GetProperty('dependencyGraph')) @('relativePath','sha256','length','reviewStatus') 'PDFium dependency policy'
    Assert-ExactObject ($root.GetProperty('inputs').GetProperty('peImports')) @('relativePath','sha256','length') 'PDFium PE-import policy'
    Assert-ExactObject ($root.GetProperty('inputs').GetProperty('dependencyReviewPolicy')) @('relativePath','sha256','length','reviewStatus') 'PDFium dependency-review policy binding'
    Assert-ExactObject ($root.GetProperty('inputs').GetProperty('deterministicNotice')) @('relativePath','sha256','length','firstLine') 'PDFium notice policy'
    Assert-ExactObject ($root.GetProperty('inputs').GetProperty('candidateApproval')) @('relativePath','sha256','length') 'PDFium candidate-approval policy'
    Assert-ExactObject ($root.GetProperty('stagedFiles')) @('runner','notice','metadata') 'PDFium staged-file policy'
}
finally {
    $policyDocument.Dispose()
}
$policy = $policyText | ConvertFrom-Json -Depth 20

if ($policy.schemaVersion -ne 1 -or $policy.policyId -ne 'graphreader-pdfium-internal-dev-portable-v1' -or
    $policy.stagingMode -ne 'internal-development-portable-only') { throw 'PDFium internal staging policy identity is invalid.' }
if ($policy.source.repository -ne 'https://pdfium.googlesource.com/pdfium' -or
    $policy.source.revision -ne '2870fa9244b0f0f69fb743fab1e08deefcb07b2b') { throw 'PDFium internal staging source is not pinned.' }
if ($policy.reviewApproved -ne $false -or $policy.cleanMachineEvidence -ne $false -or $policy.releaseApproved -ne $false) {
    throw 'PDFium internal staging approval flags must all remain false.'
}
if ($policy.inputs.dependencyGraph.reviewStatus -ne 'inventory-only-unreviewed') {
    throw 'PDFium dependency inventory must remain explicitly unreviewed.'
}
if ($policy.inputs.dependencyReviewPolicy.reviewStatus -ne 'dependency-mapped-not-approved' -or
    $policy.inputs.deterministicNotice.firstLine -ne 'REVIEW STATUS: DEPENDENCY-MAPPED') {
    throw 'PDFium dependency review and notice must remain explicitly not approved.'
}
$expectedImports = @('ADVAPI32.dll','GDI32.dll','USER32.dll','KERNEL32.dll')
if (@($policy.allowedDynamicLibraries).Count -ne $expectedImports.Count -or
    (Compare-Object -ReferenceObject $expectedImports -DifferenceObject @($policy.allowedDynamicLibraries))) {
    throw 'PDFium dynamic-library policy is ambiguous.'
}

$fullEvidenceRoot = Assert-ContainedPath -BasePath $repositoryRoot -CandidatePath $EvidenceRoot -Label 'PDFium evidence root'
if (-not (Test-Path -LiteralPath $fullEvidenceRoot -PathType Container)) { throw 'PDFium evidence root is missing.' }
Assert-NoReparsePoint -Path $fullEvidenceRoot -Label 'PDFium evidence root'
if (Test-Path -LiteralPath (Join-Path $fullEvidenceRoot 'reviewed-approval.json')) {
    throw 'Reviewed PDFium approval is present; internal unapproved staging would be ambiguous.'
}

$runnerPath = Get-VerifiedInput $fullEvidenceRoot $policy.inputs.runner 'bin/graphreader_pdfium_renderer.exe' 'PDFium runner'
$sourceLockPath = Get-VerifiedInput $fullEvidenceRoot $policy.inputs.sourceLock 'source-lock.json' 'PDFium source lock'
$manifestPath = Get-VerifiedInput $fullEvidenceRoot $policy.inputs.buildManifest 'build-manifest.json' 'PDFium build manifest'
$dependencyPath = Get-VerifiedInput $fullEvidenceRoot $policy.inputs.dependencyGraph 'target-dependencies.txt' 'PDFium dependency graph'
$importsPath = Get-VerifiedInput $fullEvidenceRoot $policy.inputs.peImports 'pe-imports.txt' 'PDFium PE imports'
$candidateApprovalPath = Get-VerifiedInput $fullEvidenceRoot $policy.inputs.candidateApproval 'reviewed-approval.candidate.json' 'PDFium candidate approval'
$dependencyReviewPolicyPath = Get-VerifiedInput $profileRoot $policy.inputs.dependencyReviewPolicy 'review/dependency-review-policy.json' 'PDFium tracked dependency review policy'
$noticePath = Get-VerifiedInput $profileRoot $policy.inputs.deterministicNotice 'review/third-party-notices.dependency-mapped.txt' 'PDFium deterministic dependency notice'

$binFiles = @(Get-ChildItem -LiteralPath (Join-Path $fullEvidenceRoot 'bin') -File -Force)
if ($binFiles.Count -ne 1 -or $binFiles[0].Name -ne 'graphreader_pdfium_renderer.exe') {
    throw 'PDFium evidence bin directory is ambiguous; exactly one pinned runner is required.'
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -Depth 20
if ($manifest.source -ne $policy.source.repository -or $manifest.sourceRevision -ne $policy.source.revision -or
    $manifest.binarySha256 -ne $policy.inputs.runner.sha256 -or
    $manifest.sourceLockSha256 -ne $policy.inputs.sourceLock.sha256 -or
    $manifest.targetDependenciesSha256 -ne $policy.inputs.dependencyGraph.sha256 -or
    $manifest.peImportsSha256 -ne $policy.inputs.peImports.sha256 -or
    $manifest.features.v8 -ne $false -or $manifest.features.xfa -ne $false -or
    $manifest.features.skia -ne $false -or $manifest.features.icuDataFile -ne $false) {
    throw 'PDFium build manifest does not match the internal staging policy.'
}
$sourceLock = Get-Content -LiteralPath $sourceLockPath -Raw | ConvertFrom-Json -Depth 20
if ($sourceLock.sources.pdfium.repository -ne $policy.source.repository -or
    $sourceLock.sources.pdfium.revision -ne $policy.source.revision -or
    $sourceLock.target.os -ne 'win' -or $sourceLock.target.cpu -ne 'x64' -or
    $sourceLock.target.v8 -ne $false -or $sourceLock.target.xfa -ne $false -or
    $sourceLock.target.skia -ne $false -or $sourceLock.target.icuDataFile -ne $false) {
    throw 'PDFium source lock does not match the internal staging policy.'
}
$candidateApproval = Get-Content -LiteralPath $candidateApprovalPath -Raw | ConvertFrom-Json -Depth 20
if ($candidateApproval.binarySha256 -ne $policy.inputs.runner.sha256 -or
    $candidateApproval.sourceRevision -ne $policy.source.revision -or
    $candidateApproval.reviewApproved -ne $false -or
    $candidateApproval.redistributionApproved -ne $false -or
    $candidateApproval.bundlingApproved -ne $false) {
    throw 'PDFium candidate approval no longer represents an unapproved exact candidate.'
}
$dependencyReviewPolicy = Get-Content -LiteralPath $dependencyReviewPolicyPath -Raw | ConvertFrom-Json -Depth 50
if ($dependencyReviewPolicy.overallReviewStatus -ne 'dependency-mapped-not-approved' -or
    $dependencyReviewPolicy.sourceRevision -ne $policy.source.revision -or
    $dependencyReviewPolicy.evidenceBinding.binarySha256 -ne $policy.inputs.runner.sha256 -or
    $dependencyReviewPolicy.evidenceBinding.sourceLockSha256 -ne $policy.inputs.sourceLock.sha256 -or
    $dependencyReviewPolicy.evidenceBinding.buildManifestSha256 -ne $policy.inputs.buildManifest.sha256 -or
    $dependencyReviewPolicy.evidenceBinding.targetDependenciesSha256 -ne $policy.inputs.dependencyGraph.sha256 -or
    $dependencyReviewPolicy.evidenceBinding.peImportsSha256 -ne $policy.inputs.peImports.sha256 -or
    $dependencyReviewPolicy.noticeBundle.reviewStatus -ne 'dependency-mapped-not-approved' -or
    $dependencyReviewPolicy.noticeBundle.sha256 -ne $policy.inputs.deterministicNotice.sha256) {
    throw 'PDFium tracked dependency review policy does not match the exact unapproved candidate.'
}
$firstNoticeLine = Get-Content -LiteralPath $noticePath -TotalCount 1
if ($firstNoticeLine -ne $policy.inputs.deterministicNotice.firstLine) { throw 'PDFium deterministic notice status is invalid.' }

$metadata = [ordered]@{
    schemaVersion = 1
    policyId = [string]$policy.policyId
    stagingMode = [string]$policy.stagingMode
    source = [ordered]@{ repository = [string]$policy.source.repository; revision = [string]$policy.source.revision }
    runner = [ordered]@{ fileName = [string]$policy.stagedFiles.runner; sha256 = [string]$policy.inputs.runner.sha256; length = [long]$policy.inputs.runner.length }
    evidence = [ordered]@{
        policySha256 = $policyHash
        sourceLockSha256 = [string]$policy.inputs.sourceLock.sha256
        buildManifestSha256 = [string]$policy.inputs.buildManifest.sha256
        dependencyGraphSha256 = [string]$policy.inputs.dependencyGraph.sha256
        peImportsSha256 = [string]$policy.inputs.peImports.sha256
        dependencyReviewPolicySha256 = [string]$policy.inputs.dependencyReviewPolicy.sha256
        noticeSha256 = [string]$policy.inputs.deterministicNotice.sha256
    }
    reviewApproved = $false
    cleanMachineEvidence = $false
    releaseApproved = $false
    warning = 'Internal development portable only. Redistribution and release are prohibited.'
}
$metadataText = ($metadata | ConvertTo-Json -Depth 10) + "`n"
$metadataBytes = [Text.UTF8Encoding]::new($false).GetBytes($metadataText)
$metadataHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($metadataBytes)).ToLowerInvariant()

$fullTargetRoot = Assert-ContainedPath -BasePath $repositoryRoot -CandidatePath $TargetRoot -Label 'PDFium staging target'
Assert-NoReparsePoint -Path $fullTargetRoot -Label 'PDFium staging target'
$expectedNames = @([string]$policy.stagedFiles.runner, [string]$policy.stagedFiles.notice, [string]$policy.stagedFiles.metadata)
if (Test-Path -LiteralPath $fullTargetRoot) {
    if (-not (Test-Path -LiteralPath $fullTargetRoot -PathType Container)) { throw 'PDFium staging target is not a directory.' }
    $actualNames = @(Get-ChildItem -LiteralPath $fullTargetRoot -File -Force | Select-Object -ExpandProperty Name)
    $directories = @(Get-ChildItem -LiteralPath $fullTargetRoot -Directory -Force)
    if ($directories.Count -ne 0 -or $actualNames.Count -ne $expectedNames.Count -or
        (Compare-Object -ReferenceObject $expectedNames -DifferenceObject $actualNames)) {
        throw 'Existing PDFium staging target contains ambiguous files.'
    }
    if ((Get-FileHash -LiteralPath (Join-Path $fullTargetRoot $policy.stagedFiles.runner) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $policy.inputs.runner.sha256 -or
        (Get-FileHash -LiteralPath (Join-Path $fullTargetRoot $policy.stagedFiles.notice) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $policy.inputs.deterministicNotice.sha256 -or
        (Get-FileHash -LiteralPath (Join-Path $fullTargetRoot $policy.stagedFiles.metadata) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $metadataHash) {
        throw 'Existing PDFium staging target does not match the exact staged payload.'
    }
} else {
    $targetParent = [IO.Path]::GetDirectoryName($fullTargetRoot)
    if (-not (Test-Path -LiteralPath $targetParent -PathType Container)) {
        New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
    }
    Assert-NoReparsePoint -Path $targetParent -Label 'PDFium staging target parent'
    $temporaryRoot = Join-Path $targetParent ('.pdfium-stage-' + [Guid]::NewGuid().ToString('N'))
    $temporaryRoot = Assert-ContainedPath -BasePath $targetParent -CandidatePath $temporaryRoot -Label 'PDFium temporary staging path'
    try {
        New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
        Copy-Item -LiteralPath $runnerPath -Destination (Join-Path $temporaryRoot $policy.stagedFiles.runner)
        Copy-Item -LiteralPath $noticePath -Destination (Join-Path $temporaryRoot $policy.stagedFiles.notice)
        [IO.File]::WriteAllBytes((Join-Path $temporaryRoot $policy.stagedFiles.metadata), $metadataBytes)
        Move-Item -LiteralPath $temporaryRoot -Destination $fullTargetRoot
    }
    finally {
        if (Test-Path -LiteralPath $temporaryRoot) {
            $verifiedTemporary = Assert-ContainedPath -BasePath $targetParent -CandidatePath $temporaryRoot -Label 'PDFium temporary cleanup path'
            Remove-Item -LiteralPath $verifiedTemporary -Recurse -Force
        }
    }
}

Write-Host "Internal PDFium portable staging: PASS $($policy.inputs.runner.sha256)"
[pscustomobject]@{
    targetRoot = $fullTargetRoot
    runnerSha256 = [string]$policy.inputs.runner.sha256
    noticeSha256 = [string]$policy.inputs.deterministicNotice.sha256
    metadataSha256 = $metadataHash
    reviewApproved = $false
    cleanMachineEvidence = $false
    releaseApproved = $false
}
