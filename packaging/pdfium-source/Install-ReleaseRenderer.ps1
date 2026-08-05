# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EvidenceRoot,

    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,

    [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
}
else {
    $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
}
$EvidenceRoot = [IO.Path]::GetFullPath($EvidenceRoot)
$DestinationRoot = [IO.Path]::GetFullPath($DestinationRoot)
$repositoryPrefix = $RepositoryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
$evidencePrefix = $EvidenceRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
$releaseAuditPath = Join-Path $RepositoryRoot 'packaging\common\release-audit.json'
$validatorPath = Join-Path $RepositoryRoot 'packaging\pdfium-source\Test-ReviewedPdfiumEvidence.ps1'
$approvalPath = Join-Path $EvidenceRoot 'reviewed-approval.json'

function Get-SafeEvidenceFile {
    param(
        [Parameter(Mandatory = $true)] [string]$RelativePath,
        [Parameter(Mandatory = $true)] [string]$Label
    )

    if ([IO.Path]::IsPathRooted($RelativePath) -or
        $RelativePath -match '(^|[\\/])\.\.([\\/]|$)') {
        throw "$Label uses an unsafe relative path: $RelativePath"
    }
    $fullPath = [IO.Path]::GetFullPath((Join-Path $EvidenceRoot $RelativePath))
    if (-not $fullPath.StartsWith($evidencePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        throw "$Label is missing or outside the PDFium evidence root: $RelativePath"
    }
    $current = $fullPath
    while ($current.StartsWith($evidencePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        if (((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Label contains a reparse point: $current"
        }
        $current = Split-Path -Parent $current
    }
    return $fullPath
}

foreach ($required in @($releaseAuditPath, $validatorPath, $approvalPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Release PDFium input is missing: $required"
    }
}

$releaseAudit = Get-Content -LiteralPath $releaseAuditPath -Raw | ConvertFrom-Json
$components = @($releaseAudit.components | Where-Object {
        [string]$_.id -ceq 'pdfium-native'
    })
if ($components.Count -ne 1) {
    throw "Release audit must contain exactly one 'pdfium-native' component."
}
$component = $components[0]
$expectedHash = [string]$component.artifactSha256
if ([string]$component.checksumPolicy -cne 'exact-binary' -or
    $expectedHash -notmatch '^[0-9a-fA-F]{64}$') {
    throw 'Release PDFium component must use exact-binary coverage with an artifact SHA-256.'
}
$expectedHash = $expectedHash.ToLowerInvariant()

$gates = @($releaseAudit.mandatoryEvidenceGates | Where-Object {
        [string]$_.id -ceq 'pdfium-clean-machine-load'
    })
if ($gates.Count -ne 1) {
    throw "Release audit must contain exactly one 'pdfium-clean-machine-load' gate."
}
$gate = $gates[0]
if ([string]$gate.status -cne 'pass' -or @($gate.evidence).Count -eq 0) {
    throw 'PDFium clean-machine evidence gate is not directly passing.'
}
foreach ($evidence in @($gate.evidence)) {
    $relativePath = [string]$evidence.path
    $evidenceHash = [string]$evidence.sha256
    if ([IO.Path]::IsPathRooted($relativePath) -or
        $relativePath -match '(^|[\\/])\.\.([\\/]|$)' -or
        $evidenceHash -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'PDFium clean-machine evidence uses an unsafe path or invalid checksum.'
    }
    $evidencePath = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $relativePath))
    if (-not $evidencePath.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $evidencePath -PathType Leaf)) {
        throw "PDFium clean-machine evidence is missing or outside the repository: $relativePath"
    }
    $actualEvidenceHash = (Get-FileHash -LiteralPath $evidencePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [string]::Equals($actualEvidenceHash, $evidenceHash, [StringComparison]::OrdinalIgnoreCase)) {
        throw "PDFium clean-machine evidence checksum differs: $relativePath"
    }
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $validatorPath -EvidenceRoot $EvidenceRoot
if ($LASTEXITCODE -ne 0) {
    throw "Reviewed PDFium evidence validation failed with exit code $LASTEXITCODE."
}

$approval = Get-Content -LiteralPath $approvalPath -Raw | ConvertFrom-Json
if ([int]$approval.schemaVersion -ne 1 -or
    [string]$approval.source -cne 'https://pdfium.googlesource.com/pdfium' -or
    [string]$approval.sourceRevision -cne '2870fa9244b0f0f69fb743fab1e08deefcb07b2b' -or
    [string]$approval.rendererVersion -cne '2870fa9244b0f0f69fb743fab1e08deefcb07b2b' -or
    [string]$approval.licenseSpdx -cne 'BSD-3-Clause' -or
    -not [bool]$approval.reviewApproved -or
    -not [bool]$approval.redistributionApproved -or
    -not [bool]$approval.bundlingApproved) {
    throw 'Reviewed PDFium approval does not satisfy the pinned release policy.'
}

$resources = @(
    [pscustomobject]@{ RelativePath = [string]$approval.binaryPath; Hash = [string]$approval.binarySha256; Label = 'PDFium runner' },
    [pscustomobject]@{ RelativePath = [string]$approval.sourceLockPath; Hash = [string]$approval.sourceLockSha256; Label = 'PDFium source lock' },
    [pscustomobject]@{ RelativePath = [string]$approval.buildManifestPath; Hash = [string]$approval.buildManifestSha256; Label = 'PDFium build manifest' },
    [pscustomobject]@{ RelativePath = [string]$approval.noticePath; Hash = [string]$approval.noticeSha256; Label = 'PDFium reviewed notice' }
)
$resolvedResources = foreach ($resource in $resources) {
    if ($resource.Hash -notmatch '^[0-9a-fA-F]{64}$') {
        throw "$($resource.Label) approval checksum is invalid."
    }
    $sourcePath = Get-SafeEvidenceFile -RelativePath $resource.RelativePath -Label $resource.Label
    $actualHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [string]::Equals($actualHash, $resource.Hash, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$($resource.Label) checksum differs from reviewed approval."
    }
    [pscustomobject]@{
        SourcePath = $sourcePath
        RelativePath = $resource.RelativePath.Replace('/', [IO.Path]::DirectorySeparatorChar)
        Hash = $actualHash
        Label = $resource.Label
    }
}
$runner = @($resolvedResources | Where-Object { $_.Label -eq 'PDFium runner' })
if ($runner.Count -ne 1 -or -not [string]::Equals($runner[0].Hash, $expectedHash, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Reviewed PDFium runner does not match the exact release-audit binary.'
}

if (-not (Test-Path -LiteralPath $DestinationRoot -PathType Container)) {
    throw "Release publish destination is missing: $DestinationRoot"
}
$targetRoot = Join-Path $DestinationRoot 'pdfium'
if (Test-Path -LiteralPath $targetRoot) {
    throw "Release PDFium target already exists: $targetRoot"
}
$temporaryRoot = Join-Path $DestinationRoot ('.pdfium-release-' + [Guid]::NewGuid().ToString('N'))
try {
    New-Item -ItemType Directory -Path $temporaryRoot -Force | Out-Null
    foreach ($resource in $resolvedResources) {
        $targetPath = [IO.Path]::GetFullPath((Join-Path $temporaryRoot $resource.RelativePath))
        $temporaryPrefix = $temporaryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        if (-not $targetPath.StartsWith($temporaryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "$($resource.Label) staging path escapes the release PDFium root."
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
        Copy-Item -LiteralPath $resource.SourcePath -Destination $targetPath
        $stagedHash = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($stagedHash -ne $resource.Hash) {
            throw "$($resource.Label) checksum changed during release staging."
        }
    }
    Copy-Item -LiteralPath $approvalPath -Destination (Join-Path $temporaryRoot 'reviewed-approval.json')
    Move-Item -LiteralPath $temporaryRoot -Destination $targetRoot
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        $resolvedTemporary = [IO.Path]::GetFullPath($temporaryRoot)
        $destinationPrefix = $DestinationRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        if ($resolvedTemporary.StartsWith($destinationPrefix, [StringComparison]::OrdinalIgnoreCase) -and
            [IO.Path]::GetFileName($resolvedTemporary).StartsWith('.pdfium-release-', [StringComparison]::Ordinal)) {
            Remove-Item -LiteralPath $resolvedTemporary -Recurse -Force
        }
    }
}

Write-Host "Release-approved PDFium renderer installed: $expectedHash"
Write-Output ([pscustomobject]@{
    ApprovalPath = Join-Path $targetRoot 'reviewed-approval.json'
    BinaryPath = Join-Path $targetRoot $runner[0].RelativePath
    BinarySha256 = $expectedHash
    CleanMachineEvidence = $true
    ReleaseApproved = $true
})
