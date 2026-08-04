# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$EvidenceRoot,
    [string]$SourceRoot,
    [string]$PolicyPath,
    [string]$NoticePath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) {
    $EvidenceRoot = Join-Path $PSScriptRoot '..\..\..\artifacts\pdfium-source\evidence'
}
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $PSScriptRoot '..\..\..\artifacts\pdfium-source\sources\pdfium'
}
if ([string]::IsNullOrWhiteSpace($PolicyPath)) {
    $PolicyPath = Join-Path $PSScriptRoot 'dependency-review-policy.json'
}

$EvidenceRoot = [IO.Path]::GetFullPath($EvidenceRoot)
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$PolicyPath = [IO.Path]::GetFullPath($PolicyPath)
$policy = Get-Content -LiteralPath $PolicyPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($NoticePath)) {
    $NoticePath = Join-Path $PSScriptRoot ([string]$policy.noticeBundle.path)
}
$NoticePath = [IO.Path]::GetFullPath($NoticePath)
$policyDirectory = [IO.Path]::GetDirectoryName($PolicyPath)
if ([string]::IsNullOrWhiteSpace($policyDirectory)) {
    throw 'PDFium dependency review policy must have a parent directory.'
}

if ([int]$policy.schemaVersion -ne 1 -or
    [string]$policy.profileId -ne 'graphreader-pdfium-minimal-win-x64' -or
    [string]$policy.sourceRevision -ne '2870fa9244b0f0f69fb743fab1e08deefcb07b2b' -or
    [string]$policy.overallReviewStatus -ne 'dependency-mapped-not-approved' -or
    [string]$policy.noticeBundle.reviewStatus -ne 'dependency-mapped-not-approved') {
    throw 'PDFium dependency review policy identity or fail-closed status is invalid.'
}

$paths = [ordered]@{
    binary = Join-Path $EvidenceRoot 'bin\graphreader_pdfium_renderer.exe'
    sourceLock = Join-Path $EvidenceRoot 'source-lock.json'
    buildManifest = Join-Path $EvidenceRoot 'build-manifest.json'
    targetDependencies = Join-Path $EvidenceRoot 'target-dependencies.txt'
    peImports = Join-Path $EvidenceRoot 'pe-imports.txt'
}
foreach ($entry in $paths.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value -PathType Leaf)) {
        throw "PDFium review input is missing: $($entry.Value)"
    }
}

foreach ($binding in @(
    @{ Name = 'binary'; Path = $paths.binary; Expected = [string]$policy.evidenceBinding.binarySha256 },
    @{ Name = 'sourceLock'; Path = $paths.sourceLock; Expected = [string]$policy.evidenceBinding.sourceLockSha256 },
    @{ Name = 'buildManifest'; Path = $paths.buildManifest; Expected = [string]$policy.evidenceBinding.buildManifestSha256 },
    @{ Name = 'targetDependencies'; Path = $paths.targetDependencies; Expected = [string]$policy.evidenceBinding.targetDependenciesSha256 },
    @{ Name = 'peImports'; Path = $paths.peImports; Expected = [string]$policy.evidenceBinding.peImportsSha256 }
)) {
    $actual = (Get-FileHash -LiteralPath $binding.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $binding.Expected) {
        throw "PDFium $($binding.Name) evidence hash mismatch."
    }
}

$trackedTargetDependencies = [IO.Path]::GetFullPath((Join-Path $policyDirectory ([string]$policy.reviewInventory.targetDependenciesPath)))
$trackedPeImports = [IO.Path]::GetFullPath((Join-Path $policyDirectory ([string]$policy.reviewInventory.peImportsPath)))
foreach ($inventoryBinding in @(
    @{ Name = 'tracked target dependency inventory'; Path = $trackedTargetDependencies; EvidencePath = $paths.targetDependencies; Expected = [string]$policy.reviewInventory.targetDependenciesSha256 },
    @{ Name = 'tracked PE import inventory'; Path = $trackedPeImports; EvidencePath = $paths.peImports; Expected = [string]$policy.reviewInventory.peImportsSha256 }
)) {
    if (-not (Test-Path -LiteralPath $inventoryBinding.Path -PathType Leaf)) {
        throw "PDFium $($inventoryBinding.Name) is missing."
    }
    $trackedHash = (Get-FileHash -LiteralPath $inventoryBinding.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    $evidenceHash = (Get-FileHash -LiteralPath $inventoryBinding.EvidencePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($trackedHash -ne $inventoryBinding.Expected -or $evidenceHash -ne $trackedHash) {
        throw "PDFium $($inventoryBinding.Name) does not exactly match retained evidence."
    }
}

$manifest = Get-Content -LiteralPath $paths.buildManifest -Raw | ConvertFrom-Json
if ([string]$manifest.profileId -ne [string]$policy.profileId -or
    [string]$manifest.sourceRevision -ne [string]$policy.sourceRevision -or
    [string]$manifest.binarySha256 -ne [string]$policy.evidenceBinding.binarySha256 -or
    [string]$manifest.targetDependenciesSha256 -ne [string]$policy.evidenceBinding.targetDependenciesSha256 -or
    [string]$manifest.peImportsSha256 -ne [string]$policy.evidenceBinding.peImportsSha256) {
    throw 'PDFium build manifest does not match the dependency review policy.'
}
if ([bool]$manifest.features.v8 -or [bool]$manifest.features.xfa -or
    [bool]$manifest.features.skia -or [bool]$manifest.features.icuDataFile) {
    throw 'PDFium reviewed dependency profile unexpectedly enables a prohibited feature.'
}

$componentIds = @{}
$noticeIds = @{}
foreach ($component in @($policy.components)) {
    $componentId = [string]$component.id
    if ([string]::IsNullOrWhiteSpace($componentId) -or $componentIds.ContainsKey($componentId)) {
        throw "PDFium dependency policy has a missing or duplicate component ID: $componentId"
    }
    $componentIds[$componentId] = $true
    if ([string]::IsNullOrWhiteSpace([string]$component.classification) -or
        [string]::IsNullOrWhiteSpace([string]$component.source) -or
        [string]::IsNullOrWhiteSpace([string]$component.sourceRevision) -or
        [string]::IsNullOrWhiteSpace([string]$component.licenseSpdx) -or
        [string]::IsNullOrWhiteSpace([string]$component.noticeDisposition)) {
        throw "PDFium component '$componentId' lacks required provenance fields."
    }

    $revisionRoot = [IO.Path]::GetFullPath((Join-Path $SourceRoot ([string]$component.sourceRoot)))
    $sourcePrefix = $SourceRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if ($revisionRoot -ne $SourceRoot -and
        -not $revisionRoot.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "PDFium component '$componentId' source root escapes the checkout."
    }
    $actualRevision = (& git -C $revisionRoot rev-parse HEAD 2>$null)
    if ($LASTEXITCODE -ne 0 -or ([string]$actualRevision).Trim() -ne [string]$component.sourceRevision) {
        throw "PDFium component '$componentId' source revision mismatch."
    }

    foreach ($license in @($component.licenses)) {
        $licensePath = [IO.Path]::GetFullPath((Join-Path $SourceRoot ([string]$license.path)))
        if (-not $licensePath.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not (Test-Path -LiteralPath $licensePath -PathType Leaf)) {
            throw "PDFium component '$componentId' license path is missing or unsafe."
        }
        $actualLicenseHash = (Get-FileHash -LiteralPath $licensePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualLicenseHash -ne [string]$license.sha256) {
            throw "PDFium component '$componentId' license hash mismatch."
        }
        if ([bool]$license.includeInNotice) {
            $noticeId = [string]$license.noticeId
            if ($noticeIds.ContainsKey($noticeId) -and $noticeIds[$noticeId] -ne [string]$license.path) {
                throw "PDFium notice ID '$noticeId' maps to inconsistent source paths."
            }
            $noticeIds[$noticeId] = [string]$license.path
        }
    }
}

$counts = @{}
foreach ($label in @(Get-Content -LiteralPath $paths.targetDependencies)) {
    if ([string]::IsNullOrWhiteSpace($label)) {
        continue
    }
    $matches = New-Object System.Collections.Generic.List[string]
    foreach ($component in @($policy.components)) {
        $matched = @($component.exactLabels) -contains $label
        if (-not $matched) {
            foreach ($prefix in @($component.labelPrefixes)) {
                if ($label.StartsWith([string]$prefix, [StringComparison]::Ordinal)) {
                    $matched = $true
                    break
                }
            }
        }
        if ($matched) {
            $matches.Add([string]$component.id)
        }
    }
    if ($matches.Count -ne 1) {
        throw "PDFium target dependency must map to exactly one component: '$label' matched $($matches.Count)."
    }
    $counts[$matches[0]] = 1 + [int]$counts[$matches[0]]
}
if (($counts.Values | Measure-Object -Sum).Sum -ne 240) {
    throw 'PDFium target dependency closure must contain exactly 240 mapped labels.'
}
foreach ($component in @($policy.components)) {
    if (-not $counts.ContainsKey([string]$component.id)) {
        throw "PDFium component '$($component.id)' has no target dependency evidence."
    }
}

$importMatches = [regex]::Matches(
    (Get-Content -LiteralPath $paths.peImports -Raw),
    '(?im)^\s+([A-Z0-9]+\.dll)\s*$')
$actualImports = @($importMatches | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
$expectedImports = @($policy.permittedPeImports | ForEach-Object { [string]$_ } | Sort-Object -Unique)
if (($actualImports -join '|') -cne ($expectedImports -join '|') -or $actualImports.Count -ne 4) {
    throw "PDFium PE import set is not the four reviewed Windows system APIs: $($actualImports -join ', ')"
}

if (-not (Test-Path -LiteralPath $NoticePath -PathType Leaf)) {
    throw "PDFium dependency-mapped notice is missing: $NoticePath"
}
$noticeFirstLine = Get-Content -LiteralPath $NoticePath -TotalCount 1
if ($noticeFirstLine.Trim() -ne 'REVIEW STATUS: DEPENDENCY-MAPPED') {
    throw 'PDFium dependency-mapped notice must not claim complete approval.'
}
$noticeHash = (Get-FileHash -LiteralPath $NoticePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($noticeHash -ne [string]$policy.noticeBundle.sha256) {
    throw 'PDFium dependency-mapped notice hash mismatch.'
}
$noticeText = Get-Content -LiteralPath $NoticePath -Raw
$sectionMatches = [regex]::Matches($noticeText, '(?m)^===== BEGIN NOTICE: ([a-z0-9.-]+) =====$')
$actualNoticeIds = @($sectionMatches | ForEach-Object { $_.Groups[1].Value } | Sort-Object)
$expectedNoticeIds = @($noticeIds.Keys | Sort-Object)
if (($actualNoticeIds -join '|') -cne ($expectedNoticeIds -join '|')) {
    throw 'PDFium dependency-mapped notice sections do not match the reviewed component matrix.'
}

Write-Host "PDFium dependency review policy: PASS ($($counts.Values | Measure-Object -Sum | Select-Object -ExpandProperty Sum) labels, $($actualNoticeIds.Count) notices, 4 system imports; approval remains false)"
