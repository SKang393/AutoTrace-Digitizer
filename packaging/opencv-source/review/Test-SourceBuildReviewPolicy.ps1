# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$EvidenceRoot,
    [string]$PolicyPath,
    [string]$NoticePath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-ReviewKey($Item) {
    return ('{0}|{1}' -f [string]$Item.kind, [string]$Item.name).ToLowerInvariant()
}

function Get-NoticeSectionText {
    param([string]$Text, [string]$Id)

    $pattern = '(?ms)^===== BEGIN NOTICE: ' + [regex]::Escape($Id) + ' =====\r?\n(?<body>.*?)^===== END NOTICE: ' + [regex]::Escape($Id) + ' =====\r?$'
    $match = [regex]::Match($Text, $pattern)
    if (-not $match.Success) {
        throw "Notice section '$Id' is missing or malformed."
    }
    return $match.Groups['body'].Value
}

function Get-TextSha256([string]$Text) {
    $normalizedText = $Text -replace "`r`n", "`n"
    $bytes = [Text.Encoding]::UTF8.GetBytes($normalizedText)
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha256.ComputeHash($bytes)).Replace('-', '')).ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
}

if ([string]::IsNullOrWhiteSpace($PolicyPath)) {
    $PolicyPath = Join-Path $PSScriptRoot 'source-build-review-policy.json'
}
if ([string]::IsNullOrWhiteSpace($NoticePath)) {
    $NoticePath = Join-Path $PSScriptRoot 'third-party-notices.candidate.txt'
}
if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) {
    $EvidenceRoot = Join-Path $PSScriptRoot '..\..\..\artifacts\goal19-opencv-source\evidence-repro-pass2-final-a'
}

if (-not (Test-Path -LiteralPath $PolicyPath -PathType Leaf)) { throw "Review policy is missing: $PolicyPath" }
if (-not (Test-Path -LiteralPath $NoticePath -PathType Leaf)) { throw "Candidate notice bundle is missing: $NoticePath" }
$inventoryPath = Join-Path $EvidenceRoot 'dependency-inventory.json'
if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) { throw "Evidence inventory is missing: $inventoryPath" }
$sourceRevisionsPath = Join-Path $EvidenceRoot 'source-revisions.json'
if (-not (Test-Path -LiteralPath $sourceRevisionsPath -PathType Leaf)) { throw "Evidence source revisions are missing: $sourceRevisionsPath" }
$binaryPath = Join-Path $EvidenceRoot 'bin\OpenCvSharpExtern.dll'
if (-not (Test-Path -LiteralPath $binaryPath -PathType Leaf)) { throw "Evidence binary is missing: $binaryPath" }

$policy = Get-Content -LiteralPath $PolicyPath -Raw | ConvertFrom-Json
$inventory = Get-Content -LiteralPath $inventoryPath -Raw | ConvertFrom-Json
$sourceRevisions = Get-Content -LiteralPath $sourceRevisionsPath -Raw | ConvertFrom-Json
$noticeText = Get-Content -LiteralPath $NoticePath -Raw
$errors = New-Object System.Collections.Generic.List[string]

if ([string]$policy.overallReviewStatus -ne 'requires-maintainer-attestation') {
    $errors.Add('Review policy must remain blocked on maintainer attestation.')
}
if ([string]$inventory.reviewStatus -ne 'requires-review') {
    $errors.Add('Mechanical evidence inventory must remain requires-review.')
}
if ([string]$inventory.profileId -ne [string]$policy.profileId) {
    $errors.Add('Evidence inventory profile does not match the review policy.')
}
foreach ($revisionName in @('openCvSharp', 'openCv', 'vcpkg')) {
    if ([string]$sourceRevisions.$revisionName -ne [string]$policy.sourceRevisions.$revisionName) {
        $errors.Add("Source revision mismatch for $revisionName.")
    }
}
$actualBinarySha256 = (Get-FileHash -LiteralPath $binaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualBinarySha256 -ne [string]$policy.binarySha256) {
    $errors.Add('Evidence binary SHA-256 does not match the review policy.')
}
if (@($policy.entries).Count -ne 15 -or @($inventory.dependencies).Count -ne 15) {
    $errors.Add('Policy and evidence inventory must each contain exactly 15 entries.')
}

$policyByKey = @{}
foreach ($entry in @($policy.entries)) {
    $key = Get-ReviewKey $entry
    if ($policyByKey.ContainsKey($key)) { $errors.Add("Duplicate policy entry: $key"); continue }
    $policyByKey[$key] = $entry
    if ([string]::IsNullOrWhiteSpace([string]$entry.source) -or [string]::IsNullOrWhiteSpace([string]$entry.license)) {
        $errors.Add("Policy entry lacks source or license: $key")
    }
}

foreach ($dependency in @($inventory.dependencies)) {
    $key = Get-ReviewKey $dependency
    if (-not $policyByKey.ContainsKey($key)) { $errors.Add("Evidence entry lacks exactly one policy disposition: $key") }
}
foreach ($key in $policyByKey.Keys) {
    if (-not (@($inventory.dependencies | ForEach-Object { Get-ReviewKey $_ }) -contains $key)) {
        $errors.Add("Policy entry has no evidence inventory match: $key")
    }
}

$noticeIds = @($policy.noticeBundle.sections | ForEach-Object { [string]$_.id })
foreach ($section in @($policy.noticeBundle.sections)) {
    try {
        $actual = Get-TextSha256 (Get-NoticeSectionText -Text $noticeText -Id ([string]$section.id))
        if ([string]::IsNullOrWhiteSpace([string]$section.sha256) -or $actual -ne [string]$section.sha256) {
            $errors.Add("Notice hash mismatch for section '$($section.id)'.")
        }
    }
    catch { $errors.Add($_.Exception.Message) }
}

foreach ($entry in @($policy.entries)) {
    $key = Get-ReviewKey $entry
    switch ([string]$entry.noticeDisposition) {
        'candidate-included' {
            foreach ($noticeId in @($entry.noticeIds)) {
                if ($noticeIds -notcontains [string]$noticeId) { $errors.Add("Policy entry references unknown notice '$noticeId': $key") }
            }
        }
        { $_ -in @('not-required-not-shipped', 'not-required-not-linked', 'merged-with-zlib-lib') } { }
        'maintainer-attestation-required' {
            if ([string]$entry.reviewStatus -ne 'requires-maintainer-attestation') { $errors.Add("Microsoft entry must require maintainer attestation: $key") }
        }
        default { $errors.Add("Unknown notice disposition for $key") }
    }
}

if ([string]$policy.microsoftStaticRuntimeAttestation.status -ne 'required' -or
    [string]::IsNullOrWhiteSpace([string]$policy.microsoftStaticRuntimeAttestation.requiredMaintainerStatement) -or
    @($policy.microsoftStaticRuntimeAttestation.termReferences).Count -lt 2) {
    $errors.Add('Microsoft static runtime attestation details are incomplete.')
}

if ($errors.Count -gt 0) {
    throw ("Source-build review policy: BLOCKED`n" + ($errors -join [Environment]::NewLine))
}

Write-Host 'Source-build review policy: PASS (overall release status remains blocked on maintainer attestation)'
