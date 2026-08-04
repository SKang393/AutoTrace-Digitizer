# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$PolicyPath,
    [string]$RuntimeRoot,
    [string]$AttestationPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-NormalizedSha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-ImageDimensions([string]$Path) {
    Add-Type -AssemblyName PresentationCore
    $stream = [IO.File]::OpenRead($Path)
    try {
        $decoder = [Windows.Media.Imaging.BitmapDecoder]::Create(
            $stream,
            [Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat,
            [Windows.Media.Imaging.BitmapCacheOption]::OnLoad)
        return [pscustomobject]@{
            Width = $decoder.Frames[0].PixelWidth
            Height = $decoder.Frames[0].PixelHeight
        }
    }
    finally {
        $stream.Dispose()
    }
}

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..'))
if ([string]::IsNullOrWhiteSpace($PolicyPath)) {
    $PolicyPath = Join-Path $PSScriptRoot 'realesrgan-ncnn-vulkan-win-x64-runtime-policy.jsonc'
}
if (-not (Test-Path -LiteralPath $PolicyPath -PathType Leaf)) {
    throw "Runtime review policy is missing: $PolicyPath"
}
$policy = Get-Content -LiteralPath $PolicyPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $RuntimeRoot = Join-Path $projectRoot ([string]$policy.defaultRuntimeRoot)
}
if ([string]::IsNullOrWhiteSpace($AttestationPath)) {
    $AttestationPath = Join-Path $projectRoot ([string]$policy.defaultPrivateAttestationPath)
}
if (-not (Test-Path -LiteralPath $RuntimeRoot -PathType Container)) {
    throw "Reviewed runtime root is missing: $RuntimeRoot"
}
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot).TrimEnd([char[]]@('\', '/'))
if (-not (Test-Path -LiteralPath $AttestationPath -PathType Leaf)) {
    throw "Private runtime authority attestation is missing: $AttestationPath"
}
$attestation = Get-Content -LiteralPath $AttestationPath -Raw | ConvertFrom-Json
$errors = New-Object System.Collections.Generic.List[string]

if ([int]$policy.schemaVersion -ne 1 -or
    [string]$policy.overallReviewStatus -ne 'reviewed-provenance-only' -or
    -not [bool]$policy.runtimeRedistributionProvenanceReviewed) {
    $errors.Add('Runtime policy review status is invalid.')
}
foreach ($approvalName in @('localAdapter', 'scientificFidelity', 'offlineCpu', 'cleanMachine', 'production', 'release')) {
    if ([bool]$policy.approvals.$approvalName) {
        $errors.Add("Approval '$approvalName' must remain false in the provenance-only policy.")
    }
}

if ([int]$attestation.schemaVersion -ne 1 -or
    -not [bool]$attestation.private -or
    [bool]$attestation.gitEligibility -or
    -not [bool]$attestation.authorityConfirmed) {
    $errors.Add('Private runtime authority attestation metadata is invalid.')
}
if ([string]$attestation.profileId -ne [string]$policy.profileId -or
    [string]$attestation.statement -cne [string]$policy.requiredMaintainerStatement) {
    $errors.Add('Private runtime authority attestation does not exactly match the policy.')
}
if ((@($attestation.termReferences) -join '|') -cne (@($policy.officialTerms) -join '|')) {
    $errors.Add('Private runtime authority attestation terms do not match the policy.')
}

$redistRecordPath = [string]$policy.sourceRedistRecord.path
if (-not (Test-Path -LiteralPath $redistRecordPath -PathType Leaf) -or
    (Get-NormalizedSha256 $redistRecordPath) -ne [string]$policy.sourceRedistRecord.sha256) {
    $errors.Add('Installed Visual Studio Redist.txt source record is missing or has the wrong hash.')
}
if ([string]$attestation.sourceRedistRecord.path -ne $redistRecordPath -or
    [string]$attestation.sourceRedistRecord.sha256 -ne [string]$policy.sourceRedistRecord.sha256) {
    $errors.Add('Attestation Redist.txt record does not match the policy.')
}

$expectedFiles = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
$reviewedNotices = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($notice in @($policy.noticeFiles)) {
    $noticePath = ([string]$notice.path) -replace '\\', '/'
    [void]$reviewedNotices.Add($noticePath)
    $absoluteNoticePath = Join-Path $projectRoot $noticePath
    if (-not (Test-Path -LiteralPath $absoluteNoticePath -PathType Leaf) -or
        (Get-NormalizedSha256 $absoluteNoticePath) -ne [string]$notice.sha256) {
        $errors.Add("Reviewed runtime notice is missing or mismatched: $noticePath")
    }
}
foreach ($asset in @($policy.runtimeAssets)) {
    $relative = ([string]$asset.path) -replace '\\', '/'
    [void]$expectedFiles.Add($relative)
    foreach ($requiredField in @('source', 'purpose', 'copyright', 'license', 'privacyStatus')) {
        if ([string]::IsNullOrWhiteSpace([string]$asset.$requiredField)) {
            $errors.Add("Runtime asset '$relative' lacks required provenance field '$requiredField'.")
        }
    }
    if ([bool]$asset.gitEligibility) {
        $errors.Add("Runtime binary asset '$relative' must remain Git-ineligible.")
    }
    $path = Join-Path $RuntimeRoot $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $errors.Add("Runtime asset is missing: $relative")
        continue
    }
    if ((Get-Item -LiteralPath $path).Length -ne [long]$asset.bytes -or
        (Get-NormalizedSha256 $path) -ne [string]$asset.sha256) {
        $errors.Add("Runtime asset size or SHA-256 mismatch: $relative")
    }
    foreach ($noticePath in @($asset.noticePaths)) {
        $normalizedNoticePath = ([string]$noticePath) -replace '\\', '/'
        if (-not $reviewedNotices.Contains($normalizedNoticePath)) {
            $errors.Add("Runtime asset notice is not checksum-bound by the policy: $normalizedNoticePath")
        }
    }
}
foreach ($evidence in @($policy.evidenceOnlyFiles)) {
    $relative = ([string]$evidence.path) -replace '\\', '/'
    [void]$expectedFiles.Add($relative)
    $path = Join-Path $RuntimeRoot $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
        (Get-NormalizedSha256 $path) -ne [string]$evidence.sha256) {
        $errors.Add("Runtime evidence-only file is missing or mismatched: $relative")
    }
}

$actualFiles = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($file in Get-ChildItem -LiteralPath $RuntimeRoot -Recurse -File) {
    $relative = $file.FullName.Substring($RuntimeRoot.Length).TrimStart([char[]]@('\', '/')) -replace '\\', '/'
    [void]$actualFiles.Add($relative)
}
foreach ($relative in $expectedFiles) {
    if (-not $actualFiles.Contains($relative)) { $errors.Add("Reviewed runtime inventory is missing: $relative") }
}
foreach ($relative in $actualFiles) {
    if (-not $expectedFiles.Contains($relative)) { $errors.Add("Reviewed runtime contains an unclassified file: $relative") }
}

$vcomp = @($policy.runtimeAssets | Where-Object { [string]$_.path -eq 'vcomp140.dll' })
if ($vcomp.Count -ne 1) {
    $errors.Add('Policy must contain exactly one vcomp140.dll asset.')
}
else {
    $vcompAsset = $vcomp[0]
    $runtimeVcomp = Join-Path $RuntimeRoot 'vcomp140.dll'
    $sourceVcomp = [string]$vcompAsset.source
    if (-not (Test-Path -LiteralPath $sourceVcomp -PathType Leaf) -or
        (Get-NormalizedSha256 $sourceVcomp) -ne [string]$vcompAsset.sha256 -or
        (Get-NormalizedSha256 $runtimeVcomp) -ne (Get-NormalizedSha256 $sourceVcomp)) {
        $errors.Add('Runtime vcomp140.dll is not an exact unmodified copy of the reviewed VC Redist source.')
    }
    if ([string](Get-Item -LiteralPath $runtimeVcomp).VersionInfo.FileVersion -ne [string]$vcompAsset.version) {
        $errors.Add('Runtime vcomp140.dll file version does not match the policy.')
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $runtimeVcomp
    if ([string]$signature.Status -ne [string]$vcompAsset.authenticode.requiredStatus -or
        [string]$signature.SignerCertificate.Subject -notlike "*$([string]$vcompAsset.authenticode.signerSubjectContains)*" -or
        [string]$signature.SignerCertificate.Thumbprint -ne [string]$vcompAsset.authenticode.signerThumbprint) {
        $errors.Add('Runtime vcomp140.dll Authenticode identity does not match the policy.')
    }
    if ([string]$attestation.component.sourcePath -ne $sourceVcomp -or
        [string]$attestation.component.sha256 -ne [string]$vcompAsset.sha256 -or
        [string]$attestation.component.version -ne [string]$vcompAsset.version -or
        -not [bool]$attestation.component.unmodified -or
        [bool]$attestation.component.gitEligibility) {
        $errors.Add('Private vcomp140.dll attestation component does not match the reviewed source asset.')
    }
}

$smoke = $policy.smokeEvidence
$smokeInput = Join-Path $projectRoot ([string]$smoke.inputPath)
$smokeOutput = Join-Path $RuntimeRoot 'authorized-vcomp-smoke.png'
if ([string]$smoke.status -ne 'execution-verified' -or [int]$smoke.exitCode -ne 0 -or
    [string]$smoke.provider -ne 'vulkan' -or [double]$smoke.elapsedMs -le 0) {
    $errors.Add('Direct runtime smoke metadata is invalid.')
}
if (-not (Test-Path -LiteralPath $smokeInput -PathType Leaf) -or
    (Get-NormalizedSha256 $smokeInput) -ne [string]$smoke.inputSha256) {
    $errors.Add('Direct runtime smoke input is missing or mismatched.')
}
if ((Get-NormalizedSha256 $smokeOutput) -ne [string]$smoke.outputSha256) {
    $errors.Add('Direct runtime smoke output is mismatched.')
}
if ((Test-Path -LiteralPath $smokeInput -PathType Leaf) -and (Test-Path -LiteralPath $smokeOutput -PathType Leaf)) {
    $inputDimensions = Get-ImageDimensions $smokeInput
    $outputDimensions = Get-ImageDimensions $smokeOutput
    if ($inputDimensions.Width -ne [int]$smoke.inputWidth -or
        $inputDimensions.Height -ne [int]$smoke.inputHeight -or
        $outputDimensions.Width -ne [int]$smoke.outputWidth -or
        $outputDimensions.Height -ne [int]$smoke.outputHeight -or
        $outputDimensions.Width -ne 2 * $inputDimensions.Width -or
        $outputDimensions.Height -ne 2 * $inputDimensions.Height) {
        $errors.Add('Direct runtime smoke dimensions do not prove exact 2x output.')
    }
}

if ($errors.Count -gt 0) {
    throw ("Reviewed Real-ESRGAN NCNN runtime: BLOCKED`n" + ($errors -join [Environment]::NewLine))
}

Write-Host 'Reviewed Real-ESRGAN NCNN runtime: PASS (redistribution provenance only)'
Write-Host 'Local adapter, scientific, CPU, clean-machine, production, and release approvals: FALSE'
