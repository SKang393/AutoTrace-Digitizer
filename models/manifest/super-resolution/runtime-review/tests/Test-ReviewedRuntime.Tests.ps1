# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param([string]$RuntimeRoot)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$reviewRoot = Split-Path -Parent $PSScriptRoot
$validator = Join-Path $reviewRoot 'Test-ReviewedRuntime.ps1'
$policy = Join-Path $reviewRoot 'realesrgan-ncnn-vulkan-win-x64-runtime-policy.jsonc'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $reviewRoot '..\..\..\..'))
$attestation = Join-Path $projectRoot 'provenance-private\realesrgan-microsoft-vcomp140-attestation.json'
if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $RuntimeRoot = Join-Path $projectRoot 'artifacts\goal19-realesrgan\runtime-authorized-vcomp-14.44.35211'
}

& $validator -PolicyPath $policy -RuntimeRoot $RuntimeRoot -AttestationPath $attestation

$temporaryPolicy = Join-Path ([IO.Path]::GetTempPath()) ('graphreader-ncnn-policy-' + [guid]::NewGuid().ToString('N') + '.json')
$temporaryAttestation = Join-Path ([IO.Path]::GetTempPath()) ('graphreader-ncnn-attestation-' + [guid]::NewGuid().ToString('N') + '.json')
try {
    $mutatedPolicy = (Get-Content -LiteralPath $policy -Raw).Replace(
        '55aba23cdcd6484fbb06f4155b8ca75adfce7a881f10afd0c49457165e677164',
        '65aba23cdcd6484fbb06f4155b8ca75adfce7a881f10afd0c49457165e677164')
    [IO.File]::WriteAllText($temporaryPolicy, $mutatedPolicy, [Text.UTF8Encoding]::new($false))
    $failedAsExpected = $false
    try {
        & $validator -PolicyPath $temporaryPolicy -RuntimeRoot $RuntimeRoot -AttestationPath $attestation
    }
    catch {
        if ($_.Exception.Message -match 'size or SHA-256 mismatch|exact unmodified copy|attestation component') {
            $failedAsExpected = $true
        }
        else { throw }
    }
    if (-not $failedAsExpected) { throw 'Validator accepted a modified vcomp140.dll policy hash.' }

    $mutatedPolicy = (Get-Content -LiteralPath $policy -Raw).Replace(
        '0ab4530b4ca4c0bc12fc97b36d8ad1d1383a66aad328b03104215da288e8c24e',
        '1ab4530b4ca4c0bc12fc97b36d8ad1d1383a66aad328b03104215da288e8c24e')
    [IO.File]::WriteAllText($temporaryPolicy, $mutatedPolicy, [Text.UTF8Encoding]::new($false))
    $failedAsExpected = $false
    try {
        & $validator -PolicyPath $temporaryPolicy -RuntimeRoot $RuntimeRoot -AttestationPath $attestation
    }
    catch {
        if ($_.Exception.Message -match 'notice is missing or mismatched') {
            $failedAsExpected = $true
        }
        else { throw }
    }
    if (-not $failedAsExpected) { throw 'Validator accepted a modified Microsoft notice hash.' }

    $mutatedPolicy = (Get-Content -LiteralPath $policy -Raw).Replace(
        '"localAdapter": false',
        '"localAdapter": true')
    [IO.File]::WriteAllText($temporaryPolicy, $mutatedPolicy, [Text.UTF8Encoding]::new($false))
    $failedAsExpected = $false
    try {
        & $validator -PolicyPath $temporaryPolicy -RuntimeRoot $RuntimeRoot -AttestationPath $attestation
    }
    catch {
        if ($_.Exception.Message -match "Approval 'localAdapter' must remain false") {
            $failedAsExpected = $true
        }
        else { throw }
    }
    if (-not $failedAsExpected) { throw 'Validator accepted local adapter approval in the provenance-only policy.' }

    $mutatedAttestation = (Get-Content -LiteralPath $attestation -Raw).Replace(
        'The release maintainer confirms that Graph Auto Reader',
        'The release maintainer has not confirmed that Graph Auto Reader')
    [IO.File]::WriteAllText($temporaryAttestation, $mutatedAttestation, [Text.UTF8Encoding]::new($false))
    $failedAsExpected = $false
    try {
        & $validator -PolicyPath $policy -RuntimeRoot $RuntimeRoot -AttestationPath $temporaryAttestation
    }
    catch {
        if ($_.Exception.Message -match 'does not exactly match the policy') {
            $failedAsExpected = $true
        }
        else { throw }
    }
    if (-not $failedAsExpected) { throw 'Validator accepted a modified authority statement.' }
}
finally {
    if (Test-Path -LiteralPath $temporaryPolicy) { Remove-Item -LiteralPath $temporaryPolicy -Force }
    if (Test-Path -LiteralPath $temporaryAttestation) { Remove-Item -LiteralPath $temporaryAttestation -Force }
}

Write-Host 'Reviewed Real-ESRGAN NCNN runtime policy tests: PASS'
