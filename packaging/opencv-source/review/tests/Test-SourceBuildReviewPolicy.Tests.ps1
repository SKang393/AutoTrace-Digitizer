# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$EvidenceRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$reviewRoot = Split-Path -Parent $PSScriptRoot
$validator = Join-Path $reviewRoot 'Test-SourceBuildReviewPolicy.ps1'
$policy = Join-Path $reviewRoot 'source-build-review-policy.json'
$notice = Join-Path $reviewRoot 'third-party-notices.candidate.txt'
$attestation = Join-Path $reviewRoot '..\..\..\provenance-private\opencv-microsoft-static-runtime-attestation.json'

& $validator -EvidenceRoot $EvidenceRoot -PolicyPath $policy -NoticePath $notice -AttestationPath $attestation

$temporaryPolicy = Join-Path ([IO.Path]::GetTempPath()) ('graphreader-review-policy-' + [guid]::NewGuid().ToString('N') + '.json')
$temporaryAttestation = Join-Path ([IO.Path]::GetTempPath()) ('graphreader-review-attestation-' + [guid]::NewGuid().ToString('N') + '.json')
try {
    $mutated = (Get-Content -LiteralPath $policy -Raw).Replace('KERNEL32.dll', 'KERNEL33.dll')
    [IO.File]::WriteAllText($temporaryPolicy, $mutated, (New-Object Text.UTF8Encoding($false)))

    $failedAsExpected = $false
    try {
        & $validator -EvidenceRoot $EvidenceRoot -PolicyPath $temporaryPolicy -NoticePath $notice -AttestationPath $attestation
    }
    catch {
        if ($_.Exception.Message -match 'Evidence entry lacks exactly one policy disposition') {
            $failedAsExpected = $true
        }
        else {
            throw
        }
    }

    if (-not $failedAsExpected) {
        throw 'Validator accepted a policy that does not map every evidence entry.'
    }

    $mutatedAttestation = (Get-Content -LiteralPath $attestation -Raw).Replace(
        'The release maintainer confirms authority to distribute',
        'The release maintainer has not confirmed authority to distribute')
    [IO.File]::WriteAllText($temporaryAttestation, $mutatedAttestation, (New-Object Text.UTF8Encoding($false)))
    $failedAsExpected = $false
    try {
        & $validator -EvidenceRoot $EvidenceRoot -PolicyPath $policy -NoticePath $notice -AttestationPath $temporaryAttestation
    }
    catch {
        if ($_.Exception.Message -match 'does not exactly match the required statement') {
            $failedAsExpected = $true
        }
        else {
            throw
        }
    }
    if (-not $failedAsExpected) {
        throw 'Validator accepted a modified maintainer attestation statement.'
    }
}
finally {
    if (Test-Path -LiteralPath $temporaryPolicy) {
        Remove-Item -LiteralPath $temporaryPolicy -Force
    }
    if (Test-Path -LiteralPath $temporaryAttestation) {
        Remove-Item -LiteralPath $temporaryAttestation -Force
    }
}

Write-Host 'Source-build review policy tests: PASS'
