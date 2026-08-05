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
$releaseAuditPath = Join-Path $RepositoryRoot 'packaging\common\release-audit.json'
$installerPath = Join-Path $RepositoryRoot 'packaging\opencv-source\Install-ReviewedRuntime.ps1'

foreach ($required in @($releaseAuditPath, $installerPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Release OpenCV input is missing: $required"
    }
}

$releaseAudit = Get-Content -LiteralPath $releaseAuditPath -Raw | ConvertFrom-Json
$components = @($releaseAudit.components | Where-Object {
        [string]$_.id -ceq 'opencvsharp-native'
    })
if ($components.Count -ne 1) {
    throw "Release audit must contain exactly one 'opencvsharp-native' component."
}
$component = $components[0]
$expectedHash = [string]$component.artifactSha256
if ([string]$component.checksumPolicy -cne 'exact-binary' -or
    $expectedHash -notmatch '^[0-9a-fA-F]{64}$') {
    throw "Release OpenCV component must use exact-binary coverage with an artifact SHA-256."
}
$expectedHash = $expectedHash.ToLowerInvariant()

$gates = @($releaseAudit.mandatoryEvidenceGates | Where-Object {
        [string]$_.id -ceq 'opencv-clean-machine-load'
    })
if ($gates.Count -ne 1) {
    throw "Release audit must contain exactly one 'opencv-clean-machine-load' gate."
}
$gate = $gates[0]
if ([string]$gate.status -cne 'pass' -or @($gate.evidence).Count -eq 0) {
    throw "OpenCV clean-machine evidence gate is not directly passing."
}
foreach ($evidence in @($gate.evidence)) {
    $relativePath = [string]$evidence.path
    $evidenceHash = [string]$evidence.sha256
    if ([IO.Path]::IsPathRooted($relativePath) -or
        $relativePath -match '(^|[\\/])\.\.([\\/]|$)' -or
        $evidenceHash -notmatch '^[0-9a-fA-F]{64}$') {
        throw "OpenCV clean-machine evidence uses an unsafe path or invalid checksum."
    }
    $evidencePath = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $relativePath))
    if (-not $evidencePath.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $evidencePath -PathType Leaf)) {
        throw "OpenCV clean-machine evidence is missing or outside the repository: $relativePath"
    }
    $actualEvidenceHash = (Get-FileHash -LiteralPath $evidencePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [string]::Equals($actualEvidenceHash, $evidenceHash, [StringComparison]::OrdinalIgnoreCase)) {
        throw "OpenCV clean-machine evidence checksum differs: $relativePath"
    }
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installerPath `
    -EvidenceRoot $EvidenceRoot `
    -DestinationRoot $DestinationRoot `
    -RepositoryRoot $RepositoryRoot
if ($LASTEXITCODE -ne 0) {
    throw "Reviewed OpenCV runtime installation failed with exit code $LASTEXITCODE."
}

$metadataPath = Join-Path $DestinationRoot 'reviewed-opencv-runtime.json'
$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if ([string]$metadata.binarySha256 -cne $expectedHash -or
    -not [bool]$metadata.provenanceValidated -or
    [string]$metadata.noticeReviewStatus -cne 'complete' -or
    [bool]$metadata.cleanMachineEvidence -or
    [bool]$metadata.releaseApproved) {
    throw "Reviewed OpenCV runtime does not match the exact release-audit boundary."
}

$metadata.cleanMachineEvidence = $true
$metadata.releaseApproved = $true
[IO.File]::WriteAllText(
    $metadataPath,
    (($metadata | ConvertTo-Json -Depth 10) + [Environment]::NewLine),
    [Text.UTF8Encoding]::new($false))

Write-Host "Release-approved OpenCV runtime installed: $($metadata.binarySha256)"
Write-Output ([pscustomobject]@{
    DestinationPath = Join-Path $DestinationRoot 'OpenCvSharpExtern.dll'
    MetadataPath = $metadataPath
    BinarySha256 = [string]$metadata.binarySha256
    CleanMachineEvidence = $true
    ReleaseApproved = $true
})
