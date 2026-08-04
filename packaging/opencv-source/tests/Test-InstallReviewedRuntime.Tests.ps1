# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$profileRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$installerPath = Join-Path $profileRoot 'Install-ReviewedRuntime.ps1'
$passed = 0
$failed = 0

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$Content
    )

    $parent = Split-Path $Path -Parent
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    [IO.File]::WriteAllText($Path, $Content, [Text.UTF8Encoding]::new($false))
}

function New-Fixture {
    param([switch]$TamperSource)

    $root = Join-Path ([IO.Path]::GetTempPath()) ('GraphReader-ReviewedOpenCvInstall-' + [Guid]::NewGuid().ToString('N'))
    $repository = Join-Path $root 'repository'
    $profile = Join-Path $repository 'packaging\opencv-source'
    $evidence = Join-Path $root 'evidence'
    $destination = Join-Path $root 'destination'
    New-Item -ItemType Directory -Path (Join-Path $profile 'review'), (Join-Path $evidence 'bin'), (Join-Path $destination 'runtimes\win-x64\native') -Force | Out-Null

    $reviewedBytes = [Text.Encoding]::UTF8.GetBytes('reviewed deterministic OpenCV runtime')
    $sourcePath = Join-Path $evidence 'bin\OpenCvSharpExtern.dll'
    [IO.File]::WriteAllBytes($sourcePath, $reviewedBytes)
    $reviewedHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($TamperSource.IsPresent) {
        [IO.File]::WriteAllText($sourcePath, 'tampered runtime', [Text.UTF8Encoding]::new($false))
    }
    [IO.File]::WriteAllText((Join-Path $destination 'runtimes\win-x64\native\OpenCvSharpExtern.dll'), 'package runtime', [Text.UTF8Encoding]::new($false))

    $validator = @'
param([string]$EvidenceRoot)
if (-not (Test-Path -LiteralPath $EvidenceRoot -PathType Container)) { exit 1 }
exit 0
'@
    Write-Utf8NoBom -Path (Join-Path $profile 'Test-SourceAuditEvidence.ps1') -Content $validator
    Write-Utf8NoBom -Path (Join-Path $profile 'review\Test-SourceBuildReviewPolicy.ps1') -Content $validator
    $policy = [ordered]@{
        schemaVersion = 1
        profileId = 'graphreader-axis-minimal-win-x64'
        evidenceRootName = 'fixture-evidence'
        overallReviewStatus = 'reviewed-provenance-only'
        binarySha256 = $reviewedHash
        sourceRevisions = [ordered]@{
            openCvSharp = 'fixture-opencvsharp'
            openCv = 'fixture-opencv'
            vcpkg = 'fixture-vcpkg'
        }
        noticeBundle = [ordered]@{ reviewStatus = 'complete' }
        microsoftStaticRuntimeAttestation = [ordered]@{ status = 'recorded-private' }
    }
    Write-Utf8NoBom -Path (Join-Path $profile 'review\source-build-review-policy.json') -Content ($policy | ConvertTo-Json -Depth 10)

    return [pscustomobject]@{
        Root = $root
        Repository = $repository
        Evidence = $evidence
        Destination = $destination
        ReviewedHash = $reviewedHash
    }
}

function Invoke-Case {
    param(
        [Parameter(Mandatory = $true)] [string]$Name,
        [Parameter(Mandatory = $true)] [scriptblock]$Action
    )

    try {
        & $Action
        $script:passed++
        Write-Host "PASS $Name"
    }
    catch {
        $script:failed++
        Write-Host "FAIL $Name`: $($_.Exception.Message)"
    }
}

function Invoke-InstallerFixture {
    param(
        [Parameter(Mandatory = $true)] [pscustomobject]$Fixture,
        [switch]$DiscardOutput
    )

    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        if ($DiscardOutput.IsPresent) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installerPath -EvidenceRoot $Fixture.Evidence -DestinationRoot $Fixture.Destination -RepositoryRoot $Fixture.Repository 2>&1 | Out-Null
        }
        else {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installerPath -EvidenceRoot $Fixture.Evidence -DestinationRoot $Fixture.Destination -RepositoryRoot $Fixture.Repository 2>&1 | Out-Host
        }
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
}

Invoke-Case -Name 'Exact reviewed runtime replaces the package payload and remains non-release-approved' -Action {
    $fixture = New-Fixture
    try {
        $exitCode = Invoke-InstallerFixture -Fixture $fixture
        if ($exitCode -ne 0) { throw "Installer exited $exitCode." }
        $installed = Get-ChildItem -LiteralPath $fixture.Destination -Recurse -File -Filter 'OpenCvSharpExtern.dll'
        $actualHash = (Get-FileHash -LiteralPath $installed.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $fixture.ReviewedHash) { throw 'Installed runtime hash does not match the reviewed runtime.' }
        $metadata = Get-Content -LiteralPath (Join-Path $fixture.Destination 'reviewed-opencv-runtime.json') -Raw | ConvertFrom-Json
        if (-not [bool]$metadata.provenanceValidated -or [bool]$metadata.cleanMachineEvidence -or [bool]$metadata.releaseApproved) {
            throw 'Runtime metadata crossed the provenance-only approval boundary.'
        }
    }
    finally {
        if (Test-Path -LiteralPath $fixture.Root) { Remove-Item -LiteralPath $fixture.Root -Recurse -Force }
    }
}

Invoke-Case -Name 'Tampered source runtime is rejected before replacement' -Action {
    $fixture = New-Fixture -TamperSource
    try {
        $exitCode = Invoke-InstallerFixture -Fixture $fixture -DiscardOutput
        if ($exitCode -eq 0) { throw 'Tampered source runtime unexpectedly passed.' }
        $remaining = (Get-FileHash -LiteralPath (Join-Path $fixture.Destination 'runtimes\win-x64\native\OpenCvSharpExtern.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($remaining -eq $fixture.ReviewedHash) { throw 'Destination changed after rejected tampering.' }
    }
    finally {
        if (Test-Path -LiteralPath $fixture.Root) { Remove-Item -LiteralPath $fixture.Root -Recurse -Force }
    }
}

Invoke-Case -Name 'Ambiguous published native destinations are rejected' -Action {
    $fixture = New-Fixture
    try {
        New-Item -ItemType Directory -Path (Join-Path $fixture.Destination 'duplicate') -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $fixture.Destination 'duplicate\OpenCvSharpExtern.dll'), 'duplicate', [Text.UTF8Encoding]::new($false))
        $exitCode = Invoke-InstallerFixture -Fixture $fixture -DiscardOutput
        if ($exitCode -eq 0) { throw 'Ambiguous destination unexpectedly passed.' }
    }
    finally {
        if (Test-Path -LiteralPath $fixture.Root) { Remove-Item -LiteralPath $fixture.Root -Recurse -Force }
    }
}

Write-Host "Reviewed OpenCV runtime packaging tests: $passed passed, $failed failed."
if ($failed -ne 0) { exit 1 }
