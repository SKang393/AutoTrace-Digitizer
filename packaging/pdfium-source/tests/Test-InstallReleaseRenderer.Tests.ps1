# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$profileRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$releaseInstallerPath = Join-Path $profileRoot 'Install-ReleaseRenderer.ps1'
$validatorPath = Join-Path $profileRoot 'Test-ReviewedPdfiumEvidence.ps1'
$passed = 0
$failed = 0

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$Content
    )

    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    [IO.File]::WriteAllText($Path, $Content, [Text.UTF8Encoding]::new($false))
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)] [string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function New-Fixture {
    $root = Join-Path ([IO.Path]::GetTempPath()) ('GraphReader-PdfiumReleaseInstall-' + [Guid]::NewGuid().ToString('N'))
    $repository = Join-Path $root 'repository'
    $profile = Join-Path $repository 'packaging\pdfium-source'
    $evidence = Join-Path $root 'evidence'
    $destination = Join-Path $root 'destination'
    New-Item -ItemType Directory -Path $profile, (Join-Path $repository 'packaging\common'), (Join-Path $repository 'packaging\evidence'), (Join-Path $evidence 'bin'), $destination -Force | Out-Null
    Copy-Item -LiteralPath $validatorPath -Destination (Join-Path $profile 'Test-ReviewedPdfiumEvidence.ps1')

    $binaryPath = Join-Path $evidence 'bin\graphreader_pdfium_renderer.exe'
    $sourceLockPath = Join-Path $evidence 'source-lock.json'
    $buildManifestPath = Join-Path $evidence 'build-manifest.json'
    $noticePath = Join-Path $evidence 'third-party-notices.reviewed.txt'
    [IO.File]::WriteAllBytes($binaryPath, [Text.Encoding]::UTF8.GetBytes('exact reviewed PDFium runner'))
    Write-Utf8NoBom -Path $sourceLockPath -Content '{"schemaVersion":1,"profileId":"graphreader-pdfium-minimal-win-x64"}'
    Write-Utf8NoBom -Path $buildManifestPath -Content '{"schemaVersion":1,"profileId":"graphreader-pdfium-minimal-win-x64"}'
    Write-Utf8NoBom -Path $noticePath -Content "REVIEW STATUS: COMPLETE`nControlled PDFium fixture notice.`n"

    $binaryHash = Get-Sha256 $binaryPath
    $approval = [ordered]@{
        schemaVersion = 1
        rendererId = 'graphreader-pdfium-renderer'
        rendererVersion = '2870fa9244b0f0f69fb743fab1e08deefcb07b2b'
        binaryPath = 'bin/graphreader_pdfium_renderer.exe'
        binarySha256 = $binaryHash
        source = 'https://pdfium.googlesource.com/pdfium'
        sourceRevision = '2870fa9244b0f0f69fb743fab1e08deefcb07b2b'
        sourceLockPath = 'source-lock.json'
        sourceLockSha256 = Get-Sha256 $sourceLockPath
        buildManifestPath = 'build-manifest.json'
        buildManifestSha256 = Get-Sha256 $buildManifestPath
        licenseSpdx = 'BSD-3-Clause'
        noticePath = 'third-party-notices.reviewed.txt'
        noticeSha256 = Get-Sha256 $noticePath
        reviewApproved = $true
        redistributionApproved = $true
        bundlingApproved = $true
    }
    Write-Utf8NoBom -Path (Join-Path $evidence 'reviewed-approval.json') -Content ($approval | ConvertTo-Json -Depth 10)

    $cleanMachineEvidencePath = Join-Path $repository 'packaging\evidence\pdfium-clean-machine.json'
    Write-Utf8NoBom -Path $cleanMachineEvidencePath -Content '{"status":"pass","machine":"fixture"}'
    $releaseAudit = [ordered]@{
        schemaVersion = 1
        mandatoryEvidenceGates = @([ordered]@{
                id = 'pdfium-clean-machine-load'
                description = 'Fixture clean-machine evidence.'
                status = 'pass'
                evidence = @([ordered]@{
                        path = 'packaging/evidence/pdfium-clean-machine.json'
                        sha256 = Get-Sha256 $cleanMachineEvidencePath
                    })
                notes = 'Fixture only.'
            })
        components = @([ordered]@{
                id = 'pdfium-native'
                checksumPolicy = 'exact-binary'
                artifactSha256 = $binaryHash
            })
    }
    Write-Utf8NoBom -Path (Join-Path $repository 'packaging\common\release-audit.json') -Content ($releaseAudit | ConvertTo-Json -Depth 10)

    return [pscustomobject]@{
        Root = $root
        Repository = $repository
        Evidence = $evidence
        Destination = $destination
        BinaryHash = $binaryHash
        CleanMachineEvidencePath = $cleanMachineEvidencePath
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

function Invoke-Installer {
    param([Parameter(Mandatory = $true)] [pscustomobject]$Fixture)
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $releaseInstallerPath `
            -EvidenceRoot $Fixture.Evidence `
            -DestinationRoot $Fixture.Destination `
            -RepositoryRoot $Fixture.Repository 2>&1 | Out-Null
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

function Remove-Fixture {
    param([Parameter(Mandatory = $true)] [pscustomobject]$Fixture)
    if (Test-Path -LiteralPath $Fixture.Root) {
        $resolved = [IO.Path]::GetFullPath($Fixture.Root)
        if ([IO.Path]::GetFileName($resolved).StartsWith('GraphReader-PdfiumReleaseInstall-', [StringComparison]::Ordinal)) {
            Remove-Item -LiteralPath $resolved -Recurse -Force
        }
    }
}

Invoke-Case -Name 'Direct reviewed and clean-machine evidence stages the exact portable PDFium payload' -Action {
    $fixture = New-Fixture
    try {
        $exitCode = Invoke-Installer $fixture
        if ($exitCode -ne 0) { throw "Installer exited $exitCode." }
        $target = Join-Path $fixture.Destination 'pdfium'
        $approval = Get-Content -LiteralPath (Join-Path $target 'reviewed-approval.json') -Raw | ConvertFrom-Json
        $runnerPath = Join-Path $target ([string]$approval.binaryPath)
        if ((Get-Sha256 $runnerPath) -ne $fixture.BinaryHash) { throw 'Staged runner hash changed.' }
        $actual = @(Get-ChildItem -LiteralPath $target -Recurse -File)
        if ($actual.Count -ne 5) { throw "Staged payload contains $($actual.Count) files instead of five." }
    }
    finally { Remove-Fixture $fixture }
}

Invoke-Case -Name 'Tampered clean-machine evidence blocks staging' -Action {
    $fixture = New-Fixture
    try {
        Write-Utf8NoBom -Path $fixture.CleanMachineEvidencePath -Content '{"status":"tampered"}'
        if ((Invoke-Installer $fixture) -eq 0) { throw 'Tampered evidence unexpectedly passed.' }
        if (Test-Path -LiteralPath (Join-Path $fixture.Destination 'pdfium')) { throw 'Rejected staging wrote a target.' }
    }
    finally { Remove-Fixture $fixture }
}

Invoke-Case -Name 'Blocked clean-machine gate blocks staging' -Action {
    $fixture = New-Fixture
    try {
        $auditPath = Join-Path $fixture.Repository 'packaging\common\release-audit.json'
        $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
        $audit.mandatoryEvidenceGates[0].status = 'blocked'
        Write-Utf8NoBom -Path $auditPath -Content ($audit | ConvertTo-Json -Depth 10)
        if ((Invoke-Installer $fixture) -eq 0) { throw 'Blocked gate unexpectedly passed.' }
        if (Test-Path -LiteralPath (Join-Path $fixture.Destination 'pdfium')) { throw 'Blocked staging wrote a target.' }
    }
    finally { Remove-Fixture $fixture }
}

Invoke-Case -Name 'Release-audit binary mismatch blocks staging' -Action {
    $fixture = New-Fixture
    try {
        $auditPath = Join-Path $fixture.Repository 'packaging\common\release-audit.json'
        $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
        $audit.components[0].artifactSha256 = ('0' * 64)
        Write-Utf8NoBom -Path $auditPath -Content ($audit | ConvertTo-Json -Depth 10)
        if ((Invoke-Installer $fixture) -eq 0) { throw 'Mismatched binary unexpectedly passed.' }
    }
    finally { Remove-Fixture $fixture }
}

Invoke-Case -Name 'Existing PDFium destination is never overwritten' -Action {
    $fixture = New-Fixture
    try {
        $target = Join-Path $fixture.Destination 'pdfium'
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Write-Utf8NoBom -Path (Join-Path $target 'owned.txt') -Content 'preserve'
        if ((Invoke-Installer $fixture) -eq 0) { throw 'Existing target unexpectedly passed.' }
        if ((Get-Content -LiteralPath (Join-Path $target 'owned.txt') -Raw) -ne 'preserve') { throw 'Existing target changed.' }
    }
    finally { Remove-Fixture $fixture }
}

Write-Host "Release PDFium renderer packaging tests: $passed passed, $failed failed."
if ($failed -ne 0) { exit 1 }
