# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$policyScript = Join-Path $repositoryRoot 'packaging\VersionPolicy.ps1'
$prepareScript = Join-Path $repositoryRoot 'packaging\Prepare-CheckpointVersion.ps1'
$releaseTagScript = Join-Path $repositoryRoot 'packaging\Test-ReleaseTag.ps1'
$hostExecutable = (Get-Process -Id $PID).Path
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('GraphReader-VersionPolicy-' + [Guid]::NewGuid().ToString('N'))
$ordinaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('GraphReader-VersionOrdinary-' + [Guid]::NewGuid().ToString('N'))
$promotionRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('GraphReader-VersionPromotion-' + [Guid]::NewGuid().ToString('N'))
$invalidPromotionRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('GraphReader-VersionInvalidPromotion-' + [Guid]::NewGuid().ToString('N'))
$rolloverRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('GraphReader-VersionRollover-' + [Guid]::NewGuid().ToString('N'))
$identicalRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('GraphReader-VersionIdentical-' + [Guid]::NewGuid().ToString('N'))
$releaseRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('GraphReader-VersionRelease-' + [Guid]::NewGuid().ToString('N'))
$historicalRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('GraphReader-VersionHistorical-' + [Guid]::NewGuid().ToString('N'))
$testRoots = @($testRoot, $ordinaryRoot, $promotionRoot, $invalidPromotionRoot, $rolloverRoot, $identicalRoot, $releaseRoot, $historicalRoot)
$historicalWorktreeCreated = $false
$passed = 0
. $policyScript

function Assert-Equal {
    param(
        [Parameter(Mandatory)][object]$Actual,
        [Parameter(Mandatory)][object]$Expected,
        [Parameter(Mandatory)][string]$Message
    )
    if ($Actual -ne $Expected) {
        throw "$Message Expected '$Expected', found '$Actual'."
    }
}

function Assert-True {
    param([Parameter(Mandatory)][bool]$Condition, [Parameter(Mandatory)][string]$Message)
    if (-not $Condition) {
        throw $Message
    }
}

function Invoke-Child {
    param(
        [Parameter(Mandatory)][string]$Script,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][bool]$ShouldPass
    )

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = & $hostExecutable -NoProfile -ExecutionPolicy Bypass -File $Script @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($ShouldPass -and $exitCode -ne 0) {
        throw "Expected success from $Script, exit code $exitCode. Output: $($output -join [Environment]::NewLine)"
    }
    if (-not $ShouldPass -and $exitCode -eq 0) {
        throw "Expected failure from $Script. Output: $($output -join [Environment]::NewLine)"
    }
}

function Invoke-Git {
    param(
        [Parameter(Mandatory)][string[]]$Arguments,
        [string]$Root = $testRoot
    )
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & git -C $Root @Arguments 1>$null 2>$null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "git failed: git -C $Root $($Arguments -join ' ')"
    }
}

function Write-TestProps {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Version
    )

    $props = @"
<Project>
  <PropertyGroup>
    <Version>$Version</Version>
    <AssemblyVersion>$Version.0</AssemblyVersion>
    <FileVersion>$Version.0</FileVersion>
    <InformationalVersion>$Version</InformationalVersion>
  </PropertyGroup>
</Project>
"@
    [System.IO.File]::WriteAllText((Join-Path $Root 'Directory.Build.props'), $props)
}

function Initialize-TestRepository {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Version
    )

    New-Item -ItemType Directory -Path $Root | Out-Null
    Write-TestProps -Root $Root -Version $Version
    Invoke-Git -Root $Root -Arguments @('init', '-b', 'main')
    Invoke-Git -Root $Root -Arguments @('config', 'user.name', 'Version Policy Test')
    Invoke-Git -Root $Root -Arguments @('config', 'user.email', 'version-policy@example.invalid')
    Invoke-Git -Root $Root -Arguments @('config', 'core.autocrlf', 'false')
    Invoke-Git -Root $Root -Arguments @('add', 'Directory.Build.props')
    Invoke-Git -Root $Root -Arguments @('commit', '-m', "Establish $Version checkpoint")
}

function Write-TestLedger {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$MaxVersion
    )

    $builds = @(
        1..431 | ForEach-Object {
            [ordered]@{ buildNumber = $_; version = '0.0.1' }
        }
    )
    $builds += [ordered]@{ buildNumber = 432; version = $MaxVersion }
    $ledgerPath = Join-Path $Root 'docs\BUILD_LEDGER.json'
    New-Item -ItemType Directory -Path (Split-Path -Parent $ledgerPath) -Force | Out-Null
    $ledger = [ordered]@{
        schemaVersion = 1
        policy = 'one produced build consumes one build number; ordinal equals build number'
        assignedUtc = '2026-08-19T00:00:00.0000000+00:00'
        builds = $builds
    }
    [System.IO.File]::WriteAllText(
        $ledgerPath,
        ($ledger | ConvertTo-Json -Depth 5),
        [System.Text.UTF8Encoding]::new($false))
}

function Assert-CanonicalBuildLedger {
    $ledgerPath = Join-Path $repositoryRoot 'docs\BUILD_LEDGER.json'
    Assert-True (Test-Path -LiteralPath $ledgerPath -PathType Leaf) 'Canonical build ledger is missing.'
    $ledger = Get-Content -LiteralPath $ledgerPath -Raw | ConvertFrom-Json
    Assert-Equal $ledger.schemaVersion 1 'Canonical build ledger schema differs.'
    $builds = @($ledger.builds)
    Assert-True ($builds.Count -ge 432) 'Canonical build ledger has fewer than 432 historical builds.'
    for ($index = 0; $index -lt $builds.Count; $index++) {
        $expectedNumber = $index + 1
        $expectedVersion = ConvertTo-GraphReaderVersion -Version (Get-NextGraphReaderVersion `
            -Version (ConvertTo-BuildVersionForTest -BuildNumber ($expectedNumber - 1)))
        $entry = $builds[$index]
        Assert-Equal ([int]$entry.buildNumber) $expectedNumber "Build ledger number differs at index $index."
        Assert-Equal ([string]$entry.version) $expectedVersion.Value "Build ledger version differs at build $expectedNumber."
        Assert-True (-not [string]::IsNullOrWhiteSpace([string]$entry.commit)) "Build $expectedNumber lacks a commit."
        Assert-True (-not [string]::IsNullOrWhiteSpace([string]$entry.buildTimeUtc)) "Build $expectedNumber lacks a build time."
        Assert-True ([string]$entry.executableSha256 -match '^[0-9a-f]{64}$') "Build $expectedNumber lacks an executable SHA-256."
        Assert-True ($entry.recordIncomplete -eq $false) "Build $expectedNumber is marked incomplete."
        Assert-Equal ([bool]$entry.releaseEligible) (($expectedNumber % 20) -eq 1) `
            "Release eligibility differs at build $expectedNumber."
    }
    Assert-Equal @($builds | Where-Object { $_.retained }).Count 1 'Canonical ledger retained count differs.'
    Assert-Equal @($builds | Select-Object -First 432 | Where-Object { $_.releaseStatus -eq 'missed-historical' }).Count 22 `
        'Historical missed-release count differs.'
    $central = Get-GraphReaderCentralVersion -RepositoryRoot $repositoryRoot
    $maximum = ConvertTo-GraphReaderVersion -Version ([string]$builds[-1].version)
    Assert-True (
        $central.Ordinal -eq $maximum.Ordinal -or
        $central.Ordinal -eq ($maximum.Ordinal + 1) -or
        $central.Value -ceq (Get-GraphReaderStablePromotionVersion)) `
        "Central version '$($central.Value)' is not the latest ledger version or its successor."
}

function ConvertTo-BuildVersionForTest {
    param([Parameter(Mandatory)][int]$BuildNumber)

    $major = [Math]::Floor($BuildNumber / 10000)
    $remainder = $BuildNumber % 10000
    return "$major.$([Math]::Floor($remainder / 100)).$($remainder % 100)"
}

try {
    Assert-CanonicalBuildLedger
    $passed++

    foreach ($case in @(
            @{ Current = '0.0.21'; Next = '0.0.22' },
            @{ Current = '0.0.99'; Next = '0.1.0' },
            @{ Current = '0.99.99'; Next = '1.0.0' },
            @{ Current = '1.99.99'; Next = '2.0.0' })) {
        Assert-Equal `
            -Actual (Get-NextGraphReaderVersion -Version $case.Current) `
            -Expected $case.Next `
            -Message "Successor differs for $($case.Current)."
        $passed++
    }

    foreach ($version in @('0.0.1', '0.0.21', '0.0.41', '0.0.61', '0.0.81', '0.1.1', '1.0.0', '1.0.1')) {
        Assert-True (Test-GraphReaderReleaseVersion -Version $version) "Expected release eligibility for $version."
        $passed++
    }
    foreach ($version in @('0.0.0', '0.0.20', '0.0.99', '0.1.0', '0.1.2')) {
        Assert-True (-not (Test-GraphReaderReleaseVersion -Version $version)) "Unexpected release eligibility for $version."
        $passed++
    }

    $stableRecord = ConvertTo-GraphReaderVersion -Version '1.0.0'
    Assert-True $stableRecord.StablePromotionRelease '1.0.0 was not classified as the stable-promotion release.'
    Assert-True (-not $stableRecord.CadenceEligible) '1.0.0 must not alter the twentieth-checkpoint cadence.'
    Assert-True (Test-GraphReaderStablePromotion -FromVersion '0.23.58' -ToVersion '1.0.0') 'Arbitrary pre-1.0 stable promotion was rejected.'
    foreach ($transition in @(
            @{ From = '0.23.58'; To = '1.0.1' },
            @{ From = '0.23.58'; To = '1.1.0' },
            @{ From = '1.0.0'; To = '1.0.0' })) {
        Assert-True `
            (-not (Test-GraphReaderStablePromotion -FromVersion $transition.From -ToVersion $transition.To)) `
            "Invalid stable promotion was accepted: $($transition.From) -> $($transition.To)."
        $passed++
    }
    $passed += 3

    foreach ($invalid in @('-1.0.0', '0.0.100', '0.01.1', '1.0', 'v1.0.0')) {
        $failed = $false
        try {
            $null = ConvertTo-GraphReaderVersion -Version $invalid
        }
        catch {
            $failed = $true
        }
        Assert-True $failed "Invalid version was accepted: $invalid"
        $passed++
    }

    Initialize-TestRepository -Root $testRoot -Version '0.4.32'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $testRoot, '-PrepareNext') -ShouldPass $true
    $prepared = Get-GraphReaderCentralVersion -RepositoryRoot $testRoot
    Assert-Equal $prepared.Value '0.4.33' 'Prepared checkpoint version differs.'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $testRoot, '-PrepareNext') -ShouldPass $true
    Assert-Equal (Get-GraphReaderCentralVersion -RepositoryRoot $testRoot).Value '0.4.33' 'Unrecorded build preparation was not idempotent.'
    Invoke-Git -Arguments @('add', 'Directory.Build.props')
    Invoke-Git -Arguments @('commit', '-m', 'Advance checkpoint version')
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $testRoot, '-CheckHead') -ShouldPass $true
    $passed++

    [System.IO.File]::WriteAllText((Join-Path $testRoot 'without-version-bump.txt'), 'invalid checkpoint')
    Invoke-Git -Arguments @('add', 'without-version-bump.txt')
    Invoke-Git -Arguments @('commit', '-m', 'Forget version advancement')
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $testRoot, '-CheckHead') -ShouldPass $false
    $passed++

    Initialize-TestRepository -Root $identicalRoot -Version '0.4.32'
    Write-TestLedger -Root $identicalRoot -MaxVersion '0.4.32'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $identicalRoot, '-PrepareNext') -ShouldPass $true
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $identicalRoot, '-PrepareNext') -ShouldPass $true
    Assert-Equal (Get-GraphReaderCentralVersion -RepositoryRoot $identicalRoot).Value '0.4.33' 'Unrecorded identical-commit preparation was not idempotent.'
    Write-TestLedger -Root $identicalRoot -MaxVersion '0.4.33'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $identicalRoot, '-PrepareNext') -ShouldPass $true
    Assert-Equal (Get-GraphReaderCentralVersion -RepositoryRoot $identicalRoot).Value '0.4.34' 'Identical-commit rebuild did not consume the next ordinal.'
    $passed += 2

    Invoke-Git -Root $identicalRoot -Arguments @('tag', '-a', 'v0.4.34', '-m', 'Invalid release 0.4.34')
    Invoke-Child -Script $releaseTagScript -Arguments @('-RepositoryRoot', $identicalRoot, '-TagName', 'v0.4.34') -ShouldPass $false
    $passed++

    Initialize-TestRepository -Root $rolloverRoot -Version '0.4.99'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $rolloverRoot, '-PrepareNext') -ShouldPass $true
    Assert-Equal (Get-GraphReaderCentralVersion -RepositoryRoot $rolloverRoot).Value '0.5.0' 'Checkpoint rollover differs.'
    $passed++

    Initialize-TestRepository -Root $releaseRoot -Version '0.0.21'
    Invoke-Git -Root $releaseRoot -Arguments @('tag', '-a', 'v0.0.21', '-m', 'Release 0.0.21')
    Invoke-Child -Script $releaseTagScript -Arguments @('-RepositoryRoot', $releaseRoot, '-TagName', 'v0.0.21') -ShouldPass $true
    Invoke-Git -Root $releaseRoot -Arguments @('tag', '-d', 'v0.0.21')
    Invoke-Git -Root $releaseRoot -Arguments @('tag', 'v0.0.21')
    Invoke-Child -Script $releaseTagScript -Arguments @('-RepositoryRoot', $releaseRoot, '-TagName', 'v0.0.21') -ShouldPass $false
    Invoke-Git -Root $releaseRoot -Arguments @('tag', '-d', 'v0.0.21')
    $passed += 2

    $historicalCorrectionCommit = '1a1f4aa87329ec0040bff68d03d0855281d5078f'
    Invoke-Git -Root $repositoryRoot -Arguments @('worktree', 'add', '--detach', $historicalRoot, $historicalCorrectionCommit)
    $historicalWorktreeCreated = $true
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $historicalRoot, '-CheckHead') -ShouldPass $true
    Assert-Equal (Get-GraphReaderCentralVersion -RepositoryRoot $historicalRoot).Value '0.4.32' 'Historical correction worktree version differs.'
    Invoke-Git -Root $historicalRoot -Arguments @('commit', '--allow-empty', '-m', 'Retain historical checkpoint version')
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $historicalRoot, '-CheckHead') -ShouldPass $false
    $passed += 2

    Initialize-TestRepository -Root $ordinaryRoot -Version '0.23.58'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $ordinaryRoot, '-PrepareNext') -ShouldPass $true
    Assert-Equal (Get-GraphReaderCentralVersion -RepositoryRoot $ordinaryRoot).Value '0.23.59' 'Ordinary pre-1.0 preparation jumped versions.'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $ordinaryRoot, '-PromoteStable') -ShouldPass $false
    $passed += 2

    Initialize-TestRepository -Root $promotionRoot -Version '0.23.58'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $promotionRoot, '-PromoteStable') -ShouldPass $true
    Assert-Equal (Get-GraphReaderCentralVersion -RepositoryRoot $promotionRoot).Value '1.0.0' 'Explicit stable promotion differs.'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $promotionRoot, '-PromoteStable') -ShouldPass $true
    Invoke-Git -Root $promotionRoot -Arguments @('add', 'Directory.Build.props')
    Invoke-Git -Root $promotionRoot -Arguments @('commit', '-m', 'Promote first stable release')
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $promotionRoot, '-CheckHead') -ShouldPass $true
    Invoke-Git -Root $promotionRoot -Arguments @('tag', '-a', 'v1.0.0', '-m', 'Release 1.0.0')
    Invoke-Child -Script $releaseTagScript -Arguments @('-RepositoryRoot', $promotionRoot, '-TagName', 'v1.0.0') -ShouldPass $true
    Invoke-Git -Root $promotionRoot -Arguments @('tag', '-d', 'v1.0.0')
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $promotionRoot, '-PromoteStable') -ShouldPass $false
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $promotionRoot, '-PrepareNext') -ShouldPass $true
    Assert-Equal (Get-GraphReaderCentralVersion -RepositoryRoot $promotionRoot).Value '1.0.1' 'Normal successor after 1.0.0 differs.'
    $passed += 7

    Initialize-TestRepository -Root $invalidPromotionRoot -Version '0.23.58'
    Write-TestProps -Root $invalidPromotionRoot -Version '1.0.1'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $invalidPromotionRoot, '-PromoteStable') -ShouldPass $false
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $invalidPromotionRoot, '-PrepareNext') -ShouldPass $false
    Invoke-Git -Root $invalidPromotionRoot -Arguments @('add', 'Directory.Build.props')
    Invoke-Git -Root $invalidPromotionRoot -Arguments @('commit', '-m', 'Create invalid stable jump')
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $invalidPromotionRoot, '-CheckHead') -ShouldPass $false
    $passed += 3

    Write-Host "Version policy tests passed: $passed"
}
finally {
    if ($historicalWorktreeCreated) {
        Invoke-Git -Root $repositoryRoot -Arguments @('worktree', 'remove', '--force', $historicalRoot)
    }
    foreach ($root in $testRoots) {
        if (Test-Path -LiteralPath $root) {
            Remove-Item -LiteralPath $root -Recurse -Force
        }
    }
}
