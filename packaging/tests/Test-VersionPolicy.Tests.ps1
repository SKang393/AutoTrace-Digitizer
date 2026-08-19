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
    param([Parameter(Mandatory)][string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & git -C $testRoot @Arguments 1>$null 2>$null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "git failed: git -C $testRoot $($Arguments -join ' ')"
    }
}

try {
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

    foreach ($version in @('0.0.1', '0.0.21', '0.0.41', '0.0.61', '0.0.81', '0.1.1', '1.0.1')) {
        Assert-True (Test-GraphReaderReleaseVersion -Version $version) "Expected release eligibility for $version."
        $passed++
    }
    foreach ($version in @('0.0.0', '0.0.20', '0.0.99', '0.1.0', '0.1.2')) {
        Assert-True (-not (Test-GraphReaderReleaseVersion -Version $version)) "Unexpected release eligibility for $version."
        $passed++
    }

    foreach ($invalid in @('-1.0.0', '0.0.100', '0.01.1', '1.0', 'v1.0.1')) {
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

    New-Item -ItemType Directory -Path $testRoot | Out-Null
    $props = @"
<Project>
  <PropertyGroup>
    <Version>0.0.21</Version>
    <AssemblyVersion>0.0.21.0</AssemblyVersion>
    <FileVersion>0.0.21.0</FileVersion>
    <InformationalVersion>0.0.21</InformationalVersion>
  </PropertyGroup>
</Project>
"@
    [System.IO.File]::WriteAllText((Join-Path $testRoot 'Directory.Build.props'), $props)
    Invoke-Git -Arguments @('init', '-b', 'main')
    Invoke-Git -Arguments @('config', 'user.name', 'Version Policy Test')
    Invoke-Git -Arguments @('config', 'user.email', 'version-policy@example.invalid')
    Invoke-Git -Arguments @('config', 'core.autocrlf', 'false')
    Invoke-Git -Arguments @('add', 'Directory.Build.props')
    Invoke-Git -Arguments @('commit', '-m', 'Establish release checkpoint')
    Invoke-Git -Arguments @('tag', '-a', 'v0.0.21', '-m', 'Release 0.0.21')

    Invoke-Child -Script $releaseTagScript -Arguments @('-RepositoryRoot', $testRoot, '-TagName', 'v0.0.21') -ShouldPass $true
    $passed++
    Invoke-Git -Arguments @('tag', '-d', 'v0.0.21')
    Invoke-Git -Arguments @('tag', 'v0.0.21')
    Invoke-Child -Script $releaseTagScript -Arguments @('-RepositoryRoot', $testRoot, '-TagName', 'v0.0.21') -ShouldPass $false
    $passed++
    Invoke-Git -Arguments @('tag', '-d', 'v0.0.21')

    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $testRoot, '-PrepareNext') -ShouldPass $true
    $prepared = Get-GraphReaderCentralVersion -RepositoryRoot $testRoot
    Assert-Equal $prepared.Value '0.0.22' 'Prepared checkpoint version differs.'
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $testRoot, '-PrepareNext') -ShouldPass $true
    Assert-Equal (Get-GraphReaderCentralVersion -RepositoryRoot $testRoot).Value '0.0.22' 'Repeated preparation was not idempotent.'
    $passed += 2

    Invoke-Git -Arguments @('add', 'Directory.Build.props')
    Invoke-Git -Arguments @('commit', '-m', 'Advance checkpoint version')
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $testRoot, '-CheckHead') -ShouldPass $true
    $passed++

    Invoke-Git -Arguments @('tag', '-a', 'v0.0.22', '-m', 'Invalid release 0.0.22')
    Invoke-Child -Script $releaseTagScript -Arguments @('-RepositoryRoot', $testRoot, '-TagName', 'v0.0.22') -ShouldPass $false
    $passed++

    [System.IO.File]::WriteAllText((Join-Path $testRoot 'without-version-bump.txt'), 'invalid checkpoint')
    Invoke-Git -Arguments @('add', 'without-version-bump.txt')
    Invoke-Git -Arguments @('commit', '-m', 'Forget version advancement')
    Invoke-Child -Script $prepareScript -Arguments @('-RepositoryRoot', $testRoot, '-CheckHead') -ShouldPass $false
    $passed++

    Write-Host "Version policy tests passed: $passed"
}
finally {
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}
