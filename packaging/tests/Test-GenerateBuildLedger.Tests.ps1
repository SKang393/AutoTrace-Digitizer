# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$scriptPath = Join-Path $repositoryRoot 'packaging\Generate-BuildLedger.ps1'
$hostExecutable = (Get-Process -Id $PID).Path
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'GraphReader-BuildLedger-' + [Guid]::NewGuid().ToString('N'))
$passed = 0

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
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][bool]$ShouldPass,
        [string]$ExpectedMessage
    )
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = & $hostExecutable -NoProfile -ExecutionPolicy Bypass -File $scriptPath @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($ShouldPass -and $exitCode -ne 0) {
        throw "Ledger generator failed unexpectedly: $($output -join [Environment]::NewLine)"
    }
    if (-not $ShouldPass -and $exitCode -eq 0) {
        throw 'Ledger generator unexpectedly accepted an invalid latest.json.'
    }
    if (-not $ShouldPass -and -not [string]::IsNullOrWhiteSpace($ExpectedMessage) -and
        (($output -join [Environment]::NewLine) -notlike "*$ExpectedMessage*")) {
        throw "Expected failure message '$ExpectedMessage' was not found. Output: $($output -join [Environment]::NewLine)"
    }
}

try {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $scriptPath,
        [ref]$tokens,
        [ref]$errors) | Out-Null
    Assert-True ($errors.Count -eq 0) "PowerShell parser errors: $($errors -join ' | ')"
    $passed++

    $realBuildRoot = Join-Path $repositoryRoot 'artifacts\dev-portable\builds'
    $realLatestPath = Join-Path $repositoryRoot 'artifacts\dev-portable\latest.json'
    if ((Test-Path -LiteralPath $realBuildRoot -PathType Container) -and
        (Test-Path -LiteralPath $realLatestPath -PathType Leaf)) {
        $realOutput = Join-Path $testRoot 'real-ledger.json'
        New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $repositoryRoot 'docs\BUILD_LEDGER.json') -Destination $realOutput
        Invoke-Child -Arguments @(
            '-BuildRoot', $realBuildRoot,
            '-LatestPath', $realLatestPath,
            '-OutputPath', $realOutput,
            '-AssignedUtc', '2026-08-19T15:00:00.0000000+00:00') -ShouldPass $true
        $ledger = Get-Content -LiteralPath $realOutput -Raw | ConvertFrom-Json
        Assert-Equal $ledger.schemaVersion 1 'Ledger schema version differs.'
        Assert-Equal $ledger.builds.Count 432 'Build count differs.'
        Assert-Equal $ledger.builds[0].buildNumber 1 'First build number differs.'
        Assert-Equal $ledger.builds[0].version '0.0.1' 'First assigned version differs.'
        Assert-Equal $ledger.builds[-1].buildNumber 432 'Last build number differs.'
        Assert-Equal $ledger.builds[-1].version '0.4.32' 'Last assigned version differs.'
        Assert-Equal @($ledger.builds | Where-Object { $_.retained }).Count 1 'Retained count differs.'
        Assert-Equal (@($ledger.builds | Where-Object { $_.retained })[0].directory) `
            '0.0.22-20260819T143609682Z-a44dfa1b' 'Retained directory differs.'
        Assert-Equal @($ledger.builds | Where-Object { $_.releaseEligible }).Count 22 'Historical release count differs.'
        Assert-Equal @($ledger.builds | Where-Object { $_.recordIncomplete }).Count 0 'Incomplete record count differs.'
        $retained = @($ledger.builds | Where-Object { $_.retained })[0]
        Assert-Equal $retained.stampedVersion '0.0.22' 'Stamped version was not preserved.'
        Assert-Equal $retained.commit 'a44dfa1b5de83c91e095b54d8fc099752d44d23e' 'Retained commit differs.'
        Assert-Equal $retained.executableSha256 '9fab0f8f940802e6b06960057995235468acbdd48ec79c46d60aa612ed6b9e30' `
            'Retained executable hash differs.'
        Assert-Equal $retained.openCvRuntimeSha256 '1fa122bdb8e94175e7719fb8aa8f2ab211268a756f5d0c7a13c710ed79ae30cd' `
            'Retained OpenCV hash differs.'
        $passed++
    }
    else {
        Write-Host 'Real portable build inventory is absent; running fixture-only generator tests.'
    }

    $fixtureBuildRoot = Join-Path $testRoot 'fixture\builds'
    $fixtureLatestPath = Join-Path $testRoot 'fixture\latest.json'
    $fixtureOutput = Join-Path $testRoot 'fixture-ledger.json'
    $fixtureExecutableHash = (Get-FileHash -InputStream ([IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes('fixture'))) -Algorithm SHA256).Hash.ToLowerInvariant()
    foreach ($name in @(
            '0.0.22-20260801T000000000Z-aaaaaaaa',
            '0.0.22-20260802T000000000Z-bbbbbbbb')) {
        $dir = Join-Path $fixtureBuildRoot $name
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        [System.IO.File]::WriteAllText((Join-Path $dir 'GraphReader.App.exe'), 'fixture')
        [System.IO.File]::WriteAllText((Join-Path $dir 'build-info.json'), (@{
                    schemaVersion = 2
                    version = '0.0.22'
                    commit = if ($name.EndsWith('aaaaaaaa')) { ('a' * 40) } else { ('b' * 40) }
                    shortCommit = if ($name.EndsWith('aaaaaaaa')) { 'aaaaaaaa' } else { 'bbbbbbbb' }
                    buildTimeUtc = if ($name.EndsWith('aaaaaaaa')) { '2026-08-01T00:00:00.0000000+00:00' } else { '2026-08-02T00:00:00.0000000+00:00' }
                    executableSha256 = $fixtureExecutableHash
                } | ConvertTo-Json -Depth 5))
    }
    [System.IO.File]::WriteAllText($fixtureLatestPath, (@{
                schemaVersion = 2
                version = '0.0.22'
                commit = ('b' * 40)
                buildTimeUtc = '2026-08-02T00:00:00.0000000+00:00'
                buildDirectory = 'builds/0.0.22-20260802T000000000Z-bbbbbbbb'
                executable = 'builds/0.0.22-20260802T000000000Z-bbbbbbbb/GraphReader.App.exe'
                executableSha256 = $fixtureExecutableHash
            } | ConvertTo-Json -Depth 5))
    Invoke-Child -Arguments @(
        '-BuildRoot', $fixtureBuildRoot,
        '-LatestPath', $fixtureLatestPath,
        '-OutputPath', $fixtureOutput,
        '-AssignedUtc', '2026-08-19T15:00:00.0000000+00:00') -ShouldPass $true
    $fixtureLedger = Get-Content -LiteralPath $fixtureOutput -Raw | ConvertFrom-Json
    Assert-Equal @($fixtureLedger.builds | Where-Object { $_.recordIncomplete }).Count 0 `
        'Missing optional OpenCV evidence incorrectly marked records incomplete.'
    Remove-Item -LiteralPath (Join-Path $fixtureBuildRoot '0.0.22-20260801T000000000Z-aaaaaaaa') -Recurse -Force
    $thirdName = '0.0.3-20260803T000000000Z-cccccccc'
    $thirdDir = Join-Path $fixtureBuildRoot $thirdName
    New-Item -ItemType Directory -Path $thirdDir -Force | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $thirdDir 'GraphReader.App.exe'), 'third')
    $thirdExecutableHash = (Get-FileHash -InputStream ([IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes('third'))) -Algorithm SHA256).Hash.ToLowerInvariant()
    [System.IO.File]::WriteAllText((Join-Path $thirdDir 'build-info.json'), (@{
                schemaVersion = 2
                version = '0.0.3'
                commit = ('c' * 40)
                shortCommit = 'cccccccc'
                buildTimeUtc = '2026-08-03T00:00:00.0000000+00:00'
                executableSha256 = $thirdExecutableHash
            } | ConvertTo-Json -Depth 5))
    [System.IO.File]::WriteAllText($fixtureLatestPath, (@{
                schemaVersion = 2
                version = '0.0.3'
                commit = ('c' * 40)
                buildTimeUtc = '2026-08-03T00:00:00.0000000+00:00'
                buildDirectory = "builds/$thirdName"
                executable = "builds/$thirdName/GraphReader.App.exe"
                executableSha256 = $thirdExecutableHash
            } | ConvertTo-Json -Depth 5))
    Invoke-Child -Arguments @(
        '-BuildRoot', $fixtureBuildRoot,
        '-LatestPath', $fixtureLatestPath,
        '-OutputPath', $fixtureOutput,
        '-AssignedUtc', '2026-08-19T15:00:00.0000000+00:00') -ShouldPass $true
    $incrementalLedger = Get-Content -LiteralPath $fixtureOutput -Raw | ConvertFrom-Json
    Assert-Equal $incrementalLedger.builds.Count 3 'Incremental ledger dropped a prior entry.'
    Assert-Equal $incrementalLedger.builds[0].buildNumber 1 'Incremental ledger renumbered the deleted entry.'
    Assert-Equal $incrementalLedger.builds[2].buildNumber 3 'Incremental ledger did not append build number 3.'
    Assert-Equal $incrementalLedger.builds[0].directory '0.0.22-20260801T000000000Z-aaaaaaaa' `
        'Incremental ledger did not preserve the deleted directory record.'
    Assert-Equal @($incrementalLedger.builds | Where-Object { $_.retained }).Count 1 `
        'Incremental ledger retained count differs.'
    Assert-Equal (@($incrementalLedger.builds | Where-Object { $_.retained })[0].directory) $thirdName `
        'Incremental ledger retained directory differs.'
    foreach ($field in @('version', 'commit', 'buildTimeUtc', 'executableSha256')) {
        $latestMismatch = Get-Content -LiteralPath $fixtureLatestPath -Raw | ConvertFrom-Json
        switch ($field) {
            'version' { $latestMismatch.version = '0.0.23' }
            'commit' { $latestMismatch.commit = ('d' * 40) }
            'buildTimeUtc' { $latestMismatch.buildTimeUtc = '2026-08-04T00:00:00.0000000+00:00' }
            'executableSha256' { $latestMismatch.executableSha256 = ('d' * 64) }
        }
        [System.IO.File]::WriteAllText($fixtureLatestPath, ($latestMismatch | ConvertTo-Json -Depth 5))
        Invoke-Child -Arguments @(
            '-BuildRoot', $fixtureBuildRoot,
            '-LatestPath', $fixtureLatestPath,
            '-OutputPath', (Join-Path $testRoot ("latest-mismatch-$field.json"))) -ShouldPass $false
    }
    [System.IO.File]::WriteAllText($fixtureLatestPath, (@{
                schemaVersion = 2
                version = '0.0.3'
                commit = ('c' * 40)
                buildTimeUtc = '2026-08-03T00:00:00.0000000+00:00'
                buildDirectory = "builds/$thirdName"
                executable = "builds/$thirdName/GraphReader.App.exe"
                executableSha256 = $thirdExecutableHash
            } | ConvertTo-Json -Depth 5))
    [System.IO.File]::WriteAllText($fixtureLatestPath, '{"buildDirectory":"builds/missing"}')
    Invoke-Child -Arguments @(
        '-BuildRoot', $fixtureBuildRoot,
        '-LatestPath', $fixtureLatestPath,
        '-OutputPath', (Join-Path $testRoot 'invalid.json')) -ShouldPass $false
    [System.IO.File]::WriteAllText(
        (Join-Path $fixtureBuildRoot "$thirdName\GraphReader.App.exe"),
        'tampered')
    [System.IO.File]::WriteAllText($fixtureLatestPath, (@{
                schemaVersion = 2
                version = '0.0.3'
                commit = ('c' * 40)
                buildTimeUtc = '2026-08-03T00:00:00.0000000+00:00'
                buildDirectory = "builds/$thirdName"
                executable = "builds/$thirdName/GraphReader.App.exe"
                executableSha256 = $thirdExecutableHash
            } | ConvertTo-Json -Depth 5))
    Invoke-Child -Arguments @(
        '-BuildRoot', $fixtureBuildRoot,
        '-LatestPath', $fixtureLatestPath,
        '-OutputPath', (Join-Path $testRoot 'tampered.json')) -ShouldPass $false
    [System.IO.File]::WriteAllText(
        (Join-Path $fixtureBuildRoot "$thirdName\GraphReader.App.exe"),
        'third')
    $reusedName = '0.0.3-20260804T000000000Z-dddddddd'
    $reusedDir = Join-Path $fixtureBuildRoot $reusedName
    New-Item -ItemType Directory -Path $reusedDir -Force | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $reusedDir 'GraphReader.App.exe'), 'reused')
    $reusedExecutableHash = (Get-FileHash -InputStream ([IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes('reused'))) -Algorithm SHA256).Hash.ToLowerInvariant()
    [System.IO.File]::WriteAllText((Join-Path $reusedDir 'build-info.json'), (@{
                schemaVersion = 2
                version = '0.0.3'
                commit = ('d' * 40)
                shortCommit = 'dddddddd'
                buildTimeUtc = '2026-08-04T00:00:00.0000000+00:00'
                executableSha256 = $reusedExecutableHash
            } | ConvertTo-Json -Depth 5))
    [System.IO.File]::WriteAllText($fixtureLatestPath, (@{
                schemaVersion = 2
                version = '0.0.3'
                commit = ('d' * 40)
                buildTimeUtc = '2026-08-04T00:00:00.0000000+00:00'
                buildDirectory = "builds/$reusedName"
                executable = "builds/$reusedName/GraphReader.App.exe"
                executableSha256 = $reusedExecutableHash
            } | ConvertTo-Json -Depth 5))
    Invoke-Child -Arguments @(
        '-BuildRoot', $fixtureBuildRoot,
        '-LatestPath', $fixtureLatestPath,
        '-OutputPath', $fixtureOutput) -ShouldPass $false -ExpectedMessage 'reuses a prior ordinal'
    $passed++

    Write-Host "Build ledger generator tests passed: $passed"
}
finally {
    $resolvedTestRoot = [System.IO.Path]::GetFullPath($testRoot)
    $temporaryPrefix = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\', '/') +
        [System.IO.Path]::DirectorySeparatorChar
    if ($resolvedTestRoot.StartsWith($temporaryPrefix, [StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $resolvedTestRoot -PathType Container)) {
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}
