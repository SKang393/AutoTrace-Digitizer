# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$reportScript = Join-Path $repositoryRoot 'packaging\Report-DevPortableSize.ps1'
$hostExecutable = (Get-Process -Id $PID).Path
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'GraphReader-DevPortableSizeTests-' + [Guid]::NewGuid().ToString('N'))
$passed = 0

function Assert-True {
    param(
        [Parameter(Mandatory)]
        [bool]$Condition,

        [Parameter(Mandatory)]
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function Write-Bytes {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [int]$Count
    )

    New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
    [System.IO.File]::WriteAllBytes($Path, [byte[]]::new($Count))
}

function New-SizeFixture {
    param([Parameter(Mandatory)][string]$Root)

    $devPortableRoot = Join-Path $Root 'artifacts\dev-portable'
    $buildsRoot = Join-Path $devPortableRoot 'builds'
    Write-Bytes -Path (Join-Path $buildsRoot 'build-a\app.bin') -Count 10
    Write-Bytes -Path (Join-Path $buildsRoot 'build-b\app.bin') -Count 20
    Write-Bytes -Path (Join-Path $buildsRoot 'build-c\app.bin') -Count 30
    Write-Bytes -Path (Join-Path $devPortableRoot 'Data\settings.bin') -Count 5
    Write-Bytes -Path (Join-Path $Root 'src\Example\bin\Release\app.bin') -Count 7
    Write-Bytes -Path (Join-Path $Root 'src\Example\obj\Release\app.obj') -Count 11
    Write-Bytes -Path (Join-Path $Root 'artifacts\goal19-opencv-source\sources\source.bin') -Count 13
    [System.IO.File]::WriteAllText(
        (Join-Path $devPortableRoot 'latest.json'),
        '{"buildDirectory":"builds/build-a"}',
        [System.Text.UTF8Encoding]::new($false))

    [System.IO.Directory]::SetLastWriteTimeUtc((Join-Path $buildsRoot 'build-a'), [datetime]'2026-01-01T00:00:00Z')
    [System.IO.Directory]::SetLastWriteTimeUtc((Join-Path $buildsRoot 'build-b'), [datetime]'2026-01-02T00:00:00Z')
    [System.IO.Directory]::SetLastWriteTimeUtc((Join-Path $buildsRoot 'build-c'), [datetime]'2026-01-03T00:00:00Z')
}

function Invoke-ExpectedFailure {
    param([Parameter(Mandatory)][string[]]$Arguments)

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $hostExecutable @Arguments 1>$null 2>$null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -eq 0) {
        throw "Expected child PowerShell to fail: $($Arguments -join ' ')"
    }
}

try {
    New-Item -ItemType Directory -Path $testRoot | Out-Null

    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $reportScript,
        [ref]$tokens,
        [ref]$errors) | Out-Null
    Assert-True ($errors.Count -eq 0) "PowerShell parser errors: $($errors -join ' | ')"
    $passed++

    $reportOnlyRoot = Join-Path $testRoot 'report-only'
    New-SizeFixture -Root $reportOnlyRoot
    $reportOnlyOutput = Join-Path $reportOnlyRoot 'artifacts\dev-portable\size-report.json'
    & $reportScript -RepositoryRoot $reportOnlyRoot -OutputPath $reportOnlyOutput
    $reportOnly = Get-Content -LiteralPath $reportOnlyOutput -Raw | ConvertFrom-Json
    Assert-True ($reportOnly.schemaVersion -eq 1) 'Size report schemaVersion was not 1.'
    Assert-True ($reportOnly.mode -eq 'report-only') 'Default size report mode was not report-only.'
    Assert-True ($reportOnly.summary.developmentPortableBuilds.count -eq 3) `
        'Report-only build count did not include all three fixture builds.'
    Assert-True ($reportOnly.summary.developmentPortableBuilds.size.bytes -eq 60) `
        'Development portable build bytes were not measured exactly.'
    Assert-True ($reportOnly.summary.developmentPortableData.bytes -eq 5) `
        'Development portable Data bytes were not measured exactly.'
    Assert-True ($reportOnly.summary.binAndObj.directoryCount -eq 2) `
        'bin/obj directory count was not measured exactly.'
    Assert-True ($reportOnly.summary.binAndObj.size.bytes -eq 18) `
        'bin/obj bytes were not measured exactly.'
    Assert-True ($reportOnly.summary.openCvAuditWorkspace.bytes -eq 13) `
        'OpenCV audit workspace bytes were not measured exactly.'
    Assert-True (@($reportOnly.largestDirectories).Count -gt 0) `
        'Largest-directory report was empty.'
    Assert-True ($reportOnly.cleanup.requested -eq $false) `
        'Report-only run incorrectly recorded cleanup as requested.'
    Assert-True (@($reportOnly.cleanup.prunedBuilds).Count -eq 0) `
        'Report-only run recorded pruned builds.'
    foreach ($name in @('build-a', 'build-b', 'build-c')) {
        Assert-True (Test-Path -LiteralPath (Join-Path $reportOnlyRoot "artifacts\dev-portable\builds\$name") -PathType Container) `
            "Report-only run deleted $name."
    }
    $passed++

    $pruneRoot = Join-Path $testRoot 'prune'
    New-SizeFixture -Root $pruneRoot
    $pruneOutput = Join-Path $pruneRoot 'artifacts\dev-portable\size-report.json'
    Invoke-ExpectedFailure @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', $reportScript,
        '-RepositoryRoot', $pruneRoot,
        '-OutputPath', $pruneOutput,
        '-PruneObsoleteBuilds',
        '-Confirm:$false')
    foreach ($name in @('build-a', 'build-b', 'build-c')) {
        Assert-True (Test-Path -LiteralPath (Join-Path $pruneRoot "artifacts\dev-portable\builds\$name") -PathType Container) `
            "Disabled direct pruning deleted $name."
    }
    $passed++

    $unsafeRoot = Join-Path $testRoot 'unsafe'
    New-SizeFixture -Root $unsafeRoot
    $unsafeOutputRoot = Join-Path $unsafeRoot 'artifacts\dev-portable'
    $outsideRoot = Join-Path $unsafeRoot 'artifacts\outside'
    Write-Bytes -Path (Join-Path $outsideRoot 'sentinel.bin') -Count 17
    [System.IO.File]::WriteAllText(
        (Join-Path $unsafeOutputRoot 'latest.json'),
        '{"buildDirectory":"../outside"}',
        [System.Text.UTF8Encoding]::new($false))
    Invoke-ExpectedFailure @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', $reportScript,
        '-RepositoryRoot', $unsafeRoot,
        '-OutputPath', (Join-Path $unsafeOutputRoot 'size-report.json'),
        '-PruneObsoleteBuilds',
        '-Confirm:$false')
    Assert-True (Test-Path -LiteralPath (Join-Path $outsideRoot 'sentinel.bin') -PathType Leaf) `
        'Path-safety rejection changed the outside sentinel.'
    foreach ($name in @('build-a', 'build-b', 'build-c')) {
        Assert-True (Test-Path -LiteralPath (Join-Path $unsafeRoot "artifacts\dev-portable\builds\$name") -PathType Container) `
            "Path-safety rejection deleted $name."
    }
    $passed++

    Write-Host "Development portable size tests passed: $passed/4"
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
