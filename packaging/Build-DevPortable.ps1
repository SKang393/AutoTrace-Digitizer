# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [switch]$AllowDirty,
    [switch]$FastTestsOnly,
    [switch]$SkipRestore,
    [string]$OutputRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'DevPortable.Common.ps1')

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repositoryRoot 'artifacts\dev-portable'
}
else {
    $OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
}

$buildsRoot = Join-Path $OutputRoot 'builds'
$stagingRoot = Join-Path $OutputRoot '.staging'
$latestPath = Join-Path $OutputRoot 'latest.json'
$failurePath = Join-Path $OutputRoot 'last-failure.json'
$sharedDataRoot = Join-Path $OutputRoot 'Data'
$currentCommand = 'initialize'
$diagnostics = [System.Collections.Generic.List[string]]::new()
$testsRun = [System.Collections.Generic.List[string]]::new()
$stagingDirectory = $null
$successfulBuildDirectory = $null

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory)]
        [string]$Description,

        [Parameter(Mandatory)]
        [scriptblock]$Command
    )

    $script:currentCommand = $Description
    $script:diagnostics.Clear()
    Write-Host "[$Description]"
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Command 2>&1 | ForEach-Object {
            $line = [string]$_
            if ($script:diagnostics.Count -ge 400) {
                $script:diagnostics.RemoveAt(0)
            }
            $script:diagnostics.Add($line)
            Write-Host $line
        }
        $commandExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    if ($commandExitCode -ne 0) {
        throw "Command failed with exit code ${commandExitCode}: $Description"
    }
}

function Get-CentralVersion {
    [xml]$props = Get-Content -LiteralPath (Join-Path $repositoryRoot 'Directory.Build.props') -Raw
    $version = [string]$props.Project.PropertyGroup.Version
    if ($version -notmatch '^\d{1,2}\.\d{1,2}\.\d{1,2}$') {
        throw "Central version is invalid: '$version'."
    }

    return $version
}

function Get-GitOutput {
    param([Parameter(Mandatory)][string[]]$Arguments)

    $output = & git -C $repositoryRoot @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($Arguments -join ' ')`n$($output -join [Environment]::NewLine)"
    }

    return ($output -join [Environment]::NewLine).Trim()
}

try {
    $currentCommand = 'read repository state'
    $version = Get-CentralVersion
    $commit = Get-GitOutput -Arguments @('rev-parse', 'HEAD')
    $shortCommit = Get-GitOutput -Arguments @('rev-parse', '--short=8', 'HEAD')
    $status = Get-GitOutput -Arguments @('status', '--porcelain', '--untracked-files=normal')
    $isDirty = -not [string]::IsNullOrWhiteSpace($status)
    if ($isDirty -and -not $AllowDirty.IsPresent) {
        throw 'The working tree is dirty. Commit the checkpoint or rerun with -AllowDirty for a local-only preview.'
    }

    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $buildsRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $stagingRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $sharedDataRoot -Force | Out-Null

    if (-not $SkipRestore.IsPresent) {
        Invoke-CheckedCommand -Description 'dotnet restore GraphAutoReader.slnx' -Command {
            & dotnet restore (Join-Path $repositoryRoot 'GraphAutoReader.slnx')
        }
    }

    $fastTestCommands = @(
        [pscustomobject]@{
            Name = 'GraphReader.App.Tests'
            Project = 'tests\GraphReader.App.Tests\GraphReader.App.Tests.csproj'
            Filter = $null
        },
        [pscustomobject]@{
            Name = 'GraphReader.Domain.Tests'
            Project = 'tests\GraphReader.Domain.Tests\GraphReader.Domain.Tests.csproj'
            Filter = $null
        },
        [pscustomobject]@{
            Name = 'Goal 19 integration and packaging smoke'
            Project = 'tests\GraphReader.Integration.Tests\GraphReader.Integration.Tests.csproj'
            Filter = 'FullyQualifiedName~IntegrationSmoke|FullyQualifiedName~PackagingContractTests|FullyQualifiedName~GitIgnoreContractTests'
        }
    )

    foreach ($test in $fastTestCommands) {
        $testsRun.Add($test.Name)
        $projectPath = Join-Path $repositoryRoot $test.Project
        if ([string]::IsNullOrWhiteSpace([string]$test.Filter)) {
            Invoke-CheckedCommand -Description "dotnet test $($test.Name)" -Command {
                & dotnet test $projectPath -c Release --no-restore
            }
        }
        else {
            $filter = [string]$test.Filter
            Invoke-CheckedCommand -Description "dotnet test $($test.Name)" -Command {
                & dotnet test $projectPath -c Release --no-restore --filter $filter
            }
        }
    }

    if (-not $FastTestsOnly.IsPresent) {
        $testsRun.Add('Full Release solution')
        Invoke-CheckedCommand -Description 'dotnet test GraphAutoReader.slnx -c Release' -Command {
            & dotnet test (Join-Path $repositoryRoot 'GraphAutoReader.slnx') -c Release --no-restore
        }
        $testsRun.Add('Public synthetic scoreboard')
        Invoke-CheckedCommand -Description 'public synthetic scoreboard' -Command {
            & dotnet run -c Release --no-restore --project (Join-Path $repositoryRoot 'tools\GraphReader.Benchmarks') -- --suite public
        }
        $testsRun.Add('Packaging regression')
        Invoke-CheckedCommand -Description 'packaging regression' -Command {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repositoryRoot 'packaging\tests\Test-ReleaseArtifact.Tests.ps1')
        }
        $testsRun.Add('Development portable packaging regression')
        Invoke-CheckedCommand -Description 'development portable packaging regression' -Command {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repositoryRoot 'packaging\tests\Test-DevPortable.Tests.ps1')
        }
        $testsRun.Add('Localization audit regression')
        Invoke-CheckedCommand -Description 'localization audit regression' -Command {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repositoryRoot 'packaging\localization\Test-LocalizationAudit.ps1')
        }
    }

    $buildTimestamp = [DateTimeOffset]::UtcNow
    $buildName = '{0}-{1}-{2}' -f $version, $buildTimestamp.ToString('yyyyMMddTHHmmssfffZ'), $shortCommit
    $successfulBuildDirectory = Join-Path $buildsRoot $buildName
    if (Test-Path -LiteralPath $successfulBuildDirectory) {
        throw "The immutable build folder already exists: $successfulBuildDirectory"
    }

    $stagingDirectory = Join-Path $stagingRoot ([Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $stagingDirectory | Out-Null

    Invoke-CheckedCommand -Description 'dotnet publish GraphReader.App' -Command {
        & dotnet publish (Join-Path $repositoryRoot 'src\GraphReader.App\GraphReader.App.csproj') `
            -c Release `
            -r win-x64 `
            --self-contained true `
            --no-restore `
            -p:PublishSingleFile=false `
            -p:DebugSymbols=false `
            -p:DebugType=None `
            -o $stagingDirectory
    }

    $executablePath = Join-Path $stagingDirectory 'GraphReader.App.exe'
    $runtimeConfigPath = Join-Path $stagingDirectory 'GraphReader.App.runtimeconfig.json'
    if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
        throw "Published executable is missing: $executablePath"
    }
    if (-not (Test-Path -LiteralPath $runtimeConfigPath -PathType Leaf)) {
        throw "Published runtime configuration is missing: $runtimeConfigPath"
    }

    Write-Utf8NoBom -Path (Join-Path $stagingDirectory 'portable.mode') -Content "development portable`r`n"
    Write-Utf8NoBom -Path (Join-Path $stagingDirectory 'DEVELOPMENT_BUILD.txt') -Content @"
Graph Auto Reader Development Preview
Local maintainer testing only. Do not redistribute or publish.
Version: $version
Commit: $commit
Dirty: $($isDirty.ToString().ToLowerInvariant())
Built UTC: $($buildTimestamp.ToString('O'))
Runtime mode: ManualPreview
"@

    $availableModelIds = @(Get-DevPortableApprovedModelIds `
            -RepositoryRoot $repositoryRoot `
            -Diagnostics $diagnostics)
    $unavailableStages = @('enhancement', 'axis', 'ocr', 'markers', 'legends', 'phases')
    $buildInfo = [ordered]@{
        schemaVersion = 1
        version = $version
        commit = $commit
        shortCommit = $shortCommit
        dirty = $isDirty
        buildTimeUtc = $buildTimestamp.ToString('O')
        runtimeMode = 'ManualPreview'
        testsRun = @($testsRun)
        availableModelIds = $availableModelIds
        unavailableStages = $unavailableStages
        localOnlyWarning = 'Development Preview. Local maintainer testing only. Do not redistribute or publish.'
    }
    Write-JsonFile -Path (Join-Path $stagingDirectory 'build-info.json') -Value $buildInfo

    $currentCommand = 'portable smoke'
    $previousDataRoot = [Environment]::GetEnvironmentVariable('GRAPHREADER_DEV_PORTABLE_DATA_ROOT', 'Process')
    try {
        [Environment]::SetEnvironmentVariable(
            'GRAPHREADER_DEV_PORTABLE_DATA_ROOT',
            [System.IO.Path]::GetFullPath($sharedDataRoot),
            'Process')
        $smokeProcess = Start-Process `
            -FilePath $executablePath `
            -ArgumentList '--portable-smoke' `
            -WorkingDirectory $stagingDirectory `
            -WindowStyle Hidden `
            -PassThru `
            -Wait
        if ($smokeProcess.ExitCode -ne 0) {
            throw "Portable smoke exited with code $($smokeProcess.ExitCode)."
        }
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            'GRAPHREADER_DEV_PORTABLE_DATA_ROOT',
            $previousDataRoot,
            'Process')
    }

    Move-Item -LiteralPath $stagingDirectory -Destination $successfulBuildDirectory
    $stagingDirectory = $null

    $currentCommand = 'finalize development portable metadata'
    $outputPrefix = [System.IO.Path]::GetFullPath($OutputRoot).TrimEnd('\', '/') +
        [System.IO.Path]::DirectorySeparatorChar
    $fullBuildDirectory = [System.IO.Path]::GetFullPath($successfulBuildDirectory)
    if (-not $fullBuildDirectory.StartsWith($outputPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The successful build directory is outside the development portable output root.'
    }
    $relativeBuildDirectory = $fullBuildDirectory.Substring($outputPrefix.Length)
    $latest = [ordered]@{
        schemaVersion = 1
        version = $version
        commit = $commit
        shortCommit = $shortCommit
        dirty = $isDirty
        buildTimeUtc = $buildTimestamp.ToString('O')
        buildDirectory = $relativeBuildDirectory.Replace('\', '/')
        executable = ($relativeBuildDirectory.Replace('\', '/') + '/GraphReader.App.exe')
        dataRoot = 'Data'
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Run-Latest-DevPortable.ps1') -Destination $OutputRoot -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Run-Latest-DevPortable.cmd') -Destination $OutputRoot -Force
    Write-JsonFileAtomic -Path $latestPath -Value $latest

    if (Test-Path -LiteralPath $failurePath -PathType Leaf) {
        Remove-Item -LiteralPath $failurePath -Force
    }

    Write-Host "Development portable ready: $successfulBuildDirectory"
    Write-Host "Executable: $(Join-Path $successfulBuildDirectory 'GraphReader.App.exe')"
    Write-Host "Shared data: $sharedDataRoot"
}
catch {
    if (Test-Path -LiteralPath $OutputRoot -PathType Container) {
        $failure = [ordered]@{
            schemaVersion = 1
            failedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
            command = $currentCommand
            error = $_.Exception.Message
            diagnostics = @($diagnostics)
            priorLatestPreserved = Test-Path -LiteralPath $latestPath -PathType Leaf
        }
        Write-JsonFileAtomic -Path $failurePath -Value $failure
    }

    Write-Error $_
    exit 1
}
finally {
    if ($null -ne $stagingDirectory -and (Test-Path -LiteralPath $stagingDirectory -PathType Container)) {
        Remove-Item -LiteralPath $stagingDirectory -Recurse -Force
    }
}
