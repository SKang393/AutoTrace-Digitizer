# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [switch]$BuildOnStart,
    [switch]$FastTestsOnly,
    [switch]$AllowDirty,
    [switch]$NoLaunch,
    [switch]$LaunchAfterBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($NoLaunch.IsPresent -and $LaunchAfterBuild.IsPresent) {
    throw '-NoLaunch and -LaunchAfterBuild cannot be used together.'
}

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$buildScript = Join-Path $PSScriptRoot 'Build-DevPortable.ps1'
$launcherScript = Join-Path $PSScriptRoot 'Run-Latest-DevPortable.ps1'
$outputRoot = Join-Path $repositoryRoot 'artifacts\dev-portable'
$latestPath = Join-Path $outputRoot 'latest.json'
$debounce = [TimeSpan]::FromSeconds(2)
$pendingBuild = $BuildOnStart.IsPresent
$lastRelevantChange = if ($pendingBuild) { [DateTimeOffset]::UtcNow.Subtract($debounce) } else { [DateTimeOffset]::MinValue }
$observedHeadCommit = $null
$subscriptions = [System.Collections.Generic.List[object]]::new()
$watcher = [System.IO.FileSystemWatcher]::new($repositoryRoot)
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]::FileName -bor
    [System.IO.NotifyFilters]::DirectoryName -bor
    [System.IO.NotifyFilters]::LastWrite -bor
    [System.IO.NotifyFilters]::Size
$watcher.EnableRaisingEvents = $true
$repositoryPrefix = $repositoryRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar

function Get-HeadCommit {
    $commit = & git -C $repositoryRoot rev-parse --verify HEAD 2>$null
    if ($LASTEXITCODE -ne 0 -or $null -eq $commit) {
        return $null
    }

    $value = ([string]$commit).Trim()
    if ($value -notmatch '^[0-9a-fA-F]{40}$') {
        return $null
    }

    return $value.ToLowerInvariant()
}

function Test-RelevantChange {
    param([Parameter(Mandatory)][string]$FullPath)

    $fullPathValue = [System.IO.Path]::GetFullPath($FullPath)
    if (-not $fullPathValue.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }
    $relative = $fullPathValue.Substring($repositoryPrefix.Length).Replace('\', '/')
    if (
        $relative -match '(^|/)(\.git|\.codex|artifacts|bin|obj|TestResults|\.cache|cache|Data)(/|$)' -or
        $relative -match '\.(tmp|log|pdb|trx)$') {
        return $false
    }

    if ($relative -match '^(src|contracts|packaging)/') {
        return $relative -match '\.(cs|xaml|csproj|props|targets|json|ps1|cmd|md|resx)$'
    }

    return $relative -in @(
        'Directory.Build.props',
        'Directory.Packages.props',
        'global.json',
        'GraphAutoReader.slnx')
}

function Drain-ChangeEvents {
    $foundRelevantChange = $false
    foreach ($event in @(Get-Event | Where-Object { $_.SourceIdentifier -like 'GraphReader.DevPortable.*' })) {
        try {
            $path = [string]$event.SourceEventArgs.FullPath
            if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-RelevantChange -FullPath $path)) {
                $foundRelevantChange = $true
            }
        }
        finally {
            Remove-Event -EventIdentifier $event.EventIdentifier -ErrorAction SilentlyContinue
        }
    }

    return $foundRelevantChange
}

function Invoke-PreviewBuild {
    $arguments = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', $buildScript
    )
    if ($AllowDirty.IsPresent) {
        $arguments += '-AllowDirty'
    }
    if ($FastTestsOnly.IsPresent) {
        $arguments += '-FastTestsOnly'
    }

    Write-Host "Building development portable at $([DateTimeOffset]::Now.ToString('T'))..."
    & powershell.exe @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Warning "Development portable build failed with exit code $exitCode. The prior latest.json remains active."
        return $false
    }

    $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
    $executable = [System.IO.Path]::GetFullPath((Join-Path $outputRoot ([string]$latest.executable)))
    Write-Host "Latest executable: $executable"

    if ($LaunchAfterBuild.IsPresent -and -not $NoLaunch.IsPresent) {
        & $launcherScript
    }

    return $true
}

try {
    $observedHeadCommit = Get-HeadCommit
    $subscriptions.Add((Register-ObjectEvent -InputObject $watcher -EventName Changed -SourceIdentifier 'GraphReader.DevPortable.Changed'))
    $subscriptions.Add((Register-ObjectEvent -InputObject $watcher -EventName Created -SourceIdentifier 'GraphReader.DevPortable.Created'))
    $subscriptions.Add((Register-ObjectEvent -InputObject $watcher -EventName Deleted -SourceIdentifier 'GraphReader.DevPortable.Deleted'))
    $subscriptions.Add((Register-ObjectEvent -InputObject $watcher -EventName Renamed -SourceIdentifier 'GraphReader.DevPortable.Renamed'))

    Write-Host 'Watching src, contracts, packaging, and central build files. Press Ctrl+C to stop.'
    while ($true) {
        Wait-Event -Timeout 1 | Out-Null
        if (Drain-ChangeEvents) {
            $pendingBuild = $true
            $lastRelevantChange = [DateTimeOffset]::UtcNow
        }

        # Linked worktrees keep HEAD and branch refs outside the watched root.
        # Comparing only the resolved commit ignores index, object, and log churn
        # while still detecting commits, checkouts, resets, and ref updates.
        $currentHeadCommit = Get-HeadCommit
        if ($null -ne $currentHeadCommit -and $currentHeadCommit -ne $observedHeadCommit) {
            $observedHeadCommit = $currentHeadCommit
            $pendingBuild = $true
            $lastRelevantChange = [DateTimeOffset]::UtcNow
            Write-Host "Git HEAD changed to $currentHeadCommit; queuing an exact-commit preview."
        }

        if (-not $pendingBuild -or ([DateTimeOffset]::UtcNow - $lastRelevantChange) -lt $debounce) {
            continue
        }

        $pendingBuild = $false
        $null = Invoke-PreviewBuild

        # Events raised while the synchronous build ran collapse into one queued build.
        if (Drain-ChangeEvents) {
            $pendingBuild = $true
            $lastRelevantChange = [DateTimeOffset]::UtcNow
        }
    }
}
finally {
    $watcher.EnableRaisingEvents = $false
    foreach ($sourceIdentifier in @(
        'GraphReader.DevPortable.Changed',
        'GraphReader.DevPortable.Created',
        'GraphReader.DevPortable.Deleted',
        'GraphReader.DevPortable.Renamed')) {
        Unregister-Event -SourceIdentifier $sourceIdentifier -ErrorAction SilentlyContinue
    }
    Get-Event | Where-Object { $_.SourceIdentifier -like 'GraphReader.DevPortable.*' } |
        Remove-Event -ErrorAction SilentlyContinue
    $watcher.Dispose()
    Write-Host 'Development portable watcher stopped.'
}
