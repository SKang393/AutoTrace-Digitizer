# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [switch]$BuildOnStart,
    [switch]$FastTestsOnly,
    [switch]$AllowDirty,
    [switch]$NoLaunch,
    [switch]$LaunchAfterBuild,
    [switch]$ApplyRetentionOnce,
    [switch]$RetentionSimulation,
    [string]$RetentionSimulationRoot,
    [string]$RetentionSimulationLedgerPath,
    [ValidateSet('PushFailure', 'PushSuccess')]
    [string]$RetentionSimulationScenario = 'PushFailure'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($NoLaunch.IsPresent -and $LaunchAfterBuild.IsPresent) {
    throw '-NoLaunch and -LaunchAfterBuild cannot be used together.'
}
if ($AllowDirty.IsPresent) {
    throw '-AllowDirty is retired: every produced build must be committed, recorded, and pushed.'
}

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$buildScript = Join-Path $PSScriptRoot 'Build-DevPortable.ps1'
$ledgerGeneratorScript = Join-Path $PSScriptRoot 'Generate-BuildLedger.ps1'
$launcherScript = Join-Path $PSScriptRoot 'Run-Latest-DevPortable.ps1'
$outputRoot = Join-Path $repositoryRoot 'artifacts\dev-portable'
$latestPath = Join-Path $outputRoot 'latest.json'
$debounce = [TimeSpan]::FromSeconds(2)
$pendingBuild = $BuildOnStart.IsPresent
$lastRelevantChange = if ($pendingBuild) { [DateTimeOffset]::UtcNow.Subtract($debounce) } else { [DateTimeOffset]::MinValue }
$observedHeadCommit = $null
$subscriptions = [System.Collections.Generic.List[object]]::new()
$watcher = $null
$repositoryPrefix = $repositoryRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar

function Resolve-DirectBuildDirectory {
    param(
        [Parameter(Mandatory)][string]$OutputRoot,
        [Parameter(Mandatory)][string]$RelativeBuildDirectory
    )

    if ([string]::IsNullOrWhiteSpace($RelativeBuildDirectory)) {
        throw 'Retention metadata does not identify a build directory.'
    }

    $buildsRoot = [System.IO.Path]::GetFullPath((Join-Path $OutputRoot 'builds'))
    $buildsPrefix = $buildsRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $OutputRoot $RelativeBuildDirectory.Replace('/', '\')))
    if (-not $candidate.StartsWith($buildsPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        [System.IO.Path]::GetDirectoryName($candidate).TrimEnd('\', '/') -ne $buildsRoot.TrimEnd('\', '/')) {
        throw "Retention target is not a direct child of artifacts/dev-portable/builds: $RelativeBuildDirectory"
    }

    return $candidate
}

function Get-DevPortableRetentionPlan {
    param(
        [Parameter(Mandatory)][string]$OutputRoot,
        [Parameter(Mandatory)][string]$LatestPath,
        [Parameter(Mandatory)][string]$LedgerPath
    )

    $outputRootFull = [System.IO.Path]::GetFullPath($OutputRoot)
    $buildsRoot = [System.IO.Path]::GetFullPath((Join-Path $outputRootFull 'builds'))
    if (-not (Test-Path -LiteralPath $buildsRoot -PathType Container)) {
        throw "Retention build root is missing: $buildsRoot"
    }
    if (-not (Test-Path -LiteralPath $LatestPath -PathType Leaf)) {
        throw "Retention latest metadata is missing: $LatestPath"
    }
    if (-not (Test-Path -LiteralPath $LedgerPath -PathType Leaf)) {
        throw "Retention ledger is missing: $LedgerPath"
    }

    $latest = Get-Content -LiteralPath $LatestPath -Raw | ConvertFrom-Json
    $latestDirectory = Resolve-DirectBuildDirectory `
        -OutputRoot $outputRootFull `
        -RelativeBuildDirectory ([string]$latest.buildDirectory)
    if (-not (Test-Path -LiteralPath $latestDirectory -PathType Container)) {
        throw "Retention latest build directory is missing: $latestDirectory"
    }
    $latestDirectoryInfo = Get-Item -LiteralPath $latestDirectory
    if (($latestDirectoryInfo.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Retention refuses reparse-point latest build directory: $latestDirectory"
    }

    $latestExecutable = [System.IO.Path]::GetFullPath((Join-Path $outputRootFull ([string]$latest.executable).Replace('/', '\')))
    $outputPrefix = $outputRootFull.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $latestExecutable.StartsWith($outputPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $latestExecutable -PathType Leaf)) {
        throw 'Retention latest executable is missing or outside the portable output root.'
    }
    $declaredExecutableHash = [string]$latest.executableSha256
    if ($declaredExecutableHash -notmatch '^[a-fA-F0-9]{64}$' -or
        (Get-FileHash -LiteralPath $latestExecutable -Algorithm SHA256).Hash.ToLowerInvariant() -ne $declaredExecutableHash.ToLowerInvariant()) {
        throw 'Retention latest executable checksum does not match latest.json.'
    }

    $ledger = Get-Content -LiteralPath $LedgerPath -Raw | ConvertFrom-Json
    $entries = @($ledger.builds)
    if ($entries.Count -eq 0) {
        throw 'Retention ledger contains no build entries.'
    }
    $incompleteEntries = @($entries | Where-Object {
            $_.PSObject.Properties.Name -contains 'recordIncomplete' -and
            $_.recordIncomplete -eq $true
        })
    if ($incompleteEntries.Count -gt 0) {
        throw 'Retention refuses a build ledger containing incomplete records.'
    }
    $retained = @($entries | Where-Object { $_.retained -eq $true })
    $latestName = [System.IO.Path]::GetFileName($latestDirectory)
    if ($retained.Count -ne 1 -or [string]$retained[0].directory -ne $latestName) {
        throw 'Retention ledger must contain exactly one retained entry matching latest.json.'
    }

    $staleDirectories = [System.Collections.Generic.List[string]]::new()
    foreach ($directory in @(Get-ChildItem -LiteralPath $buildsRoot -Directory -Force)) {
        if (($directory.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Retention refuses reparse-point build directory: $($directory.FullName)"
        }
        if ($directory.FullName -eq $latestDirectory) {
            continue
        }

        $entry = @($entries | Where-Object { [string]$_.directory -eq $directory.Name })
        if ($entry.Count -ne 1 -or
            [string]::IsNullOrWhiteSpace([string]$entry[0].commit) -or
            [string]::IsNullOrWhiteSpace([string]$entry[0].buildTimeUtc)) {
            throw "Retention target lacks a complete ledger entry: $($directory.Name)"
        }
        $staleDirectories.Add($directory.FullName)
    }

    return [pscustomobject]@{
        OutputRoot = $outputRootFull
        BuildsRoot = $buildsRoot
        LatestDirectory = $latestDirectory
        StaleDirectories = @($staleDirectories)
    }
}

function Get-TrackedStatusPaths {
    param([Parameter(Mandatory)][string]$RepositoryRoot)

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $lines = @(& git -C $RepositoryRoot status --porcelain --untracked-files=all 2>$null)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw 'Could not inspect repository status before retention commit.'
    }
    return @($lines | ForEach-Object {
        $line = ([string]$_)
        if ($line.Length -lt 4) { return $null }
        $line.Substring(3).Trim().Trim('"')
    } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Test-DevPortableLedgerPushed {
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)][string]$LedgerPath
    )

    $relativeLedger = $LedgerPath.Substring(
        ([System.IO.Path]::GetFullPath($RepositoryRoot)).Length).TrimStart('\', '/').Replace('\', '/')
    & git -C $RepositoryRoot ls-files --error-unmatch -- $relativeLedger 1>$null 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    & git -C $RepositoryRoot diff --quiet -- $relativeLedger 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    & git -C $RepositoryRoot diff --cached --quiet -- $relativeLedger 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }

    $localHead = (& git -C $RepositoryRoot rev-parse --verify HEAD 2>$null)
    if ($LASTEXITCODE -ne 0 -or $null -eq $localHead) { return $false }
    $headLedgerBlob = (& git -C $RepositoryRoot rev-parse "HEAD:$relativeLedger" 2>$null)
    if ($LASTEXITCODE -ne 0 -or $null -eq $headLedgerBlob) { return $false }
    $workingLedgerBlob = (& git -C $RepositoryRoot hash-object -- $LedgerPath 2>$null)
    if ($LASTEXITCODE -ne 0 -or $null -eq $workingLedgerBlob -or
        ([string]$workingLedgerBlob).Trim() -ne ([string]$headLedgerBlob).Trim()) { return $false }
    $remoteHead = (& git -C $RepositoryRoot ls-remote origin refs/heads/main 2>$null)
    if ($LASTEXITCODE -ne 0 -or $null -eq $remoteHead) { return $false }
    return ([string]$remoteHead).Split("`t")[0].Trim().ToLowerInvariant() -eq ([string]$localHead).Trim().ToLowerInvariant()
}

function Invoke-DevPortableLedgerCommitAndPush {
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)][string]$LedgerPath,
        [Parameter(Mandatory)][string]$CommitMessage
    )

    $allowedPaths = @('docs/BUILD_LEDGER.json')
    $statusPaths = @(Get-TrackedStatusPaths -RepositoryRoot $RepositoryRoot)
    $unexpected = @($statusPaths | Where-Object { $_.Replace('\', '/') -notin $allowedPaths })
    if ($unexpected.Count -gt 0) {
        throw "Retention refuses to commit unrelated dirty paths: $($unexpected -join ', ')"
    }

    $relativeLedger = $LedgerPath.Substring(([System.IO.Path]::GetFullPath($RepositoryRoot)).Length).TrimStart('\', '/')
    & git -C $RepositoryRoot add -- $relativeLedger
    if ($LASTEXITCODE -ne 0) { throw 'Could not stage docs/BUILD_LEDGER.json.' }
    $stagedNames = @(& git -C $RepositoryRoot diff --cached --name-only 2>$null)
    if ($LASTEXITCODE -ne 0 -or @($stagedNames | Where-Object { $_ -notin $allowedPaths }).Count -gt 0) {
        throw 'Retention staged a path outside the central version and build ledger allowlist.'
    }
    & git -C $RepositoryRoot commit -m $CommitMessage
    if ($LASTEXITCODE -ne 0) { throw 'Retention build outcome commit failed.' }
    $localHead = (& git -C $RepositoryRoot rev-parse --verify HEAD 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($localHead)) {
        throw 'Retention could not resolve the committed build outcome.'
    }
    & git -C $RepositoryRoot push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw 'Retention push to origin/main failed; build directories are retained.' }
    $remoteHead = (& git -C $RepositoryRoot ls-remote origin refs/heads/main 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]$remoteHead.Split("`t")[0].Trim().ToLowerInvariant() -ne $localHead.ToLowerInvariant()) {
        throw 'Retention could not confirm the pushed origin/main commit; build directories are retained.'
    }
    return $localHead
}

function Invoke-DevPortableRetention {
    param(
        [Parameter(Mandatory)]$Plan,
        [Parameter(Mandatory)][bool]$PushConfirmed,
        [switch]$Simulation
    )

    if (-not $PushConfirmed) {
        Write-Warning 'Portable retention skipped because the ledger push was not confirmed; previous and new builds remain.'
        return [pscustomobject]@{ Deleted = @(); Retained = @($Plan.LatestDirectory); PushConfirmed = $false }
    }

    $deleted = [System.Collections.Generic.List[string]]::new()
    foreach ($target in @($Plan.StaleDirectories)) {
        $resolvedTarget = [System.IO.Path]::GetFullPath($target)
        $buildsPrefix = $Plan.BuildsRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        if (-not $resolvedTarget.StartsWith($buildsPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            [System.IO.Path]::GetDirectoryName($resolvedTarget).TrimEnd('\', '/') -ne $Plan.BuildsRoot.TrimEnd('\', '/') -or
            $resolvedTarget -eq $Plan.LatestDirectory) {
            throw "Retention deletion target failed final path validation: $target"
        }
        if (-not $Simulation.IsPresent) {
            Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
        }
        $deleted.Add($resolvedTarget)
    }

    return [pscustomobject]@{ Deleted = @($deleted); Retained = @($Plan.LatestDirectory); PushConfirmed = $true }
}

if ($RetentionSimulation.IsPresent) {
    if ([string]::IsNullOrWhiteSpace($RetentionSimulationRoot)) {
        throw '-RetentionSimulationRoot is required with -RetentionSimulation.'
    }
    $simulationLedger = if ([string]::IsNullOrWhiteSpace($RetentionSimulationLedgerPath)) {
        Join-Path ([System.IO.Path]::GetFullPath($RetentionSimulationRoot)) 'BUILD_LEDGER.json'
    }
    else {
        [System.IO.Path]::GetFullPath($RetentionSimulationLedgerPath)
    }
    $simulationLatest = Join-Path ([System.IO.Path]::GetFullPath($RetentionSimulationRoot)) 'latest.json'
    $simulationPlan = Get-DevPortableRetentionPlan `
        -OutputRoot $RetentionSimulationRoot `
        -LatestPath $simulationLatest `
        -LedgerPath $simulationLedger
    $simulationPushConfirmed = $RetentionSimulationScenario -eq 'PushSuccess'
    $simulationResult = Invoke-DevPortableRetention -Plan $simulationPlan -PushConfirmed:$simulationPushConfirmed -Simulation
    $simulationSequence = if ($simulationPushConfirmed) {
        'build,verify,ledger,commit,push,delete'
    }
    else {
        'build,verify,ledger,commit,push(failed)'
    }
    Write-Host "Retention simulation sequence=$simulationSequence planned-deletions=$(@($simulationResult.Deleted).Count) remaining=$(@(Get-ChildItem -LiteralPath $simulationPlan.BuildsRoot -Directory).Count)."
    exit 0
}

if ($ApplyRetentionOnce.IsPresent) {
    $applyLedgerPath = Join-Path $repositoryRoot 'docs\BUILD_LEDGER.json'
    $applyPlan = Get-DevPortableRetentionPlan `
        -OutputRoot $outputRoot `
        -LatestPath $latestPath `
        -LedgerPath $applyLedgerPath
    if (-not (Test-DevPortableLedgerPushed -RepositoryRoot $repositoryRoot -LedgerPath $applyLedgerPath)) {
        throw 'Retention cleanup requires the current build ledger to be clean and confirmed on origin/main.'
    }
    $null = Invoke-DevPortableRetention -Plan $applyPlan -PushConfirmed:$true
    exit 0
}

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
    $durableBuildEligible = $false
    try {
        $preBuildStatusPaths = @(Get-TrackedStatusPaths -RepositoryRoot $repositoryRoot)
        $durableBuildEligible = $preBuildStatusPaths.Count -eq 0
        if (-not $durableBuildEligible) {
            Write-Warning 'Dirty or uncommitted source detected; this preview will not generate, commit, push, or delete retention records.'
        }
    }
    catch {
        Write-Warning "Could not establish a clean durable-build baseline: $($_.Exception.Message)"
    }

    $arguments = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', $buildScript
    )
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

    if (-not $durableBuildEligible) {
        return $true
    }

    $ledgerPath = Join-Path $repositoryRoot 'docs\BUILD_LEDGER.json'
    try {
        if (-not (Test-Path -LiteralPath $ledgerGeneratorScript -PathType Leaf)) {
            throw "Build ledger generator is missing: $ledgerGeneratorScript"
        }
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ledgerGeneratorScript `
            -BuildRoot (Join-Path $outputRoot 'builds') `
            -LatestPath $latestPath `
            -OutputPath $ledgerPath
        if ($LASTEXITCODE -ne 0) { throw 'Build ledger generation failed.' }

        $retentionPlan = Get-DevPortableRetentionPlan `
            -OutputRoot $outputRoot `
            -LatestPath $latestPath `
            -LedgerPath $ledgerPath
        $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
        $commitMessage = "Record portable build $([string]$latest.version)"
        $observedHeadCommit = Invoke-DevPortableLedgerCommitAndPush `
            -RepositoryRoot $repositoryRoot `
            -LedgerPath $ledgerPath `
            -CommitMessage $commitMessage
        $null = Invoke-DevPortableRetention -Plan $retentionPlan -PushConfirmed:$true
    }
    catch {
        Write-Warning "Portable retention deferred: $($_.Exception.Message)"
    }

    return $true
}

$watcher = [System.IO.FileSystemWatcher]::new($repositoryRoot)
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]::FileName -bor
    [System.IO.NotifyFilters]::DirectoryName -bor
    [System.IO.NotifyFilters]::LastWrite -bor
    [System.IO.NotifyFilters]::Size
$watcher.EnableRaisingEvents = $true

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
