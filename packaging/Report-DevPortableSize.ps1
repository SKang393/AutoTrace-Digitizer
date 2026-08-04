# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$RepositoryRoot,
    [string]$OutputPath,
    [switch]$PruneObsoleteBuilds,
    [ValidateRange(0, 20)]
    [int]$KeepPreviousBuilds = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Test-PathWithinRoot {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$Root
    )

    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $prefix = $resolvedRoot + [System.IO.Path]::DirectorySeparatorChar
    return $resolvedPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory)]
        [string]$Root,

        [Parameter(Mandatory)]
        [string]$Path
    )

    $rootPrefix = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/') +
        [System.IO.Path]::DirectorySeparatorChar
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $resolvedPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the repository root: $resolvedPath"
    }

    return $resolvedPath.Substring($rootPrefix.Length).Replace('\', '/')
}

function Get-DirectoryByteCount {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return [long]0
    }

    [long]$total = 0
    Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction Stop | ForEach-Object {
        $total += [long]$_.Length
    }
    return $total
}

function Get-DirectorySizeIndex {
    param(
        [Parameter(Mandatory)]
        [string]$Root,

        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [System.Collections.Generic.List[string]]$Warnings
    )

    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $rootPrefix = $resolvedRoot + [System.IO.Path]::DirectorySeparatorChar
    $sizes = [System.Collections.Generic.Dictionary[string, long]]::new(
        [StringComparer]::OrdinalIgnoreCase)
    $pending = [System.Collections.Generic.Stack[string]]::new()
    $pending.Push($resolvedRoot)

    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        if (-not $sizes.ContainsKey($directory)) {
            $sizes[$directory] = [long]0
        }

        try {
            foreach ($file in Get-ChildItem -LiteralPath $directory -File -Force -ErrorAction Stop) {
                [long]$length = $file.Length
                $current = $directory
                while ($true) {
                    $sizes[$current] = [long]$sizes[$current] + $length
                    if ($current.Equals($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
                        break
                    }

                    $parent = [System.IO.Directory]::GetParent($current)
                    if ($null -eq $parent) {
                        throw "Could not resolve a parent directory while measuring $current."
                    }
                    $current = $parent.FullName.TrimEnd('\', '/')
                    if (-not ($current.Equals($resolvedRoot, [StringComparison]::OrdinalIgnoreCase) -or
                            $current.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase))) {
                        throw "Directory traversal escaped the repository root while measuring $directory."
                    }
                    if (-not $sizes.ContainsKey($current)) {
                        $sizes[$current] = [long]0
                    }
                }
            }

            foreach ($child in Get-ChildItem -LiteralPath $directory -Directory -Force -ErrorAction Stop) {
                $childPath = [System.IO.Path]::GetFullPath($child.FullName).TrimEnd('\', '/')
                if (-not ($childPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase))) {
                    throw "Directory traversal escaped the repository root: $childPath"
                }
                if (($child.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                    $Warnings.Add("Skipped reparse-point directory: $(Get-RelativePath -Root $resolvedRoot -Path $childPath)")
                    continue
                }
                if (-not $sizes.ContainsKey($childPath)) {
                    $sizes[$childPath] = [long]0
                }
                $pending.Push($childPath)
            }
        }
        catch {
            $Warnings.Add("Could not fully measure '$directory': $($_.Exception.Message)")
        }
    }

    return $sizes
}

function ConvertTo-SizeRecord {
    param(
        [Parameter(Mandatory)]
        [long]$Bytes
    )

    return [ordered]@{
        bytes = $Bytes
        mebibytes = [Math]::Round($Bytes / 1MB, 2)
        gibibytes = [Math]::Round($Bytes / 1GB, 3)
    }
}

function Assert-ImmediateBuildDirectory {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$BuildsRoot
    )

    $resolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $resolvedBuildsRoot = [System.IO.Path]::GetFullPath($BuildsRoot).TrimEnd('\', '/')
    if (-not (Test-PathWithinRoot -Path $resolvedPath -Root $resolvedBuildsRoot)) {
        throw "Development portable cleanup target is outside artifacts/dev-portable/builds: $resolvedPath"
    }

    $parent = [System.IO.Directory]::GetParent($resolvedPath)
    if ($null -eq $parent -or
        -not $parent.FullName.TrimEnd('\', '/').Equals(
            $resolvedBuildsRoot,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Development portable cleanup target is not an immediate build directory: $resolvedPath"
    }

    if (Test-Path -LiteralPath $resolvedPath -PathType Container) {
        $item = Get-Item -LiteralPath $resolvedPath -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Development portable cleanup target cannot be a reparse point: $resolvedPath"
        }
    }

    return $resolvedPath
}

function Assert-CleanupTreeSafe {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$BuildsRoot
    )

    $resolvedRoot = Assert-ImmediateBuildDirectory -Path $Path -BuildsRoot $BuildsRoot
    $rootPrefix = $resolvedRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $pending = [System.Collections.Generic.Stack[string]]::new()
    $pending.Push($resolvedRoot)
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($item in Get-ChildItem -LiteralPath $directory -Force -ErrorAction Stop) {
            $resolvedItem = [System.IO.Path]::GetFullPath($item.FullName)
            if (-not $resolvedItem.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Development portable cleanup content escaped its build directory: $resolvedItem"
            }
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Development portable cleanup content cannot be a reparse point: $resolvedItem"
            }
            if ($item.PSIsContainer) {
                $pending.Push($resolvedItem)
            }
        }
    }
}

function Write-JsonAtomic {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [object]$Value
    )

    $directory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $temporaryPath = Join-Path $directory ([Guid]::NewGuid().ToString('N') + '.tmp')
    try {
        $json = $Value | ConvertTo-Json -Depth 12
        [System.IO.File]::WriteAllText(
            $temporaryPath,
            $json + [Environment]::NewLine,
            [System.Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = Join-Path $PSScriptRoot '..'
}
$RepositoryRoot = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
if (-not (Test-Path -LiteralPath $RepositoryRoot -PathType Container)) {
    throw "Repository root does not exist: $RepositoryRoot"
}

$artifactsRoot = Join-Path $RepositoryRoot 'artifacts'
$devPortableRoot = Join-Path $artifactsRoot 'dev-portable'
$buildsRoot = Join-Path $devPortableRoot 'builds'
$dataRoot = Join-Path $devPortableRoot 'Data'
$openCvAuditRoot = Join-Path $artifactsRoot 'goal19-opencv-source'
$latestPath = Join-Path $devPortableRoot 'latest.json'

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $devPortableRoot 'size-report.json'
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
if (-not (Test-PathWithinRoot -Path $OutputPath -Root $artifactsRoot)) {
    throw "Size report output must remain under the ignored artifacts directory: $OutputPath"
}

$latestStatus = 'missing'
$latestBuildDirectory = $null
if (Test-Path -LiteralPath $latestPath -PathType Leaf) {
    try {
        $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
    }
    catch {
        throw "Development portable latest.json is invalid: $($_.Exception.Message)"
    }
    if (-not ($latest.PSObject.Properties.Name -contains 'buildDirectory') -or
        [string]::IsNullOrWhiteSpace([string]$latest.buildDirectory)) {
        throw 'Development portable latest.json does not define buildDirectory.'
    }

    try {
        $latestValue = [string]$latest.buildDirectory
        $latestCandidate = if ([System.IO.Path]::IsPathRooted($latestValue)) {
            [System.IO.Path]::GetFullPath($latestValue)
        }
        else {
            [System.IO.Path]::GetFullPath((Join-Path $devPortableRoot $latestValue))
        }
    }
    catch {
        throw "Development portable latest.json has an invalid buildDirectory: $($_.Exception.Message)"
    }
    $latestBuildDirectory = Assert-ImmediateBuildDirectory -Path $latestCandidate -BuildsRoot $buildsRoot
    $latestStatus = if (Test-Path -LiteralPath $latestBuildDirectory -PathType Container) {
        'resolved'
    }
    else {
        'target-missing'
    }
}

$buildDirectories = if (Test-Path -LiteralPath $buildsRoot -PathType Container) {
    @(Get-ChildItem -LiteralPath $buildsRoot -Directory -Force | Sort-Object `
            @{ Expression = { $_.LastWriteTimeUtc }; Descending = $true },
            @{ Expression = { $_.Name }; Descending = $true })
}
else {
    @()
}

foreach ($buildDirectory in $buildDirectories) {
    Assert-ImmediateBuildDirectory -Path $buildDirectory.FullName -BuildsRoot $buildsRoot | Out-Null
}

if ($PruneObsoleteBuilds.IsPresent -and $latestStatus -ne 'resolved') {
    throw "Cannot prune development portable builds because latest.json status is '$latestStatus'."
}

$previousBuildsToKeep = [System.Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase)
if ($PruneObsoleteBuilds.IsPresent) {
    foreach ($buildDirectory in $buildDirectories) {
        if ($buildDirectory.FullName.Equals($latestBuildDirectory, [StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        if ($previousBuildsToKeep.Count -ge $KeepPreviousBuilds) {
            break
        }
        [void]$previousBuildsToKeep.Add($buildDirectory.FullName)
    }
}

$retainedBuilds = [System.Collections.Generic.List[object]]::new()
$pruneCandidates = [System.Collections.Generic.List[object]]::new()
$prunedBuilds = [System.Collections.Generic.List[object]]::new()
[long]$reclaimedBytes = 0
foreach ($buildDirectory in $buildDirectories) {
    $resolvedBuild = Assert-ImmediateBuildDirectory -Path $buildDirectory.FullName -BuildsRoot $buildsRoot
    [long]$buildBytes = Get-DirectoryByteCount -Path $resolvedBuild
    $record = [ordered]@{
        name = $buildDirectory.Name
        relativePath = Get-RelativePath -Root $RepositoryRoot -Path $resolvedBuild
        bytes = $buildBytes
    }

    if ($null -ne $latestBuildDirectory -and
        $resolvedBuild.Equals($latestBuildDirectory, [StringComparison]::OrdinalIgnoreCase)) {
        $record.reason = 'latest'
        $retainedBuilds.Add([pscustomobject]$record)
        continue
    }
    if (-not $PruneObsoleteBuilds.IsPresent) {
        $record.reason = 'report-only'
        $retainedBuilds.Add([pscustomobject]$record)
        continue
    }
    if ($previousBuildsToKeep.Contains($resolvedBuild)) {
        $record.reason = 'previous-fallback'
        $retainedBuilds.Add([pscustomobject]$record)
        continue
    }

    $pruneCandidates.Add([pscustomobject]$record)
    if ($PSCmdlet.ShouldProcess($resolvedBuild, 'Remove obsolete development portable build')) {
        Assert-CleanupTreeSafe -Path $resolvedBuild -BuildsRoot $buildsRoot
        Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
        if (Test-Path -LiteralPath $resolvedBuild) {
            throw "Obsolete development portable build still exists after pruning: $resolvedBuild"
        }
        $record.reason = 'pruned'
        $prunedBuilds.Add([pscustomobject]$record)
        $reclaimedBytes += $buildBytes
    }
    else {
        $record.reason = 'not-approved'
        $retainedBuilds.Add([pscustomobject]$record)
    }
}

if ($latestStatus -eq 'resolved' -and
    -not (Test-Path -LiteralPath $latestBuildDirectory -PathType Container)) {
    throw "The exact latest development portable build was not preserved: $latestBuildDirectory"
}

$scanWarnings = [System.Collections.Generic.List[string]]::new()
$sizeIndex = Get-DirectorySizeIndex -Root $RepositoryRoot -Warnings $scanWarnings

function Get-IndexedSize {
    param([Parameter(Mandatory)][string]$Path)

    $resolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    if ($sizeIndex.ContainsKey($resolvedPath)) {
        return [long]$sizeIndex[$resolvedPath]
    }
    return [long]0
}

$binObjDirectories = @($sizeIndex.Keys | Where-Object {
        $leaf = Split-Path -Leaf $_
        $leaf -ieq 'bin' -or $leaf -ieq 'obj'
    } | Sort-Object { $_.Length })
$topLevelBinObjDirectories = [System.Collections.Generic.List[string]]::new()
foreach ($candidate in $binObjDirectories) {
    $isNested = $false
    foreach ($selected in $topLevelBinObjDirectories) {
        if (Test-PathWithinRoot -Path $candidate -Root $selected) {
            $isNested = $true
            break
        }
    }
    if (-not $isNested) {
        $topLevelBinObjDirectories.Add($candidate)
    }
}
[long]$binObjBytes = 0
foreach ($directory in $topLevelBinObjDirectories) {
    $binObjBytes += [long]$sizeIndex[$directory]
}

$largestDirectories = @($sizeIndex.GetEnumerator() |
    Where-Object { -not $_.Key.Equals($RepositoryRoot, [StringComparison]::OrdinalIgnoreCase) } |
    Sort-Object @{ Expression = { [long]$_.Value }; Descending = $true },
        @{ Expression = { $_.Key }; Descending = $false } |
    Select-Object -First 20 |
    ForEach-Object {
        [ordered]@{
            relativePath = Get-RelativePath -Root $RepositoryRoot -Path $_.Key
            bytes = [long]$_.Value
            mebibytes = [Math]::Round(([long]$_.Value) / 1MB, 2)
        }
    })

$remainingBuildDirectories = if (Test-Path -LiteralPath $buildsRoot -PathType Container) {
    @(Get-ChildItem -LiteralPath $buildsRoot -Directory -Force)
}
else {
    @()
}

$report = [ordered]@{
    schemaVersion = 1
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
    mode = if ($PruneObsoleteBuilds.IsPresent) { 'prune' } else { 'report-only' }
    repositoryRoot = $RepositoryRoot
    outputPath = $OutputPath
    summary = [ordered]@{
        repository = ConvertTo-SizeRecord -Bytes (Get-IndexedSize -Path $RepositoryRoot)
        developmentPortableBuilds = [ordered]@{
            count = $remainingBuildDirectories.Count
            size = ConvertTo-SizeRecord -Bytes (Get-IndexedSize -Path $buildsRoot)
        }
        developmentPortableData = ConvertTo-SizeRecord -Bytes (Get-IndexedSize -Path $dataRoot)
        binAndObj = [ordered]@{
            directoryCount = $topLevelBinObjDirectories.Count
            size = ConvertTo-SizeRecord -Bytes $binObjBytes
        }
        openCvAuditWorkspace = ConvertTo-SizeRecord -Bytes (Get-IndexedSize -Path $openCvAuditRoot)
    }
    paths = [ordered]@{
        developmentPortableBuilds = Get-RelativePath -Root $RepositoryRoot -Path $buildsRoot
        developmentPortableData = Get-RelativePath -Root $RepositoryRoot -Path $dataRoot
        openCvAuditWorkspace = Get-RelativePath -Root $RepositoryRoot -Path $openCvAuditRoot
    }
    latest = [ordered]@{
        metadataPath = Get-RelativePath -Root $RepositoryRoot -Path $latestPath
        status = $latestStatus
        buildDirectory = if ($null -eq $latestBuildDirectory) {
            $null
        }
        else {
            Get-RelativePath -Root $RepositoryRoot -Path $latestBuildDirectory
        }
        preserved = ($latestStatus -eq 'resolved' -and
            (Test-Path -LiteralPath $latestBuildDirectory -PathType Container))
    }
    cleanup = [ordered]@{
        requested = $PruneObsoleteBuilds.IsPresent
        performed = ($prunedBuilds.Count -gt 0)
        keepPreviousBuilds = $KeepPreviousBuilds
        safetyRoot = Get-RelativePath -Root $RepositoryRoot -Path $buildsRoot
        retainedBuilds = @($retainedBuilds)
        pruneCandidates = @($pruneCandidates)
        prunedBuilds = @($prunedBuilds)
        reclaimedBytes = $reclaimedBytes
    }
    largestDirectories = $largestDirectories
    scanWarnings = @($scanWarnings)
}

Write-JsonAtomic -Path $OutputPath -Value $report
Write-Host "Development portable size report: $OutputPath"
Write-Host "Repository: $([Math]::Round($report.summary.repository.gibibytes, 3)) GiB"
Write-Host "Builds: $($report.summary.developmentPortableBuilds.count) folders, $([Math]::Round($report.summary.developmentPortableBuilds.size.gibibytes, 3)) GiB"
Write-Host "Pruned: $($prunedBuilds.Count) folders, $([Math]::Round($reclaimedBytes / 1GB, 3)) GiB"
