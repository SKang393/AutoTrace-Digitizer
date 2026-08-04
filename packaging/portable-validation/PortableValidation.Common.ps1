# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

Set-StrictMode -Version Latest

function Write-PvJsonFile {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [object]$Value
    )

    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    [System.IO.File]::WriteAllText(
        $Path,
        (($Value | ConvertTo-Json -Depth 30) + [Environment]::NewLine),
        [System.Text.UTF8Encoding]::new($false))
}

function Get-PvRequiredGateIds {
    return @(
        'path-with-spaces-and-unicode',
        'shared-preview-data-root',
        'offline-no-network-observation',
        'normal-portable-dot-data-root',
        'read-only-folder-diagnostic',
        'no-registry-configuration-dependency-observation',
        'file-system-write-trace'
    )
}

function Test-PvValidationGates {
    param(
        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [object[]]$Gates,

        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [object[]]$HarnessErrors
    )

    if ($HarnessErrors.Count -gt 0 -or $Gates.Count -eq 0 -or
        @($Gates | Where-Object { [string]$_.status -ne 'PASS' }).Count -gt 0) {
        return $false
    }

    foreach ($requiredId in Get-PvRequiredGateIds) {
        if (@($Gates | Where-Object { [string]$_.id -eq $requiredId }).Count -ne 1) {
            return $false
        }
    }

    return $true
}

function Test-PvPathUnderRoot {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$Root,

        [switch]$AllowEqual
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullRoot = [System.IO.Path]::GetFullPath($Root)
    if ($AllowEqual.IsPresent -and
        [string]::Equals(
            $fullPath.TrimEnd([char[]]@('\', '/')),
            $fullRoot.TrimEnd([char[]]@('\', '/')),
            [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    $rootPrefix = $fullRoot.TrimEnd([char[]]@('\', '/')) +
        [System.IO.Path]::DirectorySeparatorChar
    return $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)
}

function Get-PvApplicationRootSnapshot {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    $fullRoot = [System.IO.Path]::GetFullPath($Path)
    if (-not (Test-Path -LiteralPath $fullRoot)) {
        return [pscustomobject]@{
            root = $fullRoot
            exists = $false
            capturedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
            entries = @()
        }
    }
    if (-not (Test-Path -LiteralPath $fullRoot -PathType Container)) {
        throw "The application data root exists but is not a directory: $fullRoot"
    }

    $entries = [System.Collections.Generic.List[object]]::new()
    $rootItem = Get-Item -LiteralPath $fullRoot -Force
    $entries.Add([ordered]@{
            relativePath = '.'
            kind = 'directory'
            length = $null
            lastWriteTimeUtc = $rootItem.LastWriteTimeUtc.ToString('O')
            attributes = [string]$rootItem.Attributes
            sha256 = $null
        })

    foreach ($item in Get-ChildItem -LiteralPath $fullRoot -Force -Recurse | Sort-Object FullName) {
        if (-not (Test-PvPathUnderRoot -Path $item.FullName -Root $fullRoot)) {
            throw "Snapshot enumeration escaped the application data root: $($item.FullName)"
        }
        $relativePath = $item.FullName.Substring($fullRoot.Length).TrimStart([char[]]@('\', '/'))
        $isDirectory = $item.PSIsContainer
        $sha256 = $null
        if (-not $isDirectory) {
            $sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        $entries.Add([ordered]@{
                relativePath = $relativePath.Replace('\', '/')
                kind = if ($isDirectory) { 'directory' } else { 'file' }
                length = if ($isDirectory) { $null } else { [long]$item.Length }
                lastWriteTimeUtc = $item.LastWriteTimeUtc.ToString('O')
                attributes = [string]$item.Attributes
                sha256 = $sha256
            })
    }

    return [pscustomobject]@{
        root = $fullRoot
        exists = $true
        capturedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
        entries = @($entries)
    }
}

function Compare-PvApplicationRootSnapshot {
    param(
        [Parameter(Mandatory)]
        [object]$Before,

        [Parameter(Mandatory)]
        [object]$After
    )

    $beforeRoot = [System.IO.Path]::GetFullPath([string]$Before.root)
    $afterRoot = [System.IO.Path]::GetFullPath([string]$After.root)
    if (-not [string]::Equals($beforeRoot, $afterRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cannot compare snapshots from different application roots: '$beforeRoot' and '$afterRoot'."
    }

    $events = [System.Collections.Generic.List[object]]::new()
    if (-not $Before.exists -and -not $After.exists) {
        return @()
    }
    if (-not $Before.exists -and $After.exists) {
        $events.Add([ordered]@{
                observedAtUtc = [string]$After.capturedAtUtc
                watcherRoot = $afterRoot
                changeType = 'Created'
                path = $afterRoot
                oldPath = $null
                source = 'exact-root-before-after-snapshot'
            })
        foreach ($entry in $After.entries | Where-Object { [string]$_.relativePath -ne '.' }) {
            $events.Add([ordered]@{
                    observedAtUtc = [string]$After.capturedAtUtc
                    watcherRoot = $afterRoot
                    changeType = 'Created'
                    path = Join-Path $afterRoot ([string]$entry.relativePath).Replace('/', '\')
                    oldPath = $null
                    source = 'exact-root-before-after-snapshot'
                })
        }
        return @($events)
    }
    if ($Before.exists -and -not $After.exists) {
        $events.Add([ordered]@{
                observedAtUtc = [string]$After.capturedAtUtc
                watcherRoot = $afterRoot
                changeType = 'Deleted'
                path = $afterRoot
                oldPath = $null
                source = 'exact-root-before-after-snapshot'
            })
        return @($events)
    }

    $beforeEntries = [System.Collections.Generic.Dictionary[string,object]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)
    $afterEntries = [System.Collections.Generic.Dictionary[string,object]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $Before.entries) {
        $beforeEntries.Add([string]$entry.relativePath, $entry)
    }
    foreach ($entry in $After.entries) {
        $afterEntries.Add([string]$entry.relativePath, $entry)
    }

    $relativePaths = @($beforeEntries.Keys) + @($afterEntries.Keys)
    foreach ($relativePath in @($relativePaths | Sort-Object -Unique)) {
        $beforePresent = $beforeEntries.ContainsKey($relativePath)
        $afterPresent = $afterEntries.ContainsKey($relativePath)
        $fullPath = if ($relativePath -eq '.') {
            $afterRoot
        }
        else {
            Join-Path $afterRoot $relativePath.Replace('/', '\')
        }
        if (-not $beforePresent) {
            $events.Add([ordered]@{
                    observedAtUtc = [string]$After.capturedAtUtc
                    watcherRoot = $afterRoot
                    changeType = 'Created'
                    path = $fullPath
                    oldPath = $null
                    source = 'exact-root-before-after-snapshot'
                })
            continue
        }
        if (-not $afterPresent) {
            $events.Add([ordered]@{
                    observedAtUtc = [string]$After.capturedAtUtc
                    watcherRoot = $afterRoot
                    changeType = 'Deleted'
                    path = $fullPath
                    oldPath = $null
                    source = 'exact-root-before-after-snapshot'
                })
            continue
        }

        $beforeEntry = $beforeEntries[$relativePath]
        $afterEntry = $afterEntries[$relativePath]
        $changed =
            [string]$beforeEntry.kind -ne [string]$afterEntry.kind -or
            [string]$beforeEntry.length -ne [string]$afterEntry.length -or
            [string]$beforeEntry.lastWriteTimeUtc -ne [string]$afterEntry.lastWriteTimeUtc -or
            [string]$beforeEntry.attributes -ne [string]$afterEntry.attributes -or
            [string]$beforeEntry.sha256 -ne [string]$afterEntry.sha256
        if ($changed) {
            $events.Add([ordered]@{
                    observedAtUtc = [string]$After.capturedAtUtc
                    watcherRoot = $afterRoot
                    changeType = 'Changed'
                    path = $fullPath
                    oldPath = $null
                    source = 'exact-root-before-after-snapshot'
                })
        }
    }

    return @($events)
}

function Remove-PvSandboxTree {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$SandboxRoot
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullSandboxRoot = [System.IO.Path]::GetFullPath($SandboxRoot)
    if (-not (Test-PvPathUnderRoot -Path $fullPath -Root $fullSandboxRoot)) {
        throw "Refusing to remove a path outside the validation sandbox: $fullPath"
    }

    if (Test-Path -LiteralPath $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    }
}

function Start-PvWriteTrace {
    param(
        [Parameter(Mandatory)]
        [string[]]$Roots
    )

    $eventQueue = [System.Collections.Concurrent.ConcurrentQueue[object]]::new()
    $errorQueue = [System.Collections.Concurrent.ConcurrentQueue[object]]::new()
    $bindings = [System.Collections.Generic.List[object]]::new()
    $traceId = [Guid]::NewGuid().ToString('N')
    $watcherIndex = 0

    foreach ($root in @($Roots | ForEach-Object {
                [System.IO.Path]::GetFullPath($_)
            } | Sort-Object -Unique)) {
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        $watcher = [System.IO.FileSystemWatcher]::new($root)
        $watcher.IncludeSubdirectories = $true
        $watcher.InternalBufferSize = 65536
        $watcher.NotifyFilter =
            [System.IO.NotifyFilters]::FileName -bor
            [System.IO.NotifyFilters]::DirectoryName -bor
            [System.IO.NotifyFilters]::LastWrite -bor
            [System.IO.NotifyFilters]::Size -bor
            [System.IO.NotifyFilters]::Attributes

        $jobs = [System.Collections.Generic.List[object]]::new()
        foreach ($eventName in @('Created', 'Changed', 'Deleted')) {
            $sourceIdentifier = "GraphReader.Pv.${traceId}.${watcherIndex}.${eventName}"
            $job = Register-ObjectEvent `
                -InputObject $watcher `
                -EventName $eventName `
                -SourceIdentifier $sourceIdentifier `
                -MessageData ([pscustomobject]@{ Queue = $eventQueue }) `
                -Action {
                $event.MessageData.Queue.Enqueue([ordered]@{
                    observedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
                    watcherRoot = [string]$event.Sender.Path
                    changeType = [string]$event.SourceEventArgs.ChangeType
                    path = [string]$event.SourceEventArgs.FullPath
                    oldPath = $null
                    source = 'filesystem-watcher'
                })
            }
            $jobs.Add($job)
        }

        $renameIdentifier = "GraphReader.Pv.${traceId}.${watcherIndex}.Renamed"
        $renameJob = Register-ObjectEvent `
            -InputObject $watcher `
            -EventName Renamed `
            -SourceIdentifier $renameIdentifier `
            -MessageData ([pscustomobject]@{ Queue = $eventQueue }) `
            -Action {
            $event.MessageData.Queue.Enqueue([ordered]@{
                    observedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
                    watcherRoot = [string]$event.Sender.Path
                    changeType = [string]$event.SourceEventArgs.ChangeType
                    path = [string]$event.SourceEventArgs.FullPath
                    oldPath = [string]$event.SourceEventArgs.OldFullPath
                    source = 'filesystem-watcher'
                })
        }
        $jobs.Add($renameJob)

        $errorIdentifier = "GraphReader.Pv.${traceId}.${watcherIndex}.Error"
        $errorJob = Register-ObjectEvent `
            -InputObject $watcher `
            -EventName Error `
            -SourceIdentifier $errorIdentifier `
            -MessageData ([pscustomobject]@{ Queue = $errorQueue }) `
            -Action {
            $exception = $event.SourceEventArgs.GetException()
            $event.MessageData.Queue.Enqueue([ordered]@{
                    observedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
                    watcherRoot = [string]$event.Sender.Path
                    error = if ($null -eq $exception) {
                        'Unknown FileSystemWatcher error.'
                    }
                    else {
                        $exception.Message
                    }
                })
        }
        $jobs.Add($errorJob)
        $watcher.EnableRaisingEvents = $true

        $bindings.Add([pscustomobject]@{
                Watcher = $watcher
                Jobs = $jobs
            })
        $watcherIndex++
    }

    return [pscustomobject]@{
        Bindings = $bindings
        Events = $eventQueue
        Errors = $errorQueue
    }
}

function Stop-PvWriteTrace {
    param(
        [Parameter(Mandatory)]
        [object]$Trace,

        [int]$DrainMilliseconds = 400
    )

    if ($DrainMilliseconds -gt 0) {
        Start-Sleep -Milliseconds $DrainMilliseconds
    }

    foreach ($binding in $Trace.Bindings) {
        $binding.Watcher.EnableRaisingEvents = $false
        foreach ($job in $binding.Jobs) {
            Unregister-Event -SourceIdentifier $job.Name -ErrorAction SilentlyContinue
            Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
        }
        $binding.Watcher.Dispose()
    }

    return [pscustomobject]@{
        Events = @($Trace.Events.ToArray())
        Errors = @($Trace.Errors.ToArray())
    }
}

function Get-PvProcessComponentEvidence {
    param(
        [Parameter(Mandatory)]
        [System.Diagnostics.Process]$Process
    )

    $modules = [System.Collections.Generic.List[object]]::new()
    $collectionError = $null
    try {
        $Process.Refresh()
        foreach ($module in $Process.Modules) {
            $moduleName = [string]$module.ModuleName
            $fileName = [string]$module.FileName
            if ($moduleName -notmatch '(?i)^(Presentation|WindowsBase|wpfgfx|D3D|DXGI|amd|ati|nv|igd|coreclr|hostfxr|clrjit)' -and
                $fileName -notmatch '(?i)[\\/](AMD|ATI|NVIDIA|Intel)[\\/]') {
                continue
            }

            $version = $null
            try {
                $version = $module.FileVersionInfo
            }
            catch {
            }
            $modules.Add([ordered]@{
                    moduleName = $moduleName
                    fileName = $fileName
                    companyName = if ($null -eq $version) { $null } else { [string]$version.CompanyName }
                    productName = if ($null -eq $version) { $null } else { [string]$version.ProductName }
                })
        }
    }
    catch {
        $collectionError = $_.Exception.Message
    }

    $processPath = $null
    try {
        $processPath = $Process.MainModule.FileName
    }
    catch {
    }
    return [pscustomobject]@{
        processId = $Process.Id
        processName = $Process.ProcessName
        executablePath = $processPath
        modules = @($modules)
        moduleCollectionError = $collectionError
    }
}

function Resolve-PvExternalCacheComponent {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [object]$ApplicationProcessEvidence,

        [string]$ChangeType,

        [AllowEmptyCollection()]
        [string[]]$ExternalSystemRoots = @()
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $moduleNames = @($ApplicationProcessEvidence.modules | ForEach-Object {
            [string]$_.moduleName
        })

    if (-not [string]::IsNullOrWhiteSpace([string]$ApplicationProcessEvidence.executablePath)) {
        $applicationRoot = Split-Path -Parent ([string]$ApplicationProcessEvidence.executablePath)
        $isApplicationChild = Test-PvPathUnderRoot -Path $fullPath -Root $applicationRoot
        $relative = if ($isApplicationChild) {
            $fullPath.Substring($applicationRoot.Length).TrimStart([char[]]@('\', '/'))
        }
        else {
            $null
        }
        $isCultureDirectory =
            $isApplicationChild -and
            $relative -notmatch '[\\/]' -and
            $relative -match '^[a-z]{2}(?:-[A-Za-z]{2,4})?$' -and
            [string]$ChangeType -eq 'Changed' -and
            (Test-Path -LiteralPath $fullPath -PathType Container)
        if ($isCultureDirectory) {
            $matchingModules = @($moduleNames | Where-Object {
                    $_ -match '(?i)^(Presentation|WindowsBase|wpfgfx|coreclr)'
                })
            return [pscustomobject]@{
                recognized = $true
                attributed = $matchingModules.Count -gt 0
                component = 'Microsoft .NET/WPF resource manager'
                purpose = 'satellite-resource directory metadata observation without file persistence'
                evidence = if ($matchingModules.Count -gt 0) {
                    "Changed-only event targets existing culture resource directory '$relative' beside GraphReader.App, no child file mutation was traced, and the process loaded .NET/WPF modules: $($matchingModules -join ', ')."
                }
                else {
                    "Changed-only event targets existing culture resource directory '$relative', but no .NET/WPF module evidence was captured from GraphReader.App."
                }
            }
        }
    }

    foreach ($root in $ExternalSystemRoots) {
        $fullRoot = [System.IO.Path]::GetFullPath($root)
        if (-not (Test-PvPathUnderRoot -Path $fullPath -Root $fullRoot -AllowEqual)) {
            continue
        }
        $relative = $fullPath.Substring($fullRoot.Length).TrimStart([char[]]@('\', '/'))
        if ($relative -notmatch '(?i)^\.net(?:[\\/]|$)') {
            continue
        }
        $matchingModules = @($moduleNames | Where-Object {
                $_ -match '(?i)^(coreclr|hostfxr|clrjit)'
            })
        return [pscustomobject]@{
            recognized = $true
            attributed = $matchingModules.Count -gt 0
            component = 'Microsoft .NET runtime'
            purpose = '.NET runtime extraction or compilation cache'
            evidence = if ($matchingModules.Count -gt 0) {
                "Destination is under a supplied system-managed temporary root in the .net cache subtree and GraphReader.App loaded .NET runtime modules: $($matchingModules -join ', ')."
            }
            else {
                'Destination is under a supplied system-managed .net cache subtree, but no .NET runtime module evidence was captured from GraphReader.App.'
            }
        }
    }

    if ($fullPath -match '(?i)[\\/]AppData[\\/]Local[\\/]AMD(?:[\\/]|$)') {
        $matchingModules = @($moduleNames | Where-Object { $_ -match '(?i)^(amd|ati)' })
        return [pscustomobject]@{
            recognized = $true
            attributed = $matchingModules.Count -gt 0
            component = 'AMD graphics driver'
            purpose = 'GPU driver-managed shader or Direct3D cache'
            evidence = if ($matchingModules.Count -gt 0) {
                "Destination is under the AMD cache root and GraphReader.App loaded AMD/ATI modules: $($matchingModules -join ', ')."
            }
            else {
                'Destination is under the AMD cache root, but no AMD/ATI module evidence was captured from GraphReader.App.'
            }
        }
    }

    if ($fullPath -match '(?i)[\\/]AppData[\\/]Local[\\/](NVIDIA|NVIDIA Corporation)[\\/](DXCache|GLCache|NV_Cache)(?:[\\/]|$)') {
        $matchingModules = @($moduleNames | Where-Object { $_ -match '(?i)^nv' })
        return [pscustomobject]@{
            recognized = $true
            attributed = $matchingModules.Count -gt 0
            component = 'NVIDIA graphics driver'
            purpose = 'GPU driver-managed shader or Direct3D cache'
            evidence = if ($matchingModules.Count -gt 0) {
                "Destination is under an NVIDIA cache root and GraphReader.App loaded NVIDIA modules: $($matchingModules -join ', ')."
            }
            else {
                'Destination is under an NVIDIA cache root, but no NVIDIA module evidence was captured from GraphReader.App.'
            }
        }
    }

    if ($fullPath -match '(?i)[\\/]AppData[\\/]Local[\\/]Microsoft[\\/]Windows[\\/]Caches(?:[\\/]|$)' -or
        $fullPath -match '(?i)[\\/]AppData[\\/]Local[\\/]D3DSCache(?:[\\/]|$)' -or
        $fullPath -match '(?i)[\\/]AppData[\\/]Local[\\/]FontCache(?:[\\/]|$)') {
        $matchingModules = @($moduleNames | Where-Object {
                $_ -match '(?i)^(Presentation|WindowsBase|wpfgfx|D3D|DXGI)'
            })
        return [pscustomobject]@{
            recognized = $true
            attributed = $matchingModules.Count -gt 0
            component = 'Microsoft Windows/WPF graphics and resource runtime'
            purpose = 'Windows-managed WPF, font, or graphics cache'
            evidence = if ($matchingModules.Count -gt 0) {
                "Destination is under a Windows-managed cache and GraphReader.App loaded WPF/graphics modules: $($matchingModules -join ', ')."
            }
            else {
                'Destination is under a Windows-managed cache, but no WPF/graphics module evidence was captured from GraphReader.App.'
            }
        }
    }

    return [pscustomobject]@{
        recognized = $false
        attributed = $false
        component = $null
        purpose = $null
        evidence = 'No supported external cache signature matched this destination.'
    }
}

function Classify-PvWriteEvents {
    param(
        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [object[]]$Events,

        [Parameter(Mandatory)]
        [string]$ConfiguredDataRoot,

        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [string[]]$ApplicationOwnedExternalRoots,

        [Parameter(Mandatory)]
        [object]$ApplicationProcessEvidence,

        [AllowEmptyCollection()]
        [string[]]$ExternalSystemRoots = @(),

        [AllowEmptyCollection()]
        [string[]]$UserSelectedWritableRoots = @()
    )

    $configuredRoot = [System.IO.Path]::GetFullPath($ConfiguredDataRoot)
    $applicationRoots = @($ApplicationOwnedExternalRoots | ForEach-Object {
            [System.IO.Path]::GetFullPath($_)
        } | Sort-Object -Unique)
    $userRoots = @($UserSelectedWritableRoots | ForEach-Object {
            [System.IO.Path]::GetFullPath($_)
        } | Sort-Object -Unique)
    $classified = [System.Collections.Generic.List[object]]::new()
    $pending = [System.Collections.Generic.List[object]]::new()
    $externalWarnings = [System.Collections.Generic.List[object]]::new()

    $processSummary = [ordered]@{
        processId = $ApplicationProcessEvidence.processId
        processName = $ApplicationProcessEvidence.processName
        executablePath = $ApplicationProcessEvidence.executablePath
    }

    foreach ($event in $Events) {
        $eventPath = [string]$event.path
        if ([string]::IsNullOrWhiteSpace($eventPath)) {
            $classified.Add([ordered]@{
                    classification = 'failure'
                    ownership = 'unattributed'
                    purpose = 'unknown filesystem mutation'
                    responsibleProcess = $processSummary
                    responsibleComponent = 'unattributed'
                    evidence = 'The trace event had no destination path.'
                    event = $event
                })
            continue
        }

        $fullPath = [System.IO.Path]::GetFullPath($eventPath)
        if (Test-PvPathUnderRoot -Path $fullPath -Root $configuredRoot -AllowEqual) {
            $relative = $fullPath.Substring($configuredRoot.Length).TrimStart([char[]]@('\', '/'))
            $firstSegment = @($relative -split '[\\/]')[0]
            $purpose = switch -Regex ($firstSegment) {
                '(?i)^Settings$' { 'Graph Auto Reader portable settings'; break }
                '(?i)^Cache$' { 'Graph Auto Reader portable cache'; break }
                '(?i)^Logs$' { 'Graph Auto Reader portable logs'; break }
                '(?i)^Autosave$' { 'Graph Auto Reader portable autosave'; break }
                '(?i)^Recovery$' { 'Graph Auto Reader portable recovery'; break }
                default { 'Graph Auto Reader portable mutable data' }
            }
            $classified.Add([ordered]@{
                    classification = 'allowed'
                    ownership = 'graph-auto-reader'
                    purpose = $purpose
                    responsibleProcess = $processSummary
                    responsibleComponent = 'Graph Auto Reader'
                    evidence = "Destination is within the configured portable Data root '$configuredRoot'."
                    event = $event
                })
            continue
        }

        $userSelected = $false
        foreach ($root in $userRoots) {
            if (Test-PvPathUnderRoot -Path $fullPath -Root $root -AllowEqual) {
                $userSelected = $true
                break
            }
        }
        if ($userSelected) {
            $classified.Add([ordered]@{
                    classification = 'allowed'
                    ownership = 'user-selected'
                    purpose = 'user-selected project or export destination'
                    responsibleProcess = $processSummary
                    responsibleComponent = 'Graph Auto Reader'
                    evidence = 'Destination is within an explicitly supplied user-selected writable root.'
                    event = $event
                })
            continue
        }

        $applicationOwned = $false
        foreach ($root in $applicationRoots) {
            if (Test-PvPathUnderRoot -Path $fullPath -Root $root -AllowEqual) {
                $applicationOwned = $true
                break
            }
        }
        $extension = [System.IO.Path]::GetExtension($fullPath)
        if ($applicationOwned -or
            $extension -match '(?i)^\.(garproj|garrecovery)$' -or
            [System.IO.Path]::GetFileName($fullPath) -match '(?i)\.garproj\.autosave$') {
            $purpose = if ($fullPath -match '(?i)[\\/]Settings(?:[\\/]|$)') {
                'Graph Auto Reader settings'
            }
            elseif ($fullPath -match '(?i)[\\/]Cache(?:[\\/]|$)') {
                'Graph Auto Reader cache'
            }
            elseif ($fullPath -match '(?i)[\\/]Logs(?:[\\/]|$)') {
                'Graph Auto Reader logs'
            }
            elseif ($fullPath -match '(?i)[\\/]Autosave(?:[\\/]|$)') {
                'Graph Auto Reader autosave'
            }
            elseif ($fullPath -match '(?i)[\\/]Recovery(?:[\\/]|$)') {
                'Graph Auto Reader recovery'
            }
            else {
                'Graph Auto Reader project persistence'
            }
            $classified.Add([ordered]@{
                    classification = 'failure'
                    ownership = 'graph-auto-reader'
                    purpose = $purpose
                    responsibleProcess = $processSummary
                    responsibleComponent = 'Graph Auto Reader'
                    evidence = "Graph Auto Reader-owned persistence targeted '$fullPath' outside configured portable Data and user-selected writable roots."
                    event = $event
                })
            continue
        }

        $external = Resolve-PvExternalCacheComponent `
            -Path $fullPath `
            -ApplicationProcessEvidence $ApplicationProcessEvidence `
            -ChangeType ([string]$event.changeType) `
            -ExternalSystemRoots $ExternalSystemRoots
        if ($external.recognized -and $external.attributed) {
            $record = [ordered]@{
                classification = 'warning'
                ownership = 'external-system-component'
                purpose = $external.purpose
                responsibleProcess = $processSummary
                responsibleComponent = $external.component
                evidence = $external.evidence
                event = $event
            }
            $classified.Add($record)
            $externalWarnings.Add($record)
            continue
        }
        if ($external.recognized) {
            $classified.Add([ordered]@{
                    classification = 'failure'
                    ownership = 'unattributed'
                    purpose = $external.purpose
                    responsibleProcess = $processSummary
                    responsibleComponent = $external.component
                    evidence = $external.evidence
                    event = $event
                })
            continue
        }

        $pending.Add([pscustomobject]@{
                FullPath = $fullPath
                Event = $event
            })
    }

    foreach ($candidate in $pending) {
        $allowedEvidence = @($classified | Where-Object {
                [string]$candidate.Event.changeType -eq 'Changed' -and
                [string]$_.classification -eq 'allowed' -and
                (Test-PvPathUnderRoot -Path ([string]$_.event.path) -Root $candidate.FullPath)
            } | Select-Object -First 1)
        if ($allowedEvidence.Count -eq 1) {
            $source = $allowedEvidence[0]
            $classified.Add([ordered]@{
                    classification = 'allowed'
                    ownership = 'graph-auto-reader'
                    purpose = 'parent directory metadata changed by allowed portable Data persistence'
                    responsibleProcess = $processSummary
                    responsibleComponent = 'Graph Auto Reader'
                    evidence = "Parent metadata event is an ancestor of allowed portable Data event '$($source.event.path)'."
                    event = $candidate.Event
                })
            continue
        }

        $parentEvidence = @($externalWarnings | Where-Object {
                [string]$candidate.Event.changeType -in @('Created', 'Changed') -and
                (Test-PvPathUnderRoot -Path ([string]$_.event.path) -Root $candidate.FullPath)
            } | Select-Object -First 1)
        if ($parentEvidence.Count -eq 1) {
            $source = $parentEvidence[0]
            $classified.Add([ordered]@{
                    classification = 'warning'
                    ownership = 'external-system-component'
                    purpose = 'parent directory metadata changed by an attributed external cache write'
                    responsibleProcess = $processSummary
                    responsibleComponent = $source.responsibleComponent
                    evidence = "Parent metadata event is an ancestor of attributed external cache event '$($source.event.path)'. $($source.evidence)"
                    event = $candidate.Event
                })
        }
        else {
            $classified.Add([ordered]@{
                    classification = 'failure'
                    ownership = 'unattributed'
                    purpose = 'filesystem mutation outside configured destinations'
                    responsibleProcess = $processSummary
                    responsibleComponent = 'unattributed'
                    evidence = 'The mutation is outside configured destinations and has no supported external component attribution.'
                    event = $candidate.Event
                })
        }
    }

    return [pscustomobject]@{
        ClassifiedEvents = @($classified)
        Allowed = @($classified | Where-Object { [string]$_.classification -eq 'allowed' })
        Warnings = @($classified | Where-Object { [string]$_.classification -eq 'warning' })
        Failures = @($classified | Where-Object { [string]$_.classification -eq 'failure' })
    }
}

function Start-PvIsolatedProcess {
    param(
        [Parameter(Mandatory)]
        [string]$ExecutablePath,

        [Parameter(Mandatory)]
        [string]$WorkingDirectory,

        [string[]]$ArgumentList = @(),

        [Parameter(Mandatory)]
        [hashtable]$Environment,

        [ValidateSet('Hidden', 'Minimized', 'Normal')]
        [string]$WindowStyle = 'Hidden'
    )

    $priorValues = @{}
    try {
        foreach ($name in $Environment.Keys) {
            $priorValues[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $Environment[$name], 'Process')
        }

        $parameters = @{
            FilePath = $ExecutablePath
            WorkingDirectory = $WorkingDirectory
            PassThru = $true
            WindowStyle = $WindowStyle
        }
        if ($ArgumentList.Count -gt 0) {
            $parameters.ArgumentList = $ArgumentList
        }
        return Start-Process @parameters
    }
    finally {
        foreach ($name in $Environment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $priorValues[$name], 'Process')
        }
    }
}

function Wait-PvProcessExit {
    param(
        [Parameter(Mandatory)]
        [System.Diagnostics.Process]$Process,

        [int]$TimeoutMilliseconds = 15000
    )

    if (-not $Process.WaitForExit($TimeoutMilliseconds)) {
        throw "Process $($Process.Id) did not exit within $TimeoutMilliseconds ms."
    }

    return $Process.ExitCode
}

function Stop-PvProcess {
    param(
        [Parameter(Mandatory)]
        [System.Diagnostics.Process]$Process,

        [int]$TimeoutMilliseconds = 5000
    )

    if ($Process.HasExited) {
        return [pscustomobject]@{ Forced = $false; ExitCode = $Process.ExitCode }
    }

    $closed = $Process.CloseMainWindow()
    if ($closed -and $Process.WaitForExit($TimeoutMilliseconds)) {
        return [pscustomobject]@{ Forced = $false; ExitCode = $Process.ExitCode }
    }

    $Process.Kill()
    $Process.WaitForExit()
    return [pscustomobject]@{ Forced = $true; ExitCode = $Process.ExitCode }
}

function Wait-PvAutomationElement {
    param(
        [Parameter(Mandatory)]
        [System.Diagnostics.Process]$Process,

        [string]$AutomationId,

        [int]$TimeoutMilliseconds = 15000
    )

    Add-Type -AssemblyName UIAutomationClient
    Add-Type -AssemblyName UIAutomationTypes
    $deadline = [DateTimeOffset]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    $processCondition = [System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty,
        $Process.Id)

    do {
        if ($Process.HasExited) {
            return $null
        }

        try {
            $window = [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
                [System.Windows.Automation.TreeScope]::Children,
                $processCondition)
            if ($null -ne $window) {
                if ([string]::IsNullOrWhiteSpace($AutomationId)) {
                    return [pscustomobject]@{
                        Name = [string]$window.Current.Name
                        AutomationId = [string]$window.Current.AutomationId
                    }
                }

                $idCondition = [System.Windows.Automation.PropertyCondition]::new(
                    [System.Windows.Automation.AutomationElement]::AutomationIdProperty,
                    $AutomationId)
                $element = $window.FindFirst(
                    [System.Windows.Automation.TreeScope]::Descendants,
                    $idCondition)
                if ($null -ne $element) {
                    return [pscustomobject]@{
                        Name = [string]$element.Current.Name
                        AutomationId = [string]$element.Current.AutomationId
                    }
                }
            }
        }
        catch [System.Windows.Automation.ElementNotAvailableException] {
        }

        Start-Sleep -Milliseconds 100
    } while ([DateTimeOffset]::UtcNow -lt $deadline)

    return $null
}

function Observe-PvProcessNetwork {
    param(
        [Parameter(Mandatory)]
        [System.Diagnostics.Process]$Process,

        [int]$ObservationSeconds = 3,

        [int]$PollMilliseconds = 250
    )

    if ($null -eq (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) -or
        $null -eq (Get-Command Get-NetUDPEndpoint -ErrorAction SilentlyContinue)) {
        return [pscustomobject]@{
            Succeeded = $false
            SampleCount = 0
            Tcp = @()
            Udp = @()
            Error = 'Get-NetTCPConnection or Get-NetUDPEndpoint is unavailable.'
        }
    }

    $tcp = [System.Collections.Generic.Dictionary[string,object]]::new(
        [System.StringComparer]::Ordinal)
    $udp = [System.Collections.Generic.Dictionary[string,object]]::new(
        [System.StringComparer]::Ordinal)
    $sampleCount = 0
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($ObservationSeconds)
    try {
        do {
            if ($Process.HasExited) {
                break
            }

            $sampleCount++
            foreach ($connection in @(Get-NetTCPConnection -OwningProcess $Process.Id -ErrorAction SilentlyContinue)) {
                $key = '{0}|{1}|{2}|{3}|{4}' -f
                    $connection.LocalAddress,
                    $connection.LocalPort,
                    $connection.RemoteAddress,
                    $connection.RemotePort,
                    $connection.State
                if (-not $tcp.ContainsKey($key)) {
                    $tcp[$key] = [ordered]@{
                        localAddress = [string]$connection.LocalAddress
                        localPort = [int]$connection.LocalPort
                        remoteAddress = [string]$connection.RemoteAddress
                        remotePort = [int]$connection.RemotePort
                        state = [string]$connection.State
                    }
                }
            }
            foreach ($endpoint in @(Get-NetUDPEndpoint -OwningProcess $Process.Id -ErrorAction SilentlyContinue)) {
                $key = '{0}|{1}' -f $endpoint.LocalAddress, $endpoint.LocalPort
                if (-not $udp.ContainsKey($key)) {
                    $udp[$key] = [ordered]@{
                        localAddress = [string]$endpoint.LocalAddress
                        localPort = [int]$endpoint.LocalPort
                    }
                }
            }

            if ([DateTimeOffset]::UtcNow -lt $deadline) {
                Start-Sleep -Milliseconds $PollMilliseconds
            }
        } while ([DateTimeOffset]::UtcNow -lt $deadline)

        return [pscustomobject]@{
            Succeeded = $true
            SampleCount = $sampleCount
            Tcp = @($tcp.Values)
            Udp = @($udp.Values)
            Error = $null
        }
    }
    catch {
        return [pscustomobject]@{
            Succeeded = $false
            SampleCount = $sampleCount
            Tcp = @($tcp.Values)
            Udp = @($udp.Values)
            Error = $_.Exception.Message
        }
    }
}

function Set-PvDirectoryDenyWrite {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    $acl = Get-Acl -LiteralPath $fullPath
    $originalSddl = $acl.Sddl
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    if ($null -eq $identity.User) {
        throw 'The current Windows identity has no user SID.'
    }

    $inheritance =
        [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor
        [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
    $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $identity.User,
        [System.Security.AccessControl.FileSystemRights]::Write,
        $inheritance,
        [System.Security.AccessControl.PropagationFlags]::None,
        [System.Security.AccessControl.AccessControlType]::Deny)
    [void]$acl.AddAccessRule($rule)
    Set-Acl -LiteralPath $fullPath -AclObject $acl

    return [pscustomobject]@{
        Path = $fullPath
        OriginalSddl = $originalSddl
    }
}

function Restore-PvDirectoryAcl {
    param(
        [Parameter(Mandatory)]
        [object]$State
    )

    $security = [System.Security.AccessControl.DirectorySecurity]::new()
    $security.SetSecurityDescriptorSddlForm(
        [string]$State.OriginalSddl,
        [System.Security.AccessControl.AccessControlSections]::All)
    Set-Acl -LiteralPath ([string]$State.Path) -AclObject $security
}
