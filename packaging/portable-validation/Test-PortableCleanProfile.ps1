# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$DevPortableRoot,
    [string]$ExecutablePath,
    [string]$OutputRoot,

    [ValidateRange(1, 30)]
    [int]$NetworkObservationSeconds = 3,

    [ValidateRange(100, 5000)]
    [int]$NetworkPollMilliseconds = 250,

    [switch]$KeepSandbox
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'PortableValidation.Common.ps1')

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($DevPortableRoot)) {
    $DevPortableRoot = Join-Path $repositoryRoot 'artifacts\dev-portable'
}
else {
    $DevPortableRoot = [System.IO.Path]::GetFullPath($DevPortableRoot)
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repositoryRoot 'artifacts\portable-validation'
}
else {
    $OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
}

$startedAt = [DateTimeOffset]::UtcNow
$runName = $startedAt.ToString('yyyyMMddTHHmmssfffZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$runRoot = Join-Path $OutputRoot $runName
$reportPath = Join-Path $runRoot 'portable-clean-profile-report.json'
$sandboxRoot = Join-Path $runRoot 'sandbox'
$gates = [System.Collections.Generic.List[object]]::new()
$scenarios = [System.Collections.Generic.List[object]]::new()
$harnessErrors = [System.Collections.Generic.List[string]]::new()
$allTraceErrors = [System.Collections.Generic.List[object]]::new()
$allAllowedWrites = [System.Collections.Generic.List[object]]::new()
$allExternalWriteWarnings = [System.Collections.Generic.List[object]]::new()
$allWriteFailures = [System.Collections.Generic.List[object]]::new()
$activeProcess = $null
$activeTrace = $null
$readOnlyAclState = $null
$sourceExecutable = $null
$sourceBuildDirectory = $null
$portableRoot = $null
$registryPath = 'Registry::HKEY_CURRENT_USER\Software\GraphAutoReader'
$registryBefore = $null
$registryAfter = $null
$networkObservation = $null

function Add-Gate {
    param(
        [Parameter(Mandatory)][string]$Id,
        [Parameter(Mandatory)][bool]$Passed,
        [Parameter(Mandatory)][string]$Evidence
    )

    $gates.Add([ordered]@{
            id = $Id
            status = if ($Passed) { 'PASS' } else { 'FAIL' }
            evidence = $Evidence
        })
}

function Get-ExpectedMutableDirectories {
    param([Parameter(Mandatory)][string]$DataRoot)

    return @('Settings', 'Cache', 'Logs', 'Autosave', 'Recovery') |
        ForEach-Object { Join-Path $DataRoot $_ }
}

function Test-ExpectedMutableDirectories {
    param([Parameter(Mandatory)][string]$DataRoot)

    return @((Get-ExpectedMutableDirectories -DataRoot $DataRoot) |
            Where-Object { -not (Test-Path -LiteralPath $_ -PathType Container) }).Count -eq 0
}

function Get-ExactApplicationRootSnapshots {
    param(
        [Parameter(Mandatory)]
        [string[]]$Roots
    )

    return @($Roots | ForEach-Object {
            Get-PvApplicationRootSnapshot -Path $_
        })
}

function Add-ExactApplicationRootSnapshotEvents {
    param(
        [Parameter(Mandatory)]
        [object]$TraceOutcome,

        [Parameter(Mandatory)]
        [object[]]$BeforeSnapshots
    )

    $snapshotEvents = [System.Collections.Generic.List[object]]::new()
    foreach ($before in $BeforeSnapshots) {
        $after = Get-PvApplicationRootSnapshot -Path ([string]$before.root)
        foreach ($event in Compare-PvApplicationRootSnapshot -Before $before -After $after) {
            $snapshotEvents.Add($event)
        }
    }

    return [pscustomobject]@{
        Events = @($TraceOutcome.Events) + @($snapshotEvents)
        Errors = @($TraceOutcome.Errors)
    }
}

function Add-TraceOutcome {
    param(
        [Parameter(Mandatory)][string]$ScenarioId,
        [Parameter(Mandatory)][object]$TraceOutcome,
        [Parameter(Mandatory)][string]$ConfiguredDataRoot,
        [Parameter(Mandatory)][AllowEmptyCollection()][string[]]$ApplicationOwnedExternalRoots,
        [Parameter(Mandatory)][object]$ProcessEvidence,
        [AllowEmptyCollection()][string[]]$ExternalSystemRoots = @(),
        [AllowEmptyCollection()][string[]]$UserSelectedWritableRoots = @()
    )

    foreach ($traceError in $TraceOutcome.Errors) {
        $allTraceErrors.Add([ordered]@{
                scenario = $ScenarioId
                watcherRoot = $traceError.watcherRoot
                error = $traceError.error
            })
    }
    $classification = Classify-PvWriteEvents `
            -Events @($TraceOutcome.Events) `
            -ConfiguredDataRoot $ConfiguredDataRoot `
            -ApplicationOwnedExternalRoots $ApplicationOwnedExternalRoots `
            -ApplicationProcessEvidence $ProcessEvidence `
            -ExternalSystemRoots $ExternalSystemRoots `
            -UserSelectedWritableRoots $UserSelectedWritableRoots
    foreach ($record in $classification.Allowed) {
        $allAllowedWrites.Add([ordered]@{
                scenario = $ScenarioId
                purpose = $record.purpose
                responsibleProcess = $record.responsibleProcess
                responsibleComponent = $record.responsibleComponent
                evidence = $record.evidence
                event = $record.event
            })
    }
    foreach ($record in $classification.Warnings) {
        $allExternalWriteWarnings.Add([ordered]@{
                scenario = $ScenarioId
                purpose = $record.purpose
                responsibleProcess = $record.responsibleProcess
                responsibleComponent = $record.responsibleComponent
                evidence = $record.evidence
                event = $record.event
            })
    }
    foreach ($record in $classification.Failures) {
        $allWriteFailures.Add([ordered]@{
                scenario = $ScenarioId
                ownership = $record.ownership
                purpose = $record.purpose
                responsibleProcess = $record.responsibleProcess
                responsibleComponent = $record.responsibleComponent
                evidence = $record.evidence
                event = $record.event
            })
    }

    return [ordered]@{
        eventCount = @($TraceOutcome.Events).Count
        watcherErrorCount = @($TraceOutcome.Errors).Count
        allowedEventCount = @($classification.Allowed).Count
        externalWarningCount = @($classification.Warnings).Count
        failureCount = @($classification.Failures).Count
        exactRootSnapshotEventCount = @($TraceOutcome.Events | Where-Object {
                [string]$_.source -eq 'exact-root-before-after-snapshot'
            }).Count
        configuredDataRoot = [System.IO.Path]::GetFullPath($ConfiguredDataRoot)
        processEvidence = $ProcessEvidence
        classifiedEvents = @($classification.ClassifiedEvents)
    }
}

function Get-IsolatedEnvironment {
    param(
        [Parameter(Mandatory)][string]$ProfileRoot,
        [AllowNull()][string]$SharedDataRoot
    )

    $localAppData = Join-Path $ProfileRoot 'AppData\Local'
    $roamingAppData = Join-Path $ProfileRoot 'AppData\Roaming'
    $temporaryRoot = Join-Path $ProfileRoot 'Temp'
    foreach ($directory in @($ProfileRoot, $localAppData, $roamingAppData, $temporaryRoot)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }

    return @{
        GRAPHREADER_RUNTIME_MODE = 'ManualPreview'
        GRAPHREADER_DEV_PORTABLE_DATA_ROOT = $SharedDataRoot
        USERPROFILE = $ProfileRoot
        LOCALAPPDATA = $localAppData
        APPDATA = $roamingAppData
        TEMP = $temporaryRoot
        TMP = $temporaryRoot
    }
}

try {
    New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $sandboxRoot -Force | Out-Null

    if ([string]::IsNullOrWhiteSpace($ExecutablePath)) {
        $latestPath = Join-Path $DevPortableRoot 'latest.json'
        if (-not (Test-Path -LiteralPath $latestPath -PathType Leaf)) {
            throw "The development portable metadata is missing: $latestPath"
        }
        $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
        if ([string]::IsNullOrWhiteSpace([string]$latest.executable)) {
            throw "The development portable metadata does not name an executable: $latestPath"
        }
        $sourceExecutable = [System.IO.Path]::GetFullPath(
            (Join-Path $DevPortableRoot ([string]$latest.executable)))
        if (-not (Test-PvPathUnderRoot -Path $sourceExecutable -Root $DevPortableRoot)) {
            throw 'latest.json points outside the development portable root.'
        }
    }
    else {
        $sourceExecutable = [System.IO.Path]::GetFullPath($ExecutablePath)
    }

    if (-not (Test-Path -LiteralPath $sourceExecutable -PathType Leaf)) {
        throw "The portable executable is missing: $sourceExecutable"
    }
    $sourceBuildDirectory = Split-Path -Parent $sourceExecutable
    foreach ($requiredFile in @('portable.mode', 'GraphReader.App.runtimeconfig.json')) {
        $requiredPath = Join-Path $sourceBuildDirectory $requiredFile
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "The portable build is missing ${requiredFile}: $requiredPath"
        }
    }

    $unicodeSegment = -join ([char[]](0xD55C, 0xAE00))
    $portableRoot = Join-Path $sandboxRoot "Portable validation with spaces $unicodeSegment\Graph Auto Reader"
    New-Item -ItemType Directory -Path $portableRoot -Force | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $sourceBuildDirectory -Force) {
        Copy-Item -LiteralPath $item.FullName -Destination $portableRoot -Recurse
    }
    $scenarioExecutable = Join-Path $portableRoot (Split-Path -Leaf $sourceExecutable)
    if (-not (Test-Path -LiteralPath $scenarioExecutable -PathType Leaf)) {
        throw "The copied validation executable is missing: $scenarioExecutable"
    }

    $containsSpace = $portableRoot.Contains(' ')
    $containsUnicode = $portableRoot.ToCharArray().Where({ [int]$_ -gt 127 }).Count -gt 0
    Add-Gate -Id 'path-with-spaces-and-unicode' `
        -Passed ($containsSpace -and $containsUnicode) `
        -Evidence "Validation executable copied to '$scenarioExecutable'."

    $profileRoot = Join-Path $sandboxRoot 'isolated user profile'
    $actualLocalDataRoot = Join-Path (
        [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)) 'GraphAutoReader'
    $actualRoamingDataRoot = Join-Path (
        [Environment]::GetFolderPath([Environment+SpecialFolder]::ApplicationData)) 'GraphAutoReader'
    $isolatedInstalledRoot = Join-Path $profileRoot 'AppData\Local\GraphAutoReader'
    $isolatedRoamingRoot = Join-Path $profileRoot 'AppData\Roaming\GraphAutoReader'
    $applicationOwnedExternalRoots = @(
        $actualLocalDataRoot,
        $actualRoamingDataRoot,
        $isolatedInstalledRoot,
        $isolatedRoamingRoot)
    $actualApplicationRoots = @($actualLocalDataRoot, $actualRoamingDataRoot)
    $externalSystemRoots = @(
        (Join-Path $profileRoot 'Temp'),
        (Join-Path $profileRoot 'AppData\Local'),
        [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()),
        [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)) |
        Sort-Object -Unique
    $sharedDataRoot = Join-Path $sandboxRoot 'shared preview Data'
    $localPortableDataRoot = Join-Path $portableRoot 'Data'
    if (Test-Path -LiteralPath $localPortableDataRoot) {
        Remove-PvSandboxTree -Path $localPortableDataRoot -SandboxRoot $sandboxRoot
    }

    $registryBefore = Test-Path -LiteralPath $registryPath

    # Scenario 1: a live WPF process uses the development shared Data override.
    New-Item -ItemType Directory -Path $sharedDataRoot -Force | Out-Null
    $sharedEnvironment = Get-IsolatedEnvironment `
        -ProfileRoot $profileRoot `
        -SharedDataRoot $sharedDataRoot
    $sharedWatchRoots = @($portableRoot, $sharedDataRoot, $profileRoot)
    $sharedActualSnapshots = Get-ExactApplicationRootSnapshots -Roots $actualApplicationRoots
    $activeTrace = Start-PvWriteTrace -Roots $sharedWatchRoots
    $activeProcess = Start-PvIsolatedProcess `
        -ExecutablePath $scenarioExecutable `
        -WorkingDirectory $portableRoot `
        -Environment $sharedEnvironment `
        -WindowStyle Minimized
    $windowElement = Wait-PvAutomationElement -Process $activeProcess -TimeoutMilliseconds 15000
    $sharedProcessEvidence = Get-PvProcessComponentEvidence -Process $activeProcess
    $networkObservation = Observe-PvProcessNetwork `
        -Process $activeProcess `
        -ObservationSeconds $NetworkObservationSeconds `
        -PollMilliseconds $NetworkPollMilliseconds
    $sharedStop = Stop-PvProcess -Process $activeProcess
    $activeProcess = $null
    $sharedTraceOutcome = Stop-PvWriteTrace -Trace $activeTrace
    $activeTrace = $null
    $sharedTraceOutcome = Add-ExactApplicationRootSnapshotEvents `
        -TraceOutcome $sharedTraceOutcome `
        -BeforeSnapshots $sharedActualSnapshots
    $sharedTrace = Add-TraceOutcome `
        -ScenarioId 'shared-preview-data-root' `
        -TraceOutcome $sharedTraceOutcome `
        -ConfiguredDataRoot $sharedDataRoot `
        -ApplicationOwnedExternalRoots $applicationOwnedExternalRoots `
        -ProcessEvidence $sharedProcessEvidence `
        -ExternalSystemRoots $externalSystemRoots
    $sharedDirectoriesReady = Test-ExpectedMutableDirectories -DataRoot $sharedDataRoot
    $sharedLocalDataAbsent = -not (Test-Path -LiteralPath $localPortableDataRoot)
    $sharedPassed =
        $null -ne $windowElement -and
        $sharedDirectoriesReady -and
        $sharedLocalDataAbsent
    $windowName = if ($null -eq $windowElement) { $null } else { [string]$windowElement.Name }
    Add-Gate -Id 'shared-preview-data-root' `
        -Passed $sharedPassed `
        -Evidence "Window='$windowName'; expected shared directories=$sharedDirectoriesReady; local Data absent=$sharedLocalDataAbsent; traced events=$($sharedTrace.eventCount)."
    $scenarios.Add([ordered]@{
            id = 'shared-preview-data-root'
            executable = $scenarioExecutable
            dataRoot = $sharedDataRoot
            processExitCode = $sharedStop.ExitCode
            forcedTerminationAfterObservation = $sharedStop.Forced
            window = $windowElement
            expectedDirectoriesReady = $sharedDirectoriesReady
            localDataAbsent = $sharedLocalDataAbsent
            writeTrace = $sharedTrace
        })

    $networkPassed =
        $networkObservation.Succeeded -and
        $networkObservation.SampleCount -gt 0 -and
        @($networkObservation.Tcp).Count -eq 0 -and
        @($networkObservation.Udp).Count -eq 0
    Add-Gate -Id 'offline-no-network-observation' `
        -Passed $networkPassed `
        -Evidence "Network remained enabled; $($networkObservation.SampleCount) process-owned TCP/UDP table samples observed $(@($networkObservation.Tcp).Count) TCP and $(@($networkObservation.Udp).Count) UDP endpoints during a ${NetworkObservationSeconds}s requested window."

    # Scenario 2: the same portable runs without the development override and uses .\Data.
    if (Test-Path -LiteralPath $localPortableDataRoot) {
        Remove-PvSandboxTree -Path $localPortableDataRoot -SandboxRoot $sandboxRoot
    }
    $normalEnvironment = Get-IsolatedEnvironment -ProfileRoot $profileRoot -SharedDataRoot $null
    $normalWatchRoots = @($portableRoot, $profileRoot)
    $normalActualSnapshots = Get-ExactApplicationRootSnapshots -Roots $actualApplicationRoots
    $activeTrace = Start-PvWriteTrace -Roots $normalWatchRoots
    $activeProcess = Start-PvIsolatedProcess `
        -ExecutablePath $scenarioExecutable `
        -WorkingDirectory $portableRoot `
        -ArgumentList @('--portable-smoke') `
        -Environment $normalEnvironment `
        -WindowStyle Hidden
    $normalProcessEvidence = Get-PvProcessComponentEvidence -Process $activeProcess
    $normalExitCode = Wait-PvProcessExit -Process $activeProcess -TimeoutMilliseconds 15000
    $activeProcess = $null
    $normalTraceOutcome = Stop-PvWriteTrace -Trace $activeTrace
    $activeTrace = $null
    $normalTraceOutcome = Add-ExactApplicationRootSnapshotEvents `
        -TraceOutcome $normalTraceOutcome `
        -BeforeSnapshots $normalActualSnapshots
    $normalTrace = Add-TraceOutcome `
        -ScenarioId 'normal-portable-data-root' `
        -TraceOutcome $normalTraceOutcome `
        -ConfiguredDataRoot $localPortableDataRoot `
        -ApplicationOwnedExternalRoots $applicationOwnedExternalRoots `
        -ProcessEvidence $normalProcessEvidence `
        -ExternalSystemRoots $externalSystemRoots
    $normalDirectoriesReady = Test-ExpectedMutableDirectories -DataRoot $localPortableDataRoot
    $isolatedInstalledRootAbsent = -not (Test-Path -LiteralPath $isolatedInstalledRoot)
    $normalPassed =
        $normalExitCode -eq 0 -and
        $normalDirectoriesReady -and
        $isolatedInstalledRootAbsent
    Add-Gate -Id 'normal-portable-dot-data-root' `
        -Passed $normalPassed `
        -Evidence "Exit=$normalExitCode; expected .\\Data directories=$normalDirectoriesReady; isolated LocalAppData root absent=$isolatedInstalledRootAbsent; traced events=$($normalTrace.eventCount)."
    $scenarios.Add([ordered]@{
            id = 'normal-portable-data-root'
            executable = $scenarioExecutable
            dataRoot = $localPortableDataRoot
            processExitCode = $normalExitCode
            expectedDirectoriesReady = $normalDirectoriesReady
            isolatedInstalledRootAbsent = $isolatedInstalledRootAbsent
            writeTrace = $normalTrace
        })

    # Scenario 3: deny writes to .\Data and observe the live corrective UI text.
    if (Test-Path -LiteralPath $localPortableDataRoot) {
        Remove-PvSandboxTree -Path $localPortableDataRoot -SandboxRoot $sandboxRoot
    }
    New-Item -ItemType Directory -Path $localPortableDataRoot | Out-Null
    $readOnlyAclState = Set-PvDirectoryDenyWrite -Path $localPortableDataRoot
    $writeWasDenied = $false
    try {
        [System.IO.File]::WriteAllText(
            (Join-Path $localPortableDataRoot 'harness-write-should-fail.tmp'),
            'denied')
    }
    catch [UnauthorizedAccessException] {
        $writeWasDenied = $true
    }
    $readOnlyActualSnapshots = Get-ExactApplicationRootSnapshots -Roots $actualApplicationRoots
    $activeTrace = Start-PvWriteTrace -Roots @($portableRoot, $profileRoot)
    $activeProcess = Start-PvIsolatedProcess `
        -ExecutablePath $scenarioExecutable `
        -WorkingDirectory $portableRoot `
        -Environment $normalEnvironment `
        -WindowStyle Minimized
    $statusElement = Wait-PvAutomationElement `
        -Process $activeProcess `
        -AutomationId 'Workflow.Status' `
        -TimeoutMilliseconds 15000
    $readOnlyProcessEvidence = Get-PvProcessComponentEvidence -Process $activeProcess
    $readOnlyStop = Stop-PvProcess -Process $activeProcess
    $activeProcess = $null
    $readOnlyTraceOutcome = Stop-PvWriteTrace -Trace $activeTrace
    $activeTrace = $null
    $readOnlyTraceOutcome = Add-ExactApplicationRootSnapshotEvents `
        -TraceOutcome $readOnlyTraceOutcome `
        -BeforeSnapshots $readOnlyActualSnapshots
    $readOnlyTrace = Add-TraceOutcome `
        -ScenarioId 'read-only-portable-diagnostic' `
        -TraceOutcome $readOnlyTraceOutcome `
        -ConfiguredDataRoot $localPortableDataRoot `
        -ApplicationOwnedExternalRoots $applicationOwnedExternalRoots `
        -ProcessEvidence $readOnlyProcessEvidence `
        -ExternalSystemRoots $externalSystemRoots
    $expectedDiagnostic = 'This portable folder is read-only. Move it to a writable folder and try again.'
    $diagnosticObserved =
        $null -ne $statusElement -and
        [string]::Equals($statusElement.Name, $expectedDiagnostic, [StringComparison]::Ordinal)
    $observedStatusText = if ($null -eq $statusElement) { $null } else { [string]$statusElement.Name }
    $readOnlyPassed =
        $writeWasDenied -and
        $diagnosticObserved
    Add-Gate -Id 'read-only-folder-diagnostic' `
        -Passed $readOnlyPassed `
        -Evidence "Deny-write ACL effective=$writeWasDenied; UI diagnostic observed=$diagnosticObserved; text='$observedStatusText'."
    $scenarios.Add([ordered]@{
            id = 'read-only-portable-diagnostic'
            executable = $scenarioExecutable
            dataRoot = $localPortableDataRoot
            denyWriteAclEffective = $writeWasDenied
            expectedDiagnostic = $expectedDiagnostic
            observedStatus = $statusElement
            forcedTerminationAfterObservation = $readOnlyStop.Forced
            processExitCode = $readOnlyStop.ExitCode
            writeTrace = $readOnlyTrace
        })
    Restore-PvDirectoryAcl -State $readOnlyAclState
    $readOnlyAclState = $null

    $registryAfter = Test-Path -LiteralPath $registryPath
    $registryPassed = -not $registryBefore -and -not $registryAfter -and $sharedPassed -and $normalPassed
    Add-Gate -Id 'no-registry-configuration-dependency-observation' `
        -Passed $registryPassed `
        -Evidence "Application configuration key '$registryPath' existed before=$registryBefore and after=$registryAfter; both shared and normal portable startup scenarios succeeded=$($sharedPassed -and $normalPassed)."

    $fileTracePassed = $allTraceErrors.Count -eq 0 -and $allWriteFailures.Count -eq 0
    Add-Gate -Id 'file-system-write-trace' `
        -Passed $fileTracePassed `
        -Evidence "Purpose-aware classification recorded $($allAllowedWrites.Count) allowed Graph Auto Reader Data events, $($allExternalWriteWarnings.Count) attributed external cache warnings, $($allWriteFailures.Count) application-owned or unattributed failures, and $($allTraceErrors.Count) watcher errors."
}
catch {
    $harnessErrors.Add($_.Exception.Message)
    Add-Gate -Id 'harness-completion' -Passed $false -Evidence $_.Exception.Message
}
finally {
    if ($null -ne $activeProcess) {
        try {
            [void](Stop-PvProcess -Process $activeProcess)
        }
        catch {
            $harnessErrors.Add("Failed to stop process $($activeProcess.Id): $($_.Exception.Message)")
        }
    }
    if ($null -ne $activeTrace) {
        try {
            [void](Stop-PvWriteTrace -Trace $activeTrace -DrainMilliseconds 0)
        }
        catch {
            $harnessErrors.Add("Failed to stop a write trace: $($_.Exception.Message)")
        }
    }
    if ($null -ne $readOnlyAclState) {
        try {
            Restore-PvDirectoryAcl -State $readOnlyAclState
        }
        catch {
            $harnessErrors.Add("Failed to restore the read-only test ACL: $($_.Exception.Message)")
        }
    }
}

$allPassed = Test-PvValidationGates -Gates @($gates) -HarnessErrors @($harnessErrors)
$report = [ordered]@{
    schemaVersion = 1
    status = if ($allPassed) { 'PASS' } else { 'FAIL' }
    startedAtUtc = $startedAt.ToString('O')
    completedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
    evidenceScope = 'local isolated-profile simulation on the current Windows host'
    cleanVmEvidence = $false
    source = [ordered]@{
        executable = $sourceExecutable
        buildDirectory = $sourceBuildDirectory
    }
    validationCopy = $portableRoot
    registryObservation = [ordered]@{
        applicationConfigurationKey = $registryPath
        existedBefore = $registryBefore
        existedAfter = $registryAfter
        note = 'The WPF system-theme provider may read Windows theme settings. This gate observes absence of an application configuration key, not zero registry reads.'
    }
    networkObservation = $networkObservation
    gates = @($gates)
    scenarios = @($scenarios)
    fileSystemTrace = [ordered]@{
        method = 'FileSystemWatcher plus fail-closed exact-root before/after snapshots, destination-purpose classification, and live GraphReader.App process/module evidence'
        policy = 'Graph Auto Reader-owned persistence outside configured Data or a user-selected writable root fails. Attributed Windows, WPF, .NET, or GPU cache activity is retained as a warning.'
        watcherErrors = @($allTraceErrors)
        allowedApplicationWrites = @($allAllowedWrites)
        externalComponentWarnings = @($allExternalWriteWarnings)
        writeFailures = @($allWriteFailures)
    }
    harnessErrors = @($harnessErrors)
    limitations = @(
        'This is not clean-Windows-user or clean-VM evidence.',
        'Network adapters remained enabled. Process-owned TCP and UDP endpoint polling is observational and can miss a connection opened and closed between samples.',
        'FileSystemWatcher does not emit a process ID. External warnings combine the active GraphReader.App process identity, its loaded component modules, destination purpose, and cache-path evidence; unknown mutations fail closed.',
        'The exact real LocalAppData and RoamingAppData GraphAutoReader roots are SHA-256 and metadata snapshotted before and after every scenario so initially absent root creation cannot escape observation.',
        'Registry observation proves successful startup while the application-specific configuration key is absent. It is not a complete registry access trace.',
        'The harness exercises startup and path initialization. It does not prove import, save, autosave, recovery, or export writes.'
    )
}

try {
    Write-PvJsonFile -Path $reportPath -Value $report
}
catch {
    Write-Error "Portable validation finished but the report could not be written: $($_.Exception.Message)"
    exit 1
}

if (-not $KeepSandbox.IsPresent -and (Test-Path -LiteralPath $sandboxRoot)) {
    try {
        Remove-PvSandboxTree -Path $sandboxRoot -SandboxRoot $runRoot
    }
    catch {
        Write-Warning "The validation sandbox could not be removed: $($_.Exception.Message)"
    }
}

Write-Host "Portable clean-profile validation: $($report.status)"
Write-Host "Report: $reportPath"
if (-not $allPassed) {
    Write-Error 'One or more portable validation gates failed. See the JSON report.'
    exit 1
}
