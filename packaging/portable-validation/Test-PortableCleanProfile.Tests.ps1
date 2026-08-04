# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$commonScript = Join-Path $PSScriptRoot 'PortableValidation.Common.ps1'
$validationScript = Join-Path $PSScriptRoot 'Test-PortableCleanProfile.ps1'
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'GraphReader-PortableValidationTests-' + [Guid]::NewGuid().ToString('N'))
$passed = 0
$activeProcess = $null
$aclState = $null

function Assert-True {
    param(
        [Parameter(Mandatory)][bool]$Condition,
        [Parameter(Mandatory)][string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

try {
    New-Item -ItemType Directory -Path $testRoot | Out-Null

    foreach ($script in @($commonScript, $validationScript)) {
        $tokens = $null
        $errors = $null
        [System.Management.Automation.Language.Parser]::ParseFile(
            $script,
            [ref]$tokens,
            [ref]$errors) | Out-Null
        Assert-True ($errors.Count -eq 0) `
            "PowerShell parser errors in ${script}: $($errors -join ' | ')"
    }
    $passed++

    . $commonScript
    $containmentRoot = Join-Path $testRoot 'containment'
    $contained = Join-Path $containmentRoot 'child\file.txt'
    $prefixSibling = $containmentRoot + '-sibling\file.txt'
    Assert-True (Test-PvPathUnderRoot -Path $contained -Root $containmentRoot) `
        'Path containment rejected a child path.'
    Assert-True (-not (Test-PvPathUnderRoot -Path $prefixSibling -Root $containmentRoot)) `
        'Path containment accepted a prefix-sibling path.'
    Assert-True (Test-PvPathUnderRoot -Path $containmentRoot -Root $containmentRoot -AllowEqual) `
        'Path containment rejected an equal root when AllowEqual was requested.'
    $passed++

    $traceRoot = Join-Path $testRoot 'write trace'
    $configuredDataRoot = Join-Path $traceRoot 'portable\Data'
    $applicationExternalRoot = Join-Path $testRoot 'application external\AppData\Local\GraphAutoReader'
    New-Item -ItemType Directory -Path $configuredDataRoot -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $applicationExternalRoot 'Settings') -Force | Out-Null
    $testApplicationRoot = Join-Path $traceRoot 'portable'
    $cultureDirectory = Join-Path $testApplicationRoot 'zh-Hans'
    New-Item -ItemType Directory -Path $cultureDirectory -Force | Out-Null
    $processEvidence = [pscustomobject]@{
        processId = 4242
        processName = 'GraphReader.App'
        executablePath = Join-Path $testApplicationRoot 'GraphReader.App.exe'
        modules = @(
            [pscustomobject]@{ moduleName = 'atidxx64.dll' },
            [pscustomobject]@{ moduleName = 'PresentationCore.dll' },
            [pscustomobject]@{ moduleName = 'wpfgfx_cor3.dll' },
            [pscustomobject]@{ moduleName = 'coreclr.dll' })
        moduleCollectionError = $null
    }

    $trace = Start-PvWriteTrace -Roots @($traceRoot)
    $traceFile = Join-Path $configuredDataRoot 'Settings\probe.tmp'
    New-Item -ItemType Directory -Path (Split-Path -Parent $traceFile) -Force | Out-Null
    [System.IO.File]::WriteAllText($traceFile, 'first')
    [System.IO.File]::AppendAllText($traceFile, 'second')
    Remove-Item -LiteralPath $traceFile -Force
    $traceOutcome = Stop-PvWriteTrace -Trace $trace -DrainMilliseconds 600
    Assert-True (@($traceOutcome.Events).Count -gt 0) `
        'FileSystemWatcher did not record a transient create/write/delete sequence.'
    Assert-True (@($traceOutcome.Errors).Count -eq 0) `
        'FileSystemWatcher reported an error during the self-test.'
    $allowedClassification = Classify-PvWriteEvents `
            -Events @($traceOutcome.Events) `
            -ConfiguredDataRoot $configuredDataRoot `
            -ApplicationOwnedExternalRoots @($applicationExternalRoot) `
            -ApplicationProcessEvidence $processEvidence
    $allowedFailureSummary = @($allowedClassification.Failures | ForEach-Object {
            "$($_.event.changeType):$($_.event.path):$($_.evidence)"
        }) -join ' | '
    Assert-True (@($allowedClassification.Failures).Count -eq 0) `
        "A write under configured portable Data failed classification: $allowedFailureSummary"
    Assert-True (@($allowedClassification.Allowed).Count -gt 0) `
        'Configured portable Data writes were not recorded as allowed application persistence.'
    $passed++

    $negativeTrace = Start-PvWriteTrace -Roots @($applicationExternalRoot)
    $negativePath = Join-Path $applicationExternalRoot 'Settings\settings.json'
    [System.IO.File]::WriteAllText($negativePath, '{"negative":true}')
    $negativeOutcome = Stop-PvWriteTrace -Trace $negativeTrace -DrainMilliseconds 600
    $negativeClassification = Classify-PvWriteEvents `
        -Events @($negativeOutcome.Events) `
        -ConfiguredDataRoot $configuredDataRoot `
        -ApplicationOwnedExternalRoots @($applicationExternalRoot) `
        -ApplicationProcessEvidence $processEvidence
    Assert-True (@($negativeOutcome.Events).Count -gt 0) `
        'The negative probe did not record the Graph Auto Reader LocalAppData write.'
    Assert-True (@($negativeClassification.Failures).Count -gt 0) `
        'A Graph Auto Reader settings write under LocalAppData did not fail classification.'
    Assert-True (@($negativeClassification.Failures | Where-Object {
                $_.ownership -eq 'graph-auto-reader' -and
                $_.purpose -eq 'Graph Auto Reader settings' -and
                $_.responsibleComponent -eq 'Graph Auto Reader'
            }).Count -gt 0) `
        'The LocalAppData negative probe lacked application ownership and purpose evidence.'
    $passed++

    $snapshotNegativeRoot = Join-Path $testRoot 'initially absent exact roots'
    $absentLocalRoot = Join-Path $snapshotNegativeRoot 'AppData\Local\GraphAutoReader'
    $absentRoamingRoot = Join-Path $snapshotNegativeRoot 'AppData\Roaming\GraphAutoReader'
    foreach ($case in @(
            [pscustomobject]@{ Name = 'LocalAppData'; Root = $absentLocalRoot },
            [pscustomobject]@{ Name = 'RoamingAppData'; Root = $absentRoamingRoot })) {
        $beforeSnapshot = Get-PvApplicationRootSnapshot -Path $case.Root
        Assert-True (-not $beforeSnapshot.exists) `
            "$($case.Name) negative root was not initially absent."
        $settingsPath = Join-Path $case.Root 'Settings\settings.json'
        New-Item -ItemType Directory -Path (Split-Path -Parent $settingsPath) -Force | Out-Null
        [System.IO.File]::WriteAllText($settingsPath, '{"negative":true}')
        $afterSnapshot = Get-PvApplicationRootSnapshot -Path $case.Root
        $snapshotEvents = @(Compare-PvApplicationRootSnapshot `
                -Before $beforeSnapshot `
                -After $afterSnapshot)
        Assert-True ($snapshotEvents.Count -gt 0) `
            "Initially absent $($case.Name) GraphAutoReader creation produced no snapshot events."
        Assert-True (@($snapshotEvents | Where-Object {
                    $_.source -eq 'exact-root-before-after-snapshot' -and
                    $_.changeType -eq 'Created' -and
                    (Test-PvPathUnderRoot -Path $_.path -Root $case.Root -AllowEqual)
                }).Count -gt 0) `
            "Initially absent $($case.Name) GraphAutoReader creation lacked exact-root snapshot evidence."
        $snapshotClassification = Classify-PvWriteEvents `
            -Events $snapshotEvents `
            -ConfiguredDataRoot $configuredDataRoot `
            -ApplicationOwnedExternalRoots @($absentLocalRoot, $absentRoamingRoot) `
            -ApplicationProcessEvidence $processEvidence
        Assert-True (@($snapshotClassification.Failures | Where-Object {
                    $_.ownership -eq 'graph-auto-reader' -and
                    $_.responsibleComponent -eq 'Graph Auto Reader'
                }).Count -gt 0) `
            "Initially absent $($case.Name) GraphAutoReader creation did not fail as application-owned persistence."
        Assert-True (@($snapshotClassification.Failures | Where-Object {
                    $_.purpose -eq 'Graph Auto Reader settings' -and
                    $_.event.path -eq $settingsPath
                }).Count -gt 0) `
            "Initially absent $($case.Name) settings write lacked purpose-specific failure evidence."
    }
    $passed++

    $externalProfileRoot = Join-Path $traceRoot 'isolated profile\AppData\Local'
    $externalSystemRoot = Join-Path $traceRoot 'isolated profile\Temp'
    $externalEvents = @(
        [pscustomobject]@{
            observedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
            watcherRoot = $externalProfileRoot
            changeType = 'Changed'
            path = $externalProfileRoot
            oldPath = $null
        },
        [pscustomobject]@{
            observedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
            watcherRoot = $externalProfileRoot
            changeType = 'Created'
            path = Join-Path $externalProfileRoot 'AMD\DX9Cache\driver-cache.parc'
            oldPath = $null
        },
        [pscustomobject]@{
            observedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
            watcherRoot = $testApplicationRoot
            changeType = 'Changed'
            path = $cultureDirectory
            oldPath = $null
        },
        [pscustomobject]@{
            observedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
            watcherRoot = $externalSystemRoot
            changeType = 'Created'
            path = Join-Path $externalSystemRoot '.net\GraphReader.App\bundle.tmp'
            oldPath = $null
        })
    $externalClassification = Classify-PvWriteEvents `
        -Events $externalEvents `
        -ConfiguredDataRoot $configuredDataRoot `
        -ApplicationOwnedExternalRoots @($applicationExternalRoot) `
        -ApplicationProcessEvidence $processEvidence `
        -ExternalSystemRoots @($externalSystemRoot)
    Assert-True (@($externalClassification.Failures).Count -eq 0) `
        'Attributed AMD cache activity was incorrectly treated as application persistence.'
    Assert-True (@($externalClassification.Warnings).Count -eq 4) `
        'Attributed AMD cache, parent metadata, WPF culture-directory, and .NET cache events were not retained as warnings.'
    Assert-True (@($externalClassification.Warnings | Where-Object {
                $_.responsibleProcess.processName -eq 'GraphReader.App' -and
                $_.responsibleComponent -eq 'AMD graphics driver' -and
                -not [string]::IsNullOrWhiteSpace([string]$_.evidence)
            }).Count -eq 2) `
        'External cache warnings lacked responsible process, component, or evidence.'
    Assert-True (@($externalClassification.Warnings | Where-Object {
                $_.responsibleProcess.processName -eq 'GraphReader.App' -and
                $_.responsibleComponent -eq 'Microsoft .NET/WPF resource manager' -and
                $_.purpose -match 'satellite-resource' -and
                -not [string]::IsNullOrWhiteSpace([string]$_.evidence)
            }).Count -eq 1) `
        'The WPF culture-directory observation lacked process, component, purpose, or evidence.'
    Assert-True (@($externalClassification.Warnings | Where-Object {
                $_.responsibleProcess.processName -eq 'GraphReader.App' -and
                $_.responsibleComponent -eq 'Microsoft .NET runtime' -and
                $_.purpose -match '\.NET runtime' -and
                -not [string]::IsNullOrWhiteSpace([string]$_.evidence)
            }).Count -eq 1) `
        'The .NET cache warning lacked process, component, purpose, or evidence.'
    $unattributedEvidence = [pscustomobject]@{
        processId = 4242
        processName = 'GraphReader.App'
        executablePath = 'C:\Portable\GraphReader.App.exe'
        modules = @()
        moduleCollectionError = $null
    }
    $unattributed = Classify-PvWriteEvents `
        -Events @($externalEvents[1]) `
        -ConfiguredDataRoot $configuredDataRoot `
        -ApplicationOwnedExternalRoots @($applicationExternalRoot) `
        -ApplicationProcessEvidence $unattributedEvidence
    Assert-True (@($unattributed.Failures).Count -eq 1) `
        'An external-looking cache event without component evidence did not fail closed.'
    $passed++

    $jsonPath = Join-Path $testRoot 'json\report.json'
    $unicodeText = -join ([char[]](0xD55C, 0xAE00))
    Write-PvJsonFile -Path $jsonPath -Value ([ordered]@{
            status = 'PASS'
            unicode = $unicodeText
        })
    $json = [System.IO.File]::ReadAllText(
        $jsonPath,
        [System.Text.UTF8Encoding]::new($false)) | ConvertFrom-Json
    Assert-True ($json.status -eq 'PASS' -and $json.unicode -eq $unicodeText) `
        'JSON evidence did not round-trip through UTF-8.'
    $passed++

    $readOnlyRoot = Join-Path $testRoot 'read only Data'
    New-Item -ItemType Directory -Path $readOnlyRoot -Force | Out-Null
    $aclState = Set-PvDirectoryDenyWrite -Path $readOnlyRoot
    $writeDenied = $false
    try {
        [System.IO.File]::WriteAllText((Join-Path $readOnlyRoot 'denied.tmp'), 'denied')
    }
    catch [UnauthorizedAccessException] {
        $writeDenied = $true
    }
    Assert-True $writeDenied 'The deny-write ACL did not block the current identity.'
    Restore-PvDirectoryAcl -State $aclState
    $aclState = $null
    $restoredPath = Join-Path $readOnlyRoot 'restored.tmp'
    [System.IO.File]::WriteAllText($restoredPath, 'restored')
    Assert-True (Test-Path -LiteralPath $restoredPath -PathType Leaf) `
        'The original directory ACL was not restored.'
    $passed++

    $activeProcess = Start-Process `
        -FilePath 'powershell.exe' `
        -ArgumentList @('-NoProfile', '-Command', 'Start-Sleep -Seconds 2') `
        -WindowStyle Hidden `
        -PassThru
    $network = Observe-PvProcessNetwork `
        -Process $activeProcess `
        -ObservationSeconds 1 `
        -PollMilliseconds 100
    Assert-True $network.Succeeded `
        "Process-owned network observation failed: $($network.Error)"
    Assert-True ($network.SampleCount -gt 0) `
        'Process-owned network observation took no samples.'
    [void](Stop-PvProcess -Process $activeProcess)
    $activeProcess = $null
    $passed++

    $passingGates = @(Get-PvRequiredGateIds | ForEach-Object {
            [pscustomobject]@{ id = $_; status = 'PASS' }
        })
    Assert-True (Test-PvValidationGates -Gates $passingGates -HarnessErrors @()) `
        'The complete required gate set did not pass.'
    Assert-True (-not (Test-PvValidationGates -Gates @($passingGates | Select-Object -Skip 1) -HarnessErrors @())) `
        'A missing required gate did not fail closed.'
    $failingGates = @($passingGates | ForEach-Object {
            [pscustomobject]@{ id = $_.id; status = if ($_.id -eq 'file-system-write-trace') { 'FAIL' } else { 'PASS' } }
        })
    Assert-True (-not (Test-PvValidationGates -Gates $failingGates -HarnessErrors @())) `
        'A failed required gate did not fail closed.'
    Assert-True (-not (Test-PvValidationGates -Gates $passingGates -HarnessErrors @('probe failure'))) `
        'A harness error did not fail closed.'
    $passed++

    Write-Host "Portable validation self-tests passed: $passed/$passed"
}
finally {
    if ($null -ne $activeProcess) {
        try {
            [void](Stop-PvProcess -Process $activeProcess)
        }
        catch {
        }
    }
    if ($null -ne $aclState) {
        try {
            Restore-PvDirectoryAcl -State $aclState
        }
        catch {
        }
    }
    if (Test-Path -LiteralPath $testRoot) {
        Remove-PvSandboxTree -Path $testRoot -SandboxRoot ([System.IO.Path]::GetTempPath())
    }
}
