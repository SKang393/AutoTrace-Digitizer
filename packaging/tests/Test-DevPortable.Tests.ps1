# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$buildScript = Join-Path $repositoryRoot 'packaging\Build-DevPortable.ps1'
$commonScript = Join-Path $repositoryRoot 'packaging\DevPortable.Common.ps1'
$watchScript = Join-Path $repositoryRoot 'packaging\Watch-DevPortable.ps1'
$launcherScript = Join-Path $repositoryRoot 'packaging\Run-Latest-DevPortable.ps1'
$hostExecutable = (Get-Process -Id $PID).Path
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'GraphReader-DevPortableTests-' + [Guid]::NewGuid().ToString('N'))
$dirtyMarker = Join-Path $repositoryRoot ('.dev-portable-test-' + [Guid]::NewGuid().ToString('N') + '.txt')
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

function Invoke-Git {
    param(
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string[]]$Arguments
    )

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & git -C $WorkingDirectory @Arguments 1>$null 2>$null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "git failed with exit code ${exitCode}: git -C $WorkingDirectory $($Arguments -join ' ')"
    }
}

try {
    New-Item -ItemType Directory -Path $testRoot | Out-Null

    foreach ($script in @($buildScript, $commonScript, $watchScript, $launcherScript)) {
        $tokens = $null
        $errors = $null
        [System.Management.Automation.Language.Parser]::ParseFile(
            $script,
            [ref]$tokens,
            [ref]$errors) | Out-Null
        Assert-True ($errors.Count -eq 0) "PowerShell parser errors in ${script}: $($errors -join ' | ')"
    }

    $atomicOutputRoot = Join-Path $testRoot 'atomic metadata'
    New-Item -ItemType Directory -Path $atomicOutputRoot | Out-Null
    . $commonScript
    $atomicPath = Join-Path $atomicOutputRoot 'atomic-self-test.json'
    Write-JsonFileAtomic -Path $atomicPath -Value ([ordered]@{ generation = 1 })
    Write-JsonFileAtomic -Path $atomicPath -Value ([ordered]@{ generation = 2 })
    $atomicValue = Get-Content -LiteralPath (Join-Path $atomicOutputRoot 'atomic-self-test.json') -Raw | ConvertFrom-Json
    Assert-True ($atomicValue.generation -eq 2) 'Atomic metadata replacement did not publish the second generation.'
    Assert-True (@(Get-ChildItem -LiteralPath $atomicOutputRoot -File | Where-Object { $_.Name -match '\.(tmp|bak)$' }).Count -eq 0) `
        'Atomic metadata replacement left temporary files behind.'
    $passed++

    $modelRoot = Join-Path $testRoot 'approved model discovery'
    $modelManifestRoot = Join-Path $modelRoot 'models\manifest'
    $modelPayloadRoot = Join-Path $modelRoot 'models\runtime'
    New-Item -ItemType Directory -Path $modelManifestRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $modelPayloadRoot -Force | Out-Null
    $singlePayload = Join-Path $modelPayloadRoot 'single.onnx'
    $parameterPayload = Join-Path $modelPayloadRoot 'multi.param'
    $binaryPayload = Join-Path $modelPayloadRoot 'multi.bin'
    [System.IO.File]::WriteAllBytes($singlePayload, [byte[]](1, 2, 3, 4))
    [System.IO.File]::WriteAllBytes($parameterPayload, [byte[]](5, 6, 7))
    [System.IO.File]::WriteAllBytes($binaryPayload, [byte[]](8, 9, 10, 11))
    $singleHash = (Get-FileHash -LiteralPath $singlePayload -Algorithm SHA256).Hash.ToLowerInvariant()
    $parameterHash = (Get-FileHash -LiteralPath $parameterPayload -Algorithm SHA256).Hash.ToLowerInvariant()
    $binaryHash = (Get-FileHash -LiteralPath $binaryPayload -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-JsonFile -Path (Join-Path $modelManifestRoot 'approved-single.json') -Value ([ordered]@{
            model_id = 'approved-single'
            license = @{ reviewed = $true }
            commercial_use = $true
            redistribution = $true
            sha256 = $singleHash
            files = @('runtime/single.onnx')
            benchmarks = @(@{ status = 'pass'; release_eligible = $true })
        })
    Write-JsonFile -Path (Join-Path $modelManifestRoot 'approved-multi.json') -Value ([ordered]@{
            model_id = 'approved-multi'
            license = @{ reviewed = $true }
            commercial_use = $true
            redistribution = $true
            sha256 = ('0' * 64)
            files = @('runtime/multi.param', 'runtime/multi.bin')
            preprocessing = @{ model_payload_sha256 = [ordered]@{
                    'runtime/multi.param' = $parameterHash
                    'runtime/multi.bin' = $binaryHash
                } }
            benchmarks = @(@{ status = 'pass'; release_eligible = $true })
        })
    Write-JsonFile -Path (Join-Path $modelManifestRoot 'unapproved.json') -Value ([ordered]@{
            model_id = 'unapproved'
            license = @{ reviewed = $true }
            commercial_use = $true
            redistribution = $true
            sha256 = $singleHash
            files = @('runtime/single.onnx')
            benchmarks = @(@{ status = 'fail'; release_eligible = $false })
        })
    Write-JsonFile -Path (Join-Path $modelManifestRoot 'invalid-multi.json') -Value ([ordered]@{
            model_id = 'invalid-multi'
            license = @{ reviewed = $true }
            commercial_use = $true
            redistribution = $true
            sha256 = ('0' * 64)
            files = @('runtime/multi.param', 'runtime/multi.bin')
            preprocessing = @{ model_payload_sha256 = @{
                    'runtime/multi.param' = $parameterHash
                } }
            benchmarks = @(@{ status = 'pass'; release_eligible = $true })
        })
    foreach ($name in @('duplicate-a.json', 'duplicate-b.json')) {
        Write-JsonFile -Path (Join-Path $modelManifestRoot $name) -Value ([ordered]@{
                model_id = 'duplicate-approved'
                license = @{ reviewed = $true }
                commercial_use = $true
                redistribution = $true
                sha256 = $singleHash
                files = @('runtime/single.onnx')
                benchmarks = @(@{ status = 'pass'; release_eligible = $true })
            })
    }
    $modelDiagnostics = [System.Collections.Generic.List[string]]::new()
    $availableModels = @(Get-DevPortableApprovedModelIds `
            -RepositoryRoot $modelRoot `
            -Diagnostics $modelDiagnostics)
    Assert-True ($availableModels.Count -eq 2) `
        "Approved model discovery returned $($availableModels.Count) models instead of two."
    Assert-True ($availableModels -contains 'approved-single') `
        'Approved single-file model was not discovered.'
    Assert-True ($availableModels -contains 'approved-multi') `
        'Approved multi-file model was not discovered.'
    Assert-True ($availableModels -notcontains 'unapproved') `
        'A benchmark-unapproved model was reported as available.'
    Assert-True ($availableModels -notcontains 'invalid-multi') `
        'A multi-file model with an incomplete checksum map was reported as available.'
    Assert-True ($availableModels -notcontains 'duplicate-approved') `
        'A duplicated approved model ID was reported as available.'
    Assert-True ((@($modelDiagnostics) -join [Environment]::NewLine) -like '*invalid-multi.json*') `
        'Invalid multi-file model diagnostics were not retained.'
    Assert-True ((@($modelDiagnostics) -join [Environment]::NewLine) -like "*duplicate model ID 'duplicate-approved'*") `
        'Duplicated approved model ID diagnostics were not retained.'
    $passed++

    $watcherRoot = Join-Path $testRoot 'watcher exact commit'
    $watcherPackagingRoot = Join-Path $watcherRoot 'packaging'
    $watcherOutputRoot = Join-Path $watcherRoot 'artifacts\dev-portable'
    $watcherBuildLog = Join-Path $watcherRoot 'build-commits.txt'
    New-Item -ItemType Directory -Path $watcherPackagingRoot | Out-Null
    New-Item -ItemType Directory -Path $watcherOutputRoot | Out-Null
    Copy-Item -LiteralPath $watchScript -Destination $watcherPackagingRoot
    [System.IO.File]::WriteAllText(
        (Join-Path $watcherPackagingRoot 'Build-DevPortable.ps1'),
        @'
param([switch]$AllowDirty, [switch]$FastTestsOnly)
$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$commit = (& git -C $root rev-parse --verify HEAD).Trim()
[System.IO.File]::AppendAllText((Join-Path $root 'build-commits.txt'), $commit + [Environment]::NewLine)
if (Test-Path -LiteralPath (Join-Path $root 'delay-next-build') -PathType Leaf) {
    Remove-Item -LiteralPath (Join-Path $root 'delay-next-build') -Force
    Start-Sleep -Seconds 3
}
$outputRoot = Join-Path $root 'artifacts\dev-portable'
$buildRoot = Join-Path $outputRoot ('builds\0.0.0-test-' + $commit.Substring(0, 7))
New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
$executable = Join-Path $buildRoot 'GraphReader.App.exe'
[System.IO.File]::WriteAllText($executable, '')
$relativeExecutable = $executable.Substring($outputRoot.Length + 1).Replace('\', '/')
[System.IO.File]::WriteAllText(
    (Join-Path $outputRoot 'latest.json'),
    ('{"executable":"' + $relativeExecutable + '"}'))
'@)
    [System.IO.File]::WriteAllText((Join-Path $watcherRoot 'README.md'), 'watcher fixture')
    Invoke-Git -WorkingDirectory $watcherRoot -Arguments @('init', '--initial-branch=main')
    Invoke-Git -WorkingDirectory $watcherRoot -Arguments @('config', 'user.name', 'Graph Reader Test')
    Invoke-Git -WorkingDirectory $watcherRoot -Arguments @('config', 'user.email', 'graph-reader-test@example.invalid')
    Invoke-Git -WorkingDirectory $watcherRoot -Arguments @('add', '.')
    Invoke-Git -WorkingDirectory $watcherRoot -Arguments @('commit', '-m', 'Create watcher fixture')

    $watcherStdout = Join-Path $watcherRoot 'watcher.stdout.log'
    $watcherStderr = Join-Path $watcherRoot 'watcher.stderr.log'
    $watcherProcess = Start-Process -FilePath $hostExecutable -ArgumentList @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', ('"{0}"' -f (Join-Path $watcherPackagingRoot 'Watch-DevPortable.ps1')),
        '-NoLaunch') -RedirectStandardOutput $watcherStdout -RedirectStandardError $watcherStderr -PassThru -WindowStyle Hidden
    try {
        $startupDeadline = [DateTimeOffset]::UtcNow.AddSeconds(10)
        while ([DateTimeOffset]::UtcNow -lt $startupDeadline) {
            if ($watcherProcess.HasExited) {
                throw "Watcher exited during startup. stderr: $(Get-Content -LiteralPath $watcherStderr -Raw -ErrorAction SilentlyContinue)"
            }
            if ((Test-Path -LiteralPath $watcherStdout -PathType Leaf) -and
                ([string](Get-Content -LiteralPath $watcherStdout -Raw)) -like '*Press Ctrl+C to stop.*') {
                break
            }
            Start-Sleep -Milliseconds 100
        }
        Assert-True (([string](Get-Content -LiteralPath $watcherStdout -Raw)) -like '*Press Ctrl+C to stop.*') `
            'Watcher did not become ready for the exact-commit behavioral test.'

        Invoke-Git -WorkingDirectory $watcherRoot -Arguments @('commit', '--allow-empty', '-m', 'Move current branch ref')
        $expectedCommit = (& git -C $watcherRoot rev-parse --verify HEAD).Trim().ToLowerInvariant()
        $buildDeadline = [DateTimeOffset]::UtcNow.AddSeconds(12)
        while ([DateTimeOffset]::UtcNow -lt $buildDeadline -and
            -not (Test-Path -LiteralPath $watcherBuildLog -PathType Leaf)) {
            if ($watcherProcess.HasExited) {
                throw "Watcher exited before the ref-triggered build. stderr: $(Get-Content -LiteralPath $watcherStderr -Raw -ErrorAction SilentlyContinue)"
            }
            Start-Sleep -Milliseconds 100
        }
        Assert-True (Test-Path -LiteralPath $watcherBuildLog -PathType Leaf) `
            'Moving the current branch ref did not trigger a preview build.'
        Start-Sleep -Milliseconds 500
        $builtCommits = @(Get-Content -LiteralPath $watcherBuildLog | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        Assert-True ($builtCommits.Count -eq 1) `
            "A single ref move should queue one build, but recorded $($builtCommits.Count)."
        Assert-True ($builtCommits[0].Trim().ToLowerInvariant() -eq $expectedCommit) `
            "Watcher built commit $($builtCommits[0]) instead of exact HEAD $expectedCommit."
        Write-Host "Watcher ref trigger passed: commit=$expectedCommit builds=$($builtCommits.Count)"

        $sourceRoot = Join-Path $watcherRoot 'src'
        New-Item -ItemType Directory -Path $sourceRoot | Out-Null
        $debounceStart = [DateTimeOffset]::UtcNow
        [System.IO.File]::WriteAllText((Join-Path $sourceRoot 'Debounce.cs'), 'first')
        $debounceDeadline = $debounceStart.AddSeconds(12)
        do {
            $builtCommits = @(Get-Content -LiteralPath $watcherBuildLog | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
            if ($builtCommits.Count -ge 2) {
                break
            }
            Start-Sleep -Milliseconds 100
        } while ([DateTimeOffset]::UtcNow -lt $debounceDeadline)
        $debounceElapsed = [DateTimeOffset]::UtcNow - $debounceStart
        Assert-True ($builtCommits.Count -eq 2) `
            "A relevant source change should produce one debounced build; recorded $($builtCommits.Count)."
        Assert-True ($debounceElapsed.TotalSeconds -ge 1.5) `
            "Watcher built after $([Math]::Round($debounceElapsed.TotalSeconds, 2)) seconds; debounce must be at least 1.5 seconds."
        Write-Host "Watcher debounce passed: seconds=$([Math]::Round($debounceElapsed.TotalSeconds, 2)) builds=$($builtCommits.Count)"

        [System.IO.File]::WriteAllText((Join-Path $watcherRoot 'delay-next-build'), '')
        [System.IO.File]::WriteAllText((Join-Path $sourceRoot 'Queued.cs'), 'trigger delayed build')
        $delayedBuildDeadline = [DateTimeOffset]::UtcNow.AddSeconds(12)
        do {
            $builtCommits = @(Get-Content -LiteralPath $watcherBuildLog | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
            if ($builtCommits.Count -ge 3) {
                break
            }
            Start-Sleep -Milliseconds 100
        } while ([DateTimeOffset]::UtcNow -lt $delayedBuildDeadline)
        Assert-True ($builtCommits.Count -eq 3) 'The delayed build did not start.'

        [System.IO.File]::WriteAllText((Join-Path $sourceRoot 'DuringBuild.cs'), 'queue one follow-up')
        $queuedBuildDeadline = [DateTimeOffset]::UtcNow.AddSeconds(12)
        do {
            $builtCommits = @(Get-Content -LiteralPath $watcherBuildLog | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
            if ($builtCommits.Count -ge 4) {
                break
            }
            Start-Sleep -Milliseconds 100
        } while ([DateTimeOffset]::UtcNow -lt $queuedBuildDeadline)
        Assert-True ($builtCommits.Count -eq 4) `
            "A change during a build should queue one follow-up build; recorded $($builtCommits.Count)."
        Start-Sleep -Seconds 3
        $builtCommits = @(Get-Content -LiteralPath $watcherBuildLog | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        Assert-True ($builtCommits.Count -eq 4) `
            "Queued changes should collapse into one follow-up build; recorded $($builtCommits.Count)."
        Write-Host "Watcher queued-build passed: total-builds=$($builtCommits.Count)"
    }
    finally {
        if ($null -ne $watcherProcess -and -not $watcherProcess.HasExited) {
            $watcherProcess.Kill()
            $watcherProcess.WaitForExit()
        }
        if ($null -ne $watcherProcess) {
            $watcherProcess.Dispose()
        }
    }
    $passed++

    $failureRoot = Join-Path $testRoot 'prior latest preservation'
    New-Item -ItemType Directory -Path $failureRoot | Out-Null
    $latestPath = Join-Path $failureRoot 'latest.json'
    $expectedLatest = '{"sentinel":"prior-success"}'
    [System.IO.File]::WriteAllText($latestPath, $expectedLatest)
    [System.IO.File]::WriteAllText($dirtyMarker, 'force dirty-tree rejection')

    Invoke-ExpectedFailure @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', $buildScript,
        '-OutputRoot', $failureRoot)

    Assert-True ((Get-Content -LiteralPath $latestPath -Raw) -eq $expectedLatest) `
        'A failed dirty-tree build changed the prior latest.json.'
    $failurePath = Join-Path $failureRoot 'last-failure.json'
    Assert-True (Test-Path -LiteralPath $failurePath -PathType Leaf) `
        'A failed build did not produce last-failure.json.'
    $failure = Get-Content -LiteralPath $failurePath -Raw | ConvertFrom-Json
    Assert-True ($failure.priorLatestPreserved -eq $true) `
        'last-failure.json did not record preservation of the prior latest build.'
    $passed++

    $diagnosticsRoot = Join-Path $testRoot 'captured diagnostics'
    $fakeToolRoot = Join-Path $testRoot 'fake dotnet'
    New-Item -ItemType Directory -Path $diagnosticsRoot | Out-Null
    New-Item -ItemType Directory -Path $fakeToolRoot | Out-Null
    [System.IO.File]::WriteAllText(
        (Join-Path $fakeToolRoot 'dotnet.cmd'),
        "@echo off`r`necho DIAGNOSTIC_SENTINEL 1>&2`r`nexit /b 7`r`n")
    $previousProcessPath = [Environment]::GetEnvironmentVariable('PATH', 'Process')
    try {
        [Environment]::SetEnvironmentVariable(
            'PATH',
            $fakeToolRoot + [System.IO.Path]::PathSeparator + $previousProcessPath,
            'Process')
        Invoke-ExpectedFailure @(
            '-NoProfile',
            '-ExecutionPolicy', 'Bypass',
            '-File', $buildScript,
            '-OutputRoot', $diagnosticsRoot,
            '-AllowDirty',
            '-FastTestsOnly',
            '-SkipRestore')
    }
    finally {
        [Environment]::SetEnvironmentVariable('PATH', $previousProcessPath, 'Process')
    }
    $capturedFailure = Get-Content -LiteralPath (Join-Path $diagnosticsRoot 'last-failure.json') -Raw | ConvertFrom-Json
    Assert-True ($capturedFailure.command -eq 'dotnet test GraphReader.App.Tests') `
        'last-failure.json did not identify the failed command.'
    Assert-True ((@($capturedFailure.diagnostics) -join [Environment]::NewLine) -like '*DIAGNOSTIC_SENTINEL*') `
        'last-failure.json did not capture child-process diagnostics.'
    $passed++

    $launcherIsolationRoot = Join-Path $testRoot 'launcher enhancement isolation'
    $launcherIsolationBuild = Join-Path $launcherIsolationRoot 'builds\isolated'
    New-Item -ItemType Directory -Path $launcherIsolationBuild -Force | Out-Null
    Copy-Item -LiteralPath $launcherScript -Destination $launcherIsolationRoot
    $captureExecutable = Join-Path $launcherIsolationBuild 'capture.cmd'
    $captureOutput = Join-Path $launcherIsolationBuild 'enhancement-environment.txt'
    $startupImage = Join-Path $launcherIsolationRoot 'Chandler graph.png'
    [System.IO.File]::WriteAllBytes($startupImage, [byte[]](1, 2, 3))
    [System.IO.File]::WriteAllText(
        $captureExecutable,
        "@echo off`r`necho [%GRAPHREADER_REALESRGAN_RUNTIME_ROOT%] > `"%~dp0enhancement-environment.txt`"`r`necho [%GRAPHREADER_REALESRGAN_MANIFEST_PATH%] >> `"%~dp0enhancement-environment.txt`"`r`necho [%GRAPHREADER_PDFIUM_APPROVAL_PATH%] >> `"%~dp0enhancement-environment.txt`"`r`necho [%*] >> `"%~dp0enhancement-environment.txt`"`r`n")
    [System.IO.File]::WriteAllText((Join-Path $launcherIsolationBuild 'portable.mode'), '')
    $captureHash = (Get-FileHash -LiteralPath $captureExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-JsonFile -Path (Join-Path $launcherIsolationRoot 'latest.json') -Value ([ordered]@{
            schemaVersion = 2
            executable = 'builds/isolated/capture.cmd'
            executableSha256 = $captureHash
        })
    $previousRuntimeRoot = [Environment]::GetEnvironmentVariable(
        'GRAPHREADER_REALESRGAN_RUNTIME_ROOT',
        'Process')
    $previousManifestPath = [Environment]::GetEnvironmentVariable(
        'GRAPHREADER_REALESRGAN_MANIFEST_PATH',
        'Process')
    $previousPdfiumApprovalPath = [Environment]::GetEnvironmentVariable(
        'GRAPHREADER_PDFIUM_APPROVAL_PATH',
        'Process')
    try {
        [Environment]::SetEnvironmentVariable(
            'GRAPHREADER_REALESRGAN_RUNTIME_ROOT',
            'C:\stale-runtime',
            'Process')
        [Environment]::SetEnvironmentVariable(
            'GRAPHREADER_REALESRGAN_MANIFEST_PATH',
            'C:\stale-manifest.json',
            'Process')
        [Environment]::SetEnvironmentVariable(
            'GRAPHREADER_PDFIUM_APPROVAL_PATH',
            'C:\stale-pdfium-approval.json',
            'Process')
        & $hostExecutable -NoProfile -ExecutionPolicy Bypass -File `
            (Join-Path $launcherIsolationRoot 'Run-Latest-DevPortable.ps1') `
            -Wait -DisableLocalEnhancement -DisableLocalPdfium -ImagePath $startupImage | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Disabled-enhancement launcher exited with code $LASTEXITCODE."
        }
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            'GRAPHREADER_REALESRGAN_RUNTIME_ROOT',
            $previousRuntimeRoot,
            'Process')
        [Environment]::SetEnvironmentVariable(
            'GRAPHREADER_REALESRGAN_MANIFEST_PATH',
            $previousManifestPath,
            'Process')
        [Environment]::SetEnvironmentVariable(
            'GRAPHREADER_PDFIUM_APPROVAL_PATH',
            $previousPdfiumApprovalPath,
            'Process')
    }
    $capturedEnvironment = @(Get-Content -LiteralPath $captureOutput)
    Assert-True ($capturedEnvironment.Count -eq 4) `
        'Launcher did not capture all child environment values and startup arguments.'
    Assert-True ($capturedEnvironment[0].Trim() -eq '[]') `
        'Disabled-enhancement launcher leaked an inherited runtime root to the child.'
    Assert-True ($capturedEnvironment[1].Trim() -eq '[]') `
        'Disabled-enhancement launcher leaked an inherited manifest path to the child.'
    Assert-True ($capturedEnvironment[2].Trim() -eq '[]') `
        'Disabled-PDFium launcher leaked an inherited approval path to the child.'
    Assert-True ($capturedEnvironment[3].Contains('--open-image')) `
        'Launcher did not pass the explicit startup image argument.'
    Assert-True ($capturedEnvironment[3].Contains($startupImage)) `
        'Launcher did not preserve the startup image path containing spaces.'
    $passed++

    [System.IO.File]::AppendAllText($captureExecutable, "rem tampered`r`n")
    Invoke-ExpectedFailure @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', (Join-Path $launcherIsolationRoot 'Run-Latest-DevPortable.ps1'),
        '-Wait')
    $passed++

    $launcherRoot = Join-Path $testRoot 'launcher traversal'
    New-Item -ItemType Directory -Path $launcherRoot | Out-Null
    Copy-Item -LiteralPath $launcherScript -Destination $launcherRoot
    Write-JsonFile -Path (Join-Path $launcherRoot 'latest.json') -Value ([ordered]@{
            schemaVersion = 2
            executable = '../outside.exe'
            executableSha256 = ('0' * 64)
        })

    Invoke-ExpectedFailure @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', (Join-Path $launcherRoot 'Run-Latest-DevPortable.ps1'))
    $passed++

    Write-Host "Development portable packaging tests passed: $passed/8"
}
finally {
    if (Test-Path -LiteralPath $dirtyMarker -PathType Leaf) {
        Remove-Item -LiteralPath $dirtyMarker -Force
    }

    $resolvedTestRoot = [System.IO.Path]::GetFullPath($testRoot)
    $temporaryPrefix = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\', '/') +
        [System.IO.Path]::DirectorySeparatorChar
    if ($resolvedTestRoot.StartsWith($temporaryPrefix, [StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $resolvedTestRoot -PathType Container)) {
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}
