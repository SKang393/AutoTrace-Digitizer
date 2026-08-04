# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$EvidenceRoot,
    [string]$OutputRoot,
    [ValidateRange(15, 120)]
    [int]$ApplicationSmokeTimeoutSeconds = 30
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) {
    $EvidenceRoot = Join-Path $projectRoot 'artifacts\goal19-opencv-source\evidence-repro-pass2-final-a'
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $projectRoot 'artifacts\goal19-opencv-source\runtime-parity'
}

$EvidenceRoot = [IO.Path]::GetFullPath($EvidenceRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$sourceDll = Join-Path $EvidenceRoot 'bin\OpenCvSharpExtern.dll'
$policyPath = Join-Path $PSScriptRoot 'review\source-build-review-policy.json'
$integrationProject = Join-Path $projectRoot 'tests\GraphReader.Integration.Tests\GraphReader.Integration.Tests.csproj'
$appProject = Join-Path $projectRoot 'src\GraphReader.App\GraphReader.App.csproj'
$benchmarkFilter = 'FullyQualifiedName=GraphReader.Integration.Tests.IntegrationSmoke.OpenCvSourceRuntimeParityTests.PublicOpenCvAxisBenchmarkMatchesFixedMetricBudget'

if (-not (Test-Path -LiteralPath $sourceDll -PathType Leaf)) {
    throw "Source-built OpenCvSharpExtern.dll is missing: $sourceDll"
}
if (-not (Test-Path -LiteralPath $policyPath -PathType Leaf)) {
    throw "Source-build review policy is missing: $policyPath"
}

$policy = Get-Content -LiteralPath $policyPath -Raw | ConvertFrom-Json
$sourceDllHash = (Get-FileHash -LiteralPath $sourceDll -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not [string]::Equals($sourceDllHash, [string]$policy.binarySha256, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Source-built DLL hash '$sourceDllHash' does not match reviewed policy '$($policy.binarySha256)'."
}

$runName = 'run-{0}-{1}' -f [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'), ([Guid]::NewGuid().ToString('N'))
$runRoot = Join-Path $OutputRoot $runName
$resultsRoot = Join-Path $runRoot 'test-results'
$packageBackupRoot = Join-Path $runRoot 'package-runtime'
$appPackageRoot = Join-Path $runRoot 'app-package'
$appSourceRoot = Join-Path $runRoot 'app-source'
New-Item -ItemType Directory -Path $resultsRoot, $packageBackupRoot, $appPackageRoot, $appSourceRoot -Force | Out-Null

function Invoke-Checked {
    param(
        [Parameter(Mandatory)] [string]$Executable,
        [Parameter(Mandatory)] [string[]]$Arguments,
        [Parameter(Mandatory)] [string]$Description
    )

    Write-Host $Description
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Invoke-ParityBenchmark {
    param(
        [Parameter(Mandatory)] [string]$EvidencePath,
        [Parameter(Mandatory)] [string]$TrxName
    )

    $previousOutput = [Environment]::GetEnvironmentVariable('GRAPHREADER_OPENCV_PARITY_OUTPUT', 'Process')
    try {
        [Environment]::SetEnvironmentVariable('GRAPHREADER_OPENCV_PARITY_OUTPUT', $EvidencePath, 'Process')
        Invoke-Checked -Executable 'dotnet' -Arguments @(
            'test', $integrationProject,
            '-c', 'Release',
            '--no-build',
            '--filter', $benchmarkFilter,
            '--results-directory', $resultsRoot,
            '--logger', "trx;LogFileName=$TrxName"
        ) -Description "Run OpenCV public axis parity benchmark: $TrxName"
    }
    finally {
        [Environment]::SetEnvironmentVariable('GRAPHREADER_OPENCV_PARITY_OUTPUT', $previousOutput, 'Process')
    }

    if (-not (Test-Path -LiteralPath $EvidencePath -PathType Leaf)) {
        throw "OpenCV public axis benchmark did not emit evidence: $EvidencePath"
    }
}

function Get-TrxCounters {
    param([Parameter(Mandatory)] [string]$Path)

    [xml]$document = Get-Content -LiteralPath $Path -Raw
    $counters = $document.TestRun.ResultSummary.Counters
    return [ordered]@{
        total = [int]$counters.total
        executed = [int]$counters.executed
        passed = [int]$counters.passed
        failed = [int]$counters.failed
        error = [int]$counters.error
    }
}

function Invoke-ApplicationSmoke {
    param(
        [Parameter(Mandatory)] [string]$ExecutablePath,
        [Parameter(Mandatory)] [string]$Label
    )

    $process = Start-Process -FilePath $ExecutablePath -ArgumentList '--portable-smoke' -PassThru -WindowStyle Hidden
    try {
        if (-not $process.WaitForExit($ApplicationSmokeTimeoutSeconds * 1000)) {
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
            throw "$Label application smoke exceeded $ApplicationSmokeTimeoutSeconds seconds."
        }
        if ($process.ExitCode -ne 0) {
            throw "$Label application smoke exited with code $($process.ExitCode)."
        }
        return $process.ExitCode
    }
    finally {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        }
        $process.Dispose()
    }
}

Invoke-Checked -Executable 'powershell' -Arguments @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'Test-SourceAuditEvidence.ps1'),
    '-EvidenceRoot', $EvidenceRoot
) -Description 'Validate retained OpenCV source-build evidence'
Invoke-Checked -Executable 'powershell' -Arguments @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'review\Test-SourceBuildReviewPolicy.ps1'),
    '-EvidenceRoot', $EvidenceRoot
) -Description 'Validate reviewed OpenCV source-build provenance policy'
Invoke-Checked -Executable 'dotnet' -Arguments @(
    'build', $integrationProject,
    '-c', 'Release',
    '--no-restore'
) -Description 'Build integration parity host'

$integrationOutput = Join-Path $projectRoot 'tests\GraphReader.Integration.Tests\bin\Release\net10.0-windows'
$nativeCandidates = @(Get-ChildItem -LiteralPath $integrationOutput -Filter 'OpenCvSharpExtern.dll' -File -Recurse)
if ($nativeCandidates.Count -ne 1) {
    throw "Expected exactly one integration-test OpenCvSharpExtern.dll, found $($nativeCandidates.Count)."
}

$integrationNative = $nativeCandidates[0].FullName
$packageBackup = Join-Path $packageBackupRoot 'OpenCvSharpExtern.dll'
Copy-Item -LiteralPath $integrationNative -Destination $packageBackup -Force
$packageDllHash = (Get-FileHash -LiteralPath $packageBackup -Algorithm SHA256).Hash.ToLowerInvariant()
$baselineEvidence = Join-Path $runRoot 'axis-package-runtime.json'
$sourceEvidence = Join-Path $runRoot 'axis-source-runtime.json'

try {
    Invoke-ParityBenchmark -EvidencePath $baselineEvidence -TrxName 'axis-package-runtime.trx'
    Copy-Item -LiteralPath $sourceDll -Destination $integrationNative -Force
    $installedSourceHash = (Get-FileHash -LiteralPath $integrationNative -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [string]::Equals($installedSourceHash, $sourceDllHash, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The source-built DLL did not copy exactly into the integration parity host.'
    }

    Invoke-ParityBenchmark -EvidencePath $sourceEvidence -TrxName 'axis-source-runtime.trx'
    Invoke-Checked -Executable 'dotnet' -Arguments @(
        'test', $integrationProject,
        '-c', 'Release',
        '--no-build',
        '--results-directory', $resultsRoot,
        '--logger', 'trx;LogFileName=integration-source-runtime.trx'
    ) -Description 'Run the complete integration assembly with the source-built OpenCV runtime'
}
finally {
    Copy-Item -LiteralPath $packageBackup -Destination $integrationNative -Force
    $restoredHash = (Get-FileHash -LiteralPath $integrationNative -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [string]::Equals($restoredHash, $packageDllHash, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The NuGet OpenCV runtime was not restored exactly after parity testing.'
    }
}

$baselineBenchmarkHash = (Get-FileHash -LiteralPath $baselineEvidence -Algorithm SHA256).Hash.ToLowerInvariant()
$sourceBenchmarkHash = (Get-FileHash -LiteralPath $sourceEvidence -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not [string]::Equals($baselineBenchmarkHash, $sourceBenchmarkHash, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The source-built OpenCV runtime changed canonical public-axis benchmark output.'
}
$benchmarkSha256 = $sourceBenchmarkHash

Invoke-Checked -Executable 'dotnet' -Arguments @(
    'publish', $appProject,
    '-c', 'Release',
    '-r', 'win-x64',
    '--self-contained', 'true',
    '--no-restore',
    '-o', $appPackageRoot
) -Description 'Publish the WPF application for OpenCV runtime smoke'

Copy-Item -Path (Join-Path $appPackageRoot '*') -Destination $appSourceRoot -Recurse -Force
New-Item -ItemType File -Path (Join-Path $appPackageRoot 'portable.mode') -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $appSourceRoot 'portable.mode') -Force | Out-Null
$publishedPackageNative = @(Get-ChildItem -LiteralPath $appPackageRoot -Filter 'OpenCvSharpExtern.dll' -File -Recurse)
$publishedSourceNative = @(Get-ChildItem -LiteralPath $appSourceRoot -Filter 'OpenCvSharpExtern.dll' -File -Recurse)
if ($publishedPackageNative.Count -ne 1 -or $publishedSourceNative.Count -ne 1) {
    throw 'Expected exactly one OpenCvSharpExtern.dll in each application publish tree.'
}

$publishedPackageHash = (Get-FileHash -LiteralPath $publishedPackageNative[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not [string]::Equals($publishedPackageHash, $packageDllHash, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Published NuGet OpenCV runtime does not match the integration parity baseline.'
}
Copy-Item -LiteralPath $sourceDll -Destination $publishedSourceNative[0].FullName -Force
$publishedSourceHash = (Get-FileHash -LiteralPath $publishedSourceNative[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not [string]::Equals($publishedSourceHash, $sourceDllHash, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Published source-built OpenCV runtime copy failed checksum verification.'
}

$packageSmokeExitCode = Invoke-ApplicationSmoke -ExecutablePath (Join-Path $appPackageRoot 'GraphReader.App.exe') -Label 'NuGet runtime'
$sourceSmokeExitCode = Invoke-ApplicationSmoke -ExecutablePath (Join-Path $appSourceRoot 'GraphReader.App.exe') -Label 'Source runtime'

$summary = [ordered]@{
    schema = 'graphreader.opencv-source-runtime-parity.v1'
    completedUtc = [DateTimeOffset]::UtcNow.ToString('O')
    repositoryCommit = (& git -C $projectRoot rev-parse HEAD).Trim()
    repositoryDirty = [bool](& git -C $projectRoot status --porcelain=v1)
    sourceEvidenceRoot = $EvidenceRoot
    sourceDllSha256 = $sourceDllHash
    packageDllSha256 = $packageDllHash
    publicAxisBenchmark = [ordered]@{
        fixtureCount = 4
        maximumOriginErrorPixels = 5.0
        exactCanonicalParity = $true
        evidenceSha256 = $benchmarkSha256
        packageEvidence = $baselineEvidence
        sourceEvidence = $sourceEvidence
        packageCounters = Get-TrxCounters -Path (Join-Path $resultsRoot 'axis-package-runtime.trx')
        sourceCounters = Get-TrxCounters -Path (Join-Path $resultsRoot 'axis-source-runtime.trx')
    }
    fullIntegrationSourceRuntime = Get-TrxCounters -Path (Join-Path $resultsRoot 'integration-source-runtime.trx')
    applicationSmoke = [ordered]@{
        packageExitCode = $packageSmokeExitCode
        sourceExitCode = $sourceSmokeExitCode
        packagePublishRoot = $appPackageRoot
        sourcePublishRoot = $appSourceRoot
    }
    provenanceValidated = $true
    cleanMachineEvidence = $false
    releaseApproved = $false
    remainingGate = 'Clean-machine load and workflow evidence is still mandatory before replacing the release runtime.'
}
$summaryPath = Join-Path $runRoot 'runtime-parity-summary.json'
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summaryPath -Encoding utf8NoBOM

Write-Host "OpenCV source-runtime parity: PASS"
Write-Host "Evidence: $summaryPath"
Write-Output ([pscustomobject]@{
    SummaryPath = $summaryPath
    RunRoot = $runRoot
    SourceDllSha256 = $sourceDllHash
    PackageDllSha256 = $packageDllHash
    BenchmarkSha256 = $benchmarkSha256
    CleanMachineEvidence = $false
    ReleaseApproved = $false
})
