# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$auditScript = Join-Path $PSScriptRoot "Audit-Localization.ps1"
$fixtureRoot = Join-Path $PSScriptRoot "fixtures"
$hostExecutable = (Get-Process -Id $PID).Path
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("GraphReader-LocalizationAudit-" + [Guid]::NewGuid().ToString("N"))
[IO.Directory]::CreateDirectory($temporaryRoot) | Out-Null

function Invoke-FixtureAudit {
    param(
        [string] $FixtureName,
        [int] $ExpectedExitCode,
        [switch] $StrictExtra,
        [switch] $StrictUnused
    )

    $fixturePath = Join-Path $fixtureRoot $FixtureName
    $reportPath = Join-Path $temporaryRoot "$FixtureName.json"
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $auditScript,
        "-RepositoryRoot", $fixturePath,
        "-AppSourceRoot", $fixturePath,
        "-ReportPath", $reportPath
    )
    if ($StrictExtra.IsPresent) {
        $arguments += "-FailOnExtraKeys"
    }
    if ($StrictUnused.IsPresent) {
        $arguments += "-FailOnUnusedKeys"
    }

    & $hostExecutable @arguments | Out-Host
    $actualExitCode = $LASTEXITCODE
    if ($actualExitCode -ne $ExpectedExitCode) {
        throw "Fixture '$FixtureName' returned exit code $actualExitCode; expected $ExpectedExitCode."
    }

    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        throw "Fixture '$FixtureName' did not produce a JSON report."
    }

    return Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
}

try {
    $complete = Invoke-FixtureAudit -FixtureName "complete" -ExpectedExitCode 0
    if ($complete.status -ne "pass" -or $complete.counts.missing_keys -ne 0 -or $complete.counts.extra_keys -ne 0) {
        throw "Complete fixture did not report a clean localization set."
    }

    $missing = Invoke-FixtureAudit -FixtureName "missing" -ExpectedExitCode 1
    if ($missing.status -ne "fail" -or $missing.counts.missing_keys -ne 1 -or $missing.missing_keys[0] -ne "en-US:Labels.Secondary") {
        throw "Missing fixture did not report the expected key."
    }

    $extra = Invoke-FixtureAudit -FixtureName "extra" -ExpectedExitCode 0
    if ($extra.status -ne "pass" -or $extra.counts.extra_keys -ne 1 -or $extra.extra_keys[0] -ne "en-US:Labels.Unused") {
        throw "Extra fixture did not report the expected non-strict result."
    }

    $strictExtra = Invoke-FixtureAudit -FixtureName "extra" -ExpectedExitCode 2 -StrictExtra
    if ($strictExtra.status -ne "fail" -or $strictExtra.counts.extra_keys -ne 1) {
        throw "Strict extra-key fixture did not fail as expected."
    }

    $unused = Invoke-FixtureAudit -FixtureName "unused" -ExpectedExitCode 0
    if ($unused.status -ne "pass" -or $unused.counts.unused_keys -ne 1 -or $unused.unused_keys[0] -ne "Labels.Unused") {
        throw "Unused fixture did not report the expected informational result."
    }

    $strictUnused = Invoke-FixtureAudit -FixtureName "unused" -ExpectedExitCode 4 -StrictUnused
    if ($strictUnused.status -ne "fail" -or $strictUnused.counts.unused_keys -ne 1) {
        throw "Strict unused-key fixture did not fail as expected."
    }

    $duplicate = Invoke-FixtureAudit -FixtureName "duplicate" -ExpectedExitCode 1
    if ($duplicate.status -ne "fail" -or $duplicate.counts.duplicate_keys -ne 1 -or $duplicate.duplicate_keys[0] -ne "en-US:Labels.Primary") {
        throw "Duplicate fixture did not report the expected resource key."
    }

    $unresolved = Invoke-FixtureAudit -FixtureName "unresolved" -ExpectedExitCode 1
    if ($unresolved.status -ne "fail" -or $unresolved.counts.unresolved_resource_references -ne 1 -or $unresolved.unresolved_resource_references[0] -ne "Labels.Missing") {
        throw "Unresolved fixture did not report the expected WPF resource reference."
    }

    $repositoryReportPath = Join-Path $temporaryRoot "repository-default-root.json"
    $repositoryArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $auditScript,
        "-ReportPath", $repositoryReportPath,
        "-FailOnExtraKeys"
    )
    Push-Location $temporaryRoot
    try {
        & $hostExecutable @repositoryArguments | Out-Host
        $repositoryExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    if ($repositoryExitCode -ne 0) {
        throw "Default repository-root audit returned exit code $repositoryExitCode; expected 0."
    }

    $repository = Get-Content -LiteralPath $repositoryReportPath -Raw | ConvertFrom-Json
    if ($repository.status -ne "pass" -or
        $repository.source_root -ne "src/GraphReader.App" -or
        $repository.counts.missing_keys -ne 0 -or
        $repository.counts.extra_keys -ne 0 -or
        $repository.counts.duplicate_keys -ne 0 -or
        $repository.counts.unresolved_resource_references -ne 0) {
        throw "Default repository-root audit did not report a complete localization set."
    }

    Write-Host "Localization audit self-tests: PASS (9/9)"
    exit 0
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
