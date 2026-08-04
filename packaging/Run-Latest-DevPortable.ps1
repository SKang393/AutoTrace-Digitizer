# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [switch]$Wait,
    [switch]$PortableSmoke
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$adjacentLatest = Join-Path $PSScriptRoot 'latest.json'
if (Test-Path -LiteralPath $adjacentLatest -PathType Leaf) {
    $outputRoot = $PSScriptRoot
}
else {
    $repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    $outputRoot = Join-Path $repositoryRoot 'artifacts\dev-portable'
}

$latestPath = Join-Path $outputRoot 'latest.json'
if (-not (Test-Path -LiteralPath $latestPath -PathType Leaf)) {
    throw "No successful development portable exists. Run packaging\Build-DevPortable.ps1 first: $latestPath"
}

$latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace([string]$latest.executable)) {
    throw "latest.json does not identify an executable: $latestPath"
}

$executablePath = [System.IO.Path]::GetFullPath((Join-Path $outputRoot ([string]$latest.executable)))
$outputPrefix = [System.IO.Path]::GetFullPath($outputRoot).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $executablePath.StartsWith($outputPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'latest.json points outside the development portable root.'
}
if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
    throw "The latest development portable executable is missing: $executablePath"
}

$buildDirectory = Split-Path -Parent $executablePath
if (-not (Test-Path -LiteralPath (Join-Path $buildDirectory 'portable.mode') -PathType Leaf)) {
    throw "The latest development portable is missing portable.mode: $buildDirectory"
}

$sharedDataRoot = [System.IO.Path]::GetFullPath((Join-Path $outputRoot 'Data'))
New-Item -ItemType Directory -Path $sharedDataRoot -Force | Out-Null
$arguments = @()
if ($PortableSmoke.IsPresent) {
    $arguments = @('--portable-smoke')
}

$previousDataRoot = [Environment]::GetEnvironmentVariable(
    'GRAPHREADER_DEV_PORTABLE_DATA_ROOT',
    'Process')
try {
    [Environment]::SetEnvironmentVariable(
        'GRAPHREADER_DEV_PORTABLE_DATA_ROOT',
        $sharedDataRoot,
        'Process')
    $startParameters = @{
        FilePath = $executablePath
        WorkingDirectory = $buildDirectory
        PassThru = $true
        Wait = $Wait.IsPresent
    }
    if ($arguments.Count -gt 0) {
        $startParameters.ArgumentList = $arguments
    }
    $process = Start-Process @startParameters
}
finally {
    [Environment]::SetEnvironmentVariable(
        'GRAPHREADER_DEV_PORTABLE_DATA_ROOT',
        $previousDataRoot,
        'Process')
}

Write-Host "Launched: $executablePath"
Write-Host "Shared data: $sharedDataRoot"
if ($Wait.IsPresent -and $process.ExitCode -ne 0) {
    throw "Development portable exited with code $($process.ExitCode)."
}
