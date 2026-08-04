# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [switch]$Wait,
    [switch]$PortableSmoke,
    [switch]$DisableLocalEnhancement,
    [string]$RealEsrganRuntimeRoot,
    [string]$RealEsrganManifestPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$adjacentLatest = Join-Path $PSScriptRoot 'latest.json'
if (Test-Path -LiteralPath $adjacentLatest -PathType Leaf) {
    $outputRoot = $PSScriptRoot
    $repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
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
$previousEnhancementRuntimeRoot = [Environment]::GetEnvironmentVariable(
    'GRAPHREADER_REALESRGAN_RUNTIME_ROOT',
    'Process')
$previousEnhancementManifestPath = [Environment]::GetEnvironmentVariable(
    'GRAPHREADER_REALESRGAN_MANIFEST_PATH',
    'Process')
try {
    [Environment]::SetEnvironmentVariable(
        'GRAPHREADER_DEV_PORTABLE_DATA_ROOT',
        $sharedDataRoot,
        'Process')
    [Environment]::SetEnvironmentVariable(
        'GRAPHREADER_REALESRGAN_RUNTIME_ROOT',
        $null,
        'Process')
    [Environment]::SetEnvironmentVariable(
        'GRAPHREADER_REALESRGAN_MANIFEST_PATH',
        $null,
        'Process')
    if (-not $DisableLocalEnhancement.IsPresent) {
        if ([string]::IsNullOrWhiteSpace($RealEsrganRuntimeRoot)) {
            $RealEsrganRuntimeRoot = Join-Path $repositoryRoot 'artifacts\goal19-realesrgan\runtime-minimal-audit'
        }
        if ([string]::IsNullOrWhiteSpace($RealEsrganManifestPath)) {
            $RealEsrganManifestPath = Join-Path $repositoryRoot 'models\manifest\super-resolution\realesr-animevideov3-ncnn-x2.json'
        }
        if ((Test-Path -LiteralPath $RealEsrganRuntimeRoot -PathType Container) -and
            (Test-Path -LiteralPath $RealEsrganManifestPath -PathType Leaf)) {
            [Environment]::SetEnvironmentVariable(
                'GRAPHREADER_REALESRGAN_RUNTIME_ROOT',
                [System.IO.Path]::GetFullPath($RealEsrganRuntimeRoot),
                'Process')
            [Environment]::SetEnvironmentVariable(
                'GRAPHREADER_REALESRGAN_MANIFEST_PATH',
                [System.IO.Path]::GetFullPath($RealEsrganManifestPath),
                'Process')
        }
    }
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
    [Environment]::SetEnvironmentVariable(
        'GRAPHREADER_REALESRGAN_RUNTIME_ROOT',
        $previousEnhancementRuntimeRoot,
        'Process')
    [Environment]::SetEnvironmentVariable(
        'GRAPHREADER_REALESRGAN_MANIFEST_PATH',
        $previousEnhancementManifestPath,
        'Process')
}

Write-Host "Launched: $executablePath"
Write-Host "Shared data: $sharedDataRoot"
if (-not $DisableLocalEnhancement.IsPresent -and
    -not [string]::IsNullOrWhiteSpace($RealEsrganRuntimeRoot) -and
    (Test-Path -LiteralPath $RealEsrganRuntimeRoot -PathType Container)) {
    Write-Host "Local enhancement: realesr-animevideov3 x2 evaluation runtime configured outside the portable build"
}
if ($Wait.IsPresent -and $process.ExitCode -ne 0) {
    throw "Development portable exited with code $($process.ExitCode)."
}
