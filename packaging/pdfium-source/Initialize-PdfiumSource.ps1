# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param([string]$SourceRoot)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $projectRoot 'artifacts\pdfium-source\sources'
}
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$lock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'source-lock.json') -Raw | ConvertFrom-Json
$depotTools = Join-Path $SourceRoot 'depot_tools'
$pdfiumRoot = Join-Path $SourceRoot 'pdfium'

function Invoke-BatchTool {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $quote = { param([string]$Value) '"' + $Value.Replace('"', '""') + '"' }
    $quotedArguments = @($Arguments | ForEach-Object { & $quote $_ })
    $process = Start-Process -FilePath $Path -ArgumentList $quotedArguments -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Batch tool failed with exit code $($process.ExitCode)`: $Path"
    }
}

New-Item -ItemType Directory -Path $SourceRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $depotTools)) {
    & git clone --filter=blob:none --no-checkout ([string]$lock.sources.depotTools.repository) $depotTools
    if ($LASTEXITCODE -ne 0) { throw 'Unable to clone pinned depot_tools.' }
}
& git -C $depotTools fetch --depth 1 origin ([string]$lock.sources.depotTools.revision)
if ($LASTEXITCODE -ne 0) { throw 'Unable to fetch pinned depot_tools revision.' }
& git -C $depotTools -c advice.detachedHead=false checkout --detach FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw 'Unable to check out pinned depot_tools revision.' }
if ((& git -C $depotTools status --porcelain | Out-String).Trim()) {
    throw "depot_tools contains local changes after pinned checkout: $depotTools"
}

$env:DEPOT_TOOLS_UPDATE = '0'
$env:DEPOT_TOOLS_WIN_TOOLCHAIN = '0'
$env:PATH = "$depotTools;$env:PATH"
Push-Location $SourceRoot
try {
    if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot '.gclient'))) {
        Invoke-BatchTool -Path (Join-Path $depotTools 'gclient.bat') -Arguments @(
            'config',
            '--unmanaged',
            [string]$lock.sources.pdfium.repository)
    }
    $vpython = Join-Path $depotTools '.cipd_bin\vpython3.exe'
    if (-not (Test-Path -LiteralPath $vpython -PathType Leaf)) {
        throw "depot_tools did not bootstrap its pinned vpython3 runtime: $vpython"
    }
    $gclientPath = Join-Path $depotTools 'gclient.py'
    $syncArguments = @(
        $gclientPath,
        'sync',
        '--no-history',
        '--delete_unversioned_trees',
        '-D',
        '--revision',
        "pdfium@$($lock.sources.pdfium.revision)")
    $quotedSyncArguments = @($syncArguments | ForEach-Object { '"' + $_.Replace('"', '""') + '"' })
    $launcher = Start-Process -FilePath $vpython -ArgumentList $quotedSyncArguments -NoNewWindow -Wait -PassThru
    if ($launcher.ExitCode -ne 0) {
        throw "Pinned PDFium gclient launcher failed with exit code $($launcher.ExitCode)."
    }
    $deadline = [DateTime]::UtcNow.AddMinutes(30)
    do {
        $activeSync = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq 'python3.exe' -and
            $_.CommandLine -like "*$gclientPath*sync*pdfium@$($lock.sources.pdfium.revision)*"
        })
        if ($activeSync.Count -eq 0) { break }
        if ([DateTime]::UtcNow -ge $deadline) { throw 'Pinned PDFium gclient sync exceeded 30 minutes.' }
        Start-Sleep -Seconds 1
    } while ($true)
    foreach ($generatedTool in @(
        (Join-Path $pdfiumRoot 'buildtools\win\gn.exe'),
        (Join-Path $pdfiumRoot 'third_party\ninja\ninja.exe')
    )) {
        if (-not (Test-Path -LiteralPath $generatedTool -PathType Leaf)) {
            throw "Pinned PDFium gclient hooks did not produce: $generatedTool"
        }
    }
}
finally {
    Pop-Location
}

$actual = (& git -C $pdfiumRoot rev-parse HEAD | Out-String).Trim()
if (-not [string]::Equals($actual, [string]$lock.sources.pdfium.revision, [StringComparison]::OrdinalIgnoreCase)) {
    throw "PDFium resolved to $actual, expected $($lock.sources.pdfium.revision)."
}
Write-Host "Pinned PDFium source ready: $actual"
