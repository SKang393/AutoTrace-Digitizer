# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$SourceRoot,
    [switch]$SkipVcpkgBootstrap
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'OpenCvSourceAudit.Common.ps1')

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $projectRoot 'artifacts\goal19-opencv-source\sources'
}
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$lockPath = Join-Path $PSScriptRoot 'source-lock.json'
$lock = Read-OpenCvSourceLock -Path $lockPath

function Initialize-PinnedCheckout {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Revision
    )

    $checkout = Join-Path $SourceRoot $Name
    if (-not (Test-Path -LiteralPath $checkout)) {
        New-Item -ItemType Directory -Path $checkout -Force | Out-Null
        & git init -q $checkout
        if ($LASTEXITCODE -ne 0) {
            throw "git init failed for $checkout"
        }
        & git -C $checkout remote add origin $Repository
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to configure source remote for $Name."
        }
    }

    $dirty = (& git -C $checkout status --porcelain 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Existing source directory is not a Git checkout: $checkout"
    }
    if (-not [string]::IsNullOrWhiteSpace($dirty)) {
        throw "Pinned source checkout has local changes and will not be changed: $checkout"
    }

    $current = (& git -C $checkout rev-parse HEAD 2>$null | Out-String).Trim()
    if (-not [string]::Equals($current, $Revision, [StringComparison]::OrdinalIgnoreCase)) {
        & git -C $checkout fetch --depth 1 origin $Revision
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to fetch pinned $Name revision $Revision."
        }
        & git -C $checkout -c advice.detachedHead=false checkout --detach FETCH_HEAD
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to check out pinned $Name revision $Revision."
        }
    }

    $actual = Get-RepositoryRevision -RepositoryPath $checkout
    if (-not [string]::Equals($actual, $Revision, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Name resolved to $actual, expected $Revision."
    }

    Write-Host "${Name}: $actual"
    return $checkout
}

New-Item -ItemType Directory -Path $SourceRoot -Force | Out-Null
$openCvSharpPath = Initialize-PinnedCheckout -Name 'opencvsharp' -Repository $lock.sources.openCvSharp.repository -Revision $lock.sources.openCvSharp.revision
$openCvPath = Initialize-PinnedCheckout -Name 'opencv' -Repository $lock.sources.openCv.repository -Revision $lock.sources.openCv.revision
$vcpkgPath = Initialize-PinnedCheckout -Name 'vcpkg' -Repository $lock.sources.vcpkg.repository -Revision $lock.sources.vcpkg.revision

if (-not $SkipVcpkgBootstrap) {
    $vcpkgExe = Join-Path $vcpkgPath 'vcpkg.exe'
    if (-not (Test-Path -LiteralPath $vcpkgExe -PathType Leaf)) {
        & (Join-Path $vcpkgPath 'bootstrap-vcpkg.bat') -disableMetrics
        if ($LASTEXITCODE -ne 0) {
            throw 'Pinned vcpkg bootstrap failed.'
        }
    }

    $versionLine = (& $vcpkgExe version | Select-Object -First 1)
    $actualToolVersion = ([regex]::Match($versionLine, 'version\s+(?<version>\S+)')).Groups['version'].Value
    if (-not [string]::Equals($actualToolVersion, $lock.sources.vcpkg.toolVersion, [StringComparison]::Ordinal)) {
        throw "Pinned vcpkg tool version is '$actualToolVersion', expected '$($lock.sources.vcpkg.toolVersion)'."
    }
}

Write-Host "Pinned source checkouts are ready under: $SourceRoot"
