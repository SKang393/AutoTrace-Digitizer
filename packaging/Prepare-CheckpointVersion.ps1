# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$RepositoryRoot,
    [switch]$CheckHead,
    [switch]$PrepareNext
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'VersionPolicy.ps1')

if ($CheckHead.IsPresent -and $PrepareNext.IsPresent) {
    throw '-CheckHead and -PrepareNext cannot be used together.'
}
if (-not $CheckHead.IsPresent -and -not $PrepareNext.IsPresent) {
    $PrepareNext = $true
}

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
}
else {
    $RepositoryRoot = [System.IO.Path]::GetFullPath($RepositoryRoot)
}

$propsPath = Join-Path $RepositoryRoot 'Directory.Build.props'
$workingVersion = Get-GraphReaderCentralVersion -Path $propsPath

function Invoke-GitText {
    param([Parameter(Mandatory)][string[]]$Arguments)

    $output = & git -C $RepositoryRoot @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    return ($output -join [Environment]::NewLine).Trim()
}

function Get-CommittedVersion {
    param([Parameter(Mandatory)][string]$Revision)

    $content = Invoke-GitText -Arguments @('show', ('{0}:Directory.Build.props' -f $Revision))
    if ([string]::IsNullOrWhiteSpace($content)) {
        throw "Directory.Build.props is unavailable at Git revision '$Revision'."
    }
    return Get-GraphReaderVersionFromProjectXml -Content $content -Description "$Revision`:Directory.Build.props"
}

function Set-VersionFields {
    param([Parameter(Mandatory)][string]$Version)

    $content = Get-Content -LiteralPath $propsPath -Raw
    $replacements = [ordered]@{
        Version = $Version
        AssemblyVersion = "$Version.0"
        FileVersion = "$Version.0"
        InformationalVersion = $Version
    }
    foreach ($entry in $replacements.GetEnumerator()) {
        $pattern = '<{0}>[^<]*</{0}>' -f [regex]::Escape([string]$entry.Key)
        $matches = [regex]::Matches($content, $pattern)
        if ($matches.Count -ne 1) {
            throw "Expected exactly one $($entry.Key) element in $propsPath."
        }
        $replacement = '<{0}>{1}</{0}>' -f $entry.Key, $entry.Value
        $content = [regex]::Replace($content, $pattern, $replacement)
    }

    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($propsPath, $content, $encoding)
}

if ($CheckHead.IsPresent) {
    $parent = Invoke-GitText -Arguments @('rev-parse', 'HEAD^')
    $expected = if ([string]::IsNullOrWhiteSpace($parent)) {
        '0.0.1'
    }
    else {
        Get-NextGraphReaderVersion -Version (Get-CommittedVersion -Revision $parent).Value
    }

    if ($workingVersion.Value -ne $expected) {
        throw "Checkpoint version '$($workingVersion.Value)' is invalid for HEAD. Expected '$expected' from its first parent. Run packaging/Prepare-CheckpointVersion.ps1 before committing."
    }

    Write-Host "Checkpoint version verified: $($workingVersion.Value)"
    exit 0
}

$committedVersion = Get-CommittedVersion -Revision 'HEAD'
$nextVersion = Get-NextGraphReaderVersion -Version $committedVersion.Value
if ($workingVersion.Value -eq $nextVersion) {
    Write-Host "Checkpoint version already prepared: $nextVersion"
    exit 0
}
if ($workingVersion.Value -ne $committedVersion.Value) {
    throw "Working version '$($workingVersion.Value)' is neither committed version '$($committedVersion.Value)' nor its successor '$nextVersion'."
}

Set-VersionFields -Version $nextVersion
Write-Host "Prepared checkpoint version: $($committedVersion.Value) -> $nextVersion"
