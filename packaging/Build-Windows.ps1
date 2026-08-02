# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$ManifestPath = (Join-Path $PSScriptRoot 'artifacts.json'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\artifacts\windows'),
    [switch]$SkipPublish,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-ChildPath {
    param(
        [Parameter(Mandatory)]
        [string]$Parent,

        [Parameter(Mandatory)]
        [string]$Child
    )

    if ([System.IO.Path]::IsPathRooted($Child)) {
        throw "Packaging paths must be relative: $Child"
    }

    $parentFullPath = [System.IO.Path]::GetFullPath($Parent)
    $childFullPath = [System.IO.Path]::GetFullPath((Join-Path $parentFullPath $Child))
    $parentPrefix = $parentFullPath.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar

    if (-not $childFullPath.StartsWith($parentPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Packaging path leaves its parent directory: $Child"
    }

    return $childFullPath
}

function Copy-DirectoryContent {
    param(
        [Parameter(Mandatory)]
        [string]$Source,

        [Parameter(Mandatory)]
        [string]$Destination
    )

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
    }
}

$manifestFullPath = [System.IO.Path]::GetFullPath($ManifestPath)
$packagingRoot = Split-Path -Parent $manifestFullPath
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $packagingRoot '..'))
$manifest = Get-Content -LiteralPath $manifestFullPath -Raw | ConvertFrom-Json

$null = & (Join-Path $packagingRoot 'Test-ReleaseArtifact.ps1') -ManifestPath $manifestFullPath

$commonDefinitionPath = Join-Path $packagingRoot 'common\publish.json'
$commonDefinition = Get-Content -LiteralPath $commonDefinitionPath -Raw | ConvertFrom-Json
$buildRoot = Resolve-ChildPath -Parent $OutputRoot -Child "$($manifest.version)-$($manifest.rid)"

if (Test-Path -LiteralPath $buildRoot) {
    if (-not $Force) {
        throw "Build staging already exists: $buildRoot. Pass -Force to replace it."
    }

    $outputRootFullPath = [System.IO.Path]::GetFullPath($OutputRoot)
    $validatedBuildRoot = Resolve-ChildPath -Parent $outputRootFullPath -Child "$($manifest.version)-$($manifest.rid)"
    Remove-Item -LiteralPath $validatedBuildRoot -Recurse -Force
}

$commonPublishPath = Resolve-ChildPath -Parent $buildRoot -Child ([string]$manifest.commonPublish)
$installerStagePath = Resolve-ChildPath -Parent $buildRoot -Child ([string]$manifest.installer.stagingDirectory)
$portableStagePath = Resolve-ChildPath -Parent $buildRoot -Child ([string]$manifest.portable.stagingDirectory)
New-Item -ItemType Directory -Path $commonPublishPath -Force | Out-Null

if (-not $SkipPublish) {
    $projectPath = Resolve-ChildPath -Parent $repositoryRoot -Child ([string]$commonDefinition.project)
    $publishArguments = @(
        'publish',
        $projectPath,
        '--configuration', [string]$commonDefinition.configuration,
        '--runtime', [string]$manifest.rid,
        '--self-contained', ([string]$commonDefinition.selfContained).ToLowerInvariant(),
        ('-p:PublishSingleFile=' + ([string]$commonDefinition.publishSingleFile).ToLowerInvariant()),
        '--output', $commonPublishPath
    )

    & dotnet @publishArguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet publish failed with exit code $LASTEXITCODE."
    }
}

foreach ($content in $commonDefinition.requiredContent) {
    $sourcePath = Resolve-ChildPath -Parent $repositoryRoot -Child ([string]$content.source)
    $targetPath = Resolve-ChildPath -Parent $commonPublishPath -Child ([string]$content.target)

    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Required distribution content is missing: $sourcePath"
    }

    if ((Get-Item -LiteralPath $sourcePath).PSIsContainer) {
        Copy-DirectoryContent -Source $sourcePath -Destination $targetPath
    }
    else {
        $targetParent = Split-Path -Parent $targetPath
        New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
    }
}

Copy-DirectoryContent -Source $commonPublishPath -Destination $installerStagePath
Copy-DirectoryContent -Source $commonPublishPath -Destination $portableStagePath

$portableDefinitionPath = Resolve-ChildPath -Parent $packagingRoot -Child ([string]$manifest.portable.definition)
$portableDefinition = Get-Content -LiteralPath $portableDefinitionPath -Raw | ConvertFrom-Json
$portableSentinelPath = Resolve-ChildPath -Parent $portableStagePath -Child ([string]$portableDefinition.sentinel)
$null = New-Item -ItemType File -Path $portableSentinelPath -Force

[pscustomobject]@{
    Version = [string]$manifest.version
    RuntimeIdentifier = [string]$manifest.rid
    CommonPublish = $commonPublishPath
    InstallerStage = $installerStagePath
    PortableStage = $portableStagePath
    InstallerArtifact = [string]$manifest.installer.fileName
    PortableArtifact = [string]$manifest.portable.fileName
    FinalArtifactsEmitted = $false
}

Write-Warning 'Goal 00 stages both layouts only. It does not create an installer, ZIP, tag, or release.'
