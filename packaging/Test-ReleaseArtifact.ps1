# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$ManifestPath = (Join-Path $PSScriptRoot 'artifacts.json'),
    [string]$ArtifactRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Equal {
    param(
        [Parameter(Mandatory)]
        [object]$Actual,

        [Parameter(Mandatory)]
        [object]$Expected,

        [Parameter(Mandatory)]
        [string]$Description
    )

    if ($Actual -ne $Expected) {
        throw "$Description. Expected '$Expected', found '$Actual'."
    }
}

function Assert-RelativeFile {
    param(
        [Parameter(Mandatory)]
        [string]$Root,

        [Parameter(Mandatory)]
        [string]$RelativePath,

        [Parameter(Mandatory)]
        [string]$Description
    )

    if ([System.IO.Path]::IsPathRooted($RelativePath)) {
        throw "$Description must use a relative path: $RelativePath"
    }

    $rootFullPath = [System.IO.Path]::GetFullPath($Root)
    $resolvedPath = [System.IO.Path]::GetFullPath((Join-Path $rootFullPath $RelativePath))
    $rootPrefix = $rootFullPath.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar

    if (-not $resolvedPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description leaves the packaging directory: $RelativePath"
    }

    if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
        throw "$Description is missing: $resolvedPath"
    }

    return $resolvedPath
}

$manifestFullPath = [System.IO.Path]::GetFullPath($ManifestPath)
$packagingRoot = Split-Path -Parent $manifestFullPath
$manifest = Get-Content -LiteralPath $manifestFullPath -Raw | ConvertFrom-Json

Assert-Equal -Actual $manifest.schemaVersion -Expected 1 -Description 'Packaging manifest schema version is invalid'
Assert-Equal -Actual $manifest.version -Expected '0.0.1' -Description 'Initial packaging version is invalid'
Assert-Equal -Actual $manifest.rid -Expected 'win-x64' -Description 'Initial packaging RID is invalid'

if ([string]::IsNullOrWhiteSpace([string]$manifest.commonPublish)) {
    throw 'The common publish path is required.'
}

$expectedInstallerName = "GraphAutoReader-$($manifest.version)-$($manifest.rid)-setup.exe"
$expectedPortableName = "GraphAutoReader-$($manifest.version)-$($manifest.rid)-portable.zip"
Assert-Equal -Actual $manifest.installer.fileName -Expected $expectedInstallerName -Description 'Installer artifact name is invalid'
Assert-Equal -Actual $manifest.portable.fileName -Expected $expectedPortableName -Description 'Portable artifact name is invalid'

$commonDefinitionPath = Assert-RelativeFile -Root $packagingRoot -RelativePath 'common/publish.json' -Description 'Common publish definition'
$installerDefinitionPath = Assert-RelativeFile -Root $packagingRoot -RelativePath ([string]$manifest.installer.definition) -Description 'Installer definition'
$portableDefinitionPath = Assert-RelativeFile -Root $packagingRoot -RelativePath ([string]$manifest.portable.definition) -Description 'Portable definition'

$commonDefinition = Get-Content -LiteralPath $commonDefinitionPath -Raw | ConvertFrom-Json
$installerDefinition = Get-Content -LiteralPath $installerDefinitionPath -Raw | ConvertFrom-Json
$portableDefinition = Get-Content -LiteralPath $portableDefinitionPath -Raw | ConvertFrom-Json

Assert-Equal -Actual $commonDefinition.schemaVersion -Expected 1 -Description 'Common publish schema version is invalid'
Assert-Equal -Actual $commonDefinition.project -Expected 'src/GraphReader.App/GraphReader.App.csproj' -Description 'Common publish project is invalid'
Assert-Equal -Actual $commonDefinition.configuration -Expected 'Release' -Description 'Common publish configuration is invalid'
Assert-Equal -Actual $commonDefinition.selfContained -Expected $true -Description 'Common publish must be self-contained'
Assert-Equal -Actual $installerDefinition.kind -Expected 'installer' -Description 'Installer definition kind is invalid'
Assert-Equal -Actual $installerDefinition.format -Expected 'setup-exe' -Description 'Installer format is invalid'
Assert-Equal -Actual $installerDefinition.commonPublishOnly -Expected $true -Description 'Installer must consume the common publish'
Assert-Equal -Actual $installerDefinition.scope -Expected 'perUser' -Description 'Installer scope is invalid'
Assert-Equal -Actual $installerDefinition.requiresAdministrator -Expected $false -Description 'Installer must not require elevation by default'
Assert-Equal -Actual $installerDefinition.offlineCoreWorkflow -Expected $true -Description 'Installer core workflow must work offline'
Assert-Equal -Actual $installerDefinition.mutableDataRoot -Expected '%LOCALAPPDATA%\GraphAutoReader' -Description 'Installed data root is invalid'
Assert-Equal -Actual $portableDefinition.kind -Expected 'portable' -Description 'Portable definition kind is invalid'
Assert-Equal -Actual $portableDefinition.format -Expected 'zip' -Description 'Portable format is invalid'
Assert-Equal -Actual $portableDefinition.commonPublishOnly -Expected $true -Description 'Portable package must consume the common publish'
Assert-Equal -Actual $portableDefinition.requiresAdministrator -Expected $false -Description 'Portable package must not require elevation'
Assert-Equal -Actual $portableDefinition.offlineCoreWorkflow -Expected $true -Description 'Portable core workflow must work offline'
Assert-Equal -Actual $portableDefinition.sentinel -Expected 'portable.mode' -Description 'Portable sentinel name is invalid'
Assert-Equal -Actual $portableDefinition.mutableDataRoot -Expected '.\Data' -Description 'Portable data root is invalid'
Assert-Equal -Actual $portableDefinition.registryConfigurationRequired -Expected $false -Description 'Portable mode must not depend on registry configuration'

$requiredContentSources = @($commonDefinition.requiredContent | ForEach-Object { [string]$_.source })
foreach ($requiredSource in @('contracts', 'models/manifest', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'LICENSES')) {
    if ($requiredContentSources -notcontains $requiredSource) {
        throw "Common publish definition is missing required distribution content: $requiredSource"
    }
}

if (-not [string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    $artifactRootFullPath = [System.IO.Path]::GetFullPath($ArtifactRoot)
    $installerArtifactPath = Join-Path $artifactRootFullPath $manifest.installer.fileName
    $portableArtifactPath = Join-Path $artifactRootFullPath $manifest.portable.fileName

    foreach ($artifactPath in @($installerArtifactPath, $portableArtifactPath)) {
        if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
            throw "Required release artifact is missing: $artifactPath"
        }

        if ((Get-Item -LiteralPath $artifactPath).Length -eq 0) {
            throw "Release artifact is empty: $artifactPath"
        }
    }

    $installerHeader = [byte[]]::new(2)
    $installerStream = [System.IO.File]::OpenRead($installerArtifactPath)
    try {
        $installerHeaderLength = $installerStream.Read($installerHeader, 0, $installerHeader.Length)
    }
    finally {
        $installerStream.Dispose()
    }

    if ($installerHeaderLength -ne 2 -or $installerHeader[0] -ne 0x4D -or $installerHeader[1] -ne 0x5A) {
        throw "Installer artifact is not a Windows executable: $installerArtifactPath"
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($portableArtifactPath)
    try {
        $entryNames = @($archive.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
        foreach ($requiredEntry in @('GraphReader.App.exe', 'portable.mode', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md')) {
            if ($entryNames -notcontains $requiredEntry) {
                throw "Portable archive is missing required root entry: $requiredEntry"
            }
        }
    }
    finally {
        $archive.Dispose()
    }
}

[pscustomobject]@{
    Manifest = $manifestFullPath
    Version = [string]$manifest.version
    RuntimeIdentifier = [string]$manifest.rid
    Installer = [string]$manifest.installer.fileName
    Portable = [string]$manifest.portable.fileName
    ArtifactFilesChecked = -not [string]::IsNullOrWhiteSpace($ArtifactRoot)
    Status = 'PASS'
}
