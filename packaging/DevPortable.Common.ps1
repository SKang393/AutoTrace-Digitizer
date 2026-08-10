# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$Content
    )

    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    [System.IO.File]::WriteAllText(
        $Path,
        $Content,
        [System.Text.UTF8Encoding]::new($false))
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [object]$Value
    )

    Write-Utf8NoBom -Path $Path -Content (($Value | ConvertTo-Json -Depth 20) + [Environment]::NewLine)
}

function Write-JsonFileAtomic {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [object]$Value
    )

    $operationId = [Guid]::NewGuid().ToString('N')
    $temporaryPath = "$Path.$operationId.tmp"
    $backupPath = "$Path.$operationId.bak"
    try {
        Write-JsonFile -Path $temporaryPath -Value $Value
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            [System.IO.File]::Replace($temporaryPath, $Path, $backupPath)
        }
        else {
            [System.IO.File]::Move($temporaryPath, $Path)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
        if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
            Remove-Item -LiteralPath $backupPath -Force
        }
    }
}

function Copy-DevPortableRequiredContent {
    param(
        [Parameter(Mandatory)]
        [string]$RepositoryRoot,

        [Parameter(Mandatory)]
        [string]$DestinationRoot
    )

    $repositoryPath = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
    $destinationPath = [System.IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\', '/')
    $definitionPath = Join-Path $repositoryPath 'packaging\common\publish.json'
    if (-not (Test-Path -LiteralPath $definitionPath -PathType Leaf)) {
        throw "Common publish definition is missing: $definitionPath"
    }

    $definition = Get-Content -LiteralPath $definitionPath -Raw | ConvertFrom-Json
    foreach ($content in @($definition.requiredContent)) {
        $source = ([string]$content.source).Replace('/', '\')
        $target = ([string]$content.target).Replace('/', '\')
        if ([string]::IsNullOrWhiteSpace($source) -or
            [string]::IsNullOrWhiteSpace($target) -or
            [System.IO.Path]::IsPathRooted($source) -or
            [System.IO.Path]::IsPathRooted($target) -or
            @($source.Split('\')) -contains '..' -or
            @($target.Split('\')) -contains '..') {
            throw "Common publish contains an unsafe required-content mapping: '$source' -> '$target'."
        }

        $sourcePath = [System.IO.Path]::GetFullPath((Join-Path $repositoryPath $source))
        $targetPath = [System.IO.Path]::GetFullPath((Join-Path $destinationPath $target))
        if (-not $sourcePath.StartsWith($repositoryPath + [System.IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
            -not $targetPath.StartsWith($destinationPath + [System.IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Common publish required content resolves outside its allowed root: '$source' -> '$target'."
        }
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            throw "Required common publish content is missing: $sourcePath"
        }

        if (Test-Path -LiteralPath $sourcePath -PathType Container) {
            New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
            Get-ChildItem -LiteralPath $sourcePath -Force |
                Copy-Item -Destination $targetPath -Recurse -Force
        }
        else {
            $targetParent = Split-Path -Parent $targetPath
            if (-not [string]::IsNullOrWhiteSpace($targetParent)) {
                New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
            }
            Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
        }
    }
}

function Get-DevPortableApprovedModelIds {
    param(
        [Parameter(Mandatory)]
        [string]$RepositoryRoot,

        [System.Collections.Generic.List[string]]$Diagnostics
    )

    $manifestRoot = Join-Path $RepositoryRoot 'models\manifest'
    if (-not (Test-Path -LiteralPath $manifestRoot -PathType Container)) {
        return @()
    }

    $available = [System.Collections.Generic.List[string]]::new()
    $parsedManifests = [System.Collections.Generic.List[object]]::new()
    $modelIdPaths = @{}
    $manifestPaths = @(Get-ChildItem -LiteralPath $manifestRoot -Recurse -Filter '*.json' -File |
        Sort-Object FullName)
    foreach ($manifestPath in $manifestPaths) {
        try {
            $manifest = Get-Content -LiteralPath $manifestPath.FullName -Raw | ConvertFrom-Json
            $parsedManifests.Add([pscustomobject]@{
                    Path = $manifestPath
                    Manifest = $manifest
                })
            $modelId = [string]$manifest.model_id
            if (-not [string]::IsNullOrWhiteSpace($modelId)) {
                if (-not $modelIdPaths.ContainsKey($modelId)) {
                    $modelIdPaths[$modelId] = [System.Collections.Generic.List[string]]::new()
                }
                $modelIdPaths[$modelId].Add($manifestPath.FullName)
            }
        }
        catch {
            if ($null -ne $Diagnostics) {
                $Diagnostics.Add("Skipped invalid model manifest '$($manifestPath.Name)': $($_.Exception.Message)")
            }
        }
    }

    $duplicateModelIds = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::Ordinal)
    foreach ($modelId in @($modelIdPaths.Keys | Sort-Object)) {
        if ($modelIdPaths[$modelId].Count -gt 1) {
            $null = $duplicateModelIds.Add($modelId)
            if ($null -ne $Diagnostics) {
                $names = @($modelIdPaths[$modelId] | ForEach-Object { [System.IO.Path]::GetFileName($_) }) -join ', '
                $Diagnostics.Add("Skipped duplicate model ID '$modelId' declared by: $names")
            }
        }
    }

    foreach ($parsed in $parsedManifests) {
        $manifestPath = $parsed.Path
        $manifest = $parsed.Manifest
        try {
            if ([string]::IsNullOrWhiteSpace([string]$manifest.model_id) -or
                $duplicateModelIds.Contains([string]$manifest.model_id) -or
                $manifest.license.reviewed -ne $true -or
                $manifest.commercial_use -ne $true -or
                $manifest.redistribution -ne $true) {
                continue
            }

            $benchmarks = @($manifest.benchmarks)
            $releaseBenchmarks = @($benchmarks | Where-Object {
                    [string]$_.status -eq 'pass' -and $_.release_eligible -eq $true
                })
            if ($releaseBenchmarks.Count -eq 0) {
                continue
            }

            $modelFiles = @($manifest.files | ForEach-Object { ([string]$_).Replace('\', '/') })
            if ($modelFiles.Count -eq 0 -or
                @($modelFiles | Where-Object {
                        [string]::IsNullOrWhiteSpace($_) -or
                        [System.IO.Path]::IsPathRooted($_) -or
                        @($_.Split('/')) -contains '..'
                    }).Count -gt 0 -or
                @($modelFiles | Sort-Object -Unique).Count -ne $modelFiles.Count) {
                throw 'Model payload paths are missing, unsafe, or duplicated.'
            }

            $artifactHashes = [System.Collections.Generic.Dictionary[string,string]]::new(
                [System.StringComparer]::Ordinal)
            if ($modelFiles.Count -eq 1) {
                if ([string]$manifest.sha256 -notmatch '^[a-fA-F0-9]{64}$') {
                    throw 'The single model payload checksum is invalid.'
                }
                $artifactHashes.Add($modelFiles[0], ([string]$manifest.sha256).ToLowerInvariant())
            }
            else {
                $hashMap = $manifest.preprocessing.PSObject.Properties['model_payload_sha256'].Value
                if ($null -eq $hashMap -or $hashMap -is [string] -or $hashMap -is [System.Array]) {
                    throw 'A multi-file model requires a payload checksum map.'
                }
                foreach ($hashProperty in @($hashMap.PSObject.Properties)) {
                    $hashPath = ([string]$hashProperty.Name).Replace('\', '/')
                    $hashValue = [string]$hashProperty.Value
                    if ($hashValue -notmatch '^[a-fA-F0-9]{64}$' -or
                        $artifactHashes.ContainsKey($hashPath)) {
                        throw "The payload checksum entry for '$hashPath' is invalid or duplicated."
                    }
                    $artifactHashes.Add($hashPath, $hashValue.ToLowerInvariant())
                }
                if ($artifactHashes.Count -ne $modelFiles.Count -or
                    @($modelFiles | Where-Object { -not $artifactHashes.ContainsKey($_) }).Count -gt 0) {
                    throw 'The multi-file payload checksum map does not exactly match the declared files.'
                }
            }

            foreach ($modelFile in $modelFiles) {
                $candidatePaths = @(
                    (Join-Path $RepositoryRoot $modelFile),
                    (Join-Path (Join-Path $RepositoryRoot 'models') $modelFile),
                    (Join-Path $manifestPath.DirectoryName $modelFile)) |
                    ForEach-Object { [System.IO.Path]::GetFullPath($_) } |
                    Sort-Object -Unique
                $located = @($candidatePaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
                if ($located.Count -ne 1) {
                    throw "Model payload '$modelFile' does not resolve to exactly one file."
                }
                $actualHash = (Get-FileHash -LiteralPath $located[0] -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($actualHash -ne $artifactHashes[$modelFile]) {
                    throw "Model payload '$modelFile' does not match its approved checksum."
                }
            }

            $available.Add([string]$manifest.model_id)
        }
        catch {
            if ($null -ne $Diagnostics) {
                $Diagnostics.Add("Skipped invalid model manifest '$($manifestPath.Name)': $($_.Exception.Message)")
            }
        }
    }

    return @($available | Sort-Object)
}
