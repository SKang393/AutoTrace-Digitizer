# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string] $RepositoryRoot,
    [string] $AppSourceRoot,
    [string] $ReportPath,
    [switch] $FailOnExtraKeys,
    [switch] $FailOnUnusedKeys
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-OrdinalSortedUnique {
    param([string[]] $Values)

    $unique = New-Object "System.Collections.Generic.HashSet[string]" ([StringComparer]::Ordinal)
    foreach ($value in $Values) {
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            [void] $unique.Add($value)
        }
    }

    $result = [string[]] $unique
    [Array]::Sort($result, [StringComparer]::Ordinal)
    return ,$result
}

function Get-RelativeAuditPath {
    param(
        [string] $Root,
        [string] $Path
    )

    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $pathFull = [IO.Path]::GetFullPath($Path)
    if ($pathFull.StartsWith($rootFull + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        return $pathFull.Substring($rootFull.Length + 1).Replace("\", "/")
    }

    return $pathFull.Replace("\", "/")
}

function Get-XamlKeys {
    param([string] $Path)

    $document = New-Object Xml.XmlDocument
    $document.PreserveWhitespace = $true
    $document.Load($Path)

    $namespaces = New-Object Xml.XmlNamespaceManager($document.NameTable)
    $namespaces.AddNamespace("x", "http://schemas.microsoft.com/winfx/2006/xaml")
    $nodes = $document.SelectNodes("//*[@x:Key]", $namespaces)
    $keys = New-Object "System.Collections.Generic.List[string]"
    foreach ($node in $nodes) {
        $attribute = $node.Attributes.GetNamedItem("Key", "http://schemas.microsoft.com/winfx/2006/xaml")
        if ($null -ne $attribute -and -not [string]::IsNullOrWhiteSpace($attribute.Value)) {
            $keys.Add($attribute.Value)
        }
    }

    return ,([string[]] $keys)
}

function Get-DuplicateKeys {
    param([string[]] $Keys)

    $counts = @{}
    foreach ($key in $Keys) {
        if ($counts.ContainsKey($key)) {
            $counts[$key]++
        }
        else {
            $counts[$key] = 1
        }
    }

    $duplicates = @($counts.Keys | Where-Object { $counts[$_] -gt 1 })
    return ,(Get-OrdinalSortedUnique -Values $duplicates)
}

function Get-SetDifference {
    param(
        [string[]] $Left,
        [string[]] $Right
    )

    $rightSet = New-Object "System.Collections.Generic.HashSet[string]" ([StringComparer]::Ordinal)
    foreach ($item in $Right) {
        [void] $rightSet.Add($item)
    }

    $difference = @($Left | Where-Object { -not $rightSet.Contains($_) })
    return ,(Get-OrdinalSortedUnique -Values $difference)
}

try {
    if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
        $RepositoryRoot = Join-Path $PSScriptRoot "../.."
    }

    $repositoryRootFull = [IO.Path]::GetFullPath($RepositoryRoot)
    if ([string]::IsNullOrWhiteSpace($AppSourceRoot)) {
        $AppSourceRoot = Join-Path $repositoryRootFull "src/GraphReader.App"
    }

    $appSourceRootFull = [IO.Path]::GetFullPath($AppSourceRoot)
    if (-not (Test-Path -LiteralPath $appSourceRootFull -PathType Container)) {
        throw "WPF application source root does not exist: $appSourceRootFull"
    }

    $localizationRoot = Join-Path $appSourceRootFull "Localization"
    $contractPath = Join-Path $localizationRoot "LocalizationKeys.cs"
    if (-not (Test-Path -LiteralPath $contractPath -PathType Leaf)) {
        throw "Localization key contract does not exist: $contractPath"
    }

    $contractSource = [IO.File]::ReadAllText($contractPath)
    $contractPattern = 'public\s+const\s+string\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"(?<key>(?:\\.|[^"\\])*)"\s*;'
    $contractMatches = [Text.RegularExpressions.Regex]::Matches($contractSource, $contractPattern)
    if ($contractMatches.Count -eq 0) {
        throw "No public localization key constants were found in $contractPath"
    }

    $contractKeysByName = @{}
    $contractKeysRaw = New-Object "System.Collections.Generic.List[string]"
    foreach ($match in $contractMatches) {
        $name = $match.Groups["name"].Value
        $key = [Text.RegularExpressions.Regex]::Unescape($match.Groups["key"].Value)
        if ($contractKeysByName.ContainsKey($name)) {
            throw "Duplicate localization constant name '$name' in $contractPath"
        }

        $contractKeysByName[$name] = $key
        $contractKeysRaw.Add($key)
    }

    $contractDuplicateKeys = Get-DuplicateKeys -Keys ([string[]] $contractKeysRaw)
    $contractKeys = Get-OrdinalSortedUnique -Values ([string[]] $contractKeysRaw)

    $resourceFiles = @(Get-ChildItem -LiteralPath $localizationRoot -File -Filter "Resources.*.xaml")
    $resourceFiles = @($resourceFiles | Sort-Object -Property FullName)
    if ($resourceFiles.Count -eq 0) {
        throw "No localization resource dictionaries matching Resources.*.xaml were found in $localizationRoot"
    }

    $dictionaryReports = New-Object "System.Collections.Generic.List[object]"
    $allLocalizationResourceKeys = New-Object "System.Collections.Generic.List[string]"
    $allMissingKeys = New-Object "System.Collections.Generic.List[string]"
    $allExtraKeys = New-Object "System.Collections.Generic.List[string]"
    $allDuplicateKeys = New-Object "System.Collections.Generic.List[string]"

    foreach ($resourceFile in $resourceFiles) {
        if ($resourceFile.Name -notmatch '^Resources\.(?<culture>.+)\.xaml$') {
            throw "Localization resource filename does not identify a culture: $($resourceFile.Name)"
        }

        $culture = $Matches["culture"]
        $definedKeysRaw = Get-XamlKeys -Path $resourceFile.FullName
        $definedKeys = Get-OrdinalSortedUnique -Values $definedKeysRaw
        $missingKeys = Get-SetDifference -Left $contractKeys -Right $definedKeys
        $extraKeys = Get-SetDifference -Left $definedKeys -Right $contractKeys
        $duplicateKeys = Get-DuplicateKeys -Keys $definedKeysRaw

        foreach ($key in $definedKeys) { $allLocalizationResourceKeys.Add($key) }
        foreach ($key in $missingKeys) { $allMissingKeys.Add("$culture`:$key") }
        foreach ($key in $extraKeys) { $allExtraKeys.Add("$culture`:$key") }
        foreach ($key in $duplicateKeys) { $allDuplicateKeys.Add("$culture`:$key") }

        $dictionaryReports.Add([ordered]@{
            culture = $culture
            path = Get-RelativeAuditPath -Root $repositoryRootFull -Path $resourceFile.FullName
            defined_count = $definedKeys.Count
            missing_keys = $missingKeys
            extra_keys = $extraKeys
            duplicate_keys = $duplicateKeys
        })
    }

    foreach ($key in $contractDuplicateKeys) {
        $allDuplicateKeys.Add("contract:$key")
    }

    $xamlFiles = @(Get-ChildItem -LiteralPath $appSourceRootFull -Recurse -File -Filter "*.xaml" | Sort-Object -Property FullName)
    $allDefinedXamlKeys = New-Object "System.Collections.Generic.List[string]"
    $allResourceReferences = New-Object "System.Collections.Generic.List[string]"
    $resourceReferencePattern = '\{(?:DynamicResource|StaticResource)\s+(?:ResourceKey\s*=\s*)?(?<key>[^\s,\}]+)'

    foreach ($xamlFile in $xamlFiles) {
        foreach ($key in (Get-XamlKeys -Path $xamlFile.FullName)) {
            $allDefinedXamlKeys.Add($key)
        }

        $xamlSource = [IO.File]::ReadAllText($xamlFile.FullName)
        foreach ($referenceMatch in [Text.RegularExpressions.Regex]::Matches($xamlSource, $resourceReferencePattern)) {
            $key = $referenceMatch.Groups["key"].Value
            if (-not $key.StartsWith("{", [StringComparison]::Ordinal)) {
                $allResourceReferences.Add($key)
            }
        }
    }

    $definedXamlKeys = Get-OrdinalSortedUnique -Values ([string[]] $allDefinedXamlKeys)
    $resourceReferences = Get-OrdinalSortedUnique -Values ([string[]] $allResourceReferences)
    $unresolvedResourceReferences = Get-SetDifference -Left $resourceReferences -Right $definedXamlKeys

    $candidateLocalizationKeys = Get-OrdinalSortedUnique -Values @($contractKeys + [string[]] $allLocalizationResourceKeys)
    $candidateLocalizationSet = New-Object "System.Collections.Generic.HashSet[string]" ([StringComparer]::Ordinal)
    foreach ($key in $candidateLocalizationKeys) { [void] $candidateLocalizationSet.Add($key) }

    $referencedLocalizationKeysRaw = New-Object "System.Collections.Generic.List[string]"
    foreach ($key in $resourceReferences) {
        if ($candidateLocalizationSet.Contains($key)) {
            $referencedLocalizationKeysRaw.Add($key)
        }
    }

    $csharpFiles = @(Get-ChildItem -LiteralPath $appSourceRootFull -Recurse -File -Filter "*.cs" | Sort-Object -Property FullName)
    $constantReferencePattern = 'LocalizationKeys\.(?<name>[A-Za-z_][A-Za-z0-9_]*)'
    $literalReferencePattern = '(?:FindResource|TryFindResource|GetString|GetLocalizedString|ResourceText|FormatResource)\s*\(\s*"(?<key>(?:\\.|[^"\\])*)"'
    foreach ($csharpFile in $csharpFiles) {
        $csharpSource = [IO.File]::ReadAllText($csharpFile.FullName)
        foreach ($referenceMatch in [Text.RegularExpressions.Regex]::Matches($csharpSource, $constantReferencePattern)) {
            $name = $referenceMatch.Groups["name"].Value
            if ($contractKeysByName.ContainsKey($name)) {
                $referencedLocalizationKeysRaw.Add([string] $contractKeysByName[$name])
            }
        }

        foreach ($referenceMatch in [Text.RegularExpressions.Regex]::Matches($csharpSource, $literalReferencePattern)) {
            $key = [Text.RegularExpressions.Regex]::Unescape($referenceMatch.Groups["key"].Value)
            if ($candidateLocalizationSet.Contains($key)) {
                $referencedLocalizationKeysRaw.Add($key)
            }
        }
    }

    $referencedLocalizationKeys = Get-OrdinalSortedUnique -Values ([string[]] $referencedLocalizationKeysRaw)
    $unusedKeys = Get-SetDifference -Left $contractKeys -Right $referencedLocalizationKeys
    $missingKeysFlat = Get-OrdinalSortedUnique -Values ([string[]] $allMissingKeys)
    $extraKeysFlat = Get-OrdinalSortedUnique -Values ([string[]] $allExtraKeys)
    $duplicateKeysFlat = Get-OrdinalSortedUnique -Values ([string[]] $allDuplicateKeys)

    $hasRequiredFailure = $missingKeysFlat.Count -gt 0 -or
        $duplicateKeysFlat.Count -gt 0 -or
        $unresolvedResourceReferences.Count -gt 0
    $hasStrictExtraFailure = $FailOnExtraKeys.IsPresent -and $extraKeysFlat.Count -gt 0
    $hasStrictUnusedFailure = $FailOnUnusedKeys.IsPresent -and $unusedKeys.Count -gt 0
    $status = if ($hasRequiredFailure -or $hasStrictExtraFailure -or $hasStrictUnusedFailure) { "fail" } else { "pass" }

    $report = [ordered]@{
        schema_version = 1
        status = $status
        source_root = Get-RelativeAuditPath -Root $repositoryRootFull -Path $appSourceRootFull
        counts = [ordered]@{
            contract_keys = $contractKeys.Count
            referenced_localization_keys = $referencedLocalizationKeys.Count
            resource_dictionaries = $dictionaryReports.Count
            unused_keys = $unusedKeys.Count
            missing_keys = $missingKeysFlat.Count
            extra_keys = $extraKeysFlat.Count
            duplicate_keys = $duplicateKeysFlat.Count
            unresolved_resource_references = $unresolvedResourceReferences.Count
        }
        contract_keys = $contractKeys
        referenced_localization_keys = $referencedLocalizationKeys
        resource_dictionaries = [object[]] $dictionaryReports
        unused_keys = $unusedKeys
        missing_keys = $missingKeysFlat
        extra_keys = $extraKeysFlat
        duplicate_keys = $duplicateKeysFlat
        unresolved_resource_references = $unresolvedResourceReferences
    }

    $json = $report | ConvertTo-Json -Depth 8
    if (-not [string]::IsNullOrWhiteSpace($ReportPath)) {
        $reportPathFull = [IO.Path]::GetFullPath($ReportPath)
        $reportDirectory = Split-Path -Parent $reportPathFull
        if (-not [string]::IsNullOrWhiteSpace($reportDirectory)) {
            [IO.Directory]::CreateDirectory($reportDirectory) | Out-Null
        }

        [IO.File]::WriteAllText($reportPathFull, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
    }

    Write-Host "Localization audit: $status"
    Write-Host "Contract keys: $($contractKeys.Count)"
    Write-Host "Referenced localization keys: $($referencedLocalizationKeys.Count)"
    Write-Host "Resource dictionaries: $($dictionaryReports.Count)"
    Write-Host "Unused localization keys: $($unusedKeys.Count)"
    Write-Host "Missing keys: $($missingKeysFlat.Count)"
    Write-Host "Extra keys: $($extraKeysFlat.Count)"
    Write-Host "Duplicate keys: $($duplicateKeysFlat.Count)"
    Write-Host "Unresolved WPF resource references: $($unresolvedResourceReferences.Count)"

    foreach ($dictionary in $dictionaryReports) {
        Write-Host "$($dictionary.culture): $($dictionary.defined_count) defined, $($dictionary.missing_keys.Count) missing, $($dictionary.extra_keys.Count) extra, $($dictionary.duplicate_keys.Count) duplicate"
    }

    if ($unusedKeys.Count -gt 0) { Write-Host "Unused: $($unusedKeys -join ', ')" }
    if ($missingKeysFlat.Count -gt 0) { Write-Host "Missing: $($missingKeysFlat -join ', ')" }
    if ($extraKeysFlat.Count -gt 0) { Write-Host "Extra: $($extraKeysFlat -join ', ')" }
    if ($duplicateKeysFlat.Count -gt 0) { Write-Host "Duplicate: $($duplicateKeysFlat -join ', ')" }
    if ($unresolvedResourceReferences.Count -gt 0) { Write-Host "Unresolved: $($unresolvedResourceReferences -join ', ')" }

    if ($hasRequiredFailure) { exit 1 }
    if ($hasStrictExtraFailure) { exit 2 }
    if ($hasStrictUnusedFailure) { exit 4 }
    exit 0
}
catch {
    [Console]::Error.WriteLine("Localization audit failed: $($_.Exception.Message)")
    exit 3
}
