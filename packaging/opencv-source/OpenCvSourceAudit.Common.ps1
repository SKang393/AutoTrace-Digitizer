# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

Set-StrictMode -Version Latest

function Read-OpenCvSourceLock {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "OpenCV source lock was not found: $Path"
    }

    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-RepositoryRevision {
    param([Parameter(Mandatory = $true)][string]$RepositoryPath)

    if (-not (Test-Path -LiteralPath (Join-Path $RepositoryPath '.git'))) {
        throw "Pinned source checkout is missing: $RepositoryPath"
    }

    $revision = (& git -C $RepositoryPath rev-parse HEAD 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to read source revision for '$RepositoryPath': $revision"
    }

    return $revision
}

function Read-CMakeCacheFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "CMake cache was not found: $Path"
    }

    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^(?<key>[^#/:][^:=]*):(?<type>[^=]+)=(?<value>.*)$') {
            $values[$Matches.key] = $Matches.value
        }
    }

    return $values
}

function Get-OpenCvCachePolicyErrors {
    param([Parameter(Mandatory = $true)][hashtable]$Cache)

    $errors = New-Object System.Collections.Generic.List[string]
    $required = [ordered]@{
        BUILD_LIST = 'core,imgproc,imgcodecs'
        BUILD_SHARED_LIBS = 'OFF'
        OPENCV_ENABLE_NONFREE = 'OFF'
        WITH_FFMPEG = 'OFF'
        WITH_GSTREAMER = 'OFF'
        WITH_IPP = 'OFF'
        WITH_ITT = 'OFF'
        WITH_JPEG = 'OFF'
        WITH_OPENCL = 'OFF'
        WITH_OPENEXR = 'OFF'
        WITH_OPENJPEG = 'OFF'
        WITH_PNG = 'OFF'
        WITH_PROTOBUF = 'OFF'
        WITH_TIFF = 'OFF'
        WITH_WEBP = 'OFF'
    }

    foreach ($entry in $required.GetEnumerator()) {
        if (-not $Cache.ContainsKey($entry.Key)) {
            $errors.Add("CMake cache does not declare $($entry.Key).")
        }
        elseif (-not [string]::Equals([string]$Cache[$entry.Key], [string]$entry.Value, [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add("CMake cache $($entry.Key) is '$($Cache[$entry.Key])', expected '$($entry.Value)'.")
        }
    }

    if ($Cache.ContainsKey('OPENCV_MODULES_BUILD')) {
        $allowedModules = @('opencv_core', 'opencv_imgcodecs', 'opencv_imgproc')
        foreach ($module in ([string]$Cache['OPENCV_MODULES_BUILD'] -split ';' | Where-Object { $_ })) {
            if ($allowedModules -notcontains $module) {
                $errors.Add("CMake cache enables an out-of-profile module: $module")
            }
        }
    }

    return $errors.ToArray()
}

function Get-OpenCvPreflightErrors {
    param(
        [Parameter(Mandatory = $true)][object]$Preflight,
        [Parameter(Mandatory = $true)][object]$Lock
    )

    $errors = New-Object System.Collections.Generic.List[string]
    $expected = [ordered]@{
        status = 'pass'
        profileId = [string]$Lock.profileId
        visualStudioInstallationVersion = [string]$Lock.toolchain.visualStudioInstallationVersion
        vcToolsVersion = [string]$Lock.toolchain.vcToolsVersion
        windowsSdkVersion = [string]$Lock.toolchain.windowsSdkVersion
        cmakeVersion = [string]$Lock.toolchain.cmakeVersion
        vcpkgRevision = [string]$Lock.sources.vcpkg.revision
        vcpkgToolVersion = [string]$Lock.sources.vcpkg.toolVersion
    }

    foreach ($entry in $expected.GetEnumerator()) {
        $property = $Preflight.PSObject.Properties[$entry.Key]
        if ($null -eq $property) {
            $errors.Add("Preflight evidence does not declare $($entry.Key).")
        }
        elseif (-not [string]::Equals(
                [string]$property.Value,
                [string]$entry.Value,
                [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add("Preflight $($entry.Key) is '$($property.Value)', expected '$($entry.Value)'.")
        }
    }

    foreach ($pathField in @('visualStudioPath', 'cmakePath', 'vcpkgPath', 'dumpbinPath')) {
        $property = $Preflight.PSObject.Properties[$pathField]
        if ($null -eq $property -or [string]::IsNullOrWhiteSpace([string]$property.Value)) {
            $errors.Add("Preflight evidence does not declare a nonempty $pathField.")
        }
    }

    return $errors.ToArray()
}

function Get-OpenCvToolchainCacheErrors {
    param(
        [Parameter(Mandatory = $true)][hashtable]$OpenCvCache,
        [Parameter(Mandatory = $true)][hashtable]$ExternCache,
        [Parameter(Mandatory = $true)][object]$Preflight,
        [Parameter(Mandatory = $true)][object]$Lock
    )

    $errors = New-Object System.Collections.Generic.List[string]
    $expectedCommon = [ordered]@{
        CMAKE_GENERATOR = [string]$Lock.toolchain.generator
        CMAKE_GENERATOR_PLATFORM = [string]$Lock.toolchain.platform
        CMAKE_GENERATOR_TOOLSET = [string]$Lock.toolchain.toolset
        CMAKE_SYSTEM_VERSION = [string]$Lock.toolchain.windowsSdkVersion
        VCPKG_TARGET_TRIPLET = 'graphreader-x64-windows-static'
        CMAKE_MSVC_RUNTIME_LIBRARY = 'MultiThreaded'
    }

    $visualStudioProperty = $Preflight.PSObject.Properties['visualStudioPath']
    if ($null -ne $visualStudioProperty) {
        $expectedCommon.CMAKE_GENERATOR_INSTANCE = [string]$visualStudioProperty.Value
    }

    foreach ($cacheEntry in @(
            @{ Name = 'OpenCV'; Values = $OpenCvCache },
            @{ Name = 'OpenCvSharpExtern'; Values = $ExternCache })) {
        foreach ($expectedEntry in $expectedCommon.GetEnumerator()) {
            if (-not $cacheEntry.Values.ContainsKey($expectedEntry.Key)) {
                $errors.Add("$($cacheEntry.Name) CMake cache does not declare $($expectedEntry.Key).")
            }
            elseif (-not [string]::Equals(
                    [string]$cacheEntry.Values[$expectedEntry.Key],
                    [string]$expectedEntry.Value,
                    [StringComparison]::OrdinalIgnoreCase)) {
                $errors.Add("$($cacheEntry.Name) CMake cache $($expectedEntry.Key) is '$($cacheEntry.Values[$expectedEntry.Key])', expected '$($expectedEntry.Value)'.")
            }
        }

        if (-not $cacheEntry.Values.ContainsKey('CMAKE_CONFIGURATION_TYPES') -or
            @([string]$cacheEntry.Values['CMAKE_CONFIGURATION_TYPES'] -split ';') -notcontains [string]$Lock.toolchain.configuration) {
            $errors.Add("$($cacheEntry.Name) CMake cache does not include the pinned '$($Lock.toolchain.configuration)' configuration.")
        }
        foreach ($flagName in @('CMAKE_C_FLAGS', 'CMAKE_CXX_FLAGS')) {
            if (-not $cacheEntry.Values.ContainsKey($flagName) -or
                [string]$cacheEntry.Values[$flagName] -notmatch '(?i)(?:^|\s)/Brepro(?:\s|$)') {
                $errors.Add("$($cacheEntry.Name) CMake cache $flagName does not retain /Brepro.")
            }
        }
    }

    if (-not $OpenCvCache.ContainsKey('CMAKE_INSTALL_PREFIX') -or
        -not [string]::Equals(
            ([string]$OpenCvCache['CMAKE_INSTALL_PREFIX'] -replace '\\', '/').TrimEnd('/'),
            ([string]$Lock.toolchain.canonicalInstallPrefix -replace '\\', '/').TrimEnd('/'),
            [StringComparison]::OrdinalIgnoreCase)) {
        $errors.Add('OpenCV CMake cache install prefix does not match the pinned canonical install prefix.')
    }
    if (-not [string]::Equals([string]$Lock.toolchain.crtLinkage, 'static', [StringComparison]::OrdinalIgnoreCase)) {
        $errors.Add("Unsupported source-lock CRT linkage '$($Lock.toolchain.crtLinkage)'.")
    }

    return $errors.ToArray()
}

function Get-ExactEvidenceInputErrors {
    param(
        [Parameter(Mandatory = $true)][string]$EvidenceRoot,
        [Parameter(Mandatory = $true)][string]$LockPath
    )

    $errors = New-Object System.Collections.Generic.List[string]
    $inputSourceRoot = Split-Path -Parent ([IO.Path]::GetFullPath($LockPath))
    foreach ($entry in @(
            @{ Relative = 'inputs/source-lock.json'; Expected = $LockPath },
            @{ Relative = 'inputs/opencv-minimal-cache.cmake'; Expected = (Join-Path $inputSourceRoot 'opencv-minimal-cache.cmake') },
            @{ Relative = 'inputs/x64-windows-static.cmake'; Expected = (Join-Path $inputSourceRoot 'x64-windows-static.cmake') })) {
        $actualPath = Join-Path $EvidenceRoot $entry.Relative
        if (-not (Test-Path -LiteralPath $entry.Expected -PathType Leaf)) {
            $errors.Add("Tracked OpenCV source-audit input is missing: $($entry.Expected)")
            continue
        }
        if (-not (Test-Path -LiteralPath $actualPath -PathType Leaf)) {
            continue
        }
        $expectedHash = (Get-FileHash -LiteralPath $entry.Expected -Algorithm SHA256).Hash
        $actualHash = (Get-FileHash -LiteralPath $actualPath -Algorithm SHA256).Hash
        if (-not [string]::Equals($expectedHash, $actualHash, [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add("Retained audit input does not match the tracked file byte-for-byte: $($entry.Relative)")
        }
    }

    return $errors.ToArray()
}

function Get-LinkerMapLibraries {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Linker map was not found: $Path"
    }

    $names = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $content = Get-Content -LiteralPath $Path -Raw
    foreach ($match in [regex]::Matches($content, '(?i)(?<name>[A-Za-z0-9_.+\-]+\.lib)(?=[:(\s])')) {
        [void]$names.Add($match.Groups['name'].Value)
    }
    foreach ($match in [regex]::Matches($content, '(?im)\s(?<name>[A-Za-z0-9_.+\-]+):[^\s:]+\.obj\s*$')) {
        $name = $match.Groups['name'].Value
        if (-not $name.EndsWith('.lib', [StringComparison]::OrdinalIgnoreCase)) {
            $name += '.lib'
        }
        [void]$names.Add($name)
    }

    return @($names | Sort-Object)
}

function Get-PeImportNames {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "PE import report was not found: $Path"
    }

    $names = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*(?<name>[A-Za-z0-9_.+\-]+\.dll)\s*$') {
            [void]$names.Add($Matches.name)
        }
    }

    return @($names | Sort-Object)
}

function Get-MsvcPathMapFlag {
    param(
        [Parameter(Mandatory = $true)][string]$ActualRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalRoot
    )

    $actual = [IO.Path]::GetFullPath($ActualRoot).TrimEnd('\', '/')
    $canonical = $CanonicalRoot.TrimEnd('\', '/') -replace '/', '\'
    if (-not [IO.Path]::IsPathRooted($canonical)) {
        throw "Canonical path-map root must be absolute: $CanonicalRoot"
    }
    if ($actual.IndexOf('=') -ge 0 -or $canonical.IndexOf('=') -ge 0) {
        throw 'MSVC path-map roots cannot contain an equals sign.'
    }

    return "/experimental:deterministic /pathmap:$actual=$canonical"
}

function Set-CanonicalOpenCvBuildMetadata {
    param(
        [Parameter(Mandatory = $true)][string]$HeaderPath,
        [Parameter(Mandatory = $true)][string]$CanonicalBuildRoot
    )

    if (-not (Test-Path -LiteralPath $HeaderPath -PathType Leaf)) {
        throw "OpenCV build metadata header was not found: $HeaderPath"
    }
    $content = Get-Content -LiteralPath $HeaderPath -Raw
    $pattern = '(?m)^#define OPENCV_BUILD_DIR "[^"]*"\r?$'
    $matches = [regex]::Matches($content, $pattern)
    if ($matches.Count -ne 1) {
        throw "Expected one OPENCV_BUILD_DIR definition in $HeaderPath, found $($matches.Count)."
    }

    $canonical = $CanonicalBuildRoot.TrimEnd('\', '/') -replace '\\', '/'
    $replacement = "#define OPENCV_BUILD_DIR `"$canonical`""
    $normalized = [regex]::Replace($content, $pattern, $replacement)
    [IO.File]::WriteAllText($HeaderPath, $normalized, (New-Object Text.UTF8Encoding($false)))
}

function Write-Sha256Manifest {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )

    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    $resolvedOutput = [IO.Path]::GetFullPath($OutputPath)
    $lines = foreach ($file in Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File | Sort-Object FullName) {
        if ([string]::Equals($file.FullName, $resolvedOutput, [StringComparison]::OrdinalIgnoreCase)) {
            continue
        }

        $relative = $file.FullName.Substring($resolvedRoot.Length).TrimStart('\', '/') -replace '\\', '/'
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash *$relative"
    }

    [IO.File]::WriteAllLines($OutputPath, $lines, (New-Object Text.UTF8Encoding($false)))
}

function Get-Sha256ManifestErrors {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$ManifestPath
    )

    $errors = New-Object System.Collections.Generic.List[string]
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        return @("SHA-256 manifest is missing: $ManifestPath")
    }

    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    $resolvedManifest = [IO.Path]::GetFullPath($ManifestPath)
    $listedPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($line in Get-Content -LiteralPath $ManifestPath) {
        if ($line -notmatch '^(?<hash>[0-9a-fA-F]{64}) \*(?<path>.+)$') {
            $errors.Add("Malformed SHA-256 manifest line: $line")
            continue
        }

        $relative = $Matches.path -replace '\\', '/'
        if (-not $listedPaths.Add($relative)) {
            $errors.Add("Duplicate SHA-256 manifest entry: $relative")
            continue
        }

        $path = [IO.Path]::GetFullPath((Join-Path $resolvedRoot ($relative -replace '/', '\')))
        $rootPrefix = $resolvedRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        if (-not $path.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add("SHA-256 manifest path escapes the evidence root: $relative")
            continue
        }
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            $errors.Add("Hashed evidence file is missing: $relative")
            continue
        }

        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if (-not [string]::Equals($actual, $Matches.hash, [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add("Hash mismatch for evidence file: $relative")
        }
    }

    foreach ($file in Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File) {
        if ([string]::Equals($file.FullName, $resolvedManifest, [StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        $relative = $file.FullName.Substring($resolvedRoot.Length).TrimStart('\', '/') -replace '\\', '/'
        if (-not $listedPaths.Contains($relative)) {
            $errors.Add("SHA-256 manifest omits evidence file: $relative")
        }
    }

    return $errors.ToArray()
}

function Get-OpenCvSourceAuditEvidenceErrors {
    param(
        [Parameter(Mandatory = $true)][string]$EvidenceRoot,
        [Parameter(Mandatory = $true)][string]$LockPath
    )

    $errors = New-Object System.Collections.Generic.List[string]
    $requiredFiles = @(
        'preflight.json',
        'source-revisions.json',
        'inputs/source-lock.json',
        'inputs/opencv-minimal-cache.cmake',
        'inputs/x64-windows-static.cmake',
        'build/opencv/CMakeCache.txt',
        'build/opencv/opencv-dependencies.dot',
        'build/extern/CMakeCache.txt',
        'build/extern/OpenCvSharpExtern-dependencies.dot',
        'build/extern/OpenCvSharpExtern.map',
        'bin/OpenCvSharpExtern.dll',
        'OpenCvSharpExtern.imports.txt',
        'dependency-inventory.json',
        'third-party-notices.reviewed.txt',
        'hashes.sha256'
    )

    foreach ($relative in $requiredFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $EvidenceRoot $relative) -PathType Leaf)) {
            $errors.Add("Required audit evidence is missing: $relative")
        }
    }

    $openCvCache = $null
    $cachePath = Join-Path $EvidenceRoot 'build/opencv/CMakeCache.txt'
    if (Test-Path -LiteralPath $cachePath -PathType Leaf) {
      try {
        $openCvCache = Read-CMakeCacheFile -Path $cachePath
        foreach ($errorText in Get-OpenCvCachePolicyErrors -Cache $openCvCache) {
            $errors.Add($errorText)
        }
      }
      catch {
        $errors.Add($_.Exception.Message)
      }
    }

    $lock = Read-OpenCvSourceLock -Path $LockPath
    foreach ($errorText in Get-ExactEvidenceInputErrors -EvidenceRoot $EvidenceRoot -LockPath $LockPath) {
        $errors.Add($errorText)
    }

    $preflight = $null
    $preflightPath = Join-Path $EvidenceRoot 'preflight.json'
    if (Test-Path -LiteralPath $preflightPath -PathType Leaf) {
        try {
            $preflight = Get-Content -LiteralPath $preflightPath -Raw | ConvertFrom-Json
            foreach ($errorText in Get-OpenCvPreflightErrors -Preflight $preflight -Lock $lock) {
                $errors.Add($errorText)
            }
        }
        catch {
            $errors.Add("Preflight evidence could not be validated: $($_.Exception.Message)")
        }
    }

    $externCache = $null
    $externCachePath = Join-Path $EvidenceRoot 'build/extern/CMakeCache.txt'
    if (Test-Path -LiteralPath $externCachePath -PathType Leaf) {
        try {
            $externCache = Read-CMakeCacheFile -Path $externCachePath
        }
        catch {
            $errors.Add($_.Exception.Message)
        }
    }
    if ($null -ne $openCvCache -and $null -ne $externCache -and $null -ne $preflight) {
        foreach ($errorText in Get-OpenCvToolchainCacheErrors -OpenCvCache $openCvCache -ExternCache $externCache -Preflight $preflight -Lock $lock) {
            $errors.Add($errorText)
        }
    }

    $revisionsPath = Join-Path $EvidenceRoot 'source-revisions.json'
    if (Test-Path -LiteralPath $revisionsPath -PathType Leaf) {
        $revisions = Get-Content -LiteralPath $revisionsPath -Raw | ConvertFrom-Json
        if (-not [string]::Equals($revisions.openCvSharp, $lock.sources.openCvSharp.revision, [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add('OpenCvSharp evidence revision does not match source-lock.json.')
        }
        if (-not [string]::Equals($revisions.openCv, $lock.sources.openCv.revision, [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add('OpenCV evidence revision does not match source-lock.json.')
        }
        if (-not [string]::Equals($revisions.vcpkg, $lock.sources.vcpkg.revision, [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add('vcpkg evidence revision does not match source-lock.json.')
        }
    }

    $inventoryPath = Join-Path $EvidenceRoot 'dependency-inventory.json'
    if (Test-Path -LiteralPath $inventoryPath -PathType Leaf) {
        $inventory = Get-Content -LiteralPath $inventoryPath -Raw | ConvertFrom-Json
        if (-not [string]::Equals([string]$inventory.reviewStatus, 'reviewed', [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add("Dependency inventory reviewStatus must be 'reviewed'.")
        }
        foreach ($dependency in @($inventory.dependencies)) {
            if ([string]::IsNullOrWhiteSpace([string]$dependency.name) -or
                [string]::IsNullOrWhiteSpace([string]$dependency.source) -or
                [string]::IsNullOrWhiteSpace([string]$dependency.license) -or
                -not [string]::Equals([string]$dependency.reviewStatus, 'reviewed', [StringComparison]::OrdinalIgnoreCase) -or
                @('included', 'not-required') -notcontains [string]$dependency.noticeDisposition) {
                $errors.Add("Dependency inventory entry is incomplete: $($dependency.name)")
            }
        }

        $expectedCoverage = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        $mapPath = Join-Path $EvidenceRoot 'build/extern/OpenCvSharpExtern.map'
        if (Test-Path -LiteralPath $mapPath -PathType Leaf) {
            foreach ($name in Get-LinkerMapLibraries -Path $mapPath) {
                [void]$expectedCoverage.Add("static-library|$name")
            }
        }
        $importsPath = Join-Path $EvidenceRoot 'OpenCvSharpExtern.imports.txt'
        if (Test-Path -LiteralPath $importsPath -PathType Leaf) {
            foreach ($name in Get-PeImportNames -Path $importsPath) {
                [void]$expectedCoverage.Add("pe-import|$name")
            }
        }

        $actualCoverage = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        foreach ($dependency in @($inventory.dependencies) | Where-Object { @('static-library', 'pe-import') -contains [string]$_.kind }) {
            $key = "$($dependency.kind)|$($dependency.name)"
            if (-not $actualCoverage.Add($key)) {
                $errors.Add("Duplicate dependency inventory coverage entry: $key")
            }
        }
        foreach ($key in $expectedCoverage) {
            if (-not $actualCoverage.Contains($key)) {
                $errors.Add("Dependency inventory omits observed binary dependency: $key")
            }
        }
        foreach ($key in $actualCoverage) {
            if (-not $expectedCoverage.Contains($key)) {
                $errors.Add("Dependency inventory contains unobserved binary dependency: $key")
            }
        }
    }

    $reviewedNoticesPath = Join-Path $EvidenceRoot 'third-party-notices.reviewed.txt'
    if (Test-Path -LiteralPath $reviewedNoticesPath -PathType Leaf) {
        $noticeText = Get-Content -LiteralPath $reviewedNoticesPath -Raw
        if ($noticeText -notmatch '(?im)^REVIEW STATUS:\s*COMPLETE\s*$') {
            $errors.Add('Reviewed notice file does not declare REVIEW STATUS: COMPLETE.')
        }
    }

    $hashesPath = Join-Path $EvidenceRoot 'hashes.sha256'
    if (Test-Path -LiteralPath $hashesPath -PathType Leaf) {
        foreach ($errorText in Get-Sha256ManifestErrors -Root $EvidenceRoot -ManifestPath $hashesPath) {
            $errors.Add($errorText)
        }
    }

    return $errors.ToArray()
}

function Get-OpenCvReproducibilityErrors {
    param(
        [Parameter(Mandatory = $true)][string]$FirstEvidenceRoot,
        [Parameter(Mandatory = $true)][string]$SecondEvidenceRoot
    )

    $errors = New-Object System.Collections.Generic.List[string]
    foreach ($relative in @(
        'inputs/source-lock.json',
        'inputs/opencv-minimal-cache.cmake',
        'inputs/x64-windows-static.cmake',
        'bin/OpenCvSharpExtern.dll',
        'build/extern/OpenCvSharpExtern.map'
    )) {
        $first = Join-Path $FirstEvidenceRoot $relative
        $second = Join-Path $SecondEvidenceRoot $relative
        if (-not (Test-Path -LiteralPath $first -PathType Leaf) -or -not (Test-Path -LiteralPath $second -PathType Leaf)) {
            $errors.Add("Reproducibility evidence is missing from one build: $relative")
            continue
        }
        $firstHash = (Get-FileHash -LiteralPath $first -Algorithm SHA256).Hash
        $secondHash = (Get-FileHash -LiteralPath $second -Algorithm SHA256).Hash
        if (-not [string]::Equals($firstHash, $secondHash, [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add("Reproducibility hash mismatch for ${relative}: $($firstHash.ToLowerInvariant()) != $($secondHash.ToLowerInvariant())")
        }
    }

    return $errors.ToArray()
}
