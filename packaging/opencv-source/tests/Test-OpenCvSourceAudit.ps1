# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$scriptRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $scriptRoot 'OpenCvSourceAudit.Common.ps1')

$script:passed = 0
$script:failed = 0

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw $Message
    }
}

function Invoke-Test {
    param([string]$Name, [scriptblock]$Body)
    try {
        & $Body
        $script:passed++
        Write-Host "PASS $Name"
    }
    catch {
        $script:failed++
        Write-Host "FAIL $Name"
        Write-Host $_.Exception.Message
    }
}

function Write-Utf8Text {
    param([string]$Path, [string]$Value)
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    [IO.File]::WriteAllText($Path, $Value, (New-Object Text.UTF8Encoding($false)))
}

function Write-JsonFixture {
    param([string]$Path, $Value)
    Write-Utf8Text -Path $Path -Value (($Value | ConvertTo-Json -Depth 20) + [Environment]::NewLine)
}

Invoke-Test 'CMake cache parser enforces the minimal nonfree-OFF profile' {
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("graphreader-opencv-cache-{0}" -f [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $temp -Force | Out-Null
    try {
        $cachePath = Join-Path $temp 'CMakeCache.txt'
        Write-Utf8Text -Path $cachePath -Value @'
BUILD_LIST:STRING=core,imgproc,imgcodecs
BUILD_SHARED_LIBS:BOOL=OFF
OPENCV_ENABLE_NONFREE:BOOL=OFF
WITH_FFMPEG:BOOL=OFF
WITH_GSTREAMER:BOOL=OFF
WITH_IPP:BOOL=OFF
WITH_ITT:BOOL=OFF
WITH_JPEG:BOOL=OFF
WITH_OPENCL:BOOL=OFF
WITH_OPENEXR:BOOL=OFF
WITH_OPENJPEG:BOOL=OFF
WITH_PNG:BOOL=OFF
WITH_PROTOBUF:BOOL=OFF
WITH_TIFF:BOOL=OFF
WITH_WEBP:BOOL=OFF
OPENCV_MODULES_BUILD:INTERNAL=opencv_core;opencv_imgcodecs;opencv_imgproc
'@
        $cache = Read-CMakeCacheFile -Path $cachePath
        Assert-True (@(Get-OpenCvCachePolicyErrors -Cache $cache).Count -eq 0) 'Expected compliant cache to pass.'
        $cache['OPENCV_ENABLE_NONFREE'] = 'ON'
        Assert-True (@(Get-OpenCvCachePolicyErrors -Cache $cache).Count -eq 1) 'Expected nonfree-enabled cache to fail.'
    }
    finally {
        Remove-Item -LiteralPath $temp -Recurse -Force
    }
}

Invoke-Test 'Linker map and PE import parsers return deterministic unique names' {
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("graphreader-opencv-map-{0}" -f [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $temp -Force | Out-Null
    try {
        $mapPath = Join-Path $temp 'OpenCvSharpExtern.map'
        Write-Utf8Text -Path $mapPath -Value @'
 0001:00000000 opencv_core4130.lib(matrix.cpp.obj)
 0001:00000010 opencv_imgproc4130:canny.obj
 0001:00000020 opencv_core4130:other.obj
'@
        $importsPath = Join-Path $temp 'imports.txt'
        Write-Utf8Text -Path $importsPath -Value "    KERNEL32.dll`r`n    VCRUNTIME140.dll`r`n    KERNEL32.dll`r`n"
        $libraries = @(Get-LinkerMapLibraries -Path $mapPath)
        $imports = @(Get-PeImportNames -Path $importsPath)
        Assert-True ($libraries.Count -eq 2) 'Expected two unique static libraries.'
        Assert-True ($libraries[0] -eq 'opencv_core4130.lib') 'Expected sorted OpenCV core library.'
        Assert-True ($imports.Count -eq 2) 'Expected two unique PE imports.'
        Assert-True ($imports[0] -eq 'KERNEL32.dll') 'Expected sorted PE imports.'
    }
    finally {
        Remove-Item -LiteralPath $temp -Recurse -Force
    }
}

Invoke-Test 'Path mapping and OpenCV build metadata normalize isolated roots' {
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("graphreader-opencv-pathmap-{0}" -f [Guid]::NewGuid().ToString('N'))
    $first = Join-Path $temp 'first/opencv_data_config.hpp'
    $second = Join-Path $temp 'second/opencv_data_config.hpp'
    try {
        Write-Utf8Text -Path $first -Value "#define OPENCV_BUILD_DIR `"C:/isolated/first/build/opencv`"`r`n"
        Write-Utf8Text -Path $second -Value "#define OPENCV_BUILD_DIR `"D:/isolated/second/build/opencv`"`r`n"
        $canonical = 'C:/GraphAutoReader/OpenCV-source-audit-evidence/build/opencv'
        Set-CanonicalOpenCvBuildMetadata -HeaderPath $first -CanonicalBuildRoot $canonical
        Set-CanonicalOpenCvBuildMetadata -HeaderPath $second -CanonicalBuildRoot $canonical
        Assert-True ((Get-FileHash $first -Algorithm SHA256).Hash -eq (Get-FileHash $second -Algorithm SHA256).Hash) 'Expected canonicalized build metadata to match.'

        $firstFlag = Get-MsvcPathMapFlag -ActualRoot (Join-Path $temp 'first') -CanonicalRoot 'C:/GraphAutoReader/OpenCV-source-audit-evidence'
        $secondFlag = Get-MsvcPathMapFlag -ActualRoot (Join-Path $temp 'second') -CanonicalRoot 'C:/GraphAutoReader/OpenCV-source-audit-evidence'
        Assert-True ($firstFlag -match '^/experimental:deterministic /pathmap:') 'Expected deterministic MSVC path-map options.'
        Assert-True ($firstFlag -ne $secondFlag) 'Expected each isolated source root to be mapped explicitly.'
        Assert-True ($firstFlag.EndsWith('=C:\GraphAutoReader\OpenCV-source-audit-evidence')) 'Expected the shared canonical path-map target.'
        Assert-True ($secondFlag.EndsWith('=C:\GraphAutoReader\OpenCV-source-audit-evidence')) 'Expected the shared canonical path-map target.'
    }
    finally {
        if (Test-Path -LiteralPath $temp) {
            Remove-Item -LiteralPath $temp -Recurse -Force
        }
    }
}

Invoke-Test 'Evidence validator fails closed when build evidence is absent' {
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("graphreader-opencv-empty-{0}" -f [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $temp -Force | Out-Null
    try {
        $errors = @(Get-OpenCvSourceAuditEvidenceErrors -EvidenceRoot $temp -LockPath (Join-Path $scriptRoot 'source-lock.json'))
        Assert-True ($errors.Count -ge 10) 'Expected missing evidence to remain blocked.'
        Assert-True (($errors -join "`n") -match 'OpenCvSharpExtern\.map') 'Expected linker map to be mandatory.'
        Assert-True (($errors -join "`n") -match 'third-party-notices\.reviewed\.txt') 'Expected reviewed notices to be mandatory.'
    }
    finally {
        Remove-Item -LiteralPath $temp -Recurse -Force
    }
}

Invoke-Test 'Evidence validator checks reviewed inventory, notices, and every hash' {
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("graphreader-opencv-complete-{0}" -f [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $temp -Force | Out-Null
    try {
        $lock = Read-OpenCvSourceLock -Path (Join-Path $scriptRoot 'source-lock.json')
        $preflightFixture = [ordered]@{
            status = 'pass'
            profileId = $lock.profileId
            visualStudioInstallationVersion = $lock.toolchain.visualStudioInstallationVersion
            visualStudioPath = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
            vcToolsVersion = $lock.toolchain.vcToolsVersion
            windowsSdkVersion = $lock.toolchain.windowsSdkVersion
            cmakeVersion = $lock.toolchain.cmakeVersion
            cmakePath = 'C:\Program Files\CMake\bin\cmake.exe'
            vcpkgRevision = $lock.sources.vcpkg.revision
            vcpkgToolVersion = $lock.sources.vcpkg.toolVersion
            vcpkgPath = 'C:\source\vcpkg'
            dumpbinPath = 'C:\toolchain\dumpbin.exe'
        }
        Write-JsonFixture -Path (Join-Path $temp 'preflight.json') -Value $preflightFixture
        Write-JsonFixture -Path (Join-Path $temp 'source-revisions.json') -Value @{
            openCvSharp = $lock.sources.openCvSharp.revision
            openCv = $lock.sources.openCv.revision
            vcpkg = $lock.sources.vcpkg.revision
        }
        New-Item -ItemType Directory -Path (Join-Path $temp 'inputs') -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $scriptRoot 'source-lock.json') -Destination (Join-Path $temp 'inputs/source-lock.json')
        Copy-Item -LiteralPath (Join-Path $scriptRoot 'opencv-minimal-cache.cmake') -Destination (Join-Path $temp 'inputs/opencv-minimal-cache.cmake')
        Copy-Item -LiteralPath (Join-Path $scriptRoot 'x64-windows-static.cmake') -Destination (Join-Path $temp 'inputs/x64-windows-static.cmake')
        $commonToolchainCache = @"
CMAKE_CONFIGURATION_TYPES:STRING=Debug;Release
CMAKE_CXX_FLAGS:STRING=/Brepro
CMAKE_C_FLAGS:STRING=/Brepro
CMAKE_GENERATOR_INSTANCE:UNINITIALIZED=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
CMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreaded
CMAKE_SYSTEM_VERSION:UNINITIALIZED=$($lock.toolchain.windowsSdkVersion)
VCPKG_TARGET_TRIPLET:STRING=graphreader-x64-windows-static
CMAKE_GENERATOR:INTERNAL=$($lock.toolchain.generator)
CMAKE_GENERATOR_PLATFORM:INTERNAL=$($lock.toolchain.platform)
CMAKE_GENERATOR_TOOLSET:INTERNAL=$($lock.toolchain.toolset)
"@
        $cache = @"
BUILD_LIST:STRING=core,imgproc,imgcodecs
BUILD_SHARED_LIBS:BOOL=OFF
OPENCV_ENABLE_NONFREE:BOOL=OFF
WITH_FFMPEG:BOOL=OFF
WITH_GSTREAMER:BOOL=OFF
WITH_IPP:BOOL=OFF
WITH_ITT:BOOL=OFF
WITH_JPEG:BOOL=OFF
WITH_OPENCL:BOOL=OFF
WITH_OPENEXR:BOOL=OFF
WITH_OPENJPEG:BOOL=OFF
WITH_PNG:BOOL=OFF
WITH_PROTOBUF:BOOL=OFF
WITH_TIFF:BOOL=OFF
WITH_WEBP:BOOL=OFF
OPENCV_MODULES_BUILD:INTERNAL=opencv_core;opencv_imgcodecs;opencv_imgproc
$commonToolchainCache
CMAKE_INSTALL_PREFIX:PATH=$($lock.toolchain.canonicalInstallPrefix)
"@
        Write-Utf8Text -Path (Join-Path $temp 'build/opencv/CMakeCache.txt') -Value $cache
        Write-Utf8Text -Path (Join-Path $temp 'build/opencv/opencv-dependencies.dot') -Value 'digraph dependencies {}'
        $externCache = $commonToolchainCache
        Write-Utf8Text -Path (Join-Path $temp 'build/extern/CMakeCache.txt') -Value $externCache
        Write-Utf8Text -Path (Join-Path $temp 'build/extern/OpenCvSharpExtern-dependencies.dot') -Value 'digraph dependencies {}'
        Write-Utf8Text -Path (Join-Path $temp 'build/extern/OpenCvSharpExtern.map') -Value 'opencv_core4130.lib(core.obj)'
        Write-Utf8Text -Path (Join-Path $temp 'bin/OpenCvSharpExtern.dll') -Value 'deterministic fixture payload'
        Write-Utf8Text -Path (Join-Path $temp 'OpenCvSharpExtern.imports.txt') -Value '    KERNEL32.dll'
        Write-JsonFixture -Path (Join-Path $temp 'dependency-inventory.json') -Value @{
            reviewStatus = 'reviewed'
            dependencies = @(
                @{
                    name = 'opencv_core4130.lib'
                    kind = 'static-library'
                    source = 'pinned OpenCV source'
                    license = 'Apache-2.0'
                    noticeDisposition = 'included'
                    reviewStatus = 'reviewed'
                },
                @{
                    name = 'KERNEL32.dll'
                    kind = 'pe-import'
                    source = 'Windows SDK system component'
                    license = 'Microsoft Windows system component'
                    noticeDisposition = 'not-required'
                    reviewStatus = 'reviewed'
                }
            )
        }
        Write-Utf8Text -Path (Join-Path $temp 'third-party-notices.reviewed.txt') -Value "REVIEW STATUS: COMPLETE`r`nFixture notice"
        Write-Sha256Manifest -Root $temp -OutputPath (Join-Path $temp 'hashes.sha256')
        $errors = @(Get-OpenCvSourceAuditEvidenceErrors -EvidenceRoot $temp -LockPath (Join-Path $scriptRoot 'source-lock.json'))
        Assert-True ($errors.Count -eq 0) ("Expected complete fixture evidence to pass: " + ($errors -join '; '))

        $manifestPath = Join-Path $temp 'hashes.sha256'

        Write-Utf8Text -Path (Join-Path $temp 'inputs/source-lock.json') -Value '{}'
        Write-Sha256Manifest -Root $temp -OutputPath $manifestPath
        $inputErrors = @(Get-OpenCvSourceAuditEvidenceErrors -EvidenceRoot $temp -LockPath (Join-Path $scriptRoot 'source-lock.json'))
        Assert-True (($inputErrors -join "`n") -match 'Retained audit input does not match') 'Expected a retained-input mismatch to fail.'
        Copy-Item -LiteralPath (Join-Path $scriptRoot 'source-lock.json') -Destination (Join-Path $temp 'inputs/source-lock.json') -Force

        $preflightFixture.profileId = 'wrong-profile'
        Write-JsonFixture -Path (Join-Path $temp 'preflight.json') -Value $preflightFixture
        Write-Sha256Manifest -Root $temp -OutputPath $manifestPath
        $preflightErrors = @(Get-OpenCvSourceAuditEvidenceErrors -EvidenceRoot $temp -LockPath (Join-Path $scriptRoot 'source-lock.json'))
        Assert-True (($preflightErrors -join "`n") -match 'Preflight profileId') 'Expected a preflight profile mismatch to fail.'
        $preflightFixture.profileId = $lock.profileId
        Write-JsonFixture -Path (Join-Path $temp 'preflight.json') -Value $preflightFixture

        $mutatedExternCache = $externCache.Replace('CMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreaded', 'CMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreadedDLL')
        Write-Utf8Text -Path (Join-Path $temp 'build/extern/CMakeCache.txt') -Value $mutatedExternCache
        Write-Sha256Manifest -Root $temp -OutputPath $manifestPath
        $toolchainErrors = @(Get-OpenCvSourceAuditEvidenceErrors -EvidenceRoot $temp -LockPath (Join-Path $scriptRoot 'source-lock.json'))
        Assert-True (($toolchainErrors -join "`n") -match 'OpenCvSharpExtern CMake cache CMAKE_MSVC_RUNTIME_LIBRARY') 'Expected a toolchain cache mismatch to fail.'
        Write-Utf8Text -Path (Join-Path $temp 'build/extern/CMakeCache.txt') -Value $externCache
        Write-Sha256Manifest -Root $temp -OutputPath $manifestPath

        $manifestLines = @(Get-Content -LiteralPath $manifestPath | Where-Object { $_ -notmatch '\*bin/OpenCvSharpExtern\.dll$' })
        [IO.File]::WriteAllLines($manifestPath, $manifestLines, (New-Object Text.UTF8Encoding($false)))
        $omissionErrors = @(Get-Sha256ManifestErrors -Root $temp -ManifestPath $manifestPath)
        Assert-True (($omissionErrors -join "`n") -match 'omits evidence file: bin/OpenCvSharpExtern\.dll') 'Expected an omitted evidence hash to fail.'
        Write-Sha256Manifest -Root $temp -OutputPath $manifestPath

        Write-JsonFixture -Path (Join-Path $temp 'dependency-inventory.json') -Value @{
            reviewStatus = 'reviewed'
            dependencies = @(@{
                name = 'opencv_core4130.lib'
                kind = 'static-library'
                source = 'pinned OpenCV source'
                license = 'Apache-2.0'
                noticeDisposition = 'included'
                reviewStatus = 'reviewed'
            })
        }
        Write-Sha256Manifest -Root $temp -OutputPath $manifestPath
        $coverageErrors = @(Get-OpenCvSourceAuditEvidenceErrors -EvidenceRoot $temp -LockPath (Join-Path $scriptRoot 'source-lock.json'))
        Assert-True (($coverageErrors -join "`n") -match 'omits observed binary dependency: pe-import\|KERNEL32\.dll') 'Expected omitted PE import inventory coverage to fail.'

        Write-Utf8Text -Path (Join-Path $temp 'bin/OpenCvSharpExtern.dll') -Value 'tampered fixture payload'
        $tamperErrors = @(Get-OpenCvSourceAuditEvidenceErrors -EvidenceRoot $temp -LockPath (Join-Path $scriptRoot 'source-lock.json'))
        Assert-True (($tamperErrors -join "`n") -match 'Hash mismatch') 'Expected tampered binary hash to fail.'
    }
    finally {
        Remove-Item -LiteralPath $temp -Recurse -Force
    }
}

Invoke-Test 'Reproducibility guard compares retained inputs, binary, and linker map' {
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("graphreader-opencv-repro-{0}" -f [Guid]::NewGuid().ToString('N'))
    $first = Join-Path $temp 'first'
    $second = Join-Path $temp 'second'
    try {
        foreach ($relative in @(
            'inputs/source-lock.json',
            'inputs/opencv-minimal-cache.cmake',
            'inputs/x64-windows-static.cmake',
            'bin/OpenCvSharpExtern.dll',
            'build/extern/OpenCvSharpExtern.map'
        )) {
            Write-Utf8Text -Path (Join-Path $first $relative) -Value "same $relative"
            Write-Utf8Text -Path (Join-Path $second $relative) -Value "same $relative"
        }
        Assert-True (@(Get-OpenCvReproducibilityErrors -FirstEvidenceRoot $first -SecondEvidenceRoot $second).Count -eq 0) 'Expected identical builds to pass.'
        Write-Utf8Text -Path (Join-Path $second 'bin/OpenCvSharpExtern.dll') -Value 'different binary'
        $errors = @(Get-OpenCvReproducibilityErrors -FirstEvidenceRoot $first -SecondEvidenceRoot $second)
        Assert-True (($errors -join "`n") -match 'bin/OpenCvSharpExtern\.dll') 'Expected binary mismatch to fail.'
    }
    finally {
        if (Test-Path -LiteralPath $temp) {
            Remove-Item -LiteralPath $temp -Recurse -Force
        }
    }
}

Write-Host "$script:passed passed, $script:failed failed"
if ($script:failed -gt 0) {
    exit 1
}
