# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [ValidateSet('Preflight', 'Configure', 'Build', 'Collect', 'Validate', 'All')]
    [string]$Phase = 'Preflight',
    [string]$SourceRoot,
    [string]$EvidenceRoot,
    [ValidateRange(1, 64)]
    [int]$Jobs = 4
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'OpenCvSourceAudit.Common.ps1')

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $projectRoot 'artifacts\goal19-opencv-source\sources'
}
if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) {
    $EvidenceRoot = Join-Path $projectRoot 'artifacts\goal19-opencv-source\evidence'
}
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$EvidenceRoot = [IO.Path]::GetFullPath($EvidenceRoot)
$lockPath = Join-Path $PSScriptRoot 'source-lock.json'
$lock = Read-OpenCvSourceLock -Path $lockPath
$openCvSharpPath = Join-Path $SourceRoot 'opencvsharp'
$openCvPath = Join-Path $SourceRoot 'opencv'
$vcpkgPath = Join-Path $SourceRoot 'vcpkg'
$openCvBuild = Join-Path $EvidenceRoot 'build\opencv'
$externBuild = Join-Path $EvidenceRoot 'build\extern'
$openCvInstall = Join-Path $EvidenceRoot 'stage\opencv'
$binRoot = Join-Path $EvidenceRoot 'bin'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'

function Write-JsonUtf8 {
    param([Parameter(Mandatory = $true)]$Value, [Parameter(Mandatory = $true)][string]$Path)
    $json = $Value | ConvertTo-Json -Depth 20
    [IO.File]::WriteAllText($Path, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description
    )

    Write-Host "[$Description] $Executable $($Arguments -join ' ')"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Invoke-CheckedWithCompilerPathMap {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $previousCl = [Environment]::GetEnvironmentVariable('CL', [EnvironmentVariableTarget]::Process)
    $pathMap = Get-MsvcPathMapFlag -ActualRoot $EvidenceRoot -CanonicalRoot ([string]$lock.toolchain.canonicalEvidenceRoot)
    try {
        $env:CL = if ([string]::IsNullOrWhiteSpace($previousCl)) { $pathMap } else { "$previousCl $pathMap" }
        Invoke-Checked -Executable $Executable -Arguments $Arguments -Description $Description
    }
    finally {
        if ($null -eq $previousCl) {
            Remove-Item Env:CL -ErrorAction SilentlyContinue
        }
        else {
            $env:CL = $previousCl
        }
    }
}

function Invoke-Preflight {
    New-Item -ItemType Directory -Path $EvidenceRoot -Force | Out-Null

    foreach ($entry in @(
        @{ Name = 'OpenCvSharp'; Path = $openCvSharpPath; Expected = [string]$lock.sources.openCvSharp.revision },
        @{ Name = 'OpenCV'; Path = $openCvPath; Expected = [string]$lock.sources.openCv.revision },
        @{ Name = 'vcpkg'; Path = $vcpkgPath; Expected = [string]$lock.sources.vcpkg.revision }
    )) {
        $actual = Get-RepositoryRevision -RepositoryPath $entry.Path
        if (-not [string]::Equals($actual, $entry.Expected, [StringComparison]::OrdinalIgnoreCase)) {
            throw "$($entry.Name) checkout is $actual, expected $($entry.Expected)."
        }
    }

    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw "vswhere.exe is missing: $vswhere"
    }
    $vsPath = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Out-String).Trim()
    $vsVersion = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion | Out-String).Trim()
    if (-not [string]::Equals($vsVersion, [string]$lock.toolchain.visualStudioInstallationVersion, [StringComparison]::Ordinal)) {
        throw "Visual Studio installation version is '$vsVersion', expected '$($lock.toolchain.visualStudioInstallationVersion)'."
    }

    $vcToolsPath = Join-Path $vsPath ("VC\Tools\MSVC\{0}" -f $lock.toolchain.vcToolsVersion)
    $sdkLibPath = Join-Path ${env:ProgramFiles(x86)} ("Windows Kits\10\Lib\{0}" -f $lock.toolchain.windowsSdkVersion)
    $cmake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    $dumpbin = Join-Path $vcToolsPath 'bin\Hostx64\x64\dumpbin.exe'
    $vcpkgExe = Join-Path $vcpkgPath 'vcpkg.exe'
    foreach ($requiredPath in @($vcToolsPath, $sdkLibPath, $cmake, $dumpbin, $vcpkgExe)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Pinned toolchain input is missing: $requiredPath"
        }
    }

    $cmakeVersionLine = (& $cmake --version | Select-Object -First 1)
    $cmakeVersion = ([regex]::Match($cmakeVersionLine, 'cmake version\s+(?<version>\S+)')).Groups['version'].Value
    if (-not [string]::Equals($cmakeVersion, [string]$lock.toolchain.cmakeVersion, [StringComparison]::Ordinal)) {
        throw "CMake version is '$cmakeVersion', expected '$($lock.toolchain.cmakeVersion)'."
    }

    $vcpkgVersionLine = (& $vcpkgExe version | Select-Object -First 1)
    $vcpkgVersion = ([regex]::Match($vcpkgVersionLine, 'version\s+(?<version>\S+)')).Groups['version'].Value
    if (-not [string]::Equals($vcpkgVersion, [string]$lock.sources.vcpkg.toolVersion, [StringComparison]::Ordinal)) {
        throw "vcpkg tool version is '$vcpkgVersion', expected '$($lock.sources.vcpkg.toolVersion)'."
    }

    $preflight = [ordered]@{
        status = 'pass'
        checkedUtc = [DateTime]::UtcNow.ToString('o')
        profileId = [string]$lock.profileId
        visualStudioInstallationVersion = $vsVersion
        visualStudioPath = $vsPath
        vcToolsVersion = [string]$lock.toolchain.vcToolsVersion
        windowsSdkVersion = [string]$lock.toolchain.windowsSdkVersion
        cmakeVersion = $cmakeVersion
        cmakePath = $cmake
        vcpkgRevision = [string]$lock.sources.vcpkg.revision
        vcpkgToolVersion = $vcpkgVersion
        vcpkgPath = $vcpkgPath
        dumpbinPath = $dumpbin
    }
    Write-JsonUtf8 -Value $preflight -Path (Join-Path $EvidenceRoot 'preflight.json')
    Write-JsonUtf8 -Value ([ordered]@{
        openCvSharp = Get-RepositoryRevision -RepositoryPath $openCvSharpPath
        openCv = Get-RepositoryRevision -RepositoryPath $openCvPath
        vcpkg = Get-RepositoryRevision -RepositoryPath $vcpkgPath
    }) -Path (Join-Path $EvidenceRoot 'source-revisions.json')
    return $preflight
}

function Invoke-ConfigureOpenCv {
    $preflight = Invoke-Preflight
    New-Item -ItemType Directory -Path $openCvBuild, $openCvInstall -Force | Out-Null
    $cmake = [string]$preflight.cmakePath
    $toolchain = Join-Path $vcpkgPath 'scripts\buildsystems\vcpkg.cmake'
    $graph = Join-Path $openCvBuild 'opencv-dependencies.dot'
    $arguments = @(
        '-C', (Join-Path $PSScriptRoot 'opencv-minimal-cache.cmake'),
        '-S', $openCvPath,
        '-B', $openCvBuild,
        '-G', [string]$lock.toolchain.generator,
        '-A', [string]$lock.toolchain.platform,
        '-T', [string]$lock.toolchain.toolset,
        "--graphviz=$graph",
        "-DCMAKE_GENERATOR_INSTANCE=$($preflight.visualStudioPath)",
        "-DCMAKE_SYSTEM_VERSION=$($lock.toolchain.windowsSdkVersion)",
        "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
        '-DVCPKG_MANIFEST_MODE=OFF',
        '-DVCPKG_TARGET_TRIPLET=graphreader-x64-windows-static',
        "-DVCPKG_OVERLAY_TRIPLETS=$PSScriptRoot",
        "-DCMAKE_INSTALL_PREFIX=$($lock.toolchain.canonicalInstallPrefix)"
    )
    Invoke-Checked -Executable $cmake -Arguments $arguments -Description 'Configure pinned OpenCV'

    $cache = Read-CMakeCacheFile -Path (Join-Path $openCvBuild 'CMakeCache.txt')
    $policyErrors = @(Get-OpenCvCachePolicyErrors -Cache $cache)
    if ($policyErrors.Count -gt 0) {
        throw "Configured OpenCV violates the minimal policy:`n$($policyErrors -join [Environment]::NewLine)"
    }

    $metadataHeader = Join-Path $openCvBuild 'opencv_data_config.hpp'
    $canonicalBuildRoot = ([string]$lock.toolchain.canonicalEvidenceRoot).TrimEnd('/', '\') + '/build/opencv'
    Set-CanonicalOpenCvBuildMetadata -HeaderPath $metadataHeader -CanonicalBuildRoot $canonicalBuildRoot
}

function Invoke-BuildNative {
    $preflight = Invoke-Preflight
    if (-not (Test-Path -LiteralPath (Join-Path $openCvBuild 'CMakeCache.txt'))) {
        Invoke-ConfigureOpenCv
    }
    $cmake = [string]$preflight.cmakePath
    Invoke-CheckedWithCompilerPathMap -Executable $cmake -Arguments @('--build', $openCvBuild, '--config', 'Release', '--parallel', [string]$Jobs) -Description 'Build pinned OpenCV'
    Invoke-Checked -Executable $cmake -Arguments @('--install', $openCvBuild, '--config', 'Release', '--prefix', $openCvInstall) -Description 'Install pinned OpenCV'

    $openCvConfig = Get-ChildItem -LiteralPath $openCvInstall -Filter OpenCVConfig.cmake -Recurse -File |
        Sort-Object @{ Expression = { if ($_.FullName -match '[\\/]x64[\\/]vc17[\\/]staticlib[\\/]') { 0 } else { 1 } } }, FullName |
        Select-Object -First 1
    if ($null -eq $openCvConfig) {
        throw "OpenCVConfig.cmake was not installed under $openCvInstall"
    }

    New-Item -ItemType Directory -Path $externBuild -Force | Out-Null
    $externGraph = Join-Path $externBuild 'OpenCvSharpExtern-dependencies.dot'
    $linkerMap = Join-Path $externBuild 'OpenCvSharpExtern.map'
    $linkerFlags = "/Brepro /MAP:`"$linkerMap`" /MAPINFO:EXPORTS /INCREMENTAL:NO /OPT:REF /OPT:ICF"
    $toolchain = Join-Path $vcpkgPath 'scripts\buildsystems\vcpkg.cmake'
    $arguments = @(
        '-S', (Join-Path $openCvSharpPath 'src'),
        '-B', $externBuild,
        '-G', [string]$lock.toolchain.generator,
        '-A', [string]$lock.toolchain.platform,
        '-T', [string]$lock.toolchain.toolset,
        "--graphviz=$externGraph",
        "-DCMAKE_GENERATOR_INSTANCE=$($preflight.visualStudioPath)",
        "-DCMAKE_SYSTEM_VERSION=$($lock.toolchain.windowsSdkVersion)",
        "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
        '-DVCPKG_MANIFEST_MODE=OFF',
        '-DVCPKG_TARGET_TRIPLET=graphreader-x64-windows-static',
        "-DVCPKG_OVERLAY_TRIPLETS=$PSScriptRoot",
        "-DOpenCV_DIR=$($openCvConfig.Directory.FullName)",
        "-DCMAKE_PREFIX_PATH=$openCvInstall",
        '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded',
        '-DCMAKE_C_FLAGS=/DWIN32 /D_WINDOWS /W3 /Brepro',
        '-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /W3 /GR /EHsc /Brepro',
        "-DCMAKE_SHARED_LINKER_FLAGS_RELEASE=$linkerFlags",
        '-DNO_CONTRIB=ON',
        '-DNO_STITCHING=ON',
        '-DNO_CALIB3D=ON',
        '-DNO_VIDEO=ON',
        '-DNO_FEATURES2D=ON',
        '-DNO_FLANN=ON',
        '-DNO_DNN=ON',
        '-DNO_ML=ON',
        '-DNO_OBJDETECT=ON',
        '-DNO_PHOTO=ON',
        '-DNO_BARCODE=ON',
        '-DNO_HIGHGUI=ON',
        '-DNO_VIDEOIO=ON',
        '-DNO_INSTALL_TO_TEST=ON'
    )
    Invoke-Checked -Executable $cmake -Arguments $arguments -Description 'Configure pinned OpenCvSharpExtern'
    Invoke-CheckedWithCompilerPathMap -Executable $cmake -Arguments @('--build', $externBuild, '--config', 'Release', '--parallel', [string]$Jobs) -Description 'Build pinned OpenCvSharpExtern'
}

function Invoke-CollectEvidence {
    $preflight = Invoke-Preflight
    $dll = Join-Path $externBuild 'OpenCvSharpExtern\Release\OpenCvSharpExtern.dll'
    $map = Join-Path $externBuild 'OpenCvSharpExtern.map'
    foreach ($required in @($dll, $map, (Join-Path $openCvBuild 'CMakeCache.txt'), (Join-Path $openCvBuild 'opencv-dependencies.dot'), (Join-Path $externBuild 'CMakeCache.txt'), (Join-Path $externBuild 'OpenCvSharpExtern-dependencies.dot'))) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Native build evidence is missing: $required"
        }
    }

    New-Item -ItemType Directory -Path $binRoot -Force | Out-Null
    $inputRoot = Join-Path $EvidenceRoot 'inputs'
    New-Item -ItemType Directory -Path $inputRoot -Force | Out-Null
    Copy-Item -LiteralPath $lockPath -Destination (Join-Path $inputRoot 'source-lock.json') -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'opencv-minimal-cache.cmake') -Destination (Join-Path $inputRoot 'opencv-minimal-cache.cmake') -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'x64-windows-static.cmake') -Destination (Join-Path $inputRoot 'x64-windows-static.cmake') -Force
    Copy-Item -LiteralPath $dll -Destination (Join-Path $binRoot 'OpenCvSharpExtern.dll') -Force
    $importsPath = Join-Path $EvidenceRoot 'OpenCvSharpExtern.imports.txt'
    & ([string]$preflight.dumpbinPath) /nologo /imports (Join-Path $binRoot 'OpenCvSharpExtern.dll') | Out-File -LiteralPath $importsPath -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        throw "dumpbin import inspection failed with exit code $LASTEXITCODE."
    }

    $libraries = @(Get-LinkerMapLibraries -Path $map)
    $imports = @(Get-PeImportNames -Path $importsPath)
    $dependencies = New-Object System.Collections.Generic.List[object]
    foreach ($name in $libraries) {
        $source = $null
        $license = $null
        if ($name -match '^opencv_(core|imgproc|imgcodecs)') {
            $source = "https://github.com/opencv/opencv/tree/$($lock.sources.openCv.revision)"
            $license = 'Apache-2.0'
        }
        $dependencies.Add([ordered]@{
            name = $name
            kind = 'static-library'
            source = $source
            license = $license
            noticeDisposition = 'unresolved'
            reviewStatus = 'requires-review'
        })
    }
    foreach ($name in $imports) {
        $dependencies.Add([ordered]@{
            name = $name
            kind = 'pe-import'
            source = $null
            license = $null
            noticeDisposition = 'unresolved'
            reviewStatus = 'requires-review'
        })
    }
    $installedLicenseFiles = @(Get-ChildItem -LiteralPath (Join-Path $openCvInstall 'etc\licenses') -File -ErrorAction SilentlyContinue | Sort-Object Name)
    foreach ($licenseFile in $installedLicenseFiles) {
        $dependencies.Add([ordered]@{
            name = $licenseFile.Name
            kind = 'embedded-source-license'
            source = "https://github.com/opencv/opencv/tree/$($lock.sources.openCv.revision)"
            license = $null
            noticeDisposition = 'unresolved'
            reviewStatus = 'requires-review'
        })
    }

    Write-JsonUtf8 -Value ([ordered]@{
        schemaVersion = 1
        profileId = [string]$lock.profileId
        reviewStatus = 'requires-review'
        generatedUtc = [DateTime]::UtcNow.ToString('o')
        binarySha256 = (Get-FileHash -LiteralPath (Join-Path $binRoot 'OpenCvSharpExtern.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
        dependencies = $dependencies.ToArray()
        warning = 'This mechanically extracted inventory is not a completed license review.'
    }) -Path (Join-Path $EvidenceRoot 'dependency-inventory.json')

    $candidateNotice = Join-Path $EvidenceRoot 'third-party-notices.candidate.txt'
    $noticeParts = New-Object System.Collections.Generic.List[string]
    foreach ($part in @(
        'REVIEW STATUS: INCOMPLETE',
        'This candidate combines pinned source license texts. It is not approved for distribution until every linker-map and PE-import entry is reconciled.',
        '',
        '===== OpenCvSharp pinned source license =====',
        (Get-Content -LiteralPath (Join-Path $openCvSharpPath 'LICENSE') -Raw),
        '',
        '===== OpenCV pinned source license =====',
        (Get-Content -LiteralPath (Join-Path $openCvPath 'LICENSE') -Raw)
    )) {
        $noticeParts.Add([string]$part)
    }
    foreach ($licenseFile in $installedLicenseFiles) {
        $noticeParts.Add('')
        $noticeParts.Add("===== OpenCV installed component license: $($licenseFile.Name) =====")
        $noticeParts.Add((Get-Content -LiteralPath $licenseFile.FullName -Raw))
    }
    [IO.File]::WriteAllText($candidateNotice, ($noticeParts.ToArray() -join [Environment]::NewLine), (New-Object Text.UTF8Encoding($false)))
    Write-Sha256Manifest -Root $EvidenceRoot -OutputPath (Join-Path $EvidenceRoot 'hashes.sha256')
    Write-Host 'Raw build evidence collected. Audit remains blocked until dependency-inventory.json is reviewed and third-party-notices.reviewed.txt is approved.'
}

function Invoke-ValidateEvidence {
    $errors = @(Get-OpenCvSourceAuditEvidenceErrors -EvidenceRoot $EvidenceRoot -LockPath $lockPath)
    if ($errors.Count -gt 0) {
        throw "OpenCV source audit is incomplete:`n$($errors -join [Environment]::NewLine)"
    }
    Write-Host 'OpenCV source audit evidence: PASS'
}

switch ($Phase) {
    'Preflight' { Invoke-Preflight | Out-Null }
    'Configure' { Invoke-ConfigureOpenCv }
    'Build' { Invoke-BuildNative }
    'Collect' { Invoke-CollectEvidence }
    'Validate' { Invoke-ValidateEvidence }
    'All' {
        Invoke-ConfigureOpenCv
        Invoke-BuildNative
        Invoke-CollectEvidence
        Invoke-ValidateEvidence
    }
}
