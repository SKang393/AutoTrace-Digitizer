[CmdletBinding()]
param(
  [ValidateSet("release", "debug")]
  [string] $Configuration = "release",

  [string[]] $Config = @("log4cpp_null"),

  [string] $BuildDir = "",

  [string] $OutputDir = "",

  [switch] $SkipBuild,

  [switch] $Clean,

  [switch] $NoZip,

  [string[]] $QMakeArg = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProductName = "AutoTrace Digitizer"
$ReleaseVersion = "v1.0.0"
$BuildExecutableFileName = "Engauge.exe"
$PortableExecutableBaseName = "AutoTraceDigitizer"
$PortableExecutableFileName = "$PortableExecutableBaseName.exe"
$PortableFolderName = "AutoTrace-Digitizer-Windows-Portable"
$ReleaseZipName = "AutoTrace-Digitizer-$ReleaseVersion-Windows-Portable.zip"

function ConvertTo-FullPath {
  param([Parameter(Mandatory = $true)][string] $Path)
  return [System.IO.Path]::GetFullPath($Path)
}

function ConvertTo-BuildToolPath {
  param([Parameter(Mandatory = $true)][string] $Path)

  $fullPath = ConvertTo-FullPath $Path

  if (-not [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    return $fullPath
  }

  if ($null -eq ("Kernel32ShortPath" -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class Kernel32ShortPath
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern uint GetShortPathName(string longPath, StringBuilder shortPath, uint bufferLength);
}
"@
  }

  $buffer = New-Object System.Text.StringBuilder 32768
  $length = [Kernel32ShortPath]::GetShortPathName($fullPath, $buffer, [uint32] $buffer.Capacity)

  if ($length -gt 0 -and $length -lt $buffer.Capacity) {
    return $buffer.ToString()
  }

  return $fullPath
}

function ConvertTo-QMakeProjectPath {
  param([Parameter(Mandatory = $true)][string] $Path)
  return (ConvertTo-BuildToolPath $Path) -replace "\\", "/"
}

function Assert-PathInside {
  param(
    [Parameter(Mandatory = $true)][string] $Path,
    [Parameter(Mandatory = $true)][string] $Root
  )

  $fullPath = ConvertTo-FullPath $Path
  $fullRoot = (ConvertTo-FullPath $Root).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
  $comparison = [System.StringComparison]::OrdinalIgnoreCase

  if ($fullPath.Equals($fullRoot, $comparison) -or
      -not $fullPath.StartsWith($fullRoot + [System.IO.Path]::DirectorySeparatorChar, $comparison)) {
    throw "Refusing to remove path outside the repository: $fullPath"
  }
}

function Reset-Directory {
  param(
    [Parameter(Mandatory = $true)][string] $Path,
    [Parameter(Mandatory = $true)][string] $RepoRoot
  )

  if (Test-Path -LiteralPath $Path) {
    Assert-PathInside -Path $Path -Root $RepoRoot
    Remove-Item -LiteralPath $Path -Recurse -Force
  }

  New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Ensure-Directory {
  param([Parameter(Mandatory = $true)][string] $Path)
  New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Find-CommandPath {
  param([Parameter(Mandatory = $true)][string[]] $Names)

  foreach ($name in $Names) {
    $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
      return $command.Source
    }
  }

  return $null
}

function Invoke-Native {
  param(
    [Parameter(Mandatory = $true)][string] $FilePath,
    [string[]] $Arguments = @()
  )

  Write-Host "> $FilePath $($Arguments -join ' ')"
  $previousErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $output = & $FilePath @Arguments 2>&1
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }
  $output | ForEach-Object { Write-Host $_ }
  if ($exitCode -ne 0) {
    throw "Command failed with exit code ${exitCode}: $FilePath $($Arguments -join ' ')"
  }
}

function Get-QMakeValue {
  param(
    [Parameter(Mandatory = $true)][string] $QMake,
    [Parameter(Mandatory = $true)][string] $Name,
    [string] $DefaultValue = ""
  )

  $output = & $QMake -query $Name
  if ($LASTEXITCODE -ne 0) {
    if (-not [string]::IsNullOrWhiteSpace($DefaultValue)) {
      return $DefaultValue
    }
    throw "qmake -query $Name failed"
  }

  $value = ($output | Select-Object -First 1).ToString().Trim()
  if ([string]::IsNullOrWhiteSpace($value)) {
    if (-not [string]::IsNullOrWhiteSpace($DefaultValue)) {
      return $DefaultValue
    }
    throw "qmake did not return a value for $Name"
  }

  return $value
}

function Resolve-QtTool {
  param(
    [Parameter(Mandatory = $true)][string] $QMake,
    [Parameter(Mandatory = $true)][string] $Name,
    [switch] $Optional
  )

  $qtBins = Get-QMakeValue -QMake $QMake -Name "QT_INSTALL_BINS"
  $candidate = Join-Path $qtBins "$Name.exe"
  if (Test-Path -LiteralPath $candidate) {
    return $candidate
  }

  $fromPath = Find-CommandPath @($Name, "$Name.exe")
  if ($null -ne $fromPath) {
    return $fromPath
  }

  if ($Optional) {
    return $null
  }

  throw "Required Qt tool was not found: $Name"
}

function Resolve-QMake {
  $qmake = Find-CommandPath @("qmake", "qmake6", "qmake-qt6", "qmake-qt5")
  if ($null -eq $qmake) {
    throw "qmake was not found. Run this from a Qt command prompt or add Qt bin to PATH."
  }

  return $qmake
}

function Resolve-MakeTool {
  param(
    [Parameter(Mandatory = $true)][string] $QMake,
    [Parameter(Mandatory = $true)][string] $QMakeSpec
  )

  if ($QMakeSpec -match "msvc") {
    $makeTool = Find-CommandPath @("jom", "nmake")
  } else {
    $makeTool = Find-CommandPath @("mingw32-make", "make", "jom", "nmake")
  }

  if ($null -eq $makeTool) {
    throw "No supported make tool was found. Use a Qt/MSVC command prompt or install jom/nmake/mingw32-make."
  }

  return $makeTool
}

function Test-FftwHome {
  param([Parameter(Mandatory = $true)][string] $Path)

  return ((Test-Path -LiteralPath (Join-Path $Path "include\fftw3.h")) -and
          (Test-Path -LiteralPath (Join-Path $Path "lib\libfftw3-3.dll")))
}

function Ensure-FftwMsvcImportLibrary {
  param(
    [Parameter(Mandatory = $true)][string] $FftwHome,
    [Parameter(Mandatory = $true)][int] $Bits,
    [Parameter(Mandatory = $true)][string] $RepoRoot
  )

  $library = Join-Path $FftwHome "lib\libfftw3-3.lib"
  if (Test-Path -LiteralPath $library) {
    return
  }

  $definition = Join-Path $FftwHome "lib\libfftw3-3.def"
  if (-not (Test-Path -LiteralPath $definition)) {
    throw "Missing FFTW import library and definition file: $library"
  }

  Assert-PathInside -Path $library -Root $RepoRoot
  $libTool = Find-CommandPath @("lib")
  if ($null -eq $libTool) {
    throw "MSVC lib.exe was not found, so FFTW import library could not be created."
  }

  $machine = if ($Bits -eq 64) { "x64" } else { "x86" }
  Invoke-Native -FilePath $libTool -Arguments @("/def:$definition", "/out:$library", "/machine:$machine")
}

function Resolve-FftwHome {
  param(
    [Parameter(Mandatory = $true)][string] $RepoRoot,
    [Parameter(Mandatory = $true)][string] $BuildDir,
    [Parameter(Mandatory = $true)][int] $Bits,
    [Parameter(Mandatory = $true)][string] $QMakeSpec
  )

  $candidatePaths = @()
  if (-not [string]::IsNullOrWhiteSpace($env:FFTW_HOME)) {
    $candidatePaths += $env:FFTW_HOME
  }
  $candidatePaths += (Join-Path $RepoRoot "dev\windows\unzip_fftw")

  foreach ($candidate in $candidatePaths) {
    if (Test-FftwHome $candidate) {
      if ($QMakeSpec -match "msvc") {
        Ensure-FftwMsvcImportLibrary -FftwHome $candidate -Bits $Bits -RepoRoot $RepoRoot
      }
      return (ConvertTo-FullPath $candidate)
    }
  }

  $zip = Join-Path $RepoRoot "dev\windows\appveyor\fftw-3.3.5-dll$Bits.zip"
  if (-not (Test-Path -LiteralPath $zip)) {
    throw "FFTW_HOME is not set and bundled FFTW zip was not found: $zip"
  }

  $target = Join-Path $BuildDir "deps\fftw"
  Reset-Directory -Path $target -RepoRoot $RepoRoot
  Expand-Archive -LiteralPath $zip -DestinationPath $target -Force

  $includeDir = Join-Path $target "include"
  $libDir = Join-Path $target "lib"
  Ensure-Directory $includeDir
  Ensure-Directory $libDir

  $header = Get-ChildItem -LiteralPath $target -Recurse -Filter "fftw3.h" | Select-Object -First 1
  if ($null -eq $header) {
    throw "Bundled FFTW zip did not contain fftw3.h"
  }
  Copy-Item -LiteralPath $header.FullName -Destination $includeDir -Force

  Get-ChildItem -LiteralPath $target -Recurse -Include "*.dll", "*.def", "*.lib" |
    Where-Object { -not $_.PSIsContainer -and $_.DirectoryName -ne $libDir } |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $libDir -Force }

  if ($QMakeSpec -match "msvc") {
    Ensure-FftwMsvcImportLibrary -FftwHome $target -Bits $Bits -RepoRoot $RepoRoot
  }

  return (ConvertTo-FullPath $target)
}

function Copy-RequiredFile {
  param(
    [Parameter(Mandatory = $true)][string] $Source,
    [Parameter(Mandatory = $true)][string] $Destination
  )

  if (-not (Test-Path -LiteralPath $Source)) {
    throw "Required file was not found: $Source"
  }

  Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Copy-RuntimeDlls {
  param(
    [Parameter(Mandatory = $true)][string] $PackageDir,
    [Parameter(Mandatory = $true)][string] $FftwHome,
    [Parameter(Mandatory = $true)][string[]] $Config
  )

  Copy-RequiredFile -Source (Join-Path $FftwHome "lib\libfftw3-3.dll") -Destination $PackageDir

  if (-not ($Config -contains "log4cpp_null")) {
    if ([string]::IsNullOrWhiteSpace($env:LOG4CPP_HOME)) {
      throw "LOG4CPP_HOME is required when CONFIG does not include log4cpp_null."
    }
    $log4cppDlls = @(
      (Join-Path $env:LOG4CPP_HOME "bin\log4cpp.dll"),
      (Join-Path $env:LOG4CPP_HOME "lib\log4cpp.dll")
    )
    $log4cppDll = $log4cppDlls | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($null -eq $log4cppDll) {
      throw "log4cpp.dll was not found under LOG4CPP_HOME."
    }
    Copy-RequiredFile -Source $log4cppDll -Destination $PackageDir
  }

  if ($Config -contains "pdf") {
    if ([string]::IsNullOrWhiteSpace($env:POPPLER_LIB)) {
      throw "POPPLER_LIB is required when CONFIG includes pdf."
    }
    Get-ChildItem -LiteralPath $env:POPPLER_LIB -Filter "*.dll" |
      ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $PackageDir -Force }
  }

  if ($Config -contains "jpeg2000") {
    if ([string]::IsNullOrWhiteSpace($env:OPENJPEG_LIB)) {
      throw "OPENJPEG_LIB is required when CONFIG includes jpeg2000."
    }
    Get-ChildItem -LiteralPath $env:OPENJPEG_LIB -Filter "*openjp2*.dll" |
      ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $PackageDir -Force }
  }
}

function Copy-Documentation {
  param(
    [Parameter(Mandatory = $true)][string] $RepoRoot,
    [Parameter(Mandatory = $true)][string] $BuildDir,
    [Parameter(Mandatory = $true)][string] $PackageDir,
    [Parameter(Mandatory = $true)][string] $QMake
  )

  $documentationDir = Join-Path $PackageDir "documentation"
  Ensure-Directory $documentationDir
  $qcollectionGenerator = Resolve-QtTool -QMake $QMake -Name "qcollectiongenerator" -Optional

  if ($null -eq $qcollectionGenerator) {
    Write-Warning "qcollectiongenerator was not found. The portable package will not include searchable help."
    return
  }

  $helpBuildDir = Join-Path $BuildDir "help"
  Reset-Directory -Path $helpBuildDir -RepoRoot $RepoRoot
  Copy-Item -LiteralPath (Join-Path $RepoRoot "help") -Destination $helpBuildDir -Recurse -Force
  $helpWorkDir = Join-Path $helpBuildDir "help"

  Push-Location $helpWorkDir
  try {
    Invoke-Native -FilePath $qcollectionGenerator -Arguments @("engauge.qhcp", "-o", "engauge.qhc")
  } finally {
    Pop-Location
  }

  Copy-RequiredFile -Source (Join-Path $helpWorkDir "engauge.qch") -Destination $documentationDir
  Copy-RequiredFile -Source (Join-Path $helpWorkDir "engauge.qhc") -Destination $documentationDir
}

function Copy-Translations {
  param(
    [Parameter(Mandatory = $true)][string] $RepoRoot,
    [Parameter(Mandatory = $true)][string] $BuildDir,
    [Parameter(Mandatory = $true)][string] $PackageDir,
    [Parameter(Mandatory = $true)][string] $QMake
  )

  $translationPackageDir = Join-Path $PackageDir "translations"
  Ensure-Directory $translationPackageDir
  $lrelease = Resolve-QtTool -QMake $QMake -Name "lrelease" -Optional

  if ($null -eq $lrelease) {
    Write-Warning "lrelease was not found. The portable package will use English-only application text."
    return
  }

  $translationBuildDir = Join-Path $BuildDir "translations"
  Reset-Directory -Path $translationBuildDir -RepoRoot $RepoRoot
  $translationBuildDirForTool = ConvertTo-BuildToolPath $translationBuildDir

  Get-ChildItem -LiteralPath (Join-Path $RepoRoot "translations") -Filter "engauge_*.ts" |
    ForEach-Object {
      $qmFile = Join-Path $translationBuildDir ($_.BaseName + ".qm")
      $qmFileForTool = Join-Path $translationBuildDirForTool ($_.BaseName + ".qm")
      Invoke-Native -FilePath $lrelease -Arguments @((ConvertTo-BuildToolPath $_.FullName), "-qm", $qmFileForTool)
      Copy-RequiredFile -Source $qmFile -Destination $translationPackageDir
    }
}

function Write-PortableFiles {
  param([Parameter(Mandatory = $true)][string] $PackageDir)

  $launcher = @"
@echo off
set "ENGAUGE_PORTABLE=1"
start "" /D "%~dp0" "%~dp0AutoTraceDigitizer.exe" %*
"@

  Set-Content -LiteralPath (Join-Path $PackageDir "Start AutoTrace Digitizer.cmd") -Value $launcher -Encoding ASCII

  $qtConf = @"
[Paths]
Prefix=.
"@
  Set-Content -LiteralPath (Join-Path $PackageDir "qt.conf") -Value $qtConf -Encoding ASCII

  $portableMarker = @"
; Keep this file beside AutoTraceDigitizer.exe to store settings in .\settings instead of the Windows registry.
"@
  Set-Content -LiteralPath (Join-Path $PackageDir "portable-settings.ini") -Value $portableMarker -Encoding ASCII
  Ensure-Directory (Join-Path $PackageDir "settings")

  $readme = @"
AutoTrace Digitizer portable package

AutoTrace Digitizer is a modified fork of Engauge Digitizer. It is not the official upstream Engauge Digitizer release.
Original Engauge Digitizer license and attribution notices are preserved.

Run Start AutoTrace Digitizer.cmd or AutoTraceDigitizer.exe.
Settings are stored under .\settings because portable-settings.ini is present.
This folder can be copied to another Windows computer without installing AutoTrace Digitizer.
Auto Axis detects the bottom x-axis, supports mild tilt correction, creates editable axis points, sets x start to 1, sets y minimum to 0, and prompts for y maximum.
Auto Curve detects visible plotted markers in the calibrated plot area and creates editable curve points on the selected curve.
Auto Curve groups visually matching markers and cycles detected marker groups on repeated clicks.
The default locale is US English unless the user selects another locale in Settings / Main Window.
Portable paths are resolved relative to this folder and are intended to support Unicode and non-English Windows paths.

Limitations: Auto Axis does not read axis numbers automatically. Auto Curve is an early implementation and needs manual review.
"@
  Set-Content -LiteralPath (Join-Path $PackageDir "PORTABLE_README.txt") -Value $readme -Encoding ASCII
}

$scriptDir = $PSScriptRoot
$repoRoot = ConvertTo-FullPath (Join-Path $scriptDir "..\..")
$projectFile = Join-Path $repoRoot "engauge.pro"
$projectFileForQMake = ConvertTo-QMakeProjectPath $projectFile

if ([string]::IsNullOrWhiteSpace($BuildDir)) {
  $BuildDir = Join-Path $repoRoot "dev\windows\staging\windows-portable-build"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $OutputDir = Join-Path $repoRoot "release\$PortableFolderName"
}

$BuildDir = ConvertTo-FullPath $BuildDir
$OutputDir = ConvertTo-FullPath $OutputDir
$packageDir = $OutputDir

Assert-PathInside -Path $BuildDir -Root $repoRoot
Assert-PathInside -Path $packageDir -Root $repoRoot

$qmake = Resolve-QMake
$qtVersion = Get-QMakeValue -QMake $qmake -Name "QT_VERSION"
$qmakeSpec = Get-QMakeValue -QMake $qmake -Name "QMAKE_XSPEC"
$qtPrefix = Get-QMakeValue -QMake $qmake -Name "QT_INSTALL_PREFIX"
$qtArchDefault = if ($qtPrefix -match "64") { "x86_64" } else { "x86" }
$qtArch = Get-QMakeValue -QMake $qmake -Name "QT_ARCH" -DefaultValue $qtArchDefault
$bits = if ($qtArch -match "64|x86_64|amd64") { 64 } else { 32 }
$makeTool = Resolve-MakeTool -QMake $qmake -QMakeSpec $qmakeSpec
$windeployqt = Resolve-QtTool -QMake $qmake -Name "windeployqt"

if ($Clean) {
  Reset-Directory -Path $BuildDir -RepoRoot $repoRoot
} else {
  Ensure-Directory $BuildDir
}

$fftwHome = ConvertTo-BuildToolPath (Resolve-FftwHome -RepoRoot $repoRoot -BuildDir $BuildDir -Bits $bits -QMakeSpec $qmakeSpec)
$env:FFTW_HOME = $fftwHome

if (-not $SkipBuild) {
  Push-Location $BuildDir
  try {
    $qmakeConfig = "CONFIG+=$($Config -join ' ')"
    $qmakeArgs = @($projectFileForQMake, $qmakeConfig, "CONFIG-=debug_and_release", "CONFIG+=release", "DEFINES+=WIN_RELEASE") + $QMakeArg
    Invoke-Native -FilePath $qmake -Arguments $qmakeArgs
    Invoke-Native -FilePath $makeTool
  } finally {
    Pop-Location
  }
}

$builtExe = if ($SkipBuild) {
  Join-Path $repoRoot "bin\$BuildExecutableFileName"
} else {
  Join-Path $BuildDir "bin\$BuildExecutableFileName"
}

if (-not (Test-Path -LiteralPath $builtExe)) {
  throw "$BuildExecutableFileName was not found: $builtExe"
}

Reset-Directory -Path $packageDir -RepoRoot $repoRoot
Copy-RequiredFile -Source $builtExe -Destination (Join-Path $packageDir $PortableExecutableFileName)
Copy-RequiredFile -Source (Join-Path $repoRoot "LICENSE") -Destination $packageDir
Copy-RequiredFile -Source (Join-Path $repoRoot "README.md") -Destination $packageDir
Copy-RuntimeDlls -PackageDir $packageDir -FftwHome $fftwHome -Config $Config

$deployMode = if ($Configuration -eq "debug") { "--debug" } else { "--release" }
Invoke-Native -FilePath $windeployqt -Arguments @($deployMode, "--compiler-runtime", (Join-Path $packageDir $PortableExecutableFileName))

Copy-Documentation -RepoRoot $repoRoot -BuildDir $BuildDir -PackageDir $packageDir -QMake $qmake
Copy-Translations -RepoRoot $repoRoot -BuildDir $BuildDir -PackageDir $packageDir -QMake $qmake
Write-PortableFiles -PackageDir $packageDir

if (-not $NoZip) {
  $zipPath = Join-Path (Split-Path -Parent $packageDir) $ReleaseZipName
  if (Test-Path -LiteralPath $zipPath) {
    Assert-PathInside -Path $zipPath -Root $repoRoot
    Remove-Item -LiteralPath $zipPath -Force
  }
  Compress-Archive -LiteralPath $packageDir -DestinationPath $zipPath -Force
}

Write-Host ""
Write-Host "Portable folder: $packageDir"
if (-not $NoZip) {
  Write-Host "Portable zip:    $zipPath"
}
Write-Host "Qt version:      $qtVersion"
Write-Host "Qt spec:         $qmakeSpec"
Write-Host "Qt arch:         $qtArch"
