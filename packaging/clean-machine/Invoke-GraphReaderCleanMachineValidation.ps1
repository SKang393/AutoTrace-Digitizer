# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PayloadRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [Parameter(Mandatory = $true)]
    [string]$VmProvenancePath,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedCommit,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedVersion,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedExecutableSha256,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedOpenCvSha256,

    [ValidateRange(15, 120)]
    [int]$ApplicationSmokeTimeoutSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-CanonicalSha256 {
    param([Parameter(Mandatory = $true)][string]$Value)

    return $Value -cmatch '^[0-9a-f]{64}$'
}

function Add-Failure {
    param([Parameter(Mandatory = $true)][string]$Message)

    $script:failures.Add($Message)
}

function Get-RequiredJsonProperty {
    param(
        [Parameter(Mandatory = $true)][object]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        Add-Failure "$Description is missing required property '$Name'."
        return $null
    }
    return $property.Value
}

function Get-RequiredBooleanJsonProperty {
    param(
        [Parameter(Mandatory = $true)][object]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $value = Get-RequiredJsonProperty -Object $Object -Name $Name -Description $Description
    if ($null -eq $value -or -not ($value -is [bool])) {
        Add-Failure "$Description property '$Name' must be a JSON Boolean."
        return $false
    }
    return [bool]$value
}

$PayloadRoot = [IO.Path]::GetFullPath($PayloadRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$VmProvenancePath = [IO.Path]::GetFullPath($VmProvenancePath)
$ExpectedExecutableSha256 = $ExpectedExecutableSha256.ToLowerInvariant()
$ExpectedOpenCvSha256 = $ExpectedOpenCvSha256.ToLowerInvariant()

foreach ($value in @($ExpectedExecutableSha256, $ExpectedOpenCvSha256)) {
    if (-not (Test-CanonicalSha256 -Value $value)) {
        throw "Expected checksums must be canonical lowercase SHA-256 values."
    }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$evidencePath = Join-Path $OutputRoot 'opencv-clean-machine.json'
$failures = [Collections.Generic.List[string]]::new()
$observedAtUtc = [DateTimeOffset]::UtcNow

$result = [ordered]@{
    schema = 'graphreader.opencv-clean-machine-load.v1'
    status = 'fail'
    observedAtUtc = $observedAtUtc.ToString('O')
    harnessSha256 = $null
    vmProvenance = $null
    machine = $null
    payload = $null
    nativeLoad = $null
    applicationSmoke = $null
    failures = @()
}

try {
    $result.harnessSha256 = Get-Sha256 -Path $PSCommandPath

    if (-not (Test-Path -LiteralPath $VmProvenancePath -PathType Leaf)) {
        Add-Failure "VM provenance is missing: $VmProvenancePath"
        $vmProvenance = $null
    }
    else {
        $vmProvenance = Get-Content -LiteralPath $VmProvenancePath -Raw | ConvertFrom-Json
        $provenanceSchema = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'schema' -Description 'VM provenance')
        $isoSha256 = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'isoSha256' -Description 'VM provenance')
        $officialIsoSha256 = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'officialIsoSha256' -Description 'VM provenance')
        $isoSha256Verified = Get-RequiredBooleanJsonProperty -Object $vmProvenance -Name 'isoSha256Verified' -Description 'VM provenance'
        $environmentKind = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'environmentKind' -Description 'VM provenance')
        $networkMode = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'networkMode' -Description 'VM provenance')
        $freshInstall = Get-RequiredBooleanJsonProperty -Object $vmProvenance -Name 'freshInstall' -Description 'VM provenance'
        $vmId = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'vmId' -Description 'VM provenance')
        $vmConfigurationSha256 = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'vmConfigurationSha256' -Description 'VM provenance')
        $qemuInstallerSha256 = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'qemuInstallerSha256' -Description 'VM provenance')
        $qemuVersion = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'qemuVersion' -Description 'VM provenance')
        $expectedOsBuild = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedOsBuild' -Description 'VM provenance')
        $expectedOsUbrValue = Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedOsUbr' -Description 'VM provenance'
        $expectedCommitFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedCommit' -Description 'VM provenance')
        $expectedVersionFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedVersion' -Description 'VM provenance')
        $expectedExecutableFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedExecutableSha256' -Description 'VM provenance')
        $expectedOpenCvFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedOpenCvSha256' -Description 'VM provenance')
        $expectedHarnessFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedHarnessSha256' -Description 'VM provenance')
        $result.vmProvenance = [ordered]@{
            path = $VmProvenancePath
            sha256 = Get-Sha256 -Path $VmProvenancePath
            schema = $provenanceSchema
            isoSha256 = $isoSha256
            officialIsoSha256 = $officialIsoSha256
            isoSha256Verified = $isoSha256Verified
            environmentKind = $environmentKind
            networkMode = $networkMode
            freshInstall = $freshInstall
            vmId = $vmId
            vmConfigurationSha256 = $vmConfigurationSha256
            qemuInstallerSha256 = $qemuInstallerSha256
            qemuVersion = $qemuVersion
            expectedOsBuild = $expectedOsBuild
            expectedOsUbr = $expectedOsUbrValue
            expectedCommit = $expectedCommitFromProvenance
            expectedVersion = $expectedVersionFromProvenance
            expectedExecutableSha256 = $expectedExecutableFromProvenance
            expectedOpenCvSha256 = $expectedOpenCvFromProvenance
            expectedHarnessSha256 = $expectedHarnessFromProvenance
        }

        if ($provenanceSchema -cne 'graphreader.clean-windows-vm-provenance.v1') {
            Add-Failure 'VM provenance schema is not supported.'
        }
        if (-not (Test-CanonicalSha256 -Value $isoSha256) -or
            -not (Test-CanonicalSha256 -Value $officialIsoSha256) -or
            $isoSha256 -cne $officialIsoSha256 -or
            -not $isoSha256Verified) {
            Add-Failure 'The official Windows evaluation ISO checksum was not verified.'
        }
        if ($environmentKind -cne 'fresh-windows-evaluation-vm') {
            Add-Failure 'The execution environment is not identified as a fresh Windows evaluation VM.'
        }
        if ($networkMode -cne 'none') {
            Add-Failure 'The clean-machine validation was not provisioned with networking disabled.'
        }
        if (-not $freshInstall) {
            Add-Failure 'The VM provenance does not identify a fresh installation.'
        }
        if ($vmId -notmatch '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' -or
            -not (Test-CanonicalSha256 -Value $vmConfigurationSha256) -or
            -not (Test-CanonicalSha256 -Value $qemuInstallerSha256) -or
            [string]::IsNullOrWhiteSpace($qemuVersion) -or
            $expectedOsBuild -notmatch '^\d{5}$' -or
            -not ($expectedOsUbrValue -is [long] -or $expectedOsUbrValue -is [int])) {
            Add-Failure 'VM provenance is missing a canonical VM, QEMU, or expected OS identity.'
        }
        if ($expectedCommitFromProvenance -cne $ExpectedCommit -or
            $expectedVersionFromProvenance -cne $ExpectedVersion -or
            $expectedExecutableFromProvenance -cne $ExpectedExecutableSha256 -or
            $expectedOpenCvFromProvenance -cne $ExpectedOpenCvSha256 -or
            $expectedHarnessFromProvenance -cne [string]$result.harnessSha256) {
            Add-Failure 'VM provenance does not bind the exact requested portable payload.'
        }
    }

    $os = Get-CimInstance Win32_OperatingSystem
    $computerSystem = Get-CimInstance Win32_ComputerSystem
    $computerSystemProduct = Get-CimInstance Win32_ComputerSystemProduct
    $bios = Get-CimInstance Win32_BIOS
    $currentVersion = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
    $installTimeUtc = ([DateTime]$os.InstallDate).ToUniversalTime()
    $installAgeHours = ([DateTimeOffset]::UtcNow - [DateTimeOffset]$installTimeUtc).TotalHours
    $developerTools = [Collections.Generic.List[object]]::new()
    foreach ($commandName in @('dotnet.exe', 'git.exe', 'cmake.exe', 'ninja.exe', 'msbuild.exe', 'devenv.exe')) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            $developerTools.Add([ordered]@{
                    command = $commandName
                    source = [string]$command.Source
                })
        }
    }

    $networkQuerySucceeded = $false
    $networkAdapters = @()
    if ($null -eq (Get-Command Get-NetAdapter -ErrorAction SilentlyContinue)) {
        Add-Failure 'Get-NetAdapter is unavailable, so offline state cannot be verified.'
    }
    else {
        try {
            $networkAdapters = @(Get-NetAdapter -ErrorAction Stop | ForEach-Object {
                [ordered]@{
                    name = [string]$_.Name
                    status = [string]$_.Status
                    interfaceDescription = [string]$_.InterfaceDescription
                }
            })
            $networkQuerySucceeded = $true
        }
        catch {
            Add-Failure "Network adapter discovery failed: $($_.Exception.Message)"
        }
    }
    $networkAdaptersUp = @($networkAdapters | Where-Object { [string]$_.status -ceq 'Up' })

    $result.machine = [ordered]@{
        productName = [string]$os.Caption
        version = [string]$os.Version
        buildNumber = [string]$os.BuildNumber
        architecture = [string]$os.OSArchitecture
        installTimeUtc = $installTimeUtc.ToString('O')
        installAgeHours = [Math]::Round($installAgeHours, 3)
        powerShellVersion = $PSVersionTable.PSVersion.ToString()
        is64BitOperatingSystem = [Environment]::Is64BitOperatingSystem
        is64BitProcess = [Environment]::Is64BitProcess
        manufacturer = [string]$computerSystem.Manufacturer
        model = [string]$computerSystem.Model
        biosManufacturer = [string]$bios.Manufacturer
        machineUuid = ([string]$computerSystemProduct.UUID).ToLowerInvariant()
        updateBuildRevision = [int]$currentVersion.UBR
        developerToolsOnPath = @($developerTools)
        networkQuerySucceeded = $networkQuerySucceeded
        networkAdapters = @($networkAdapters)
        networkAdaptersUp = $networkAdaptersUp.Count
    }

    if (-not [Environment]::Is64BitOperatingSystem -or
        -not [Environment]::Is64BitProcess -or
        [string]$os.Caption -notmatch 'Windows') {
        Add-Failure 'The validation host is not Windows x64.'
    }
    if ($null -ne $vmProvenance -and
        (([string]$computerSystemProduct.UUID).ToLowerInvariant() -cne ([string]$result.vmProvenance.vmId).ToLowerInvariant() -or
            [string]$os.BuildNumber -cne [string]$result.vmProvenance.expectedOsBuild -or
            [int]$currentVersion.UBR -ne [int]$result.vmProvenance.expectedOsUbr -or
            [string]$computerSystem.Manufacturer -notmatch '(?i)qemu')) {
        Add-Failure 'Observed guest identity does not match the checksum-bound QEMU VM provenance.'
    }
    if ($installAgeHours -lt 0 -or $installAgeHours -gt 24) {
        Add-Failure "The Windows installation is not fresh enough for this evidence run: $([Math]::Round($installAgeHours, 3)) hours."
    }
    if ($developerTools.Count -ne 0) {
        Add-Failure 'Developer tools are present on PATH in the claimed clean environment.'
    }
    if (-not $networkQuerySucceeded -or $networkAdaptersUp.Count -ne 0) {
        Add-Failure 'One or more network adapters are up during the offline clean-machine run.'
    }

    $executablePath = Join-Path $PayloadRoot 'GraphReader.App.exe'
    $portableModePath = Join-Path $PayloadRoot 'portable.mode'
    $buildInfoPath = Join-Path $PayloadRoot 'build-info.json'
    $nativeCandidates = @(Get-ChildItem -LiteralPath $PayloadRoot -Filter 'OpenCvSharpExtern.dll' -File -Recurse)
    $modelPayloads = @(Get-ChildItem -LiteralPath $PayloadRoot -File -Recurse | Where-Object {
            $_.Extension -in @('.onnx', '.param', '.bin')
        })
    $reparsePoints = @(Get-Item -LiteralPath $PayloadRoot -Force; Get-ChildItem -LiteralPath $PayloadRoot -Force -Recurse) |
        Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }

    if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
        Add-Failure "Application executable is missing: $executablePath"
    }
    if (-not (Test-Path -LiteralPath $portableModePath -PathType Leaf)) {
        Add-Failure "portable.mode is missing: $portableModePath"
    }
    if (-not (Test-Path -LiteralPath $buildInfoPath -PathType Leaf)) {
        Add-Failure "build-info.json is missing: $buildInfoPath"
    }
    if ($nativeCandidates.Count -ne 1) {
        Add-Failure "Expected exactly one OpenCvSharpExtern.dll, found $($nativeCandidates.Count)."
    }
    if ($modelPayloads.Count -ne 0) {
        Add-Failure "The manual portable contains $($modelPayloads.Count) model payload files."
    }
    if ($reparsePoints.Count -ne 0) {
        Add-Failure "The portable payload contains $($reparsePoints.Count) reparse-point entries."
    }

    $executableSha256 = if (Test-Path -LiteralPath $executablePath -PathType Leaf) {
        Get-Sha256 -Path $executablePath
    }
    else {
        $null
    }
    $openCvPath = if ($nativeCandidates.Count -eq 1) { $nativeCandidates[0].FullName } else { $null }
    $openCvSha256 = if ($null -ne $openCvPath) { Get-Sha256 -Path $openCvPath } else { $null }
    $buildInfo = if (Test-Path -LiteralPath $buildInfoPath -PathType Leaf) {
        Get-Content -LiteralPath $buildInfoPath -Raw | ConvertFrom-Json
    }
    else {
        $null
    }

    if ($executableSha256 -cne $ExpectedExecutableSha256) {
        Add-Failure 'Application executable checksum differs from the expected clean build.'
    }
    if ($openCvSha256 -cne $ExpectedOpenCvSha256) {
        Add-Failure 'OpenCV native runtime checksum differs from the reviewed source build.'
    }
    if ($null -ne $buildInfo) {
        $dirtyProperty = $buildInfo.PSObject.Properties['dirty']
        $availableModelsProperty = $buildInfo.PSObject.Properties['availableModelIds']
        if ($buildInfo.PSObject.Properties['schemaVersion'] -eq $null -or [int]$buildInfo.schemaVersion -ne 2 -or
            $buildInfo.PSObject.Properties['commit'] -eq $null -or [string]$buildInfo.commit -cne $ExpectedCommit -or
            $buildInfo.PSObject.Properties['version'] -eq $null -or [string]$buildInfo.version -cne $ExpectedVersion -or
            $null -eq $dirtyProperty -or -not ($dirtyProperty.Value -is [bool]) -or [bool]$dirtyProperty.Value -or
            $buildInfo.PSObject.Properties['executableSha256'] -eq $null -or [string]$buildInfo.executableSha256 -cne $ExpectedExecutableSha256 -or
            $null -eq $availableModelsProperty -or @($availableModelsProperty.Value).Count -ne 0) {
            Add-Failure 'build-info.json does not bind the expected clean manual-preview commit, version, executable, and empty model set.'
        }
    }

    $result.payload = [ordered]@{
        root = $PayloadRoot
        version = if ($null -ne $buildInfo) { [string]$buildInfo.version } else { $null }
        commit = if ($null -ne $buildInfo) { [string]$buildInfo.commit } else { $null }
        dirty = if ($null -ne $buildInfo) { [bool]$buildInfo.dirty } else { $null }
        executablePath = $executablePath
        executableSha256 = $executableSha256
        portableModePresent = Test-Path -LiteralPath $portableModePath -PathType Leaf
        openCvPath = $openCvPath
        openCvSha256 = $openCvSha256
        modelPayloadFileCount = $modelPayloads.Count
        reparsePointCount = $reparsePoints.Count
    }

    $nativeLoaded = $false
    $nativeLoadError = $null
    if ($null -ne $openCvPath -and $openCvSha256 -ceq $ExpectedOpenCvSha256) {
        if ($null -eq ('GraphReaderCleanMachineNativeMethods' -as [type])) {
            Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class GraphReaderCleanMachineNativeMethods
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr LoadLibraryEx(string fileName, IntPtr reserved, uint flags);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool FreeLibrary(IntPtr module);
}
'@
        }

        $module = [GraphReaderCleanMachineNativeMethods]::LoadLibraryEx(
            $openCvPath,
            [IntPtr]::Zero,
            [uint32]0x00001100)
        if ($module -eq [IntPtr]::Zero) {
            $nativeLoadError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            Add-Failure "OpenCvSharpExtern.dll failed to load with Win32 error $nativeLoadError."
        }
        else {
            $nativeLoaded = $true
            if (-not [GraphReaderCleanMachineNativeMethods]::FreeLibrary($module)) {
                $nativeLoadError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
                Add-Failure "OpenCvSharpExtern.dll loaded but failed to unload with Win32 error $nativeLoadError."
            }
        }
    }
    $result.nativeLoad = [ordered]@{
        attempted = $null -ne $openCvPath
        succeeded = $nativeLoaded
        win32Error = $nativeLoadError
        flags = 'LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS'
    }

    $smokeStartedUtc = $null
    $smokeFinishedUtc = $null
    $smokeExitCode = $null
    $smokeTimedOut = $false
    if ($executableSha256 -ceq $ExpectedExecutableSha256) {
        $previousDataRoot = [Environment]::GetEnvironmentVariable('GRAPHREADER_DEV_PORTABLE_DATA_ROOT', 'Process')
        $previousEnhancementRoot = [Environment]::GetEnvironmentVariable('GRAPHREADER_REALESRGAN_RUNTIME_ROOT', 'Process')
        $previousEnhancementManifest = [Environment]::GetEnvironmentVariable('GRAPHREADER_REALESRGAN_MANIFEST_PATH', 'Process')
        $previousPdfium = [Environment]::GetEnvironmentVariable('GRAPHREADER_PDFIUM_APPROVAL_PATH', 'Process')
        try {
            [Environment]::SetEnvironmentVariable('GRAPHREADER_DEV_PORTABLE_DATA_ROOT', (Join-Path $OutputRoot 'Data'), 'Process')
            [Environment]::SetEnvironmentVariable('GRAPHREADER_REALESRGAN_RUNTIME_ROOT', $null, 'Process')
            [Environment]::SetEnvironmentVariable('GRAPHREADER_REALESRGAN_MANIFEST_PATH', $null, 'Process')
            [Environment]::SetEnvironmentVariable('GRAPHREADER_PDFIUM_APPROVAL_PATH', $null, 'Process')
            $smokeStartedUtc = [DateTimeOffset]::UtcNow
            $process = Start-Process -FilePath $executablePath -ArgumentList '--portable-smoke' -WorkingDirectory $PayloadRoot -PassThru -WindowStyle Hidden
            try {
                if (-not $process.WaitForExit($ApplicationSmokeTimeoutSeconds * 1000)) {
                    $smokeTimedOut = $true
                    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                    Add-Failure "Application portable smoke exceeded $ApplicationSmokeTimeoutSeconds seconds."
                }
                else {
                    $smokeExitCode = $process.ExitCode
                    if ($smokeExitCode -ne 0) {
                        Add-Failure "Application portable smoke exited with code $smokeExitCode."
                    }
                }
            }
            finally {
                if (-not $process.HasExited) {
                    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                }
                $process.Dispose()
                $smokeFinishedUtc = [DateTimeOffset]::UtcNow
            }
        }
        finally {
            [Environment]::SetEnvironmentVariable('GRAPHREADER_DEV_PORTABLE_DATA_ROOT', $previousDataRoot, 'Process')
            [Environment]::SetEnvironmentVariable('GRAPHREADER_REALESRGAN_RUNTIME_ROOT', $previousEnhancementRoot, 'Process')
            [Environment]::SetEnvironmentVariable('GRAPHREADER_REALESRGAN_MANIFEST_PATH', $previousEnhancementManifest, 'Process')
            [Environment]::SetEnvironmentVariable('GRAPHREADER_PDFIUM_APPROVAL_PATH', $previousPdfium, 'Process')
        }
    }

    $result.applicationSmoke = [ordered]@{
        argument = '--portable-smoke'
        startedUtc = if ($null -ne $smokeStartedUtc) { $smokeStartedUtc.ToString('O') } else { $null }
        finishedUtc = if ($null -ne $smokeFinishedUtc) { $smokeFinishedUtc.ToString('O') } else { $null }
        timeoutSeconds = $ApplicationSmokeTimeoutSeconds
        timedOut = $smokeTimedOut
        exitCode = $smokeExitCode
        passed = -not $smokeTimedOut -and $smokeExitCode -eq 0
    }
}
catch {
    Add-Failure "Harness error: $($_.Exception.Message)"
}

$result.failures = @($failures)
if ($failures.Count -eq 0 -and
    $null -ne $result.nativeLoad -and [bool]$result.nativeLoad.succeeded -and
    $null -ne $result.applicationSmoke -and [bool]$result.applicationSmoke.passed) {
    $result.status = 'pass'
}

[IO.File]::WriteAllText(
    $evidencePath,
    (($result | ConvertTo-Json -Depth 12) + [Environment]::NewLine),
    [Text.UTF8Encoding]::new($false))

Write-Host "OpenCV clean-machine validation: $($result.status.ToUpperInvariant())"
Write-Host "Evidence: $evidencePath"
if ($result.status -cne 'pass') {
    foreach ($failure in $failures) {
        Write-Host "BLOCKED: $failure"
    }
    exit 2
}

exit 0
