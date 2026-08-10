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
    [string]$ExpectedApplicationDllSha256,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedRendererSha256,

    [string]$RendererSmokeScriptPath,

    [ValidateRange(5, 300)]
    [int]$RendererSmokeTimeoutSeconds = 90,

    [ValidateRange(5, 300)]
    [int]$ApplicationSmokeTimeoutSeconds = 90
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PayloadRoot = [IO.Path]::GetFullPath($PayloadRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$VmProvenancePath = [IO.Path]::GetFullPath($VmProvenancePath)
if ([string]::IsNullOrWhiteSpace($RendererSmokeScriptPath)) {
    $RendererSmokeScriptPath = Join-Path $PSScriptRoot '..\pdfium-source\Test-PdfiumRunner.ps1'
}
$RendererSmokeScriptPath = [IO.Path]::GetFullPath($RendererSmokeScriptPath)
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$evidencePath = Join-Path $OutputRoot 'pdfium-clean-machine.json'
$failures = [Collections.Generic.List[string]]::new()

function Add-Failure([string]$Message) {
    $failures.Add($Message)
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-CanonicalSha256([AllowNull()][object]$Value) {
    return $Value -is [string] -and [string]$Value -cmatch '^[0-9a-f]{64}$'
}

function Get-RequiredJsonProperty {
    param(
        [Parameter(Mandatory = $true)][object]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "$Description is missing required property '$Name'."
    }
    return $property.Value
}

function Test-JsonBoolean {
    param(
        [Parameter(Mandatory = $true)][object]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][bool]$Expected
    )

    $property = $Object.PSObject.Properties[$Name]
    return $null -ne $property -and $property.Value -is [bool] -and [bool]$property.Value -eq $Expected
}

function Resolve-SafePayloadFile {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if ([IO.Path]::IsPathRooted($RelativePath) -or $RelativePath -match '(^|[\\/])\.\.([\\/]|$)') {
        throw "$Description uses an unsafe relative path: $RelativePath"
    }
    $rootPrefix = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $path = [IO.Path]::GetFullPath((Join-Path $Root $RelativePath.Replace('/', '\')))
    if (-not $path.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "$Description is missing or outside the packaged PDFium root: $RelativePath"
    }
    $current = $path
    while ($current.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        if (((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Description contains a reparse point: $current"
        }
        $current = Split-Path -Parent $current
    }
    return $path
}

function Quote-ProcessArgument([string]$Value) {
    if ($Value.Contains('"')) {
        throw 'A clean-machine process argument contains a quotation mark.'
    }
    return '"' + $Value + '"'
}

$harnessSha256 = Get-Sha256 -Path $PSCommandPath
$result = [ordered]@{
    schema = 'graphreader.pdfium-clean-machine-load.v1'
    status = 'fail'
    observedAtUtc = [DateTimeOffset]::UtcNow.ToString('O')
    harnessSha256 = $harnessSha256
    requested = [ordered]@{
        commit = $ExpectedCommit
        version = $ExpectedVersion
        executableSha256 = $ExpectedExecutableSha256
        applicationDllSha256 = $ExpectedApplicationDllSha256
        rendererSha256 = $ExpectedRendererSha256
    }
    vmProvenance = $null
    machine = $null
    payload = $null
    rendererSmoke = $null
    applicationPackagedFallback = $null
    failures = @()
}

try {
    if ($ExpectedCommit -notmatch '^[0-9a-f]{40}$' -or
        $ExpectedVersion -notmatch '^\d{1,2}\.\d{1,2}\.\d{1,2}$' -or
        -not (Test-CanonicalSha256 $ExpectedExecutableSha256) -or
        -not (Test-CanonicalSha256 $ExpectedApplicationDllSha256) -or
        -not (Test-CanonicalSha256 $ExpectedRendererSha256)) {
        throw 'Expected commit, version, executable hash, or renderer hash is not canonical.'
    }
    if (-not (Test-Path -LiteralPath $VmProvenancePath -PathType Leaf)) {
        throw "VM provenance is missing: $VmProvenancePath"
    }
    if (-not (Test-Path -LiteralPath $RendererSmokeScriptPath -PathType Leaf)) {
        throw "PDFium renderer smoke script is missing: $RendererSmokeScriptPath"
    }
    if (-not (Test-Path -LiteralPath $PayloadRoot -PathType Container)) {
        throw "PDFium clean-machine payload is missing: $PayloadRoot"
    }

    $smokeScriptSha256 = Get-Sha256 -Path $RendererSmokeScriptPath
    $vmProvenance = Get-Content -LiteralPath $VmProvenancePath -Raw | ConvertFrom-Json
    $provenanceSchema = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'schema' -Description 'VM provenance')
    $isoSha256 = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'isoSha256' -Description 'VM provenance')
    $officialIsoSha256 = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'officialIsoSha256' -Description 'VM provenance')
    $isoSha256Verified = Test-JsonBoolean -Object $vmProvenance -Name 'isoSha256Verified' -Expected $true
    $environmentKind = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'environmentKind' -Description 'VM provenance')
    $networkMode = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'networkMode' -Description 'VM provenance')
    $freshInstall = Test-JsonBoolean -Object $vmProvenance -Name 'freshInstall' -Expected $true
    $vmId = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'vmId' -Description 'VM provenance')
    $vmConfigurationSha256 = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'vmConfigurationSha256' -Description 'VM provenance')
    $qemuInstallerSha256 = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'qemuInstallerSha256' -Description 'VM provenance')
    $qemuVersion = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'qemuVersion' -Description 'VM provenance')
    $expectedOsBuild = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedOsBuild' -Description 'VM provenance')
    $expectedOsUbrValue = Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedOsUbr' -Description 'VM provenance'
    $expectedCommitFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedCommit' -Description 'VM provenance')
    $expectedVersionFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedVersion' -Description 'VM provenance')
    $expectedExecutableFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedExecutableSha256' -Description 'VM provenance')
    $expectedApplicationDllFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedApplicationDllSha256' -Description 'VM provenance')
    $expectedRendererFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedPdfiumRendererSha256' -Description 'VM provenance')
    $expectedHarnessFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedPdfiumHarnessSha256' -Description 'VM provenance')
    $expectedSmokeScriptFromProvenance = [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'expectedRendererSmokeScriptSha256' -Description 'VM provenance')
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
        expectedApplicationDllSha256 = $expectedApplicationDllFromProvenance
        expectedPdfiumRendererSha256 = $expectedRendererFromProvenance
        expectedPdfiumHarnessSha256 = $expectedHarnessFromProvenance
        expectedRendererSmokeScriptSha256 = $expectedSmokeScriptFromProvenance
    }

    if ($provenanceSchema -cne 'graphreader.clean-windows-vm-provenance.v1') {
        Add-Failure 'VM provenance schema is not supported.'
    }
    if (-not (Test-CanonicalSha256 $isoSha256) -or
        -not (Test-CanonicalSha256 $officialIsoSha256) -or
        $isoSha256 -cne $officialIsoSha256 -or
        -not $isoSha256Verified) {
        Add-Failure 'The official Windows evaluation ISO checksum was not verified.'
    }
    if ($environmentKind -cne 'fresh-windows-evaluation-vm' -or $networkMode -cne 'none' -or -not $freshInstall) {
        Add-Failure 'The execution boundary is not a fresh network-disabled Windows evaluation VM.'
    }
    if ($vmId -notmatch '^[0-9a-fA-F-]{36}$' -or
        -not (Test-CanonicalSha256 $vmConfigurationSha256) -or
        -not (Test-CanonicalSha256 $qemuInstallerSha256) -or
        [string]::IsNullOrWhiteSpace($qemuVersion) -or
        $expectedOsBuild -notmatch '^\d{5}$' -or
        -not ($expectedOsUbrValue -is [long] -or $expectedOsUbrValue -is [int])) {
        Add-Failure 'VM provenance is missing a canonical VM, QEMU, or OS identity.'
    }
    if ($expectedCommitFromProvenance -cne $ExpectedCommit -or
        $expectedVersionFromProvenance -cne $ExpectedVersion -or
        $expectedExecutableFromProvenance -cne $ExpectedExecutableSha256 -or
        $expectedApplicationDllFromProvenance -cne $ExpectedApplicationDllSha256 -or
        $expectedRendererFromProvenance -cne $ExpectedRendererSha256 -or
        $expectedHarnessFromProvenance -cne $harnessSha256 -or
        $expectedSmokeScriptFromProvenance -cne $smokeScriptSha256) {
        Add-Failure 'VM provenance does not bind the exact application, renderer, and current guest scripts.'
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
            $developerTools.Add([ordered]@{ command = $commandName; source = [string]$command.Source })
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
    if (([string]$computerSystemProduct.UUID).ToLowerInvariant() -cne $vmId.ToLowerInvariant() -or
        [string]$os.BuildNumber -cne $expectedOsBuild -or
        [int]$currentVersion.UBR -ne [int]$expectedOsUbrValue -or
        [string]$computerSystem.Manufacturer -notmatch '(?i)qemu') {
        Add-Failure 'Observed guest identity does not match the checksum-bound QEMU VM provenance.'
    }
    if ($installAgeHours -lt 0 -or $installAgeHours -gt 24) {
        Add-Failure "The Windows installation is not fresh enough: $([Math]::Round($installAgeHours, 3)) hours."
    }
    if ($developerTools.Count -ne 0) {
        Add-Failure 'Developer tools are present on PATH in the claimed clean environment.'
    }
    if (-not $networkQuerySucceeded -or $networkAdaptersUp.Count -ne 0) {
        Add-Failure 'One or more network adapters are up during the offline clean-machine run.'
    }

    $executablePath = Join-Path $PayloadRoot 'GraphReader.App.exe'
    $applicationDllPath = Join-Path $PayloadRoot 'GraphReader.App.dll'
    $pdfiumRoot = Join-Path $PayloadRoot 'pdfium'
    $approvalPath = Join-Path $pdfiumRoot 'reviewed-approval.json'
    if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
        throw "Application executable is missing: $executablePath"
    }
    if (-not (Test-Path -LiteralPath $applicationDllPath -PathType Leaf)) {
        throw "Application assembly is missing: $applicationDllPath"
    }
    if (-not (Test-Path -LiteralPath $approvalPath -PathType Leaf)) {
        throw "Packaged PDFium approval is missing: $approvalPath"
    }
    $payloadReparsePoints = @(
        @(Get-Item -LiteralPath $PayloadRoot -Force; Get-ChildItem -LiteralPath $PayloadRoot -Force -Recurse) |
            Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }
    )
    $approval = Get-Content -LiteralPath $approvalPath -Raw | ConvertFrom-Json
    if ([int](Get-RequiredJsonProperty -Object $approval -Name 'schemaVersion' -Description 'PDFium approval') -ne 1 -or
        [string](Get-RequiredJsonProperty -Object $approval -Name 'rendererId' -Description 'PDFium approval') -cne 'graphreader-pdfium-renderer' -or
        [string](Get-RequiredJsonProperty -Object $approval -Name 'rendererVersion' -Description 'PDFium approval') -cne '2870fa9244b0f0f69fb743fab1e08deefcb07b2b' -or
        [string](Get-RequiredJsonProperty -Object $approval -Name 'source' -Description 'PDFium approval') -cne 'https://pdfium.googlesource.com/pdfium' -or
        [string](Get-RequiredJsonProperty -Object $approval -Name 'sourceRevision' -Description 'PDFium approval') -cne '2870fa9244b0f0f69fb743fab1e08deefcb07b2b' -or
        [string](Get-RequiredJsonProperty -Object $approval -Name 'licenseSpdx' -Description 'PDFium approval') -cne 'BSD-3-Clause' -or
        -not (Test-JsonBoolean -Object $approval -Name 'reviewApproved' -Expected $true) -or
        -not (Test-JsonBoolean -Object $approval -Name 'redistributionApproved' -Expected $true) -or
        -not (Test-JsonBoolean -Object $approval -Name 'bundlingApproved' -Expected $true)) {
        Add-Failure 'Packaged PDFium approval does not satisfy the exact reviewed policy.'
    }

    $resourceDefinitions = @(
        [pscustomobject]@{ PathField = 'binaryPath'; HashField = 'binarySha256'; Label = 'renderer' },
        [pscustomobject]@{ PathField = 'sourceLockPath'; HashField = 'sourceLockSha256'; Label = 'source lock' },
        [pscustomobject]@{ PathField = 'buildManifestPath'; HashField = 'buildManifestSha256'; Label = 'build manifest' },
        [pscustomobject]@{ PathField = 'noticePath'; HashField = 'noticeSha256'; Label = 'notice' }
    )
    $resourceRecords = [Collections.Generic.List[object]]::new()
    $runnerPath = $null
    foreach ($definition in $resourceDefinitions) {
        $relativePath = [string](Get-RequiredJsonProperty -Object $approval -Name $definition.PathField -Description 'PDFium approval')
        $expectedHash = [string](Get-RequiredJsonProperty -Object $approval -Name $definition.HashField -Description 'PDFium approval')
        if (-not (Test-CanonicalSha256 $expectedHash)) {
            throw "PDFium $($definition.Label) checksum is not canonical."
        }
        $resourcePath = Resolve-SafePayloadFile -Root $pdfiumRoot -RelativePath $relativePath -Description "PDFium $($definition.Label)"
        $actualHash = Get-Sha256 -Path $resourcePath
        if ($actualHash -cne $expectedHash) {
            Add-Failure "PDFium $($definition.Label) checksum differs from the packaged approval."
        }
        $resourceRecords.Add([ordered]@{
            label = $definition.Label
            relativePath = $relativePath.Replace('\', '/')
            sha256 = $actualHash
        })
        if ($definition.Label -ceq 'renderer') { $runnerPath = $resourcePath }
    }

    $pdfiumFiles = @(Get-ChildItem -LiteralPath $pdfiumRoot -File -Recurse)
    $executableSha256 = Get-Sha256 -Path $executablePath
    $applicationDllSha256 = Get-Sha256 -Path $applicationDllPath
    $runnerSha256 = if ($null -ne $runnerPath) { Get-Sha256 -Path $runnerPath } else { $null }
    $result.payload = [ordered]@{
        root = $PayloadRoot
        commit = $ExpectedCommit
        version = $ExpectedVersion
        executablePath = $executablePath
        executableSha256 = $executableSha256
        applicationDllPath = $applicationDllPath
        applicationDllSha256 = $applicationDllSha256
        pdfiumRoot = $pdfiumRoot
        approvalPath = $approvalPath
        approvalSha256 = Get-Sha256 -Path $approvalPath
        rendererPath = $runnerPath
        rendererSha256 = $runnerSha256
        resources = @($resourceRecords)
        pdfiumFileCount = $pdfiumFiles.Count
        reparsePointCount = $payloadReparsePoints.Count
    }
    if ($executableSha256 -cne $ExpectedExecutableSha256 -or
        $applicationDllSha256 -cne $ExpectedApplicationDllSha256 -or
        $runnerSha256 -cne $ExpectedRendererSha256 -or
        $pdfiumFiles.Count -ne 5 -or
        $payloadReparsePoints.Count -ne 0) {
        Add-Failure 'Packaged application or PDFium bytes differ from the exact clean-machine boundary.'
    }

    $renderRoot = Join-Path $OutputRoot 'pdfium-render-smoke'
    if (Test-Path -LiteralPath $renderRoot) {
        throw "PDFium renderer smoke output already exists: $renderRoot"
    }
    New-Item -ItemType Directory -Path $renderRoot | Out-Null
    $renderStdoutPath = Join-Path $OutputRoot 'pdfium-render-smoke.stdout.txt'
    $renderStderrPath = Join-Path $OutputRoot 'pdfium-render-smoke.stderr.txt'
    $renderArguments = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', (Quote-ProcessArgument $RendererSmokeScriptPath),
        '-RunnerPath', (Quote-ProcessArgument $runnerPath),
        '-EvidenceRoot', (Quote-ProcessArgument $renderRoot)
    )
    $renderStartedUtc = [DateTimeOffset]::UtcNow
    $renderStartInfo = [Diagnostics.ProcessStartInfo]::new()
    $renderStartInfo.FileName = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $renderStartInfo.Arguments = $renderArguments -join ' '
    $renderStartInfo.UseShellExecute = $false
    $renderStartInfo.CreateNoWindow = $true
    $renderStartInfo.RedirectStandardOutput = $true
    $renderStartInfo.RedirectStandardError = $true
    $renderProcess = [Diagnostics.Process]::new()
    $renderProcess.StartInfo = $renderStartInfo
    $renderTimedOut = $false
    $renderExitCode = $null
    try {
        if (-not $renderProcess.Start()) {
            throw 'PDFium renderer smoke process did not start.'
        }
        $renderStdoutTask = $renderProcess.StandardOutput.ReadToEndAsync()
        $renderStderrTask = $renderProcess.StandardError.ReadToEndAsync()
        if (-not $renderProcess.WaitForExit($RendererSmokeTimeoutSeconds * 1000)) {
            $renderTimedOut = $true
            Stop-Process -Id $renderProcess.Id -Force -ErrorAction SilentlyContinue
            Add-Failure "PDFium renderer smoke exceeded $RendererSmokeTimeoutSeconds seconds."
        }
        else {
            $renderProcess.WaitForExit()
            $renderExitCode = $renderProcess.ExitCode
            if ($renderExitCode -ne 0) {
                Add-Failure "PDFium renderer smoke exited with code $renderExitCode."
            }
        }
        $renderStdout = $renderStdoutTask.GetAwaiter().GetResult()
        $renderStderr = $renderStderrTask.GetAwaiter().GetResult()
        [IO.File]::WriteAllText($renderStdoutPath, $renderStdout, [Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText($renderStderrPath, $renderStderr, [Text.UTF8Encoding]::new($false))
    }
    finally {
        if (-not $renderProcess.HasExited) { Stop-Process -Id $renderProcess.Id -Force -ErrorAction SilentlyContinue }
        $renderProcess.Dispose()
    }
    $renderFinishedUtc = [DateTimeOffset]::UtcNow
    $renderReportPath = Join-Path $renderRoot 'native-render-smoke.json'
    $renderReport = if (Test-Path -LiteralPath $renderReportPath -PathType Leaf) {
        Get-Content -LiteralPath $renderReportPath -Raw | ConvertFrom-Json
    }
    else { $null }
    $renderContractPassed = $null -ne $renderReport -and
        [int](Get-RequiredJsonProperty -Object $renderReport -Name 'schemaVersion' -Description 'PDFium render report') -eq 1 -and
        [string](Get-RequiredJsonProperty -Object $renderReport -Name 'runnerSha256' -Description 'PDFium render report') -ceq $ExpectedRendererSha256 -and
        [string](Get-RequiredJsonProperty -Object $renderReport -Name 'inputKind' -Description 'PDFium render report') -ceq 'controlled-synthetic-fixture' -and
        [string](Get-RequiredJsonProperty -Object $renderReport -Name 'inputSha256' -Description 'PDFium render report') -ceq '2ebb9f3a7cdec5c76773fc7796f3056a80cc0bdaa1935b1b27e10bb9d581cf8b' -and
        (Test-JsonBoolean -Object $renderReport -Name 'inputUnchanged' -Expected $true) -and
        [string](Get-RequiredJsonProperty -Object $renderReport -Name 'rawSha256' -Description 'PDFium render report') -ceq 'b549878f8641965a3c78a659177f4b4e027fa250d161d33b9a3df551e8f72158' -and
        [int](Get-RequiredJsonProperty -Object $renderReport -Name 'width' -Description 'PDFium render report') -eq 72 -and
        [int](Get-RequiredJsonProperty -Object $renderReport -Name 'height' -Description 'PDFium render report') -eq 72 -and
        [int](Get-RequiredJsonProperty -Object $renderReport -Name 'stride' -Description 'PDFium render report') -eq 288 -and
        [int64](Get-RequiredJsonProperty -Object $renderReport -Name 'payloadLength' -Description 'PDFium render report') -eq 20736 -and
        [string](Get-RequiredJsonProperty -Object $renderReport -Name 'stdout' -Description 'PDFium render report') -ceq 'OK 72 72 288' -and
        [string](Get-RequiredJsonProperty -Object $renderReport -Name 'stderr' -Description 'PDFium render report') -ceq ''
    if (-not $renderContractPassed) {
        Add-Failure 'PDFium renderer did not produce the exact fixed-fixture output contract.'
    }
    $result.rendererSmoke = [ordered]@{
        scriptPath = $RendererSmokeScriptPath
        scriptSha256 = $smokeScriptSha256
        startedUtc = $renderStartedUtc.ToString('O')
        finishedUtc = $renderFinishedUtc.ToString('O')
        timeoutSeconds = $RendererSmokeTimeoutSeconds
        timedOut = $renderTimedOut
        exitCode = $renderExitCode
        reportPath = $renderReportPath
        reportSha256 = if ($null -ne $renderReport) { Get-Sha256 -Path $renderReportPath } else { $null }
        contractPassed = $renderContractPassed
        result = $renderReport
    }

    $previousPdfiumApproval = [Environment]::GetEnvironmentVariable('GRAPHREADER_PDFIUM_APPROVAL_PATH', 'Process')
    $applicationStartedUtc = [DateTimeOffset]::UtcNow
    $applicationTimedOut = $false
    $applicationExitCode = $null
    try {
        [Environment]::SetEnvironmentVariable('GRAPHREADER_PDFIUM_APPROVAL_PATH', $null, 'Process')
        $applicationProcess = Start-Process -FilePath $executablePath `
            -ArgumentList @('--production-runtime-smoke', '--require-packaged-pdfium') `
            -WorkingDirectory $PayloadRoot -WindowStyle Hidden -PassThru
        try {
            if (-not $applicationProcess.WaitForExit($ApplicationSmokeTimeoutSeconds * 1000)) {
                $applicationTimedOut = $true
                Stop-Process -Id $applicationProcess.Id -Force -ErrorAction SilentlyContinue
                Add-Failure "Packaged PDFium application fallback exceeded $ApplicationSmokeTimeoutSeconds seconds."
            }
            else {
                $applicationExitCode = $applicationProcess.ExitCode
                if ($applicationExitCode -ne 0) {
                    Add-Failure "Packaged PDFium application fallback exited with code $applicationExitCode."
                }
            }
        }
        finally {
            if (-not $applicationProcess.HasExited) {
                Stop-Process -Id $applicationProcess.Id -Force -ErrorAction SilentlyContinue
            }
            $applicationProcess.Dispose()
        }
    }
    finally {
        [Environment]::SetEnvironmentVariable('GRAPHREADER_PDFIUM_APPROVAL_PATH', $previousPdfiumApproval, 'Process')
    }
    $applicationFinishedUtc = [DateTimeOffset]::UtcNow
    $result.applicationPackagedFallback = [ordered]@{
        arguments = @('--production-runtime-smoke', '--require-packaged-pdfium')
        environmentApprovalUnset = $true
        expectedPackagedApprovalPath = $approvalPath
        startedUtc = $applicationStartedUtc.ToString('O')
        finishedUtc = $applicationFinishedUtc.ToString('O')
        timeoutSeconds = $ApplicationSmokeTimeoutSeconds
        timedOut = $applicationTimedOut
        exitCode = $applicationExitCode
        passed = -not $applicationTimedOut -and $applicationExitCode -eq 0
    }
}
catch {
    Add-Failure "Harness error: $($_.Exception.Message)"
}

$result.failures = @($failures)
if ($failures.Count -eq 0 -and
    $null -ne $result.rendererSmoke -and [bool]$result.rendererSmoke.contractPassed -and
    $null -ne $result.applicationPackagedFallback -and [bool]$result.applicationPackagedFallback.passed) {
    $result.status = 'pass'
}

[IO.File]::WriteAllText(
    $evidencePath,
    (($result | ConvertTo-Json -Depth 14) + [Environment]::NewLine),
    [Text.UTF8Encoding]::new($false))

Write-Host "PDFium clean-machine validation: $($result.status.ToUpperInvariant())"
Write-Host "Evidence: $evidencePath"
if ($result.status -cne 'pass') {
    foreach ($failure in $failures) { Write-Host "BLOCKED: $failure" }
    exit 2
}

exit 0
