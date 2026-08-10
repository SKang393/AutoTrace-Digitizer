# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EvidenceRoot,

    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedCommit,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedVersion,

    [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
}
else {
    $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
}
$EvidenceRoot = [IO.Path]::GetFullPath($EvidenceRoot)
$DestinationRoot = [IO.Path]::GetFullPath($DestinationRoot)
$repositoryPrefix = $RepositoryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
$evidencePrefix = $EvidenceRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
$releaseAuditPath = Join-Path $RepositoryRoot 'packaging\common\release-audit.json'
$validatorPath = Join-Path $RepositoryRoot 'packaging\pdfium-source\Test-ReviewedPdfiumEvidence.ps1'
$rendererSmokeScriptPath = Join-Path $RepositoryRoot 'packaging\pdfium-source\Test-PdfiumRunner.ps1'
$cleanMachineHarnessPath = Join-Path $RepositoryRoot 'packaging\clean-machine\Invoke-GraphReaderPdfiumCleanMachineValidation.ps1'
$approvalPath = Join-Path $EvidenceRoot 'reviewed-approval.json'

function Get-SafeEvidenceFile {
    param(
        [Parameter(Mandatory = $true)] [string]$RelativePath,
        [Parameter(Mandatory = $true)] [string]$Label
    )

    if ([IO.Path]::IsPathRooted($RelativePath) -or
        $RelativePath -match '(^|[\\/])\.\.([\\/]|$)') {
        throw "$Label uses an unsafe relative path: $RelativePath"
    }
    $fullPath = [IO.Path]::GetFullPath((Join-Path $EvidenceRoot $RelativePath))
    if (-not $fullPath.StartsWith($evidencePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        throw "$Label is missing or outside the PDFium evidence root: $RelativePath"
    }
    $current = $fullPath
    while ($current.StartsWith($evidencePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        if (((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Label contains a reparse point: $current"
        }
        $current = Split-Path -Parent $current
    }
    return $fullPath
}

foreach ($required in @($releaseAuditPath, $validatorPath, $rendererSmokeScriptPath, $cleanMachineHarnessPath, $approvalPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Release PDFium input is missing: $required"
    }
}
if ($ExpectedCommit -notmatch '^[0-9a-f]{40}$' -or
    $ExpectedVersion -notmatch '^\d{1,2}\.\d{1,2}\.\d{1,2}$') {
    throw 'Release PDFium evidence requires a canonical expected commit and version.'
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

function Assert-JsonBoolean {
    param(
        [Parameter(Mandatory = $true)][object]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][bool]$Expected,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $value = Get-RequiredJsonProperty -Object $Object -Name $Name -Description $Description
    if (-not ($value -is [bool]) -or [bool]$value -ne $Expected) {
        throw "$Description property '$Name' must be the JSON Boolean '$Expected'."
    }
}

function Assert-CanonicalSha256 {
    param(
        [AllowNull()][object]$Value,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (-not ($Value -is [string]) -or [string]$Value -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Description must be a canonical lowercase SHA-256 value."
    }
}

function Assert-PdfiumCleanMachineEvidence {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedRendererSha256,
        [Parameter(Mandatory = $true)][string]$ExpectedHarnessSha256,
        [Parameter(Mandatory = $true)][string]$ExpectedSmokeScriptSha256,
        [Parameter(Mandatory = $true)][string]$ExpectedExecutableSha256,
        [Parameter(Mandatory = $true)][string]$ExpectedApplicationDllSha256,
        [Parameter(Mandatory = $true)][string]$ExpectedApprovalSha256,
        [Parameter(Mandatory = $true)][object[]]$ExpectedResources
    )

    $evidence = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ([string](Get-RequiredJsonProperty -Object $evidence -Name 'schema' -Description 'PDFium clean-machine evidence') -cne
        'graphreader.pdfium-clean-machine-load.v1' -or
        [string](Get-RequiredJsonProperty -Object $evidence -Name 'status' -Description 'PDFium clean-machine evidence') -cne 'pass') {
        throw 'PDFium clean-machine evidence schema or status is not directly passing.'
    }
    $reportedHarnessSha256 = Get-RequiredJsonProperty -Object $evidence -Name 'harnessSha256' -Description 'PDFium clean-machine evidence'
    Assert-CanonicalSha256 -Value $reportedHarnessSha256 -Description 'PDFium clean-machine harness checksum'
    if ([string]$reportedHarnessSha256 -cne $ExpectedHarnessSha256) {
        throw 'PDFium clean-machine evidence was not generated by the current reviewed harness.'
    }
    if (@(Get-RequiredJsonProperty -Object $evidence -Name 'failures' -Description 'PDFium clean-machine evidence').Count -ne 0) {
        throw 'PDFium clean-machine evidence retains one or more failures.'
    }

    $payload = Get-RequiredJsonProperty -Object $evidence -Name 'payload' -Description 'PDFium clean-machine evidence'
    if ([string](Get-RequiredJsonProperty -Object $payload -Name 'commit' -Description 'Clean-machine payload') -cne $ExpectedCommit -or
        [string](Get-RequiredJsonProperty -Object $payload -Name 'version' -Description 'Clean-machine payload') -cne $ExpectedVersion) {
        throw 'Clean-machine payload does not bind the release commit and version.'
    }
    $payloadExecutableSha256 = Get-RequiredJsonProperty -Object $payload -Name 'executableSha256' -Description 'Clean-machine payload'
    $payloadApplicationDllSha256 = Get-RequiredJsonProperty -Object $payload -Name 'applicationDllSha256' -Description 'Clean-machine payload'
    $payloadRendererSha256 = Get-RequiredJsonProperty -Object $payload -Name 'rendererSha256' -Description 'Clean-machine payload'
    $payloadApprovalSha256 = Get-RequiredJsonProperty -Object $payload -Name 'approvalSha256' -Description 'Clean-machine payload'
    foreach ($hashValue in @($payloadExecutableSha256, $payloadApplicationDllSha256, $payloadRendererSha256, $payloadApprovalSha256)) {
        Assert-CanonicalSha256 -Value $hashValue -Description 'Clean-machine payload checksum'
    }
    if ([string]$payloadExecutableSha256 -cne $ExpectedExecutableSha256 -or
        [string]$payloadApplicationDllSha256 -cne $ExpectedApplicationDllSha256 -or
        [string]$payloadRendererSha256 -cne $ExpectedRendererSha256 -or
        [string]$payloadApprovalSha256 -cne $ExpectedApprovalSha256 -or
        [int](Get-RequiredJsonProperty -Object $payload -Name 'pdfiumFileCount' -Description 'Clean-machine payload') -ne 5 -or
        [int](Get-RequiredJsonProperty -Object $payload -Name 'reparsePointCount' -Description 'Clean-machine payload') -ne 0) {
        throw 'Clean-machine application, PDFium bytes, or containment policy differ from release staging.'
    }

    $reportedResources = @(Get-RequiredJsonProperty -Object $payload -Name 'resources' -Description 'Clean-machine payload')
    if ($reportedResources.Count -ne $ExpectedResources.Count) {
        throw 'Clean-machine PDFium resource inventory has an unexpected size.'
    }
    foreach ($expectedResource in $ExpectedResources) {
        $matches = @($reportedResources | Where-Object {
                [string]$_.label -ceq [string]$expectedResource.Label -and
                [string]$_.relativePath -ceq [string]$expectedResource.RelativePath
            })
        if ($matches.Count -ne 1) {
            throw "Clean-machine PDFium resource is missing or duplicated: $($expectedResource.Label)."
        }
        $reportedHash = Get-RequiredJsonProperty -Object $matches[0] -Name 'sha256' -Description "PDFium $($expectedResource.Label) resource"
        Assert-CanonicalSha256 -Value $reportedHash -Description "PDFium $($expectedResource.Label) resource checksum"
        if ([string]$reportedHash -cne [string]$expectedResource.Hash) {
            throw "Clean-machine PDFium resource checksum differs: $($expectedResource.Label)."
        }
    }

    $rendererSmoke = Get-RequiredJsonProperty -Object $evidence -Name 'rendererSmoke' -Description 'PDFium clean-machine evidence'
    $rendererSmokeScriptSha256 = Get-RequiredJsonProperty -Object $rendererSmoke -Name 'scriptSha256' -Description 'PDFium renderer smoke'
    Assert-CanonicalSha256 -Value $rendererSmokeScriptSha256 -Description 'PDFium renderer smoke script checksum'
    Assert-JsonBoolean -Object $rendererSmoke -Name 'timedOut' -Expected $false -Description 'PDFium renderer smoke'
    Assert-JsonBoolean -Object $rendererSmoke -Name 'contractPassed' -Expected $true -Description 'PDFium renderer smoke'
    if ([string]$rendererSmokeScriptSha256 -cne $ExpectedSmokeScriptSha256 -or
        [int](Get-RequiredJsonProperty -Object $rendererSmoke -Name 'exitCode' -Description 'PDFium renderer smoke') -ne 0) {
        throw 'PDFium renderer smoke did not use the current script or exit successfully.'
    }
    $renderResult = Get-RequiredJsonProperty -Object $rendererSmoke -Name 'result' -Description 'PDFium renderer smoke'
    Assert-JsonBoolean -Object $renderResult -Name 'inputUnchanged' -Expected $true -Description 'PDFium renderer output'
    if ([int](Get-RequiredJsonProperty -Object $renderResult -Name 'schemaVersion' -Description 'PDFium renderer output') -ne 1 -or
        [string](Get-RequiredJsonProperty -Object $renderResult -Name 'runnerSha256' -Description 'PDFium renderer output') -cne $ExpectedRendererSha256 -or
        [string](Get-RequiredJsonProperty -Object $renderResult -Name 'inputKind' -Description 'PDFium renderer output') -cne 'controlled-synthetic-fixture' -or
        [string](Get-RequiredJsonProperty -Object $renderResult -Name 'inputSha256' -Description 'PDFium renderer output') -cne '2ebb9f3a7cdec5c76773fc7796f3056a80cc0bdaa1935b1b27e10bb9d581cf8b' -or
        [string](Get-RequiredJsonProperty -Object $renderResult -Name 'rawSha256' -Description 'PDFium renderer output') -cne 'b549878f8641965a3c78a659177f4b4e027fa250d161d33b9a3df551e8f72158' -or
        [int](Get-RequiredJsonProperty -Object $renderResult -Name 'width' -Description 'PDFium renderer output') -ne 72 -or
        [int](Get-RequiredJsonProperty -Object $renderResult -Name 'height' -Description 'PDFium renderer output') -ne 72 -or
        [int](Get-RequiredJsonProperty -Object $renderResult -Name 'stride' -Description 'PDFium renderer output') -ne 288 -or
        [int64](Get-RequiredJsonProperty -Object $renderResult -Name 'payloadLength' -Description 'PDFium renderer output') -ne 20736 -or
        [string](Get-RequiredJsonProperty -Object $renderResult -Name 'stdout' -Description 'PDFium renderer output') -cne 'OK 72 72 288' -or
        [string](Get-RequiredJsonProperty -Object $renderResult -Name 'stderr' -Description 'PDFium renderer output') -cne '') {
        throw 'PDFium clean-machine render output differs from the fixed public fixture contract.'
    }

    $applicationFallback = Get-RequiredJsonProperty -Object $evidence -Name 'applicationPackagedFallback' -Description 'PDFium clean-machine evidence'
    Assert-JsonBoolean -Object $applicationFallback -Name 'environmentApprovalUnset' -Expected $true -Description 'Packaged PDFium application fallback'
    Assert-JsonBoolean -Object $applicationFallback -Name 'timedOut' -Expected $false -Description 'Packaged PDFium application fallback'
    Assert-JsonBoolean -Object $applicationFallback -Name 'passed' -Expected $true -Description 'Packaged PDFium application fallback'
    $applicationArguments = @(Get-RequiredJsonProperty -Object $applicationFallback -Name 'arguments' -Description 'Packaged PDFium application fallback')
    if ($applicationArguments.Count -ne 2 -or
        [string]$applicationArguments[0] -cne '--production-runtime-smoke' -or
        [string]$applicationArguments[1] -cne '--require-packaged-pdfium' -or
        [int](Get-RequiredJsonProperty -Object $applicationFallback -Name 'exitCode' -Description 'Packaged PDFium application fallback') -ne 0) {
        throw 'The production application did not exercise the packaged PDFium fallback.'
    }

    $machine = Get-RequiredJsonProperty -Object $evidence -Name 'machine' -Description 'PDFium clean-machine evidence'
    Assert-JsonBoolean -Object $machine -Name 'is64BitOperatingSystem' -Expected $true -Description 'Clean Windows machine'
    Assert-JsonBoolean -Object $machine -Name 'is64BitProcess' -Expected $true -Description 'Clean Windows machine'
    Assert-JsonBoolean -Object $machine -Name 'networkQuerySucceeded' -Expected $true -Description 'Clean Windows machine'
    $installAgeHours = [double](Get-RequiredJsonProperty -Object $machine -Name 'installAgeHours' -Description 'Clean Windows machine')
    if ([string](Get-RequiredJsonProperty -Object $machine -Name 'productName' -Description 'Clean Windows machine') -notmatch 'Windows' -or
        [string](Get-RequiredJsonProperty -Object $machine -Name 'architecture' -Description 'Clean Windows machine') -notmatch '64' -or
        [string](Get-RequiredJsonProperty -Object $machine -Name 'manufacturer' -Description 'Clean Windows machine') -notmatch '(?i)qemu' -or
        $installAgeHours -lt 0 -or $installAgeHours -gt 24 -or
        @(Get-RequiredJsonProperty -Object $machine -Name 'developerToolsOnPath' -Description 'Clean Windows machine').Count -ne 0 -or
        [int](Get-RequiredJsonProperty -Object $machine -Name 'networkAdaptersUp' -Description 'Clean Windows machine') -ne 0) {
        throw 'Clean Windows host identity, freshness, toolchain, or offline state is not passing.'
    }

    $vmProvenance = Get-RequiredJsonProperty -Object $evidence -Name 'vmProvenance' -Description 'PDFium clean-machine evidence'
    if ([string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'schema' -Description 'VM provenance') -cne 'graphreader.clean-windows-vm-provenance.v1' -or
        [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'environmentKind' -Description 'VM provenance') -cne 'fresh-windows-evaluation-vm' -or
        [string](Get-RequiredJsonProperty -Object $vmProvenance -Name 'networkMode' -Description 'VM provenance') -cne 'none') {
        throw 'VM provenance schema or execution boundary is invalid.'
    }
    Assert-JsonBoolean -Object $vmProvenance -Name 'isoSha256Verified' -Expected $true -Description 'VM provenance'
    Assert-JsonBoolean -Object $vmProvenance -Name 'freshInstall' -Expected $true -Description 'VM provenance'
    foreach ($hashField in @('isoSha256', 'officialIsoSha256', 'vmConfigurationSha256', 'qemuInstallerSha256', 'expectedExecutableSha256', 'expectedApplicationDllSha256', 'expectedPdfiumRendererSha256', 'expectedPdfiumHarnessSha256', 'expectedRendererSmokeScriptSha256')) {
        Assert-CanonicalSha256 -Value (Get-RequiredJsonProperty -Object $vmProvenance -Name $hashField -Description 'VM provenance') -Description "VM provenance $hashField"
    }
    if ([string]$vmProvenance.isoSha256 -cne [string]$vmProvenance.officialIsoSha256 -or
        [string]$vmProvenance.expectedCommit -cne $ExpectedCommit -or
        [string]$vmProvenance.expectedVersion -cne $ExpectedVersion -or
        [string]$vmProvenance.expectedExecutableSha256 -cne $ExpectedExecutableSha256 -or
        [string]$vmProvenance.expectedApplicationDllSha256 -cne $ExpectedApplicationDllSha256 -or
        [string]$vmProvenance.expectedPdfiumRendererSha256 -cne $ExpectedRendererSha256 -or
        [string]$vmProvenance.expectedPdfiumHarnessSha256 -cne $ExpectedHarnessSha256 -or
        [string]$vmProvenance.expectedRendererSmokeScriptSha256 -cne $ExpectedSmokeScriptSha256 -or
        ([string]$vmProvenance.vmId).ToLowerInvariant() -cne ([string]$machine.machineUuid).ToLowerInvariant() -or
        [string]$vmProvenance.expectedOsBuild -cne [string]$machine.buildNumber -or
        [int]$vmProvenance.expectedOsUbr -ne [int]$machine.updateBuildRevision -or
        [string]::IsNullOrWhiteSpace([string]$vmProvenance.qemuVersion)) {
        throw 'VM provenance does not bind the observed guest, application, renderer, and guest scripts.'
    }
}

$releaseAudit = Get-Content -LiteralPath $releaseAuditPath -Raw | ConvertFrom-Json
$components = @($releaseAudit.components | Where-Object {
        [string]$_.id -ceq 'pdfium-native'
    })
if ($components.Count -ne 1) {
    throw "Release audit must contain exactly one 'pdfium-native' component."
}
$component = $components[0]
$expectedHash = [string]$component.artifactSha256
if ([string]$component.checksumPolicy -cne 'exact-binary' -or
    $expectedHash -notmatch '^[0-9a-fA-F]{64}$') {
    throw 'Release PDFium component must use exact-binary coverage with an artifact SHA-256.'
}
$expectedHash = $expectedHash.ToLowerInvariant()

$gates = @($releaseAudit.mandatoryEvidenceGates | Where-Object {
        [string]$_.id -ceq 'pdfium-clean-machine-load'
    })
if ($gates.Count -ne 1) {
    throw "Release audit must contain exactly one 'pdfium-clean-machine-load' gate."
}
$gate = $gates[0]
if ([string]$gate.status -cne 'pass' -or @($gate.evidence).Count -eq 0) {
    throw 'PDFium clean-machine evidence gate is not directly passing.'
}
$gateEvidence = @($gate.evidence)
if ($gateEvidence.Count -ne 1) {
    throw 'PDFium clean-machine evidence gate must contain exactly one reviewed report.'
}
$cleanMachineEvidencePaths = [Collections.Generic.List[string]]::new()
foreach ($evidence in @($gate.evidence)) {
    $relativePath = [string]$evidence.path
    $evidenceHash = [string]$evidence.sha256
    if ([IO.Path]::IsPathRooted($relativePath) -or
        $relativePath -match '(^|[\\/])\.\.([\\/]|$)' -or
        $evidenceHash -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'PDFium clean-machine evidence uses an unsafe path or invalid checksum.'
    }
    $evidencePath = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $relativePath))
    if (-not $evidencePath.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $evidencePath -PathType Leaf)) {
        throw "PDFium clean-machine evidence is missing or outside the repository: $relativePath"
    }
    $actualEvidenceHash = (Get-FileHash -LiteralPath $evidencePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [string]::Equals($actualEvidenceHash, $evidenceHash, [StringComparison]::OrdinalIgnoreCase)) {
        throw "PDFium clean-machine evidence checksum differs: $relativePath"
    }
    $cleanMachineEvidencePaths.Add($evidencePath)
}

$destinationExecutablePath = Join-Path $DestinationRoot 'GraphReader.App.exe'
$destinationApplicationDllPath = Join-Path $DestinationRoot 'GraphReader.App.dll'
if (-not (Test-Path -LiteralPath $destinationExecutablePath -PathType Leaf)) {
    throw "Published application executable is missing: $destinationExecutablePath"
}
if (-not (Test-Path -LiteralPath $destinationApplicationDllPath -PathType Leaf)) {
    throw "Published application assembly is missing: $destinationApplicationDllPath"
}
$destinationExecutableSha256 = (Get-FileHash -LiteralPath $destinationExecutablePath -Algorithm SHA256).Hash.ToLowerInvariant()
$destinationApplicationDllSha256 = (Get-FileHash -LiteralPath $destinationApplicationDllPath -Algorithm SHA256).Hash.ToLowerInvariant()
$cleanMachineHarnessSha256 = (Get-FileHash -LiteralPath $cleanMachineHarnessPath -Algorithm SHA256).Hash.ToLowerInvariant()
$rendererSmokeScriptSha256 = (Get-FileHash -LiteralPath $rendererSmokeScriptPath -Algorithm SHA256).Hash.ToLowerInvariant()

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $validatorPath -EvidenceRoot $EvidenceRoot
if ($LASTEXITCODE -ne 0) {
    throw "Reviewed PDFium evidence validation failed with exit code $LASTEXITCODE."
}

$approval = Get-Content -LiteralPath $approvalPath -Raw | ConvertFrom-Json
Assert-JsonBoolean -Object $approval -Name 'reviewApproved' -Expected $true -Description 'Reviewed PDFium approval'
Assert-JsonBoolean -Object $approval -Name 'redistributionApproved' -Expected $true -Description 'Reviewed PDFium approval'
Assert-JsonBoolean -Object $approval -Name 'bundlingApproved' -Expected $true -Description 'Reviewed PDFium approval'
if ([int]$approval.schemaVersion -ne 1 -or
    [string]$approval.source -cne 'https://pdfium.googlesource.com/pdfium' -or
    [string]$approval.sourceRevision -cne '2870fa9244b0f0f69fb743fab1e08deefcb07b2b' -or
    [string]$approval.rendererVersion -cne '2870fa9244b0f0f69fb743fab1e08deefcb07b2b' -or
    [string]$approval.licenseSpdx -cne 'BSD-3-Clause') {
    throw 'Reviewed PDFium approval does not satisfy the pinned release policy.'
}

$resources = @(
    [pscustomobject]@{ RelativePath = [string]$approval.binaryPath; Hash = [string]$approval.binarySha256; Label = 'PDFium runner' },
    [pscustomobject]@{ RelativePath = [string]$approval.sourceLockPath; Hash = [string]$approval.sourceLockSha256; Label = 'PDFium source lock' },
    [pscustomobject]@{ RelativePath = [string]$approval.buildManifestPath; Hash = [string]$approval.buildManifestSha256; Label = 'PDFium build manifest' },
    [pscustomobject]@{ RelativePath = [string]$approval.noticePath; Hash = [string]$approval.noticeSha256; Label = 'PDFium reviewed notice' }
)
$resolvedResources = foreach ($resource in $resources) {
    if ($resource.Hash -notmatch '^[0-9a-fA-F]{64}$') {
        throw "$($resource.Label) approval checksum is invalid."
    }
    $sourcePath = Get-SafeEvidenceFile -RelativePath $resource.RelativePath -Label $resource.Label
    $actualHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [string]::Equals($actualHash, $resource.Hash, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$($resource.Label) checksum differs from reviewed approval."
    }
    [pscustomobject]@{
        SourcePath = $sourcePath
        RelativePath = $resource.RelativePath.Replace('/', [IO.Path]::DirectorySeparatorChar)
        Hash = $actualHash
        Label = $resource.Label
    }
}
$runner = @($resolvedResources | Where-Object { $_.Label -eq 'PDFium runner' })
if ($runner.Count -ne 1 -or -not [string]::Equals($runner[0].Hash, $expectedHash, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Reviewed PDFium runner does not match the exact release-audit binary.'
}

$resourceLabelMap = @{
    'PDFium runner' = 'renderer'
    'PDFium source lock' = 'source lock'
    'PDFium build manifest' = 'build manifest'
    'PDFium reviewed notice' = 'notice'
}
$expectedCleanMachineResources = @($resolvedResources | ForEach-Object {
        [pscustomobject]@{
            Label = [string]$resourceLabelMap[[string]$_.Label]
            RelativePath = ([string]$_.RelativePath).Replace('\', '/')
            Hash = [string]$_.Hash
        }
    })
$approvalSha256 = (Get-FileHash -LiteralPath $approvalPath -Algorithm SHA256).Hash.ToLowerInvariant()
foreach ($cleanMachineEvidencePath in $cleanMachineEvidencePaths) {
    Assert-PdfiumCleanMachineEvidence `
        -Path $cleanMachineEvidencePath `
        -ExpectedRendererSha256 $expectedHash `
        -ExpectedHarnessSha256 $cleanMachineHarnessSha256 `
        -ExpectedSmokeScriptSha256 $rendererSmokeScriptSha256 `
        -ExpectedExecutableSha256 $destinationExecutableSha256 `
        -ExpectedApplicationDllSha256 $destinationApplicationDllSha256 `
        -ExpectedApprovalSha256 $approvalSha256 `
        -ExpectedResources $expectedCleanMachineResources
}

if (-not (Test-Path -LiteralPath $DestinationRoot -PathType Container)) {
    throw "Release publish destination is missing: $DestinationRoot"
}
$targetRoot = Join-Path $DestinationRoot 'pdfium'
if (Test-Path -LiteralPath $targetRoot) {
    throw "Release PDFium target already exists: $targetRoot"
}
$temporaryRoot = Join-Path $DestinationRoot ('.pdfium-release-' + [Guid]::NewGuid().ToString('N'))
try {
    New-Item -ItemType Directory -Path $temporaryRoot -Force | Out-Null
    foreach ($resource in $resolvedResources) {
        $targetPath = [IO.Path]::GetFullPath((Join-Path $temporaryRoot $resource.RelativePath))
        $temporaryPrefix = $temporaryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        if (-not $targetPath.StartsWith($temporaryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "$($resource.Label) staging path escapes the release PDFium root."
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
        Copy-Item -LiteralPath $resource.SourcePath -Destination $targetPath
        $stagedHash = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($stagedHash -ne $resource.Hash) {
            throw "$($resource.Label) checksum changed during release staging."
        }
    }
    Copy-Item -LiteralPath $approvalPath -Destination (Join-Path $temporaryRoot 'reviewed-approval.json')
    Move-Item -LiteralPath $temporaryRoot -Destination $targetRoot
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        $resolvedTemporary = [IO.Path]::GetFullPath($temporaryRoot)
        $destinationPrefix = $DestinationRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        if ($resolvedTemporary.StartsWith($destinationPrefix, [StringComparison]::OrdinalIgnoreCase) -and
            [IO.Path]::GetFileName($resolvedTemporary).StartsWith('.pdfium-release-', [StringComparison]::Ordinal)) {
            Remove-Item -LiteralPath $resolvedTemporary -Recurse -Force
        }
    }
}

Write-Host "Release-approved PDFium renderer installed: $expectedHash"
Write-Output ([pscustomobject]@{
    ApprovalPath = Join-Path $targetRoot 'reviewed-approval.json'
    BinaryPath = Join-Path $targetRoot $runner[0].RelativePath
    BinarySha256 = $expectedHash
    CleanMachineEvidence = $true
    ReleaseApproved = $true
})
