# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$EvidenceRoot = (Join-Path $PSScriptRoot '..\..\..\artifacts\pdfium-source\evidence'),
    [string]$ReportPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$profileRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $profileRoot '..\..'))
$testParent = [IO.Path]::GetFullPath((Join-Path $repositoryRoot 'artifacts\pdfium-source\staging-tests'))
$testRoot = Join-Path $testParent ([Guid]::NewGuid().ToString('N'))
$stageScript = Join-Path $profileRoot 'Stage-InternalPortablePdfium.ps1'
$policyPath = Join-Path $profileRoot 'internal-portable-policy.json'
$fullEvidenceRoot = [IO.Path]::GetFullPath($EvidenceRoot)
$results = [Collections.Generic.List[string]]::new()

function Assert-Fails {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [Parameter(Mandatory = $true)][string]$MessagePattern,
        [Parameter(Mandatory = $true)][string]$Scenario
    )

    try {
        & $Action
        throw "$Scenario unexpectedly succeeded."
    }
    catch {
        if ($_.Exception.Message -notmatch $MessagePattern) { throw }
        $results.Add("PASS $Scenario : $($_.Exception.Message)")
    }
}

function New-EvidenceFixture {
    param([Parameter(Mandatory = $true)][string]$Name)
    $root = Join-Path $testRoot $Name
    New-Item -ItemType Directory -Path (Join-Path $root 'bin') -Force | Out-Null
    foreach ($fileName in @(
        'source-lock.json',
        'build-manifest.json',
        'target-dependencies.txt',
        'pe-imports.txt',
        'third-party-notices.candidate.txt',
        'reviewed-approval.candidate.json'
    )) {
        Copy-Item -LiteralPath (Join-Path $fullEvidenceRoot $fileName) -Destination (Join-Path $root $fileName)
    }
    Copy-Item -LiteralPath (Join-Path $fullEvidenceRoot 'bin\graphreader_pdfium_renderer.exe') -Destination (Join-Path $root 'bin\graphreader_pdfium_renderer.exe')
    return $root
}

function Remove-VerifiedTestRoot {
    if (-not (Test-Path -LiteralPath $testRoot)) { return }
    $full = [IO.Path]::GetFullPath($testRoot)
    $prefix = [IO.Path]::TrimEndingDirectorySeparator($testParent) + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove unexpected staging test path: $full"
    }
    Remove-Item -LiteralPath $full -Recurse -Force
}

New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
$temporaryPolicy = Join-Path $profileRoot ('.internal-portable-policy-test-' + [Guid]::NewGuid().ToString('N') + '.json')
$junction = $null
try {
    $target = Join-Path $testRoot 'valid-stage'
    $null = & $stageScript -EvidenceRoot $fullEvidenceRoot -TargetRoot $target
    $null = & $stageScript -EvidenceRoot $fullEvidenceRoot -TargetRoot $target
    $files = @(Get-ChildItem -LiteralPath $target -File)
    if ($files.Count -ne 3) { throw 'Valid staging did not contain exactly three files.' }
    $metadata = Get-Content -LiteralPath (Join-Path $target 'pdfium-internal-metadata.json') -Raw | ConvertFrom-Json
    if ($metadata.reviewApproved -ne $false -or $metadata.cleanMachineEvidence -ne $false -or $metadata.releaseApproved -ne $false) {
        throw 'Valid staging metadata approval flags are not all false.'
    }
    $results.Add('PASS exact staging and idempotent verification')

    Add-Content -LiteralPath (Join-Path $target 'pdfium-internal-metadata.json') -Value 'tampered'
    Assert-Fails { & $stageScript -EvidenceRoot $fullEvidenceRoot -TargetRoot $target | Out-Null } 'does not match the exact staged payload' 'staged metadata tampering rejection'

    $tamperedEvidence = New-EvidenceFixture 'tampered-evidence'
    [IO.File]::AppendAllText((Join-Path $tamperedEvidence 'bin\graphreader_pdfium_renderer.exe'), 'tampered')
    Assert-Fails { & $stageScript -EvidenceRoot $tamperedEvidence -TargetRoot (Join-Path $testRoot 'tampered-output') | Out-Null } 'length does not match policy|SHA-256 does not match policy' 'runner tampering rejection'

    $ambiguousEvidence = New-EvidenceFixture 'ambiguous-evidence'
    [IO.File]::WriteAllText((Join-Path $ambiguousEvidence 'bin\unexpected.dll'), 'unexpected')
    Assert-Fails { & $stageScript -EvidenceRoot $ambiguousEvidence -TargetRoot (Join-Path $testRoot 'ambiguous-output') | Out-Null } 'bin directory is ambiguous' 'ambiguous runner directory rejection'

    $outsideTarget = [IO.Path]::GetFullPath((Join-Path $repositoryRoot ('..\pdfium-stage-outside-' + [Guid]::NewGuid().ToString('N'))))
    Assert-Fails { & $stageScript -EvidenceRoot $fullEvidenceRoot -TargetRoot $outsideTarget | Out-Null } 'must remain under' 'target traversal rejection'
    if (Test-Path -LiteralPath $outsideTarget) { throw 'Traversal rejection created the outside target.' }

    Copy-Item -LiteralPath $policyPath -Destination $temporaryPolicy
    Add-Content -LiteralPath $temporaryPolicy -Value ' '
    Assert-Fails { & $stageScript -EvidenceRoot $fullEvidenceRoot -TargetRoot (Join-Path $testRoot 'policy-output') -PolicyPath $temporaryPolicy | Out-Null } 'policy SHA-256 is not the tracked value' 'policy tampering rejection'

    if ([OperatingSystem]::IsWindows()) {
        $reparseEvidence = Join-Path $testRoot 'reparse-evidence'
        New-Item -ItemType Directory -Path $reparseEvidence | Out-Null
        foreach ($fileName in @(
            'source-lock.json',
            'build-manifest.json',
            'target-dependencies.txt',
            'pe-imports.txt',
            'third-party-notices.candidate.txt',
            'reviewed-approval.candidate.json'
        )) {
            Copy-Item -LiteralPath (Join-Path $fullEvidenceRoot $fileName) -Destination (Join-Path $reparseEvidence $fileName)
        }
        $junction = Join-Path $reparseEvidence 'bin'
        $junctionTarget = Join-Path $fullEvidenceRoot 'bin'
        & cmd.exe /d /c mklink /J $junction $junctionTarget | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Unable to create controlled staging-test junction.' }
        Assert-Fails { & $stageScript -EvidenceRoot $reparseEvidence -TargetRoot (Join-Path $testRoot 'reparse-output') | Out-Null } 'contains a reparse point' 'reparse-point input rejection'
    }

    $results.Add('PASS approval flags remain false')
    $results | ForEach-Object { Write-Host $_ }
    if (-not [string]::IsNullOrWhiteSpace($ReportPath)) {
        $fullReport = [IO.Path]::GetFullPath($ReportPath)
        $reportParent = [IO.Path]::GetDirectoryName($fullReport)
        if (-not (Test-Path -LiteralPath $reportParent)) { New-Item -ItemType Directory -Path $reportParent -Force | Out-Null }
        [IO.File]::WriteAllLines($fullReport, $results, [Text.UTF8Encoding]::new($false))
    }
}
finally {
    if (Test-Path -LiteralPath $temporaryPolicy) { Remove-Item -LiteralPath $temporaryPolicy -Force }
    if ($null -ne $junction -and (Test-Path -LiteralPath $junction)) { [IO.Directory]::Delete($junction) }
    Remove-VerifiedTestRoot
}
