# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$profileRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$validatorPath = Join-Path $profileRoot 'Invoke-GraphReaderCleanMachineValidation.ps1'
$pdfiumValidatorPath = Join-Path $profileRoot 'Invoke-GraphReaderPdfiumCleanMachineValidation.ps1'
$readmePath = Join-Path $profileRoot 'README.md'
$passed = 0
$failed = 0

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )

    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    [IO.File]::WriteAllText($Path, $Content, [Text.UTF8Encoding]::new($false))
}

function Invoke-Case {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    try {
        & $Action
        $script:passed++
        Write-Host "PASS $Name"
    }
    catch {
        $script:failed++
        Write-Host "FAIL $Name`: $($_.Exception.Message)"
    }
}

Invoke-Case -Name 'Validator parses in Windows PowerShell' -Action {
    $tokens = $null
    $errors = $null
    [Management.Automation.Language.Parser]::ParseFile(
        $validatorPath,
        [ref]$tokens,
        [ref]$errors) | Out-Null
    if ($errors.Count -ne 0) {
        throw ($errors | ForEach-Object { $_.Message } | Out-String)
    }
}

Invoke-Case -Name 'PDFium validator parses in Windows PowerShell' -Action {
    $tokens = $null
    $errors = $null
    [Management.Automation.Language.Parser]::ParseFile(
        $pdfiumValidatorPath,
        [ref]$tokens,
        [ref]$errors) | Out-Null
    if ($errors.Count -ne 0) {
        throw ($errors | ForEach-Object { $_.Message } | Out-String)
    }
}

Invoke-Case -Name 'Missing payload fails closed and still writes structured evidence' -Action {
    $root = Join-Path ([IO.Path]::GetTempPath()) ('GraphReader-CleanMachineHarness-' + [Guid]::NewGuid().ToString('N'))
    $output = Join-Path $root 'output'
    $provenancePath = Join-Path $root 'vm-provenance.json'
    try {
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        $hash = ('a' * 64)
        $provenance = [ordered]@{
            schema = 'graphreader.clean-windows-vm-provenance.v1'
            isoSha256 = ('b' * 64)
            isoSha256Verified = $true
            environmentKind = 'fresh-windows-evaluation-vm'
            networkMode = 'none'
            freshInstall = $true
            expectedCommit = ('c' * 40)
            expectedVersion = '0.0.21'
            expectedExecutableSha256 = $hash
            expectedOpenCvSha256 = $hash
        }
        Write-Utf8NoBom -Path $provenancePath -Content (($provenance | ConvertTo-Json -Depth 5) + [Environment]::NewLine)

        $previousErrorPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $validatorPath `
                -PayloadRoot (Join-Path $root 'missing-payload') `
                -OutputRoot $output `
                -VmProvenancePath $provenancePath `
                -ExpectedCommit ('c' * 40) `
                -ExpectedVersion '0.0.21' `
                -ExpectedExecutableSha256 $hash `
                -ExpectedOpenCvSha256 $hash 2>&1 | Out-Null
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorPreference
        }

        if ($exitCode -eq 0) { throw 'Missing payload unexpectedly passed.' }
        $evidencePath = Join-Path $output 'opencv-clean-machine.json'
        if (-not (Test-Path -LiteralPath $evidencePath -PathType Leaf)) {
            throw 'Fail-closed run did not write evidence.'
        }
        $evidence = Get-Content -LiteralPath $evidencePath -Raw | ConvertFrom-Json
        if ([string]$evidence.schema -cne 'graphreader.opencv-clean-machine-load.v1' -or
            [string]$evidence.status -cne 'fail' -or
            @($evidence.failures).Count -eq 0) {
            throw 'Failure evidence does not preserve the expected schema and blockers.'
        }
    }
    finally {
        if (Test-Path -LiteralPath $root) {
            Remove-Item -LiteralPath $root -Recurse -Force
        }
    }
}

Invoke-Case -Name 'Validator keeps release promotion outside the guest harness' -Action {
    $source = Get-Content -LiteralPath $validatorPath -Raw
    $readme = Get-Content -LiteralPath $readmePath -Raw
    foreach ($required in @(
            'LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS',
            '--portable-smoke',
            'graphreader.opencv-clean-machine-load.v1',
            'networkAdaptersUp',
            'developerToolsOnPath')) {
        if ($source.IndexOf($required, [StringComparison]::Ordinal) -lt 0) {
            throw "Validator is missing required boundary '$required'."
        }
    }
    if ($source.IndexOf('releaseApproved = $true', [StringComparison]::Ordinal) -ge 0 -or
        $readme.IndexOf('cannot approve', [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw 'Guest validation can cross or obscures the public release boundary.'
    }
}

Invoke-Case -Name 'Missing PDFium payload fails closed and writes structured evidence' -Action {
    $root = Join-Path ([IO.Path]::GetTempPath()) ('GraphReader-PdfiumCleanMachineHarness-' + [Guid]::NewGuid().ToString('N'))
    $output = Join-Path $root 'output'
    $provenancePath = Join-Path $root 'vm-provenance.json'
    try {
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        Write-Utf8NoBom -Path $provenancePath -Content '{}'
        $hash = ('a' * 64)
        $previousErrorPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $pdfiumValidatorPath `
                -PayloadRoot (Join-Path $root 'missing-payload') `
                -OutputRoot $output `
                -VmProvenancePath $provenancePath `
                -ExpectedCommit ('c' * 40) `
                -ExpectedVersion '0.0.21' `
                -ExpectedExecutableSha256 $hash `
                -ExpectedApplicationDllSha256 $hash `
                -ExpectedRendererSha256 $hash 2>&1 | Out-Null
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorPreference
        }

        if ($exitCode -eq 0) { throw 'Missing PDFium payload unexpectedly passed.' }
        $evidencePath = Join-Path $output 'pdfium-clean-machine.json'
        if (-not (Test-Path -LiteralPath $evidencePath -PathType Leaf)) {
            throw 'Fail-closed PDFium run did not write evidence.'
        }
        $evidence = Get-Content -LiteralPath $evidencePath -Raw | ConvertFrom-Json
        if ([string]$evidence.schema -cne 'graphreader.pdfium-clean-machine-load.v1' -or
            [string]$evidence.status -cne 'fail' -or
            @($evidence.failures).Count -eq 0) {
            throw 'PDFium failure evidence does not preserve its schema and blockers.'
        }
    }
    finally {
        if (Test-Path -LiteralPath $root) {
            Remove-Item -LiteralPath $root -Recurse -Force
        }
    }
}

Invoke-Case -Name 'PDFium validator keeps release promotion outside the guest harness' -Action {
    $source = Get-Content -LiteralPath $pdfiumValidatorPath -Raw
    foreach ($required in @(
            'graphreader.pdfium-clean-machine-load.v1',
            'expectedPdfiumRendererSha256',
            'controlled-synthetic-fixture',
            '--require-packaged-pdfium',
            'networkAdaptersUp',
            'developerToolsOnPath')) {
        if ($source.IndexOf($required, [StringComparison]::Ordinal) -lt 0) {
            throw "PDFium validator is missing required boundary '$required'."
        }
    }
    if ($source.IndexOf('releaseApproved = $true', [StringComparison]::Ordinal) -ge 0) {
        throw 'PDFium guest validation can cross the public release boundary.'
    }
}

Write-Host "Clean-machine validation tests: $passed passed, $failed failed"
if ($failed -ne 0) {
    exit 1
}
