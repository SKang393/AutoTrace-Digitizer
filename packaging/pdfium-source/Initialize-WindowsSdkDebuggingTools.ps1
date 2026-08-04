# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param([string]$SetupPath, [string]$ArtifactRoot)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) { $ArtifactRoot = Join-Path $projectRoot 'artifacts\pdfium-source' }
$ArtifactRoot = [IO.Path]::GetFullPath($ArtifactRoot)
$lock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'source-lock.json') -Raw | ConvertFrom-Json
$debuggerLock = $lock.toolchain.windowsSdkDebuggers
$layoutRoot = Join-Path $ArtifactRoot 'windows-sdk-layout'
$extractRoot = Join-Path $ArtifactRoot 'windows-sdk-debuggers-x64'
$msi = Join-Path $layoutRoot 'Installers\X64 Debuggers And Tools-x64_en-us.msi'
$debuggerRoot = Join-Path $extractRoot 'Windows Kits\10\Debuggers\x64'

function Assert-Hash([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Pinned Windows SDK debugger input is missing: $Path" }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) { throw "Windows SDK debugger hash mismatch for $Path`: $actual" }
}

if (-not (Test-Path -LiteralPath $msi -PathType Leaf)) {
    if ([string]::IsNullOrWhiteSpace($SetupPath)) {
        $candidate = Get-ChildItem -LiteralPath 'C:\ProgramData\Package Cache' -Filter winsdksetup.exe -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.VersionInfo.ProductVersion -eq [string]$debuggerLock.installerVersion } |
            Select-Object -First 1
        if ($null -eq $candidate) { throw "Windows SDK setup $($debuggerLock.installerVersion) is unavailable." }
        $SetupPath = $candidate.FullName
    }
    New-Item -ItemType Directory -Path $layoutRoot -Force | Out-Null
    $arguments = @('/layout', ('"' + $layoutRoot + '"'), '/features', [string]$debuggerLock.featureId, '/quiet', '/norestart')
    [void](Start-Process -FilePath $SetupPath -ArgumentList $arguments -NoNewWindow -Wait -PassThru)
    $deadline = [DateTime]::UtcNow.AddMinutes(10)
    do {
        $active = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq 'winsdksetup.exe' -and $_.CommandLine -like "*/layout*$layoutRoot*"
        })
        if ($active.Count -eq 0) { break }
        if ([DateTime]::UtcNow -ge $deadline) { throw 'Windows SDK debugger layout exceeded 10 minutes.' }
        Start-Sleep -Seconds 1
    } while ($true)
}
Assert-Hash $msi ([string]$debuggerLock.x64MsiSha256)

$dbghelp = Join-Path $debuggerRoot 'dbghelp.dll'
if (-not (Test-Path -LiteralPath $dbghelp -PathType Leaf)) {
    New-Item -ItemType Directory -Path $extractRoot -Force | Out-Null
    $log = Join-Path $ArtifactRoot 'windows-sdk-admin-extract.log'
    $arguments = @('/a', ('"' + $msi + '"'), ('TARGETDIR="' + $extractRoot + '"'), '/qn', '/norestart', '/l*v', ('"' + $log + '"'))
    $process = Start-Process -FilePath msiexec.exe -ArgumentList $arguments -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "Windows SDK debugger administrative extraction failed with exit code $($process.ExitCode)." }
}
Assert-Hash $dbghelp ([string]$debuggerLock.dbghelpSha256)
Assert-Hash (Join-Path $debuggerRoot 'dbgcore.dll') ([string]$debuggerLock.dbgcoreSha256)
Assert-Hash (Join-Path $debuggerRoot 'symsrv.dll') ([string]$debuggerLock.symsrvSha256)
Write-Host "Pinned Windows SDK x64 debugging tools ready: $debuggerRoot"
