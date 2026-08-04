# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param([string]$EvidenceRoot)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'OpenCvSourceAudit.Common.ps1')

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) {
    $EvidenceRoot = Join-Path $projectRoot 'artifacts\goal19-opencv-source\evidence'
}

$errors = @(Get-OpenCvSourceAuditEvidenceErrors -EvidenceRoot ([IO.Path]::GetFullPath($EvidenceRoot)) -LockPath (Join-Path $PSScriptRoot 'source-lock.json'))
if ($errors.Count -gt 0) {
    Write-Error ("OpenCV source audit evidence: BLOCKED`n" + ($errors -join [Environment]::NewLine))
    exit 1
}

Write-Host 'OpenCV source audit evidence: PASS'
