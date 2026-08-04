# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$PolicyPath,
    [string]$SourceRoot,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($PolicyPath)) {
    $PolicyPath = Join-Path $PSScriptRoot 'dependency-review-policy.json'
}
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $PSScriptRoot '..\..\..\artifacts\pdfium-source\sources\pdfium'
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $PSScriptRoot 'third-party-notices.dependency-mapped.txt'
}

$PolicyPath = [IO.Path]::GetFullPath($PolicyPath)
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
$policy = Get-Content -LiteralPath $PolicyPath -Raw | ConvertFrom-Json

if ([string]$policy.overallReviewStatus -ne 'dependency-mapped-not-approved') {
    throw 'PDFium dependency policy must remain dependency-mapped-not-approved.'
}

$builder = New-Object Text.StringBuilder
[void]$builder.AppendLine('REVIEW STATUS: DEPENDENCY-MAPPED')
[void]$builder.AppendLine("PROFILE: $($policy.profileId)")
[void]$builder.AppendLine("PDFIUM REVISION: $($policy.sourceRevision)")
[void]$builder.AppendLine("RUNNER SHA-256: $($policy.evidenceBinding.binarySha256)")
[void]$builder.AppendLine()
[void]$builder.AppendLine('This deterministic bundle contains the exact license and notice files mapped to the retained runner target-dependency closure. Build-only NASM and Windows system API imports are dispositioned in the policy and are not bundled here. This dependency mapping is not independent approval, clean-machine evidence, or release authorization.')
[void]$builder.AppendLine()

$seen = @{}
foreach ($component in @($policy.components)) {
    foreach ($license in @($component.licenses)) {
        if (-not [bool]$license.includeInNotice) {
            continue
        }
        $noticeId = [string]$license.noticeId
        $relativePath = [string]$license.path
        if ($seen.ContainsKey($noticeId)) {
            if ([string]$seen[$noticeId] -ne $relativePath) {
                throw "Notice ID '$noticeId' maps to more than one source path."
            }
            continue
        }
        $seen[$noticeId] = $relativePath

        $sourcePath = [IO.Path]::GetFullPath((Join-Path $SourceRoot $relativePath))
        $sourcePrefix = $SourceRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        if (-not $sourcePath.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Notice source escapes the PDFium source root: $relativePath"
        }
        if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
            throw "Notice source is missing: $sourcePath"
        }
        $actualHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne [string]$license.sha256) {
            throw "Notice source hash mismatch: $relativePath"
        }

        $body = (Get-Content -LiteralPath $sourcePath -Raw) -replace "`r`n", "`n"
        $body = $body.TrimEnd("`r", "`n")
        [void]$builder.AppendLine("===== BEGIN NOTICE: $noticeId =====")
        [void]$builder.AppendLine("SOURCE PATH: $relativePath")
        [void]$builder.AppendLine("SOURCE SHA-256: $actualHash")
        [void]$builder.AppendLine()
        [void]$builder.AppendLine($body)
        [void]$builder.AppendLine("===== END NOTICE: $noticeId =====")
        [void]$builder.AppendLine()
    }
}

$parent = [IO.Path]::GetDirectoryName($OutputPath)
if ([string]::IsNullOrWhiteSpace($parent)) {
    throw 'Output path must have a parent directory.'
}
[IO.Directory]::CreateDirectory($parent) | Out-Null
$text = $builder.ToString() -replace "`r`n", "`n"
[IO.File]::WriteAllText($OutputPath, $text, [Text.UTF8Encoding]::new($false))
Write-Host "PDFium dependency-mapped notice: $OutputPath"
Write-Output ([pscustomobject]@{
    OutputPath = $OutputPath
    Sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    NoticeCount = $seen.Count
    ReviewApproved = $false
})
