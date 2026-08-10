# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

[CmdletBinding()]
param(
    [string]$OutputPath = (Join-Path $PSScriptRoot 'GraphAutoReader.ico')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

function New-RoundedRectanglePath {
    param(
        [float]$X,
        [float]$Y,
        [float]$Width,
        [float]$Height,
        [float]$Radius
    )

    $diameter = $Radius * 2
    $path = [System.Drawing.Drawing2D.GraphicsPath]::new()
    $path.AddArc($X, $Y, $diameter, $diameter, 180, 90)
    $path.AddArc($X + $Width - $diameter, $Y, $diameter, $diameter, 270, 90)
    $path.AddArc($X + $Width - $diameter, $Y + $Height - $diameter, $diameter, $diameter, 0, 90)
    $path.AddArc($X, $Y + $Height - $diameter, $diameter, $diameter, 90, 90)
    $path.CloseFigure()
    return $path
}

function New-IconFrame {
    param([int]$Size)

    $bitmap = [System.Drawing.Bitmap]::new(
        $Size,
        $Size,
        [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
        $graphics.Clear([System.Drawing.Color]::Transparent)
        $graphics.ScaleTransform($Size / 64.0, $Size / 64.0)

        $background = New-RoundedRectanglePath -X 2 -Y 2 -Width 60 -Height 60 -Radius 13
        $primaryBrush = [System.Drawing.SolidBrush]::new(
            [System.Drawing.Color]::FromArgb(255, 54, 89, 227))
        $whitePen = [System.Drawing.Pen]::new(
            [System.Drawing.Color]::FromArgb(245, 255, 255, 255),
            3.4)
        $tracePen = [System.Drawing.Pen]::new(
            [System.Drawing.Color]::FromArgb(255, 225, 231, 255),
            3.0)
        $rowPen = [System.Drawing.Pen]::new(
            [System.Drawing.Color]::FromArgb(225, 255, 255, 255),
            2.4)
        $anchorBrush = [System.Drawing.SolidBrush]::new(
            [System.Drawing.Color]::FromArgb(255, 255, 255, 255))
        $evidenceBrush = [System.Drawing.SolidBrush]::new(
            [System.Drawing.Color]::FromArgb(255, 90, 215, 190))
        try {
            $whitePen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
            $whitePen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
            $whitePen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
            $tracePen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
            $tracePen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
            $tracePen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
            $rowPen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
            $rowPen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round

            $graphics.FillPath($primaryBrush, $background)
            $graphics.DrawLine($whitePen, 14, 12, 14, 48)
            $graphics.DrawLine($whitePen, 14, 48, 48, 48)
            $graphics.DrawLines($tracePen, [System.Drawing.PointF[]]@(
                [System.Drawing.PointF]::new(19, 41),
                [System.Drawing.PointF]::new(28, 33),
                [System.Drawing.PointF]::new(37, 36),
                [System.Drawing.PointF]::new(46, 22)))

            foreach ($point in @(
                [System.Drawing.PointF]::new(19, 41),
                [System.Drawing.PointF]::new(28, 33),
                [System.Drawing.PointF]::new(37, 36))) {
                $graphics.FillEllipse($anchorBrush, $point.X - 2.7, $point.Y - 2.7, 5.4, 5.4)
            }
            $graphics.FillEllipse($evidenceBrush, 42.2, 18.2, 7.6, 7.6)
            $graphics.DrawLine($rowPen, 46, 27, 46, 36)
            $graphics.DrawLine($rowPen, 46, 36, 54, 36)
            $graphics.DrawLine($rowPen, 45, 41, 54, 41)
            $graphics.DrawLine($rowPen, 45, 46, 54, 46)
        }
        finally {
            $background.Dispose()
            $primaryBrush.Dispose()
            $whitePen.Dispose()
            $tracePen.Dispose()
            $rowPen.Dispose()
            $anchorBrush.Dispose()
            $evidenceBrush.Dispose()
        }

        $stream = [System.IO.MemoryStream]::new()
        $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
        return $stream.ToArray()
    }
    finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

$sizes = @(16, 20, 24, 32, 40, 48, 64, 128, 256)
$frames = @($sizes | ForEach-Object {
    [pscustomobject]@{
        Size = $_
        Bytes = New-IconFrame -Size $_
    }
})

$outputDirectory = Split-Path -Parent $OutputPath
if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
}

$file = [System.IO.File]::Open($OutputPath, [System.IO.FileMode]::Create)
$writer = [System.IO.BinaryWriter]::new($file)
try {
    $writer.Write([uint16]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]$frames.Count)

    $offset = 6 + (16 * $frames.Count)
    foreach ($frame in $frames) {
        [byte]$dimension = if ($frame.Size -eq 256) { 0 } else { $frame.Size }
        $writer.Write($dimension)
        $writer.Write($dimension)
        $writer.Write([byte]0)
        $writer.Write([byte]0)
        $writer.Write([uint16]1)
        $writer.Write([uint16]32)
        $writer.Write([uint32]$frame.Bytes.Length)
        $writer.Write([uint32]$offset)
        $offset += $frame.Bytes.Length
    }

    foreach ($frame in $frames) {
        $writer.Write([byte[]]$frame.Bytes)
    }
}
finally {
    $writer.Dispose()
    $file.Dispose()
}

Write-Host "Generated $OutputPath with sizes: $($sizes -join ', ')"
