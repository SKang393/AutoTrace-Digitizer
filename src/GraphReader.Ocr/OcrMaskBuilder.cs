// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Ocr;

public static class OcrMaskBuilder
{
    public static IReadOnlyList<OcrMask> Build(
        IReadOnlyList<OcrDetectedRegion> regions,
        int? originalWidth,
        int? originalHeight,
        double paddingPixels = 1)
    {
        ArgumentNullException.ThrowIfNull(regions);
        if ((originalWidth.HasValue != originalHeight.HasValue) ||
            originalWidth is <= 0 || originalHeight is <= 0 ||
            paddingPixels < 0 || !double.IsFinite(paddingPixels))
        {
            throw new ArgumentOutOfRangeException(nameof(originalWidth));
        }

        var masks = new List<OcrMask>(regions.Count);
        foreach (var region in regions)
        {
            if (region.CoordinateSpace != OcrContract.CoordinateSpace)
            {
                throw new ArgumentException("OCR mask input must use original_pixels.", nameof(regions));
            }

            var bounds = region.Polygon.Bounds;
            var left = bounds.Left - paddingPixels;
            var top = bounds.Top - paddingPixels;
            var right = bounds.Right + paddingPixels;
            var bottom = bounds.Bottom + paddingPixels;
            if (originalWidth.HasValue && originalHeight.HasValue)
            {
                left = Math.Clamp(left, 0, originalWidth.Value);
                top = Math.Clamp(top, 0, originalHeight.Value);
                right = Math.Clamp(right, 0, originalWidth.Value);
                bottom = Math.Clamp(bottom, 0, originalHeight.Value);
            }

            if (right <= left || bottom <= top)
            {
                continue;
            }

            masks.Add(new OcrMask(
                region.RegionId,
                OcrPolygon.FromRectangle(new OcrRectangle(left, top, right - left, bottom - top)),
                Math.Clamp(region.DetectionConfidence, 0, 1)));
        }

        return OcrCollections.Freeze(masks);
    }
}
