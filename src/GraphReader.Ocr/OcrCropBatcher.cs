// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Buffers.Binary;
using System.Security.Cryptography;

namespace GraphReader.Ocr;

public sealed record OcrCropBatcherOptions
{
    public int TargetWidth { get; init; } = 128;

    public int TargetHeight { get; init; } = 32;

    public int BatchSize { get; init; } = 16;

    public double PaddingPixels { get; init; } = 1;
}

public static class OcrCropBatcher
{
    public static IReadOnlyList<IReadOnlyList<OcrCrop>> CreateBatches(
        OcrImage image,
        IReadOnlyList<OcrDetectedRegion> regions,
        OcrCropBatcherOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(regions);
        options ??= new OcrCropBatcherOptions();
        Validate(image, options);

        var crops = new List<OcrCrop>(regions.Count);
        foreach (var region in regions.OrderBy(static item => item.RegionId, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            crops.Add(CreateCrop(image, region, options, cancellationToken));
        }

        var batches = new List<IReadOnlyList<OcrCrop>>();
        for (var index = 0; index < crops.Count; index += options.BatchSize)
        {
            batches.Add(Array.AsReadOnly(crops.Skip(index).Take(options.BatchSize).ToArray()));
        }

        return OcrCollections.Freeze(batches);
    }

    private static OcrCrop CreateCrop(
        OcrImage image,
        OcrDetectedRegion region,
        OcrCropBatcherOptions options,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(region);
        if (region.CoordinateSpace != OcrContract.CoordinateSpace)
        {
            throw new ArgumentException("OCR regions must use original_pixels.", nameof(region));
        }

        var bounds = region.Polygon.Bounds;
        var padded = new OcrRectangle(
            bounds.X - options.PaddingPixels,
            bounds.Y - options.PaddingPixels,
            bounds.Width + (2 * options.PaddingPixels),
            bounds.Height + (2 * options.PaddingPixels));
        var output = new float[checked(options.TargetWidth * options.TargetHeight)];
        var orientation = GraphTextRoleClassifier.GetOrientation(region.OrientationDegrees);
        for (var targetY = 0; targetY < options.TargetHeight; targetY++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var v = (targetY + 0.5) / options.TargetHeight;
            for (var targetX = 0; targetX < options.TargetWidth; targetX++)
            {
                var u = (targetX + 0.5) / options.TargetWidth;
                var original = MapSample(padded, u, v, orientation);
                var source = image.OriginalToImage.MapFromOriginal(original);
                output[(targetY * options.TargetWidth) + targetX] = Sample(image, source.X, source.Y);
            }
        }

        return new OcrCrop(
            region.RegionId,
            image.SourceImage,
            options.TargetWidth,
            options.TargetHeight,
            output,
            HashCrop(output, image.SourceImage, options.TargetWidth, options.TargetHeight),
            region.Polygon);
    }

    private static OcrPoint MapSample(
        OcrRectangle bounds,
        double u,
        double v,
        OcrOrientation orientation) =>
        orientation switch
        {
            OcrOrientation.RotatedClockwise =>
                new OcrPoint(bounds.Left + (v * bounds.Width), bounds.Top + ((1 - u) * bounds.Height)),
            OcrOrientation.RotatedCounterClockwise =>
                new OcrPoint(bounds.Left + ((1 - v) * bounds.Width), bounds.Top + (u * bounds.Height)),
            _ => new OcrPoint(bounds.Left + (u * bounds.Width), bounds.Top + (v * bounds.Height)),
        };

    private static float Sample(OcrImage image, double x, double y)
    {
        var sourceX = Math.Clamp((int)Math.Round(x), 0, image.Width - 1);
        var sourceY = Math.Clamp((int)Math.Round(y), 0, image.Height - 1);
        return image.Pixels.Span[(sourceY * image.Stride) + sourceX] / 255f;
    }

    private static string HashCrop(
        ReadOnlySpan<float> values,
        OcrSourceImage source,
        int width,
        int height)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> integer = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(integer, width);
        hash.AppendData(integer);
        BinaryPrimitives.WriteInt32LittleEndian(integer, height);
        hash.AppendData(integer);
        BinaryPrimitives.WriteInt32LittleEndian(integer, (int)source);
        hash.AppendData(integer);
        foreach (var value in values)
        {
            BinaryPrimitives.WriteSingleLittleEndian(integer, value);
            hash.AppendData(integer);
        }

        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }

    private static void Validate(OcrImage image, OcrCropBatcherOptions options)
    {
        if (image.Width <= 0 || image.Height <= 0 || image.Stride < image.Width ||
            image.Pixels.Length < checked(image.Stride * image.Height))
        {
            throw new ArgumentException("OCR image dimensions, stride, or pixel buffer are invalid.", nameof(image));
        }

        if (!image.OriginalToImage.IsInvertible)
        {
            throw new ArgumentException("OCR image transform must be finite and invertible.", nameof(image));
        }

        if (options.TargetWidth <= 0 || options.TargetHeight <= 0 || options.BatchSize <= 0 ||
            !double.IsFinite(options.PaddingPixels) || options.PaddingPixels < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options));
        }
    }
}
