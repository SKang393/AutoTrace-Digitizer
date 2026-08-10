// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Buffers.Binary;
using System.Security.Cryptography;

namespace GraphReader.Ocr;

public enum OcrCropResizeMode
{
    PreserveAspectRatioPad,
    Stretch,
}

public sealed record OcrCropBatcherOptions
{
    public int TargetWidth { get; init; } = 128;

    public int TargetHeight { get; init; } = 32;

    public int BatchSize { get; init; } = 16;

    public double PaddingPixels { get; init; } = 1;

    /// <summary>
    /// Controls how a detected text region is mapped into the recognition tensor.
    /// PP-OCR recognition preserves the source aspect ratio and pads the remaining
    /// width instead of stretching every crop to the model's maximum width.
    /// </summary>
    public OcrCropResizeMode ResizeMode { get; init; } =
        OcrCropResizeMode.PreserveAspectRatioPad;

    /// <summary>
    /// Source-space value used for right padding. The PP-OCR normalization
    /// (value - 0.5) * 2 maps the default 0.5 value to tensor-space zero.
    /// </summary>
    public float PaddingValue { get; init; } = 0.5f;
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
        var orientation = GraphTextRoleClassifier.GetOrientation(region.OrientationDegrees);
        int contentWidth = ContentWidth(padded, orientation, options);
        var output = Enumerable.Repeat(
                options.PaddingValue,
                checked(options.TargetWidth * options.TargetHeight))
            .ToArray();
        float[]? bgrOutput = image.BgrPixels is null
            ? null
            : Enumerable.Repeat(
                    options.PaddingValue,
                    checked(options.TargetWidth * options.TargetHeight * 3))
                .ToArray();
        for (var targetY = 0; targetY < options.TargetHeight; targetY++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var v = (targetY + 0.5) / options.TargetHeight;
            for (var targetX = 0; targetX < contentWidth; targetX++)
            {
                var u = (targetX + 0.5) / contentWidth;
                var original = MapSample(padded, u, v, orientation);
                var source = image.OriginalToImage.MapFromOriginal(original);
                output[(targetY * options.TargetWidth) + targetX] = Sample(image, source.X, source.Y);
                if (bgrOutput is not null)
                {
                    int targetOffset = checked(((targetY * options.TargetWidth) + targetX) * 3);
                    for (int channel = 0; channel < 3; channel++)
                    {
                        bgrOutput[targetOffset + channel] = SampleBgr(image, source.X, source.Y, channel);
                    }
                }
            }
        }

        var bgrPixels = bgrOutput is null
            ? null
            : new OcrBgrFloatPixels(checked(options.TargetWidth * 3), bgrOutput);

        return new OcrCrop(
            region.RegionId,
            image.SourceImage,
            options.TargetWidth,
            options.TargetHeight,
            output,
            HashCrop(output, bgrOutput, image.SourceImage, options.TargetWidth, options.TargetHeight),
            region.Polygon,
            bgrPixels);
    }

    private static int ContentWidth(
        OcrRectangle bounds,
        OcrOrientation orientation,
        OcrCropBatcherOptions options)
    {
        if (options.ResizeMode == OcrCropResizeMode.Stretch)
        {
            return options.TargetWidth;
        }

        double orientedWidth = orientation is
            OcrOrientation.RotatedClockwise or OcrOrientation.RotatedCounterClockwise
                ? bounds.Height
                : bounds.Width;
        double orientedHeight = orientation is
            OcrOrientation.RotatedClockwise or OcrOrientation.RotatedCounterClockwise
                ? bounds.Width
                : bounds.Height;
        double ratio = orientedWidth / orientedHeight;
        return Math.Clamp(
            checked((int)Math.Ceiling(options.TargetHeight * ratio)),
            1,
            options.TargetWidth);
    }

    private static OcrPoint MapSample(
        OcrRectangle bounds,
        double u,
        double v,
        OcrOrientation orientation) =>
        orientation switch
        {
            OcrOrientation.RotatedClockwise =>
                new OcrPoint(
                    bounds.Left + (v * bounds.Width) - 0.5,
                    bounds.Top + ((1 - u) * bounds.Height) - 0.5),
            OcrOrientation.RotatedCounterClockwise =>
                new OcrPoint(
                    bounds.Left + ((1 - v) * bounds.Width) - 0.5,
                    bounds.Top + (u * bounds.Height) - 0.5),
            _ => new OcrPoint(
                bounds.Left + (u * bounds.Width) - 0.5,
                bounds.Top + (v * bounds.Height) - 0.5),
        };

    private static float Sample(OcrImage image, double x, double y)
    {
        double boundedX = Math.Clamp(x, 0, image.Width - 1d);
        double boundedY = Math.Clamp(y, 0, image.Height - 1d);
        int x0 = (int)Math.Floor(boundedX);
        int y0 = (int)Math.Floor(boundedY);
        int x1 = Math.Min(x0 + 1, image.Width - 1);
        int y1 = Math.Min(y0 + 1, image.Height - 1);
        double xWeight = boundedX - x0;
        double yWeight = boundedY - y0;
        ReadOnlySpan<byte> pixels = image.Pixels.Span;
        double top = (pixels[(y0 * image.Stride) + x0] * (1 - xWeight)) +
            (pixels[(y0 * image.Stride) + x1] * xWeight);
        double bottom = (pixels[(y1 * image.Stride) + x0] * (1 - xWeight)) +
            (pixels[(y1 * image.Stride) + x1] * xWeight);
        return (float)(((top * (1 - yWeight)) + (bottom * yWeight)) / 255d);
    }

    private static float SampleBgr(OcrImage image, double x, double y, int channel)
    {
        OcrBgrBytePixels bgr = image.BgrPixels ??
            throw new InvalidOperationException("BGR sampling requires a BGR24 image plane.");
        double boundedX = Math.Clamp(x, 0, image.Width - 1d);
        double boundedY = Math.Clamp(y, 0, image.Height - 1d);
        int x0 = (int)Math.Floor(boundedX);
        int y0 = (int)Math.Floor(boundedY);
        int x1 = Math.Min(x0 + 1, image.Width - 1);
        int y1 = Math.Min(y0 + 1, image.Height - 1);
        double xWeight = boundedX - x0;
        double yWeight = boundedY - y0;
        ReadOnlySpan<byte> pixels = bgr.Pixels.Span;
        double top = (pixels[(y0 * bgr.Stride) + (x0 * 3) + channel] * (1 - xWeight)) +
            (pixels[(y0 * bgr.Stride) + (x1 * 3) + channel] * xWeight);
        double bottom = (pixels[(y1 * bgr.Stride) + (x0 * 3) + channel] * (1 - xWeight)) +
            (pixels[(y1 * bgr.Stride) + (x1 * 3) + channel] * xWeight);
        return (float)(((top * (1 - yWeight)) + (bottom * yWeight)) / 255d);
    }

    private static string HashCrop(
        ReadOnlySpan<float> values,
        float[]? bgrValues,
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

        BinaryPrimitives.WriteInt32LittleEndian(integer, bgrValues is null ? 0 : 3);
        hash.AppendData(integer);
        if (bgrValues is not null)
        {
            foreach (float value in bgrValues)
            {
                BinaryPrimitives.WriteSingleLittleEndian(integer, value);
                hash.AppendData(integer);
            }
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
            !double.IsFinite(options.PaddingPixels) || options.PaddingPixels < 0 ||
            !Enum.IsDefined(options.ResizeMode) ||
            !float.IsFinite(options.PaddingValue) || options.PaddingValue is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options));
        }

        if (image.BgrPixels is { } bgr &&
            (bgr.Stride < checked(image.Width * 3) ||
             bgr.Pixels.Length != checked(bgr.Stride * image.Height)))
        {
            throw new ArgumentException("OCR BGR24 dimensions, stride, or pixel buffer are invalid.", nameof(image));
        }
    }
}
