// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang
// Derived from Pillow 12.3.0 src/libImaging/Resample.c at
// bb1d8e8ab8d29048624d96e3ee53cecf7c13d13d. See
// LICENSES/Pillow-12.3.0-HPND.txt.

namespace GraphReader.Ocr;

/// <summary>
/// Reproduces Pillow 12.3.0's 8-bit BILINEAR resize path. The reviewed OCR
/// payloads were trained with that fixed-point, horizontal-then-vertical
/// implementation, so a mathematically similar floating-point sampler is not
/// byte-equivalent at the model boundary.
/// </summary>
internal static class PillowBilinearResizer
{
    private const int PrecisionBits = 22;
    private const int Precision = 1 << PrecisionBits;
    private const int Rounding = 1 << (PrecisionBits - 1);

    public static byte[] Resize(
        ReadOnlySpan<byte> source,
        int sourceWidth,
        int sourceHeight,
        int destinationWidth,
        int destinationHeight)
    {
        if (sourceWidth <= 0 || sourceHeight <= 0 ||
            destinationWidth <= 0 || destinationHeight <= 0 ||
            source.Length != checked(sourceWidth * sourceHeight))
        {
            throw new ArgumentOutOfRangeException(nameof(source));
        }

        if (sourceWidth == destinationWidth && sourceHeight == destinationHeight)
        {
            return source.ToArray();
        }

        byte[] horizontal;
        int horizontalWidth;
        if (sourceWidth == destinationWidth)
        {
            horizontal = source.ToArray();
            horizontalWidth = sourceWidth;
        }
        else
        {
            horizontalWidth = destinationWidth;
            horizontal = ResizeHorizontal(
                source,
                sourceWidth,
                sourceHeight,
                destinationWidth);
        }

        return sourceHeight == destinationHeight
            ? horizontal
            : ResizeVertical(
                horizontal,
                horizontalWidth,
                sourceHeight,
                destinationHeight);
    }

    private static byte[] ResizeHorizontal(
        ReadOnlySpan<byte> source,
        int sourceWidth,
        int sourceHeight,
        int destinationWidth)
    {
        Coefficients[] coefficients = PrecomputeCoefficients(sourceWidth, destinationWidth);
        var destination = new byte[checked(destinationWidth * sourceHeight)];
        for (var y = 0; y < sourceHeight; y++)
        {
            for (var x = 0; x < destinationWidth; x++)
            {
                Coefficients kernel = coefficients[x];
                long accumulator = Rounding;
                for (var index = 0; index < kernel.Values.Length; index++)
                {
                    accumulator += source[(y * sourceWidth) + kernel.Start + index] *
                        (long)kernel.Values[index];
                }

                destination[(y * destinationWidth) + x] = Clip(accumulator);
            }
        }

        return destination;
    }

    private static byte[] ResizeVertical(
        ReadOnlySpan<byte> source,
        int width,
        int sourceHeight,
        int destinationHeight)
    {
        Coefficients[] coefficients = PrecomputeCoefficients(sourceHeight, destinationHeight);
        var destination = new byte[checked(width * destinationHeight)];
        for (var y = 0; y < destinationHeight; y++)
        {
            Coefficients kernel = coefficients[y];
            for (var x = 0; x < width; x++)
            {
                long accumulator = Rounding;
                for (var index = 0; index < kernel.Values.Length; index++)
                {
                    accumulator += source[((kernel.Start + index) * width) + x] *
                        (long)kernel.Values[index];
                }

                destination[(y * width) + x] = Clip(accumulator);
            }
        }

        return destination;
    }

    private static Coefficients[] PrecomputeCoefficients(int sourceSize, int destinationSize)
    {
        double scale = sourceSize / (double)destinationSize;
        double filterScale = Math.Max(scale, 1d);
        double support = filterScale;
        var result = new Coefficients[destinationSize];
        for (var destination = 0; destination < destinationSize; destination++)
        {
            double center = (destination + 0.5d) * scale;
            int minimum = Math.Max(0, checked((int)(center - support + 0.5d)));
            int maximum = Math.Min(sourceSize, checked((int)(center + support + 0.5d)));
            int count = maximum - minimum;
            var weights = new double[count];
            double total = 0;
            for (var index = 0; index < count; index++)
            {
                double distance = Math.Abs((index + minimum - center + 0.5d) / filterScale);
                double weight = distance < 1d ? 1d - distance : 0d;
                weights[index] = weight;
                total += weight;
            }

            if (total == 0)
            {
                throw new InvalidOperationException("Pillow bilinear coefficient normalization produced zero weight.");
            }

            var fixedPoint = new int[count];
            for (var index = 0; index < count; index++)
            {
                double normalized = weights[index] / total;
                fixedPoint[index] = checked((int)(0.5d + (normalized * Precision)));
            }

            result[destination] = new Coefficients(minimum, fixedPoint);
        }

        return result;
    }

    private static byte Clip(long accumulator) =>
        (byte)Math.Clamp(accumulator >> PrecisionBits, byte.MinValue, byte.MaxValue);

    private sealed record Coefficients(int Start, int[] Values);
}
