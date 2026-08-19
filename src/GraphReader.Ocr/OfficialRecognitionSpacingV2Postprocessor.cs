// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace GraphReader.Ocr;

/// <summary>
/// Byte-bound port of the public-passing official PP-OCRv5 image-spacing V2 P2
/// rule. It may insert source-evidenced spaces and change one isolated
/// lowercase l to capital I only when the immutable crop supplies the frozen
/// top-and-bottom serif evidence. It never creates a text region or marker.
/// </summary>
public static class OfficialRecognitionSpacingV2Postprocessor
{
    public const string Revision = "official-ppocrv5-image-spacing-v2-p2";
    public const int MinimumGapPixels = 4;
    public const double MinimumGapToInkHeightRatio = 0.25;
    public const int MinimumSourceGroups = 3;
    public const double ForegroundContrastFraction = 0.30;
    public const double MinimumForegroundContrast = 10.0;
    public const double CapitalIMinimumWidthHeightRatio = 0.25;
    public const double CapitalIMinimumTopCoverage = 0.75;
    public const double CapitalIMinimumBottomCoverage = 0.75;
    public const int MaximumCharacterCount = 128;
    public const int MaximumSourceGroups = 32;

    public static string Restore(OcrV8SourceCrop crop, string rawPrediction)
    {
        ValidateCrop(crop);
        ArgumentNullException.ThrowIfNull(rawPrediction);
        Rune[] characters = rawPrediction.EnumerateRunes().ToArray();
        if (characters.Length < 2 || characters.Length > MaximumCharacterCount ||
            characters.Any(static value => Rune.IsWhiteSpace(value)))
        {
            return rawPrediction;
        }

        GroupFeature[] features = SourceGroupFeatures(crop);
        if (features.Length > MaximumSourceGroups)
        {
            return rawPrediction;
        }

        int[] counts = PartitionCharacterCounts(
            characters.Length,
            features.Select(static value => value.Width).ToArray());
        if (counts.Length == 0)
        {
            return rawPrediction;
        }

        var chunks = new string[counts.Length];
        var offset = 0;
        for (var index = 0; index < counts.Length; index++)
        {
            string chunk = string.Concat(
                characters.AsSpan(offset, counts[index]).ToArray()
                    .Select(static value => value.ToString()));
            offset += counts[index];
            GroupFeature feature = features[index];
            if (string.Equals(chunk, "l", StringComparison.Ordinal) &&
                feature.Width / (double)Math.Max(1, feature.Height) >= CapitalIMinimumWidthHeightRatio &&
                feature.TopCoverage >= CapitalIMinimumTopCoverage &&
                feature.BottomCoverage >= CapitalIMinimumBottomCoverage)
            {
                chunk = "I";
            }

            chunks[index] = chunk;
        }

        return string.Join(' ', chunks);
    }

    internal static string ConfigurationFingerprint()
    {
        string material = string.Join('|',
        [
            Revision,
            MinimumGapPixels.ToString(CultureInfo.InvariantCulture),
            MinimumGapToInkHeightRatio.ToString("R", CultureInfo.InvariantCulture),
            MinimumSourceGroups.ToString(CultureInfo.InvariantCulture),
            ForegroundContrastFraction.ToString("R", CultureInfo.InvariantCulture),
            MinimumForegroundContrast.ToString("R", CultureInfo.InvariantCulture),
            CapitalIMinimumWidthHeightRatio.ToString("R", CultureInfo.InvariantCulture),
            CapitalIMinimumTopCoverage.ToString("R", CultureInfo.InvariantCulture),
            CapitalIMinimumBottomCoverage.ToString("R", CultureInfo.InvariantCulture),
            MaximumCharacterCount.ToString(CultureInfo.InvariantCulture),
            MaximumSourceGroups.ToString(CultureInfo.InvariantCulture),
        ]);
        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(material)));
    }

    private static GroupFeature[] SourceGroupFeatures(OcrV8SourceCrop crop)
    {
        ReadOnlySpan<byte> pixels = crop.Pixels.Span;
        var edge = new byte[checked((crop.Width * 2) + (crop.Height * 2))];
        var edgeIndex = 0;
        pixels[..crop.Width].CopyTo(edge.AsSpan(edgeIndex, crop.Width));
        edgeIndex += crop.Width;
        pixels.Slice((crop.Height - 1) * crop.Width, crop.Width)
            .CopyTo(edge.AsSpan(edgeIndex, crop.Width));
        edgeIndex += crop.Width;
        for (var y = 0; y < crop.Height; y++)
        {
            edge[edgeIndex++] = pixels[y * crop.Width];
        }
        for (var y = 0; y < crop.Height; y++)
        {
            edge[edgeIndex++] = pixels[(y * crop.Width) + crop.Width - 1];
        }

        double background = Percentile(edge, 50.0);
        double darkest = Percentile(pixels.ToArray(), 1.0);
        double contrast = Math.Max(0.0, background - darkest);
        double threshold = Math.Max(
            MinimumForegroundContrast,
            contrast * ForegroundContrastFraction);
        double foregroundMaximum = background - threshold;

        var foreground = new bool[pixels.Length];
        int top = crop.Height;
        int left = crop.Width;
        var bottom = -1;
        var right = -1;
        for (var y = 0; y < crop.Height; y++)
        {
            for (var x = 0; x < crop.Width; x++)
            {
                int index = (y * crop.Width) + x;
                if (pixels[index] > foregroundMaximum)
                {
                    continue;
                }

                foreground[index] = true;
                top = Math.Min(top, y);
                left = Math.Min(left, x);
                bottom = Math.Max(bottom, y);
                right = Math.Max(right, x);
            }
        }

        if (bottom < top || right < left)
        {
            return [];
        }

        var activeColumns = new List<int>();
        for (var x = left; x <= right; x++)
        {
            bool active = false;
            for (var y = top; y <= bottom && !active; y++)
            {
                active = foreground[(y * crop.Width) + x];
            }
            if (active)
            {
                activeColumns.Add(x);
            }
        }

        if (activeColumns.Count == 0)
        {
            return [];
        }

        int inkHeight = bottom - top + 1;
        int largeGap = Math.Max(
            MinimumGapPixels,
            checked((int)Math.Ceiling(inkHeight * MinimumGapToInkHeightRatio)));
        var starts = new List<int> { activeColumns[0] };
        var ends = new List<int>();
        for (var index = 1; index < activeColumns.Count; index++)
        {
            if (activeColumns[index] - activeColumns[index - 1] - 1 >= largeGap)
            {
                ends.Add(activeColumns[index - 1]);
                starts.Add(activeColumns[index]);
            }
        }
        ends.Add(activeColumns[^1]);

        var result = new GroupFeature[starts.Count];
        for (var groupIndex = 0; groupIndex < starts.Count; groupIndex++)
        {
            int start = starts[groupIndex];
            int end = ends[groupIndex];
            int groupTop = crop.Height;
            var groupBottom = -1;
            for (var y = top; y <= bottom; y++)
            {
                for (var x = start; x <= end; x++)
                {
                    if (!foreground[(y * crop.Width) + x])
                    {
                        continue;
                    }
                    groupTop = Math.Min(groupTop, y);
                    groupBottom = Math.Max(groupBottom, y);
                }
            }

            int width = end - start + 1;
            int height = groupBottom - groupTop + 1;
            var topInk = 0;
            var bottomInk = 0;
            for (var x = start; x <= end; x++)
            {
                if (foreground[(groupTop * crop.Width) + x])
                {
                    topInk++;
                }
                if (foreground[(groupBottom * crop.Width) + x])
                {
                    bottomInk++;
                }
            }
            result[groupIndex] = new(
                width,
                height,
                topInk / (double)width,
                bottomInk / (double)width);
        }

        return result;
    }

    private static int[] PartitionCharacterCounts(int characterCount, int[] widths)
    {
        int groupCount = widths.Length;
        if (groupCount < MinimumSourceGroups || groupCount > characterCount)
        {
            return [];
        }

        double totalWidth = widths.Sum(static value => (double)value);
        var cumulative = new double[groupCount - 1];
        var running = 0.0;
        for (var index = 0; index < cumulative.Length; index++)
        {
            running += widths[index];
            cumulative[index] = running / totalWidth;
        }

        double bestCost = double.PositiveInfinity;
        int[]? bestCounts = null;
        var boundaries = new int[groupCount - 1];
        Search(0, 1);
        return bestCounts ?? [];

        void Search(int depth, int minimumBoundary)
        {
            if (depth == boundaries.Length)
            {
                var counts = new int[groupCount];
                var prior = 0;
                var cost = 0.0;
                for (var index = 0; index < boundaries.Length; index++)
                {
                    int boundary = boundaries[index];
                    counts[index] = boundary - prior;
                    prior = boundary;
                    double difference = (boundary / (double)characterCount) - cumulative[index];
                    cost += difference * difference;
                }
                counts[^1] = characterCount - prior;
                if (cost < bestCost ||
                    (cost.Equals(bestCost) && LexicographicallyLess(counts, bestCounts)))
                {
                    bestCost = cost;
                    bestCounts = counts;
                }
                return;
            }

            int boundariesRemaining = boundaries.Length - depth - 1;
            int maximumBoundary = characterCount - boundariesRemaining - 1;
            for (int boundary = minimumBoundary; boundary <= maximumBoundary; boundary++)
            {
                boundaries[depth] = boundary;
                Search(depth + 1, boundary + 1);
            }
        }
    }

    private static bool LexicographicallyLess(int[] candidate, int[]? current)
    {
        if (current is null)
        {
            return true;
        }
        for (var index = 0; index < candidate.Length; index++)
        {
            if (candidate[index] != current[index])
            {
                return candidate[index] < current[index];
            }
        }
        return false;
    }

    private static double Percentile(byte[] values, double percentile)
    {
        Array.Sort(values);
        double position = (values.Length - 1) * (percentile / 100.0);
        int lower = checked((int)Math.Floor(position));
        int upper = checked((int)Math.Ceiling(position));
        if (lower == upper)
        {
            return values[lower];
        }
        double fraction = position - lower;
        return values[lower] + ((values[upper] - values[lower]) * fraction);
    }

    private static void ValidateCrop(OcrV8SourceCrop crop)
    {
        ArgumentNullException.ThrowIfNull(crop);
        if (crop.Width <= 0 || crop.Height <= 0 ||
            crop.Pixels.Length != checked(crop.Width * crop.Height) ||
            string.IsNullOrWhiteSpace(crop.PixelSha256) ||
            crop.PixelSha256.Length != 64 ||
            crop.PixelSha256.Any(static value => !Uri.IsHexDigit(value)) ||
            !string.Equals(
                crop.PixelSha256,
                Convert.ToHexStringLower(SHA256.HashData(crop.Pixels.Span)),
                StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                "Official recognition spacing V2 requires a checksum-matched Gray8 source crop.",
                nameof(crop));
        }
    }

    private readonly record struct GroupFeature(
        int Width,
        int Height,
        double TopCoverage,
        double BottomCoverage);
}
