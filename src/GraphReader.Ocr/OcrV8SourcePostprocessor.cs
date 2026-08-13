// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using GraphReader.Inference;

namespace GraphReader.Ocr;

public sealed record OcrV8SourceCrop(
    int Width,
    int Height,
    ReadOnlyMemory<byte> Pixels,
    string PixelSha256,
    OcrPolygon OriginalPolygon);

public sealed record OcrV8AmbiguityResult(
    string Text,
    int ChangedCharacterCount,
    bool ModelExecuted,
    string InputTensorSha256,
    double InferenceMilliseconds);

public sealed record LocalOnnxAmbiguitySourceGroupOptions(ModelIdentity Model)
{
    public string InputName { get; init; } = "glyphs";

    public string OutputName { get; init; } = "logits";

    public string StageVersion { get; init; } = "0.0.21-p2";

    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

    public IReadOnlyList<InferenceProvider>? AllowedProviders { get; init; }

    public bool BypassCache { get; init; }
}

/// <summary>
/// Exact source-byte preprocessing used by the public-passing OCR production
/// composition V8. It restores only whitespace supported by large source gaps
/// and routes the reviewed O/o/l/I ambiguity family through its checksum-bound
/// four-class ONNX model. It never creates text regions or marker evidence.
/// </summary>
public sealed partial class OcrV8SourcePostprocessor
{
    public const string CompositionRevision =
        "graphreader-v10-bounded-zero-consensus-ambiguity-alias-composition-v8";
    public const string SpacingRevision = "official-ppocrv5-conservative-spacing-v3-p1";
    public const string AmbiguityRevision = "graph-ambiguity-source-group-v3";
    public const int SourceHorizontalPadding = 8;
    public const int SourceVerticalPadding = 2;
    public const int AmbiguityImageSize = 32;

    private const int MinimumGapPixels = 5;
    private const double MinimumGapToInkHeightRatio = 0.40;
    private const int MinimumSourceGroups = 2;
    private const double ForegroundContrastFraction = 0.30;
    private const double MinimumForegroundContrast = 10.0;
    private const string CanonicalAmbiguityGlyphs = "OolI";
    private const string ExtendedAmbiguityGlyphs = "OolI!i";

    private readonly InferenceRuntime runtime;
    private readonly LocalOnnxAmbiguitySourceGroupOptions options;
    private readonly string configurationFingerprint;

    public OcrV8SourcePostprocessor(
        InferenceRuntime runtime,
        LocalOnnxAmbiguitySourceGroupOptions options)
    {
        this.runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        ArgumentNullException.ThrowIfNull(options);
        this.options = options with
        {
            AllowedProviders = options.AllowedProviders is null
                ? null
                : Array.AsReadOnly(options.AllowedProviders.ToArray()),
        };
        ValidateOptions(this.options);
        configurationFingerprint = HashStrings(
        [
            CompositionRevision,
            SpacingRevision,
            AmbiguityRevision,
            this.options.Model.Sha256.ToLowerInvariant(),
            SourceHorizontalPadding.ToString(CultureInfo.InvariantCulture),
            SourceVerticalPadding.ToString(CultureInfo.InvariantCulture),
            AmbiguityImageSize.ToString(CultureInfo.InvariantCulture),
            MinimumGapPixels.ToString(CultureInfo.InvariantCulture),
            MinimumGapToInkHeightRatio.ToString("R", CultureInfo.InvariantCulture),
            ForegroundContrastFraction.ToString("R", CultureInfo.InvariantCulture),
            MinimumForegroundContrast.ToString("R", CultureInfo.InvariantCulture),
            CanonicalAmbiguityGlyphs,
            ExtendedAmbiguityGlyphs,
            this.options.InputName,
            this.options.OutputName,
            this.options.StageVersion,
            ProviderFingerprint(this.options.AllowedProviders),
        ]);
    }

    public string ModelId => options.Model.ModelId;

    public string ModelVersion => options.Model.Version;

    public string ModelSha256 => options.Model.Sha256;

    public string ConfigurationFingerprint => configurationFingerprint;

    public static OcrV8SourceCrop ExtractSourceCrop(
        OcrImage image,
        OcrDetectedRegion region)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(region);
        if (image.SourceImage != OcrSourceImage.Original ||
            image.OriginalToImage != OcrFrameTransform.Identity ||
            image.Width <= 0 || image.Height <= 0 || image.Stride < image.Width ||
            image.Pixels.Length < checked(image.Stride * image.Height) ||
            !string.Equals(image.CoordinateSpace, OcrContract.CoordinateSpace, StringComparison.Ordinal) ||
            !string.Equals(region.CoordinateSpace, OcrContract.CoordinateSpace, StringComparison.Ordinal))
        {
            throw new ArgumentException(
                "OCR V8 source processing requires a valid immutable original Gray8 frame in original_pixels.",
                nameof(image));
        }

        OcrRectangle bounds = region.Polygon.Bounds;
        int left = Math.Max(0, checked((int)Math.Floor(bounds.Left)) - SourceHorizontalPadding);
        int top = Math.Max(0, checked((int)Math.Floor(bounds.Top)) - SourceVerticalPadding);
        int right = Math.Min(
            image.Width,
            checked((int)Math.Ceiling(bounds.Right)) + SourceHorizontalPadding);
        int bottom = Math.Min(
            image.Height,
            checked((int)Math.Ceiling(bounds.Bottom)) + SourceVerticalPadding);
        if (right <= left || bottom <= top)
        {
            throw new InvalidDataException("OCR V8 source crop is empty.");
        }

        int width = right - left;
        int height = bottom - top;
        var pixels = new byte[checked(width * height)];
        ReadOnlySpan<byte> source = image.Pixels.Span;
        for (var y = 0; y < height; y++)
        {
            source.Slice(checked(((top + y) * image.Stride) + left), width)
                .CopyTo(pixels.AsSpan(y * width, width));
        }

        return new OcrV8SourceCrop(
            width,
            height,
            pixels,
            Convert.ToHexStringLower(SHA256.HashData(pixels)),
            region.Polygon);
    }

    public static string RestoreConservativeSourceSpaces(
        OcrV8SourceCrop crop,
        string rawPrediction)
    {
        ValidateCrop(crop);
        ArgumentNullException.ThrowIfNull(rawPrediction);
        Rune[] characters = rawPrediction
            .EnumerateRunes()
            .Where(static character => !Rune.IsWhiteSpace(character))
            .ToArray();
        if (characters.Length < 2)
        {
            return rawPrediction;
        }

        int[] widths = SourceGroupWidths(crop);
        int[] counts = PartitionCharacterCounts(characters.Length, widths);
        if (counts.Length == 0)
        {
            return rawPrediction;
        }

        var chunks = new string[counts.Length];
        var offset = 0;
        for (var index = 0; index < counts.Length; index++)
        {
            chunks[index] = string.Concat(
                characters.AsSpan(offset, counts[index]).ToArray().Select(static character => character.ToString()));
            offset += counts[index];
        }

        return string.Join(' ', chunks);
    }

    public async ValueTask<OcrV8AmbiguityResult> ResolveAmbiguityAsync(
        OcrV8SourceCrop crop,
        string conservativePrediction,
        CancellationToken cancellationToken)
    {
        ValidateCrop(crop);
        ArgumentNullException.ThrowIfNull(conservativePrediction);
        cancellationToken.ThrowIfCancellationRequested();
        Rune[] nonspace = conservativePrediction
            .EnumerateRunes()
            .Where(static character => !Rune.IsWhiteSpace(character))
            .ToArray();
        bool allExtended = nonspace.Length > 0 &&
            nonspace.All(static character => ContainsAsciiRune(ExtendedAmbiguityGlyphs, character));
        Group[] groups = ActiveGroups(crop);
        if (allExtended)
        {
            groups = ForceExpectedAmbiguityGroups(crop, nonspace.Length, groups);
        }

        if (groups.Length != nonspace.Length ||
            !nonspace.Any(static character => ContainsAsciiRune(ExtendedAmbiguityGlyphs, character)))
        {
            return Unchanged(conservativePrediction);
        }

        int[] indices = allExtended
            ? Enumerable.Range(0, nonspace.Length).ToArray()
            : Enumerable.Range(0, nonspace.Length)
                .Where(index => ContainsAsciiRune(CanonicalAmbiguityGlyphs, nonspace[index]))
                .ToArray();
        if (indices.Length == 0)
        {
            return Unchanged(conservativePrediction);
        }

        int valuesPerGlyph = checked(AmbiguityImageSize * AmbiguityImageSize);
        var values = new float[checked(indices.Length * valuesPerGlyph)];
        for (var targetIndex = 0; targetIndex < indices.Length; targetIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EncodeGroupTensor(
                crop,
                groups,
                indices[targetIndex],
                values.AsSpan(targetIndex * valuesPerGlyph, valuesPerGlyph));
        }

        string inputTensorSha256 = HashFloatTensor(values);
        var request = new InferenceRequest(
            options.Model,
            new InferenceInput(
                values,
                [indices.Length, 1, AmbiguityImageSize, AmbiguityImageSize],
                options.InputName,
                options.OutputName),
            new StageCacheMaterial(
                crop.PixelSha256,
                PolygonMaterial(crop.OriginalPolygon),
                "identity",
                "ocr_v8_ambiguity_source_group",
                options.StageVersion,
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["configuration_sha256"] = configurationFingerprint,
                    ["conservative_prediction_sha256"] = Convert.ToHexStringLower(
                        SHA256.HashData(Encoding.UTF8.GetBytes(conservativePrediction))),
                    ["input_tensor_sha256"] = inputTensorSha256,
                    ["group_count"] = groups.Length,
                    ["executed_group_count"] = indices.Length,
                    ["all_extended_alias_route"] = allExtended,
                    ["allowed_providers"] = ProviderFingerprint(options.AllowedProviders),
                },
                OcrContract.Version),
            options.Timeout,
            options.AllowedProviders,
            options.BypassCache);

        InferenceResponse response = await runtime.RunAsync(request, cancellationToken).ConfigureAwait(false);
        if (!response.Succeeded || response.Execution is null)
        {
            string diagnostic = response.Error is null
                ? "The OCR ambiguity runtime returned no execution evidence."
                : $"{response.Error.Code}: {response.Error.TechnicalMessage}";
            throw new InvalidOperationException(diagnostic);
        }

        if (options.AllowedProviders is not null &&
            !options.AllowedProviders.Contains(response.Execution.Provider))
        {
            throw new InvalidDataException(
                $"OCR ambiguity classification executed with undeclared provider '{response.Execution.Provider}'.");
        }

        const int classCount = 4;
        if (response.Execution.Output.Count != checked(indices.Length * classCount))
        {
            throw new InvalidDataException(
                $"OCR ambiguity classifier returned {response.Execution.Output.Count} values; " +
                $"{indices.Length * classCount} were required.");
        }

        float[] logits = response.Execution.Output.ToArray();
        var changed = 0;
        for (var targetIndex = 0; targetIndex < indices.Length; targetIndex++)
        {
            int bestClass = ArgMaxFinite(logits.AsSpan(targetIndex * classCount, classCount));
            int characterIndex = indices[targetIndex];
            var replacement = new Rune(CanonicalAmbiguityGlyphs[bestClass]);
            changed += nonspace[characterIndex] == replacement ? 0 : 1;
            nonspace[characterIndex] = replacement;
        }

        string final = nonspace.All(static character => ContainsAsciiRune(CanonicalAmbiguityGlyphs, character))
            ? string.Join(' ', nonspace.Select(static character => character.ToString()))
            : ReinsertWhitespace(conservativePrediction, nonspace);
        return new OcrV8AmbiguityResult(
            final,
            changed,
            true,
            inputTensorSha256,
            response.Execution.Timing.InferenceMilliseconds);
    }

    public static void ValidateOptions(LocalOnnxAmbiguitySourceGroupOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Model.Validate();
        if (!string.Equals(options.InputName, "glyphs", StringComparison.Ordinal) ||
            !string.Equals(options.OutputName, "logits", StringComparison.Ordinal) ||
            string.IsNullOrWhiteSpace(options.StageVersion) ||
            options.Timeout <= TimeSpan.Zero || options.Timeout > TimeSpan.FromMinutes(5) ||
            !ValidProviderPolicy(options.AllowedProviders))
        {
            throw new ArgumentException(
                "Local ONNX ambiguity source-group options do not match the frozen V8 contract.",
                nameof(options));
        }
    }

    private static int[] SourceGroupWidths(OcrV8SourceCrop crop)
    {
        bool[] foreground = ForegroundMask(crop, out Bounds inkBounds);
        if (!inkBounds.IsValid)
        {
            return [];
        }

        bool[] active = ActiveColumns(foreground, crop.Width, crop.Height);
        int gapMinimum = Math.Max(
            MinimumGapPixels,
            checked((int)Math.Ceiling(inkBounds.Height * MinimumGapToInkHeightRatio)));
        var widths = new List<int>();
        int? start = null;
        var prior = -1;
        for (var x = inkBounds.Left; x <= inkBounds.Right; x++)
        {
            if (!active[x])
            {
                continue;
            }

            if (start is null)
            {
                start = x;
            }
            else if (x - prior - 1 >= gapMinimum)
            {
                widths.Add(prior - start.Value + 1);
                start = x;
            }

            prior = x;
        }

        if (start.HasValue)
        {
            widths.Add(prior - start.Value + 1);
        }

        return widths.ToArray();
    }

    private static int[] PartitionCharacterCounts(int characterCount, int[] widths)
    {
        int groupCount = widths.Length;
        if (groupCount < MinimumSourceGroups || groupCount > characterCount || characterCount > 64)
        {
            return [];
        }

        double totalWidth = widths.Sum();
        double[] cumulative = new double[groupCount - 1];
        var running = 0d;
        for (var index = 0; index < cumulative.Length; index++)
        {
            running += widths[index];
            cumulative[index] = running / totalWidth;
        }

        double bestCost = double.PositiveInfinity;
        int[]? bestCounts = null;
        int[] boundaries = new int[groupCount - 1];
        Enumerate(boundaryIndex: 0, minimum: 1);
        return bestCounts ?? [];

        void Enumerate(int boundaryIndex, int minimum)
        {
            if (boundaryIndex == boundaries.Length)
            {
                var counts = new int[groupCount];
                var left = 0;
                var cost = 0d;
                for (var index = 0; index < boundaries.Length; index++)
                {
                    counts[index] = boundaries[index] - left;
                    left = boundaries[index];
                    double difference = (boundaries[index] / (double)characterCount) - cumulative[index];
                    cost += difference * difference;
                }

                counts[^1] = characterCount - left;
                if (cost < bestCost ||
                    (cost == bestCost && LexicographicallyLess(counts, bestCounts)))
                {
                    bestCost = cost;
                    bestCounts = counts;
                }

                return;
            }

            int remainingBoundaries = boundaries.Length - boundaryIndex - 1;
            int maximum = characterCount - remainingBoundaries - 1;
            for (int value = minimum; value <= maximum; value++)
            {
                boundaries[boundaryIndex] = value;
                Enumerate(boundaryIndex + 1, value + 1);
            }
        }
    }

    private static Group[] ActiveGroups(OcrV8SourceCrop crop)
    {
        bool[] foreground = ForegroundMask(crop, out Bounds inkBounds);
        if (!inkBounds.IsValid)
        {
            return [];
        }

        bool[] active = ActiveColumns(foreground, crop.Width, crop.Height);
        int gapMinimum = Math.Max(
            MinimumGapPixels,
            checked((int)Math.Ceiling(inkBounds.Height * MinimumGapToInkHeightRatio)));
        var intervals = new List<(int Left, int Right)>();
        int? start = null;
        var prior = -1;
        for (var x = 0; x < crop.Width; x++)
        {
            if (!active[x])
            {
                continue;
            }

            if (start is null)
            {
                start = x;
            }
            else if (x - prior - 1 >= gapMinimum)
            {
                intervals.Add((start.Value, prior + 1));
                start = x;
            }

            prior = x;
        }

        if (start.HasValue)
        {
            intervals.Add((start.Value, prior + 1));
        }

        return intervals.Select(interval => GroupFromInterval(
                foreground,
                crop.Width,
                crop.Height,
                interval.Left,
                interval.Right))
            .Where(static group => group.IsValid)
            .ToArray();
    }

    private static Group[] ForceExpectedAmbiguityGroups(
        OcrV8SourceCrop crop,
        int expectedCount,
        Group[] existing)
    {
        if (existing.Length == expectedCount)
        {
            return existing;
        }

        bool[] foreground = ForegroundMask(crop, out Bounds bounds);
        if (!bounds.IsValid || expectedCount < 2)
        {
            return existing;
        }

        bool[] active = ActiveColumns(foreground, crop.Width, crop.Height);
        var activeColumns = Enumerable.Range(0, crop.Width).Where(x => active[x]).ToArray();
        var gaps = activeColumns.Zip(activeColumns.Skip(1), static (prior, current) =>
                new Gap(current - prior - 1, prior, current))
            .Where(static gap => gap.Size > 0)
            .OrderByDescending(static gap => gap.Size)
            .ThenByDescending(static gap => gap.Prior)
            .ThenByDescending(static gap => gap.Current)
            .Take(expectedCount - 1)
            .Where(gap => gap.Size >= Math.Max(1, checked((int)Math.Ceiling(bounds.Height * 0.25))))
            .OrderBy(static gap => gap.Prior)
            .ThenBy(static gap => gap.Current)
            .ToArray();
        if (gaps.Length != expectedCount - 1)
        {
            return existing;
        }

        int[] starts = [bounds.Left, .. gaps.Select(static gap => gap.Current)];
        int[] ends = [.. gaps.Select(static gap => gap.Prior + 1), bounds.Right + 1];
        Group[] result = starts.Zip(ends, (left, right) =>
                GroupFromInterval(foreground, crop.Width, crop.Height, left, right))
            .ToArray();
        return result.All(static group => group.IsValid) ? result : existing;
    }

    private static void EncodeGroupTensor(
        OcrV8SourceCrop crop,
        IReadOnlyList<Group> groups,
        int index,
        Span<float> destination)
    {
        if (index < 0 || index >= groups.Count || destination.Length != AmbiguityImageSize * AmbiguityImageSize)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        Group group = groups[index];
        int lineHeight = groups.Max(static candidate => candidate.Height);
        int baseline = groups.Max(static candidate => candidate.Bottom);
        double scale = 11.5 / Math.Max(1, lineHeight);
        int resizedWidth = Math.Max(1, checked((int)Math.Round(group.Width * scale, MidpointRounding.ToEven)));
        int resizedHeight = Math.Max(1, checked((int)Math.Round(group.Height * scale, MidpointRounding.ToEven)));
        var source = new byte[checked(group.Width * group.Height)];
        for (var y = 0; y < group.Height; y++)
        {
            crop.Pixels.Span.Slice(
                    ((group.Top + y) * crop.Width) + group.Left,
                    group.Width)
                .CopyTo(source.AsSpan(y * group.Width, group.Width));
        }

        byte[] resized = PillowBilinearResizer.Resize(
            source,
            group.Width,
            group.Height,
            resizedWidth,
            resizedHeight);
        destination.Clear();
        int pasteX = (AmbiguityImageSize - resizedWidth) / 2;
        int pasteY = checked((int)Math.Round(
            21 - ((baseline - group.Top) * scale),
            MidpointRounding.ToEven));
        for (var y = 0; y < resizedHeight; y++)
        {
            int targetY = pasteY + y;
            if (targetY < 0 || targetY >= AmbiguityImageSize)
            {
                continue;
            }

            for (var x = 0; x < resizedWidth; x++)
            {
                int targetX = pasteX + x;
                if (targetX < 0 || targetX >= AmbiguityImageSize)
                {
                    continue;
                }

                destination[(targetY * AmbiguityImageSize) + targetX] =
                    1f - (resized[(y * resizedWidth) + x] / 255f);
            }
        }
    }

    private static bool[] ForegroundMask(OcrV8SourceCrop crop, out Bounds bounds)
    {
        ValidateCrop(crop);
        ReadOnlySpan<byte> pixels = crop.Pixels.Span;
        var edge = new byte[checked((2 * crop.Width) + (2 * crop.Height))];
        var offset = 0;
        pixels[..crop.Width].CopyTo(edge.AsSpan(offset, crop.Width));
        offset += crop.Width;
        pixels.Slice((crop.Height - 1) * crop.Width, crop.Width).CopyTo(edge.AsSpan(offset, crop.Width));
        offset += crop.Width;
        for (var y = 0; y < crop.Height; y++)
        {
            edge[offset++] = pixels[y * crop.Width];
        }

        for (var y = 0; y < crop.Height; y++)
        {
            edge[offset++] = pixels[(y * crop.Width) + crop.Width - 1];
        }

        double background = Median(edge);
        double contrast = Math.Max(0, background - Percentile(pixels, 0.01));
        double threshold = background - Math.Max(
            MinimumForegroundContrast,
            contrast * ForegroundContrastFraction);
        var foreground = new bool[pixels.Length];
        int left = crop.Width;
        int top = crop.Height;
        var right = -1;
        var bottom = -1;
        for (var y = 0; y < crop.Height; y++)
        {
            for (var x = 0; x < crop.Width; x++)
            {
                int index = (y * crop.Width) + x;
                if (pixels[index] > threshold)
                {
                    continue;
                }

                foreground[index] = true;
                left = Math.Min(left, x);
                right = Math.Max(right, x);
                top = Math.Min(top, y);
                bottom = Math.Max(bottom, y);
            }
        }

        bounds = new Bounds(left, top, right, bottom);
        return foreground;
    }

    private static bool[] ActiveColumns(bool[] foreground, int width, int height)
    {
        var active = new bool[width];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                active[x] |= foreground[(y * width) + x];
            }
        }

        return active;
    }

    private static Group GroupFromInterval(
        bool[] foreground,
        int width,
        int height,
        int left,
        int right)
    {
        var top = height;
        var bottom = -1;
        for (var y = 0; y < height; y++)
        {
            for (var x = left; x < right; x++)
            {
                if (!foreground[(y * width) + x])
                {
                    continue;
                }

                top = Math.Min(top, y);
                bottom = Math.Max(bottom, y);
            }
        }

        return new Group(left, top, right, bottom + 1);
    }

    private static double Median(ReadOnlySpan<byte> values)
    {
        byte[] sorted = values.ToArray();
        Array.Sort(sorted);
        int midpoint = sorted.Length / 2;
        return sorted.Length % 2 == 0
            ? (sorted[midpoint - 1] + sorted[midpoint]) / 2d
            : sorted[midpoint];
    }

    private static double Percentile(ReadOnlySpan<byte> values, double fraction)
    {
        byte[] sorted = values.ToArray();
        Array.Sort(sorted);
        double position = (sorted.Length - 1) * fraction;
        int lower = (int)Math.Floor(position);
        int upper = (int)Math.Ceiling(position);
        double weight = position - lower;
        return (sorted[lower] * (1 - weight)) + (sorted[upper] * weight);
    }

    private static int ArgMaxFinite(ReadOnlySpan<float> row)
    {
        if (row.Length == 0 || !float.IsFinite(row[0]))
        {
            throw new InvalidDataException("OCR ambiguity logits contain a non-finite value.");
        }

        var best = 0;
        for (var index = 1; index < row.Length; index++)
        {
            if (!float.IsFinite(row[index]))
            {
                throw new InvalidDataException("OCR ambiguity logits contain a non-finite value.");
            }

            if (row[index] > row[best])
            {
                best = index;
            }
        }

        return best;
    }

    private static bool ContainsAsciiRune(string values, Rune candidate) =>
        candidate.IsAscii && values.Contains((char)candidate.Value, StringComparison.Ordinal);

    private static string ReinsertWhitespace(string source, Rune[] nonspace)
    {
        var result = new StringBuilder(source.Length);
        var index = 0;
        foreach (Rune character in source.EnumerateRunes())
        {
            result.Append(Rune.IsWhiteSpace(character) ? character.ToString() : nonspace[index++].ToString());
        }

        return result.ToString();
    }

    private static bool LexicographicallyLess(int[] left, int[]? right)
    {
        if (right is null)
        {
            return true;
        }

        for (var index = 0; index < left.Length; index++)
        {
            if (left[index] != right[index])
            {
                return left[index] < right[index];
            }
        }

        return false;
    }

    private static OcrV8AmbiguityResult Unchanged(string text) =>
        new(text, 0, false, string.Empty, 0);

    private static string HashFloatTensor(ReadOnlySpan<float> values)
    {
        using var stream = new MemoryStream(values.Length * sizeof(float));
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        foreach (float value in values)
        {
            writer.Write(value);
        }

        writer.Flush();
        return Convert.ToHexStringLower(SHA256.HashData(stream.GetBuffer().AsSpan(0, checked((int)stream.Length))));
    }

    private static string PolygonMaterial(OcrPolygon polygon) =>
        string.Join(';', polygon.Points.Select(static point =>
            FormattableString.Invariant($"{point.X:R},{point.Y:R}")));

    private static void ValidateCrop(OcrV8SourceCrop crop)
    {
        ArgumentNullException.ThrowIfNull(crop);
        if (crop.Width <= 0 || crop.Height <= 0 ||
            crop.Pixels.Length != checked(crop.Width * crop.Height) ||
            crop.PixelSha256.Length != 64 || crop.PixelSha256.Any(static character => !Uri.IsHexDigit(character)) ||
            !string.Equals(
                crop.PixelSha256,
                Convert.ToHexStringLower(SHA256.HashData(crop.Pixels.Span)),
                StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("OCR V8 source crop bytes or checksum are invalid.", nameof(crop));
        }
    }

    private static bool ValidProviderPolicy(IReadOnlyList<InferenceProvider>? providers) =>
        providers is null ||
        (providers.Count > 0 &&
         providers.Contains(InferenceProvider.Cpu) &&
         providers.All(static provider => provider is InferenceProvider.Cpu or InferenceProvider.DirectMl) &&
         providers.Distinct().Count() == providers.Count);

    private static string ProviderFingerprint(IReadOnlyList<InferenceProvider>? providers) =>
        providers is null
            ? "policy-default"
            : string.Join(',', providers.OrderBy(static provider => provider));

    private static string HashStrings(IEnumerable<string> values)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (string value in values)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            hash.AppendData(BitConverter.GetBytes(bytes.Length));
            hash.AppendData(bytes);
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    private readonly record struct Bounds(int Left, int Top, int Right, int Bottom)
    {
        public bool IsValid => Left <= Right && Top <= Bottom;

        public int Height => Bottom - Top + 1;
    }

    private readonly record struct Group(int Left, int Top, int Right, int Bottom)
    {
        public bool IsValid => Left < Right && Top < Bottom;

        public int Width => Right - Left;

        public int Height => Bottom - Top;
    }

    private readonly record struct Gap(int Size, int Prior, int Current);

    [GeneratedRegex("^-?\\d+(?:\\.\\d+)?%?$", RegexOptions.CultureInvariant)]
    internal static partial Regex GraphNumber();
}
