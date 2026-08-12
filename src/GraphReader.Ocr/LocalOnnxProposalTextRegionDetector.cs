// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using GraphReader.Inference;

namespace GraphReader.Ocr;

public sealed record LocalOnnxProposalTextRegionDetectorOptions(ModelIdentity Model)
{
    public int CropWidth { get; init; } = 128;

    public int CropHeight { get; init; } = 32;

    public int InputChannels { get; init; } = 2;

    public int GeometryFeatureCount { get; init; } = 12;

    public int MinimumComponentArea { get; init; } = 2;

    public double MaximumComponentWidthRatio { get; init; } = 0.15;

    public double MaximumComponentHeightRatio { get; init; } = 0.20;

    public double MinimumVerticalOverlapRatio { get; init; } = 0.35;

    public double MaximumHorizontalGapHeightRatio { get; init; } = 2.5;

    public double MaximumComponentHeightRatioWithinLine { get; init; } = 2.0;

    public double MaximumMergedHeightGrowthRatio { get; init; } = 1.6;

    public double TightHorizontalPaddingPixels { get; init; } = 1.0;

    public double TightVerticalPaddingRatio { get; init; } = 0.25;

    public double ContextHorizontalPaddingHeightRatio { get; init; } = 2.0;

    public double ContextVerticalPaddingHeightRatio { get; init; } = 1.5;

    public double ContextMinimumPaddingPixels { get; init; } = 8.0;

    public double ProposalThresholdMeanRatio { get; init; } = 0.8;

    public int ProposalThresholdMinimum { get; init; } = 32;

    public int ProposalThresholdMaximum { get; init; } = 224;

    public float ConfidenceThreshold { get; init; } = 0.95f;

    public int MaximumProposals { get; init; } = 4096;

    public string InputName { get; init; } = "region_proposals";

    public string OutputName { get; init; } = "region_logits";

    public string StageVersion { get; init; } = "0.0.21-p2";

    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

    public IReadOnlyList<InferenceProvider>? AllowedProviders { get; init; }

    public bool BypassCache { get; init; }
}

/// <summary>
/// Executes the checksum-bound V8 component-fusion classifier over proposals
/// produced from the immutable Gray8 frame. This runtime contains no model
/// bytes and does not grant production approval to a candidate payload.
/// </summary>
public sealed class LocalOnnxProposalTextRegionDetector : ITextRegionDetector
{
    public const string ProposalAlgorithm = "adaptive-gray-baseline-bounded-line-grouping-v2";
    public const string EncodingAlgorithm = "graph-text-component-context-v7-encoding-v1";
    public const string PostprocessingAlgorithm = "component-fusion-proposal-classifier-v1";

    private readonly InferenceRuntime runtime;
    private readonly LocalOnnxProposalTextRegionDetectorOptions options;
    private readonly string configurationFingerprint;

    public LocalOnnxProposalTextRegionDetector(
        InferenceRuntime runtime,
        LocalOnnxProposalTextRegionDetectorOptions options)
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
        configurationFingerprint = CreateConfigurationFingerprint(this.options);
    }

    public string ConfigurationFingerprint => configurationFingerprint;

    public async ValueTask<IReadOnlyList<OcrDetectedRegion>> DetectAsync(
        OcrImage image,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(image);
        ValidateImage(image);
        cancellationToken.ThrowIfCancellationRequested();

        byte threshold = EstimateThreshold(image, options, cancellationToken);
        Component[] proposals = BuildProposals(image, threshold, options, cancellationToken);
        if (proposals.Length == 0)
        {
            return Array.Empty<OcrDetectedRegion>();
        }

        if (proposals.Length > options.MaximumProposals)
        {
            throw new InvalidDataException(
                $"OCR proposal count {proposals.Length} exceeds the reviewed safety limit {options.MaximumProposals}.");
        }

        int encodedWidth = checked(options.CropWidth + options.GeometryFeatureCount);
        int valuesPerProposal = checked(options.InputChannels * options.CropHeight * encodedWidth);
        var tensor = new float[checked(proposals.Length * valuesPerProposal)];
        for (var proposalIndex = 0; proposalIndex < proposals.Length; proposalIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EncodeProposal(
                image,
                proposals[proposalIndex],
                threshold,
                options,
                tensor.AsSpan(proposalIndex * valuesPerProposal, valuesPerProposal));
        }

        string imageSha256 = Convert.ToHexStringLower(SHA256.HashData(image.Pixels.Span));
        var request = new InferenceRequest(
            options.Model,
            new InferenceInput(
                tensor,
                [proposals.Length, options.InputChannels, options.CropHeight, encodedWidth],
                options.InputName,
                options.OutputName),
            new StageCacheMaterial(
                imageSha256,
                FormattableString.Invariant($"0,0,{image.Width},{image.Height}"),
                TransformFingerprint(image.OriginalToImage),
                "ocr_proposal_detection",
                options.StageVersion,
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["configuration_sha256"] = configurationFingerprint,
                    ["proposal_algorithm"] = ProposalAlgorithm,
                    ["encoding_algorithm"] = EncodingAlgorithm,
                    ["postprocessing_algorithm"] = PostprocessingAlgorithm,
                    ["proposal_count"] = proposals.Length,
                    ["proposal_threshold"] = threshold,
                    ["confidence_threshold"] = options.ConfidenceThreshold,
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
                ? "The OCR proposal-classifier runtime returned no execution evidence."
                : $"{response.Error.Code}: {response.Error.TechnicalMessage}";
            throw new InvalidOperationException(diagnostic);
        }

        if (options.AllowedProviders is not null &&
            !options.AllowedProviders.Contains(response.Execution.Provider))
        {
            throw new InvalidDataException(
                $"OCR proposal classification executed with undeclared provider '{response.Execution.Provider}'.");
        }

        int expectedOutputCount = checked(proposals.Length * 2);
        if (response.Execution.Output.Count != expectedOutputCount)
        {
            throw new InvalidDataException(
                $"OCR proposal classifier returned {response.Execution.Output.Count} values; " +
                $"{expectedOutputCount} were required.");
        }

        var accepted = new List<OcrDetectedRegion>();
        ReadOnlySpan<float> logits = response.Execution.Output.ToArray();
        for (var proposalIndex = 0; proposalIndex < proposals.Length; proposalIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            float rejectedLogit = logits[proposalIndex * 2];
            float acceptedLogit = logits[(proposalIndex * 2) + 1];
            float confidence = SoftmaxClassOne(rejectedLogit, acceptedLogit);
            if (confidence < options.ConfidenceThreshold)
            {
                continue;
            }

            Component proposal = proposals[proposalIndex];
            OcrRectangle rectangle = MapToOriginal(image, proposal);
            double density = proposal.Area / (double)Math.Max(1, proposal.Width * proposal.Height);
            accepted.Add(new OcrDetectedRegion(
                DeterministicRegionId(options.Model.Sha256, rectangle),
                OcrPolygon.FromRectangle(rectangle),
                0,
                confidence,
                Evidence: new OcrRegionEvidence(
                    proposal.Count,
                    density,
                    confidence,
                    1d - confidence,
                    false,
                    Array.AsReadOnly(new[] { PostprocessingAlgorithm }))));
        }

        return Array.AsReadOnly(accepted.ToArray());
    }

    public static void ValidateOptions(LocalOnnxProposalTextRegionDetectorOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Model.Validate();
        if (options.CropWidth != 128 || options.CropHeight != 32 ||
            options.InputChannels != 2 || options.GeometryFeatureCount != 12 ||
            options.MinimumComponentArea != 2 ||
            options.MaximumComponentWidthRatio != 0.15 ||
            options.MaximumComponentHeightRatio != 0.20 ||
            options.MinimumVerticalOverlapRatio != 0.35 ||
            options.MaximumHorizontalGapHeightRatio != 2.5 ||
            options.MaximumComponentHeightRatioWithinLine != 2.0 ||
            options.MaximumMergedHeightGrowthRatio != 1.6 ||
            options.TightHorizontalPaddingPixels != 1.0 ||
            options.TightVerticalPaddingRatio != 0.25 ||
            options.ContextHorizontalPaddingHeightRatio != 2.0 ||
            options.ContextVerticalPaddingHeightRatio != 1.5 ||
            options.ContextMinimumPaddingPixels != 8.0 ||
            options.ProposalThresholdMeanRatio != 0.8 ||
            options.ProposalThresholdMinimum != 32 ||
            options.ProposalThresholdMaximum != 224 ||
            options.ConfidenceThreshold != 0.95f ||
            options.MaximumProposals != 4096 ||
            !string.Equals(options.InputName, "region_proposals", StringComparison.Ordinal) ||
            !string.Equals(options.OutputName, "region_logits", StringComparison.Ordinal) ||
            string.IsNullOrWhiteSpace(options.StageVersion) ||
            options.Timeout <= TimeSpan.Zero || options.Timeout > TimeSpan.FromMinutes(5) ||
            !ValidProviderPolicy(options.AllowedProviders))
        {
            throw new ArgumentException(
                "Local ONNX proposal-classifier OCR detector options do not match the frozen V8 contract.",
                nameof(options));
        }
    }

    private static Component[] BuildProposals(
        OcrImage image,
        byte threshold,
        LocalOnnxProposalTextRegionDetectorOptions options,
        CancellationToken cancellationToken)
    {
        List<Component> remaining = ConnectedComponents(image, threshold, options, cancellationToken)
            .OrderBy(static item => item.Top)
            .ThenBy(static item => item.Left)
            .ThenBy(static item => item.Bottom)
            .ThenBy(static item => item.Right)
            .ToList();
        var lines = new List<Component>();
        while (remaining.Count > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Component line = remaining[0];
            remaining.RemoveAt(0);
            var changed = true;
            while (changed)
            {
                changed = false;
                for (int index = remaining.Count - 1; index >= 0; index--)
                {
                    Component candidate = remaining[index];
                    int minimumHeight = Math.Max(1, Math.Min(line.Height, candidate.Height));
                    int maximumHeight = Math.Max(line.Height, candidate.Height);
                    if (maximumHeight / (double)minimumHeight > options.MaximumComponentHeightRatioWithinLine)
                    {
                        continue;
                    }

                    int overlap = Math.Max(
                        0,
                        Math.Min(line.Bottom, candidate.Bottom) - Math.Max(line.Top, candidate.Top) + 1);
                    if (overlap / (double)minimumHeight < options.MinimumVerticalOverlapRatio)
                    {
                        continue;
                    }

                    int gap = candidate.Left > line.Right
                        ? candidate.Left - line.Right - 1
                        : line.Left > candidate.Right
                            ? line.Left - candidate.Right - 1
                            : 0;
                    if (gap > maximumHeight * options.MaximumHorizontalGapHeightRatio)
                    {
                        continue;
                    }

                    Component merged = line.Merge(candidate);
                    if (merged.Height > maximumHeight * options.MaximumMergedHeightGrowthRatio)
                    {
                        continue;
                    }

                    line = merged;
                    remaining.RemoveAt(index);
                    changed = true;
                }
            }

            lines.Add(line);
        }

        return lines
            .OrderBy(static item => item.Top)
            .ThenBy(static item => item.Left)
            .ThenBy(static item => item.Bottom)
            .ThenBy(static item => item.Right)
            .ToArray();
    }

    private static List<Component> ConnectedComponents(
        OcrImage image,
        byte threshold,
        LocalOnnxProposalTextRegionDetectorOptions options,
        CancellationToken cancellationToken)
    {
        var visited = new bool[checked(image.Width * image.Height)];
        var components = new List<Component>();
        var queue = new Queue<int>();
        double maximumWidth = Math.Max(2d, image.Width * options.MaximumComponentWidthRatio);
        double maximumHeight = Math.Max(2d, image.Height * options.MaximumComponentHeightRatio);
        for (var y = 0; y < image.Height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (var x = 0; x < image.Width; x++)
            {
                int start = checked((y * image.Width) + x);
                if (visited[start])
                {
                    continue;
                }

                visited[start] = true;
                if (!IsForeground(image, x, y, threshold))
                {
                    continue;
                }

                queue.Enqueue(start);
                int left = x;
                int right = x;
                int top = y;
                int bottom = y;
                var area = 0;
                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    if ((area & 1023) == 0)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    int currentX = current % image.Width;
                    int currentY = current / image.Width;
                    left = Math.Min(left, currentX);
                    right = Math.Max(right, currentX);
                    top = Math.Min(top, currentY);
                    bottom = Math.Max(bottom, currentY);
                    area++;
                    Visit(currentX - 1, currentY);
                    Visit(currentX + 1, currentY);
                    Visit(currentX, currentY - 1);
                    Visit(currentX, currentY + 1);
                }

                var component = new Component(left, top, right, bottom, area, 1);
                if (area >= options.MinimumComponentArea &&
                    component.Width <= maximumWidth && component.Height <= maximumHeight)
                {
                    components.Add(component);
                }

                void Visit(int nextX, int nextY)
                {
                    if (nextX < 0 || nextY < 0 || nextX >= image.Width || nextY >= image.Height)
                    {
                        return;
                    }

                    int next = checked((nextY * image.Width) + nextX);
                    if (visited[next])
                    {
                        return;
                    }

                    visited[next] = true;
                    if (IsForeground(image, nextX, nextY, threshold))
                    {
                        queue.Enqueue(next);
                    }
                }
            }
        }

        return components;
    }

    private static void EncodeProposal(
        OcrImage image,
        Component proposal,
        byte threshold,
        LocalOnnxProposalTextRegionDetectorOptions options,
        Span<float> destination)
    {
        int encodedWidth = checked(options.CropWidth + options.GeometryFeatureCount);
        int valuesPerChannel = checked(options.CropHeight * encodedWidth);
        destination.Clear();

        double tightVerticalPadding = Math.Max(1d, proposal.Height * options.TightVerticalPaddingRatio);
        float tightMean = EncodeCrop(
            image,
            proposal.Left - options.TightHorizontalPaddingPixels,
            proposal.Top - tightVerticalPadding,
            proposal.Width + (2d * options.TightHorizontalPaddingPixels),
            proposal.Height + (2d * tightVerticalPadding),
            options,
            destination[..valuesPerChannel]);

        double contextHorizontalPadding = Math.Max(
            options.ContextMinimumPaddingPixels,
            proposal.Height * options.ContextHorizontalPaddingHeightRatio);
        double contextVerticalPadding = Math.Max(
            options.ContextMinimumPaddingPixels,
            proposal.Height * options.ContextVerticalPaddingHeightRatio);
        double contextLeft = proposal.Left - contextHorizontalPadding;
        double contextTop = proposal.Top - contextVerticalPadding;
        double contextWidth = proposal.Width + (2d * contextHorizontalPadding);
        double contextHeight = proposal.Height + (2d * contextVerticalPadding);
        float contextMean = EncodeCrop(
            image,
            contextLeft,
            contextTop,
            contextWidth,
            contextHeight,
            options,
            destination[valuesPerChannel..]);

        int x0 = Math.Max(0, checked((int)Math.Floor(contextLeft)));
        int y0 = Math.Max(0, checked((int)Math.Floor(contextTop)));
        int x1 = Math.Min(image.Width, checked((int)Math.Ceiling(contextLeft + contextWidth)));
        int y1 = Math.Min(image.Height, checked((int)Math.Ceiling(contextTop + contextHeight)));
        (float maximumRowDensity, float maximumColumnDensity, float edgeDensity) =
            ContextDensities(image, threshold, x0, y0, x1, y1);
        float[] geometry =
        [
            proposal.Width / (float)image.Width,
            proposal.Height / (float)image.Height,
            proposal.Area / (float)Math.Max(1, proposal.Width * proposal.Height),
            Math.Min(1f, proposal.Width / Math.Max(1f, proposal.Height * 8f)),
            Math.Min(1f, proposal.Height / Math.Max(1f, proposal.Width * 4f)),
            Math.Min(1f, proposal.Count / 16f),
            threshold / 255f,
            tightMean,
            contextMean,
            maximumRowDensity,
            maximumColumnDensity,
            edgeDensity,
        ];

        for (var channel = 0; channel < options.InputChannels; channel++)
        {
            int channelOffset = channel * valuesPerChannel;
            for (var y = 0; y < options.CropHeight; y++)
            {
                geometry.CopyTo(destination[(channelOffset + (y * encodedWidth) + options.CropWidth)..]);
            }
        }
    }

    private static float EncodeCrop(
        OcrImage image,
        double left,
        double top,
        double width,
        double height,
        LocalOnnxProposalTextRegionDetectorOptions options,
        Span<float> channelDestination)
    {
        int encodedWidth = checked(options.CropWidth + options.GeometryFeatureCount);
        int contentWidth = Math.Clamp(
            checked((int)Math.Ceiling(options.CropHeight * width / height)),
            1,
            options.CropWidth);
        for (var targetY = 0; targetY < options.CropHeight; targetY++)
        {
            double sourceY = top + (((targetY + 0.5) / options.CropHeight) * height) - 0.5;
            for (var targetX = 0; targetX < options.CropWidth; targetX++)
            {
                float ink = 0;
                if (targetX < contentWidth)
                {
                    double sourceX = left + (((targetX + 0.5) / contentWidth) * width) - 0.5;
                    if (sourceX >= 0 && sourceX < image.Width && sourceY >= 0 && sourceY < image.Height)
                    {
                        float sampled = SampleBilinear(image, sourceX, sourceY);
                        float rounded = MathF.Round(sampled, MidpointRounding.ToEven);
                        ink = 1f - (Math.Clamp(rounded, 0f, 255f) / 255f);
                    }
                }

                channelDestination[(targetY * encodedWidth) + targetX] = ink;
            }
        }

        return NumpyFloat32Mean(channelDestination, options.CropHeight, encodedWidth, options.CropWidth);
    }

    private static float NumpyFloat32Mean(
        ReadOnlySpan<float> source,
        int rowCount,
        int stride,
        int columnCount)
    {
        var contiguous = new float[checked(rowCount * columnCount)];
        for (var row = 0; row < rowCount; row++)
        {
            source.Slice(row * stride, columnCount).CopyTo(contiguous.AsSpan(row * columnCount));
        }

        return PairwiseFloat32Sum(contiguous) / contiguous.Length;
    }

    private static float PairwiseFloat32Sum(ReadOnlySpan<float> values)
    {
        if (values.Length < 8)
        {
            float result = -0f;
            foreach (float value in values)
            {
                result += value;
            }

            return result;
        }

        if (values.Length <= 128)
        {
            Span<float> partial = stackalloc float[8];
            values[..8].CopyTo(partial);
            var index = 8;
            for (; index + 7 < values.Length; index += 8)
            {
                for (var lane = 0; lane < 8; lane++)
                {
                    partial[lane] += values[index + lane];
                }
            }

            float result =
                ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                ((partial[4] + partial[5]) + (partial[6] + partial[7]));
            for (; index < values.Length; index++)
            {
                result += values[index];
            }

            return result;
        }

        int midpoint = values.Length / 2;
        midpoint -= midpoint % 8;
        return PairwiseFloat32Sum(values[..midpoint]) + PairwiseFloat32Sum(values[midpoint..]);
    }

    private static float SampleBilinear(OcrImage image, double x, double y)
    {
        double clippedX = Math.Clamp(x, 0, image.Width - 1d);
        double clippedY = Math.Clamp(y, 0, image.Height - 1d);
        int x0 = checked((int)Math.Floor(clippedX));
        int y0 = checked((int)Math.Floor(clippedY));
        int x1 = Math.Min(x0 + 1, image.Width - 1);
        int y1 = Math.Min(y0 + 1, image.Height - 1);
        double xWeight = clippedX - x0;
        double yWeight = clippedY - y0;
        double top = Pixel(image, x0, y0) * (1d - xWeight) + Pixel(image, x1, y0) * xWeight;
        double bottom = Pixel(image, x0, y1) * (1d - xWeight) + Pixel(image, x1, y1) * xWeight;
        return (float)(top * (1d - yWeight) + bottom * yWeight);
    }

    private static (float MaximumRow, float MaximumColumn, float Edge) ContextDensities(
        OcrImage image,
        byte threshold,
        int x0,
        int y0,
        int x1,
        int y1)
    {
        int width = Math.Max(0, x1 - x0);
        int height = Math.Max(0, y1 - y0);
        if (width == 0 || height == 0)
        {
            return (0, 0, 0);
        }

        var columns = new int[width];
        int maximumRow = 0;
        var edgeCount = 0;
        for (var y = y0; y < y1; y++)
        {
            var rowCount = 0;
            for (var x = x0; x < x1; x++)
            {
                if (!IsForeground(image, x, y, threshold))
                {
                    continue;
                }

                rowCount++;
                columns[x - x0]++;
                if (y == y0)
                {
                    edgeCount++;
                }

                if (y == y1 - 1)
                {
                    edgeCount++;
                }

                if (x == x0)
                {
                    edgeCount++;
                }

                if (x == x1 - 1)
                {
                    edgeCount++;
                }
            }

            maximumRow = Math.Max(maximumRow, rowCount);
        }

        int maximumColumn = columns.Max();
        int edgeLength = checked((2 * width) + (2 * height));
        return (
            maximumRow / (float)width,
            maximumColumn / (float)height,
            edgeCount / (float)edgeLength);
    }

    private static OcrRectangle MapToOriginal(OcrImage image, Component proposal)
    {
        OcrPoint first = image.OriginalToImage.MapToOriginal(new OcrPoint(proposal.Left, proposal.Top));
        OcrPoint second = image.OriginalToImage.MapToOriginal(new OcrPoint(proposal.Right + 1, proposal.Bottom + 1));
        int canonicalWidth = image.CanonicalOriginalWidth ?? image.Width;
        int canonicalHeight = image.CanonicalOriginalHeight ?? image.Height;
        double left = Math.Clamp(Math.Min(first.X, second.X), 0, canonicalWidth);
        double top = Math.Clamp(Math.Min(first.Y, second.Y), 0, canonicalHeight);
        double right = Math.Clamp(Math.Max(first.X, second.X), 0, canonicalWidth);
        double bottom = Math.Clamp(Math.Max(first.Y, second.Y), 0, canonicalHeight);
        var rectangle = new OcrRectangle(left, top, right - left, bottom - top);
        if (!rectangle.IsValid)
        {
            throw new InvalidDataException("OCR proposal mapped to an invalid original-pixel rectangle.");
        }

        return rectangle;
    }

    private static byte EstimateThreshold(
        OcrImage image,
        LocalOnnxProposalTextRegionDetectorOptions options,
        CancellationToken cancellationToken)
    {
        long sum = 0;
        for (var y = 0; y < image.Height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (var x = 0; x < image.Width; x++)
            {
                sum += Pixel(image, x, y);
            }
        }

        double mean = sum / (double)checked(image.Width * image.Height);
        return (byte)Math.Clamp(
            Math.Round(mean * options.ProposalThresholdMeanRatio, MidpointRounding.ToEven),
            options.ProposalThresholdMinimum,
            options.ProposalThresholdMaximum);
    }

    private static byte Pixel(OcrImage image, int x, int y) =>
        image.Pixels.Span[checked((y * image.Stride) + x)];

    private static bool IsForeground(OcrImage image, int x, int y, byte threshold) =>
        Pixel(image, x, y) <= threshold;

    private static float SoftmaxClassOne(float classZero, float classOne)
    {
        if (!float.IsFinite(classZero) || !float.IsFinite(classOne))
        {
            throw new InvalidDataException("OCR proposal classifier output contains a non-finite logit.");
        }

        double maximum = Math.Max(classZero, classOne);
        double zero = Math.Exp(classZero - maximum);
        double one = Math.Exp(classOne - maximum);
        return (float)(one / (zero + one));
    }

    private static void ValidateImage(OcrImage image)
    {
        if (image.Width <= 0 || image.Height <= 0 || image.Stride < image.Width ||
            image.Pixels.Length < checked(image.Stride * image.Height) ||
            !image.OriginalToImage.IsInvertible ||
            !string.Equals(image.CoordinateSpace, OcrContract.CoordinateSpace, StringComparison.Ordinal) ||
            image.CanonicalOriginalWidth is <= 0 || image.CanonicalOriginalHeight is <= 0)
        {
            throw new ArgumentException("OCR proposal-classifier image is invalid.", nameof(image));
        }
    }

    private static bool ValidProviderPolicy(IReadOnlyList<InferenceProvider>? providers) =>
        providers is null ||
        (providers.Count > 0 &&
         providers.Contains(InferenceProvider.Cpu) &&
         providers.All(static provider => provider is InferenceProvider.Cpu or InferenceProvider.DirectMl) &&
         providers.Distinct().Count() == providers.Count);

    private static string CreateConfigurationFingerprint(
        LocalOnnxProposalTextRegionDetectorOptions options) =>
        HashStrings(
        [
            ProposalAlgorithm,
            EncodingAlgorithm,
            PostprocessingAlgorithm,
            options.Model.Sha256.ToLowerInvariant(),
            options.CropWidth.ToString(CultureInfo.InvariantCulture),
            options.CropHeight.ToString(CultureInfo.InvariantCulture),
            options.InputChannels.ToString(CultureInfo.InvariantCulture),
            options.GeometryFeatureCount.ToString(CultureInfo.InvariantCulture),
            options.MinimumComponentArea.ToString(CultureInfo.InvariantCulture),
            options.MaximumComponentWidthRatio.ToString("R", CultureInfo.InvariantCulture),
            options.MaximumComponentHeightRatio.ToString("R", CultureInfo.InvariantCulture),
            options.MinimumVerticalOverlapRatio.ToString("R", CultureInfo.InvariantCulture),
            options.MaximumHorizontalGapHeightRatio.ToString("R", CultureInfo.InvariantCulture),
            options.MaximumComponentHeightRatioWithinLine.ToString("R", CultureInfo.InvariantCulture),
            options.MaximumMergedHeightGrowthRatio.ToString("R", CultureInfo.InvariantCulture),
            options.TightHorizontalPaddingPixels.ToString("R", CultureInfo.InvariantCulture),
            options.TightVerticalPaddingRatio.ToString("R", CultureInfo.InvariantCulture),
            options.ContextHorizontalPaddingHeightRatio.ToString("R", CultureInfo.InvariantCulture),
            options.ContextVerticalPaddingHeightRatio.ToString("R", CultureInfo.InvariantCulture),
            options.ContextMinimumPaddingPixels.ToString("R", CultureInfo.InvariantCulture),
            options.ProposalThresholdMeanRatio.ToString("R", CultureInfo.InvariantCulture),
            options.ProposalThresholdMinimum.ToString(CultureInfo.InvariantCulture),
            options.ProposalThresholdMaximum.ToString(CultureInfo.InvariantCulture),
            options.ConfidenceThreshold.ToString("R", CultureInfo.InvariantCulture),
            options.MaximumProposals.ToString(CultureInfo.InvariantCulture),
            options.InputName,
            options.OutputName,
            options.StageVersion,
            ProviderFingerprint(options.AllowedProviders),
        ]);

    private static string DeterministicRegionId(string modelSha256, OcrRectangle rectangle)
    {
        string material = string.Create(
            CultureInfo.InvariantCulture,
            $"{modelSha256.ToLowerInvariant()}:{PostprocessingAlgorithm}:" +
            $"{rectangle.X:R},{rectangle.Y:R},{rectangle.Width:R},{rectangle.Height:R}");
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(material));
        return new Guid(hash.AsSpan(0, 16)).ToString("D");
    }

    private static string TransformFingerprint(OcrFrameTransform transform) =>
        FormattableString.Invariant(
            $"{transform.ScaleX:R},{transform.ScaleY:R},{transform.OffsetX:R},{transform.OffsetY:R}");

    private static string ProviderFingerprint(IReadOnlyList<InferenceProvider>? providers) =>
        providers is null
            ? "policy-default"
            : string.Join(',', providers
                .Distinct()
                .OrderBy(static provider => provider)
                .Select(static provider => provider.ToString()));

    private static string HashStrings(IEnumerable<string> values)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (string value in values)
        {
            hash.AppendData(Encoding.UTF8.GetBytes(value));
            hash.AppendData([0]);
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    private readonly record struct Component(
        int Left,
        int Top,
        int Right,
        int Bottom,
        int Area,
        int Count)
    {
        public int Width => Right - Left + 1;

        public int Height => Bottom - Top + 1;

        public Component Merge(Component other) =>
            new(
                Math.Min(Left, other.Left),
                Math.Min(Top, other.Top),
                Math.Max(Right, other.Right),
                Math.Max(Bottom, other.Bottom),
                checked(Area + other.Area),
                checked(Count + other.Count));
    }
}
