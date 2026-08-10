// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using GraphReader.Inference;

namespace GraphReader.Ocr;

public enum OcrDetectionOutputActivation
{
    Probability,
    SigmoidLogit,
}

public sealed record LocalOnnxTextRegionDetectorOptions(ModelIdentity Model)
{
    public int MaximumSideLength { get; init; } = 960;

    public int DimensionMultiple { get; init; } = 32;

    public int InputChannels { get; init; } = 3;

    public OcrTensorLayout InputLayout { get; init; } = OcrTensorLayout.ChannelsFirst;

    public OcrTensorColorMode InputColorMode { get; init; } =
        OcrTensorColorMode.GrayscaleReplicated;

    public IReadOnlyList<float> ChannelMeans { get; init; } = [0.485f, 0.456f, 0.406f];

    public IReadOnlyList<float> ChannelScales { get; init; } = [1f / 0.229f, 1f / 0.224f, 1f / 0.225f];

    public string InputName { get; init; } = "input";

    public string OutputName { get; init; } = "output";

    public string StageVersion { get; init; } = "0.1.0";

    public OcrDetectionOutputActivation OutputActivation { get; init; } =
        OcrDetectionOutputActivation.Probability;

    public float ProbabilityThreshold { get; init; } = 0.30f;

    public float BoxConfidenceThreshold { get; init; } = 0.55f;

    public double UnclipRatio { get; init; } = 1.5;

    public int MinimumComponentArea { get; init; } = 3;

    public int MinimumSideLength { get; init; } = 2;

    public int MaximumRegions { get; init; } = 1000;

    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

    public IReadOnlyList<InferenceProvider>? AllowedProviders { get; init; }

    public bool BypassCache { get; init; }
}

/// <summary>
/// Executes a checksum-bound local ONNX dense text-probability model and maps
/// deterministic axis-aligned regions back to immutable original pixels. This
/// adapter contains no weights and does not imply that any model is approved.
/// </summary>
public sealed class LocalOnnxTextRegionDetector : ITextRegionDetector
{
    private readonly InferenceRuntime runtime;
    private readonly LocalOnnxTextRegionDetectorOptions options;
    private readonly string configurationFingerprint;

    public LocalOnnxTextRegionDetector(
        InferenceRuntime runtime,
        LocalOnnxTextRegionDetectorOptions options)
    {
        this.runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        ArgumentNullException.ThrowIfNull(options);
        this.options = options with
        {
            ChannelMeans = options.ChannelMeans is null
                ? Array.Empty<float>()
                : Array.AsReadOnly(options.ChannelMeans.ToArray()),
            ChannelScales = options.ChannelScales is null
                ? Array.Empty<float>()
                : Array.AsReadOnly(options.ChannelScales.ToArray()),
            AllowedProviders = options.AllowedProviders is null
                ? null
                : Array.AsReadOnly(options.AllowedProviders.ToArray()),
        };
        options.Model.Validate();
        ValidateOptions(this.options);
        configurationFingerprint = CreateConfigurationFingerprint(this.options);
    }

    public string ConfigurationFingerprint => configurationFingerprint;

    public async ValueTask<IReadOnlyList<OcrDetectedRegion>> DetectAsync(
        OcrImage image,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(image);
        ValidateImage(image, options);
        cancellationToken.ThrowIfCancellationRequested();

        (int tensorWidth, int tensorHeight) = TensorDimensions(image, options);
        float[] tensor = CreateTensor(image, tensorWidth, tensorHeight, options, cancellationToken);
        IReadOnlyList<long> shape = options.InputLayout == OcrTensorLayout.ChannelsFirst
            ? [1, options.InputChannels, tensorHeight, tensorWidth]
            : [1, tensorHeight, tensorWidth, options.InputChannels];
        ReadOnlySpan<byte> consumedPixels = options.InputColorMode == OcrTensorColorMode.Bgr
            ? image.BgrPixels!.Pixels.Span
            : image.Pixels.Span;
        string imageSha256 = Convert.ToHexStringLower(SHA256.HashData(consumedPixels));
        var request = new InferenceRequest(
            options.Model,
            new InferenceInput(tensor, shape, options.InputName, options.OutputName),
            new StageCacheMaterial(
                imageSha256,
                FormattableString.Invariant($"0,0,{image.Width},{image.Height}"),
                TransformFingerprint(image.OriginalToImage),
                "ocr_detection",
                options.StageVersion,
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["input_width"] = tensorWidth,
                    ["input_height"] = tensorHeight,
                    ["input_channels"] = options.InputChannels,
                    ["input_layout"] = options.InputLayout.ToString(),
                    ["input_color_mode"] = options.InputColorMode.ToString(),
                    ["channel_means"] = options.ChannelMeans.ToArray(),
                    ["channel_scales"] = options.ChannelScales.ToArray(),
                    ["output_activation"] = options.OutputActivation.ToString(),
                    ["probability_threshold"] = options.ProbabilityThreshold,
                    ["box_confidence_threshold"] = options.BoxConfidenceThreshold,
                    ["unclip_ratio"] = options.UnclipRatio,
                    ["minimum_component_area"] = options.MinimumComponentArea,
                    ["minimum_side_length"] = options.MinimumSideLength,
                    ["maximum_regions"] = options.MaximumRegions,
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
                ? "The OCR detection runtime returned no execution evidence."
                : $"{response.Error.Code}: {response.Error.TechnicalMessage}";
            throw new InvalidOperationException(diagnostic);
        }

        if (options.AllowedProviders is not null &&
            !options.AllowedProviders.Contains(response.Execution.Provider))
        {
            throw new InvalidDataException(
                $"OCR detection executed with undeclared provider '{response.Execution.Provider}'.");
        }

        int expectedOutputCount = checked(tensorWidth * tensorHeight);
        if (response.Execution.Output.Count != expectedOutputCount)
        {
            throw new InvalidDataException(
                $"OCR detection output contained {response.Execution.Output.Count} values; {expectedOutputCount} were required.");
        }

        float[] probabilities = response.Execution.Output.ToArray();
        for (var index = 0; index < probabilities.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            probabilities[index] = ToProbability(probabilities[index], options.OutputActivation);
        }

        Component[] components = FindComponents(
                probabilities,
                tensorWidth,
                tensorHeight,
                options,
                cancellationToken)
            .Where(component => component.Confidence >= options.BoxConfidenceThreshold)
            .OrderByDescending(static component => component.Confidence)
            .ThenByDescending(static component => component.PixelCount)
            .ThenBy(static component => component.Top)
            .ThenBy(static component => component.Left)
            .Take(options.MaximumRegions)
            .OrderBy(static component => component.Top)
            .ThenBy(static component => component.Left)
            .ToArray();

        var regions = new List<OcrDetectedRegion>(components.Length);
        foreach (Component component in components)
        {
            cancellationToken.ThrowIfCancellationRequested();
            OcrRectangle rectangle = MapToOriginal(
                component,
                image,
                tensorWidth,
                tensorHeight,
                options.UnclipRatio);
            if (!rectangle.IsValid)
            {
                continue;
            }

            double density = component.PixelCount /
                (double)checked(component.Width * component.Height);
            regions.Add(new OcrDetectedRegion(
                DeterministicRegionId(options.Model.Sha256, rectangle),
                OcrPolygon.FromRectangle(rectangle),
                OrientationDegrees: 0,
                DetectionConfidence: component.Confidence,
                CoordinateSpace: OcrContract.CoordinateSpace,
                Evidence: new OcrRegionEvidence(
                    ComponentCount: 1,
                    InkDensity: Math.Clamp(density, 0, 1),
                    TextLikelihood: component.Confidence,
                    StructureLikelihood: 1 - component.Confidence,
                    LikelyGraphStructure: false,
                    Reasons: Array.AsReadOnly(["onnx_dense_text_probability"]))));
        }

        return regions.AsReadOnly();
    }

    private static IEnumerable<Component> FindComponents(
        float[] probabilities,
        int width,
        int height,
        LocalOnnxTextRegionDetectorOptions options,
        CancellationToken cancellationToken)
    {
        var visited = new bool[probabilities.Length];
        var queue = new Queue<int>();
        for (var y = 0; y < height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (var x = 0; x < width; x++)
            {
                int seed = checked((y * width) + x);
                if (visited[seed])
                {
                    continue;
                }

                visited[seed] = true;
                if (probabilities[seed] < options.ProbabilityThreshold)
                {
                    continue;
                }

                queue.Enqueue(seed);
                int left = x;
                int right = x;
                int top = y;
                int bottom = y;
                int pixelCount = 0;
                double confidenceSum = 0;
                while (queue.Count > 0)
                {
                    if ((pixelCount & 1023) == 0)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    int current = queue.Dequeue();
                    int currentX = current % width;
                    int currentY = current / width;
                    left = Math.Min(left, currentX);
                    right = Math.Max(right, currentX);
                    top = Math.Min(top, currentY);
                    bottom = Math.Max(bottom, currentY);
                    pixelCount++;
                    confidenceSum += probabilities[current];
                    for (var offsetY = -1; offsetY <= 1; offsetY++)
                    {
                        for (var offsetX = -1; offsetX <= 1; offsetX++)
                        {
                            if (offsetX == 0 && offsetY == 0)
                            {
                                continue;
                            }

                            int neighborX = currentX + offsetX;
                            int neighborY = currentY + offsetY;
                            if (neighborX < 0 || neighborY < 0 || neighborX >= width || neighborY >= height)
                            {
                                continue;
                            }

                            int neighbor = checked((neighborY * width) + neighborX);
                            if (visited[neighbor])
                            {
                                continue;
                            }

                            visited[neighbor] = true;
                            if (probabilities[neighbor] >= options.ProbabilityThreshold)
                            {
                                queue.Enqueue(neighbor);
                            }
                        }
                    }
                }

                var component = new Component(
                    left,
                    top,
                    right,
                    bottom,
                    pixelCount,
                    Math.Clamp(confidenceSum / Math.Max(1, pixelCount), 0, 1));
                if (component.PixelCount >= options.MinimumComponentArea &&
                    component.Width >= options.MinimumSideLength &&
                    component.Height >= options.MinimumSideLength)
                {
                    yield return component;
                }
            }
        }
    }

    private static OcrRectangle MapToOriginal(
        Component component,
        OcrImage image,
        int tensorWidth,
        int tensorHeight,
        double unclipRatio)
    {
        double area = checked(component.Width * component.Height);
        double perimeter = 2d * (component.Width + component.Height);
        double expansion = perimeter <= 0 ? 0 : (area * unclipRatio) / perimeter;
        double imageLeft = Math.Clamp(
            (component.Left - expansion) * image.Width / tensorWidth,
            0,
            image.Width);
        double imageTop = Math.Clamp(
            (component.Top - expansion) * image.Height / tensorHeight,
            0,
            image.Height);
        double imageRight = Math.Clamp(
            (component.Right + 1 + expansion) * image.Width / tensorWidth,
            0,
            image.Width);
        double imageBottom = Math.Clamp(
            (component.Bottom + 1 + expansion) * image.Height / tensorHeight,
            0,
            image.Height);
        OcrPoint first = image.OriginalToImage.MapToOriginal(new OcrPoint(imageLeft, imageTop));
        OcrPoint second = image.OriginalToImage.MapToOriginal(new OcrPoint(imageRight, imageBottom));
        double canonicalWidth = image.CanonicalOriginalWidth ?? image.Width;
        double canonicalHeight = image.CanonicalOriginalHeight ?? image.Height;
        double left = Math.Clamp(Math.Min(first.X, second.X), 0, canonicalWidth);
        double top = Math.Clamp(Math.Min(first.Y, second.Y), 0, canonicalHeight);
        double right = Math.Clamp(Math.Max(first.X, second.X), 0, canonicalWidth);
        double bottom = Math.Clamp(Math.Max(first.Y, second.Y), 0, canonicalHeight);
        return new OcrRectangle(left, top, right - left, bottom - top);
    }

    private static float[] CreateTensor(
        OcrImage image,
        int targetWidth,
        int targetHeight,
        LocalOnnxTextRegionDetectorOptions options,
        CancellationToken cancellationToken)
    {
        int pixelsPerChannel = checked(targetWidth * targetHeight);
        var values = new float[checked(pixelsPerChannel * options.InputChannels)];
        for (var targetY = 0; targetY < targetHeight; targetY++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            double sourceY = ((targetY + 0.5) * image.Height / targetHeight) - 0.5;
            int y0 = Math.Clamp((int)Math.Floor(sourceY), 0, image.Height - 1);
            int y1 = Math.Min(y0 + 1, image.Height - 1);
            double yWeight = Math.Clamp(sourceY - Math.Floor(sourceY), 0, 1);
            for (var targetX = 0; targetX < targetWidth; targetX++)
            {
                double sourceX = ((targetX + 0.5) * image.Width / targetWidth) - 0.5;
                int x0 = Math.Clamp((int)Math.Floor(sourceX), 0, image.Width - 1);
                int x1 = Math.Min(x0 + 1, image.Width - 1);
                double xWeight = Math.Clamp(sourceX - Math.Floor(sourceX), 0, 1);
                int pixelIndex = checked((targetY * targetWidth) + targetX);
                for (var channel = 0; channel < options.InputChannels; channel++)
                {
                    double top = SourceValue(image, x0, y0, channel, options.InputColorMode) * (1 - xWeight) +
                        SourceValue(image, x1, y0, channel, options.InputColorMode) * xWeight;
                    double bottom = SourceValue(image, x0, y1, channel, options.InputColorMode) * (1 - xWeight) +
                        SourceValue(image, x1, y1, channel, options.InputColorMode) * xWeight;
                    float sample = (float)((top * (1 - yWeight) + bottom * yWeight) / 255d);
                    int destination = options.InputLayout == OcrTensorLayout.ChannelsFirst
                        ? checked((channel * pixelsPerChannel) + pixelIndex)
                        : checked((pixelIndex * options.InputChannels) + channel);
                    values[destination] =
                        (sample - options.ChannelMeans[channel]) * options.ChannelScales[channel];
                }
            }
        }

        return values;
    }

    private static byte SourceValue(
        OcrImage image,
        int x,
        int y,
        int channel,
        OcrTensorColorMode colorMode) =>
        colorMode == OcrTensorColorMode.Bgr
            ? image.BgrPixels!.Pixels.Span[checked(
                (y * image.BgrPixels.Stride) + (x * 3) + channel)]
            : image.Pixels.Span[checked((y * image.Stride) + x)];

    private static (int Width, int Height) TensorDimensions(
        OcrImage image,
        LocalOnnxTextRegionDetectorOptions options)
    {
        int alignedMaximum = options.MaximumSideLength / options.DimensionMultiple * options.DimensionMultiple;
        double scale = Math.Min(1d, alignedMaximum / (double)Math.Max(image.Width, image.Height));
        int width = RoundAligned(image.Width * scale, options.DimensionMultiple, alignedMaximum);
        int height = RoundAligned(image.Height * scale, options.DimensionMultiple, alignedMaximum);
        return (width, height);
    }

    private static int RoundAligned(double value, int multiple, int maximum) =>
        Math.Clamp(
            checked((int)Math.Round(
                value / multiple,
                MidpointRounding.AwayFromZero) * multiple),
            multiple,
            maximum);

    private static float ToProbability(float value, OcrDetectionOutputActivation activation)
    {
        if (!float.IsFinite(value))
        {
            throw new InvalidDataException("OCR detection output contains a non-finite value.");
        }

        return activation switch
        {
            OcrDetectionOutputActivation.Probability when value is >= 0 and <= 1 => value,
            OcrDetectionOutputActivation.Probability => throw new InvalidDataException(
                "OCR detection probability output must remain within [0,1]."),
            OcrDetectionOutputActivation.SigmoidLogit =>
                (float)(1d / (1d + Math.Exp(-Math.Clamp(value, -80f, 80f)))),
            _ => throw new ArgumentOutOfRangeException(nameof(activation)),
        };
    }

    private static void ValidateImage(
        OcrImage image,
        LocalOnnxTextRegionDetectorOptions options)
    {
        if (image.Width <= 0 || image.Height <= 0 || image.Stride < image.Width ||
            image.Pixels.Length < checked(image.Stride * image.Height) ||
            !image.OriginalToImage.IsInvertible ||
            !string.Equals(image.CoordinateSpace, OcrContract.CoordinateSpace, StringComparison.Ordinal) ||
            image.CanonicalOriginalWidth is <= 0 || image.CanonicalOriginalHeight is <= 0 ||
            (options.InputColorMode == OcrTensorColorMode.Bgr && !HasValidBgrPlane(image)))
        {
            throw new ArgumentException("OCR detection image is invalid.", nameof(image));
        }
    }

    private static bool HasValidBgrPlane(OcrImage image) =>
        image.BgrPixels is { } bgr &&
        bgr.Stride >= checked(image.Width * 3) &&
        bgr.Pixels.Length == checked(bgr.Stride * image.Height);

    public static void ValidateOptions(LocalOnnxTextRegionDetectorOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Model.Validate();
        bool invalidProviders = options.AllowedProviders is not null &&
            (options.AllowedProviders.Count == 0 ||
             options.AllowedProviders.Any(static provider =>
                 provider is not (InferenceProvider.Cpu or InferenceProvider.DirectMl)) ||
             !options.AllowedProviders.Contains(InferenceProvider.Cpu) ||
             options.AllowedProviders.Distinct().Count() != options.AllowedProviders.Count);
        if (options.MaximumSideLength is < 32 or > 4096 ||
            options.DimensionMultiple is < 1 or > 128 ||
            options.DimensionMultiple > options.MaximumSideLength ||
            options.InputChannels is < 1 or > 4 || !Enum.IsDefined(options.InputLayout) ||
            !Enum.IsDefined(options.InputColorMode) ||
            (options.InputColorMode == OcrTensorColorMode.Bgr && options.InputChannels != 3) ||
            options.ChannelMeans.Count != options.InputChannels ||
            options.ChannelScales.Count != options.InputChannels ||
            options.ChannelMeans.Any(static value => !float.IsFinite(value)) ||
            options.ChannelScales.Any(static value => !float.IsFinite(value) || value == 0) ||
            string.IsNullOrWhiteSpace(options.InputName) || string.IsNullOrWhiteSpace(options.OutputName) ||
            string.IsNullOrWhiteSpace(options.StageVersion) || !Enum.IsDefined(options.OutputActivation) ||
            options.ProbabilityThreshold is < 0 or > 1 ||
            options.BoxConfidenceThreshold is < 0 or > 1 ||
            !double.IsFinite(options.UnclipRatio) || options.UnclipRatio is < 0 or > 10 ||
            options.MinimumComponentArea is < 1 or > 16_777_216 ||
            options.MinimumSideLength is < 1 or > 4096 ||
            options.MaximumRegions is < 1 or > 10_000 ||
            options.Timeout <= TimeSpan.Zero || options.Timeout > TimeSpan.FromMinutes(5) || invalidProviders)
        {
            throw new ArgumentException("Local ONNX OCR detector options are invalid.", nameof(options));
        }
    }

    private static string CreateConfigurationFingerprint(LocalOnnxTextRegionDetectorOptions options)
    {
        string material = string.Join('|',
            options.Model.Sha256.ToLowerInvariant(),
            options.MaximumSideLength.ToString(CultureInfo.InvariantCulture),
            options.DimensionMultiple.ToString(CultureInfo.InvariantCulture),
            options.InputChannels.ToString(CultureInfo.InvariantCulture),
            options.InputLayout,
            options.InputColorMode,
            string.Join(',', options.ChannelMeans.Select(static value => value.ToString("R", CultureInfo.InvariantCulture))),
            string.Join(',', options.ChannelScales.Select(static value => value.ToString("R", CultureInfo.InvariantCulture))),
            options.InputName,
            options.OutputName,
            options.StageVersion,
            options.OutputActivation,
            options.ProbabilityThreshold.ToString("R", CultureInfo.InvariantCulture),
            options.BoxConfidenceThreshold.ToString("R", CultureInfo.InvariantCulture),
            options.UnclipRatio.ToString("R", CultureInfo.InvariantCulture),
            options.MinimumComponentArea.ToString(CultureInfo.InvariantCulture),
            options.MinimumSideLength.ToString(CultureInfo.InvariantCulture),
            options.MaximumRegions.ToString(CultureInfo.InvariantCulture),
            ProviderFingerprint(options.AllowedProviders));
        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(material)));
    }

    private static string DeterministicRegionId(string modelSha256, OcrRectangle rectangle)
    {
        string material = FormattableString.Invariant(
            $"{modelSha256.ToLowerInvariant()}:{rectangle.X:R},{rectangle.Y:R},{rectangle.Width:R},{rectangle.Height:R}");
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

    private sealed record Component(
        int Left,
        int Top,
        int Right,
        int Bottom,
        int PixelCount,
        double Confidence)
    {
        public int Width => Right - Left + 1;

        public int Height => Bottom - Top + 1;
    }
}
