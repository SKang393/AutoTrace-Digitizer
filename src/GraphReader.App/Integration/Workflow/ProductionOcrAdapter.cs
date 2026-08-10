// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using GraphReader.Inference;
using GraphReader.Ocr;

namespace GraphReader.App.Integration.Workflow;

public interface IProductionOcrAdapter
{
    string AdapterId { get; }

    bool IsApproved { get; }

    Task<ProductionOcrEvidence> RecognizeAsync(
        ProductionWorkflowDetectionRequest request,
        ProductionDecodedRaster originalRaster,
        OcrRectangle plotBounds,
        OcrDetectorImage detectorImage,
        CancellationToken cancellationToken);
}

public sealed class ProductionOcrEvidence
{
    internal ProductionOcrEvidence(
        OcrResult result,
        IEnumerable<ProductionOcrModelEvidence> modelEvidence)
    {
        Result = result ?? throw new ArgumentNullException(nameof(result));
        ModelEvidence = Array.AsReadOnly(modelEvidence.ToArray());
    }

    public OcrResult Result { get; }

    public IReadOnlyList<ProductionOcrModelEvidence> ModelEvidence { get; }
}

/// <summary>
/// Binds the existing OCR pipeline to exact detector and recognizer identities.
/// Composition remains disabled until both payloads have independently passed
/// the production model-store, benchmark, provider, notice, and checksum gates.
/// </summary>
public sealed class ProductionOcrAdapter : IProductionOcrAdapter
{
    public const string ApprovalBenchmarkProfile = "graphreader-ocr-public-gate-v1";
    private const string CombinedTimingWarning = "ocr_pipeline_timing_not_model_isolated";
    private readonly Lazy<OcrPipeline> pipeline;
    private readonly ModelIdentity detectionModel;
    private readonly ModelIdentity recognitionModel;
    private readonly InferenceProvider detectionProvider;
    private readonly InferenceProvider recognitionProvider;

    public ProductionOcrAdapter(
        OcrPipeline pipeline,
        ModelIdentity detectionModel,
        InferenceProvider detectionProvider,
        ModelIdentity recognitionModel,
        InferenceProvider recognitionProvider,
        bool isApproved)
        : this(
            () => pipeline ?? throw new ArgumentNullException(nameof(pipeline)),
            detectionModel,
            detectionProvider,
            recognitionModel,
            recognitionProvider,
            isApproved)
    {
    }

    private ProductionOcrAdapter(
        Func<OcrPipeline> pipelineFactory,
        ModelIdentity detectionModel,
        InferenceProvider detectionProvider,
        ModelIdentity recognitionModel,
        InferenceProvider recognitionProvider,
        bool isApproved)
    {
        ArgumentNullException.ThrowIfNull(pipelineFactory);
        pipeline = new Lazy<OcrPipeline>(
            pipelineFactory,
            LazyThreadSafetyMode.ExecutionAndPublication);
        this.detectionModel = ValidateModel(detectionModel, nameof(detectionModel));
        this.recognitionModel = ValidateModel(recognitionModel, nameof(recognitionModel));
        this.detectionProvider = ValidateProvider(detectionProvider, nameof(detectionProvider));
        this.recognitionProvider = ValidateProvider(recognitionProvider, nameof(recognitionProvider));
        IsApproved = isApproved;
    }

    public string AdapterId =>
        $"graphreader-ocr:{detectionModel.Sha256[..12].ToLowerInvariant()}:{recognitionModel.Sha256[..12].ToLowerInvariant()}";

    public bool IsApproved { get; }

    public static async Task<ProductionOcrAdapter> CreateAsync(
        ResolvedProductionModel detectionModel,
        ResolvedProductionModel recognitionModel,
        ProductionInferenceRuntimeHost runtimeHost,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(detectionModel);
        ArgumentNullException.ThrowIfNull(recognitionModel);
        ArgumentNullException.ThrowIfNull(runtimeHost);
        RequireTask(detectionModel, "ocr_detection");
        RequireTask(recognitionModel, "ocr_recognition");
        RequireCpu(detectionModel);
        RequireCpu(recognitionModel);
        VerifyChecksum(
            detectionModel.ManifestPath,
            detectionModel.ManifestSha256,
            "OCR detection manifest");
        VerifyChecksum(
            recognitionModel.ManifestPath,
            recognitionModel.ManifestSha256,
            "OCR recognition manifest");
        ProductionOcrApprovalGate.Validate(detectionModel, recognitionModel);

        LocalOnnxTextRegionDetectorOptions detectorOptions = ReadDetectionOptions(detectionModel);
        (LocalOnnxTextRecognizerOptions Recognizer, OcrPipelineOptions Pipeline) recognition =
            ReadRecognitionOptions(recognitionModel);
        await ValidateExecutablePairAsync(
                detectorOptions,
                recognition.Recognizer,
                runtimeHost.Runtime,
                cancellationToken)
            .ConfigureAwait(false);
        return new ProductionOcrAdapter(
            () =>
            {
                InferenceRuntime runtime = runtimeHost.Runtime;
                var detector = new LocalOnnxTextRegionDetector(runtime, detectorOptions);
                var recognizer = new LocalOnnxTextRecognizer(runtime, recognition.Recognizer);
                return new OcrPipeline(
                    detector,
                    recognizer,
                    new MemoryOcrResultCache(),
                    recognition.Pipeline);
            },
            detectionModel.Identity,
            InferenceProvider.Cpu,
            recognitionModel.Identity,
            InferenceProvider.Cpu,
            isApproved: true);
    }

    private static Task ValidateExecutablePairAsync(
        LocalOnnxTextRegionDetectorOptions detectorOptions,
        LocalOnnxTextRecognizerOptions recognizerOptions,
        InferenceRuntime runtime,
        CancellationToken cancellationToken) =>
        Task.Run(async () =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            var detector = new LocalOnnxTextRegionDetector(
                runtime,
                detectorOptions with { BypassCache = true });
            const int detectorProbeSize = 32;
            var detectorImage = new OcrImage(
                detectorProbeSize,
                detectorProbeSize,
                detectorProbeSize,
                new byte[detectorProbeSize * detectorProbeSize],
                OcrSourceImage.Original,
                OcrFrameTransform.Identity,
                CanonicalOriginalWidth: detectorProbeSize,
                CanonicalOriginalHeight: detectorProbeSize);
            _ = await detector.DetectAsync(detectorImage, cancellationToken).ConfigureAwait(false);

            var recognizer = new LocalOnnxTextRecognizer(
                runtime,
                recognizerOptions with { BypassCache = true });
            var cropPixels = new float[checked(recognizerOptions.InputWidth * recognizerOptions.InputHeight)];
            string cropSha256 = Convert.ToHexStringLower(
                SHA256.HashData(new byte[checked(cropPixels.Length * sizeof(float))]));
            OcrPolygon polygon = OcrPolygon.FromRectangle(new OcrRectangle(
                0,
                0,
                recognizerOptions.InputWidth,
                recognizerOptions.InputHeight));
            OcrCrop[] crops =
            [
                new("probe-1", OcrSourceImage.Original, recognizerOptions.InputWidth,
                    recognizerOptions.InputHeight, cropPixels, cropSha256, polygon),
                new("probe-2", OcrSourceImage.Original, recognizerOptions.InputWidth,
                    recognizerOptions.InputHeight, cropPixels, cropSha256, polygon),
            ];
            IReadOnlyList<OcrRecognition> results = await recognizer
                .RecognizeBatchAsync(crops, cancellationToken)
                .ConfigureAwait(false);
            if (results.Count != crops.Length || results.Any(static result => result.Failure is not null))
            {
                string failures = string.Join(
                    ", ",
                    results.Where(static result => result.Failure is not null)
                        .Select(static result =>
                            $"{result.Failure!.Code}: {result.Failure.TechnicalMessage}"));
                throw new InvalidDataException(
                    $"OCR recognition payload failed the two-item CPU executable probe: {failures}.");
            }
        }, cancellationToken);

    public async Task<ProductionOcrEvidence> RecognizeAsync(
        ProductionWorkflowDetectionRequest request,
        ProductionDecodedRaster originalRaster,
        OcrRectangle plotBounds,
        OcrDetectorImage detectorImage,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(originalRaster);
        ArgumentNullException.ThrowIfNull(detectorImage);
        cancellationToken.ThrowIfCancellationRequested();
        if (!IsApproved)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionModelsUnavailable,
                "Errors.ModelNotFound",
                $"OCR adapter '{AdapterId}' does not have two independently approved production payloads.",
                "Install checksum-verified approved OCR detection and recognition models or continue in manual mode.");
        }

        ValidateInput(request, originalRaster, plotBounds);
        var ocrRequest = new OcrRequest(
            request.ProjectId.ToString("D"),
            request.Panel.ImportedPanel.PanelId.ToString("D"),
            request.Image.Sha256,
            originalRaster.CreateOcrImage(),
            plotBounds,
            EnhancedImage: null,
            DetectedRegions: null,
            OcrContract.Version,
            TransformChain: "identity",
            DetectorImage: detectorImage);

        OcrResult result = await pipeline.Value
            .RecognizeAsync(ocrRequest, cancellationToken)
            .ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        ValidateOutput(request, originalRaster, result);

        string detectorInputWarning = $"ocr_detector_input_sha256:{detectorImage.PixelSha256.ToLowerInvariant()}";
        result = result with
        {
            Warnings = Array.AsReadOnly(result.Warnings
                .Append("ocr_detector_axis_geometry_mask_applied")
                .Append(detectorInputWarning)
                .Distinct(StringComparer.Ordinal)
                .ToArray()),
        };

        IReadOnlyList<string> envelopeWarnings = result.Warnings
            .Append(CombinedTimingWarning)
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        WorkflowVisionTiming combinedTiming = new(
            result.Timing.PreprocessMilliseconds,
            result.Timing.InferenceMilliseconds,
            result.Timing.PostprocessMilliseconds,
            result.Timing.TotalMilliseconds);
        var models = new List<ProductionOcrModelEvidence>(capacity: 2);
        bool detectionCompleted = !string.Equals(
            result.Failure?.Code,
            "OCR_REGION_DETECTION_FAILED",
            StringComparison.Ordinal);
        if (detectionCompleted)
        {
            models.Add(new(
                "ocr_detection",
                CreateEnvelope(
                    request,
                    result,
                    detectionModel,
                    detectionProvider,
                    combinedTiming,
                    envelopeWarnings)));
        }

        if (result.Succeeded && result.Cache.CropCount > 0)
        {
            models.Add(new(
                "ocr_recognition",
                CreateEnvelope(
                    request,
                    result,
                    recognitionModel,
                    recognitionProvider,
                    combinedTiming,
                    envelopeWarnings)));
        }

        if (!result.Succeeded)
        {
            OcrFailure failure = result.Failure!;
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                failure.UserMessageKey,
                $"OCR pipeline failed with '{failure.Code}': {failure.TechnicalMessage}",
                failure.SuggestedAction,
                models.Select(static model => model.Envelope));
        }

        return new ProductionOcrEvidence(result, models);
    }

    private static WorkflowVisionEnvelope CreateEnvelope(
        ProductionWorkflowDetectionRequest request,
        OcrResult result,
        ModelIdentity model,
        InferenceProvider provider,
        WorkflowVisionTiming timing,
        IReadOnlyList<string> warnings) =>
        new(
            contractVersion: 1,
            request.RunId,
            request.ProjectId,
            request.Panel.ImportedPanel.PanelId,
            OcrContract.Stage,
            result.StageVersion,
            request.Image.Sha256,
            new WorkflowVisionModel(
                model.ModelId,
                model.Version,
                model.Sha256.ToLowerInvariant(),
                ProviderName(provider)),
            timing,
            result.Confidence,
            warnings,
            request.Transforms,
            OcrContract.CoordinateSpace);

    private static void ValidateInput(
        ProductionWorkflowDetectionRequest request,
        ProductionDecodedRaster originalRaster,
        OcrRectangle plotBounds)
    {
        bool invalidPlot = !plotBounds.IsValid ||
            plotBounds.Left < 0 ||
            plotBounds.Top < 0 ||
            plotBounds.Right > originalRaster.Width ||
            plotBounds.Bottom > originalRaster.Height;
        if (request.ImageVariant != WorkflowImageVariant.Original ||
            request.Image.Variant != WorkflowImageVariant.Original ||
            originalRaster.Variant != WorkflowImageVariant.Original ||
            originalRaster.OriginalToFrame != GraphReader.Markers.Detection.MarkerAffineTransform.Identity ||
            originalRaster.Width != request.Image.Width ||
            originalRaster.Height != request.Image.Height ||
            !string.Equals(
                originalRaster.InputSha256,
                request.Image.Sha256,
                StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(
                request.Panel.Original.Sha256,
                request.Image.Sha256,
                StringComparison.OrdinalIgnoreCase) ||
            invalidPlot)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "Production OCR requires the checksum-matched immutable original raster and plot bounds inside that raster.",
                "Decode the current original image and recompute plot geometry before OCR.");
        }
    }

    private static void ValidateOutput(
        ProductionWorkflowDetectionRequest request,
        ProductionDecodedRaster raster,
        OcrResult result)
    {
        bool identityMismatch = result.ContractVersion != OcrContract.Version ||
            !Guid.TryParse(result.ProjectId, out Guid resultProjectId) ||
            resultProjectId != request.ProjectId ||
            !Guid.TryParse(result.PanelId, out Guid resultPanelId) ||
            resultPanelId != request.Panel.ImportedPanel.PanelId ||
            !string.Equals(result.Stage, OcrContract.Stage, StringComparison.Ordinal) ||
            string.IsNullOrWhiteSpace(result.StageVersion) ||
            !string.Equals(result.InputSha256, request.Image.Sha256, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(result.CoordinateSpace, OcrContract.CoordinateSpace, StringComparison.Ordinal) ||
            result.Regions.Any(region =>
                !string.Equals(region.CoordinateSpace, OcrContract.CoordinateSpace, StringComparison.Ordinal) ||
                !PolygonIsBounded(region.Polygon, raster.Width, raster.Height)) ||
            result.Masks.Any(mask =>
                !string.Equals(mask.CoordinateSpace, OcrContract.CoordinateSpace, StringComparison.Ordinal) ||
                !PolygonIsBounded(mask.Polygon, raster.Width, raster.Height));
        if (identityMismatch)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "The OCR pipeline returned evidence for a different project, panel, image, contract, or coordinate space.",
                "Reject the result and rerun the checksum-bound OCR pipeline.");
        }
    }

    private static bool PolygonIsBounded(OcrPolygon polygon, int width, int height) =>
        polygon.Points.All(point =>
            point.IsFinite &&
            point.X >= 0 && point.X <= width &&
            point.Y >= 0 && point.Y <= height);

    private static ModelIdentity ValidateModel(ModelIdentity model, string parameterName)
    {
        ArgumentNullException.ThrowIfNull(model, parameterName);
        model.Validate();
        return model;
    }

    private static InferenceProvider ValidateProvider(InferenceProvider provider, string parameterName) =>
        provider is InferenceProvider.Cpu or InferenceProvider.DirectMl
            ? provider
            : throw new ArgumentOutOfRangeException(
                parameterName,
                provider,
                "Production OCR supports only CPU or DirectML execution evidence.");

    private static string ProviderName(InferenceProvider provider) => provider switch
    {
        InferenceProvider.Cpu => "cpu",
        InferenceProvider.DirectMl => "directml",
        _ => throw new ArgumentOutOfRangeException(nameof(provider)),
    };

    private static LocalOnnxTextRegionDetectorOptions ReadDetectionOptions(
        ResolvedProductionModel resolvedModel)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(resolvedModel.ManifestPath));
        JsonElement root = document.RootElement;
        JsonElement input = SingleObject(root, "inputs", "OCR detection");
        JsonElement output = SingleObject(root, "outputs", "OCR detection");
        RequireString(input, "element_type", "float32", "OCR detection input");
        RequireString(input, "layout", "NCHW", "OCR detection input");
        RequireShape(input, "shape", ["1", "3", "H", "W"], "OCR detection input");
        RequireStringArray(
            input,
            "channels",
            ["grayscale", "grayscale", "grayscale"],
            "OCR detection input");
        RequireString(output, "element_type", "float32", "OCR detection output");
        RequireString(output, "layout", "NCHW", "OCR detection output");
        RequireShape(output, "shape", ["1", "1", "H", "W"], "OCR detection output");
        RequireStringArray(output, "channels", ["text_probability"], "OCR detection output");
        string activation = RequiredString(output, "activation", "OCR detection output");
        OcrDetectionOutputActivation outputActivation = activation switch
        {
            "probability" => OcrDetectionOutputActivation.Probability,
            "sigmoid_logit" => OcrDetectionOutputActivation.SigmoidLogit,
            _ => throw new InvalidDataException(
                "OCR detection output activation must be probability or sigmoid_logit."),
        };

        JsonElement preprocessing = RequiredObject(root, "preprocessing", "OCR detection manifest");
        float[] means = RequiredSingles(preprocessing, "channel_means", 3, "OCR detection preprocessing");
        float[] scales = RequiredSingles(preprocessing, "channel_scales", 3, "OCR detection preprocessing");
        JsonElement postprocessing = RequiredObject(root, "postprocessing", "OCR detection manifest");
        RequireString(
            postprocessing,
            "algorithm",
            "dense_probability_components_v1",
            "OCR detection postprocessing");
        var options = new LocalOnnxTextRegionDetectorOptions(resolvedModel.Identity)
        {
            MaximumSideLength = RequiredInt32(
                preprocessing,
                "maximum_side_length",
                "OCR detection preprocessing"),
            DimensionMultiple = RequiredInt32(
                preprocessing,
                "dimension_multiple",
                "OCR detection preprocessing"),
            InputChannels = 3,
            InputLayout = OcrTensorLayout.ChannelsFirst,
            ChannelMeans = means,
            ChannelScales = scales,
            InputName = RequiredString(input, "name", "OCR detection input"),
            OutputName = RequiredString(output, "name", "OCR detection output"),
            StageVersion = resolvedModel.Identity.Version,
            OutputActivation = outputActivation,
            ProbabilityThreshold = RequiredSingle(
                postprocessing,
                "probability_threshold",
                "OCR detection postprocessing"),
            BoxConfidenceThreshold = RequiredSingle(
                postprocessing,
                "box_confidence_threshold",
                "OCR detection postprocessing"),
            UnclipRatio = RequiredDouble(
                postprocessing,
                "unclip_ratio",
                "OCR detection postprocessing"),
            MinimumComponentArea = RequiredInt32(
                postprocessing,
                "minimum_component_area",
                "OCR detection postprocessing"),
            MinimumSideLength = RequiredInt32(
                postprocessing,
                "minimum_side_length",
                "OCR detection postprocessing"),
            MaximumRegions = RequiredInt32(
                postprocessing,
                "maximum_regions",
                "OCR detection postprocessing"),
            AllowedProviders = [InferenceProvider.Cpu],
        };
        LocalOnnxTextRegionDetector.ValidateOptions(options);
        return options;
    }

    private static (LocalOnnxTextRecognizerOptions Recognizer, OcrPipelineOptions Pipeline)
        ReadRecognitionOptions(ResolvedProductionModel resolvedModel)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(resolvedModel.ManifestPath));
        JsonElement root = document.RootElement;
        JsonElement input = SingleObject(root, "inputs", "OCR recognition");
        JsonElement output = SingleObject(root, "outputs", "OCR recognition");
        RequireString(input, "element_type", "float32", "OCR recognition input");
        RequireString(input, "layout", "NCHW", "OCR recognition input");
        JsonElement inputShape = RequiredArray(input, "shape", "OCR recognition input");
        if (inputShape.GetArrayLength() != 4 ||
            !StringValueEquals(inputShape[0], "N") ||
            !TryReadInt32(inputShape[1], out int channels) || channels != 3 ||
            !TryReadInt32(inputShape[2], out int inputHeight) || inputHeight != 48 ||
            !TryReadInt32(inputShape[3], out int inputWidth) || inputWidth != 320)
        {
            throw new InvalidDataException(
                "OCR recognition input shape must be the reviewed [N,3,48,320] contract.");
        }

        RequireStringArray(
            input,
            "channels",
            ["grayscale", "grayscale", "grayscale"],
            "OCR recognition input");
        RequireString(output, "element_type", "float32", "OCR recognition output");
        string outputLayout = RequiredString(output, "layout", "OCR recognition output");
        OcrOutputLayout runtimeOutputLayout = outputLayout switch
        {
            "NTC" => OcrOutputLayout.BatchTimeClass,
            "TNC" => OcrOutputLayout.TimeBatchClass,
            _ => throw new InvalidDataException("OCR recognition output layout must be NTC or TNC."),
        };
        RequireShape(
            output,
            "shape",
            runtimeOutputLayout == OcrOutputLayout.BatchTimeClass
                ? ["N", "T", "C"]
                : ["T", "N", "C"],
            "OCR recognition output");
        string alphabet = RequiredString(output, "alphabet", "OCR recognition output");
        int expectedTimeSteps = RequiredInt32(output, "time_steps", "OCR recognition output");
        int blankClassIndex = RequiredInt32(output, "blank_class_index", "OCR recognition output");
        JsonElement preprocessing = RequiredObject(root, "preprocessing", "OCR recognition manifest");
        float[] means = RequiredSingles(preprocessing, "channel_means", 3, "OCR recognition preprocessing");
        float[] scales = RequiredSingles(preprocessing, "channel_scales", 3, "OCR recognition preprocessing");
        JsonElement postprocessing = RequiredObject(root, "postprocessing", "OCR recognition manifest");
        RequireString(
            postprocessing,
            "algorithm",
            "ctc_greedy_alternatives_v1",
            "OCR recognition postprocessing");
        int maximumAlternatives = RequiredInt32(
            postprocessing,
            "maximum_alternatives",
            "OCR recognition postprocessing");
        var recognizer = new LocalOnnxTextRecognizerOptions(resolvedModel.Identity, alphabet)
        {
            InputWidth = inputWidth,
            InputHeight = inputHeight,
            InputChannels = channels,
            InputLayout = OcrTensorLayout.ChannelsFirst,
            OutputLayout = runtimeOutputLayout,
            ExpectedTimeSteps = expectedTimeSteps,
            BlankClassIndex = blankClassIndex,
            MaximumAlternatives = maximumAlternatives,
            InputName = RequiredString(input, "name", "OCR recognition input"),
            OutputName = RequiredString(output, "name", "OCR recognition output"),
            StageVersion = resolvedModel.Identity.Version,
            ChannelMeans = means,
            ChannelScales = scales,
            AllowedProviders = [InferenceProvider.Cpu],
        };
        var pipeline = new OcrPipelineOptions
        {
            StageVersion = resolvedModel.Identity.Version,
            CropWidth = inputWidth,
            CropHeight = inputHeight,
        };
        LocalOnnxTextRecognizer.ValidateOptions(recognizer);
        return (recognizer, pipeline);
    }

    private static void RequireTask(ResolvedProductionModel model, string expected)
    {
        if (!string.Equals(model.Task, expected, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Resolved model task '{model.Task}' is not {expected}.");
        }
    }

    private static void RequireCpu(ResolvedProductionModel model)
    {
        if (!model.AvailableProviders.Contains(InferenceProvider.Cpu))
        {
            throw new InvalidDataException(
                $"Resolved OCR model '{model.Identity.ModelId}' lacks mandatory CPU approval.");
        }
    }

    private static JsonElement SingleObject(JsonElement root, string propertyName, string label)
    {
        JsonElement values = RequiredArray(root, propertyName, label);
        if (values.GetArrayLength() != 1 || values[0].ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"{label} '{propertyName}' must contain exactly one object.");
        }

        return values[0];
    }

    private static JsonElement RequiredObject(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be an object.");
        }

        return value;
    }

    private static JsonElement RequiredArray(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be an array.");
        }

        return value;
    }

    private static string RequiredString(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(value.GetString()))
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be a non-empty string.");
        }

        return value.GetString()!;
    }

    private static void RequireString(
        JsonElement parent,
        string propertyName,
        string expected,
        string label)
    {
        string actual = RequiredString(parent, propertyName, label);
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"{label} field '{propertyName}' must be '{expected}', found '{actual}'.");
        }
    }

    private static void RequireStringArray(
        JsonElement parent,
        string propertyName,
        string[] expected,
        string label)
    {
        JsonElement values = RequiredArray(parent, propertyName, label);
        string?[] actual = values.EnumerateArray()
            .Select(static value => value.ValueKind == JsonValueKind.String ? value.GetString() : null)
            .ToArray();
        if (!actual.SequenceEqual(expected, StringComparer.Ordinal))
        {
            throw new InvalidDataException(
                $"{label} field '{propertyName}' does not match the frozen order.");
        }
    }

    private static void RequireShape(
        JsonElement parent,
        string propertyName,
        string[] expected,
        string label)
    {
        JsonElement values = RequiredArray(parent, propertyName, label);
        string?[] actual = values.EnumerateArray().Select(static value => value.ValueKind switch
        {
            JsonValueKind.String => value.GetString(),
            JsonValueKind.Number when value.TryGetInt32(out int integer) =>
                integer.ToString(System.Globalization.CultureInfo.InvariantCulture),
            _ => null,
        }).ToArray();
        if (!actual.SequenceEqual(expected, StringComparer.Ordinal))
        {
            throw new InvalidDataException(
                $"{label} field '{propertyName}' does not match [{string.Join(',', expected)}].");
        }
    }

    private static float[] RequiredSingles(
        JsonElement parent,
        string propertyName,
        int count,
        string label)
    {
        JsonElement values = RequiredArray(parent, propertyName, label);
        float[] result;
        try
        {
            result = values.EnumerateArray().Select(static value => value.GetSingle()).ToArray();
        }
        catch (Exception exception) when (exception is InvalidOperationException or FormatException)
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must contain float32 values.", exception);
        }

        if (result.Length != count || result.Any(static value => !float.IsFinite(value)))
        {
            throw new InvalidDataException(
                $"{label} field '{propertyName}' must contain {count} finite values.");
        }

        return result;
    }

    private static float RequiredSingle(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Number || !value.TryGetSingle(out float result) ||
            !float.IsFinite(result))
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be finite float32.");
        }

        return result;
    }

    private static double RequiredDouble(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Number || !value.TryGetDouble(out double result) ||
            !double.IsFinite(result))
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be finite.");
        }

        return result;
    }

    private static int RequiredInt32(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            !TryReadInt32(value, out int result))
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be int32.");
        }

        return result;
    }

    private static bool TryReadInt32(JsonElement value, out int result)
    {
        result = 0;
        return value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out result);
    }

    private static bool StringValueEquals(JsonElement value, string expected) =>
        value.ValueKind == JsonValueKind.String &&
        string.Equals(value.GetString(), expected, StringComparison.Ordinal);

    private static void VerifyChecksum(string path, string expectedSha256, string label)
    {
        if (!File.Exists(path))
        {
            throw new InvalidDataException($"The checksum-resolved {label} is missing: {path}");
        }

        string actual = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
        if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"The checksum-resolved {label} changed after model-store validation.");
        }
    }

    private static ProductionWorkflowStageException Failure(
        string code,
        string userMessageKey,
        string technicalMessage,
        string suggestedAction,
        IEnumerable<WorkflowVisionEnvelope>? completedEvidence = null) =>
        new(
            new ProductionWorkflowFailure(
                code,
                userMessageKey,
                technicalMessage,
                Recoverable: true,
                suggestedAction),
            completedEvidence);
}
