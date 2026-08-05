// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

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
    private const string CombinedTimingWarning = "ocr_pipeline_timing_not_model_isolated";
    private readonly OcrPipeline pipeline;
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
    {
        this.pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
        this.detectionModel = ValidateModel(detectionModel, nameof(detectionModel));
        this.recognitionModel = ValidateModel(recognitionModel, nameof(recognitionModel));
        this.detectionProvider = ValidateProvider(detectionProvider, nameof(detectionProvider));
        this.recognitionProvider = ValidateProvider(recognitionProvider, nameof(recognitionProvider));
        IsApproved = isApproved;
    }

    public string AdapterId =>
        $"graphreader-ocr:{detectionModel.Sha256[..12].ToLowerInvariant()}:{recognitionModel.Sha256[..12].ToLowerInvariant()}";

    public bool IsApproved { get; }

    public async Task<ProductionOcrEvidence> RecognizeAsync(
        ProductionWorkflowDetectionRequest request,
        ProductionDecodedRaster originalRaster,
        OcrRectangle plotBounds,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(originalRaster);
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
            TransformChain: "identity");

        OcrResult result = await pipeline
            .RecognizeAsync(ocrRequest, cancellationToken)
            .ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        ValidateOutput(request, originalRaster, result);

        IReadOnlyList<string> envelopeWarnings = result.Warnings
            .Append(CombinedTimingWarning)
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        WorkflowVisionTiming combinedTiming = new(
            result.Timing.PreprocessMilliseconds,
            result.Timing.InferenceMilliseconds,
            result.Timing.PostprocessMilliseconds,
            result.Timing.TotalMilliseconds);
        ProductionOcrModelEvidence[] models =
        [
            new(
                "ocr_detection",
                CreateEnvelope(
                    request,
                    result,
                    detectionModel,
                    detectionProvider,
                    combinedTiming,
                    envelopeWarnings)),
            new(
                "ocr_recognition",
                CreateEnvelope(
                    request,
                    result,
                    recognitionModel,
                    recognitionProvider,
                    combinedTiming,
                    envelopeWarnings)),
        ];

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
