// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Security;
using System.Text.Json;
using GraphReader.App.Integration.Workflow;
using GraphReader.App.Services;
using GraphReader.Domain;
using GraphReader.Export;
using GraphReader.Imaging;
using GraphReader.Inference;
using GraphReader.Pdf;
using GraphReader.SuperResolution;

namespace GraphReader.App.Integration;

public sealed record ApplicationCompositionResult(
    WorkflowRuntimeEnvironment Environment,
    IWorkspaceService WorkspaceService,
    DomainError? StartupError,
    ProductionInferenceRuntimeHost? InferenceRuntimeHost = null,
    IProductionAxisGeometryAdapter? AxisGeometryAdapter = null,
    IProductionMarkerCenterAdapter? MarkerCenterAdapter = null,
    IProductionMarkerClassificationAdapter? MarkerClassificationAdapter = null,
    IProductionLegendReasoningAdapter? LegendReasoningAdapter = null,
    IProductionPhaseReasoningAdapter? PhaseReasoningAdapter = null,
    IProductionRasterFrameDecoder? RasterFrameDecoder = null);

public static class ApplicationComposition
{
    public const string RealEsrganManifestEnvironmentVariable = "GRAPHREADER_REALESRGAN_MANIFEST_PATH";
    public const string RealEsrganRuntimeRootEnvironmentVariable = "GRAPHREADER_REALESRGAN_RUNTIME_ROOT";
    public const string PdfiumApprovalEnvironmentVariable = "GRAPHREADER_PDFIUM_APPROVAL_PATH";

    public static ApplicationCompositionResult Create(
        WorkflowRuntimeEnvironment environment,
        IApplicationPaths? applicationPaths = null,
        string? applicationRoot = null)
        => CreateCore(
            environment,
            applicationPaths,
            applicationRoot ?? AppContext.BaseDirectory,
            environment == WorkflowRuntimeEnvironment.Production
                ? CapturedUiThreadGuard.CaptureCurrentThread()
                : null,
            modelAvailability: null,
            runtimeAvailability: null);

    public static async Task<ApplicationCompositionResult> CreateAsync(
        WorkflowRuntimeEnvironment environment,
        IApplicationPaths? applicationPaths = null,
        string? applicationRoot = null,
        CancellationToken cancellationToken = default)
    {
        IUiThreadGuard? uiThreadGuard = environment == WorkflowRuntimeEnvironment.Production
            ? CapturedUiThreadGuard.CaptureCurrentThread()
            : null;
        ProductionModelAvailabilitySnapshot? modelAvailability = environment == WorkflowRuntimeEnvironment.Production
            ? await ProductionModelAvailabilityProbe.InspectAsync(
                applicationPaths?.ModelRoot,
                cancellationToken)
                .ConfigureAwait(false)
            : null;
        ProductionRuntimeAvailabilitySnapshot? runtimeAvailability = environment == WorkflowRuntimeEnvironment.Production
            ? await ProductionRuntimeAvailabilityProbe.InspectAsync(
                applicationRoot ?? AppContext.BaseDirectory,
                cancellationToken).ConfigureAwait(false)
            : null;
        return CreateCore(
            environment,
            applicationPaths,
            applicationRoot ?? AppContext.BaseDirectory,
            uiThreadGuard,
            modelAvailability,
            runtimeAvailability);
    }

    private static ApplicationCompositionResult CreateCore(
        WorkflowRuntimeEnvironment environment,
        IApplicationPaths? applicationPaths,
        string applicationRoot,
        IUiThreadGuard? uiThreadGuard,
        ProductionModelAvailabilitySnapshot? modelAvailability,
        ProductionRuntimeAvailabilitySnapshot? runtimeAvailability)
    {
        Func<CancellationToken, Task<RealEsrganBackendResolution>>? enhancementResolver =
            CreateLocalEnhancementResolver(applicationPaths);
        (IPdfImportService? PdfImporter, DomainError? Error) pdf =
            CreateReviewedPdfImporter(applicationRoot);
        DomainResult<ProductionInferenceRuntimeHost>? inference =
            environment == WorkflowRuntimeEnvironment.Production && applicationPaths is not null
                ? ProductionInferenceRuntimeFactory.Create(
                    applicationPaths,
                    uiThreadGuard ?? CapturedUiThreadGuard.CaptureCurrentThread())
                : null;
        var rasterFrameDecoder = new ProductionRasterFrameDecoder();
        ProductionAxisGeometryAdapter? axisAdapter = CreateApprovedAxisAdapter(
            runtimeAvailability,
            rasterFrameDecoder);
        (ProductionMarkerCenterAdapter? Adapter, DomainError? Error) markerCenter =
            CreateApprovedMarkerCenterAdapter(modelAvailability, inference?.Value);
        (ProductionMarkerClassificationAdapter? Adapter, DomainError? Error) markerClassifier =
            CreateApprovedMarkerClassifierAdapter(modelAvailability, inference?.Value);
        var legendAdapter = new ProductionLegendReasoningAdapter();
        var phaseAdapter = new ProductionPhaseReasoningAdapter();
        var approvedAdapterStages = new List<string>();
        var adapterEvidence = new List<string>();
        if (axisAdapter is not null)
        {
            approvedAdapterStages.Add("axis");
            adapterEvidence.Add($"Concrete production axis adapter '{axisAdapter.AdapterId}' is composed.");
        }

        if (markerClassifier.Adapter is not null)
        {
            adapterEvidence.Add(
                $"Concrete production marker-classifier component '{markerClassifier.Adapter.AdapterId}' is composed; marker-center remains independently required.");
        }

        if (markerCenter.Adapter is not null)
        {
            adapterEvidence.Add(
                $"Concrete production marker-center component '{markerCenter.Adapter.AdapterId}' is composed; the complete marker workflow adapter remains independently required.");
        }

        approvedAdapterStages.Add("legends");
        approvedAdapterStages.Add("phases");
        adapterEvidence.Add(
            $"Deterministic production semantic adapters '{legendAdapter.AdapterId}' and '{phaseAdapter.AdapterId}' are composed and remain dependency-gated.");

        ProductionDetectionAdapterAvailabilitySnapshot? adapterAvailability = adapterEvidence.Count == 0
            ? null
            : new ProductionDetectionAdapterAvailabilitySnapshot(
                approvedAdapterStages,
                string.Join(' ', adapterEvidence));
        IReadOnlyList<AutomaticStageStatus> automaticStages =
            ProductionStageAvailabilityRegistry.Create(
                enhancementResolver is not null,
                modelAvailability,
                pdf.PdfImporter is not null,
                runtimeAvailability,
                inference?.Value is not null,
                adapterAvailability);
        return environment switch
        {
            WorkflowRuntimeEnvironment.Production => CreateProduction(
                environment,
                applicationPaths,
                enhancementResolver,
                automaticStages,
                pdf,
                inference,
                axisAdapter,
                markerCenter.Adapter,
                markerClassifier.Adapter,
                legendAdapter,
                phaseAdapter,
                rasterFrameDecoder,
                markerCenter.Error ?? markerClassifier.Error),
            WorkflowRuntimeEnvironment.ManualPreview => new ApplicationCompositionResult(
                environment,
                new ManualPreviewWorkspaceService(
                    applicationPaths,
                    automaticStages: automaticStages,
                    enhancementResolver: enhancementResolver),
                null),
            WorkflowRuntimeEnvironment.RecordedFake => new ApplicationCompositionResult(
                environment,
                new FakeWorkspaceService(),
                null),
            _ => throw new ArgumentOutOfRangeException(nameof(environment)),
        };
    }

    private static ApplicationCompositionResult CreateProduction(
        WorkflowRuntimeEnvironment environment,
        IApplicationPaths? applicationPaths,
        Func<CancellationToken, Task<RealEsrganBackendResolution>>? enhancementResolver,
        IReadOnlyList<AutomaticStageStatus> automaticStages,
        (IPdfImportService? PdfImporter, DomainError? Error) pdf,
        DomainResult<ProductionInferenceRuntimeHost>? inference,
        IProductionAxisGeometryAdapter? axisAdapter,
        IProductionMarkerCenterAdapter? markerCenterAdapter,
        IProductionMarkerClassificationAdapter? markerClassificationAdapter,
        IProductionLegendReasoningAdapter? legendReasoningAdapter,
        IProductionPhaseReasoningAdapter? phaseReasoningAdapter,
        IProductionRasterFrameDecoder rasterFrameDecoder,
        DomainError? markerAdapterError)
    {
        var imageImporter = new ImageImportService();
        var exportService = new ExportService();
        var panelStore = new ProductionWorkflowPanelStore();
        var services = new WorkflowServiceSet(
            new ProductionWorkflowImportStage(panelStore, imageImporter, pdf.PdfImporter),
            new ProductionWorkflowPrepareStage(panelStore),
            new ProductionWorkflowDetectionStage(panelStore),
            new ProductionWorkflowExportStage(panelStore, exportService));
        var orchestrator = new WorkflowOrchestrator(services);
        DomainError? inferenceError = inference is { Errors.Count: > 0 }
            ? inference.Errors[0]
            : null;
        return new ApplicationCompositionResult(
            environment,
            new ProductionWorkspaceService(
                applicationPaths,
                imageImporter,
                exportService: exportService,
                automaticStages: automaticStages,
                enhancementResolver: enhancementResolver,
                workflowOrchestrator: orchestrator,
                panelStore: panelStore),
            pdf.Error ?? inferenceError ?? markerAdapterError,
            inference?.Value,
            axisAdapter,
            markerCenterAdapter,
            markerClassificationAdapter,
            legendReasoningAdapter,
            phaseReasoningAdapter,
            rasterFrameDecoder);
    }

    private static ProductionAxisGeometryAdapter? CreateApprovedAxisAdapter(
        ProductionRuntimeAvailabilitySnapshot? runtimeAvailability,
        IProductionRasterFrameDecoder frameDecoder)
    {
        if (runtimeAvailability is not { AxisApproved: true } ||
            string.IsNullOrWhiteSpace(runtimeAvailability.RuntimeSha256))
        {
            return null;
        }

        return new ProductionAxisGeometryAdapter(
            runtimeAvailability.RuntimeSha256,
            isApproved: true,
            frameDecoder: frameDecoder);
    }

    private static (ProductionMarkerClassificationAdapter? Adapter, DomainError? Error)
        CreateApprovedMarkerClassifierAdapter(
            ProductionModelAvailabilitySnapshot? modelAvailability,
            ProductionInferenceRuntimeHost? runtimeHost)
    {
        if (runtimeHost is null ||
            modelAvailability is null ||
            !modelAvailability.ApprovedCpuModels.TryGetValue(
                "marker_classifier",
                out ResolvedProductionModel? model))
        {
            return (null, null);
        }

        try
        {
            return (ProductionMarkerClassificationAdapter.Create(model, runtimeHost), null);
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            return (
                null,
                new DomainError(
                    "MARKER_CLASSIFIER_ADAPTER_UNAVAILABLE",
                    DomainErrorSeverity.Warning,
                    "Errors.ProductionWorkflowUnavailable",
                    $"The checksum-resolved marker classifier could not be composed: {exception.Message}",
                    Recoverable: true,
                    "continue_manual_or_repair_model_store"));
        }
    }

    private static (ProductionMarkerCenterAdapter? Adapter, DomainError? Error)
        CreateApprovedMarkerCenterAdapter(
            ProductionModelAvailabilitySnapshot? modelAvailability,
            ProductionInferenceRuntimeHost? runtimeHost)
    {
        if (runtimeHost is null ||
            modelAvailability is null ||
            !modelAvailability.ApprovedCpuModels.TryGetValue(
                "marker_center",
                out ResolvedProductionModel? model))
        {
            return (null, null);
        }

        try
        {
            return (ProductionMarkerCenterAdapter.Create(model, runtimeHost), null);
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            return (
                null,
                new DomainError(
                    "MARKER_CENTER_ADAPTER_UNAVAILABLE",
                    DomainErrorSeverity.Warning,
                    "Errors.ProductionWorkflowUnavailable",
                    $"The checksum-resolved marker-center model could not be composed: {exception.Message}",
                    Recoverable: true,
                    "continue_manual_or_repair_model_store"));
        }
    }

    private static Func<CancellationToken, Task<RealEsrganBackendResolution>>? CreateLocalEnhancementResolver(
        IApplicationPaths? applicationPaths)
    {
        string? manifestPath = Environment.GetEnvironmentVariable(RealEsrganManifestEnvironmentVariable);
        string? runtimeRoot = Environment.GetEnvironmentVariable(RealEsrganRuntimeRootEnvironmentVariable);
        if (applicationPaths is null ||
            string.IsNullOrWhiteSpace(manifestPath) ||
            string.IsNullOrWhiteSpace(runtimeRoot))
        {
            return null;
        }

        string cacheRoot = Path.Combine(applicationPaths.CacheRoot, "RealESRGAN");
        return cancellationToken => ManifestDrivenRealEsrganBackend.ResolveFromRuntimeRootAsync(
            manifestPath,
            runtimeRoot,
            cacheRoot,
            RealEsrganBackendPurpose.LocalEvaluation,
            cancellationToken);
    }

    private static (IPdfImportService? PdfImporter, DomainError? Error) CreateReviewedPdfImporter(
        string applicationRoot)
    {
        string? approvalPath = Environment.GetEnvironmentVariable(PdfiumApprovalEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(approvalPath))
        {
            string packagedApprovalPath = Path.Combine(
                Path.GetFullPath(applicationRoot),
                "pdfium",
                "reviewed-approval.json");
            if (!File.Exists(packagedApprovalPath))
            {
                return (null, null);
            }

            approvalPath = packagedApprovalPath;
        }

        try
        {
            ReviewedPdfiumPageRendererBackend backend = ReviewedPdfiumPageRendererBackend.Load(approvalPath);
            return (
                new PdfImportService(
                    new PdfPigDocumentInspector(),
                    new PanelizationEngine(),
                    backend.CreateRenderingService()),
                null);
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or
            UnauthorizedAccessException or SecurityException or JsonException)
        {
            return (
                null,
                new DomainError(
                    "PDFIUM_APPROVAL_REJECTED",
                    DomainErrorSeverity.Warning,
                    "Errors.PdfRendererUnavailable",
                    exception.Message,
                    Recoverable: true,
                    "restore_reviewed_pdfium_approval"));
        }
    }
}
