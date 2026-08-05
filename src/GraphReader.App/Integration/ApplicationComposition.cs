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
    IProductionAxisGeometryAdapter? AxisGeometryAdapter = null);

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
        ProductionAxisGeometryAdapter? axisAdapter = CreateApprovedAxisAdapter(runtimeAvailability);
        ProductionDetectionAdapterAvailabilitySnapshot? adapterAvailability = axisAdapter is null
            ? null
            : new ProductionDetectionAdapterAvailabilitySnapshot(
                ["axis"],
                $"Concrete production axis adapter '{axisAdapter.AdapterId}' is composed.");
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
                axisAdapter),
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
        IProductionAxisGeometryAdapter? axisAdapter)
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
            pdf.Error ?? inferenceError,
            inference?.Value,
            axisAdapter);
    }

    private static ProductionAxisGeometryAdapter? CreateApprovedAxisAdapter(
        ProductionRuntimeAvailabilitySnapshot? runtimeAvailability)
    {
        if (runtimeAvailability is not { AxisApproved: true } ||
            string.IsNullOrWhiteSpace(runtimeAvailability.RuntimeSha256))
        {
            return null;
        }

        return new ProductionAxisGeometryAdapter(
            runtimeAvailability.RuntimeSha256,
            isApproved: true);
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
