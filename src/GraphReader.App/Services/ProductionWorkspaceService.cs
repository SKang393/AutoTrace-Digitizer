// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration.Workflow;
using GraphReader.App.ViewModels;
using GraphReader.Domain;
using GraphReader.Export;
using GraphReader.Imaging;
using GraphReader.Pdf;
using GraphReader.Phases;
using GraphReader.SuperResolution;

namespace GraphReader.App.Services;

/// <summary>
/// Real-data production workspace. Automatic detection remains fail-closed
/// until composition supplies a workflow backed by approved runtime assets.
/// </summary>
public sealed class ProductionWorkspaceService : ManualPreviewWorkspaceService, IAutomaticWorkspaceService
{
    private static readonly string[] RequiredDetectionStages =
    [
        "axis",
        "ocr",
        "markers",
        "legends",
        "phases",
    ];

    private readonly Func<WorkflowReviewState?, CancellationToken, Task<WorkflowRunResult>>? _automaticWorkflow;
    private readonly WorkflowOrchestrator? _workflowOrchestrator;
    private readonly ProductionWorkflowPanelStore _panelStore;

    public ProductionWorkspaceService(
        IApplicationPaths? applicationPaths = null,
        IImageImportService? imageImportService = null,
        ProjectFileStore? projectFileStore = null,
        IExportService? exportService = null,
        IPhaseManualEditor? phaseEditor = null,
        IReadOnlyList<AutomaticStageStatus>? automaticStages = null,
        Func<CancellationToken, Task<RealEsrganBackendResolution>>? enhancementResolver = null,
        Func<WorkflowReviewState?, CancellationToken, Task<WorkflowRunResult>>? automaticWorkflow = null,
        WorkflowOrchestrator? workflowOrchestrator = null,
        ProductionWorkflowPanelStore? panelStore = null,
        IPdfImportService? pdfImportService = null)
        : base(
            applicationPaths,
            imageImportService,
            projectFileStore,
            exportService,
            phaseEditor,
            WorkflowRuntimeEnvironment.Production,
            automaticStages,
            enhancementResolver,
            pdfImportService)
    {
        _automaticWorkflow = automaticWorkflow;
        _workflowOrchestrator = workflowOrchestrator;
        _panelStore = panelStore ?? new ProductionWorkflowPanelStore();
    }

    public WorkflowRunResult? LastAutomaticRun { get; private set; }

    public async Task<WorkflowRunResult> RunAutomaticDetectionAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        AutomaticStageStatus? blocked = RequiredDetectionStages
            .Select(required => AutomaticStages.SingleOrDefault(status =>
                string.Equals(status.Stage, required, StringComparison.Ordinal)))
            .FirstOrDefault(status => status is null ||
                status.State is not (AutomaticStageState.Available or AutomaticStageState.Approved));
        if (blocked is not null)
        {
            throw new InvalidOperationException(blocked.Explanation);
        }

        if (_automaticWorkflow is null && _workflowOrchestrator is null)
        {
            throw new InvalidOperationException(
                "Automatic detection is not composed with approved production adapters.");
        }

        WorkflowRunResult result;
        if (_automaticWorkflow is not null)
        {
            result = await _automaticWorkflow(
                LastAutomaticRun?.Review,
                cancellationToken).ConfigureAwait(false);
        }
        else
        {
            bool enhancementEnabled = AutomaticStages.Any(status =>
                string.Equals(status.Stage, "enhancement", StringComparison.Ordinal) &&
                status.State == AutomaticStageState.Approved);
            var request = new WorkflowRunRequest(
                Guid.NewGuid(),
                CreateProductionWorkflowImportRequest(enhancementEnabled));
            result = await _workflowOrchestrator!.RunThroughReviewAsync(
                request,
                LastAutomaticRun?.Review,
                cancellationToken).ConfigureAwait(false);
        }

        if (result is null)
        {
            throw new InvalidOperationException(
                "The production automatic workflow returned no result.");
        }
        ProductionReviewProjectionResult projection = ProjectProductionReview(result, _panelStore);
        if (!projection.Succeeded)
        {
            throw new ProductionWorkflowStageException(projection.Failure ?? new ProductionWorkflowFailure(
                ProductionWorkflowFailureCodes.ReviewProjectionRejected,
                "Errors.ProductionReviewProjectionRejected",
                "The production review projection failed without structured evidence.",
                Recoverable: true,
                "Keep the current manual review and rerun automatic detection."));
        }

        LastAutomaticRun = projection.ProjectedRun ?? throw new InvalidOperationException(
            "A successful production review projection returned no corrected workflow run.");
        return LastAutomaticRun;
    }

    public override async Task RunStageAsync(WorkflowStage stage, CancellationToken cancellationToken)
    {
        if (stage == WorkflowStage.Detect)
        {
            await RunAutomaticDetectionAsync(cancellationToken).ConfigureAwait(false);
            return;
        }

        await base.RunStageAsync(stage, cancellationToken).ConfigureAwait(false);
    }
}
