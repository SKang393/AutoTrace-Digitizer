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
            WorkflowRunResult? accumulated = LastAutomaticRun;
            var completedPanelIds = new HashSet<Guid>();
            result = await _workflowOrchestrator!.RunThroughReviewAsync(
                request,
                LastAutomaticRun?.Review,
                (checkpoint, _) =>
                {
                    ProductionReviewProjectionResult checkpointProjection =
                        ProjectProductionReview(checkpoint, _panelStore);
                    if (!checkpointProjection.Succeeded)
                    {
                        throw new ProductionWorkflowStageException(
                            checkpointProjection.Failure ?? new ProductionWorkflowFailure(
                                ProductionWorkflowFailureCodes.ReviewProjectionRejected,
                                "Errors.ProductionReviewProjectionRejected",
                                "A completed production batch panel could not be checkpointed.",
                                Recoverable: true,
                                "Keep prior completed panels and rerun automatic detection."));
                    }

                    WorkflowRunResult projectedCheckpoint = checkpointProjection.ProjectedRun ??
                        throw new InvalidOperationException(
                            "A successful production batch checkpoint returned no projected run.");
                    accumulated = MergeCheckpoint(accumulated, projectedCheckpoint);
                    LastAutomaticRun = accumulated;
                    completedPanelIds.Add(projectedCheckpoint.Review.Panels.Single().PanelId);
                    return Task.CompletedTask;
                },
                cancellationToken).ConfigureAwait(false);

            Guid[] resultPanelIds = result.Review.Panels.Select(static panel => panel.PanelId).ToArray();
            if (resultPanelIds.Length == 0 ||
                resultPanelIds.Any(panelId => !completedPanelIds.Contains(panelId)) ||
                accumulated is null)
            {
                throw new InvalidOperationException(
                    "The production batch completed without a checkpoint for every review panel.");
            }

            Dictionary<Guid, WorkflowReviewPanel> projectedByPanelId = accumulated.Review.Panels
                .ToDictionary(static panel => panel.PanelId);
            WorkflowReviewPanel[] projectedPanels = resultPanelIds
                .Select(panelId => projectedByPanelId[panelId])
                .ToArray();
            HashSet<Guid> currentPanelIds = resultPanelIds.ToHashSet();
            WorkflowCorrection[] corrections = accumulated.Review.CorrectionJournal
                .Where(correction => currentPanelIds.Contains(correction.PanelId))
                .ToArray();
            var completedReview = new WorkflowReviewState(
                result.Review.ProjectId,
                projectedPanels,
                corrections,
                result.Review.Warnings
                    .Concat(accumulated.Review.Warnings)
                    .Distinct(StringComparer.Ordinal));
            result = new WorkflowRunResult(result.RunId, completedReview, result.Steps);
            LastAutomaticRun = result;
            return result;
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

    private static WorkflowRunResult MergeCheckpoint(
        WorkflowRunResult? accumulated,
        WorkflowRunResult checkpoint)
    {
        if (accumulated is null)
        {
            return checkpoint;
        }

        if (accumulated.Review.ProjectId != checkpoint.Review.ProjectId)
        {
            throw new InvalidOperationException(
                "A production batch checkpoint belongs to a different project.");
        }

        WorkflowReviewPanel checkpointPanel = checkpoint.Review.Panels.Single();
        var panels = accumulated.Review.Panels
            .Where(panel => panel.PanelId != checkpointPanel.PanelId)
            .Append(checkpointPanel)
            .ToArray();
        WorkflowCorrection[] corrections = accumulated.Review.CorrectionJournal
            .Where(correction => correction.PanelId != checkpointPanel.PanelId)
            .Concat(checkpoint.Review.CorrectionJournal)
            .ToArray();
        var review = new WorkflowReviewState(
            checkpoint.Review.ProjectId,
            panels,
            corrections,
            accumulated.Review.Warnings
                .Concat(checkpoint.Review.Warnings)
                .Distinct(StringComparer.Ordinal));
        return new WorkflowRunResult(checkpoint.RunId, review, checkpoint.Steps);
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
