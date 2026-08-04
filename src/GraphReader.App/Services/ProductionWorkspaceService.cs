// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration.Workflow;
using GraphReader.App.ViewModels;
using GraphReader.Domain;
using GraphReader.Export;
using GraphReader.Imaging;
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

    public ProductionWorkspaceService(
        IApplicationPaths? applicationPaths = null,
        IImageImportService? imageImportService = null,
        ProjectFileStore? projectFileStore = null,
        IExportService? exportService = null,
        IPhaseManualEditor? phaseEditor = null,
        IReadOnlyList<AutomaticStageStatus>? automaticStages = null,
        Func<CancellationToken, Task<RealEsrganBackendResolution>>? enhancementResolver = null,
        Func<WorkflowReviewState?, CancellationToken, Task<WorkflowRunResult>>? automaticWorkflow = null)
        : base(
            applicationPaths,
            imageImportService,
            projectFileStore,
            exportService,
            phaseEditor,
            WorkflowRuntimeEnvironment.Production,
            automaticStages,
            enhancementResolver)
    {
        _automaticWorkflow = automaticWorkflow;
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

        if (_automaticWorkflow is null)
        {
            throw new InvalidOperationException(
                "Automatic detection is not composed with approved production adapters.");
        }

        WorkflowRunResult result = await _automaticWorkflow(
            LastAutomaticRun?.Review,
            cancellationToken).ConfigureAwait(false);
        LastAutomaticRun = result ?? throw new InvalidOperationException(
            "The production automatic workflow returned no result.");
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
