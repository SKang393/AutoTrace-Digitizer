// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.App.Integration.Workflow;

public static class WorkflowComposition
{
    public static WorkflowOrchestrator Create(
        WorkflowRuntimeEnvironment environment,
        WorkflowServiceSet production,
        WorkflowServiceSet recordedFake)
    {
        ArgumentNullException.ThrowIfNull(production);
        ArgumentNullException.ThrowIfNull(recordedFake);

        WorkflowServiceSet selected = environment switch
        {
            WorkflowRuntimeEnvironment.Production => production,
            WorkflowRuntimeEnvironment.RecordedFake => recordedFake,
            _ => throw new ArgumentOutOfRangeException(nameof(environment)),
        };

        return new WorkflowOrchestrator(selected);
    }
}

public sealed class RecordedWorkflowData
{
    public RecordedWorkflowData(
        WorkflowImportSnapshot import,
        IEnumerable<WorkflowPreparedPanel> preparedPanels,
        IEnumerable<WorkflowDetectionBatch> detections,
        WorkflowExportResult export)
    {
        Import = import ?? throw new ArgumentNullException(nameof(import));
        PreparedPanels = WorkflowCollections.Freeze(preparedPanels);
        Detections = WorkflowCollections.Freeze(detections);
        Export = export ?? throw new ArgumentNullException(nameof(export));
        if (PreparedPanels.Select(static panel => panel.ImportedPanel.PanelId).Distinct().Count() != PreparedPanels.Count)
        {
            throw new ArgumentException("Recorded prepared panels must be unique.", nameof(preparedPanels));
        }

        if (Detections
            .Select(static detection => (detection.PanelId, detection.SourceImage))
            .Distinct()
            .Count() != Detections.Count)
        {
            throw new ArgumentException("Recorded detections must be unique by panel and image variant.", nameof(detections));
        }
    }

    public WorkflowImportSnapshot Import { get; }

    public IReadOnlyList<WorkflowPreparedPanel> PreparedPanels { get; }

    public IReadOnlyList<WorkflowDetectionBatch> Detections { get; }

    public WorkflowExportResult Export { get; }

    public WorkflowServiceSet CreateServiceSet()
    {
        var stages = new RecordedWorkflowStages(this);
        return new WorkflowServiceSet(stages, stages, stages, stages);
    }
}

public sealed class RecordedWorkflowStages :
    IWorkflowImportStage,
    IWorkflowPrepareStage,
    IWorkflowDetectionStage,
    IWorkflowExportStage
{
    private readonly RecordedWorkflowData data;

    public RecordedWorkflowStages(RecordedWorkflowData data)
    {
        this.data = data ?? throw new ArgumentNullException(nameof(data));
    }

    public Task<WorkflowImportSnapshot> ImportAsync(
        WorkflowImportRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        if (request.ProjectId != data.Import.ProjectId)
        {
            throw new InvalidOperationException("The recorded import does not match the requested project.");
        }

        return Task.FromResult(data.Import);
    }

    public Task<WorkflowPreparedPanel> PrepareAsync(
        WorkflowImportedPanel panel,
        bool enhancementEnabled,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(panel);
        cancellationToken.ThrowIfCancellationRequested();
        WorkflowPreparedPanel recorded = data.PreparedPanels.SingleOrDefault(candidate =>
            candidate.ImportedPanel.PanelId == panel.PanelId)
            ?? throw new InvalidOperationException("No recorded preparation exists for the requested panel.");
        if (!string.Equals(recorded.Original.Sha256, panel.Original.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("The recorded preparation does not match the imported original.");
        }

        WorkflowPreparedPanel result = enhancementEnabled
            ? recorded
            : new WorkflowPreparedPanel(recorded.ImportedPanel, recorded.Original, enhanced: null, recorded.Warnings);
        return Task.FromResult(result);
    }

    public Task<WorkflowDetectionBatch> DetectAsync(
        WorkflowPreparedPanel panel,
        WorkflowImageVariant imageVariant,
        Guid runId,
        Guid projectId,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(panel);
        cancellationToken.ThrowIfCancellationRequested();
        WorkflowDetectionBatch recorded = data.Detections.SingleOrDefault(candidate =>
            candidate.PanelId == panel.ImportedPanel.PanelId && candidate.SourceImage == imageVariant)
            ?? throw new InvalidOperationException("No recorded detection exists for the requested panel and image variant.");
        if (recorded.Envelope.RunId != runId || recorded.Envelope.ProjectId != projectId)
        {
            throw new InvalidOperationException("The recorded detection does not match the requested run and project.");
        }

        return Task.FromResult(recorded);
    }

    public Task<WorkflowExportResult> ExportAsync(
        WorkflowReviewState review,
        WorkflowExportRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(review);
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        if (review.ProjectId != data.Import.ProjectId)
        {
            throw new InvalidOperationException("The recorded export does not match the requested project.");
        }

        return Task.FromResult(data.Export);
    }
}
