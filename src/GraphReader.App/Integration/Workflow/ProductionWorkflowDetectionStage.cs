// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;

namespace GraphReader.App.Integration.Workflow;

public interface IProductionWorkflowDetectionAdapter
{
    string AdapterId { get; }

    bool IsApproved { get; }

    Task<WorkflowDetectionBatch> DetectAsync(
        ProductionWorkflowDetectionRequest request,
        CancellationToken cancellationToken);
}

public sealed class ProductionWorkflowDetectionRequest
{
    private readonly byte[] imageBytes;

    public ProductionWorkflowDetectionRequest(
        WorkflowPreparedPanel panel,
        WorkflowImageEvidence image,
        WorkflowImageVariant imageVariant,
        Guid runId,
        Guid projectId,
        byte[] imageBytes,
        IEnumerable<WorkflowTransformProvenance>? transforms = null)
    {
        Panel = panel ?? throw new ArgumentNullException(nameof(panel));
        Image = image ?? throw new ArgumentNullException(nameof(image));
        ImageVariant = imageVariant;
        RunId = runId;
        ProjectId = projectId;
        ArgumentNullException.ThrowIfNull(imageBytes);
        this.imageBytes = (byte[])imageBytes.Clone();
        Transforms = Array.AsReadOnly((transforms ?? []).ToArray());
    }

    public WorkflowPreparedPanel Panel { get; }

    public WorkflowImageEvidence Image { get; }

    public WorkflowImageVariant ImageVariant { get; }

    public Guid RunId { get; }

    public Guid ProjectId { get; }

    public IReadOnlyList<WorkflowTransformProvenance> Transforms { get; }

    public byte[] CopyImageBytes() => (byte[])imageBytes.Clone();
}

public sealed class ProductionWorkflowDetectionStage : IWorkflowDetectionStage
{
    private readonly ProductionWorkflowPanelStore panelStore;
    private readonly IProductionWorkflowDetectionAdapter? adapter;

    public ProductionWorkflowDetectionStage(
        ProductionWorkflowPanelStore panelStore,
        IProductionWorkflowDetectionAdapter? adapter = null)
    {
        this.panelStore = panelStore ?? throw new ArgumentNullException(nameof(panelStore));
        this.adapter = adapter;
    }

    public async Task<WorkflowDetectionBatch> DetectAsync(
        WorkflowPreparedPanel panel,
        WorkflowImageVariant imageVariant,
        Guid runId,
        Guid projectId,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(panel);
        cancellationToken.ThrowIfCancellationRequested();
        if (imageVariant is not (WorkflowImageVariant.Original or WorkflowImageVariant.Enhanced))
        {
            throw new ArgumentOutOfRangeException(nameof(imageVariant));
        }

        if (runId == Guid.Empty || projectId == Guid.Empty)
        {
            throw new ArgumentException("Run and project IDs are required.");
        }

        if (adapter is null || !adapter.IsApproved)
        {
            string adapterStatus = adapter is null
                ? "No production detection adapter is configured."
                : $"Detection adapter '{adapter.AdapterId}' is not approved.";
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionModelsUnavailable,
                "Errors.ModelNotFound",
                adapterStatus,
                "Install checksum-verified approved production models or continue in manual mode.");
        }

        ProductionPanelEvidence stored = panelStore.Get(panel.ImportedPanel.PanelId);
        WorkflowImageEvidence image;
        byte[]? bytes;
        IReadOnlyList<WorkflowTransformProvenance> transforms;
        if (imageVariant == WorkflowImageVariant.Original)
        {
            image = panel.Original;
            bytes = stored.CopyOriginalBytes();
            transforms = [];
        }
        else
        {
            image = panel.Enhanced ?? throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.EnhancementUnavailable",
                "Enhanced detection was requested without retained enhanced evidence.",
                "Run detection on the immutable original or prepare an approved derivative.");
            bytes = stored.CopyEnhancedBytes();
            transforms = stored.EnhancementTransforms;
        }

        if (bytes is null || !ChecksumMatches(bytes, image.Sha256))
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.SourceChanged",
                "Retained detection bytes do not match the selected image checksum.",
                "Re-import the source and retry detection.");
        }

        WorkflowDetectionBatch batch = await adapter.DetectAsync(
                new ProductionWorkflowDetectionRequest(
                    panel,
                    image,
                    imageVariant,
                    runId,
                    projectId,
                    bytes,
                    transforms),
                cancellationToken)
            .ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        if (batch.PanelId != panel.ImportedPanel.PanelId ||
            batch.SourceImage != imageVariant ||
            batch.Envelope.RunId != runId ||
            batch.Envelope.ProjectId != projectId ||
            !string.Equals(batch.Envelope.InputSha256, image.Sha256, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(batch.CoordinateSpace, "original_pixels", StringComparison.Ordinal))
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "The approved detection adapter returned evidence for a different run, project, panel, image, or coordinate space.",
                "Reject the adapter result and rerun after verifying model composition.");
        }

        return batch;
    }

    private static bool ChecksumMatches(byte[] bytes, string expected) =>
        string.Equals(
            Convert.ToHexString(SHA256.HashData(bytes)),
            expected,
            StringComparison.OrdinalIgnoreCase);

    private static ProductionWorkflowStageException Failure(
        string code,
        string userMessageKey,
        string technicalMessage,
        string suggestedAction) =>
        new(new ProductionWorkflowFailure(
            code,
            userMessageKey,
            technicalMessage,
            Recoverable: true,
            suggestedAction));
}
