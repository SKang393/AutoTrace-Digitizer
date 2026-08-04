// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Security.Cryptography;

namespace GraphReader.App.Integration.Workflow;

public interface IProductionWorkflowEnhancementAdapter
{
    string AdapterId { get; }

    bool IsApproved { get; }

    Task<ProductionWorkflowEnhancementResult> EnhanceAsync(
        ProductionWorkflowEnhancementRequest request,
        CancellationToken cancellationToken);
}

public sealed class ProductionWorkflowEnhancementRequest
{
    private readonly byte[] originalBytes;

    public ProductionWorkflowEnhancementRequest(
        Guid panelId,
        WorkflowImageEvidence original,
        byte[] originalBytes)
    {
        if (panelId == Guid.Empty)
        {
            throw new ArgumentException("A panel ID is required.", nameof(panelId));
        }

        PanelId = panelId;
        Original = original ?? throw new ArgumentNullException(nameof(original));
        ArgumentNullException.ThrowIfNull(originalBytes);
        this.originalBytes = (byte[])originalBytes.Clone();
    }

    public Guid PanelId { get; }

    public WorkflowImageEvidence Original { get; }

    public byte[] CopyOriginalBytes() => (byte[])originalBytes.Clone();
}

public sealed class ProductionWorkflowEnhancedImage
{
    private readonly byte[] bytes;

    public ProductionWorkflowEnhancedImage(
        string reference,
        string sha256,
        int width,
        int height,
        byte[] bytes,
        IEnumerable<WorkflowTransformProvenance> transforms)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(reference);
        ArgumentException.ThrowIfNullOrWhiteSpace(sha256);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentNullException.ThrowIfNull(bytes);
        Reference = reference;
        Sha256 = sha256.ToLowerInvariant();
        Width = width;
        Height = height;
        this.bytes = (byte[])bytes.Clone();
        Transforms = new ReadOnlyCollection<WorkflowTransformProvenance>(
            (transforms ?? throw new ArgumentNullException(nameof(transforms))).ToArray());
    }

    public string Reference { get; }

    public string Sha256 { get; }

    public int Width { get; }

    public int Height { get; }

    public IReadOnlyList<WorkflowTransformProvenance> Transforms { get; }

    public byte[] CopyBytes() => (byte[])bytes.Clone();
}

public sealed record ProductionWorkflowEnhancementResult(
    ProductionWorkflowEnhancedImage? Image,
    IReadOnlyList<string> Warnings)
{
    public static ProductionWorkflowEnhancementResult ContinueOriginal(params string[] warnings) =>
        new(null, Array.AsReadOnly(warnings ?? []));
}

public sealed class ProductionWorkflowPrepareStage : IWorkflowPrepareStage
{
    private readonly ProductionWorkflowPanelStore panelStore;
    private readonly IProductionWorkflowEnhancementAdapter? enhancementAdapter;

    public ProductionWorkflowPrepareStage(
        ProductionWorkflowPanelStore panelStore,
        IProductionWorkflowEnhancementAdapter? enhancementAdapter = null)
    {
        this.panelStore = panelStore ?? throw new ArgumentNullException(nameof(panelStore));
        this.enhancementAdapter = enhancementAdapter;
    }

    public async Task<WorkflowPreparedPanel> PrepareAsync(
        WorkflowImportedPanel panel,
        bool enhancementEnabled,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(panel);
        cancellationToken.ThrowIfCancellationRequested();
        ProductionPanelEvidence stored = panelStore.Get(panel.PanelId);
        if (stored.Panel.SourceId != panel.SourceId ||
            !string.Equals(stored.Panel.Original.Sha256, panel.Original.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("The requested panel does not match retained production import evidence.");
        }

        if (!enhancementEnabled)
        {
            panelStore.SetPreparation(panel.PanelId, null, null, [], []);
            return new WorkflowPreparedPanel(panel, panel.Original, enhanced: null);
        }

        if (enhancementAdapter is null)
        {
            const string warning = "Approved enhancement is unavailable; preparation continued with immutable original evidence.";
            panelStore.SetPreparation(panel.PanelId, null, null, [], [warning]);
            return new WorkflowPreparedPanel(panel, panel.Original, enhanced: null, [warning]);
        }

        if (!enhancementAdapter.IsApproved)
        {
            string warning = $"Enhancement adapter '{enhancementAdapter.AdapterId}' is not approved; preparation continued with immutable original evidence.";
            panelStore.SetPreparation(panel.PanelId, null, null, [], [warning]);
            return new WorkflowPreparedPanel(panel, panel.Original, enhanced: null, [warning]);
        }

        ProductionWorkflowEnhancementResult result = await enhancementAdapter.EnhanceAsync(
                new ProductionWorkflowEnhancementRequest(
                    panel.PanelId,
                    panel.Original,
                    stored.CopyOriginalBytes()),
                cancellationToken)
            .ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(result);
        string[] warnings = (result.Warnings ?? Array.Empty<string>())
            .Where(static warning => !string.IsNullOrWhiteSpace(warning))
            .Distinct(StringComparer.Ordinal)
            .ToArray();

        if (result.Image is null)
        {
            panelStore.SetPreparation(panel.PanelId, null, null, [], warnings);
            return new WorkflowPreparedPanel(panel, panel.Original, enhanced: null, warnings);
        }

        ProductionWorkflowEnhancedImage candidate = result.Image;
        byte[] enhancedBytes = candidate.CopyBytes();
        if (!IsValidDerivative(panel.Original, candidate, enhancedBytes, out string? invalidReason))
        {
            string[] failedWarnings = warnings
                .Append($"Approved enhancement output was rejected: {invalidReason}")
                .Distinct(StringComparer.Ordinal)
                .ToArray();
            panelStore.SetPreparation(panel.PanelId, null, null, [], failedWarnings);
            return new WorkflowPreparedPanel(panel, panel.Original, enhanced: null, failedWarnings);
        }

        var enhanced = new WorkflowImageEvidence(
            candidate.Reference,
            candidate.Sha256,
            candidate.Width,
            candidate.Height,
            WorkflowImageVariant.Enhanced);
        panelStore.SetPreparation(
            panel.PanelId,
            enhanced,
            enhancedBytes,
            candidate.Transforms,
            warnings);
        return new WorkflowPreparedPanel(panel, panel.Original, enhanced, warnings);
    }

    private static bool IsValidDerivative(
        WorkflowImageEvidence original,
        ProductionWorkflowEnhancedImage candidate,
        byte[] bytes,
        out string? reason)
    {
        string actualSha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        if (!string.Equals(actualSha256, candidate.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            reason = "the derivative checksum does not match its bytes.";
            return false;
        }

        if (candidate.Transforms.Count == 0 ||
            !string.Equals(candidate.Transforms[0].InputCoordinateSpace, "original_pixels", StringComparison.Ordinal) ||
            !string.Equals(candidate.Transforms[^1].OutputCoordinateSpace, "enhanced_pixels", StringComparison.Ordinal) ||
            candidate.Transforms.Any(static transform => transform.Lossy || transform.OutputToInputMatrix is null))
        {
            reason = "a complete reversible original-to-enhanced transform chain is required.";
            return false;
        }

        if (candidate.Width < original.Width || candidate.Height < original.Height)
        {
            reason = "the derivative dimensions cannot be smaller than the immutable original.";
            return false;
        }

        reason = null;
        return true;
    }
}
