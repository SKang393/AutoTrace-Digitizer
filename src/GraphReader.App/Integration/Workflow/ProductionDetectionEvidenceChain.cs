// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;

namespace GraphReader.App.Integration.Workflow;

/// <summary>
/// Retains validated production-stage envelopes so a later structured failure
/// can return all earlier evidence without projecting incomplete graph data.
/// </summary>
public sealed class ProductionDetectionEvidenceChain
{
    private static readonly ReadOnlyDictionary<string, int> StageOrder =
        new ReadOnlyDictionary<string, int>(new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["axis"] = 0,
            ["ocr"] = 1,
            ["markers"] = 2,
            ["legends"] = 3,
            ["phases"] = 4,
        });

    private readonly ProductionWorkflowDetectionRequest request;
    private readonly List<WorkflowVisionEnvelope> completed = [];
    private readonly HashSet<string> evidenceIdentities = new(StringComparer.Ordinal);
    private int lastStageOrder = -1;

    public ProductionDetectionEvidenceChain(ProductionWorkflowDetectionRequest request) =>
        this.request = request ?? throw new ArgumentNullException(nameof(request));

    public IReadOnlyList<WorkflowVisionEnvelope> Snapshot =>
        Array.AsReadOnly(completed.ToArray());

    public void Append(WorkflowVisionEnvelope envelope)
    {
        ArgumentNullException.ThrowIfNull(envelope);
        if (!StageOrder.TryGetValue(envelope.Stage, out int stageOrder))
        {
            throw new ArgumentException(
                $"Production evidence stage '{envelope.Stage}' is not registered.",
                nameof(envelope));
        }

        string identity = EvidenceIdentity(envelope);
        if (stageOrder < lastStageOrder || evidenceIdentities.Contains(identity) ||
            envelope.RunId != request.RunId || envelope.ProjectId != request.ProjectId ||
            envelope.PanelId != request.Panel.ImportedPanel.PanelId ||
            !string.Equals(envelope.InputSha256, request.Image.Sha256, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(envelope.CoordinateSpace, "original_pixels", StringComparison.Ordinal))
        {
            throw new ArgumentException(
                "Production evidence must be ordered, unique, non-fake, original-pixel evidence for the current run, project, panel, and image.",
                nameof(envelope));
        }

        evidenceIdentities.Add(identity);
        completed.Add(envelope);
        lastStageOrder = stageOrder;
    }

    public ProductionWorkflowStageException Reject(ProductionWorkflowFailure failure) =>
        new(
            failure ?? throw new ArgumentNullException(nameof(failure)),
            completed);

    private static string EvidenceIdentity(WorkflowVisionEnvelope envelope) =>
        string.Join(
            '|',
            envelope.Stage,
            envelope.StageVersion,
            envelope.Model?.ModelId ?? "deterministic",
            envelope.Model?.Version ?? "none",
            envelope.Model?.Sha256 ?? "none");
}
