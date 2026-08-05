// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Legends;
using GraphReader.Phases;

namespace GraphReader.App.Integration.Workflow;

public interface IProductionLegendReasoningAdapter
{
    string AdapterId { get; }

    bool IsApproved { get; }

    Task<ProductionLegendReasoningEvidence> ResolveAsync(
        ProductionWorkflowDetectionRequest request,
        LegendReasoningRequest legendRequest,
        CancellationToken cancellationToken);
}

public sealed record ProductionLegendReasoningEvidence(
    WorkflowVisionEnvelope Envelope,
    LegendReasoningPayload Payload);

public sealed class ProductionLegendReasoningAdapter : IProductionLegendReasoningAdapter
{
    private readonly ILegendReasoningService service;

    public ProductionLegendReasoningAdapter(
        bool isApproved = true,
        ILegendReasoningService? service = null)
    {
        IsApproved = isApproved;
        this.service = service ?? new LegendReasoningService();
    }

    public string AdapterId => $"graphreader-legends:{LegendReasoningContract.StageVersion}";

    public bool IsApproved { get; }

    public async Task<ProductionLegendReasoningEvidence> ResolveAsync(
        ProductionWorkflowDetectionRequest request,
        LegendReasoningRequest legendRequest,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(legendRequest);
        cancellationToken.ThrowIfCancellationRequested();
        if (!IsApproved)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionModelsUnavailable,
                "Errors.ProductionWorkflowUnavailable",
                $"Legend adapter '{AdapterId}' is not production-approved.",
                "Continue in manual mode until the deterministic adapter is approved.");
        }

        ValidateLegendRequest(request, legendRequest);
        LegendReasoningResult result;
        try
        {
            result = await service.ResolveAsync(legendRequest, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                $"Legend reasoning failed: {exception.Message}",
                "Retain earlier OCR and marker evidence and continue manual review.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        ValidateLegendResult(request, result);
        var envelope = new WorkflowVisionEnvelope(
            LegendReasoningContract.Version,
            request.RunId,
            request.ProjectId,
            request.Panel.ImportedPanel.PanelId,
            LegendReasoningContract.Stage,
            result.StageVersion,
            request.Image.Sha256,
            model: null,
            new WorkflowVisionTiming(
                result.Timing.PreprocessMilliseconds,
                result.Timing.InferenceMilliseconds,
                result.Timing.PostprocessMilliseconds,
                result.Timing.TotalMilliseconds),
            result.Confidence,
            result.Warnings,
            request.Transforms);
        return new ProductionLegendReasoningEvidence(envelope, result.Payload);
    }

    private static void ValidateLegendRequest(
        ProductionWorkflowDetectionRequest request,
        LegendReasoningRequest legendRequest)
    {
        if (legendRequest.ContractVersion != LegendReasoningContract.Version ||
            !string.Equals(
                legendRequest.Options.StageVersion,
                LegendReasoningContract.StageVersion,
                StringComparison.Ordinal) ||
            !string.Equals(legendRequest.ProjectId, request.ProjectId.ToString("D"), StringComparison.Ordinal) ||
            !string.Equals(
                legendRequest.PanelId,
                request.Panel.ImportedPanel.PanelId.ToString("D"),
                StringComparison.Ordinal) ||
            !string.Equals(legendRequest.InputSha256, request.Image.Sha256, StringComparison.OrdinalIgnoreCase) ||
            !legendRequest.PanelBounds.IsValid ||
            legendRequest.PanelBounds.Left < 0 || legendRequest.PanelBounds.Top < 0 ||
            legendRequest.PanelBounds.Right > request.Image.Width ||
            legendRequest.PanelBounds.Bottom > request.Image.Height ||
            !legendRequest.PlotBounds.IsValid)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "Legend reasoning request does not match the current run, panel, image, or original-pixel bounds.",
                "Rebuild legend evidence from the verified OCR and marker stages.");
        }
    }

    private static void ValidateLegendResult(
        ProductionWorkflowDetectionRequest request,
        LegendReasoningResult result)
    {
        if (result is null)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "Legend reasoning returned no result.",
                "Retain earlier evidence and continue manual review.");
        }

        if (!result.Succeeded)
        {
            LegendReasoningFailure? failure = result.Failure;
            throw Failure(
                failure?.Code ?? ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                failure?.UserMessageKey ?? "Errors.DetectionEvidenceRejected",
                failure?.TechnicalMessage ?? "Legend reasoning failed without a diagnostic.",
                failure?.SuggestedAction ?? "Retain earlier evidence and continue manual review.",
                failure?.Recoverable ?? true);
        }

        if (result.ContractVersion != LegendReasoningContract.Version ||
            !string.Equals(result.ProjectId, request.ProjectId.ToString("D"), StringComparison.Ordinal) ||
            !string.Equals(
                result.PanelId,
                request.Panel.ImportedPanel.PanelId.ToString("D"),
                StringComparison.Ordinal) ||
            !string.Equals(result.Stage, LegendReasoningContract.Stage, StringComparison.Ordinal) ||
            !string.Equals(result.StageVersion, LegendReasoningContract.StageVersion, StringComparison.Ordinal) ||
            !string.Equals(result.InputSha256, request.Image.Sha256, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(
                result.CoordinateSpace,
                LegendReasoningContract.CoordinateSpace,
                StringComparison.Ordinal) ||
            result.Model is not null ||
            result.Regions.Select(static region => region.RegionId)
                .Distinct(StringComparer.Ordinal).Count() != result.Regions.Count ||
            result.Series.Select(static series => series.SeriesId)
                .Distinct(StringComparer.Ordinal).Count() != result.Series.Count ||
            result.Participants.Any(static participant =>
                string.IsNullOrWhiteSpace(participant.Name) || !participant.Bounds.IsValid))
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "Legend reasoning returned mismatched identity, coordinate, deterministic-model, or payload evidence.",
                "Reject the semantic result and retain earlier OCR and marker evidence.");
        }
    }

    private static ProductionWorkflowStageException Failure(
        string code,
        string userMessageKey,
        string technicalMessage,
        string suggestedAction,
        bool recoverable = true) =>
        new(new ProductionWorkflowFailure(
            code,
            userMessageKey,
            technicalMessage,
            recoverable,
            suggestedAction));
}

public interface IProductionPhaseReasoningAdapter
{
    string AdapterId { get; }

    bool IsApproved { get; }

    Task<ProductionPhaseReasoningEvidence> ResolveAsync(
        ProductionWorkflowDetectionRequest request,
        PhaseReasoningRequest phaseRequest,
        CancellationToken cancellationToken);
}

public sealed record ProductionPhaseReasoningEvidence(
    WorkflowVisionEnvelope Envelope,
    PhaseReasoningPayload Payload);

public sealed class ProductionPhaseReasoningAdapter : IProductionPhaseReasoningAdapter
{
    private readonly IPhaseReasoningService service;

    public ProductionPhaseReasoningAdapter(
        bool isApproved = true,
        IPhaseReasoningService? service = null)
    {
        IsApproved = isApproved;
        this.service = service ?? new PhaseReasoningService();
    }

    public string AdapterId => $"graphreader-phases:{PhaseReasoningContract.StageVersion}";

    public bool IsApproved { get; }

    public async Task<ProductionPhaseReasoningEvidence> ResolveAsync(
        ProductionWorkflowDetectionRequest request,
        PhaseReasoningRequest phaseRequest,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(phaseRequest);
        cancellationToken.ThrowIfCancellationRequested();
        if (!IsApproved)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionModelsUnavailable,
                "Errors.ProductionWorkflowUnavailable",
                $"Phase adapter '{AdapterId}' is not production-approved.",
                "Continue in manual mode until the deterministic adapter is approved.");
        }

        ValidatePhaseRequest(request, phaseRequest);
        PhaseReasoningResult result;
        try
        {
            result = await service.ResolveAsync(phaseRequest, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                $"Phase reasoning failed: {exception.Message}",
                "Retain earlier axis, OCR, and marker evidence and continue manual review.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        ValidatePhaseResult(request, result);
        var envelope = new WorkflowVisionEnvelope(
            PhaseReasoningContract.Version,
            request.RunId,
            request.ProjectId,
            request.Panel.ImportedPanel.PanelId,
            PhaseReasoningContract.Stage,
            result.StageVersion,
            request.Image.Sha256,
            model: null,
            new WorkflowVisionTiming(
                result.Timing.PreprocessMilliseconds,
                result.Timing.InferenceMilliseconds,
                result.Timing.PostprocessMilliseconds,
                result.Timing.TotalMilliseconds),
            result.Confidence,
            result.Warnings,
            request.Transforms);
        return new ProductionPhaseReasoningEvidence(envelope, result.Payload);
    }

    private static void ValidatePhaseRequest(
        ProductionWorkflowDetectionRequest request,
        PhaseReasoningRequest phaseRequest)
    {
        if (phaseRequest.ContractVersion != PhaseReasoningContract.Version ||
            !string.Equals(
                phaseRequest.Options.StageVersion,
                PhaseReasoningContract.StageVersion,
                StringComparison.Ordinal) ||
            !string.Equals(phaseRequest.ProjectId, request.ProjectId.ToString("D"), StringComparison.Ordinal) ||
            !string.Equals(
                phaseRequest.PanelId,
                request.Panel.ImportedPanel.PanelId.ToString("D"),
                StringComparison.Ordinal) ||
            !string.Equals(phaseRequest.InputSha256, request.Image.Sha256, StringComparison.OrdinalIgnoreCase) ||
            !phaseRequest.PlotBounds.IsValid ||
            phaseRequest.PlotBounds.Left < 0 || phaseRequest.PlotBounds.Top < 0 ||
            phaseRequest.PlotBounds.Right > request.Image.Width ||
            phaseRequest.PlotBounds.Bottom > request.Image.Height)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "Phase reasoning request does not match the current run, panel, image, or original-pixel bounds.",
                "Rebuild phase evidence from the verified axis, OCR, and marker stages.");
        }
    }

    private static void ValidatePhaseResult(
        ProductionWorkflowDetectionRequest request,
        PhaseReasoningResult result)
    {
        if (result is null)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "Phase reasoning returned no result.",
                "Retain earlier evidence and continue manual phase editing.");
        }

        if (!result.Succeeded)
        {
            PhaseReasoningFailure? failure = result.Failure;
            throw Failure(
                failure?.Code ?? ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                failure?.UserMessageKey ?? "Errors.DetectionEvidenceRejected",
                failure?.TechnicalMessage ?? "Phase reasoning failed without a diagnostic.",
                failure?.SuggestedAction ?? "Retain earlier evidence and continue manual phase editing.",
                failure?.Recoverable ?? true);
        }

        if (result.ContractVersion != PhaseReasoningContract.Version ||
            !string.Equals(result.ProjectId, request.ProjectId.ToString("D"), StringComparison.Ordinal) ||
            !string.Equals(
                result.PanelId,
                request.Panel.ImportedPanel.PanelId.ToString("D"),
                StringComparison.Ordinal) ||
            !string.Equals(result.Stage, PhaseReasoningContract.Stage, StringComparison.Ordinal) ||
            !string.Equals(result.StageVersion, PhaseReasoningContract.StageVersion, StringComparison.Ordinal) ||
            !string.Equals(result.InputSha256, request.Image.Sha256, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(
                result.CoordinateSpace,
                PhaseReasoningContract.CoordinateSpace,
                StringComparison.Ordinal) ||
            result.Payload.Dividers.Select(static divider => divider.DividerId)
                .Distinct(StringComparer.Ordinal).Count() != result.Payload.Dividers.Count ||
            result.Payload.Phases.Select(static phase => phase.PhaseId)
                .Distinct(StringComparer.Ordinal).Count() != result.Payload.Phases.Count ||
            result.Payload.Assignments.Select(static assignment => assignment.PointId)
                .Distinct(StringComparer.Ordinal).Count() != result.Payload.Assignments.Count ||
            result.Payload.Dividers.Any(static divider => !double.IsFinite(divider.OriginalX)) ||
            result.Payload.Phases.Any(static phase =>
                !double.IsFinite(phase.OriginalXMinimum) ||
                !double.IsFinite(phase.OriginalXMaximum) ||
                phase.OriginalXMinimum >= phase.OriginalXMaximum))
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "Phase reasoning returned mismatched identity, coordinate, or payload evidence.",
                "Reject the semantic result and retain earlier axis, OCR, and marker evidence.");
        }
    }

    private static ProductionWorkflowStageException Failure(
        string code,
        string userMessageKey,
        string technicalMessage,
        string suggestedAction,
        bool recoverable = true) =>
        new(new ProductionWorkflowFailure(
            code,
            userMessageKey,
            technicalMessage,
            recoverable,
            suggestedAction));
}
