// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.App.Integration.Workflow;

public static class ManualCorrectionOverlay
{
    private const string RerunPreservationWarning = "AUTOMATION_RERUN_REMOVED_CORRECTED_POINT";

    public static WorkflowReviewState Apply(
        WorkflowReviewState state,
        WorkflowCorrection correction)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateCorrection(correction);
        if (state.CorrectionJournal.Any(existing =>
            string.Equals(existing.CorrectionId, correction.CorrectionId, StringComparison.Ordinal)))
        {
            throw new ArgumentException("Correction IDs must be unique.", nameof(correction));
        }

        int panelIndex = FindPanel(state.Panels, correction.PanelId);
        if (panelIndex < 0)
        {
            throw new ArgumentException("The correction panel does not exist in the review state.", nameof(correction));
        }

        var panels = state.Panels.ToArray();
        WorkflowReviewPanel panel = panels[panelIndex];
        panels[panelIndex] = new WorkflowReviewPanel(
            panel.PreparedPanel,
            ApplyToPoints(panel.Points, correction, requireTarget: true),
            panel.DetectionProvenance);

        return new WorkflowReviewState(
            state.ProjectId,
            panels,
            state.CorrectionJournal.Append(correction),
            state.Warnings);
    }

    public static WorkflowReviewPanel Reapply(
        WorkflowReviewPanel currentAutomation,
        WorkflowReviewPanel? previousReview,
        IEnumerable<WorkflowCorrection> correctionJournal)
    {
        ArgumentNullException.ThrowIfNull(currentAutomation);
        ArgumentNullException.ThrowIfNull(correctionJournal);
        WorkflowCorrection[] panelCorrections = correctionJournal
            .Where(correction => correction.PanelId == currentAutomation.PanelId)
            .ToArray();
        foreach (WorkflowCorrection correction in panelCorrections)
        {
            ValidateCorrection(correction);
        }

        IReadOnlyList<WorkflowPoint> points = PreserveReviewedIdentities(
            currentAutomation.Points,
            previousReview);
        foreach (WorkflowCorrection correction in panelCorrections)
        {
            points = ApplyToPoints(points, correction, requireTarget: false);
        }

        return new WorkflowReviewPanel(
            currentAutomation.PreparedPanel,
            points,
            currentAutomation.DetectionProvenance);
    }

    private static IReadOnlyList<WorkflowPoint> PreserveReviewedIdentities(
        IReadOnlyList<WorkflowPoint> current,
        WorkflowReviewPanel? previous)
    {
        if (previous is null)
        {
            return current;
        }

        Dictionary<string, WorkflowPoint> previousByDetectionKey = previous.Points
            .Where(static point => !string.IsNullOrWhiteSpace(point.DetectionKey))
            .GroupBy(static point => point.DetectionKey!, StringComparer.Ordinal)
            .ToDictionary(
                static group => group.Key,
                static group => group.OrderBy(static point => point.PointId, StringComparer.Ordinal).First(),
                StringComparer.Ordinal);
        var preserved = new List<WorkflowPoint>(current.Count + previous.Points.Count);
        var presentPointIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (WorkflowPoint point in current)
        {
            WorkflowPoint retained = point;
            if (point.DetectionKey is { } detectionKey &&
                previousByDetectionKey.TryGetValue(detectionKey, out WorkflowPoint? prior))
            {
                retained = point with
                {
                    PointId = prior.PointId,
                    ReviewStatus = prior.ReviewStatus,
                };
            }

            retained = retained with { PointId = MakeUniquePointId(retained.PointId, presentPointIds) };
            preserved.Add(retained);
            presentPointIds.Add(retained.PointId);
        }

        foreach (WorkflowPoint prior in previous.Points
                     .Where(static point => point.IsManual || point.CorrectionIds.Count > 0)
                     .OrderBy(static point => point.PointId, StringComparer.Ordinal))
        {
            bool detectionStillPresent = prior.DetectionKey is { } key && preserved.Any(point =>
                string.Equals(point.DetectionKey, key, StringComparison.Ordinal));
            if (detectionStillPresent || presentPointIds.Contains(prior.PointId))
            {
                continue;
            }

            WorkflowPoint orphan = prior with
            {
                Warnings = WorkflowCollections.Freeze(
                    prior.Warnings.Append(RerunPreservationWarning).Distinct(StringComparer.Ordinal)),
            };
            preserved.Add(orphan);
            presentPointIds.Add(orphan.PointId);
        }

        return WorkflowCollections.Freeze(preserved
            .OrderBy(static point => point.OriginalPixelX)
            .ThenBy(static point => point.OriginalPixelY)
            .ThenBy(static point => point.PointId, StringComparer.Ordinal));
    }

    private static IReadOnlyList<WorkflowPoint> ApplyToPoints(
        IReadOnlyList<WorkflowPoint> source,
        WorkflowCorrection correction,
        bool requireTarget)
    {
        var points = source.ToList();
        switch (correction)
        {
            case AddWorkflowPointCorrection add:
                if (points.Any(point => string.Equals(point.PointId, add.Point.PointId, StringComparison.Ordinal)))
                {
                    if (requireTarget)
                    {
                        throw new ArgumentException("The added point ID already exists.", nameof(correction));
                    }

                    break;
                }

                points.Add(add.Point with
                {
                    IsManual = true,
                    ReviewStatus = WorkflowReviewStatus.Corrected,
                    SourceImage = WorkflowImageVariant.Original,
                    CorrectionIds = AppendCorrection(add.Point.CorrectionIds, add.CorrectionId),
                });
                break;

            case DeleteWorkflowPointCorrection delete:
                {
                    int index = FindPoint(points, delete.TargetPointId, delete.TargetDetectionKey);
                    if (index >= 0)
                    {
                        points.RemoveAt(index);
                    }
                    else if (requireTarget)
                    {
                        throw new ArgumentException("The deleted point does not exist.", nameof(correction));
                    }

                    break;
                }

            case MoveWorkflowPointCorrection move:
                {
                    int index = FindPoint(points, move.TargetPointId, move.TargetDetectionKey);
                    if (!RequireTarget(index, requireTarget, nameof(correction)))
                    {
                        break;
                    }

                    points[index] = points[index] with
                    {
                        OriginalPixelX = move.OriginalPixelX,
                        OriginalPixelY = move.OriginalPixelY,
                        GraphX = null,
                        GraphY = null,
                        ReviewStatus = WorkflowReviewStatus.Corrected,
                        CorrectionIds = AppendCorrection(points[index].CorrectionIds, move.CorrectionId),
                    };
                    break;
                }

            case ReassignWorkflowPointCorrection reassign:
                {
                    int index = FindPoint(points, reassign.TargetPointId, reassign.TargetDetectionKey);
                    if (!RequireTarget(index, requireTarget, nameof(correction)))
                    {
                        break;
                    }

                    points[index] = points[index] with
                    {
                        SeriesId = reassign.SeriesId,
                        ReviewStatus = WorkflowReviewStatus.Corrected,
                        CorrectionIds = AppendCorrection(points[index].CorrectionIds, reassign.CorrectionId),
                    };
                    break;
                }

            case AssignWorkflowPointPhaseCorrection assignPhase:
                {
                    int index = FindPoint(points, assignPhase.TargetPointId, assignPhase.TargetDetectionKey);
                    if (!RequireTarget(index, requireTarget, nameof(correction)))
                    {
                        break;
                    }

                    points[index] = points[index] with
                    {
                        PhaseId = assignPhase.PhaseId,
                        ReviewStatus = WorkflowReviewStatus.Corrected,
                        CorrectionIds = AppendCorrection(points[index].CorrectionIds, assignPhase.CorrectionId),
                    };
                    break;
                }

            default:
                throw new ArgumentException("The correction type is not supported.", nameof(correction));
        }

        return WorkflowCollections.Freeze(points
            .OrderBy(static point => point.OriginalPixelX)
            .ThenBy(static point => point.OriginalPixelY)
            .ThenBy(static point => point.PointId, StringComparer.Ordinal));
    }

    private static void ValidateCorrection(WorkflowCorrection correction)
    {
        ArgumentNullException.ThrowIfNull(correction);
        ArgumentException.ThrowIfNullOrWhiteSpace(correction.CorrectionId);
        if (correction.PanelId == Guid.Empty)
        {
            throw new ArgumentException("A correction panel ID is required.", nameof(correction));
        }

        switch (correction)
        {
            case AddWorkflowPointCorrection add:
                ArgumentNullException.ThrowIfNull(add.Point);
                if (add.Point.DetectionKey is not null)
                {
                    throw new ArgumentException("A manually added point cannot claim automated detection evidence.", nameof(correction));
                }

                break;
            case DeleteWorkflowPointCorrection delete:
                RequireTargetSelector(delete.TargetPointId, delete.TargetDetectionKey, nameof(correction));
                break;
            case MoveWorkflowPointCorrection move:
                RequireTargetSelector(move.TargetPointId, move.TargetDetectionKey, nameof(correction));
                WorkflowContractGuards.RequireFinite(move.OriginalPixelX, nameof(move.OriginalPixelX));
                WorkflowContractGuards.RequireFinite(move.OriginalPixelY, nameof(move.OriginalPixelY));
                break;
            case ReassignWorkflowPointCorrection reassign:
                RequireTargetSelector(reassign.TargetPointId, reassign.TargetDetectionKey, nameof(correction));
                ArgumentException.ThrowIfNullOrWhiteSpace(reassign.SeriesId);
                break;
            case AssignWorkflowPointPhaseCorrection assignPhase:
                RequireTargetSelector(assignPhase.TargetPointId, assignPhase.TargetDetectionKey, nameof(correction));
                ArgumentException.ThrowIfNullOrWhiteSpace(assignPhase.PhaseId);
                break;
            default:
                throw new ArgumentException("The correction type is not supported.", nameof(correction));
        }
    }

    private static void RequireTargetSelector(string pointId, string? detectionKey, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(pointId) && string.IsNullOrWhiteSpace(detectionKey))
        {
            throw new ArgumentException("A point ID or detection key is required.", parameterName);
        }
    }

    private static int FindPanel(IReadOnlyList<WorkflowReviewPanel> panels, Guid panelId)
    {
        for (int index = 0; index < panels.Count; index++)
        {
            if (panels[index].PanelId == panelId)
            {
                return index;
            }
        }

        return -1;
    }

    private static int FindPoint(
        List<WorkflowPoint> points,
        string pointId,
        string? detectionKey)
    {
        if (!string.IsNullOrWhiteSpace(detectionKey))
        {
            for (int index = 0; index < points.Count; index++)
            {
                if (string.Equals(points[index].DetectionKey, detectionKey, StringComparison.Ordinal))
                {
                    return index;
                }
            }
        }

        for (int index = 0; index < points.Count; index++)
        {
            if (string.Equals(points[index].PointId, pointId, StringComparison.Ordinal))
            {
                return index;
            }
        }

        return -1;
    }

    private static bool RequireTarget(int index, bool requireTarget, string parameterName)
    {
        if (index >= 0)
        {
            return true;
        }

        if (requireTarget)
        {
            throw new ArgumentException("The corrected point does not exist.", parameterName);
        }

        return false;
    }

    private static IReadOnlyList<string> AppendCorrection(
        IReadOnlyList<string> correctionIds,
        string correctionId) =>
        WorkflowCollections.Freeze(correctionIds.Append(correctionId).Distinct(StringComparer.Ordinal));

    private static string MakeUniquePointId(string requestedPointId, HashSet<string> usedPointIds)
    {
        if (!usedPointIds.Contains(requestedPointId))
        {
            return requestedPointId;
        }

        int suffix = 2;
        string candidate = $"{requestedPointId}:rerun:{suffix}";
        while (usedPointIds.Contains(candidate))
        {
            suffix++;
            candidate = $"{requestedPointId}:rerun:{suffix}";
        }

        return candidate;
    }
}
