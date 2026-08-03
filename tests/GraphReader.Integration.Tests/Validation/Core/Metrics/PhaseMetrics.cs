// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests.Validation.Core.Metrics;

public sealed record PhaseDividerObservation(
    string DividerId,
    double ExpectedPixelX,
    double? ActualPixelX,
    double? ExpectedSession,
    double? ActualSession);

public sealed record PhaseAssignmentObservation(
    string PointId,
    string ExpectedPhaseCode,
    string ActualPhaseCode);

public sealed record PhaseDividerError(
    string DividerId,
    double? PixelError,
    double? SessionError);

public sealed record PhaseMetricInput(
    MetricCaseIdentity Identity,
    IReadOnlyList<PhaseDividerObservation> Dividers,
    IReadOnlyList<PhaseAssignmentObservation> Assignments);

public sealed record PhaseMetrics(
    double MeanDividerErrorPixels,
    double MaximumDividerErrorPixels,
    double MeanDividerErrorSessions,
    double MaximumDividerErrorSessions,
    ClassificationScore Assignment,
    double ExactPhaseCodeAccuracy,
    IReadOnlyList<PhaseDividerError> DividerErrors);

public static class PhaseMetricsCalculator
{
    public static MetricOutcome<PhaseMetrics> Calculate(PhaseMetricInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        MetricGuard.Identity(input.Identity);
        ArgumentNullException.ThrowIfNull(input.Dividers);
        ArgumentNullException.ThrowIfNull(input.Assignments);

        var failures = new List<MetricFailure>();
        var dividerErrors = new List<PhaseDividerError>();
        var dividerIds = new HashSet<string>(StringComparer.Ordinal);

        foreach (var divider in input.Dividers)
        {
            ArgumentNullException.ThrowIfNull(divider);
            if (string.IsNullOrWhiteSpace(divider.DividerId) || !dividerIds.Add(divider.DividerId))
            {
                throw new ArgumentException("Phase divider IDs must be non-empty and unique.", nameof(input));
            }

            MetricGuard.Finite(divider.ExpectedPixelX, nameof(input.Dividers));
            if (divider.ExpectedSession.HasValue)
            {
                MetricGuard.Finite(divider.ExpectedSession.Value, nameof(input.Dividers));
            }

            if (divider.ActualSession.HasValue)
            {
                MetricGuard.Finite(divider.ActualSession.Value, nameof(input.Dividers));
            }

            double? pixelError = null;
            double? sessionError = null;
            if (divider.ActualPixelX.HasValue)
            {
                MetricGuard.Finite(divider.ActualPixelX.Value, nameof(input.Dividers));
                pixelError = Math.Abs(divider.ActualPixelX.Value - divider.ExpectedPixelX);
            }
            else
            {
                failures.Add(MissingDividerFailure(input.Identity, divider.DividerId, "phase_divider_error_pixels"));
            }

            if (divider.ExpectedSession.HasValue && divider.ActualSession.HasValue)
            {
                sessionError = Math.Abs(divider.ActualSession.Value - divider.ExpectedSession.Value);
            }
            else if (divider.ExpectedSession.HasValue)
            {
                failures.Add(MissingDividerFailure(input.Identity, divider.DividerId, "phase_divider_error_sessions"));
            }

            dividerErrors.Add(new PhaseDividerError(divider.DividerId, pixelError, sessionError));
        }

        foreach (var assignmentItem in input.Assignments)
        {
            ArgumentNullException.ThrowIfNull(assignmentItem);
        }

        var assignment = ClassificationMetricsCalculator.Calculate(
            input.Identity,
            "phase_assignment_f1",
            input.Assignments.Select(item => new ClassificationObservation(
                item.PointId,
                item.ExpectedPhaseCode,
                item.ActualPhaseCode)).ToArray());
        failures.AddRange(assignment.Failures);

        var pixelErrors = dividerErrors
            .Where(error => error.PixelError.HasValue)
            .Select(error => error.PixelError!.Value)
            .ToArray();
        var sessionErrors = dividerErrors
            .Where(error => error.SessionError.HasValue)
            .Select(error => error.SessionError!.Value)
            .ToArray();
        var score = new PhaseMetrics(
            pixelErrors.Length == 0 ? 0 : pixelErrors.Average(),
            pixelErrors.Length == 0 ? 0 : pixelErrors.Max(),
            sessionErrors.Length == 0 ? 0 : sessionErrors.Average(),
            sessionErrors.Length == 0 ? 0 : sessionErrors.Max(),
            assignment.Value,
            assignment.Value.Accuracy,
            dividerErrors);

        return new MetricOutcome<PhaseMetrics>(score, failures);
    }

    private static MetricFailure MissingDividerFailure(
        MetricCaseIdentity identity,
        string dividerId,
        string metric)
        => new(
            identity.Module,
            identity.CaseId,
            metric,
            dividerId,
            "missing",
            $"Phase divider '{dividerId}' has no measured location.");
}
