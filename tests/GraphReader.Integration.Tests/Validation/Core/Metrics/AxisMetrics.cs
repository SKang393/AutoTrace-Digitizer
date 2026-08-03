// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests.Validation.Core.Metrics;

public sealed record AxisAnchorObservation(
    string AnchorId,
    MetricPoint Expected,
    MetricPoint? Actual);

public sealed record CalibrationObservation(
    string SampleId,
    MetricPoint ExpectedGraphValue,
    MetricPoint? ActualGraphValue);

public sealed record AxisMetricInput(
    MetricCaseIdentity Identity,
    IReadOnlyList<AxisAnchorObservation> Anchors,
    IReadOnlyList<CalibrationObservation> CalibrationSamples);

public sealed record AxisAnchorError(string AnchorId, double? ErrorPixels);

public sealed record CalibrationError(
    string SampleId,
    double? XErrorGraphUnits,
    double? YErrorGraphUnits);

public sealed record AxisMetrics(
    double MeanAnchorErrorPixels,
    double MaximumAnchorErrorPixels,
    double CalibrationRmseGraphUnits,
    int AnchorSupport,
    int CalibrationCoordinateSupport,
    IReadOnlyList<AxisAnchorError> AnchorErrors,
    IReadOnlyList<CalibrationError> CalibrationErrors);

public static class AxisMetricsCalculator
{
    public static MetricOutcome<AxisMetrics> Calculate(AxisMetricInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        MetricGuard.Identity(input.Identity);
        ArgumentNullException.ThrowIfNull(input.Anchors);
        ArgumentNullException.ThrowIfNull(input.CalibrationSamples);

        var failures = new List<MetricFailure>();
        var anchorErrors = new List<AxisAnchorError>();
        var calibrationErrors = new List<CalibrationError>();
        var seenAnchorIds = new HashSet<string>(StringComparer.Ordinal);
        var seenSampleIds = new HashSet<string>(StringComparer.Ordinal);

        foreach (var anchor in input.Anchors)
        {
            ArgumentNullException.ThrowIfNull(anchor);
            ValidateIdentifier(anchor.AnchorId, seenAnchorIds, nameof(input.Anchors));
            ValidatePoint(anchor.Expected, nameof(input.Anchors));
            if (anchor.Actual is null)
            {
                anchorErrors.Add(new AxisAnchorError(anchor.AnchorId, null));
                failures.Add(MissingFailure(input.Identity, "axis_anchor_error_pixels", anchor.AnchorId));
                continue;
            }

            ValidatePoint(anchor.Actual.Value, nameof(input.Anchors));
            anchorErrors.Add(new AxisAnchorError(anchor.AnchorId, anchor.Expected.DistanceTo(anchor.Actual.Value)));
        }

        foreach (var sample in input.CalibrationSamples)
        {
            ArgumentNullException.ThrowIfNull(sample);
            ValidateIdentifier(sample.SampleId, seenSampleIds, nameof(input.CalibrationSamples));
            ValidatePoint(sample.ExpectedGraphValue, nameof(input.CalibrationSamples));
            if (sample.ActualGraphValue is null)
            {
                calibrationErrors.Add(new CalibrationError(sample.SampleId, null, null));
                failures.Add(MissingFailure(input.Identity, "calibration_rmse_graph_units", sample.SampleId));
                continue;
            }

            ValidatePoint(sample.ActualGraphValue.Value, nameof(input.CalibrationSamples));
            calibrationErrors.Add(new CalibrationError(
                sample.SampleId,
                sample.ActualGraphValue.Value.X - sample.ExpectedGraphValue.X,
                sample.ActualGraphValue.Value.Y - sample.ExpectedGraphValue.Y));
        }

        var measuredAnchorErrors = anchorErrors
            .Where(error => error.ErrorPixels.HasValue)
            .Select(error => error.ErrorPixels!.Value)
            .ToArray();
        var coordinateErrors = calibrationErrors
            .SelectMany(error => new[] { error.XErrorGraphUnits, error.YErrorGraphUnits })
            .Where(error => error.HasValue)
            .Select(error => error!.Value)
            .ToArray();
        var score = new AxisMetrics(
            measuredAnchorErrors.Length == 0 ? 0 : measuredAnchorErrors.Average(),
            measuredAnchorErrors.Length == 0 ? 0 : measuredAnchorErrors.Max(),
            coordinateErrors.Length == 0
                ? 0
                : Math.Sqrt(coordinateErrors.Average(error => error * error)),
            measuredAnchorErrors.Length,
            coordinateErrors.Length,
            anchorErrors,
            calibrationErrors);

        return new MetricOutcome<AxisMetrics>(score, failures);
    }

    private static void ValidateIdentifier(string identifier, HashSet<string> seen, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(identifier))
        {
            throw new ArgumentException("An observation ID is required.", parameterName);
        }

        if (!seen.Add(identifier))
        {
            throw new ArgumentException($"Observation ID '{identifier}' is duplicated.", parameterName);
        }
    }

    private static void ValidatePoint(MetricPoint point, string parameterName)
    {
        MetricGuard.Finite(point.X, parameterName);
        MetricGuard.Finite(point.Y, parameterName);
    }

    private static MetricFailure MissingFailure(
        MetricCaseIdentity identity,
        string metric,
        string observationId)
        => new(
            identity.Module,
            identity.CaseId,
            metric,
            observationId,
            "missing",
            $"Observation '{observationId}' has no measured value.");
}
