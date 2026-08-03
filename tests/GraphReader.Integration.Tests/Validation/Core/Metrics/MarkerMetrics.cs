// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests.Validation.Core.Metrics;

public sealed record MarkerObservation(string MarkerId, MetricPoint Center);

public sealed record MarkerMetricCase(
    MetricCaseIdentity Identity,
    double ImageWidthPixels,
    double ImageHeightPixels,
    IReadOnlyList<MarkerObservation> Expected,
    IReadOnlyList<MarkerObservation> Actual);

public sealed record MarkerThresholdScore(
    double TolerancePixels,
    int TruePositives,
    int FalsePositives,
    int FalseNegatives,
    int DuplicateDetections,
    int PredictedCount,
    double Precision,
    double Recall,
    double F1,
    double DuplicateRate,
    double FalsePositivesPerMegapixel);

public sealed record MarkerCaseMetrics(
    MetricCaseIdentity Identity,
    MarkerThresholdScore At3Pixels,
    MarkerThresholdScore At5Pixels,
    OneToOneMatchingResult MatchesAt3Pixels,
    OneToOneMatchingResult MatchesAt5Pixels,
    IReadOnlyList<MetricFailure> Failures);

public sealed record MarkerMetricsReport(
    IReadOnlyList<MarkerCaseMetrics> Cases,
    MarkerThresholdScore At3Pixels,
    MarkerThresholdScore At5Pixels,
    IReadOnlyList<MetricFailure> Failures);

public static class MarkerMetricsCalculator
{
    public const double StrictTolerancePixels = 3;
    public const double LenientTolerancePixels = 5;

    public static MarkerMetricsReport Calculate(IReadOnlyList<MarkerMetricCase> cases)
    {
        ArgumentNullException.ThrowIfNull(cases);

        var results = cases.Select(CalculateCase).ToArray();
        var failures = results.SelectMany(result => result.Failures).ToArray();

        return new MarkerMetricsReport(
            results,
            Aggregate(cases, results.Select(result => result.At3Pixels), StrictTolerancePixels),
            Aggregate(cases, results.Select(result => result.At5Pixels), LenientTolerancePixels),
            failures);
    }

    public static MarkerCaseMetrics CalculateCase(MarkerMetricCase input)
    {
        ArgumentNullException.ThrowIfNull(input);
        MetricGuard.Identity(input.Identity);
        ArgumentNullException.ThrowIfNull(input.Expected);
        ArgumentNullException.ThrowIfNull(input.Actual);
        MetricGuard.NonNegativeFinite(input.ImageWidthPixels, nameof(input.ImageWidthPixels));
        MetricGuard.NonNegativeFinite(input.ImageHeightPixels, nameof(input.ImageHeightPixels));

        if (input.ImageWidthPixels == 0 || input.ImageHeightPixels == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(input), "Image dimensions must be greater than zero.");
        }

        ValidateMarkers(input.Expected, nameof(input.Expected));
        ValidateMarkers(input.Actual, nameof(input.Actual));

        var expectedPoints = input.Expected.Select(marker => marker.Center).ToArray();
        var actualPoints = input.Actual.Select(marker => marker.Center).ToArray();
        var at3 = CalculateThreshold(input, expectedPoints, actualPoints, StrictTolerancePixels);
        var at5 = CalculateThreshold(input, expectedPoints, actualPoints, LenientTolerancePixels);
        var failures = at3.Failures.Concat(at5.Failures).ToArray();

        return new MarkerCaseMetrics(
            input.Identity,
            at3.Score,
            at5.Score,
            at3.Matching,
            at5.Matching,
            failures);
    }

    private static ThresholdCalculation CalculateThreshold(
        MarkerMetricCase input,
        MetricPoint[] expectedPoints,
        MetricPoint[] actualPoints,
        double tolerance)
    {
        var matching = OneToOneMatcher.MatchPoints(expectedPoints, actualPoints, tolerance);
        var duplicateIndices = matching.UnmatchedActualIndices
            .Where(actualIndex => expectedPoints.Any(
                expected => expected.DistanceTo(actualPoints[actualIndex]) <= tolerance))
            .ToHashSet();

        var score = CreateScore(
            tolerance,
            matching.Matches.Count,
            matching.UnmatchedActualIndices.Count,
            matching.UnmatchedExpectedIndices.Count,
            duplicateIndices.Count,
            input.Actual.Count,
            input.ImageWidthPixels * input.ImageHeightPixels);

        var metric = $"marker_{tolerance:0}_px";
        var failures = new List<MetricFailure>();

        foreach (var expectedIndex in matching.UnmatchedExpectedIndices)
        {
            failures.Add(new MetricFailure(
                input.Identity.Module,
                input.Identity.CaseId,
                metric,
                input.Expected[expectedIndex].MarkerId,
                "missing",
                $"No prediction was within {tolerance:0} pixels of the expected marker."));
        }

        foreach (var actualIndex in matching.UnmatchedActualIndices)
        {
            var duplicate = duplicateIndices.Contains(actualIndex);
            failures.Add(new MetricFailure(
                input.Identity.Module,
                input.Identity.CaseId,
                duplicate ? $"{metric}_duplicate" : $"{metric}_false_positive",
                duplicate ? "one prediction per expected marker" : "no marker",
                input.Actual[actualIndex].MarkerId,
                duplicate
                    ? $"The extra prediction was within {tolerance:0} pixels of an expected marker already claimed by a closer one-to-one match."
                    : $"The prediction was not within {tolerance:0} pixels of any expected marker."));
        }

        return new ThresholdCalculation(score, matching, failures);
    }

    private static MarkerThresholdScore Aggregate(
        IReadOnlyList<MarkerMetricCase> cases,
        IEnumerable<MarkerThresholdScore> scores,
        double tolerance)
    {
        var materialized = scores.ToArray();
        return CreateScore(
            tolerance,
            materialized.Sum(score => score.TruePositives),
            materialized.Sum(score => score.FalsePositives),
            materialized.Sum(score => score.FalseNegatives),
            materialized.Sum(score => score.DuplicateDetections),
            materialized.Sum(score => score.PredictedCount),
            cases.Sum(item => item.ImageWidthPixels * item.ImageHeightPixels));
    }

    private static MarkerThresholdScore CreateScore(
        double tolerance,
        int truePositives,
        int falsePositives,
        int falseNegatives,
        int duplicates,
        int predictedCount,
        double imageAreaPixels)
    {
        var precision = MetricMath.Ratio(truePositives, truePositives + falsePositives, whenEmpty: 1);
        var recall = MetricMath.Ratio(truePositives, truePositives + falseNegatives, whenEmpty: 1);

        return new MarkerThresholdScore(
            tolerance,
            truePositives,
            falsePositives,
            falseNegatives,
            duplicates,
            predictedCount,
            precision,
            recall,
            MetricMath.F1(precision, recall),
            MetricMath.Ratio(duplicates, predictedCount),
            imageAreaPixels == 0 ? 0 : falsePositives * 1_000_000d / imageAreaPixels);
    }

    private static void ValidateMarkers(IReadOnlyList<MarkerObservation> markers, string parameterName)
    {
        var identifiers = new HashSet<string>(StringComparer.Ordinal);
        foreach (var marker in markers)
        {
            ArgumentNullException.ThrowIfNull(marker);
            if (string.IsNullOrWhiteSpace(marker.MarkerId))
            {
                throw new ArgumentException("Every marker requires an ID.", parameterName);
            }

            if (!identifiers.Add(marker.MarkerId))
            {
                throw new ArgumentException($"Marker ID '{marker.MarkerId}' is duplicated.", parameterName);
            }

            MetricGuard.Finite(marker.Center.X, parameterName);
            MetricGuard.Finite(marker.Center.Y, parameterName);
        }
    }

    private sealed record ThresholdCalculation(
        MarkerThresholdScore Score,
        OneToOneMatchingResult Matching,
        IReadOnlyList<MetricFailure> Failures);
}
