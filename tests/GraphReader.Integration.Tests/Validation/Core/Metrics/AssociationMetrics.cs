// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests.Validation.Core.Metrics;

public sealed record SeriesAssociationObservation(
    string PointId,
    string ExpectedSeriesId,
    string? ActualSeriesId);

public sealed record LegendMappingObservation(
    string LegendItemId,
    string ExpectedSeriesId,
    string? ActualSeriesId);

public sealed record AssociationScore(int Support, int Correct, double Accuracy);

public static class AssociationMetricsCalculator
{
    public static MetricOutcome<AssociationScore> CalculateSeries(
        MetricCaseIdentity identity,
        IReadOnlyList<SeriesAssociationObservation> observations)
    {
        ArgumentNullException.ThrowIfNull(observations);
        foreach (var observation in observations)
        {
            ArgumentNullException.ThrowIfNull(observation);
        }

        return Calculate(
            identity,
            "series_association_accuracy",
            observations.Select(observation => new AssociationItem(
                observation.PointId,
                observation.ExpectedSeriesId,
                observation.ActualSeriesId)).ToArray());
    }

    public static MetricOutcome<AssociationScore> CalculateLegend(
        MetricCaseIdentity identity,
        IReadOnlyList<LegendMappingObservation> observations)
    {
        ArgumentNullException.ThrowIfNull(observations);
        foreach (var observation in observations)
        {
            ArgumentNullException.ThrowIfNull(observation);
        }

        return Calculate(
            identity,
            "legend_mapping_accuracy",
            observations.Select(observation => new AssociationItem(
                observation.LegendItemId,
                observation.ExpectedSeriesId,
                observation.ActualSeriesId)).ToArray());
    }

    private static MetricOutcome<AssociationScore> Calculate(
        MetricCaseIdentity identity,
        string metric,
        IReadOnlyList<AssociationItem> observations)
    {
        MetricGuard.Identity(identity);
        var identifiers = new HashSet<string>(StringComparer.Ordinal);
        var correct = 0;
        var failures = new List<MetricFailure>();

        foreach (var observation in observations)
        {
            if (string.IsNullOrWhiteSpace(observation.ItemId) ||
                string.IsNullOrWhiteSpace(observation.Expected))
            {
                throw new ArgumentException("Association item IDs and expected series IDs are required.", nameof(observations));
            }

            if (!identifiers.Add(observation.ItemId))
            {
                throw new ArgumentException($"Association item ID '{observation.ItemId}' is duplicated.", nameof(observations));
            }

            if (string.Equals(observation.Expected, observation.Actual, StringComparison.Ordinal))
            {
                correct++;
                continue;
            }

            failures.Add(new MetricFailure(
                identity.Module,
                identity.CaseId,
                metric,
                observation.Expected,
                observation.Actual ?? "missing",
                $"Association item '{observation.ItemId}' was mapped to the wrong series."));
        }

        return new MetricOutcome<AssociationScore>(
            new AssociationScore(
                observations.Count,
                correct,
                MetricMath.Ratio(correct, observations.Count, whenEmpty: 1)),
            failures);
    }

    private sealed record AssociationItem(string ItemId, string Expected, string? Actual);
}
