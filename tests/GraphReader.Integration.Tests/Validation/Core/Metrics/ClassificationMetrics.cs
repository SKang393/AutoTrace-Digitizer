// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests.Validation.Core.Metrics;

public sealed record ClassificationObservation(string ItemId, string Expected, string Actual);

public sealed record ClassScore(
    string Label,
    int TruePositives,
    int FalsePositives,
    int FalseNegatives,
    double Precision,
    double Recall,
    double F1);

public sealed record ClassificationScore(
    int Support,
    int Correct,
    double Accuracy,
    double MacroF1,
    IReadOnlyList<ClassScore> Classes);

public sealed record MarkerClassificationObservation(
    string MarkerId,
    string ExpectedShape,
    string ActualShape,
    string ExpectedFill,
    string ActualFill);

public sealed record MarkerClassificationMetrics(
    ClassificationScore Shape,
    ClassificationScore Fill);

public static class ClassificationMetricsCalculator
{
    public static MetricOutcome<ClassificationScore> Calculate(
        MetricCaseIdentity identity,
        string metric,
        IReadOnlyList<ClassificationObservation> observations)
    {
        MetricGuard.Identity(identity);
        if (string.IsNullOrWhiteSpace(metric))
        {
            throw new ArgumentException("A metric name is required.", nameof(metric));
        }

        ArgumentNullException.ThrowIfNull(observations);
        ValidateObservations(observations);

        var labels = observations
            .SelectMany(observation => new[] { observation.Expected, observation.Actual })
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToArray();

        var classScores = labels.Select(label => CalculateClass(label, observations)).ToArray();
        var correct = observations.Count(observation =>
            string.Equals(observation.Expected, observation.Actual, StringComparison.Ordinal));
        var failures = observations
            .Where(observation => !string.Equals(observation.Expected, observation.Actual, StringComparison.Ordinal))
            .Select(observation => new MetricFailure(
                identity.Module,
                identity.CaseId,
                metric,
                observation.Expected,
                observation.Actual,
                $"Item '{observation.ItemId}' was assigned the wrong class."))
            .ToArray();

        var score = new ClassificationScore(
            observations.Count,
            correct,
            MetricMath.Ratio(correct, observations.Count, whenEmpty: 1),
            classScores.Length == 0 ? 1 : classScores.Average(item => item.F1),
            classScores);

        return new MetricOutcome<ClassificationScore>(score, failures);
    }

    private static ClassScore CalculateClass(
        string label,
        IReadOnlyList<ClassificationObservation> observations)
    {
        var truePositives = observations.Count(observation =>
            string.Equals(observation.Expected, label, StringComparison.Ordinal) &&
            string.Equals(observation.Actual, label, StringComparison.Ordinal));
        var falsePositives = observations.Count(observation =>
            !string.Equals(observation.Expected, label, StringComparison.Ordinal) &&
            string.Equals(observation.Actual, label, StringComparison.Ordinal));
        var falseNegatives = observations.Count(observation =>
            string.Equals(observation.Expected, label, StringComparison.Ordinal) &&
            !string.Equals(observation.Actual, label, StringComparison.Ordinal));
        var precision = MetricMath.Ratio(truePositives, truePositives + falsePositives, whenEmpty: 1);
        var recall = MetricMath.Ratio(truePositives, truePositives + falseNegatives, whenEmpty: 1);

        return new ClassScore(
            label,
            truePositives,
            falsePositives,
            falseNegatives,
            precision,
            recall,
            MetricMath.F1(precision, recall));
    }

    private static void ValidateObservations(IReadOnlyList<ClassificationObservation> observations)
    {
        var identifiers = new HashSet<string>(StringComparer.Ordinal);
        foreach (var observation in observations)
        {
            ArgumentNullException.ThrowIfNull(observation);
            if (string.IsNullOrWhiteSpace(observation.ItemId) ||
                string.IsNullOrWhiteSpace(observation.Expected) ||
                string.IsNullOrWhiteSpace(observation.Actual))
            {
                throw new ArgumentException("Classification item IDs and labels must be non-empty.", nameof(observations));
            }

            if (!identifiers.Add(observation.ItemId))
            {
                throw new ArgumentException($"Classification item ID '{observation.ItemId}' is duplicated.", nameof(observations));
            }
        }
    }
}

public static class MarkerClassificationMetricsCalculator
{
    public static MetricOutcome<MarkerClassificationMetrics> Calculate(
        MetricCaseIdentity identity,
        IReadOnlyList<MarkerClassificationObservation> observations)
    {
        MetricGuard.Identity(identity);
        ArgumentNullException.ThrowIfNull(observations);

        foreach (var observation in observations)
        {
            ArgumentNullException.ThrowIfNull(observation);
        }

        var shape = ClassificationMetricsCalculator.Calculate(
            identity,
            "marker_shape_macro_f1",
            observations.Select(observation => new ClassificationObservation(
                observation.MarkerId,
                observation.ExpectedShape,
                observation.ActualShape)).ToArray());
        var fill = ClassificationMetricsCalculator.Calculate(
            identity,
            "marker_fill_macro_f1",
            observations.Select(observation => new ClassificationObservation(
                observation.MarkerId,
                observation.ExpectedFill,
                observation.ActualFill)).ToArray());

        return new MetricOutcome<MarkerClassificationMetrics>(
            new MarkerClassificationMetrics(shape.Value, fill.Value),
            shape.Failures.Concat(fill.Failures).ToArray());
    }
}
