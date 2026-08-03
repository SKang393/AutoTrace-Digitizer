// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests.Validation.Core.Metrics;

public sealed record ConfidenceObservation(string ItemId, double Confidence, bool Correct);

public sealed record ConfidenceBin(
    int Index,
    double LowerBoundInclusive,
    double UpperBoundInclusive,
    int Count,
    double MeanConfidence,
    double Accuracy,
    double CalibrationGap);

public sealed record ConfidenceCalibrationMetrics(
    int Support,
    double ExpectedCalibrationError,
    double MaximumCalibrationError,
    double BrierScore,
    IReadOnlyList<ConfidenceBin> Bins);

public static class ConfidenceCalibrationCalculator
{
    public static MetricOutcome<ConfidenceCalibrationMetrics> Calculate(
        MetricCaseIdentity identity,
        IReadOnlyList<ConfidenceObservation> observations,
        int binCount = 10)
    {
        MetricGuard.Identity(identity);
        ArgumentNullException.ThrowIfNull(observations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(binCount);

        var identifiers = new HashSet<string>(StringComparer.Ordinal);
        var bins = Enumerable.Range(0, binCount)
            .Select(_ => new List<ConfidenceObservation>())
            .ToArray();
        foreach (var observation in observations)
        {
            ArgumentNullException.ThrowIfNull(observation);
            if (string.IsNullOrWhiteSpace(observation.ItemId) || !identifiers.Add(observation.ItemId))
            {
                throw new ArgumentException("Confidence observation IDs must be non-empty and unique.", nameof(observations));
            }

            MetricGuard.Probability(observation.Confidence, nameof(observations));
            var binIndex = Math.Min((int)(observation.Confidence * binCount), binCount - 1);
            bins[binIndex].Add(observation);
        }

        var scores = bins.Select((items, index) =>
        {
            var meanConfidence = items.Count == 0 ? 0 : items.Average(item => item.Confidence);
            var accuracy = items.Count == 0 ? 0 : items.Count(item => item.Correct) / (double)items.Count;
            return new ConfidenceBin(
                index,
                index / (double)binCount,
                (index + 1) / (double)binCount,
                items.Count,
                meanConfidence,
                accuracy,
                Math.Abs(meanConfidence - accuracy));
        }).ToArray();
        var expectedCalibrationError = observations.Count == 0
            ? 0
            : scores.Sum(bin => bin.CalibrationGap * bin.Count / observations.Count);
        var maximumCalibrationError = scores.Where(bin => bin.Count > 0)
            .Select(bin => bin.CalibrationGap)
            .DefaultIfEmpty(0)
            .Max();
        var brierScore = observations.Count == 0
            ? 0
            : observations.Average(item =>
            {
                var outcome = item.Correct ? 1d : 0d;
                var difference = item.Confidence - outcome;
                return difference * difference;
            });

        return new MetricOutcome<ConfidenceCalibrationMetrics>(
            new ConfidenceCalibrationMetrics(
                observations.Count,
                expectedCalibrationError,
                maximumCalibrationError,
                brierScore,
                scores),
            Array.Empty<MetricFailure>());
    }
}
