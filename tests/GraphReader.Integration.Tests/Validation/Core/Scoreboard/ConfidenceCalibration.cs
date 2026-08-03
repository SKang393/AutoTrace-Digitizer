// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Validation.Scoreboard;

public static class ConfidenceCalibration
{
    public static ConfidenceCalibrationReport Calculate(
        IEnumerable<ConfidenceObservation> observations,
        int binCount = 10)
    {
        ArgumentNullException.ThrowIfNull(observations);
        ArgumentOutOfRangeException.ThrowIfLessThan(binCount, 1);

        ConfidenceObservation[] materialized = observations
            .OrderBy(observation => observation.ModuleId, StringComparer.Ordinal)
            .ThenBy(observation => observation.CaseId, StringComparer.Ordinal)
            .ThenBy(observation => observation.Confidence)
            .ThenBy(observation => observation.IsCorrect)
            .ToArray();

        foreach (ConfidenceObservation observation in materialized)
        {
            if (!double.IsFinite(observation.Confidence) ||
                observation.Confidence < 0 ||
                observation.Confidence > 1)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(observations),
                    observation.Confidence,
                    $"Confidence for {observation.ModuleId}/{observation.CaseId} must be within [0, 1].");
            }
        }

        CalibrationBin[] bins = Enumerable.Range(0, binCount)
            .Select(index => BuildBin(index, binCount, materialized))
            .ToArray();

        if (materialized.Length == 0)
        {
            return new ConfidenceCalibrationReport(0, null, null, bins);
        }

        double expectedCalibrationError = bins
            .Where(bin => bin.Count > 0)
            .Sum(bin => bin.AbsoluteGap!.Value * bin.Count / materialized.Length);
        double brierScore = materialized.Average(
            observation => Math.Pow(observation.Confidence - (observation.IsCorrect ? 1 : 0), 2));

        return new ConfidenceCalibrationReport(
            materialized.Length,
            expectedCalibrationError,
            brierScore,
            bins);
    }

    private static CalibrationBin BuildBin(
        int index,
        int binCount,
        IReadOnlyList<ConfidenceObservation> observations)
    {
        double lowerBound = (double)index / binCount;
        double upperBound = (double)(index + 1) / binCount;
        ConfidenceObservation[] members = observations
            .Where(observation => BinIndex(observation.Confidence, binCount) == index)
            .ToArray();

        if (members.Length == 0)
        {
            return new CalibrationBin(
                index,
                lowerBound,
                upperBound,
                0,
                null,
                null,
                null);
        }

        double meanConfidence = members.Average(observation => observation.Confidence);
        double accuracy = members.Count(observation => observation.IsCorrect) / (double)members.Length;
        return new CalibrationBin(
            index,
            lowerBound,
            upperBound,
            members.Length,
            meanConfidence,
            accuracy,
            Math.Abs(meanConfidence - accuracy));
    }

    private static int BinIndex(double confidence, int binCount) =>
        Math.Min((int)(confidence * binCount), binCount - 1);
}
