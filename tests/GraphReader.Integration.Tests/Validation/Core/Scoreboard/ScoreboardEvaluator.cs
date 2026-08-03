// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Validation.Scoreboard;

public static class ScoreboardEvaluator
{
    public static ScoreboardResult Evaluate(
        ScoreboardInput input,
        IReadOnlyList<RegressionThreshold> thresholds,
        int calibrationBinCount = 10)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(thresholds);

        IEnumerable<string> scoredCaseIds = input.Metrics.Select(metric => metric.CaseId)
            .Concat(input.Performance.Select(performance => performance.CaseId))
            .Concat(input.ConfidenceObservations.Select(observation => observation.CaseId));
        if (input.ConfidenceObservations.Count > 0)
        {
            scoredCaseIds = scoredCaseIds.Append(input.SuiteName);
        }

        ArticleSplitValidation splitValidation = ArticleSplitValidator.Validate(
            input.ArticleSplits,
            scoredCaseIds);
        ArticleSplitRecord[] orderedSplitMetadata = input.ArticleSplits
            .OrderBy(record => record.DatasetId, StringComparer.Ordinal)
            .ThenBy(record => record.ArticleId, StringComparer.Ordinal)
            .ThenBy(record => record.Split)
            .ThenBy(record => record.CaseId, StringComparer.Ordinal)
            .ToArray();
        ConfidenceCalibrationReport calibration = ConfidenceCalibration.Calculate(
            input.ConfidenceObservations,
            calibrationBinCount);

        CaseMetricRecord[] normalizedMetrics = NormalizeMetrics(input, calibration);
        MetricSummary[] summaries = BuildSummaries(normalizedMetrics);
        RegressionThreshold[] orderedThresholds = thresholds
            .OrderBy(threshold => threshold.MetricId, StringComparer.Ordinal)
            .ToArray();
        LicenseManifestValidation[] orderedManifests = input.LicenseManifests
            .OrderBy(manifest => manifest.ManifestPath, StringComparer.Ordinal)
            .ThenBy(manifest => manifest.ComponentId, StringComparer.Ordinal)
            .ToArray();
        PerformanceRecord[] orderedPerformance = input.Performance
            .OrderBy(record => record.ModuleId, StringComparer.Ordinal)
            .ThenBy(record => record.CaseId, StringComparer.Ordinal)
            .ToArray();
        GateOutcome gate = RegressionGate.EvaluateCore(
            normalizedMetrics,
            orderedThresholds,
            splitValidation,
            orderedManifests);

        return new ScoreboardResult(
            input.SuiteName,
            input.SuiteVersion,
            input.EvidenceScope,
            summaries,
            orderedSplitMetadata,
            splitValidation,
            calibration,
            orderedPerformance,
            orderedManifests,
            orderedThresholds,
            gate);
    }

    private static CaseMetricRecord[] NormalizeMetrics(
        ScoreboardInput input,
        ConfidenceCalibrationReport calibration)
    {
        List<CaseMetricRecord> metrics = [.. input.Metrics];

        foreach (PerformanceRecord performance in input.Performance)
        {
            metrics.Add(new CaseMetricRecord(
                performance.ModuleId,
                performance.CaseId,
                ScoreboardMetricIds.ColdRuntimeMilliseconds,
                performance.ColdRuntimeMilliseconds,
                performance.ColdRuntimeMilliseconds is null ? 0 : 1,
                performance.Machine));
            metrics.Add(new CaseMetricRecord(
                performance.ModuleId,
                performance.CaseId,
                ScoreboardMetricIds.WarmRuntimeMilliseconds,
                performance.WarmRuntimeMilliseconds,
                performance.WarmRuntimeMilliseconds is null ? 0 : 1,
                performance.Machine));
            metrics.Add(new CaseMetricRecord(
                performance.ModuleId,
                performance.CaseId,
                ScoreboardMetricIds.PeakMemoryBytes,
                performance.PeakMemoryBytes,
                performance.PeakMemoryBytes is null ? 0 : 1,
                performance.Machine));
        }

        metrics.Add(new CaseMetricRecord(
            "confidence",
            input.SuiteName,
            ScoreboardMetricIds.ConfidenceExpectedCalibrationError,
            calibration.ExpectedCalibrationError,
            calibration.ObservationCount));
        metrics.Add(new CaseMetricRecord(
            "confidence",
            input.SuiteName,
            ScoreboardMetricIds.ConfidenceBrierScore,
            calibration.BrierScore,
            calibration.ObservationCount));

        return metrics
            .OrderBy(metric => metric.MetricId, StringComparer.Ordinal)
            .ThenBy(metric => metric.ModuleId, StringComparer.Ordinal)
            .ThenBy(metric => metric.CaseId, StringComparer.Ordinal)
            .ThenBy(metric => metric.Value)
            .ToArray();
    }

    private static MetricSummary[] BuildSummaries(IReadOnlyList<CaseMetricRecord> metrics)
    {
        Dictionary<string, MetricDefinition> definitions = ScoreboardMetricIds.All
            .ToDictionary(definition => definition.Id, StringComparer.Ordinal);

        foreach (string metricId in metrics.Select(metric => metric.MetricId).Distinct(StringComparer.Ordinal))
        {
            definitions.TryAdd(
                metricId,
                new MetricDefinition(
                    metricId,
                    metricId,
                    "value",
                    MetricDirection.HigherIsBetter));
        }

        return definitions.Values
            .OrderBy(definition => definition.Id, StringComparer.Ordinal)
            .Select(definition => BuildSummary(
                definition,
                metrics.Where(metric => string.Equals(
                        metric.MetricId,
                        definition.Id,
                        StringComparison.Ordinal))
                    .ToArray()))
            .ToArray();
    }

    private static MetricSummary BuildSummary(
        MetricDefinition definition,
        IReadOnlyList<CaseMetricRecord> cases)
    {
        CaseMetricRecord[] measured = cases
            .Where(metric => metric.Value is not null && double.IsFinite(metric.Value.Value))
            .ToArray();
        int totalWeight = measured.Sum(metric => Math.Max(metric.SampleCount, 1));
        double? aggregate = measured.Length == 0
            ? null
            : measured.Sum(metric => metric.Value!.Value * Math.Max(metric.SampleCount, 1)) /
              totalWeight;

        return new MetricSummary(
            definition,
            measured.Length > 0,
            aggregate,
            measured.Sum(metric => metric.SampleCount),
            cases);
    }
}
