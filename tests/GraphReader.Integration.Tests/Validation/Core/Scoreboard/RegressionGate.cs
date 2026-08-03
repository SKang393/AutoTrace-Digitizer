// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;

namespace GraphReader.Validation.Scoreboard;

public static class RegressionGate
{
    public static GateOutcome Evaluate(
        ScoreboardInput input,
        IReadOnlyList<RegressionThreshold> thresholds,
        int calibrationBinCount = 10) =>
        ScoreboardEvaluator.Evaluate(input, thresholds, calibrationBinCount).Gate;

    internal static GateOutcome EvaluateCore(
        IReadOnlyList<CaseMetricRecord> metrics,
        IReadOnlyList<RegressionThreshold> thresholds,
        ArticleSplitValidation splitValidation,
        IReadOnlyList<LicenseManifestValidation> licenseManifests)
    {
        List<GateFailure> failures = [];

        ValidateThresholds(metrics, thresholds, failures);
        AddSplitFailures(splitValidation, failures);
        AddLicenseFailures(licenseManifests, failures);

        GateFailure[] orderedFailures = failures
            .OrderBy(failure => failure.ModuleId, StringComparer.Ordinal)
            .ThenBy(failure => failure.CaseId, StringComparer.Ordinal)
            .ThenBy(failure => failure.CriterionId, StringComparer.Ordinal)
            .ThenBy(failure => failure.Message, StringComparer.Ordinal)
            .ToArray();

        return new GateOutcome(
            orderedFailures.Length == 0,
            orderedFailures.Length == 0 ? 0 : 1,
            orderedFailures);
    }

    private static void ValidateThresholds(
        IReadOnlyList<CaseMetricRecord> metrics,
        IReadOnlyList<RegressionThreshold> thresholds,
        List<GateFailure> failures)
    {
        foreach (RegressionThreshold threshold in thresholds
                     .OrderBy(item => item.MetricId, StringComparer.Ordinal))
        {
            if (threshold.Minimum is not null && threshold.Maximum is not null &&
                threshold.Minimum > threshold.Maximum)
            {
                failures.Add(new GateFailure(
                    "scoreboard",
                    "thresholds",
                    threshold.MetricId,
                    null,
                    threshold.Minimum,
                    threshold.Maximum,
                    "Minimum threshold exceeds maximum threshold."));
                continue;
            }

            CaseMetricRecord[] matching = metrics
                .Where(metric => string.Equals(
                    metric.MetricId,
                    threshold.MetricId,
                    StringComparison.Ordinal))
                .OrderBy(metric => metric.ModuleId, StringComparer.Ordinal)
                .ThenBy(metric => metric.CaseId, StringComparer.Ordinal)
                .ToArray();

            if (matching.Length == 0)
            {
                if (threshold.Required)
                {
                    failures.Add(new GateFailure(
                        "scoreboard",
                        "public-suite",
                        threshold.MetricId,
                        null,
                        threshold.Minimum,
                        threshold.Maximum,
                        "Required metric was not reported."));
                }

                continue;
            }

            foreach (CaseMetricRecord metric in matching)
            {
                AddMetricFailure(metric, threshold, failures);
            }
        }
    }

    private static void AddMetricFailure(
        CaseMetricRecord metric,
        RegressionThreshold threshold,
        List<GateFailure> failures)
    {
        if (threshold.Required && metric.SampleCount <= 0)
        {
            failures.Add(new GateFailure(
                metric.ModuleId,
                metric.CaseId,
                threshold.MetricId,
                metric.Value,
                threshold.Minimum,
                threshold.Maximum,
                "Required metric has zero sample support."));
            return;
        }

        if (metric.Value is null || !double.IsFinite(metric.Value.Value))
        {
            if (threshold.Required)
            {
                failures.Add(new GateFailure(
                    metric.ModuleId,
                    metric.CaseId,
                    threshold.MetricId,
                    metric.Value,
                    threshold.Minimum,
                    threshold.Maximum,
                    "Required metric is unmeasured or non-finite."));
            }

            return;
        }

        if (threshold.Minimum is not null && metric.Value < threshold.Minimum)
        {
            failures.Add(new GateFailure(
                metric.ModuleId,
                metric.CaseId,
                threshold.MetricId,
                metric.Value,
                threshold.Minimum,
                threshold.Maximum,
                FormattableString.Invariant(
                    $"Value {metric.Value.Value:G17} is below minimum {threshold.Minimum.Value:G17}.")));
        }

        if (threshold.Maximum is not null && metric.Value > threshold.Maximum)
        {
            failures.Add(new GateFailure(
                metric.ModuleId,
                metric.CaseId,
                threshold.MetricId,
                metric.Value,
                threshold.Minimum,
                threshold.Maximum,
                string.Format(
                    CultureInfo.InvariantCulture,
                    "Value {0:G17} exceeds maximum {1:G17}.",
                    metric.Value.Value,
                    threshold.Maximum.Value)));
        }
    }

    private static void AddSplitFailures(
        ArticleSplitValidation validation,
        List<GateFailure> failures)
    {
        foreach (ArticleSplitMetadataIssue issue in validation.MetadataIssues)
        {
            failures.Add(new GateFailure(
                "data_split",
                issue.CaseId,
                "article_split_metadata",
                null,
                1,
                1,
                issue.Message));
        }

        foreach (ArticleSplitLeak leak in validation.Leaks)
        {
            failures.Add(new GateFailure(
                "data_split",
                string.Join(",", leak.CaseIds),
                "article_split_leakage",
                leak.Splits.Count,
                null,
                1,
                $"Article '{leak.DatasetId}/{leak.ArticleId}' appears in splits: " +
                string.Join(", ", leak.Splits)));
        }
    }

    private static void AddLicenseFailures(
        IReadOnlyList<LicenseManifestValidation> manifests,
        List<GateFailure> failures)
    {
        if (manifests.Count == 0)
        {
            failures.Add(new GateFailure(
                "release",
                "model-manifests",
                "license_manifest_validation",
                null,
                1,
                null,
                "No model license manifest validation results were reported."));
            return;
        }

        foreach (LicenseManifestValidation manifest in manifests.Where(item => !item.IsValid))
        {
            failures.Add(new GateFailure(
                "release",
                manifest.ManifestPath,
                "license_manifest_validation",
                0,
                1,
                null,
                manifest.Issues.Count == 0
                    ? $"License manifest for '{manifest.ComponentId}' is invalid."
                    : string.Join("; ", manifest.Issues)));
        }
    }
}
