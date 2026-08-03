// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Validation.Scoreboard;

public enum DataSplit
{
    Train,
    Validation,
    Test,
}

public enum MetricDirection
{
    HigherIsBetter,
    LowerIsBetter,
}

public sealed record MetricDefinition(
    string Id,
    string DisplayName,
    string Unit,
    MetricDirection Direction);

public sealed record CaseMetricRecord(
    string ModuleId,
    string CaseId,
    string MetricId,
    double? Value,
    int SampleCount = 0,
    string? Note = null);

public sealed record ArticleSplitRecord(
    string DatasetId,
    string ArticleId,
    string CaseId,
    DataSplit Split);

public sealed record ArticleSplitLeak(
    string DatasetId,
    string ArticleId,
    IReadOnlyList<DataSplit> Splits,
    IReadOnlyList<string> CaseIds);

public sealed record ArticleSplitMetadataIssue(
    string CaseId,
    string Message);

public sealed record ArticleSplitValidation(
    bool IsValid,
    IReadOnlyList<ArticleSplitLeak> Leaks,
    IReadOnlyList<ArticleSplitMetadataIssue> MetadataIssues);

public sealed record ConfidenceObservation(
    string ModuleId,
    string CaseId,
    double Confidence,
    bool IsCorrect);

public sealed record CalibrationBin(
    int Index,
    double LowerBound,
    double UpperBound,
    int Count,
    double? MeanConfidence,
    double? Accuracy,
    double? AbsoluteGap);

public sealed record ConfidenceCalibrationReport(
    int ObservationCount,
    double? ExpectedCalibrationError,
    double? BrierScore,
    IReadOnlyList<CalibrationBin> Bins);

public sealed record PerformanceRecord(
    string ModuleId,
    string CaseId,
    double? ColdRuntimeMilliseconds,
    double? WarmRuntimeMilliseconds,
    long? PeakMemoryBytes,
    string? Machine = null);

public sealed record LicenseManifestValidation(
    string ManifestPath,
    string ComponentId,
    bool IsValid,
    IReadOnlyList<string> Issues);

public sealed record RegressionThreshold(
    string MetricId,
    double? Minimum,
    double? Maximum,
    bool Required = true);

public sealed record GateFailure(
    string ModuleId,
    string CaseId,
    string CriterionId,
    double? Actual,
    double? Minimum,
    double? Maximum,
    string Message);

public sealed record GateOutcome(
    bool Passed,
    int ExitCode,
    IReadOnlyList<GateFailure> Failures);

public sealed record MetricSummary(
    MetricDefinition Definition,
    bool IsMeasured,
    double? AggregateValue,
    int SampleCount,
    IReadOnlyList<CaseMetricRecord> Cases);

public sealed record ScoreboardInput(
    string SuiteName,
    string SuiteVersion,
    IReadOnlyList<CaseMetricRecord> Metrics,
    IReadOnlyList<ArticleSplitRecord> ArticleSplits,
    IReadOnlyList<ConfidenceObservation> ConfidenceObservations,
    IReadOnlyList<PerformanceRecord> Performance,
    IReadOnlyList<LicenseManifestValidation> LicenseManifests,
    string EvidenceScope = "Unspecified validation evidence.");

public sealed record ScoreboardResult(
    string SuiteName,
    string SuiteVersion,
    string EvidenceScope,
    IReadOnlyList<MetricSummary> Metrics,
    IReadOnlyList<ArticleSplitRecord> ArticleSplitMetadata,
    ArticleSplitValidation ArticleSplits,
    ConfidenceCalibrationReport ConfidenceCalibration,
    IReadOnlyList<PerformanceRecord> Performance,
    IReadOnlyList<LicenseManifestValidation> LicenseManifests,
    IReadOnlyList<RegressionThreshold> Thresholds,
    GateOutcome Gate);

public sealed record ScoreboardReport(
    string Json,
    string Markdown,
    string Html);

public sealed record ScoreboardReportPaths(
    string JsonPath,
    string MarkdownPath,
    string HtmlPath);
