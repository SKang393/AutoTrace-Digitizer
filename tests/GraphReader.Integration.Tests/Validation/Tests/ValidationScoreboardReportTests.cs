// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
using GraphReader.Validation.Scoreboard;
using Microsoft.VisualStudio.TestTools.UnitTesting;

#pragma warning disable CA1861 // Small collection expressions are intentional test fixtures.

namespace GraphReader.Integration.Tests.Validation.Tests;

[TestClass]
public sealed class ValidationScoreboardReportTests
{
    [TestMethod]
    public void ValidationJsonMarkdownAndHtmlReportsAreDeterministic()
    {
        CaseMetricRecord firstMetric = new("axis", "case-b", "quality", 0.9, 1);
        CaseMetricRecord secondMetric = new("markers", "case-a", "quality", 0.8, 1);
        PerformanceRecord firstPerformance = new("axis", "case-b", 10, 5, 1024, "fixture");
        PerformanceRecord secondPerformance = new("markers", "case-a", 8, 4, 512, "fixture");
        LicenseManifestValidation firstManifest =
            new("models/z.json", "z-model", true, Array.Empty<string>());
        LicenseManifestValidation secondManifest =
            new("models/a.json", "a-model", true, Array.Empty<string>());

        ScoreboardResult forward = ScoreboardEvaluator.Evaluate(
            CreateInput(
                [firstMetric, secondMetric],
                [firstPerformance, secondPerformance],
                [firstManifest, secondManifest]),
            [new RegressionThreshold("quality", Minimum: 0.5, Maximum: null)]);
        ScoreboardResult reverse = ScoreboardEvaluator.Evaluate(
            CreateInput(
                [secondMetric, firstMetric],
                [secondPerformance, firstPerformance],
                [secondManifest, firstManifest]),
            [new RegressionThreshold("quality", Minimum: 0.5, Maximum: null)]);

        ScoreboardReport first = ScoreboardReportGenerator.Generate(forward);
        ScoreboardReport second = ScoreboardReportGenerator.Generate(reverse);

        Assert.AreEqual(first.Json, second.Json);
        Assert.AreEqual(first.Markdown, second.Markdown);
        Assert.AreEqual(first.Html, second.Html);
        using JsonDocument json = JsonDocument.Parse(first.Json);
        Assert.AreEqual("public", json.RootElement.GetProperty("suite_name").GetString());
        Assert.IsTrue(json.RootElement.TryGetProperty("evidence_scope", out _));
        StringAssert.Contains(first.Markdown, "Evidence scope:");
        StringAssert.Contains(first.Html, "Evidence scope");
        StringAssert.StartsWith(first.Markdown, "# Graph Auto Reader validation scoreboard\n");
        StringAssert.StartsWith(first.Html, "<!doctype html>\n");
        Assert.IsFalse(first.Json.Contains('\r'));
        Assert.IsFalse(first.Markdown.Contains('\r'));
        Assert.IsFalse(first.Html.Contains('\r'));
    }

    [TestMethod]
    public void ValidationScoreboardWriterCreatesAllThreeByteStableReports()
    {
        ScoreboardResult result = ScoreboardEvaluator.Evaluate(
            CreateInput(
                [new CaseMetricRecord("markers", "case-a", "quality", 0.9, 1)],
                Array.Empty<PerformanceRecord>(),
                [new LicenseManifestValidation("models/a.json", "a-model", true, Array.Empty<string>())]),
            [new RegressionThreshold("quality", Minimum: 0.5, Maximum: null)]);
        ScoreboardReport report = ScoreboardReportGenerator.Generate(result);
        string outputDirectory = CreateTemporaryDirectory();

        try
        {
            ScoreboardReportPaths paths = ScoreboardReportWriter.WriteAll(
                outputDirectory,
                report,
                "validation-scoreboard");

            Assert.AreEqual(report.Json, File.ReadAllText(paths.JsonPath));
            Assert.AreEqual(report.Markdown, File.ReadAllText(paths.MarkdownPath));
            Assert.AreEqual(report.Html, File.ReadAllText(paths.HtmlPath));
            Assert.IsTrue(new FileInfo(paths.JsonPath).Length > 0);
            Assert.IsTrue(new FileInfo(paths.MarkdownPath).Length > 0);
            Assert.IsTrue(new FileInfo(paths.HtmlPath).Length > 0);
        }
        finally
        {
            Directory.Delete(outputDirectory, recursive: true);
        }
    }

    [TestMethod]
    public void ValidationPublicBenchmarkSmokeEvaluatesEveryDefaultGate()
    {
        string repositoryRoot = RepositoryRoot.Find();
        ScoreboardInput input = PublicScoreboardFactory.CreateSmokeInput(repositoryRoot);
        IReadOnlyList<RegressionThreshold> thresholds = PublicScoreboardFactory.CreateThresholds();

        ScoreboardResult result = ScoreboardEvaluator.Evaluate(input, thresholds);

        Assert.IsTrue(result.Gate.Passed, FailureSummary(result.Gate.Failures));
        Assert.AreEqual(0, result.Gate.ExitCode);
        Assert.HasCount(thresholds.Count, result.Thresholds);
        StringAssert.Contains(result.EvidenceScope, "not detector accuracy");
        Assert.IsTrue(result.Metrics.Any(metric =>
            metric.AggregateValue is > 0 and < 1));
        Assert.IsTrue(result.Performance.Any(performance =>
            performance.ColdRuntimeMilliseconds > 0 &&
            performance.WarmRuntimeMilliseconds > 0 &&
            performance.PeakMemoryBytes > 0));
        Assert.IsTrue(result.Metrics.All(metric =>
            !thresholds.Any(threshold => threshold.MetricId == metric.Definition.Id) ||
            metric.IsMeasured));
    }

    private static ScoreboardInput CreateInput(
        IReadOnlyList<CaseMetricRecord> metrics,
        IReadOnlyList<PerformanceRecord> performance,
        IReadOnlyList<LicenseManifestValidation> manifests) =>
        new(
            "public",
            "1",
            metrics,
            metrics
                .Select(metric => metric.CaseId)
                .Append("public")
                .Distinct(StringComparer.Ordinal)
                .Order(StringComparer.Ordinal)
                .Select((caseId, index) => new ArticleSplitRecord(
                    "public",
                    $"article-{index + 1}",
                    caseId,
                    DataSplit.Test))
                .ToArray(),
            [
                new GraphReader.Validation.Scoreboard.ConfidenceObservation(
                    "markers",
                    "case-a",
                    0.9,
                    IsCorrect: true),
            ],
            performance,
            manifests);

    private static string FailureSummary(IReadOnlyList<GateFailure> failures) =>
        string.Join(
            Environment.NewLine,
            failures.Select(failure =>
                $"[{failure.ModuleId}/{failure.CaseId}] {failure.CriterionId}: {failure.Message}"));

    private static string CreateTemporaryDirectory() =>
        Directory.CreateDirectory(Path.Combine(
            Path.GetTempPath(),
            $"GraphReader-Report-Validation-{Guid.NewGuid():N}")).FullName;
}
