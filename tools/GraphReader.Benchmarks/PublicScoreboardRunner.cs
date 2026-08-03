// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using GraphReader.Validation.Scoreboard;

namespace GraphReader.Benchmarks;

internal sealed record PublicGateFailure(
    string Module,
    string CaseId,
    string Gate,
    string Detail);

internal sealed record PublicScoreboardResult(
    bool Passed,
    int PassedGateCount,
    int TotalGateCount,
    double ElapsedMilliseconds,
    long PeakManagedMemoryBytes,
    IReadOnlyList<string> ReportPaths,
    IReadOnlyList<PublicGateFailure> Failures);

internal static class PublicScoreboardRunner
{
    public static Task<PublicScoreboardResult> RunAsync(
        string repositoryRoot,
        string outputDirectory,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        using PeakManagedMemorySampler memorySampler = new();
        long suiteStarted = Stopwatch.GetTimestamp();
        long coldStarted = Stopwatch.GetTimestamp();
        ScoreboardInput input = PublicScoreboardFactory.CreateSmokeInput(repositoryRoot);
        IReadOnlyList<RegressionThreshold> thresholds = PublicScoreboardFactory.CreateThresholds();
        _ = ScoreboardEvaluator.Evaluate(input, thresholds);
        double coldMilliseconds = Stopwatch.GetElapsedTime(coldStarted).TotalMilliseconds;

        cancellationToken.ThrowIfCancellationRequested();
        long warmStarted = Stopwatch.GetTimestamp();
        _ = ScoreboardEvaluator.Evaluate(input, thresholds);
        double warmMilliseconds = Stopwatch.GetElapsedTime(warmStarted).TotalMilliseconds;
        memorySampler.Observe();
        memorySampler.Dispose();
        long peakManagedMemoryBytes = memorySampler.PeakBytes;

        PerformanceRecord measuredPerformance = new(
            "validation_harness",
            "scoreboard-construction",
            coldMilliseconds,
            warmMilliseconds,
            peakManagedMemoryBytes,
            "local scoreboard process only; excludes image processing and inference");
        ScoreboardInput measuredInput = PublicScoreboardFactory.CreateSmokeInput(
            repositoryRoot,
            measuredPerformance);
        ScoreboardResult scoreboard = ScoreboardEvaluator.Evaluate(measuredInput, thresholds);
        ScoreboardReport report = ScoreboardReportGenerator.Generate(scoreboard);
        ScoreboardReportPaths paths = ScoreboardReportWriter.WriteAll(
            outputDirectory,
            report,
            "public-scoreboard");
        IReadOnlyList<string> reportPaths =
        [
            paths.JsonPath,
            paths.MarkdownPath,
            paths.HtmlPath,
        ];

        double elapsedMilliseconds = Stopwatch.GetElapsedTime(suiteStarted).TotalMilliseconds;
        PublicGateFailure[] failures = scoreboard.Gate.Failures
            .Select(failure => new PublicGateFailure(
                failure.ModuleId,
                failure.CaseId,
                failure.CriterionId,
                failure.Message))
            .ToArray();
        int totalGateCount = thresholds.Count + scoreboard.LicenseManifests.Count + 1;
        int passedGateCount = Math.Max(0, totalGateCount - failures.Length);

        return Task.FromResult(new PublicScoreboardResult(
            scoreboard.Gate.Passed,
            passedGateCount,
            totalGateCount,
            elapsedMilliseconds,
            peakManagedMemoryBytes,
            reportPaths,
            failures));
    }
}
