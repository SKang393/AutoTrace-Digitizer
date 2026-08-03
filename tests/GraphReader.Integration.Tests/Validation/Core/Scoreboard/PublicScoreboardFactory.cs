// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using GraphReader.Integration.Tests.Validation.Core.Metrics;

namespace GraphReader.Validation.Scoreboard;

public static class PublicScoreboardFactory
{
    public static ScoreboardInput CreateSmokeInput() =>
        CreateSmokeInput(FindRepositoryRoot());

    public static ScoreboardInput CreateInput(string repositoryRoot) =>
        CreateSmokeInput(repositoryRoot);

    public static ScoreboardInput CreateSmokeInput(
        string repositoryRoot,
        PerformanceRecord? measuredPerformance = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repositoryRoot);

        List<CaseMetricRecord> metrics = CalculatePublicMetrics();

        GraphReader.Validation.Scoreboard.ConfidenceObservation[] confidence =
        [
            new("markers", "synthetic-marker-01", 0.99, true),
            new("markers", "synthetic-marker-02", 0.99, true),
            new("axis", "synthetic-axis-01", 0.99, true),
            new("ocr", "synthetic-ocr-01", 0.01, false),
            new("phases", "synthetic-phase-01", 0.01, false),
        ];

        PerformanceRecord performance = measuredPerformance ?? MeasureCalculatorPerformance();
        ArticleSplitRecord[] splits = metrics.Select(metric => metric.CaseId)
            .Concat(confidence.Select(observation => observation.CaseId))
            .Append(performance.CaseId)
            .Append("public")
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .Select((caseId, index) => new ArticleSplitRecord(
                "public-synthetic-contract-smoke-v1",
                $"synthetic-article-{index + 1:D2}",
                caseId,
                DataSplit.Test))
            .ToArray();

        return new ScoreboardInput(
            "public",
            "1",
            metrics,
            splits,
            confidence,
            [performance],
            LicenseManifestValidator.ValidateRepository(repositoryRoot),
            "Synthetic metric-contract smoke only. It validates calculators, gates, reports, and manifest-policy plumbing; it is not detector accuracy or end-to-end application performance evidence.");
    }

    public static IReadOnlyList<RegressionThreshold> CreateThresholds() =>
        RegressionThresholds.PublicDefaults;

    public static ScoreboardResult Evaluate(
        string repositoryRoot,
        PerformanceRecord? measuredPerformance = null) =>
        ScoreboardEvaluator.Evaluate(
            CreateSmokeInput(repositoryRoot, measuredPerformance),
            RegressionThresholds.PublicDefaults);

    private static List<CaseMetricRecord> CalculatePublicMetrics()
    {
        List<CaseMetricRecord> metrics = [];
        AddMarkerMetrics(metrics);
        AddClassificationMetrics(metrics);
        AddAxisMetrics(metrics);
        AddOcrMetrics(metrics);
        AddAssociationMetrics(metrics);
        AddPhaseMetrics(metrics);
        AddCsvMetrics(metrics);
        return metrics;
    }

    private static PerformanceRecord MeasureCalculatorPerformance()
    {
        long memoryBefore = GC.GetTotalMemory(forceFullCollection: false);
        Stopwatch stopwatch = Stopwatch.StartNew();
        _ = CalculatePublicMetrics();
        stopwatch.Stop();
        double coldMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
        long memoryAfterCold = GC.GetTotalMemory(forceFullCollection: false);

        stopwatch.Restart();
        _ = CalculatePublicMetrics();
        stopwatch.Stop();
        double warmMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
        long memoryAfterWarm = GC.GetTotalMemory(forceFullCollection: false);

        return new PerformanceRecord(
            "scoreboard",
            "public-calculator-smoke",
            coldMilliseconds,
            warmMilliseconds,
            Math.Max(memoryBefore, Math.Max(memoryAfterCold, memoryAfterWarm)),
            $"live managed-memory sample; {Environment.OSVersion.Platform}; {Environment.ProcessorCount} logical processors");
    }

    private static void AddMarkerMetrics(List<CaseMetricRecord> metrics)
    {
        MetricCaseIdentity identity = new("markers", "metric-contract-smoke-markers");
        MarkerObservation[] truth =
        [
            new("truth-1", new MetricPoint(10, 10)),
            new("truth-2", new MetricPoint(20, 20)),
            new("truth-3", new MetricPoint(30, 30)),
            new("truth-4", new MetricPoint(40, 40)),
        ];
        MarkerObservation[] prediction =
        [
            new("prediction-1", new MetricPoint(11, 10)),
            new("prediction-2", new MetricPoint(20, 21)),
            new("prediction-3", new MetricPoint(29, 30)),
            new("prediction-4", new MetricPoint(40, 39)),
        ];
        MarkerCaseMetrics calculated = MarkerMetricsCalculator.CalculateCase(
            new MarkerMetricCase(identity, 1000, 1000, truth, prediction));

        Add(metrics, identity, ScoreboardMetricIds.MarkerCenterPrecision3Px, calculated.At3Pixels.Precision, truth.Length);
        Add(metrics, identity, ScoreboardMetricIds.MarkerCenterRecall3Px, calculated.At3Pixels.Recall, truth.Length);
        Add(metrics, identity, ScoreboardMetricIds.MarkerCenterF13Px, calculated.At3Pixels.F1, truth.Length);
        Add(metrics, identity, ScoreboardMetricIds.MarkerCenterPrecision5Px, calculated.At5Pixels.Precision, truth.Length);
        Add(metrics, identity, ScoreboardMetricIds.MarkerCenterRecall5Px, calculated.At5Pixels.Recall, truth.Length);
        Add(metrics, identity, ScoreboardMetricIds.MarkerCenterF15Px, calculated.At5Pixels.F1, truth.Length);
        Add(metrics, identity, ScoreboardMetricIds.DuplicateDetectionRate, calculated.At3Pixels.DuplicateRate, prediction.Length);
        Add(metrics, identity, ScoreboardMetricIds.FalsePositivesPerMegapixel, calculated.At3Pixels.FalsePositivesPerMegapixel, prediction.Length);
    }

    private static void AddClassificationMetrics(List<CaseMetricRecord> metrics)
    {
        MetricCaseIdentity identity = new("markers", "metric-contract-smoke-classification");
        MarkerClassificationObservation[] observations =
        [
            new("marker-1", "circle", "circle", "filled", "filled"),
            new("marker-2", "circle", "circle", "open", "open"),
            new("marker-3", "square", "square", "filled", "filled"),
            new("marker-4", "square", "square", "open", "open"),
        ];
        MarkerClassificationMetrics calculated = MarkerClassificationMetricsCalculator
            .Calculate(identity, observations).Value;
        Add(metrics, identity, ScoreboardMetricIds.ShapeMacroF1, calculated.Shape.MacroF1, observations.Length);
        Add(metrics, identity, ScoreboardMetricIds.FillStateMacroF1, calculated.Fill.MacroF1, observations.Length);
    }

    private static void AddAxisMetrics(List<CaseMetricRecord> metrics)
    {
        MetricCaseIdentity identity = new("axis", "metric-contract-smoke-axis");
        AxisMetrics calculated = AxisMetricsCalculator.Calculate(new AxisMetricInput(
            identity,
            [
                new("session1_y0", new MetricPoint(100, 300), new MetricPoint(101, 300)),
                new("session1_ymax", new MetricPoint(100, 50), new MetricPoint(100, 51)),
                new("sessionmax_y0", new MetricPoint(700, 300), new MetricPoint(699, 300)),
            ],
            [
                new("sample-1", new MetricPoint(1, 0), new MetricPoint(1, 0)),
                new("sample-2", new MetricPoint(1, 100), new MetricPoint(1, 100)),
                new("sample-3", new MetricPoint(24, 0), new MetricPoint(24, 0)),
            ])).Value;
        Add(metrics, identity, ScoreboardMetricIds.AxisAnchorErrorPixels, calculated.MeanAnchorErrorPixels, calculated.AnchorSupport);
        Add(metrics, identity, ScoreboardMetricIds.CalibrationRmseGraphUnits, calculated.CalibrationRmseGraphUnits, calculated.CalibrationCoordinateSupport);
    }

    private static void AddOcrMetrics(List<CaseMetricRecord> metrics)
    {
        MetricCaseIdentity identity = new("ocr", "metric-contract-smoke-ocr");
        OcrMetrics calculated = OcrMetricsCalculator.Calculate(new OcrMetricInput(
            identity,
            [
                new("x-1", TickAxis.X, "1", "1"),
                new("x-24", TickAxis.X, "24", "24"),
                new("y-0", TickAxis.Y, "0", "0"),
                new("y-100", TickAxis.Y, "100", "100"),
            ],
            [
                new("phase-a", "Baseline", "Baseline"),
                new("phase-b", "Intervention", "Intervention"),
            ])).Value;
        Add(metrics, identity, ScoreboardMetricIds.XTickNumericExactMatch, calculated.XTickNumericExactMatch, calculated.XTickSupport);
        Add(metrics, identity, ScoreboardMetricIds.YTickNumericExactMatch, calculated.YTickNumericExactMatch, calculated.YTickSupport);
        Add(metrics, identity, ScoreboardMetricIds.OcrCharacterErrorRate, calculated.CharacterErrorRate, calculated.ExpectedCharacterCount);
    }

    private static void AddAssociationMetrics(List<CaseMetricRecord> metrics)
    {
        MetricCaseIdentity seriesIdentity = new("markers", "metric-contract-smoke-grouping");
        AssociationScore series = AssociationMetricsCalculator.CalculateSeries(
            seriesIdentity,
            [
                new("point-1", "series-a", "series-a"),
                new("point-2", "series-a", "series-a"),
                new("point-3", "series-b", "series-b"),
            ]).Value;
        Add(metrics, seriesIdentity, ScoreboardMetricIds.SeriesAssociationAccuracy, series.Accuracy, series.Support);

        MetricCaseIdentity legendIdentity = new("legends", "metric-contract-smoke-legends");
        AssociationScore legend = AssociationMetricsCalculator.CalculateLegend(
            legendIdentity,
            [
                new("legend-1", "series-a", "series-a"),
                new("legend-2", "series-b", "series-b"),
            ]).Value;
        Add(metrics, legendIdentity, ScoreboardMetricIds.LegendMappingAccuracy, legend.Accuracy, legend.Support);
    }

    private static void AddPhaseMetrics(List<CaseMetricRecord> metrics)
    {
        MetricCaseIdentity identity = new("phases", "metric-contract-smoke-phases");
        PhaseMetrics calculated = PhaseMetricsCalculator.Calculate(new PhaseMetricInput(
            identity,
            [
                new("divider-1", 300, 301, 10, 10.25),
                new("divider-2", 500, 499, 18, 17.75),
            ],
            [
                new("point-1", "a", "a"),
                new("point-2", "a", "a"),
                new("point-3", "b", "b"),
                new("point-4", "phase3", "phase3"),
            ])).Value;
        Add(metrics, identity, ScoreboardMetricIds.PhaseDividerErrorPixels, calculated.MeanDividerErrorPixels, calculated.DividerErrors.Count);
        Add(metrics, identity, ScoreboardMetricIds.PhaseDividerErrorSessions, calculated.MeanDividerErrorSessions, calculated.DividerErrors.Count);
        Add(metrics, identity, ScoreboardMetricIds.PhaseAssignmentF1, calculated.Assignment.MacroF1, calculated.Assignment.Support);
    }

    private static void AddCsvMetrics(List<CaseMetricRecord> metrics)
    {
        MetricCaseIdentity identity = new("export", "metric-contract-smoke-csv");
        CsvPoint[] truth =
        [
            new("truth-1", 1, 20, "a"),
            new("truth-2", 2, 30, "a"),
            new("truth-3", 3, 40, "b"),
        ];
        CsvPoint[] prediction =
        [
            new("prediction-1", 1, 20.25, "a"),
            new("prediction-2", 2, 29.75, "a"),
            new("prediction-3", 3, 40.25, "b"),
        ];
        CsvMetrics calculated = CsvMetricsCalculator.Calculate(
            new CsvMetricInput(
                identity,
                truth,
                prediction,
                AxisYRange: 100,
                YTolerance: 1)).Value;
        Add(metrics, identity, ScoreboardMetricIds.CsvPointPrecision, calculated.PointPrecision, calculated.ActualPointCount);
        Add(metrics, identity, ScoreboardMetricIds.CsvPointRecall, calculated.PointRecall, calculated.ExpectedPointCount);
        Add(metrics, identity, ScoreboardMetricIds.YMaeAxisRangePercent, calculated.YMeanAbsoluteErrorPercentAxisRange, calculated.RowComparisons.Count);
        Add(metrics, identity, ScoreboardMetricIds.ExactPhaseCodeAccuracy, calculated.ExactPhaseCodeAccuracy, calculated.ExpectedPointCount);
    }

    private static void Add(
        List<CaseMetricRecord> metrics,
        MetricCaseIdentity identity,
        string metricId,
        double value,
        int support) =>
        metrics.Add(new CaseMetricRecord(
            identity.Module,
            identity.CaseId,
            metricId,
            value,
            support,
            "calculated from explicit deterministic synthetic truth and prediction observations"));

    private static string FindRepositoryRoot()
    {
        foreach (string start in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        {
            DirectoryInfo? directory = new(Path.GetFullPath(start));
            while (directory is not null)
            {
                if (File.Exists(Path.Combine(directory.FullName, "GraphAutoReader.slnx")) &&
                    Directory.Exists(Path.Combine(directory.FullName, "models", "manifest")))
                {
                    return directory.FullName;
                }

                directory = directory.Parent;
            }
        }

        throw new DirectoryNotFoundException(
            "Could not locate the Graph Auto Reader repository root.");
    }
}
