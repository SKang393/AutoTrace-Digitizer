// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Integration.Tests.Validation.Core.Metrics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

#pragma warning disable CA1861 // Small collection expressions are intentional test fixtures.

namespace GraphReader.Integration.Tests.Validation.Tests;

[TestClass]
public sealed class ValidationMetricCalculatorTests
{
    private static readonly MetricCaseIdentity Identity = new("markers", "public-case-01");

    [TestMethod]
    public void ValidationMarkerMetricsCoverThreeAndFivePixelThresholdsDuplicatesAndFalsePositives()
    {
        MarkerMetricsReport report = MarkerMetricsCalculator.Calculate(
        [
            new MarkerMetricCase(
                Identity,
                ImageWidthPixels: 1000,
                ImageHeightPixels: 1000,
                Expected:
                [
                    new MarkerObservation("expected-1", new MetricPoint(0, 0)),
                    new MarkerObservation("expected-2", new MetricPoint(20, 0)),
                ],
                Actual:
                [
                    new MarkerObservation("actual-match", new MetricPoint(2, 0)),
                    new MarkerObservation("actual-duplicate-at-5", new MetricPoint(4, 0)),
                    new MarkerObservation("actual-false-positive", new MetricPoint(100, 100)),
                ])
        ]);

        Assert.AreEqual(1, report.At3Pixels.TruePositives);
        Assert.AreEqual(2, report.At3Pixels.FalsePositives);
        Assert.AreEqual(1, report.At3Pixels.FalseNegatives);
        Assert.AreEqual(0, report.At3Pixels.DuplicateDetections);
        Assert.AreEqual(2, report.At3Pixels.FalsePositivesPerMegapixel, 1e-12);
        Assert.AreEqual(1d / 3d, report.At3Pixels.Precision, 1e-12);
        Assert.AreEqual(0.5, report.At3Pixels.Recall, 1e-12);
        Assert.AreEqual(0.4, report.At3Pixels.F1, 1e-12);

        Assert.AreEqual(1, report.At5Pixels.DuplicateDetections);
        Assert.AreEqual(1d / 3d, report.At5Pixels.DuplicateRate, 1e-12);
        AssertFailuresIdentifyCase(report.Failures);
        Assert.IsTrue(report.Failures.Any(failure => failure.Metric == "marker_5_px_duplicate"));
    }

    [TestMethod]
    public void ValidationMarkerShapeAndFillMetricsReportIndependentMacroScores()
    {
        MetricOutcome<MarkerClassificationMetrics> outcome =
            MarkerClassificationMetricsCalculator.Calculate(
                Identity,
                [
                    new MarkerClassificationObservation("m1", "circle", "circle", "filled", "filled"),
                    new MarkerClassificationObservation("m2", "square", "circle", "open", "filled"),
                ]);

        Assert.AreEqual(0.5, outcome.Value.Shape.Accuracy, 1e-12);
        Assert.AreEqual(0.5, outcome.Value.Fill.Accuracy, 1e-12);
        Assert.AreEqual(1d / 3d, outcome.Value.Shape.MacroF1, 1e-12);
        Assert.AreEqual(1d / 3d, outcome.Value.Fill.MacroF1, 1e-12);
        Assert.HasCount(2, outcome.Failures);
        AssertFailuresIdentifyCase(outcome.Failures);
    }

    [TestMethod]
    public void ValidationAxisMetricsMeasureAnchorAndCalibrationErrors()
    {
        MetricOutcome<AxisMetrics> outcome = AxisMetricsCalculator.Calculate(
            new AxisMetricInput(
                new MetricCaseIdentity("axis", "axis-case-01"),
                [
                    new AxisAnchorObservation("origin", new MetricPoint(0, 0), new MetricPoint(3, 4)),
                    new AxisAnchorObservation("x-max", new MetricPoint(10, 0), new MetricPoint(10, 0)),
                ],
                [
                    new CalibrationObservation("sample-1", new MetricPoint(1, 2), new MetricPoint(2, 4)),
                    new CalibrationObservation("sample-2", new MetricPoint(0, 0), new MetricPoint(0, 0)),
                ]));

        Assert.AreEqual(2.5, outcome.Value.MeanAnchorErrorPixels, 1e-12);
        Assert.AreEqual(5, outcome.Value.MaximumAnchorErrorPixels, 1e-12);
        Assert.AreEqual(Math.Sqrt(1.25), outcome.Value.CalibrationRmseGraphUnits, 1e-12);
        Assert.AreEqual(2, outcome.Value.AnchorSupport);
        Assert.AreEqual(4, outcome.Value.CalibrationCoordinateSupport);
        Assert.IsEmpty(outcome.Failures);
    }

    [TestMethod]
    public void ValidationOcrMetricsMeasureTickExactMatchAndCharacterErrorRate()
    {
        MetricOutcome<OcrMetrics> outcome = OcrMetricsCalculator.Calculate(
            new OcrMetricInput(
                new MetricCaseIdentity("ocr", "ocr-case-01"),
                [
                    new TickRecognition("x-1", TickAxis.X, "1", "1"),
                    new TickRecognition("x-2", TickAxis.X, "2", "3"),
                    new TickRecognition("y-1", TickAxis.Y, "100", "100"),
                ],
                [
                    new OcrTextRecognition("label-1", "abc", "adc"),
                    new OcrTextRecognition("label-2", "10", null),
                ]));

        Assert.AreEqual(0.5, outcome.Value.XTickNumericExactMatch, 1e-12);
        Assert.AreEqual(1, outcome.Value.YTickNumericExactMatch, 1e-12);
        Assert.AreEqual(3, outcome.Value.CharacterEditCount);
        Assert.AreEqual(0.6, outcome.Value.CharacterErrorRate, 1e-12);
        Assert.HasCount(3, outcome.Failures);
        Assert.IsTrue(outcome.Failures.Any(failure => failure.Metric == "x_tick_numeric_exact_match"));
        AssertFailuresIdentify(outcome.Failures, "ocr", "ocr-case-01");
    }

    [TestMethod]
    public void ValidationGroupingAndLegendMetricsMeasureAssociationsSeparately()
    {
        MetricCaseIdentity identity = new("legends", "association-case-01");
        MetricOutcome<AssociationScore> series = AssociationMetricsCalculator.CalculateSeries(
            identity,
            [
                new SeriesAssociationObservation("point-1", "series-a", "series-a"),
                new SeriesAssociationObservation("point-2", "series-b", "series-a"),
            ]);
        MetricOutcome<AssociationScore> legend = AssociationMetricsCalculator.CalculateLegend(
            identity,
            [
                new LegendMappingObservation("legend-1", "series-a", "series-a"),
                new LegendMappingObservation("legend-2", "series-b", null),
            ]);

        Assert.AreEqual(0.5, series.Value.Accuracy, 1e-12);
        Assert.AreEqual(0.5, legend.Value.Accuracy, 1e-12);
        Assert.AreEqual("series_association_accuracy", AssertFailure(series.Failures).Metric);
        Assert.AreEqual("legend_mapping_accuracy", AssertFailure(legend.Failures).Metric);
    }

    [TestMethod]
    public void ValidationPhaseMetricsMeasureDividerAndExactAssignmentErrors()
    {
        MetricOutcome<PhaseMetrics> outcome = PhaseMetricsCalculator.Calculate(
            new PhaseMetricInput(
                new MetricCaseIdentity("phases", "phase-case-01"),
                [new PhaseDividerObservation("divider-1", 10, 12, 3, 3.5)],
                [
                    new PhaseAssignmentObservation("point-1", "a", "a"),
                    new PhaseAssignmentObservation("point-2", "b", "phase3"),
                ]));

        Assert.AreEqual(2, outcome.Value.MeanDividerErrorPixels, 1e-12);
        Assert.AreEqual(2, outcome.Value.MaximumDividerErrorPixels, 1e-12);
        Assert.AreEqual(0.5, outcome.Value.MeanDividerErrorSessions, 1e-12);
        Assert.AreEqual(0.5, outcome.Value.ExactPhaseCodeAccuracy, 1e-12);
        Assert.AreEqual("phase_assignment_f1", AssertFailure(outcome.Failures).Metric);
    }

    [TestMethod]
    public void ValidationCsvMetricsMeasurePointsValuesAndPhaseCodes()
    {
        MetricOutcome<CsvMetrics> outcome = CsvMetricsCalculator.Calculate(
            new CsvMetricInput(
                new MetricCaseIdentity("export", "csv-case-01"),
                [
                    new CsvPoint("expected-1", 1, 10, "a"),
                    new CsvPoint("expected-2", 2, 20, "b"),
                ],
                [
                    new CsvPoint("actual-1", 1, 11, "a"),
                    new CsvPoint("actual-2", 2, 20, "phase3"),
                    new CsvPoint("actual-3", 3, 30, "phase3"),
                ],
                AxisYRange: 100,
                XTolerance: 0.01,
                YTolerance: 1.1));

        Assert.AreEqual(2, outcome.Value.TruePositivePoints);
        Assert.AreEqual(1, outcome.Value.FalsePositivePoints);
        Assert.AreEqual(0, outcome.Value.FalseNegativePoints);
        Assert.AreEqual(2d / 3d, outcome.Value.PointPrecision, 1e-12);
        Assert.AreEqual(1, outcome.Value.PointRecall, 1e-12);
        Assert.AreEqual(0.5, outcome.Value.YMeanAbsoluteError, 1e-12);
        Assert.AreEqual(0.5, outcome.Value.YMeanAbsoluteErrorPercentAxisRange, 1e-12);
        Assert.AreEqual(0.5, outcome.Value.ExactPhaseCodeAccuracy, 1e-12);
        Assert.HasCount(2, outcome.Failures);
        Assert.IsTrue(outcome.Failures.Any(failure => failure.Metric == "csv_point_precision"));
        Assert.IsTrue(outcome.Failures.Any(failure => failure.Metric == "csv_phase_code_accuracy"));
        AssertFailuresIdentify(outcome.Failures, "export", "csv-case-01");
    }

    [TestMethod]
    public void ValidationConfidenceCalibrationMeasuresReliabilityAndBrierScore()
    {
        MetricOutcome<ConfidenceCalibrationMetrics> outcome =
            ConfidenceCalibrationCalculator.Calculate(
                new MetricCaseIdentity("markers", "confidence-case-01"),
                [
                    new ConfidenceObservation("correct", 0.8, Correct: true),
                    new ConfidenceObservation("incorrect", 0.2, Correct: false),
                ],
                binCount: 2);

        Assert.AreEqual(2, outcome.Value.Support);
        Assert.AreEqual(0.2, outcome.Value.ExpectedCalibrationError, 1e-12);
        Assert.AreEqual(0.2, outcome.Value.MaximumCalibrationError, 1e-12);
        Assert.AreEqual(0.04, outcome.Value.BrierScore, 1e-12);
        Assert.HasCount(2, outcome.Value.Bins);
        Assert.IsEmpty(outcome.Failures);
    }

    private static MetricFailure AssertFailure(IReadOnlyList<MetricFailure> failures)
    {
        Assert.HasCount(1, failures);
        AssertFailuresIdentify(failures, failures[0].Module, failures[0].CaseId);
        return failures[0];
    }

    private static void AssertFailuresIdentifyCase(IReadOnlyList<MetricFailure> failures) =>
        AssertFailuresIdentify(failures, Identity.Module, Identity.CaseId);

    private static void AssertFailuresIdentify(
        IReadOnlyList<MetricFailure> failures,
        string module,
        string caseId)
    {
        Assert.IsNotEmpty(failures);
        Assert.IsTrue(failures.All(failure => failure.Module == module));
        Assert.IsTrue(failures.All(failure => failure.CaseId == caseId));
    }
}
