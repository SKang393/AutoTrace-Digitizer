// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Validation.Scoreboard;

public static class RegressionThresholds
{
    private static readonly IReadOnlyList<RegressionThreshold> Defaults = Array.AsReadOnly(
    [
        Minimum(ScoreboardMetricIds.MarkerCenterPrecision3Px, 0.80),
        Minimum(ScoreboardMetricIds.MarkerCenterRecall3Px, 0.80),
        Minimum(ScoreboardMetricIds.MarkerCenterF13Px, 0.80),
        Minimum(ScoreboardMetricIds.MarkerCenterPrecision5Px, 0.90),
        Minimum(ScoreboardMetricIds.MarkerCenterRecall5Px, 0.90),
        Minimum(ScoreboardMetricIds.MarkerCenterF15Px, 0.90),
        Maximum(ScoreboardMetricIds.DuplicateDetectionRate, 0.05),
        Maximum(ScoreboardMetricIds.FalsePositivesPerMegapixel, 0.50),
        Minimum(ScoreboardMetricIds.ShapeMacroF1, 0.80),
        Minimum(ScoreboardMetricIds.FillStateMacroF1, 0.80),
        Maximum(ScoreboardMetricIds.AxisAnchorErrorPixels, 3.0),
        Maximum(ScoreboardMetricIds.CalibrationRmseGraphUnits, 1.0),
        Minimum(ScoreboardMetricIds.XTickNumericExactMatch, 0.90),
        Minimum(ScoreboardMetricIds.YTickNumericExactMatch, 0.90),
        Maximum(ScoreboardMetricIds.OcrCharacterErrorRate, 0.10),
        Minimum(ScoreboardMetricIds.SeriesAssociationAccuracy, 0.85),
        Minimum(ScoreboardMetricIds.LegendMappingAccuracy, 0.85),
        Maximum(ScoreboardMetricIds.PhaseDividerErrorPixels, 3.0),
        Maximum(ScoreboardMetricIds.PhaseDividerErrorSessions, 1.0),
        Minimum(ScoreboardMetricIds.PhaseAssignmentF1, 0.90),
        Minimum(ScoreboardMetricIds.CsvPointPrecision, 0.95),
        Minimum(ScoreboardMetricIds.CsvPointRecall, 0.95),
        Maximum(ScoreboardMetricIds.YMaeAxisRangePercent, 2.0),
        Minimum(ScoreboardMetricIds.ExactPhaseCodeAccuracy, 0.95),
        Maximum(ScoreboardMetricIds.ColdRuntimeMilliseconds, 6000.0),
        Maximum(ScoreboardMetricIds.WarmRuntimeMilliseconds, 2000.0),
        Maximum(ScoreboardMetricIds.PeakMemoryBytes, 1_073_741_824.0),
        Maximum(ScoreboardMetricIds.ConfidenceExpectedCalibrationError, 0.10),
        Maximum(ScoreboardMetricIds.ConfidenceBrierScore, 0.10),
    ]);

    public static IReadOnlyList<RegressionThreshold> PublicDefaults => Defaults;

    private static RegressionThreshold Minimum(string metricId, double value) =>
        new(metricId, value, null);

    private static RegressionThreshold Maximum(string metricId, double value) =>
        new(metricId, null, value);
}
