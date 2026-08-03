// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Validation.Scoreboard;

public static class ScoreboardMetricIds
{
    public const string MarkerCenterPrecision3Px = "marker_center_precision_3px";
    public const string MarkerCenterRecall3Px = "marker_center_recall_3px";
    public const string MarkerCenterF13Px = "marker_center_f1_3px";
    public const string MarkerCenterPrecision5Px = "marker_center_precision_5px";
    public const string MarkerCenterRecall5Px = "marker_center_recall_5px";
    public const string MarkerCenterF15Px = "marker_center_f1_5px";
    public const string DuplicateDetectionRate = "duplicate_detection_rate";
    public const string FalsePositivesPerMegapixel = "false_positives_per_megapixel";
    public const string ShapeMacroF1 = "shape_macro_f1";
    public const string FillStateMacroF1 = "fill_state_macro_f1";
    public const string AxisAnchorErrorPixels = "axis_anchor_error_px";
    public const string CalibrationRmseGraphUnits = "calibration_rmse_graph_units";
    public const string XTickNumericExactMatch = "x_tick_numeric_exact_match";
    public const string YTickNumericExactMatch = "y_tick_numeric_exact_match";
    public const string OcrCharacterErrorRate = "ocr_character_error_rate";
    public const string SeriesAssociationAccuracy = "series_association_accuracy";
    public const string LegendMappingAccuracy = "legend_mapping_accuracy";
    public const string PhaseDividerErrorPixels = "phase_divider_error_px";
    public const string PhaseDividerErrorSessions = "phase_divider_error_sessions";
    public const string PhaseAssignmentF1 = "phase_assignment_f1";
    public const string CsvPointPrecision = "csv_point_precision";
    public const string CsvPointRecall = "csv_point_recall";
    public const string YMaeAxisRangePercent = "y_mae_axis_range_percent";
    public const string ExactPhaseCodeAccuracy = "exact_phase_code_accuracy";
    public const string ColdRuntimeMilliseconds = "cold_runtime_ms";
    public const string WarmRuntimeMilliseconds = "warm_runtime_ms";
    public const string PeakMemoryBytes = "peak_memory_bytes";
    public const string ConfidenceExpectedCalibrationError = "confidence_expected_calibration_error";
    public const string ConfidenceBrierScore = "confidence_brier_score";

    private static readonly IReadOnlyList<MetricDefinition> Definitions =
    [
        Higher(MarkerCenterPrecision3Px, "Marker center precision at 3 px", "ratio"),
        Higher(MarkerCenterRecall3Px, "Marker center recall at 3 px", "ratio"),
        Higher(MarkerCenterF13Px, "Marker center F1 at 3 px", "ratio"),
        Higher(MarkerCenterPrecision5Px, "Marker center precision at 5 px", "ratio"),
        Higher(MarkerCenterRecall5Px, "Marker center recall at 5 px", "ratio"),
        Higher(MarkerCenterF15Px, "Marker center F1 at 5 px", "ratio"),
        Lower(DuplicateDetectionRate, "Duplicate detection rate", "ratio"),
        Lower(FalsePositivesPerMegapixel, "False positives per megapixel", "count/MP"),
        Higher(ShapeMacroF1, "Shape macro-F1", "ratio"),
        Higher(FillStateMacroF1, "Fill-state macro-F1", "ratio"),
        Lower(AxisAnchorErrorPixels, "Axis-anchor error", "px"),
        Lower(CalibrationRmseGraphUnits, "Calibration RMSE", "graph units"),
        Higher(XTickNumericExactMatch, "X tick numeric exact match", "ratio"),
        Higher(YTickNumericExactMatch, "Y tick numeric exact match", "ratio"),
        Lower(OcrCharacterErrorRate, "OCR character error rate", "ratio"),
        Higher(SeriesAssociationAccuracy, "Series association accuracy", "ratio"),
        Higher(LegendMappingAccuracy, "Legend mapping accuracy", "ratio"),
        Lower(PhaseDividerErrorPixels, "Phase divider error", "px"),
        Lower(PhaseDividerErrorSessions, "Phase divider error", "sessions"),
        Higher(PhaseAssignmentF1, "Phase assignment F1", "ratio"),
        Higher(CsvPointPrecision, "CSV point precision", "ratio"),
        Higher(CsvPointRecall, "CSV point recall", "ratio"),
        Lower(YMaeAxisRangePercent, "Y MAE as axis range", "%"),
        Higher(ExactPhaseCodeAccuracy, "Exact phase-code accuracy", "ratio"),
        Lower(ColdRuntimeMilliseconds, "Cold runtime", "ms"),
        Lower(WarmRuntimeMilliseconds, "Warm runtime", "ms"),
        Lower(PeakMemoryBytes, "Peak memory", "bytes"),
        Lower(ConfidenceExpectedCalibrationError, "Confidence expected calibration error", "ratio"),
        Lower(ConfidenceBrierScore, "Confidence Brier score", "score"),
    ];

    public static IReadOnlyList<MetricDefinition> All => Definitions;

    public static MetricDefinition? Find(string metricId) =>
        Definitions.FirstOrDefault(
            definition => string.Equals(definition.Id, metricId, StringComparison.Ordinal));

    private static MetricDefinition Higher(string id, string name, string unit) =>
        new(id, name, unit, MetricDirection.HigherIsBetter);

    private static MetricDefinition Lower(string id, string name, string unit) =>
        new(id, name, unit, MetricDirection.LowerIsBetter);
}
