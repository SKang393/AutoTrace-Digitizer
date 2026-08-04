// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.App.Localization;

public static class LocalizationKeys
{
    public const string AppTitle = "App.Title";
    public const string WorkflowImport = "Workflow.Import";
    public const string WorkflowEnhance = "Workflow.Enhance";
    public const string WorkflowAutoDetect = "Workflow.AutoDetect";
    public const string WorkflowReview = "Workflow.Review";
    public const string WorkflowExport = "Workflow.Export";
    public const string WorkflowImportToolTip = "Workflow.Import.ToolTip";
    public const string WorkflowEnhanceToolTip = "Workflow.Enhance.ToolTip";
    public const string WorkflowAutoDetectToolTip = "Workflow.AutoDetect.ToolTip";
    public const string WorkflowReviewToolTip = "Workflow.Review.ToolTip";
    public const string WorkflowExportToolTip = "Workflow.Export.ToolTip";
    public const string WorkflowEnhanceUnavailable = "Workflow.Enhance.Unavailable";
    public const string WorkflowEnhanceExperimental = "Workflow.Enhance.Experimental";
    public const string WorkflowAutoDetectUnavailable = "Workflow.AutoDetect.Unavailable";
    public const string ApplicationPathsUnavailable = "Errors.ApplicationPathsUnavailable";
    public const string ApplicationDataNotWritable = "Errors.ApplicationDataNotWritable";
    public const string PortableDataNotWritable = "Errors.PortableDataNotWritable";
    public const string ProductionWorkflowUnavailable = "Errors.ProductionWorkflowUnavailable";
    public const string EnhancementConfigurationInvalid = "Errors.EnhancementConfigurationInvalid";
    public const string EnhancementModelIncompatible = "Errors.EnhancementModelIncompatible";
    public const string EnhancementRuntimeUnavailable = "Errors.EnhancementRuntimeUnavailable";
    public const string EnhancementRuntimeChecksumMismatch = "Errors.EnhancementRuntimeChecksumMismatch";
    public const string EnhancementRedistributionBlocked = "Errors.EnhancementRedistributionBlocked";
    public const string EnhancementStorageUnavailable = "Errors.EnhancementStorageUnavailable";
    public const string EnhancementExecutionFailed = "Errors.EnhancementExecutionFailed";
    public const string ModelManifestNotFound = "Errors.ModelManifestNotFound";
    public const string ModelManifestInvalid = "Errors.ModelManifestInvalid";
    public const string ModelNoticeNotFound = "Errors.ModelNoticeNotFound";
    public const string ModelNotFound = "Errors.ModelNotFound";
    public const string ModelChecksumMismatch = "Errors.ModelChecksumMismatch";
    public const string EnhancementLocalEvaluationOnly = "Warnings.EnhancementLocalEvaluationOnly";
    public const string PreviewDevelopment = "Preview.Development";
    public const string PreviewVersion = "Preview.Version";
    public const string PreviewCommit = "Preview.Commit";
    public const string PreviewRuntime = "Preview.Runtime";
    public const string PreviewAvailableStages = "Preview.AvailableStages";
    public const string PreviewMissingStages = "Preview.MissingStages";
    public const string PreviewNone = "Preview.None";
    public const string PreviewUnknown = "Preview.Unknown";
    public const string ProjectOpen = "Project.Open";
    public const string ProjectSave = "Project.Save";
    public const string ProjectSaveAs = "Project.SaveAs";
    public const string ProjectRecover = "Project.Recover";
    public const string ProjectCloseTab = "Project.CloseTab";
    public const string ProjectCloseDirtyBlocked = "Project.CloseDirtyBlocked";
    public const string ManualTitle = "Manual.Title";
    public const string ManualInstruction = "Manual.Instruction";
    public const string ManualCalibration = "Manual.Calibration";
    public const string ManualYMaximum = "Manual.YMaximum";
    public const string ManualXMaximum = "Manual.XMaximum";
    public const string ManualSeriesName = "Manual.SeriesName";
    public const string ManualSeriesShape = "Manual.SeriesShape";
    public const string ManualSeriesFill = "Manual.SeriesFill";
    public const string ManualSeriesRole = "Manual.SeriesRole";
    public const string ManualCreateSeries = "Manual.CreateSeries";
    public const string ManualEditSeries = "Manual.EditSeries";
    public const string ManualAddPoint = "Manual.AddPoint";
    public const string ManualAddFilledPoint = "Manual.AddFilledPoint";
    public const string ManualAddOpenPoint = "Manual.AddOpenPoint";
    public const string ManualMovePoint = "Manual.MovePoint";
    public const string ManualDeletePoint = "Manual.DeletePoint";
    public const string ManualPhaseCode = "Manual.PhaseCode";
    public const string ManualPhaseLabel = "Manual.PhaseLabel";
    public const string ManualAddDivider = "Manual.AddDivider";
    public const string ManualMoveDivider = "Manual.MoveDivider";
    public const string ManualDeleteDivider = "Manual.DeleteDivider";
    public const string ManualLabelDivider = "Manual.LabelDivider";
    public const string ManualSharedBaseline = "Manual.SharedBaseline";
    public const string ManualApplicableProbes = "Manual.ApplicableProbes";
    public const string ManualNoSharedBaseline = "Manual.NoSharedBaseline";
    public const string ManualApplySeriesRelations = "Manual.ApplySeriesRelations";
    public const string ManualRelationsApplied = "Manual.Status.RelationsApplied";
    public const string ManualDefaultIntervention = "Manual.Default.Intervention";
    public const string ManualInstructionCalibration = "Manual.Instruction.Calibration";
    public const string ManualInstructionAddPoint = "Manual.Instruction.AddPoint";
    public const string ManualInstructionMovePoint = "Manual.Instruction.MovePoint";
    public const string ManualInstructionAddDivider = "Manual.Instruction.AddDivider";
    public const string ManualInstructionMoveDivider = "Manual.Instruction.MoveDivider";
    public const string ManualInstructionSelect = "Manual.Instruction.Select";
    public const string ManualCalibrationPrompt = "Manual.Status.CalibrationPrompt";
    public const string ManualCalibrationSaved = "Manual.Status.CalibrationSaved";
    public const string ManualSelectSeriesFirst = "Manual.Status.SelectSeriesFirst";
    public const string ManualPointAdded = "Manual.Status.PointAdded";
    public const string ManualPointMoved = "Manual.Status.PointMoved";
    public const string ManualPointSelected = "Manual.Status.PointSelected";
    public const string ManualDividerAdded = "Manual.Status.DividerAdded";
    public const string ManualDividerMoved = "Manual.Status.DividerMoved";
    public const string ManualDividerDeleted = "Manual.Status.DividerDeleted";
    public const string ManualDividerLabeled = "Manual.Status.DividerLabeled";
    public const string ManualDividerSelected = "Manual.Status.DividerSelected";
    public const string ManualSeriesCreatedFormat = "Manual.Status.SeriesCreatedFormat";
    public const string ManualSeriesSelectedFormat = "Manual.Status.SeriesSelectedFormat";
    public const string ManualSeriesNameRequired = "Manual.Status.SeriesNameRequired";
    public const string ManualPhaseLabelRequired = "Manual.Status.PhaseLabelRequired";
    public const string MarkerShapeCircle = "MarkerShape.Circle";
    public const string MarkerShapeSquare = "MarkerShape.Square";
    public const string MarkerShapeTriangleUp = "MarkerShape.TriangleUp";
    public const string MarkerShapeTriangleDown = "MarkerShape.TriangleDown";
    public const string MarkerShapeDiamond = "MarkerShape.Diamond";
    public const string MarkerShapeStar = "MarkerShape.Star";
    public const string MarkerShapeAsterisk = "MarkerShape.Asterisk";
    public const string MarkerShapeCross = "MarkerShape.Cross";
    public const string MarkerShapeOther = "MarkerShape.Other";
    public const string MarkerFillFilled = "MarkerFill.Filled";
    public const string MarkerFillOpen = "MarkerFill.Open";
    public const string MarkerFillUnknown = "MarkerFill.Unknown";
    public const string SemanticRoleBaseline = "SemanticRole.Baseline";
    public const string SemanticRoleIntervention = "SemanticRole.Intervention";
    public const string SemanticRoleMaintenance = "SemanticRole.Maintenance";
    public const string SemanticRoleGeneralization = "SemanticRole.Generalization";
    public const string SemanticRoleUnknown = "SemanticRole.Unknown";
    public const string StatusImportedFormat = "Status.ImportedFormat";
    public const string StatusImportFailuresFormat = "Status.ImportFailuresFormat";
    public const string StatusOpenedFormat = "Status.OpenedFormat";
    public const string StatusSavedFormat = "Status.SavedFormat";
    public const string StatusRecoveredFormat = "Status.RecoveredFormat";
    public const string StatusExportedFormat = "Status.ExportedFormat";
    public const string StatusCancelled = "Status.Cancelled";
    public const string StatusManualEditRejectedFormat = "Status.ManualEditRejectedFormat";
    public const string StatusEnhancementReadyFormat = "Status.EnhancementReadyFormat";
    public const string DialogGraphImages = "Dialog.GraphImages";
    public const string DialogAllFiles = "Dialog.AllFiles";
    public const string DialogProjectFiles = "Dialog.ProjectFiles";
    public const string DialogExportFolder = "Dialog.ExportFolder";
    public const string NavigationTitle = "Navigation.Title";
    public const string NavigationGraphs = "Navigation.Graphs";
    public const string NavigationEmpty = "Navigation.Empty";
    public const string NavigationAutomationName = "Navigation.AutomationName";
    public const string CanvasEmpty = "Canvas.Empty";
    public const string CanvasPhaseOverlay = "Canvas.PhaseOverlay";
    public const string CanvasZoomIn = "Canvas.ZoomIn";
    public const string CanvasZoomOut = "Canvas.ZoomOut";
    public const string CanvasFit = "Canvas.Fit";
    public const string CanvasResetView = "Canvas.ResetView";
    public const string GraphCanvasAutomationName = "GraphCanvas.AutomationName";
    public const string GraphCanvasEmptyState = "GraphCanvas.EmptyState";
    public const string GraphCanvasPhaseOverlayAutomationName =
        "GraphCanvas.PhaseOverlay.AutomationName";
    public const string GraphCanvasCrosshairAutomationName =
        "GraphCanvas.Crosshair.AutomationName";
    public const string InspectorTitle = "Inspector.Title";
    public const string MagnifierTitle = "Magnifier.Title";
    public const string MagnifierOriginal = "Magnifier.Original";
    public const string MagnifierEnhanced = "Magnifier.Enhanced";
    public const string EnhancementComparison = "Enhancement.Comparison";
    public const string MagnifierImageModeGroupName = "Magnifier.ImageMode.GroupName";
    public const string MagnifierImageModeOriginal = "Magnifier.ImageMode.Original";
    public const string MagnifierImageModeEnhanced = "Magnifier.ImageMode.Enhanced";
    public const string MagnifierViewportAutomationName = "Magnifier.Viewport.AutomationName";
    public const string MagnifierCrosshairAutomationName = "Magnifier.Crosshair.AutomationName";
    public const string MagnifierEmptyImage = "Magnifier.EmptyImage";
    public const string MagnifierPixelCoordinates = "Magnifier.PixelCoordinates";
    public const string MagnifierGraphCoordinates = "Magnifier.GraphCoordinates";
    public const string MagnifierNearestDetection = "Magnifier.NearestDetection";
    public const string MagnifierNearestDetectionNone = "Magnifier.NearestDetection.None";
    public const string MagnifierConfidence = "Magnifier.Confidence";
    public const string MagnifierZoom = "Magnifier.Zoom";
    public const string MagnifierCoordinateFormat = "Magnifier.CoordinateFormat";
    public const string MagnifierConfidenceFormat = "Magnifier.ConfidenceFormat";
    public const string MagnifierZoomFormat = "Magnifier.ZoomFormat";
    public const string SeriesTitle = "Series.Title";
    public const string SeriesEmpty = "Series.Empty";
    public const string SeriesCardGroupName = "SeriesCard.GroupName";
    public const string SeriesCardSymbolAutomationName = "SeriesCard.Symbol.AutomationName";
    public const string SeriesCardInferredLabel = "SeriesCard.InferredLabel";
    public const string SeriesCardMarkerCount = "SeriesCard.MarkerCount";
    public const string SeriesCardConfidence = "SeriesCard.Confidence";
    public const string SeriesCardVisibility = "SeriesCard.Visibility";
    public const string SeriesCardSelect = "SeriesCard.Select";
    public const string SeriesCardReassign = "SeriesCard.Reassign";
    public const string SeriesCardMarkerCountFormat = "SeriesCard.MarkerCountFormat";
    public const string SeriesCardConfidenceFormat = "SeriesCard.ConfidenceFormat";
    public const string SeriesSplitSymbolName = "Series.Split.SymbolName";
    public const string SeriesSplitLabel = "Series.Split.Label";
    public const string ThemeLabel = "Theme.Label";
    public const string ThemeSystem = "Theme.System";
    public const string ThemeLight = "Theme.Light";
    public const string ThemeDark = "Theme.Dark";
    public const string ThemeSelectAutomationName = "Theme.Select.AutomationName";

    public static IReadOnlyList<string> All { get; } =
    [
        AppTitle,
        WorkflowImport,
        WorkflowEnhance,
        WorkflowAutoDetect,
        WorkflowReview,
        WorkflowExport,
        WorkflowImportToolTip,
        WorkflowEnhanceToolTip,
        WorkflowAutoDetectToolTip,
        WorkflowReviewToolTip,
        WorkflowExportToolTip,
        WorkflowEnhanceUnavailable,
        WorkflowEnhanceExperimental,
        WorkflowAutoDetectUnavailable,
        ApplicationPathsUnavailable,
        ApplicationDataNotWritable,
        PortableDataNotWritable,
        ProductionWorkflowUnavailable,
        EnhancementConfigurationInvalid,
        EnhancementModelIncompatible,
        EnhancementRuntimeUnavailable,
        EnhancementRuntimeChecksumMismatch,
        EnhancementRedistributionBlocked,
        EnhancementStorageUnavailable,
        EnhancementExecutionFailed,
        ModelManifestNotFound,
        ModelManifestInvalid,
        ModelNoticeNotFound,
        ModelNotFound,
        ModelChecksumMismatch,
        EnhancementLocalEvaluationOnly,
        PreviewDevelopment,
        PreviewVersion,
        PreviewCommit,
        PreviewRuntime,
        PreviewAvailableStages,
        PreviewMissingStages,
        PreviewNone,
        PreviewUnknown,
        ProjectOpen,
        ProjectSave,
        ProjectSaveAs,
        ProjectRecover,
        ProjectCloseTab,
        ProjectCloseDirtyBlocked,
        ManualTitle,
        ManualInstruction,
        ManualCalibration,
        ManualYMaximum,
        ManualXMaximum,
        ManualSeriesName,
        ManualSeriesShape,
        ManualSeriesFill,
        ManualSeriesRole,
        ManualCreateSeries,
        ManualEditSeries,
        ManualAddPoint,
        ManualAddFilledPoint,
        ManualAddOpenPoint,
        ManualMovePoint,
        ManualDeletePoint,
        ManualPhaseCode,
        ManualPhaseLabel,
        ManualAddDivider,
        ManualMoveDivider,
        ManualDeleteDivider,
        ManualLabelDivider,
        ManualSharedBaseline,
        ManualApplicableProbes,
        ManualNoSharedBaseline,
        ManualApplySeriesRelations,
        ManualRelationsApplied,
        ManualDefaultIntervention,
        ManualInstructionCalibration,
        ManualInstructionAddPoint,
        ManualInstructionMovePoint,
        ManualInstructionAddDivider,
        ManualInstructionMoveDivider,
        ManualInstructionSelect,
        ManualCalibrationPrompt,
        ManualCalibrationSaved,
        ManualSelectSeriesFirst,
        ManualPointAdded,
        ManualPointMoved,
        ManualPointSelected,
        ManualDividerAdded,
        ManualDividerMoved,
        ManualDividerDeleted,
        ManualDividerLabeled,
        ManualDividerSelected,
        ManualSeriesCreatedFormat,
        ManualSeriesSelectedFormat,
        ManualSeriesNameRequired,
        ManualPhaseLabelRequired,
        MarkerShapeCircle,
        MarkerShapeSquare,
        MarkerShapeTriangleUp,
        MarkerShapeTriangleDown,
        MarkerShapeDiamond,
        MarkerShapeStar,
        MarkerShapeAsterisk,
        MarkerShapeCross,
        MarkerShapeOther,
        MarkerFillFilled,
        MarkerFillOpen,
        MarkerFillUnknown,
        SemanticRoleBaseline,
        SemanticRoleIntervention,
        SemanticRoleMaintenance,
        SemanticRoleGeneralization,
        SemanticRoleUnknown,
        StatusImportedFormat,
        StatusImportFailuresFormat,
        StatusOpenedFormat,
        StatusSavedFormat,
        StatusRecoveredFormat,
        StatusExportedFormat,
        StatusCancelled,
        StatusManualEditRejectedFormat,
        StatusEnhancementReadyFormat,
        DialogGraphImages,
        DialogAllFiles,
        DialogProjectFiles,
        DialogExportFolder,
        NavigationTitle,
        NavigationGraphs,
        NavigationEmpty,
        NavigationAutomationName,
        CanvasEmpty,
        CanvasPhaseOverlay,
        CanvasZoomIn,
        CanvasZoomOut,
        CanvasFit,
        CanvasResetView,
        GraphCanvasAutomationName,
        GraphCanvasEmptyState,
        GraphCanvasPhaseOverlayAutomationName,
        GraphCanvasCrosshairAutomationName,
        InspectorTitle,
        MagnifierTitle,
        MagnifierOriginal,
        MagnifierEnhanced,
        EnhancementComparison,
        MagnifierImageModeGroupName,
        MagnifierImageModeOriginal,
        MagnifierImageModeEnhanced,
        MagnifierViewportAutomationName,
        MagnifierCrosshairAutomationName,
        MagnifierEmptyImage,
        MagnifierPixelCoordinates,
        MagnifierGraphCoordinates,
        MagnifierNearestDetection,
        MagnifierNearestDetectionNone,
        MagnifierConfidence,
        MagnifierZoom,
        MagnifierCoordinateFormat,
        MagnifierConfidenceFormat,
        MagnifierZoomFormat,
        SeriesTitle,
        SeriesEmpty,
        SeriesCardGroupName,
        SeriesCardSymbolAutomationName,
        SeriesCardInferredLabel,
        SeriesCardMarkerCount,
        SeriesCardConfidence,
        SeriesCardVisibility,
        SeriesCardSelect,
        SeriesCardReassign,
        SeriesCardMarkerCountFormat,
        SeriesCardConfidenceFormat,
        SeriesSplitSymbolName,
        SeriesSplitLabel,
        ThemeLabel,
        ThemeSystem,
        ThemeLight,
        ThemeDark,
        ThemeSelectAutomationName,
    ];
}
