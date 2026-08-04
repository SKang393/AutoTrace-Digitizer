// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Input;
using GraphReader.App.Appearance;
using GraphReader.App.Controls;
using GraphReader.App.Integration;
using GraphReader.App.Localization;
using GraphReader.App.Models;
using GraphReader.App.Services;
using GraphReader.Domain;
using GraphPoint = GraphReader.App.Models.GraphPoint;

namespace GraphReader.App.ViewModels;

public enum ManualEditorMode
{
    Select,
    Calibration,
    AddPoint,
    MovePoint,
    AddPhaseDivider,
    MovePhaseDivider,
}

public sealed class MainWindowViewModel : ObservableObject, IDisposable
{
    private static readonly string[] RequiredAutomaticDetectionStages =
        ["axis", "ocr", "markers", "legends", "phases"];
    private readonly IWorkspaceService _workspaceService;
    private readonly ILocalizationService? _localizationService;
    private readonly IManualWorkspaceService? _manualWorkspaceService;
    private readonly IWorkspaceDialogService? _dialogService;
    private readonly bool _isWorkflowAvailable;
    private readonly CancellationTokenSource _shutdown = new();
    private WorkspaceTabViewModel? _selectedTab;
    private WorkflowStage _currentStage = WorkflowStage.Review;
    private bool _isBusy;
    private string _statusMessageKey = "Workflow.Review";
    private string? _statusMessageOverride;
    private bool _isPhaseOverlayVisible = true;
    private ApplicationTheme _appearanceMode = ApplicationTheme.System;
    private string? _selectedPointId;
    private string? _selectedSeriesId;
    private int _nextManualPointId = 1;
    private int _nextSplitSeriesId = 1;
    private CancellationTokenSource? _activeOperation;
    private bool _disposed;
    private ManualEditorMode _editorMode;
    private readonly List<Point> _calibrationAnchors = [];
    private string _newSeriesName = string.Empty;
    private MarkerShape _newSeriesShape = MarkerShape.Circle;
    private MarkerFill _newSeriesFill = MarkerFill.Filled;
    private SemanticRole _newSeriesRole = SemanticRole.Intervention;
    private string _phaseCode = "b";
    private string _phaseLabel = string.Empty;
    private string? _selectedDividerId;
    private string? _selectedSharedBaselineSeriesId;
    private double _manualYMaximum = 100;
    private double _manualXMaximum = 20;

    public MainWindowViewModel(IWorkspaceService workspaceService)
        : this(workspaceService, null)
    {
    }

    public MainWindowViewModel(
        IWorkspaceService workspaceService,
        ILocalizationService? localizationService,
        string? startupErrorMessageKey = null,
        IWorkspaceDialogService? dialogService = null)
    {
        _workspaceService = workspaceService ?? throw new ArgumentNullException(nameof(workspaceService));
        _localizationService = localizationService;
        _manualWorkspaceService = workspaceService as IManualWorkspaceService;
        _dialogService = dialogService;
        _newSeriesName = GetLocalizedString(LocalizationKeys.ManualDefaultIntervention);
        _phaseLabel = GetLocalizedString(LocalizationKeys.ManualDefaultIntervention);
        MarkerShapeChoices =
        [
            Choice(MarkerShape.Circle, LocalizationKeys.MarkerShapeCircle),
            Choice(MarkerShape.Square, LocalizationKeys.MarkerShapeSquare),
            Choice(MarkerShape.TriangleUp, LocalizationKeys.MarkerShapeTriangleUp),
            Choice(MarkerShape.TriangleDown, LocalizationKeys.MarkerShapeTriangleDown),
            Choice(MarkerShape.Diamond, LocalizationKeys.MarkerShapeDiamond),
            Choice(MarkerShape.Star, LocalizationKeys.MarkerShapeStar),
            Choice(MarkerShape.Asterisk, LocalizationKeys.MarkerShapeAsterisk),
            Choice(MarkerShape.Cross, LocalizationKeys.MarkerShapeCross),
            Choice(MarkerShape.Other, LocalizationKeys.MarkerShapeOther),
        ];
        MarkerFillChoices =
        [
            Choice(MarkerFill.Filled, LocalizationKeys.MarkerFillFilled),
            Choice(MarkerFill.Open, LocalizationKeys.MarkerFillOpen),
            Choice(MarkerFill.Unknown, LocalizationKeys.MarkerFillUnknown),
        ];
        SeriesRoleChoices =
        [
            Choice(SemanticRole.Baseline, LocalizationKeys.SemanticRoleBaseline),
            Choice(SemanticRole.Intervention, LocalizationKeys.SemanticRoleIntervention),
            Choice(SemanticRole.Maintenance, LocalizationKeys.SemanticRoleMaintenance),
            Choice(SemanticRole.Generalization, LocalizationKeys.SemanticRoleGeneralization),
            Choice(SemanticRole.Unknown, LocalizationKeys.SemanticRoleUnknown),
        ];
        BaselineRelationChoices = [];
        ProbeRelationChoices = [];
        _isWorkflowAvailable = string.IsNullOrWhiteSpace(startupErrorMessageKey);
        if (!_isWorkflowAvailable)
        {
            _statusMessageKey = startupErrorMessageKey!;
        }
        Tabs = new ObservableCollection<WorkspaceTabViewModel>(_workspaceService.CreateWorkspace());
        foreach (WorkspaceTabViewModel tab in Tabs)
        {
            EnsureManualOverlay(tab);
            ConfigureSeriesSelection(tab);
        }
        _selectedTab = Tabs.FirstOrDefault();
        _selectedPointId = _selectedTab?.Points.FirstOrDefault()?.PointId;
        Magnifier = new MagnifierViewModel();
        Magnifier.IsCrosshairVisible = _selectedTab is not null;

        ImportCommand = new AsyncRelayCommand(
            ImportImagesFromDialogAsync,
            () => !IsBusy && _isWorkflowAvailable);
        EnhanceCommand = _manualWorkspaceService is null
            ? CreateWorkflowCommand(WorkflowStage.Prepare, "Workflow.Enhance")
            : new AsyncRelayCommand(
                EnhanceSelectedTabAsync,
                () => !IsBusy && _isWorkflowAvailable && SelectedTab is not null &&
                    CanRunAutomaticStage(WorkflowStage.Prepare));
        AutoDetectCommand = CreateWorkflowCommand(WorkflowStage.Detect, "Workflow.AutoDetect");
        ReviewCommand = CreateWorkflowCommand(WorkflowStage.Review, "Workflow.Review");
        ExportCommand = new AsyncRelayCommand(
            ExportFromDialogAsync,
            () => !IsBusy && _isWorkflowAvailable && (_manualWorkspaceService is null || SelectedTab is not null));
        CancelCommand = new RelayCommand(_ => CancelActiveOperation(), _ => IsBusy);
        AddPointCommand = new RelayCommand(
            parameter => AddPoint((string)parameter!),
            parameter => parameter is string seriesId && FindSeries(seriesId) is not null);
        DeletePointCommand = new RelayCommand(
            parameter => DeletePoint((string)parameter!),
            parameter => parameter is string pointId && FindPoint(pointId) is not null);
        MovePointCommand = new RelayCommand(
            parameter =>
            {
                MovePointRequest request = (MovePointRequest)parameter!;
                MovePoint(request.PointId, request.PixelX, request.PixelY);
            },
            parameter => parameter is MovePointRequest request && FindPoint(request.PointId) is not null);
        MergeSeriesCommand = new RelayCommand(
            parameter =>
            {
                SeriesPairRequest request = (SeriesPairRequest)parameter!;
                MergeSeries(request.SourceSeriesId, request.TargetSeriesId);
            },
            parameter => parameter is SeriesPairRequest request &&
                request.SourceSeriesId != request.TargetSeriesId &&
                FindSeries(request.SourceSeriesId) is not null &&
                FindSeries(request.TargetSeriesId) is not null);
        SplitSeriesCommand = new RelayCommand(
            parameter =>
            {
                SplitSeriesRequest request = (SplitSeriesRequest)parameter!;
                SplitSeries(request.SourceSeriesId, request.PointIds);
            },
            parameter => parameter is SplitSeriesRequest request &&
                request.PointIds.Count > 0 &&
                FindSeries(request.SourceSeriesId) is not null);
        ReassignPointCommand = new RelayCommand(
            parameter => ExecuteReassignCommand(parameter),
            CanExecuteReassignCommand);
        TogglePhaseOverlayCommand = new RelayCommand(
            _ => IsPhaseOverlayVisible = !IsPhaseOverlayVisible);
        ZoomInCommand = new RelayCommand(_ => ChangeZoom(1.25), _ => SelectedTab is not null);
        ZoomOutCommand = new RelayCommand(_ => ChangeZoom(0.8), _ => SelectedTab is not null);
        FitZoomCommand = new RelayCommand(_ => SetZoom(1), _ => SelectedTab is not null);
        ResetViewCommand = new RelayCommand(_ => SetZoom(1), _ => SelectedTab is not null);
        ShowOriginalPreviewCommand = new RelayCommand(
            _ => SetEnhancementPreviewMode(EnhancementPreviewMode.Original),
            _ => SelectedTab is not null);
        ShowEnhancedPreviewCommand = new RelayCommand(
            _ => SetEnhancementPreviewMode(EnhancementPreviewMode.Enhanced),
            _ => SelectedTab?.HasEnhancedPreview == true);
        ShowComparisonPreviewCommand = new RelayCommand(
            _ => SetEnhancementPreviewMode(EnhancementPreviewMode.Comparison),
            _ => SelectedTab?.HasEnhancedPreview == true);
        NextTabCommand = new RelayCommand(_ => SelectRelativeTab(1), _ => Tabs.Count > 1);
        PreviousTabCommand = new RelayCommand(_ => SelectRelativeTab(-1), _ => Tabs.Count > 1);
        OpenProjectCommand = new AsyncRelayCommand(OpenProjectFromDialogAsync, () => !IsBusy && _manualWorkspaceService is not null);
        SaveProjectCommand = new AsyncRelayCommand(
            cancellationToken => SaveProjectFromDialogAsync(saveAs: false, cancellationToken),
            () => !IsBusy && _manualWorkspaceService is not null && Tabs.Count > 0);
        SaveProjectAsCommand = new AsyncRelayCommand(
            cancellationToken => SaveProjectFromDialogAsync(saveAs: true, cancellationToken),
            () => !IsBusy && _manualWorkspaceService is not null && Tabs.Count > 0);
        RecoverProjectCommand = new AsyncRelayCommand(
            RecoverProjectFromAutosaveAsync,
            () => !IsBusy && _manualWorkspaceService is not null);
        CloseTabCommand = new RelayCommand(
            parameter => CloseTab(parameter as WorkspaceTabViewModel ?? SelectedTab),
            parameter => parameter is WorkspaceTabViewModel || SelectedTab is not null);
        StartCalibrationCommand = new RelayCommand(_ => BeginCalibration(), _ => SelectedTab is not null && _manualWorkspaceService is not null);
        CreateSeriesCommand = new RelayCommand(_ => CreateSeries(), _ => SelectedTab is not null && _manualWorkspaceService is not null);
        EditSeriesCommand = new RelayCommand(_ => EditSelectedSeries(), _ => CanEditSelectedSeries());
        ApplySeriesRelationsCommand = new RelayCommand(_ => ApplySeriesRelations(), _ => CanApplySeriesRelations());
        BeginAddPointCommand = new RelayCommand(_ => EditorMode = ManualEditorMode.AddPoint, _ => CanEditPoint());
        BeginAddFilledPointCommand = new RelayCommand(
            _ => BeginAddPointForFill(MarkerFill.Filled),
            _ => HasSeriesWithFill(MarkerFill.Filled));
        BeginAddOpenPointCommand = new RelayCommand(
            _ => BeginAddPointForFill(MarkerFill.Open),
            _ => HasSeriesWithFill(MarkerFill.Open));
        BeginMovePointCommand = new RelayCommand(_ => EditorMode = ManualEditorMode.MovePoint, _ => CanMoveSelectedPoint());
        BeginAddPhaseDividerCommand = new RelayCommand(_ => EditorMode = ManualEditorMode.AddPhaseDivider, _ => SelectedTab is not null && _manualWorkspaceService is not null);
        BeginMovePhaseDividerCommand = new RelayCommand(_ => EditorMode = ManualEditorMode.MovePhaseDivider, _ => SelectedDividerId is not null && _manualWorkspaceService is not null);
        DeletePhaseDividerCommand = new RelayCommand(_ => DeleteSelectedPhaseDivider(), _ => SelectedDividerId is not null && _manualWorkspaceService is not null);
        LabelPhaseDividerCommand = new RelayCommand(_ => LabelSelectedPhaseDivider(), _ => SelectedDividerId is not null && _manualWorkspaceService is not null);

        BuildIdentity identity = BuildIdentity.Current();
        VersionText = identity.Version;
        ShortCommit = identity.ShortCommit;
        RuntimeModeText = workspaceService is IRuntimeWorkspaceService runtime
            ? runtime.RuntimeEnvironment.ToString()
            : GetLocalizedString(LocalizationKeys.PreviewUnknown);
        AvailableAutomaticStagesText = workspaceService is IRuntimeWorkspaceService available
            ? JoinStages(available.AutomaticStages, static state => state is AutomaticStageState.Available or AutomaticStageState.Approved or AutomaticStageState.Experimental)
            : GetLocalizedString(LocalizationKeys.PreviewNone);
        MissingAutomaticStagesText = workspaceService is IRuntimeWorkspaceService missing
            ? JoinStages(missing.AutomaticStages, static state => state == AutomaticStageState.Unavailable)
            : GetLocalizedString(LocalizationKeys.PreviewUnknown);
        AutomaticStageStatus? enhancementStatus = (workspaceService as IRuntimeWorkspaceService)?.AutomaticStages
            .FirstOrDefault(status => string.Equals(status.Stage, "enhancement", StringComparison.Ordinal));
        EnhancementAvailabilityText = enhancementStatus?.State == AutomaticStageState.Experimental
            ? GetLocalizedString(LocalizationKeys.WorkflowEnhanceExperimental)
            : CanRunAutomaticStage(WorkflowStage.Prepare)
                ? string.Empty
                : GetLocalizedString(LocalizationKeys.WorkflowEnhanceUnavailable);
        AutoDetectionAvailabilityText = CanRunAutomaticStage(WorkflowStage.Detect)
            ? string.Empty
            : GetLocalizedString(LocalizationKeys.WorkflowAutoDetectUnavailable);
    }

    public ObservableCollection<WorkspaceTabViewModel> Tabs { get; }

    public WorkspaceTabViewModel? SelectedTab
    {
        get => _selectedTab;
        set
        {
            if (SetProperty(ref _selectedTab, value))
            {
                OnPropertyChanged(nameof(SeriesCards));
                OnPropertyChanged(nameof(HasEnhancedPreview));
                OnPropertyChanged(nameof(CurrentEnhancementPreviewMode));
                SelectedPointId = value?.Points.FirstOrDefault()?.PointId;
                SelectedSeriesId = value?.SeriesCards.FirstOrDefault()?.SeriesId;
                SelectedDividerId = value?.PhaseDividers.FirstOrDefault()?.DividerId;
                Magnifier.IsCrosshairVisible = value is not null;
                Magnifier.PixelPosition = default;
                Magnifier.CrosshairPosition = new Point(0.5, 0.5);
                Magnifier.GraphPosition = null;
                Magnifier.NearestDetectionName = null;
                Magnifier.NearestDetectionConfidence = null;
                RelayCommand.RaiseCanExecuteChanged();
                AsyncRelayCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public MagnifierViewModel Magnifier { get; }

    public ObservableCollection<SeriesCardViewModel> SeriesCards =>
        SelectedTab?.SeriesCards ?? EmptySeriesCards;

    public WorkflowStage CurrentStage
    {
        get => _currentStage;
        private set => SetProperty(ref _currentStage, value);
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (SetProperty(ref _isBusy, value))
            {
                AsyncRelayCommand.RaiseCanExecuteChanged();
                RelayCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public string StatusMessageKey
    {
        get => _statusMessageKey;
        private set
        {
            if (SetProperty(ref _statusMessageKey, value))
            {
                _statusMessageOverride = null;
                OnPropertyChanged(nameof(StatusMessage));
            }
        }
    }

    public string StatusMessage => _statusMessageOverride ?? GetLocalizedString(StatusMessageKey);

    public string VersionText { get; }

    public string ShortCommit { get; }

    public string RuntimeModeText { get; }

    public string AvailableAutomaticStagesText { get; }

    public string MissingAutomaticStagesText { get; }

    public string EnhancementAvailabilityText { get; }

    public string AutoDetectionAvailabilityText { get; }

    public bool HasEnhancedPreview => SelectedTab?.HasEnhancedPreview == true;

    public EnhancementPreviewMode CurrentEnhancementPreviewMode =>
        SelectedTab?.EnhancementPreviewMode ?? EnhancementPreviewMode.Original;

    public IReadOnlyList<LocalizedChoice<MarkerShape>> MarkerShapeChoices { get; }

    public IReadOnlyList<LocalizedChoice<MarkerFill>> MarkerFillChoices { get; }

    public IReadOnlyList<LocalizedChoice<SemanticRole>> SeriesRoleChoices { get; }

    public bool IsPhaseOverlayVisible
    {
        get => _isPhaseOverlayVisible;
        set => SetProperty(ref _isPhaseOverlayVisible, value);
    }

    public ApplicationTheme AppearanceMode
    {
        get => _appearanceMode;
        set => SetProperty(ref _appearanceMode, value);
    }

    public string? SelectedPointId
    {
        get => _selectedPointId;
        set
        {
            if (SetProperty(ref _selectedPointId, value))
            {
                RelayCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public string? SelectedSeriesId
    {
        get => _selectedSeriesId;
        set
        {
            if (SetProperty(ref _selectedSeriesId, value))
            {
                LoadSelectedSeriesEditor();
                RefreshSeriesRelationChoices();
                RelayCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public ObservableCollection<SeriesRelationChoiceViewModel> BaselineRelationChoices { get; }

    public ObservableCollection<SeriesRelationChoiceViewModel> ProbeRelationChoices { get; }

    public string? SelectedSharedBaselineSeriesId
    {
        get => _selectedSharedBaselineSeriesId;
        set => SetProperty(ref _selectedSharedBaselineSeriesId, value);
    }

    public string? SelectedDividerId
    {
        get => _selectedDividerId;
        set
        {
            if (SetProperty(ref _selectedDividerId, value))
            {
                RelayCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public ManualEditorMode EditorMode
    {
        get => _editorMode;
        private set
        {
            if (SetProperty(ref _editorMode, value))
            {
                OnPropertyChanged(nameof(EditorInstruction));
            }
        }
    }

    public string EditorInstruction => EditorMode switch
    {
        ManualEditorMode.Calibration => FormatLocalizedString(
            LocalizationKeys.ManualInstructionCalibration,
            _calibrationAnchors.Count + 1),
        ManualEditorMode.AddPoint => GetLocalizedString(LocalizationKeys.ManualInstructionAddPoint),
        ManualEditorMode.MovePoint => GetLocalizedString(LocalizationKeys.ManualInstructionMovePoint),
        ManualEditorMode.AddPhaseDivider => GetLocalizedString(LocalizationKeys.ManualInstructionAddDivider),
        ManualEditorMode.MovePhaseDivider => GetLocalizedString(LocalizationKeys.ManualInstructionMoveDivider),
        _ => GetLocalizedString(LocalizationKeys.ManualInstructionSelect),
    };

    public double ManualYMaximum
    {
        get => _manualYMaximum;
        set => SetProperty(ref _manualYMaximum, value);
    }

    public double ManualXMaximum
    {
        get => _manualXMaximum;
        set => SetProperty(ref _manualXMaximum, value);
    }

    public string NewSeriesName
    {
        get => _newSeriesName;
        set => SetProperty(ref _newSeriesName, value);
    }

    public MarkerShape NewSeriesShape
    {
        get => _newSeriesShape;
        set => SetProperty(ref _newSeriesShape, value);
    }

    public MarkerFill NewSeriesFill
    {
        get => _newSeriesFill;
        set => SetProperty(ref _newSeriesFill, value);
    }

    public SemanticRole NewSeriesRole
    {
        get => _newSeriesRole;
        set => SetProperty(ref _newSeriesRole, value);
    }

    public string PhaseCode
    {
        get => _phaseCode;
        set => SetProperty(ref _phaseCode, value);
    }

    public string PhaseLabel
    {
        get => _phaseLabel;
        set => SetProperty(ref _phaseLabel, value);
    }

    public ICommand ImportCommand { get; }

    public ICommand EnhanceCommand { get; }

    public ICommand AutoDetectCommand { get; }

    public ICommand ReviewCommand { get; }

    public ICommand ExportCommand { get; }

    public ICommand CancelCommand { get; }

    public ICommand AddPointCommand { get; }

    public ICommand DeletePointCommand { get; }

    public ICommand MovePointCommand { get; }

    public ICommand MergeSeriesCommand { get; }

    public ICommand SplitSeriesCommand { get; }

    public ICommand ReassignPointCommand { get; }

    public ICommand TogglePhaseOverlayCommand { get; }

    public ICommand ZoomInCommand { get; }

    public ICommand ZoomOutCommand { get; }

    public ICommand FitZoomCommand { get; }

    public ICommand ResetViewCommand { get; }

    public ICommand ShowOriginalPreviewCommand { get; }

    public ICommand ShowEnhancedPreviewCommand { get; }

    public ICommand ShowComparisonPreviewCommand { get; }

    public ICommand NextTabCommand { get; }

    public ICommand PreviousTabCommand { get; }

    public ICommand OpenProjectCommand { get; }

    public ICommand SaveProjectCommand { get; }

    public ICommand SaveProjectAsCommand { get; }

    public ICommand RecoverProjectCommand { get; }

    public ICommand CloseTabCommand { get; }

    public ICommand StartCalibrationCommand { get; }

    public ICommand CreateSeriesCommand { get; }

    public ICommand EditSeriesCommand { get; }

    public ICommand ApplySeriesRelationsCommand { get; }

    public ICommand BeginAddPointCommand { get; }

    public ICommand BeginAddFilledPointCommand { get; }

    public ICommand BeginAddOpenPointCommand { get; }

    public ICommand BeginMovePointCommand { get; }

    public ICommand BeginAddPhaseDividerCommand { get; }

    public ICommand BeginMovePhaseDividerCommand { get; }

    public ICommand DeletePhaseDividerCommand { get; }

    public ICommand LabelPhaseDividerCommand { get; }

    private static ObservableCollection<SeriesCardViewModel> EmptySeriesCards { get; } = [];

    public void AddPoint(string seriesId)
    {
        if (_manualWorkspaceService is not null && SelectedTab is { } manualTab)
        {
            GraphPoint added = _manualWorkspaceService.AddPoint(
                manualTab.TabId,
                seriesId,
                Magnifier.PixelPosition.X,
                Magnifier.PixelPosition.Y);
            SelectedPointId = added.PointId;
            SelectedSeriesId = seriesId;
            QueueAutosave(SnapshotTrigger.PointEdited, manualTab.TabId, added.PointId);
            return;
        }

        SeriesCardViewModel series = RequireSeries(seriesId);
        WorkspaceTabViewModel tab = RequireSelectedTab();
        GraphPoint point = new(
            $"manual-point-{_nextManualPointId++:D4}",
            seriesId,
            Magnifier.PixelPosition.X,
            Magnifier.PixelPosition.Y,
            Magnifier.GraphPosition?.X ?? 0,
            Magnifier.GraphPosition?.Y ?? 0,
            "unknown");
        tab.Points.Add(point);
        SelectedPointId = point.PointId;
        series.NotifyCountChanged();
    }

    public void DeletePoint(string pointId)
    {
        if (_manualWorkspaceService is not null && SelectedTab is { } manualTab)
        {
            GraphPoint manualPoint = RequirePoint(pointId);
            _manualWorkspaceService.DeletePoint(manualTab.TabId, pointId);
            if (SelectedPointId == pointId)
            {
                SelectedPointId = manualTab.Points.FirstOrDefault()?.PointId;
            }

            FindSeries(manualPoint.SeriesId)?.NotifyCountChanged();
            QueueAutosave(SnapshotTrigger.PointEdited, manualTab.TabId, pointId);
            return;
        }

        WorkspaceTabViewModel tab = RequireSelectedTab();
        GraphPoint point = RequirePoint(pointId);
        string seriesId = point.SeriesId;
        tab.Points.Remove(point);
        if (SelectedPointId == pointId)
        {
            SelectedPointId = tab.Points.FirstOrDefault()?.PointId;
        }

        FindSeries(seriesId)?.NotifyCountChanged();
    }

    public void MovePoint(string pointId, double pixelX, double pixelY)
    {
        if (!double.IsFinite(pixelX) || !double.IsFinite(pixelY))
        {
            throw new ArgumentOutOfRangeException(nameof(pixelX));
        }

        if (_manualWorkspaceService is not null && SelectedTab is { } manualTab)
        {
            _manualWorkspaceService.MovePoint(manualTab.TabId, pointId, pixelX, pixelY);
            SelectedPointId = pointId;
            QueueAutosave(SnapshotTrigger.PointEdited, manualTab.TabId, pointId);
            return;
        }

        GraphPoint point = RequirePoint(pointId);
        point.PixelX = pixelX;
        point.PixelY = pixelY;
        SelectedPointId = point.PointId;
        FindSeries(point.SeriesId)?.NotifyCountChanged();
    }

    public void MergeSeries(string sourceSeriesId, string targetSeriesId)
    {
        if (sourceSeriesId == targetSeriesId)
        {
            throw new ArgumentException("Source and target series must differ.", nameof(targetSeriesId));
        }

        WorkspaceTabViewModel tab = RequireSelectedTab();
        SeriesCardViewModel source = RequireSeries(sourceSeriesId);
        SeriesCardViewModel target = RequireSeries(targetSeriesId);
        foreach (GraphPoint point in tab.Points.Where(point => point.SeriesId == sourceSeriesId))
        {
            point.SeriesId = targetSeriesId;
        }

        source.NotifyCountChanged();
        target.NotifyCountChanged();
        tab.SeriesCards.Remove(source);
    }

    public void SplitSeries(string sourceSeriesId, IReadOnlyCollection<string> pointIds)
    {
        ArgumentNullException.ThrowIfNull(pointIds);
        if (pointIds.Count == 0)
        {
            throw new ArgumentException("At least one point is required.", nameof(pointIds));
        }

        WorkspaceTabViewModel tab = RequireSelectedTab();
        SeriesCardViewModel source = RequireSeries(sourceSeriesId);
        HashSet<string> selectedIds = pointIds.ToHashSet(StringComparer.Ordinal);
        GraphPoint[] selectedPoints = tab.Points
            .Where(point => selectedIds.Contains(point.PointId) && point.SeriesId == sourceSeriesId)
            .ToArray();
        if (selectedPoints.Length != selectedIds.Count)
        {
            throw new ArgumentException("Every split point must belong to the source series.", nameof(pointIds));
        }

        string createdSeriesId = $"series-split-{_nextSplitSeriesId++:D2}";
        SeriesCardViewModel created = new(
            createdSeriesId,
            "◆",
            GetLocalizedString(LocalizationKeys.SeriesSplitSymbolName),
            GetLocalizedString(LocalizationKeys.SeriesSplitLabel),
            source.Confidence,
            tab.Points);
        tab.SeriesCards.Add(created);
        foreach (GraphPoint point in selectedPoints)
        {
            point.SeriesId = createdSeriesId;
        }

        source.NotifyCountChanged();
        created.NotifyCountChanged();
    }

    private string GetLocalizedString(string key) =>
        _localizationService?.GetString(key) ?? key;

    private LocalizedChoice<T> Choice<T>(T value, string key)
        where T : struct, Enum =>
        new(value, GetLocalizedString(key));

    private string FormatLocalizedString(string key, params object[] arguments) =>
        string.Format(CultureInfo.CurrentCulture, GetLocalizedString(key), arguments);

    public void ReassignPoint(string pointId, string targetSeriesId)
    {
        if (_manualWorkspaceService is not null && SelectedTab is { } manualTab)
        {
            _manualWorkspaceService.ReassignPoint(manualTab.TabId, pointId, targetSeriesId);
            SelectedPointId = pointId;
            SelectedSeriesId = targetSeriesId;
            QueueAutosave(SnapshotTrigger.PointEdited, manualTab.TabId, pointId);
            return;
        }

        GraphPoint point = RequirePoint(pointId);
        SeriesCardViewModel target = RequireSeries(targetSeriesId);
        SelectedPointId = pointId;
        string sourceSeriesId = point.SeriesId;
        if (sourceSeriesId == targetSeriesId)
        {
            target.NotifyCountChanged();
            return;
        }

        point.SeriesId = targetSeriesId;
        FindSeries(sourceSeriesId)?.NotifyCountChanged();
        target.NotifyCountChanged();
    }

    public async Task HandleCanvasPointAsync(Point point, CancellationToken cancellationToken = default)
    {
        if (SelectedTab is not { } tab || _manualWorkspaceService is null)
        {
            return;
        }

        Magnifier.PixelPosition = point;
        Magnifier.CrosshairPosition = new Point(
            tab.PixelWidth > 0 ? Math.Clamp(point.X / tab.PixelWidth, 0, 1) : 0,
            tab.PixelHeight > 0 ? Math.Clamp(point.Y / tab.PixelHeight, 0, 1) : 0);
        if (tab.Calibration is { } calibration)
        {
            Magnifier.GraphPosition = new Point(
                calibration.XTransform.PixelToGraph(point.X),
                calibration.YTransform.PixelToGraph(point.Y));
        }
        else
        {
            Magnifier.GraphPosition = null;
        }

        try
        {
            switch (EditorMode)
            {
                case ManualEditorMode.Calibration:
                _calibrationAnchors.Add(point);
                OnPropertyChanged(nameof(EditorInstruction));
                if (_calibrationAnchors.Count == 3)
                {
                    _manualWorkspaceService.Calibrate(
                        tab.TabId,
                        new ManualCalibrationRequest(
                            new GraphReader.Axis.PixelPoint(_calibrationAnchors[0].X, _calibrationAnchors[0].Y),
                            new GraphReader.Axis.PixelPoint(_calibrationAnchors[1].X, _calibrationAnchors[1].Y),
                            new GraphReader.Axis.PixelPoint(_calibrationAnchors[2].X, _calibrationAnchors[2].Y),
                            ManualYMaximum,
                            ManualXMaximum));
                    EditorMode = ManualEditorMode.Select;
                    SetStatus(GetLocalizedString(LocalizationKeys.ManualCalibrationSaved));
                    await TryAutosaveAsync(SnapshotTrigger.CalibrationChanged, tab.TabId, null, cancellationToken);
                }
                    break;
                case ManualEditorMode.AddPoint:
                if (SelectedSeriesId is null)
                {
                    SetStatus(GetLocalizedString(LocalizationKeys.ManualSelectSeriesFirst));
                    break;
                }

                GraphPoint added = _manualWorkspaceService.AddPoint(
                    tab.TabId,
                    SelectedSeriesId,
                    point.X,
                    point.Y);
                SelectedPointId = added.PointId;
                EditorMode = ManualEditorMode.Select;
                SetStatus(GetLocalizedString(LocalizationKeys.ManualPointAdded));
                await TryAutosaveAsync(SnapshotTrigger.PointEdited, tab.TabId, added.PointId, cancellationToken);
                    break;
                case ManualEditorMode.MovePoint:
                if (SelectedPointId is not null)
                {
                    _manualWorkspaceService.MovePoint(tab.TabId, SelectedPointId, point.X, point.Y);
                    EditorMode = ManualEditorMode.Select;
                    SetStatus(GetLocalizedString(LocalizationKeys.ManualPointMoved));
                    await TryAutosaveAsync(SnapshotTrigger.PointEdited, tab.TabId, SelectedPointId, cancellationToken);
                }
                    break;
                case ManualEditorMode.AddPhaseDivider:
                EditablePhaseDivider divider = _manualWorkspaceService.AddPhaseDivider(
                    tab.TabId,
                    point.X,
                    PhaseCode,
                    PhaseLabel);
                SelectedDividerId = divider.DividerId;
                EditorMode = ManualEditorMode.Select;
                SetStatus(GetLocalizedString(LocalizationKeys.ManualDividerAdded));
                await TryAutosaveAsync(SnapshotTrigger.PhaseEdited, tab.TabId, divider.DividerId, cancellationToken);
                    break;
                case ManualEditorMode.MovePhaseDivider:
                if (SelectedDividerId is not null)
                {
                    _manualWorkspaceService.MovePhaseDivider(tab.TabId, SelectedDividerId, point.X);
                    EditorMode = ManualEditorMode.Select;
                    SetStatus(GetLocalizedString(LocalizationKeys.ManualDividerMoved));
                    await TryAutosaveAsync(SnapshotTrigger.PhaseEdited, tab.TabId, SelectedDividerId, cancellationToken);
                }
                    break;
                default:
                    SelectNearest(tab, point);
                    break;
            }
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException)
        {
            _calibrationAnchors.Clear();
            OnPropertyChanged(nameof(EditorInstruction));
            EditorMode = ManualEditorMode.Select;
            SetStatus(FormatLocalizedString(LocalizationKeys.StatusManualEditRejectedFormat, exception.Message));
        }
    }

    private async Task ImportImagesFromDialogAsync(CancellationToken cancellationToken)
    {
        if (_manualWorkspaceService is null || _dialogService is null)
        {
            await RunWorkflowStageAsync(WorkflowStage.Import, "Workflow.Import", cancellationToken);
            return;
        }

        IReadOnlyList<string> paths = _dialogService.SelectImages();
        if (paths.Count == 0)
        {
            return;
        }

        await RunBusyAsync(async token =>
        {
            IReadOnlyList<WorkspaceTabViewModel> imported =
                await _manualWorkspaceService.ImportImagesAsync(paths, token);
            foreach (WorkspaceTabViewModel tab in imported)
            {
                EnsureManualOverlay(tab);
                ConfigureSeriesSelection(tab);
                Tabs.Add(tab);
            }

            SelectedTab = imported.Count > 0 ? imported[^1] : SelectedTab;
            CurrentStage = WorkflowStage.Import;
            SetStatus(_manualWorkspaceService.LastImportErrors.Count == 0
                ? FormatLocalizedString(LocalizationKeys.StatusImportedFormat, imported.Count)
                : FormatLocalizedString(
                    LocalizationKeys.StatusImportFailuresFormat,
                    imported.Count,
                    _manualWorkspaceService.LastImportErrors.Count));
        }, cancellationToken);
    }

    private async Task EnhanceSelectedTabAsync(CancellationToken cancellationToken)
    {
        if (_manualWorkspaceService is null || SelectedTab is not { } tab)
        {
            return;
        }

        await RunBusyAsync(async token =>
        {
            WorkspaceEnhancementResult result = await _manualWorkspaceService
                .EnhanceAsync(tab.TabId, token);
            SetStatus(result.UserMessageKey is null
                ? result.Message
                : FormatLocalizedString(
                    result.UserMessageKey,
                    result.MessageArguments?.ToArray() ?? []));
            if (!result.Succeeded)
            {
                return;
            }

            CurrentStage = WorkflowStage.Prepare;
            Magnifier.IsEnhanced = true;
            OnPropertyChanged(nameof(HasEnhancedPreview));
            OnPropertyChanged(nameof(CurrentEnhancementPreviewMode));
            RelayCommand.RaiseCanExecuteChanged();
            QueueAutosave(SnapshotTrigger.PointEdited, tab.TabId, "enhancement");
        }, cancellationToken);
    }

    private void SetEnhancementPreviewMode(EnhancementPreviewMode mode)
    {
        if (_manualWorkspaceService is null || SelectedTab is not { } tab)
        {
            return;
        }

        _manualWorkspaceService.SetEnhancementPreviewMode(tab.TabId, mode);
        Magnifier.IsEnhanced = mode != EnhancementPreviewMode.Original;
        OnPropertyChanged(nameof(CurrentEnhancementPreviewMode));
        QueueAutosave(SnapshotTrigger.ExportSettingsChanged, tab.TabId, "enhancement-preview");
    }

    private async Task OpenProjectFromDialogAsync(CancellationToken cancellationToken)
    {
        if (_manualWorkspaceService is null || _dialogService?.SelectProjectToOpen() is not string path)
        {
            return;
        }

        await RunBusyAsync(async token =>
        {
            IReadOnlyList<WorkspaceTabViewModel> opened =
                await _manualWorkspaceService.OpenProjectAsync(path, token);
            Tabs.Clear();
            foreach (WorkspaceTabViewModel tab in opened)
            {
                EnsureManualOverlay(tab);
                ConfigureSeriesSelection(tab);
                Tabs.Add(tab);
            }

            SelectedTab = Tabs.FirstOrDefault();
            SetStatus(FormatLocalizedString(LocalizationKeys.StatusOpenedFormat, Path.GetFileName(path)));
        }, cancellationToken);
    }

    private async Task SaveProjectFromDialogAsync(bool saveAs, CancellationToken cancellationToken)
    {
        if (_manualWorkspaceService is null || _dialogService is null)
        {
            return;
        }

        string? path = saveAs || _manualWorkspaceService.CurrentProjectPath is null
            ? _dialogService.SelectProjectToSave(_manualWorkspaceService.CurrentProjectPath)
            : _manualWorkspaceService.CurrentProjectPath;
        if (path is null)
        {
            return;
        }

        await RunBusyAsync(async token =>
        {
            DomainResult<ProjectSaveReceipt> saved =
                await _manualWorkspaceService.SaveProjectAsync(path, token);
            if (!saved.IsSuccess)
            {
                throw new InvalidOperationException(string.Join(" | ", saved.Errors.Select(error => error.TechnicalMessage)));
            }

            foreach (WorkspaceTabViewModel tab in Tabs)
            {
                tab.IsDirty = false;
            }
            SetStatus(FormatLocalizedString(LocalizationKeys.StatusSavedFormat, Path.GetFileName(saved.Value!.Path)));
        }, cancellationToken);
    }

    private async Task RecoverProjectFromAutosaveAsync(CancellationToken cancellationToken)
    {
        if (_manualWorkspaceService is null || _dialogService is null)
        {
            return;
        }

        string suggestedPath = _manualWorkspaceService.CurrentProjectPath is { } currentPath
            ? Path.Combine(
                Path.GetDirectoryName(currentPath) ?? string.Empty,
                $"{Path.GetFileNameWithoutExtension(currentPath)}-recovered.garproj")
            : "graph-project-recovered.garproj";
        string? destinationPath = _dialogService.SelectProjectToSave(suggestedPath);
        if (destinationPath is null)
        {
            return;
        }

        await RunBusyAsync(async token =>
        {
            DomainResult<ProjectSaveReceipt> recovered = await _manualWorkspaceService
                .RecoverLatestToNewFileAsync(destinationPath, token);
            if (!recovered.IsSuccess || recovered.Value is null)
            {
                throw new InvalidOperationException(
                    string.Join(" | ", recovered.Errors.Select(error => error.TechnicalMessage)));
            }

            IReadOnlyList<WorkspaceTabViewModel> opened = await _manualWorkspaceService
                .OpenProjectAsync(recovered.Value.Path, token);
            Tabs.Clear();
            foreach (WorkspaceTabViewModel tab in opened)
            {
                EnsureManualOverlay(tab);
                ConfigureSeriesSelection(tab);
                Tabs.Add(tab);
            }

            SelectedTab = Tabs.FirstOrDefault();
            SetStatus(FormatLocalizedString(
                LocalizationKeys.StatusRecoveredFormat,
                Path.GetFileName(recovered.Value.Path)));
        }, cancellationToken);
    }

    private async Task ExportFromDialogAsync(CancellationToken cancellationToken)
    {
        if (_manualWorkspaceService is null || _dialogService is null)
        {
            await RunWorkflowStageAsync(WorkflowStage.Export, "Workflow.Export", cancellationToken);
            return;
        }

        if (SelectedTab is not { } tab || _dialogService.SelectExportDirectory() is not string directory)
        {
            return;
        }

        await RunBusyAsync(async token =>
        {
            GraphReader.Export.ExportResult exported =
                await _manualWorkspaceService.ExportAsync(tab.TabId, directory, token);
            if (!exported.Succeeded)
            {
                throw new InvalidOperationException(string.Join(" | ", exported.Failures.Select(failure => failure.TechnicalMessage)));
            }

            CurrentStage = WorkflowStage.Export;
            SetStatus(FormatLocalizedString(
                LocalizationKeys.StatusExportedFormat,
                exported.MinimalArtifacts.Count));
        }, cancellationToken);
    }

    private async Task RunBusyAsync(Func<CancellationToken, Task> action, CancellationToken cancellationToken)
    {
        if (IsBusy)
        {
            return;
        }

        _activeOperation?.Dispose();
        _activeOperation = CancellationTokenSource.CreateLinkedTokenSource(_shutdown.Token, cancellationToken);
        IsBusy = true;
        try
        {
            await action(_activeOperation.Token);
        }
        catch (OperationCanceledException) when (_activeOperation.IsCancellationRequested)
        {
            SetStatus(GetLocalizedString(LocalizationKeys.StatusCancelled));
        }
        catch (InvalidOperationException exception)
        {
            SetStatus(exception.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void BeginCalibration()
    {
        _calibrationAnchors.Clear();
        EditorMode = ManualEditorMode.Calibration;
        SetStatus(GetLocalizedString(LocalizationKeys.ManualCalibrationPrompt));
    }

    private void CreateSeries()
    {
        if (SelectedTab is not { } tab || _manualWorkspaceService is null)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(NewSeriesName))
        {
            SetStatus(GetLocalizedString(LocalizationKeys.ManualSeriesNameRequired));
            return;
        }

        try
        {
            string symbol = SymbolFor(NewSeriesShape, NewSeriesFill);
            SeriesCardViewModel series = _manualWorkspaceService.AddSeries(
                tab.TabId,
                new ManualSeriesDefinition(NewSeriesName, symbol, NewSeriesShape, NewSeriesFill, NewSeriesRole));
            ConfigureSeriesSelection(series);
            SelectedSeriesId = series.SeriesId;
            SetStatus(FormatLocalizedString(LocalizationKeys.ManualSeriesCreatedFormat, series.Label));
            QueueAutosave(SnapshotTrigger.PointEdited, tab.TabId, series.SeriesId);
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException)
        {
            SetStatus(FormatLocalizedString(LocalizationKeys.StatusManualEditRejectedFormat, exception.Message));
        }
    }

    private void EditSelectedSeries()
    {
        if (SelectedTab is not { } tab ||
            SelectedSeriesId is not string seriesId ||
            _manualWorkspaceService is null)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(NewSeriesName))
        {
            SetStatus(GetLocalizedString(LocalizationKeys.ManualSeriesNameRequired));
            return;
        }

        try
        {
            _manualWorkspaceService.UpdateSeries(
                tab.TabId,
                seriesId,
                new ManualSeriesDefinition(
                    NewSeriesName,
                    SymbolFor(NewSeriesShape, NewSeriesFill),
                    NewSeriesShape,
                    NewSeriesFill,
                    NewSeriesRole));
            SetStatus(FormatLocalizedString(LocalizationKeys.ManualSeriesSelectedFormat, NewSeriesName));
            QueueAutosave(SnapshotTrigger.PointEdited, tab.TabId, seriesId);
            RefreshSeriesRelationChoices();
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException)
        {
            SetStatus(FormatLocalizedString(LocalizationKeys.StatusManualEditRejectedFormat, exception.Message));
        }
    }

    private bool CanEditSelectedSeries() =>
        _manualWorkspaceService is not null &&
        SelectedTab is not null &&
        SelectedSeriesId is not null;

    private void LoadSelectedSeriesEditor()
    {
        if (FindSeries(SelectedSeriesId ?? string.Empty) is not { } series)
        {
            return;
        }

        NewSeriesName = series.Label;
        NewSeriesShape = series.Shape;
        NewSeriesFill = series.Fill;
        NewSeriesRole = series.SemanticRole;
    }

    private bool HasSeriesWithFill(MarkerFill fill) =>
        _manualWorkspaceService is not null &&
        SelectedTab?.SeriesCards.Any(series => series.Fill == fill) == true;

    private void BeginAddPointForFill(MarkerFill fill)
    {
        SeriesCardViewModel? series = SelectedTab?.SeriesCards.FirstOrDefault(candidate =>
            candidate.Fill == fill && string.Equals(candidate.SeriesId, SelectedSeriesId, StringComparison.Ordinal))
            ?? SelectedTab?.SeriesCards.FirstOrDefault(candidate => candidate.Fill == fill);
        if (series is null)
        {
            SetStatus(GetLocalizedString(LocalizationKeys.ManualSelectSeriesFirst));
            return;
        }

        SelectedSeriesId = series.SeriesId;
        EditorMode = ManualEditorMode.AddPoint;
    }

    private void ApplySeriesRelations()
    {
        if (SelectedTab is not { } tab ||
            SelectedSeriesId is not string interventionSeriesId ||
            _manualWorkspaceService is null)
        {
            return;
        }

        try
        {
            string[] probeSeriesIds = ProbeRelationChoices
                .Where(static choice => choice.IsSelected && choice.SeriesId is not null)
                .Select(static choice => choice.SeriesId!)
                .ToArray();
            _manualWorkspaceService.SetSeriesRelations(
                tab.TabId,
                interventionSeriesId,
                SelectedSharedBaselineSeriesId,
                probeSeriesIds);
            SetStatus(GetLocalizedString(LocalizationKeys.ManualRelationsApplied));
            QueueAutosave(SnapshotTrigger.ExportSettingsChanged, tab.TabId, interventionSeriesId);
            RefreshSeriesRelationChoices();
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException or KeyNotFoundException)
        {
            SetStatus(FormatLocalizedString(LocalizationKeys.StatusManualEditRejectedFormat, exception.Message));
        }
    }

    private bool CanApplySeriesRelations() =>
        _manualWorkspaceService is not null &&
        SelectedTab is not null &&
        SelectedSeriesId is string seriesId &&
        FindSeries(seriesId)?.SemanticRole == SemanticRole.Intervention;

    private void RefreshSeriesRelationChoices()
    {
        BaselineRelationChoices.Clear();
        ProbeRelationChoices.Clear();
        _selectedSharedBaselineSeriesId = null;

        if (_manualWorkspaceService is null ||
            SelectedTab is not { } tab ||
            SelectedSeriesId is not string selectedSeriesId ||
            FindSeries(selectedSeriesId)?.SemanticRole != SemanticRole.Intervention)
        {
            OnPropertyChanged(nameof(SelectedSharedBaselineSeriesId));
            return;
        }

        SeriesRecord? persisted = _manualWorkspaceService.CurrentProject.Panels
            .Where(panel => string.Equals(panel.PanelId.Value.ToString("D"), tab.PanelId, StringComparison.Ordinal))
            .SelectMany(static panel => panel.Series)
            .FirstOrDefault(series => string.Equals(
                series.SeriesId.Value.ToString("D"),
                selectedSeriesId,
                StringComparison.Ordinal));

        BaselineRelationChoices.Add(new SeriesRelationChoiceViewModel(
            null,
            GetLocalizedString(LocalizationKeys.ManualNoSharedBaseline)));
        foreach (SeriesCardViewModel series in tab.SeriesCards.Where(static item => item.SemanticRole == SemanticRole.Baseline))
        {
            BaselineRelationChoices.Add(new SeriesRelationChoiceViewModel(series.SeriesId, series.Label));
        }

        _selectedSharedBaselineSeriesId = persisted?.SharedBaselineSeriesId?.Value.ToString("D");
        OnPropertyChanged(nameof(SelectedSharedBaselineSeriesId));
        HashSet<string> selectedProbeIds = persisted?.ApplicableProbeSeriesIds
            .Select(static id => id.Value.ToString("D"))
            .ToHashSet(StringComparer.Ordinal) ?? [];
        foreach (SeriesCardViewModel series in tab.SeriesCards.Where(static item =>
                     item.SemanticRole is SemanticRole.Maintenance or SemanticRole.Generalization))
        {
            ProbeRelationChoices.Add(new SeriesRelationChoiceViewModel(
                series.SeriesId,
                series.Label,
                selectedProbeIds.Contains(series.SeriesId)));
        }
    }

    private void CloseTab(WorkspaceTabViewModel? tab)
    {
        if (tab is null || !Tabs.Contains(tab))
        {
            return;
        }

        if (tab.IsDirty)
        {
            SetStatus(GetLocalizedString(LocalizationKeys.ProjectCloseDirtyBlocked));
            return;
        }

        int index = Tabs.IndexOf(tab);
        if (_manualWorkspaceService is not null)
        {
            _manualWorkspaceService.CloseTab(tab.TabId);
        }

        Tabs.Remove(tab);
        SelectedTab = Tabs.Count == 0 ? null : Tabs[Math.Min(index, Tabs.Count - 1)];
        RelayCommand.RaiseCanExecuteChanged();
    }

    private void DeleteSelectedPhaseDivider()
    {
        if (SelectedTab is not { } tab || SelectedDividerId is not string dividerId || _manualWorkspaceService is null)
        {
            return;
        }

        _manualWorkspaceService.DeletePhaseDivider(tab.TabId, dividerId);
        SelectedDividerId = tab.PhaseDividers.FirstOrDefault()?.DividerId;
        SetStatus(GetLocalizedString(LocalizationKeys.ManualDividerDeleted));
        QueueAutosave(SnapshotTrigger.PhaseEdited, tab.TabId, dividerId);
    }

    private void LabelSelectedPhaseDivider()
    {
        if (SelectedTab is not { } tab || SelectedDividerId is not string dividerId || _manualWorkspaceService is null)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(PhaseCode) || string.IsNullOrWhiteSpace(PhaseLabel))
        {
            SetStatus(GetLocalizedString(LocalizationKeys.ManualPhaseLabelRequired));
            return;
        }

        try
        {
            _manualWorkspaceService.LabelPhaseDivider(tab.TabId, dividerId, PhaseCode, PhaseLabel);
            SetStatus(GetLocalizedString(LocalizationKeys.ManualDividerLabeled));
            QueueAutosave(SnapshotTrigger.PhaseEdited, tab.TabId, dividerId);
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidOperationException)
        {
            SetStatus(FormatLocalizedString(LocalizationKeys.StatusManualEditRejectedFormat, exception.Message));
        }
    }

    private void SelectNearest(WorkspaceTabViewModel tab, Point point)
    {
        GraphPoint? nearestPoint = tab.Points
            .OrderBy(candidate => DistanceSquared(candidate.PixelX, candidate.PixelY, point.X, point.Y))
            .FirstOrDefault();
        EditablePhaseDivider? nearestDivider = tab.PhaseDividers
            .OrderBy(divider => Math.Abs(divider.OriginalX - point.X))
            .FirstOrDefault();
        double pointDistance = nearestPoint is null
            ? double.MaxValue
            : Math.Sqrt(DistanceSquared(nearestPoint.PixelX, nearestPoint.PixelY, point.X, point.Y));
        double dividerDistance = nearestDivider is null ? double.MaxValue : Math.Abs(nearestDivider.OriginalX - point.X);
        if (pointDistance <= 12 && pointDistance <= dividerDistance)
        {
            SelectedPointId = nearestPoint!.PointId;
            SelectedSeriesId = nearestPoint.SeriesId;
            SetStatus(GetLocalizedString(LocalizationKeys.ManualPointSelected));
        }
        else if (dividerDistance <= 12)
        {
            SelectedDividerId = nearestDivider!.DividerId;
            PhaseCode = nearestDivider.Code;
            PhaseLabel = nearestDivider.Label;
            SetStatus(GetLocalizedString(LocalizationKeys.ManualDividerSelected));
        }
    }

    private async Task TryAutosaveAsync(
        SnapshotTrigger trigger,
        string? tabId,
        string? entityId,
        CancellationToken cancellationToken)
    {
        if (_manualWorkspaceService is null)
        {
            return;
        }

        WorkspaceTabViewModel? editedTab = tabId is null
            ? null
            : Tabs.FirstOrDefault(tab => string.Equals(tab.TabId, tabId, StringComparison.Ordinal));
        if (editedTab is not null)
        {
            editedTab.IsDirty = true;
        }

        DomainResult<ProjectSnapshotReceipt> result = await _manualWorkspaceService
            .AutosaveAsync(trigger, tabId, entityId, cancellationToken);
        if (!result.IsSuccess && result.Errors.All(static error =>
            error.Code is not "AUTOSAVE_PATHS_UNAVAILABLE" and not "AUTOSAVE_NOT_ELIGIBLE"))
        {
            SetStatus(string.Join(" | ", result.Errors.Select(static error => error.TechnicalMessage)));
        }
    }

    public async Task RunTimerAutosaveAsync(CancellationToken cancellationToken = default)
    {
        if (_manualWorkspaceService is null || IsBusy)
        {
            return;
        }

        DomainResult<ProjectSnapshotReceipt> result = await _manualWorkspaceService
            .TimerAutosaveAsync(DateTimeOffset.UtcNow, cancellationToken);
        if (!result.IsSuccess && result.Errors.All(static error =>
            error.Code is not "AUTOSAVE_NOT_ELIGIBLE" and not "AUTOSAVE_NOT_DUE" and not "AUTOSAVE_PATHS_UNAVAILABLE"))
        {
            SetStatus(string.Join(" | ", result.Errors.Select(static error => error.TechnicalMessage)));
        }
    }

    private async void QueueAutosave(
        SnapshotTrigger trigger,
        string? tabId,
        string? entityId)
    {
        try
        {
            await TryAutosaveAsync(trigger, tabId, entityId, _shutdown.Token);
        }
        catch (OperationCanceledException) when (_shutdown.IsCancellationRequested)
        {
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or InvalidOperationException)
        {
            SetStatus(exception.Message);
        }
    }

    private bool CanEditPoint() =>
        _manualWorkspaceService is not null && SelectedTab is not null && SelectedSeriesId is not null;

    private bool CanMoveSelectedPoint() =>
        _manualWorkspaceService is not null && SelectedTab is not null && SelectedPointId is not null;

    private void SetStatus(string message)
    {
        _statusMessageOverride = message;
        OnPropertyChanged(nameof(StatusMessage));
    }

    private static string SymbolFor(MarkerShape shape, MarkerFill fill) => (shape, fill) switch
    {
        (MarkerShape.Circle, MarkerFill.Open) => "○",
        (MarkerShape.Circle, MarkerFill.Unknown) => "◌",
        (MarkerShape.Circle, MarkerFill.Filled) => "●",
        (MarkerShape.Square, MarkerFill.Open) => "□",
        (MarkerShape.Square, MarkerFill.Unknown) => "▧",
        (MarkerShape.Square, MarkerFill.Filled) => "■",
        (MarkerShape.TriangleUp, MarkerFill.Open) => "△",
        (MarkerShape.TriangleUp, MarkerFill.Unknown) => "△",
        (MarkerShape.TriangleUp, MarkerFill.Filled) => "▲",
        (MarkerShape.TriangleDown, MarkerFill.Open or MarkerFill.Unknown) => "▽",
        (MarkerShape.TriangleDown, MarkerFill.Filled) => "▼",
        (MarkerShape.Diamond, MarkerFill.Open) => "◇",
        (MarkerShape.Diamond, MarkerFill.Unknown) => "◇",
        (MarkerShape.Diamond, MarkerFill.Filled) => "◆",
        (MarkerShape.Star or MarkerShape.Asterisk, _) => "✱",
        (MarkerShape.Cross, _) => "✕",
        _ => "?",
    };

    private string JoinStages(
        IEnumerable<AutomaticStageStatus> stages,
        Func<AutomaticStageState, bool> include)
    {
        string[] names = stages.Where(stage => include(stage.State)).Select(stage => stage.Stage).ToArray();
        return names.Length == 0
            ? GetLocalizedString(LocalizationKeys.PreviewNone)
            : string.Join(", ", names);
    }

    private static double DistanceSquared(double x1, double y1, double x2, double y2)
    {
        double dx = x1 - x2;
        double dy = y1 - y2;
        return (dx * dx) + (dy * dy);
    }

    private static void EnsureManualOverlay(WorkspaceTabViewModel tab)
    {
        if (tab.OverlayContent is null && tab.PixelWidth > 0 && tab.PixelHeight > 0)
        {
            tab.SetOverlayContent(new ManualReviewOverlay(
                tab.Points,
                tab.SeriesCards,
                tab.PhaseDividers,
                tab.PixelWidth,
                tab.PixelHeight));
        }
    }

    private void ConfigureSeriesSelection(WorkspaceTabViewModel tab)
    {
        foreach (SeriesCardViewModel series in tab.SeriesCards)
        {
            ConfigureSeriesSelection(series);
        }
    }

    private void ConfigureSeriesSelection(SeriesCardViewModel series)
    {
        series.SetSelectAction(seriesId =>
        {
            SelectedSeriesId = seriesId;
            SetStatus(FormatLocalizedString(LocalizationKeys.ManualSeriesSelectedFormat, series.Label));
        });
    }

    private void ChangeZoom(double factor)
    {
        if (SelectedTab is { } tab)
        {
            SetZoom(tab.ZoomLevel * factor);
        }
    }

    private void SelectRelativeTab(int offset)
    {
        if (Tabs.Count < 2)
        {
            return;
        }

        int current = SelectedTab is null ? 0 : Math.Max(0, Tabs.IndexOf(SelectedTab));
        int next = (current + offset) % Tabs.Count;
        if (next < 0)
        {
            next += Tabs.Count;
        }

        SelectedTab = Tabs[next];
    }

    private void SetZoom(double value)
    {
        if (SelectedTab is { } tab)
        {
            tab.ZoomLevel = Math.Clamp(value, 0.25, 8);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _activeOperation?.Cancel();
        _activeOperation?.Dispose();
        _shutdown.Cancel();
        _shutdown.Dispose();
        _disposed = true;
    }

    private AsyncRelayCommand CreateWorkflowCommand(WorkflowStage stage, string statusMessageKey)
    {
        AsyncRelayCommand command = new(
            cancellationToken => RunWorkflowStageAsync(stage, statusMessageKey, cancellationToken),
            () => !IsBusy && _isWorkflowAvailable && CanRunAutomaticStage(stage));
        return command;
    }

    private bool CanRunAutomaticStage(WorkflowStage stage)
    {
        if (_workspaceService is not IRuntimeWorkspaceService runtime)
        {
            return true;
        }

        static bool IsEnabled(AutomaticStageStatus status, bool allowExperimental = false) =>
            status.State is AutomaticStageState.Available or AutomaticStageState.Approved ||
            (allowExperimental && status.State == AutomaticStageState.Experimental);

        return stage switch
        {
            WorkflowStage.Prepare => runtime.AutomaticStages
                .Any(status => string.Equals(status.Stage, "enhancement", StringComparison.Ordinal) && IsEnabled(status, allowExperimental: true)),
            WorkflowStage.Detect => RequiredAutomaticDetectionStages
                .All(required => runtime.AutomaticStages.Any(status =>
                    string.Equals(status.Stage, required, StringComparison.Ordinal) && IsEnabled(status))),
            _ => true,
        };
    }

    private async Task RunWorkflowStageAsync(
        WorkflowStage stage,
        string statusMessageKey,
        CancellationToken commandCancellationToken)
    {
        if (IsBusy)
        {
            return;
        }

        _activeOperation?.Dispose();
        _activeOperation = CancellationTokenSource.CreateLinkedTokenSource(
            _shutdown.Token,
            commandCancellationToken);
        IsBusy = true;
        try
        {
            await _workspaceService.RunStageAsync(stage, _activeOperation.Token);
            CurrentStage = stage;
            StatusMessageKey = statusMessageKey;
        }
        catch (OperationCanceledException) when (_activeOperation.IsCancellationRequested)
        {
            StatusMessageKey = "Workflow.Review";
        }
        catch (InvalidOperationException exception)
        {
            SetStatus(exception.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void CancelActiveOperation() => _activeOperation?.Cancel();

    private bool CanExecuteReassignCommand(object? parameter)
    {
        if (parameter is ReassignPointRequest request)
        {
            return FindPoint(request.PointId) is not null &&
                FindSeries(request.TargetSeriesId) is not null;
        }

        return parameter is string targetSeriesId &&
            SelectedPointId is string pointId &&
            FindPoint(pointId) is GraphPoint point &&
            point.SeriesId != targetSeriesId &&
            FindSeries(targetSeriesId) is not null;
    }

    private void ExecuteReassignCommand(object? parameter)
    {
        if (parameter is ReassignPointRequest request)
        {
            ReassignPoint(request.PointId, request.TargetSeriesId);
            return;
        }

        if (parameter is string targetSeriesId && SelectedPointId is string pointId)
        {
            ReassignPoint(pointId, targetSeriesId);
            return;
        }

        throw new ArgumentException("A point and target series are required.", nameof(parameter));
    }

    private WorkspaceTabViewModel RequireSelectedTab() =>
        SelectedTab ?? throw new InvalidOperationException("No workspace tab is selected.");

    private GraphReader.App.Models.GraphPoint? FindPoint(string pointId) =>
        SelectedTab?.Points.FirstOrDefault(point => point.PointId == pointId);

    private GraphReader.App.Models.GraphPoint RequirePoint(string pointId) =>
        FindPoint(pointId) ?? throw new KeyNotFoundException($"Point '{pointId}' does not exist.");

    private SeriesCardViewModel? FindSeries(string seriesId) =>
        SelectedTab?.SeriesCards.FirstOrDefault(series => series.SeriesId == seriesId);

    private SeriesCardViewModel RequireSeries(string seriesId) =>
        FindSeries(seriesId) ?? throw new KeyNotFoundException($"Series '{seriesId}' does not exist.");
}
