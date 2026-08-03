// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Windows.Input;
using GraphReader.App.Appearance;
using GraphReader.App.Localization;
using GraphReader.App.Models;
using GraphReader.App.Services;

namespace GraphReader.App.ViewModels;

public sealed class MainWindowViewModel : ObservableObject, IDisposable
{
    private readonly IWorkspaceService _workspaceService;
    private readonly ILocalizationService? _localizationService;
    private readonly CancellationTokenSource _shutdown = new();
    private WorkspaceTabViewModel? _selectedTab;
    private WorkflowStage _currentStage = WorkflowStage.Review;
    private bool _isBusy;
    private string _statusMessageKey = "Workflow.Review";
    private bool _isPhaseOverlayVisible = true;
    private ApplicationTheme _appearanceMode = ApplicationTheme.System;
    private string? _selectedPointId;
    private int _nextManualPointId = 1;
    private int _nextSplitSeriesId = 1;
    private CancellationTokenSource? _activeOperation;
    private bool _disposed;

    public MainWindowViewModel()
        : this(new FakeWorkspaceService())
    {
    }

    public MainWindowViewModel(IWorkspaceService workspaceService)
        : this(workspaceService, null)
    {
    }

    public MainWindowViewModel(
        IWorkspaceService workspaceService,
        ILocalizationService? localizationService)
    {
        _workspaceService = workspaceService ?? throw new ArgumentNullException(nameof(workspaceService));
        _localizationService = localizationService;
        Tabs = new ObservableCollection<WorkspaceTabViewModel>(_workspaceService.CreateWorkspace());
        _selectedTab = Tabs.FirstOrDefault();
        _selectedPointId = _selectedTab?.Points.FirstOrDefault()?.PointId;
        Magnifier = new MagnifierViewModel();

        ImportCommand = CreateWorkflowCommand(WorkflowStage.Import, "Workflow.Import");
        EnhanceCommand = CreateWorkflowCommand(WorkflowStage.Prepare, "Workflow.Enhance");
        AutoDetectCommand = CreateWorkflowCommand(WorkflowStage.Detect, "Workflow.AutoDetect");
        ReviewCommand = CreateWorkflowCommand(WorkflowStage.Review, "Workflow.Review");
        ExportCommand = CreateWorkflowCommand(WorkflowStage.Export, "Workflow.Export");
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
        private set => SetProperty(ref _statusMessageKey, value);
    }

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

    private static ObservableCollection<SeriesCardViewModel> EmptySeriesCards { get; } = [];

    public void AddPoint(string seriesId)
    {
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

    public void ReassignPoint(string pointId, string targetSeriesId)
    {
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
            () => !IsBusy);
        return command;
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

    private GraphPoint? FindPoint(string pointId) =>
        SelectedTab?.Points.FirstOrDefault(point => point.PointId == pointId);

    private GraphPoint RequirePoint(string pointId) =>
        FindPoint(pointId) ?? throw new KeyNotFoundException($"Point '{pointId}' does not exist.");

    private SeriesCardViewModel? FindSeries(string seriesId) =>
        SelectedTab?.SeriesCards.FirstOrDefault(series => series.SeriesId == seriesId);

    private SeriesCardViewModel RequireSeries(string seriesId) =>
        FindSeries(seriesId) ?? throw new KeyNotFoundException($"Series '{seriesId}' does not exist.");
}
