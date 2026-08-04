// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.IO;
using System.Reflection;
using System.Text.Json;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using GraphReader.App.Integration.Workflow;
using GraphReader.App.Localization;
using GraphReader.App.Models;
using GraphReader.App.ViewModels;
using GraphReader.Axis;
using GraphReader.Domain;
using GraphReader.Export;
using GraphReader.Imaging;
using GraphReader.Phases;
using GraphReader.SuperResolution;
using AxisPixelPoint = GraphReader.Axis.PixelPoint;
using AppGraphPoint = GraphReader.App.Models.GraphPoint;
using DomainCalibrationAnchor = GraphReader.Domain.CalibrationAnchor;
using DomainCalibrationAnchorKind = GraphReader.Domain.CalibrationAnchorKind;
using DomainCalibrationStatus = GraphReader.Domain.CalibrationStatus;
using DomainGraphPoint = GraphReader.Domain.GraphPoint;
using DomainPixelPoint = GraphReader.Domain.PixelPoint;
using DomainPhaseNormalizedType = GraphReader.Domain.PhaseNormalizedType;
using ExportPhaseType = GraphReader.Export.ExportPhaseType;
using ExportReviewStatus = GraphReader.Export.ExportReviewStatus;
using PhaseNormalizedType = GraphReader.Phases.PhaseNormalizedType;

namespace GraphReader.App.Services;

/// <summary>
/// Real-data manual workspace. It intentionally begins empty and never supplies
/// recorded detections when production models are absent.
/// </summary>
public sealed class ManualPreviewWorkspaceService : IManualWorkspaceService
{
    private readonly IApplicationPaths? _applicationPaths;
    private readonly IImageImportService _imageImportService;
    private readonly ProjectFileStore _projectFileStore;
    private readonly IExportService _exportService;
    private readonly IPhaseManualEditor _phaseEditor;
    private readonly WorkflowRuntimeEnvironment _runtimeEnvironment;
    private readonly IReadOnlyList<AutomaticStageStatus> _automaticStages;
    private readonly Func<CancellationToken, Task<RealEsrganBackendResolution>>? _enhancementResolver;
    private readonly List<WorkspaceTabViewModel> _tabs = [];
    private readonly Dictionary<string, PhaseManualOverrides> _phaseOverrides = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ManualPointXState> _pointXStates = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<PointModification>> _pointModificationHistories = new(StringComparer.Ordinal);
    private readonly Dictionary<string, JsonElement> _enhancementByTab = new(StringComparer.Ordinal);
    private readonly Dictionary<string, EnhancementEnvelope> _enhancementEnvelopes = new(StringComparer.Ordinal);
    private IReadOnlyList<ImageImportError> _lastImportErrors = [];

    public ManualPreviewWorkspaceService(
        IApplicationPaths? applicationPaths = null,
        IImageImportService? imageImportService = null,
        ProjectFileStore? projectFileStore = null,
        IExportService? exportService = null,
        IPhaseManualEditor? phaseEditor = null,
        WorkflowRuntimeEnvironment runtimeEnvironment = WorkflowRuntimeEnvironment.ManualPreview,
        IReadOnlyList<AutomaticStageStatus>? automaticStages = null,
        Func<CancellationToken, Task<RealEsrganBackendResolution>>? enhancementResolver = null)
    {
        if (runtimeEnvironment == WorkflowRuntimeEnvironment.RecordedFake)
        {
            throw new ArgumentOutOfRangeException(
                nameof(runtimeEnvironment),
                "The real-data manual workspace cannot use the recorded-fake runtime identity.");
        }

        _applicationPaths = applicationPaths;
        _imageImportService = imageImportService ?? new ImageImportService();
        _projectFileStore = projectFileStore ?? new ProjectFileStore();
        _exportService = exportService ?? new ExportService();
        _phaseEditor = phaseEditor ?? new PhaseManualEditor();
        _runtimeEnvironment = runtimeEnvironment;
        _automaticStages = (automaticStages ?? DefaultAutomaticStages).ToArray();
        _enhancementResolver = enhancementResolver;
        CurrentProject = ProjectDocument.Create(GetApplicationVersion(), DateTimeOffset.UtcNow);
    }

    private static IReadOnlyList<AutomaticStageStatus> DefaultAutomaticStages { get; } =
    [
        new("enhancement", AutomaticStageState.Unavailable, "Enhancement is unavailable. Install an approved Real-ESRGAN runtime and model to enable it."),
        new("axis", AutomaticStageState.Unavailable, "Automatic axis detection is unavailable. Use manual three-anchor calibration."),
        new("ocr", AutomaticStageState.Unavailable, "Install approved OCR models to enable text recognition."),
        new("markers", AutomaticStageState.Unavailable, "Install approved marker models to enable marker detection."),
        new("legends", AutomaticStageState.Unavailable, "Automatic legend reasoning is unavailable. Create and label series manually."),
        new("phases", AutomaticStageState.Unavailable, "Automatic phase detection is unavailable. Add and label dividers manually."),
    ];

    public WorkflowRuntimeEnvironment RuntimeEnvironment => _runtimeEnvironment;

    public IReadOnlyList<AutomaticStageStatus> AutomaticStages => _automaticStages;

    public bool UsesFakeGraphData => false;

    public ProjectDocument CurrentProject { get; private set; }

    public string? CurrentProjectPath { get; private set; }

    public IReadOnlyList<ImageImportError> LastImportErrors => _lastImportErrors;

    public IReadOnlyList<WorkspaceTabViewModel> CreateWorkspace() => _tabs.ToArray();

    public async Task<IReadOnlyList<WorkspaceTabViewModel>> ImportImagesAsync(
        IEnumerable<string> paths,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(paths);
        string[] requestedPaths = paths.ToArray();
        if (requestedPaths.Length == 0)
        {
            return [];
        }

        BatchImportResult result = await _imageImportService
            .ImportBatchAsync(requestedPaths, cancellationToken)
            .ConfigureAwait(false);
        _lastImportErrors = result.Items
            .Where(static item => item.Error is not null)
            .Select(static item => item.Error!)
            .ToArray();

        var addedTabs = new List<WorkspaceTabViewModel>();
        var addedSources = new List<SourceReference>();
        foreach (ImageImportResult item in result.Items.Where(static item => item.Image is not null))
        {
            cancellationToken.ThrowIfCancellationRequested();
            ImportedImage image = item.Image!;
            var source = new SourceReference(
                SourceId.New(),
                SourceKind.Image,
                Path.GetFileName(image.SourcePath),
                image.SourcePath,
                image.Sha256,
                ArticleMetadata: null);
            var panelId = PanelId.New();
            WorkspaceTabViewModel tab = CreateEmptyImageTab(panelId, source, image);
            _tabs.Add(tab);
            addedTabs.Add(tab);
            addedSources.Add(source);
            _phaseOverrides[tab.TabId] = new PhaseManualOverrides();
        }

        if (addedTabs.Count > 0)
        {
            CurrentProject = CurrentProject with
            {
                ModifiedUtc = DateTimeOffset.UtcNow,
                Sources = CurrentProject.Sources.Concat(addedSources).ToArray(),
            };
            SynchronizeProject(DomainEventKind.DetectionAccepted, panelId: null, entityId: null, "Real image import");
        }

        return addedTabs;
    }

    public async Task<IReadOnlyList<WorkspaceTabViewModel>> OpenProjectAsync(
        string path,
        CancellationToken cancellationToken)
    {
        DomainResult<ProjectDocument> loaded = await _projectFileStore.LoadAsync(path, cancellationToken)
            .ConfigureAwait(false);
        if (!loaded.IsSuccess || loaded.Value is null)
        {
            throw new InvalidOperationException(FormatErrors(loaded.Errors));
        }

        ProjectDocument project = loaded.Value;
        var importedBySource = new Dictionary<SourceId, ImportedImage>();
        var errors = new List<ImageImportError>();
        foreach (SourceReference source in project.Sources.Where(static source => source.Kind == SourceKind.Image))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(source.LocalPath))
            {
                throw new InvalidOperationException($"Source '{source.DisplayName}' has no local image path.");
            }

            ImageImportResult imported = await _imageImportService.ImportAsync(source.LocalPath, cancellationToken)
                .ConfigureAwait(false);
            if (imported.Image is null)
            {
                errors.Add(imported.Error!);
                continue;
            }

            if (!string.Equals(imported.Image.Sha256, source.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException($"Source '{source.DisplayName}' no longer matches its saved SHA-256.");
            }

            importedBySource[source.SourceId] = imported.Image;
        }

        if (errors.Count > 0)
        {
            _lastImportErrors = errors;
            throw new InvalidOperationException(string.Join(" | ", errors.Select(static error => error.TechnicalMessage)));
        }

        _tabs.Clear();
        _phaseOverrides.Clear();
        _pointXStates.Clear();
        _pointModificationHistories.Clear();
        _enhancementByTab.Clear();
        _enhancementEnvelopes.Clear();
        foreach (PanelRecord panel in project.Panels)
        {
            if (!importedBySource.TryGetValue(panel.SourceId, out ImportedImage? image))
            {
                continue;
            }

            SourceReference source = project.Sources.Single(item => item.SourceId == panel.SourceId);
            WorkspaceTabViewModel tab = CreateTabFromProject(panel, source, image);
            ReindexAllSeries(tab);
            _tabs.Add(tab);
            _phaseOverrides[tab.TabId] = CreatePhaseOverrides(tab);
            if (panel.Enhancement is JsonElement enhancement)
            {
                _enhancementByTab[tab.TabId] = enhancement.Clone();
            }
            foreach (PointRecord point in panel.Points)
            {
                string pointId = point.PointId.Value.ToString("D");
                _pointXStates[pointId] = new ManualPointXState(
                    point.PrintedXValue,
                    point.EstimatedXValue,
                    point.XSource,
                    point.XConfidence,
                    point.GraphX.HasValue);
                _pointModificationHistories[pointId] = point.ModificationHistory.ToList();
            }
        }

        CurrentProject = project;
        CurrentProjectPath = Path.GetFullPath(path);
        _lastImportErrors = [];
        return CreateWorkspace();
    }

    public async Task<WorkspaceEnhancementResult> EnhanceAsync(
        string tabId,
        CancellationToken cancellationToken)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        if (_enhancementResolver is null)
        {
            return new WorkspaceEnhancementResult(
                false,
                "Official Real-ESRGAN enhancement is not configured. The original image remains unchanged.",
                UserMessageKey: LocalizationKeys.WorkflowEnhanceUnavailable);
        }

        if (_applicationPaths is null || string.IsNullOrWhiteSpace(tab.SourcePath) || tab.PanelId is null)
        {
            return new WorkspaceEnhancementResult(
                false,
                "Enhancement storage or source metadata is unavailable. The original image remains unchanged.",
                UserMessageKey: LocalizationKeys.EnhancementStorageUnavailable);
        }

        RealEsrganBackendResolution backend = await _enhancementResolver(cancellationToken).ConfigureAwait(false);
        if (!backend.IsAvailable || backend.Service is null || backend.Model is null)
        {
            string message = backend.Diagnostic?.TechnicalMessage ??
                "The configured Real-ESRGAN runtime or model is unavailable.";
            return new WorkspaceEnhancementResult(
                false,
                $"{message} The original image remains unchanged.",
                UserMessageKey: backend.Diagnostic?.UserMessageKey ?? LocalizationKeys.EnhancementRuntimeUnavailable);
        }

        string derivativeRoot = Path.Combine(_applicationPaths.CacheRoot, "Enhancement", "Derivatives");
        Directory.CreateDirectory(derivativeRoot);
        string outputPath = Path.Combine(
            derivativeRoot,
            $"{Guid.Parse(tab.PanelId):N}-{Guid.NewGuid():N}-realesrgan-x2.png");
        var request = new EnhancementRequest(
            CurrentProject.ProjectId.Value,
            Guid.Parse(tab.PanelId),
            tab.SourcePath,
            outputPath,
            new PixelDimensions(tab.PixelWidth, tab.PixelHeight),
            backend.Model,
            new EnhancementOptions(Scale: 2, ContinueWithoutEnhancement: true));
        EnhancementResult result = await backend.Service.EnhanceAsync(request, cancellationToken)
            .ConfigureAwait(false);
        if (!result.IsSuccess || result.OutputPath is null || result.Envelope is null)
        {
            return new WorkspaceEnhancementResult(
                false,
                $"{result.Diagnostic.Message} The original image remains unchanged.",
                RuntimeMilliseconds: result.Envelope?.TimingMs.Total ?? 0,
                UserMessageKey: LocalizationKeys.EnhancementExecutionFailed);
        }

        byte[] enhancedBytes = await File.ReadAllBytesAsync(result.OutputPath, cancellationToken)
            .ConfigureAwait(false);
        tab.EnhancedImageSource = CreateBitmap(new ImmutableImageBytes(enhancedBytes));
        tab.EnhancementPreviewMode = EnhancementPreviewMode.Enhanced;
        _enhancementEnvelopes[tab.TabId] = result.Envelope;
        UpdateEnhancementProvenance(tab, result.Envelope);
        SynchronizeProject(
            DomainEventKind.DetectionAccepted,
            Guid.Parse(tab.PanelId),
            backend.Model.ModelId,
            "Official Real-ESRGAN x2 derivative recorded for local evaluation");
        return new WorkspaceEnhancementResult(
            true,
            $"Enhanced preview ready with {backend.Model.ModelId} at 2x. The original image is unchanged.",
            result.OutputPath,
            result.Envelope.TimingMs.Total,
            LocalizationKeys.StatusEnhancementReadyFormat,
            [backend.Model.ModelId]);
    }

    public void SetEnhancementPreviewMode(string tabId, EnhancementPreviewMode mode)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        tab.EnhancementPreviewMode = mode;
        if (_enhancementEnvelopes.TryGetValue(tab.TabId, out EnhancementEnvelope? envelope))
        {
            UpdateEnhancementProvenance(tab, envelope);
            SynchronizeProject(
                DomainEventKind.ExportSettingsChanged,
                Guid.Parse(tab.PanelId!),
                envelope.Model.ModelId,
                $"Enhancement preview mode changed to {tab.EnhancementPreviewMode}");
        }
    }

    public bool CloseTab(string tabId)
    {
        WorkspaceTabViewModel? tab = _tabs.SingleOrDefault(
            item => string.Equals(item.TabId, tabId, StringComparison.Ordinal));
        if (tab is null)
        {
            return false;
        }

        _tabs.Remove(tab);
        _phaseOverrides.Remove(tab.TabId);
        _enhancementByTab.Remove(tab.TabId);
        _enhancementEnvelopes.Remove(tab.TabId);
        foreach (AppGraphPoint point in tab.Points)
        {
            _pointXStates.Remove(point.PointId);
            _pointModificationHistories.Remove(point.PointId);
        }

        return true;
    }

    public async Task<DomainResult<ProjectSaveReceipt>> SaveProjectAsync(
        string? path,
        CancellationToken cancellationToken)
    {
        string? selectedPath = string.IsNullOrWhiteSpace(path) ? CurrentProjectPath : Path.GetFullPath(path);
        if (string.IsNullOrWhiteSpace(selectedPath))
        {
            return DomainResult<ProjectSaveReceipt>.Failure(new DomainError(
                "PROJECT_SAVE_PATH_REQUIRED",
                DomainErrorSeverity.Warning,
                "Errors.ProjectSavePathInvalid",
                "Select a project file path before saving.",
                Recoverable: true,
                "select_project_path"));
        }

        SynchronizeProject(DomainEventKind.ExportSettingsChanged, panelId: null, entityId: null, "Manual project save");
        DomainResult<ProjectSaveReceipt> saved = await _projectFileStore
            .SaveAsync(CurrentProject, selectedPath, cancellationToken)
            .ConfigureAwait(false);
        if (saved.IsSuccess)
        {
            CurrentProjectPath = selectedPath;
        }

        return saved;
    }

    public ManualCalibrationState Calibrate(string tabId, ManualCalibrationRequest request)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        ValidateCalibrationRequest(request);
        LinearAxisTransform xTransform = FitTransform(
            request.Session1Y0.X,
            1,
            request.SessionMaximumY0.X,
            request.XMaximum);
        LinearAxisTransform yTransform = FitTransform(
            request.Session1Y0.Y,
            0,
            request.Session1YMaximum.Y,
            request.YMaximum);

        var state = new ManualCalibrationState(
            request.Session1Y0,
            request.Session1YMaximum,
            request.SessionMaximumY0,
            request.YMaximum,
            request.XMaximum,
            xTransform,
            yTransform,
            Confidence: 1);
        tab.Calibration = state;
        foreach (AppGraphPoint point in tab.Points)
        {
            UpdatePointCoordinates(tab, point);
            MarkPointXAsManualEstimate(tab, point);
        }

        SynchronizeProject(
            DomainEventKind.CalibrationChanged,
            Guid.Parse(tab.PanelId!),
            entityId: null,
            "Manual three-anchor calibration");
        return state;
    }

    public SeriesCardViewModel AddSeries(string tabId, ManualSeriesDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentException.ThrowIfNullOrWhiteSpace(definition.DisplayName);
        ArgumentException.ThrowIfNullOrWhiteSpace(definition.Symbol);
        WorkspaceTabViewModel tab = RequireTab(tabId);
        var series = new SeriesCardViewModel(
            Guid.NewGuid().ToString("D"),
            definition.Symbol,
            $"{definition.Fill} {definition.Shape}",
            definition.DisplayName,
            1,
            tab.Points,
            definition.Shape,
            definition.Fill,
            definition.SemanticRole);
        tab.SeriesCards.Add(series);
        SynchronizeProject(DomainEventKind.PointEdited, Guid.Parse(tab.PanelId!), series.SeriesId, "Manual series created");
        return series;
    }

    public void UpdateSeries(string tabId, string seriesId, ManualSeriesDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentException.ThrowIfNullOrWhiteSpace(definition.DisplayName);
        ArgumentException.ThrowIfNullOrWhiteSpace(definition.Symbol);
        WorkspaceTabViewModel tab = RequireTab(tabId);
        SeriesCardViewModel series = RequireSeries(tab, seriesId);
        series.Label = definition.DisplayName;
        series.Symbol = definition.Symbol;
        series.AccessibleName = $"{definition.Fill} {definition.Shape}";
        series.Shape = definition.Shape;
        series.Fill = definition.Fill;
        series.SemanticRole = definition.SemanticRole;
        SynchronizeProject(
            DomainEventKind.PointEdited,
            Guid.Parse(tab.PanelId!),
            series.SeriesId,
            "Manual series edited");
    }

    public void SetSeriesRelations(
        string tabId,
        string interventionSeriesId,
        string? sharedBaselineSeriesId,
        IEnumerable<string> applicableProbeSeriesIds)
    {
        ArgumentNullException.ThrowIfNull(applicableProbeSeriesIds);
        WorkspaceTabViewModel tab = RequireTab(tabId);
        SeriesCardViewModel intervention = RequireSeries(tab, interventionSeriesId);
        if (intervention.SemanticRole != SemanticRole.Intervention)
        {
            throw new ArgumentException("Series relations can be assigned only to an intervention series.", nameof(interventionSeriesId));
        }

        SeriesId? baselineId = null;
        if (sharedBaselineSeriesId is not null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(sharedBaselineSeriesId);
            if (string.Equals(sharedBaselineSeriesId, intervention.SeriesId, StringComparison.Ordinal))
            {
                throw new ArgumentException("An intervention series cannot link to itself as its shared baseline.", nameof(sharedBaselineSeriesId));
            }

            SeriesCardViewModel baseline = RequireSeries(tab, sharedBaselineSeriesId);
            if (baseline.SemanticRole != SemanticRole.Baseline)
            {
                throw new ArgumentException("A shared baseline link must reference a baseline series in the same tab.", nameof(sharedBaselineSeriesId));
            }

            baselineId = SeriesId.FromGuid(Guid.Parse(baseline.SeriesId));
        }

        string[] requestedProbeIds = applicableProbeSeriesIds.ToArray();
        var probeIds = new SeriesId[requestedProbeIds.Length];
        for (int index = 0; index < requestedProbeIds.Length; index++)
        {
            string probeSeriesId = requestedProbeIds[index];
            ArgumentException.ThrowIfNullOrWhiteSpace(probeSeriesId);
            if (string.Equals(probeSeriesId, intervention.SeriesId, StringComparison.Ordinal))
            {
                throw new ArgumentException("An intervention series cannot link to itself as an applicable probe.", nameof(applicableProbeSeriesIds));
            }

            SeriesCardViewModel probe = RequireSeries(tab, probeSeriesId);
            if (probe.SemanticRole is not (SemanticRole.Maintenance or SemanticRole.Generalization))
            {
                throw new ArgumentException(
                    "An applicable probe link must reference a maintenance or generalization series in the same tab.",
                    nameof(applicableProbeSeriesIds));
            }

            probeIds[index] = SeriesId.FromGuid(Guid.Parse(probe.SeriesId));
        }

        PanelId panelId = PanelId.FromGuid(Guid.Parse(tab.PanelId!));
        SeriesId interventionId = SeriesId.FromGuid(Guid.Parse(intervention.SeriesId));
        CurrentProject = CurrentProject with
        {
            Panels = CurrentProject.Panels.Select(panel => panel.PanelId == panelId
                ? panel with
                {
                    Series = panel.Series.Select(series => series.SeriesId == interventionId
                        ? series with
                        {
                            SharedBaselineSeriesId = baselineId,
                            ApplicableProbeSeriesIds = probeIds,
                        }
                        : series).ToArray(),
                }
                : panel).ToArray(),
        };
        SynchronizeProject(
            DomainEventKind.ExportSettingsChanged,
            panelId.Value,
            intervention.SeriesId,
            "Manual intervention series relations changed");
    }

    public AppGraphPoint AddPoint(string tabId, string seriesId, double pixelX, double pixelY)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        _ = RequireSeries(tab, seriesId);
        EnsureFinitePoint(pixelX, pixelY);
        var point = new AppGraphPoint(
            Guid.NewGuid().ToString("D"),
            seriesId,
            pixelX,
            pixelY,
            0,
            0,
            "a",
            observationIndex: 1);
        UpdatePointCoordinates(tab, point);
        MarkPointXAsManualEstimate(tab, point);
        _pointModificationHistories[point.PointId] = [];
        tab.Points.Add(point);
        ReindexSeries(tab, seriesId);
        RequireSeries(tab, seriesId).NotifyCountChanged();
        SynchronizeProject(DomainEventKind.PointEdited, Guid.Parse(tab.PanelId!), point.PointId, "Manual point added");
        return point;
    }

    public void MovePoint(string tabId, string pointId, double pixelX, double pixelY)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        EnsureFinitePoint(pixelX, pixelY);
        AppGraphPoint point = RequirePoint(tab, pointId);
        DomainPixelPoint previousPixel = new(point.PixelX, point.PixelY);
        DomainGraphPoint? previousGraph = GetPreviousGraph(tab, point);
        point.PixelX = pixelX;
        point.PixelY = pixelY;
        UpdatePointCoordinates(tab, point);
        MarkPointXAsManualEstimate(tab, point);
        AppendPointModification(point, previousPixel, previousGraph, "Manual point moved");
        ReindexSeries(tab, point.SeriesId);
        SynchronizeProject(DomainEventKind.PointEdited, Guid.Parse(tab.PanelId!), point.PointId, "Manual point moved");
    }

    public void DeletePoint(string tabId, string pointId)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        AppGraphPoint point = RequirePoint(tab, pointId);
        string sourceSeriesId = point.SeriesId;
        tab.Points.Remove(point);
        _pointXStates.Remove(point.PointId);
        _pointModificationHistories.Remove(point.PointId);
        ReindexSeries(tab, sourceSeriesId);
        RequireSeries(tab, point.SeriesId).NotifyCountChanged();
        SynchronizeProject(DomainEventKind.PointEdited, Guid.Parse(tab.PanelId!), point.PointId, "Manual point deleted");
    }

    public void ReassignPoint(string tabId, string pointId, string targetSeriesId)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        AppGraphPoint point = RequirePoint(tab, pointId);
        SeriesCardViewModel target = RequireSeries(tab, targetSeriesId);
        SeriesCardViewModel source = RequireSeries(tab, point.SeriesId);
        if (string.Equals(source.SeriesId, target.SeriesId, StringComparison.Ordinal))
        {
            return;
        }

        DomainPixelPoint previousPixel = new(point.PixelX, point.PixelY);
        DomainGraphPoint? previousGraph = GetPreviousGraph(tab, point);
        point.SeriesId = target.SeriesId;
        UpdatePointCoordinates(tab, point);
        AppendPointModification(
            point,
            previousPixel,
            previousGraph,
            $"Manual point reassigned from series '{source.SeriesId}' to '{target.SeriesId}'");
        ReindexSeries(tab, source.SeriesId);
        ReindexSeries(tab, target.SeriesId);
        source.NotifyCountChanged();
        target.NotifyCountChanged();
        SynchronizeProject(DomainEventKind.PointEdited, Guid.Parse(tab.PanelId!), point.PointId, "Manual point reassigned");
    }

    public EditablePhaseDivider AddPhaseDivider(string tabId, double originalX, string code, string label)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        ValidatePhaseLabel(code, label);
        string dividerId = Guid.NewGuid().ToString("D");
        PhaseManualOverrides current = GetOverrides(tab);
        PhaseEditResult edited = _phaseEditor.Apply(
            current,
            new AddPhaseDividerCommand(Guid.NewGuid().ToString("D"), dividerId, originalX, PhaseDividerStyle.Dashed),
            PlotBounds(tab),
            CancellationToken.None);
        RequirePhaseSuccess(edited);
        _phaseOverrides[tab.TabId] = edited.Overrides;
        var divider = new EditablePhaseDivider(dividerId, originalX, code.Trim(), label.Trim());
        tab.PhaseDividers.Add(divider);
        SortDividers(tab);
        UpdateAllPointPhases(tab);
        SynchronizeProject(DomainEventKind.PhaseEdited, Guid.Parse(tab.PanelId!), dividerId, "Manual phase divider added");
        return divider;
    }

    public void MovePhaseDivider(string tabId, string dividerId, double originalX)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        EditablePhaseDivider divider = RequireDivider(tab, dividerId);
        PhaseEditResult edited = _phaseEditor.Apply(
            GetOverrides(tab),
            new MovePhaseDividerCommand(
                Guid.NewGuid().ToString("D"),
                divider.DividerId,
                originalX,
                PhaseDividerStyle.Dashed,
                divider.OriginalX),
            PlotBounds(tab),
            CancellationToken.None);
        RequirePhaseSuccess(edited);
        _phaseOverrides[tab.TabId] = edited.Overrides;
        divider.OriginalX = originalX;
        SortDividers(tab);
        UpdateAllPointPhases(tab);
        SynchronizeProject(DomainEventKind.PhaseEdited, Guid.Parse(tab.PanelId!), dividerId, "Manual phase divider moved");
    }

    public void DeletePhaseDivider(string tabId, string dividerId)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        EditablePhaseDivider divider = RequireDivider(tab, dividerId);
        PhaseEditResult edited = _phaseEditor.Apply(
            GetOverrides(tab),
            new DeletePhaseDividerCommand(Guid.NewGuid().ToString("D"), divider.DividerId, divider.OriginalX),
            PlotBounds(tab),
            CancellationToken.None);
        RequirePhaseSuccess(edited);
        _phaseOverrides[tab.TabId] = edited.Overrides;
        tab.PhaseDividers.Remove(divider);
        UpdateAllPointPhases(tab);
        SynchronizeProject(DomainEventKind.PhaseEdited, Guid.Parse(tab.PanelId!), dividerId, "Manual phase divider deleted");
    }

    public void LabelPhaseDivider(string tabId, string dividerId, string code, string label)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        EditablePhaseDivider divider = RequireDivider(tab, dividerId);
        ValidatePhaseLabel(code, label);
        double right = tab.PhaseDividers
            .Where(item => item.OriginalX > divider.OriginalX)
            .Select(item => item.OriginalX)
            .DefaultIfEmpty(tab.PixelWidth)
            .Min();
        PhaseEditResult edited = _phaseEditor.Apply(
            GetOverrides(tab),
            new RelabelPhaseCommand(
                Guid.NewGuid().ToString("D"),
                divider.DividerId,
                code.Trim(),
                MapPhaseType(code),
                label.Trim(),
                divider.OriginalX,
                right),
            PlotBounds(tab),
            CancellationToken.None);
        RequirePhaseSuccess(edited);
        _phaseOverrides[tab.TabId] = edited.Overrides;
        divider.Code = code.Trim();
        divider.Label = label.Trim();
        UpdateAllPointPhases(tab);
        SynchronizeProject(DomainEventKind.PhaseEdited, Guid.Parse(tab.PanelId!), dividerId, "Manual phase divider labeled");
    }

    public async Task<DomainResult<ProjectSnapshotReceipt>> AutosaveAsync(
        SnapshotTrigger trigger,
        string? tabId,
        string? entityId,
        CancellationToken cancellationToken)
    {
        if (_applicationPaths is null)
        {
            return DomainResult<ProjectSnapshotReceipt>.Failure(new DomainError(
                "AUTOSAVE_PATHS_UNAVAILABLE",
                DomainErrorSeverity.Warning,
                "Errors.ApplicationPathsUnavailable",
                "Application paths were not supplied to the manual runtime.",
                Recoverable: true,
                "restart_application"));
        }

        SynchronizeProject(DomainEventKind.PointEdited, panelId: null, entityId, "Manual workspace autosave");
        var snapshots = new ProjectSnapshotService(_applicationPaths.AutosaveRoot, _projectFileStore);
        PanelId? panelId = tabId is null ? null : PanelId.FromGuid(Guid.Parse(RequireTab(tabId).PanelId!));
        DomainResult<ProjectSnapshotReceipt> result = await snapshots.SaveEventSnapshotAsync(
            CurrentProject,
            trigger,
            DateTimeOffset.UtcNow,
            panelId,
            entityId,
            cancellationToken).ConfigureAwait(false);
        if (result.IsSuccess && result.Value is not null)
        {
            CurrentProject = result.Value.Snapshot;
        }

        return result;
    }

    public async Task<DomainResult<ProjectSnapshotReceipt>> TimerAutosaveAsync(
        DateTimeOffset occurredUtc,
        CancellationToken cancellationToken)
    {
        if (_applicationPaths is null)
        {
            return DomainResult<ProjectSnapshotReceipt>.Failure(new DomainError(
                "AUTOSAVE_PATHS_UNAVAILABLE",
                DomainErrorSeverity.Warning,
                "Errors.ApplicationPathsUnavailable",
                "Application paths were not supplied to the manual runtime.",
                Recoverable: true,
                "restart_application"));
        }

        var scheduler = new FiveMinuteAutosaveScheduler();
        AutosaveSchedule schedule = scheduler.CreateSchedule(CurrentProject, CurrentProject.ModifiedUtc);
        var snapshots = new ProjectSnapshotService(_applicationPaths.AutosaveRoot, _projectFileStore, scheduler);
        DomainResult<ProjectSnapshotReceipt> result = await snapshots.SaveTimerSnapshotAsync(
            CurrentProject,
            schedule,
            occurredUtc,
            cancellationToken).ConfigureAwait(false);
        if (result.IsSuccess && result.Value is not null)
        {
            CurrentProject = result.Value.Snapshot;
        }

        return result;
    }

    public Task<DomainResult<RecoveryDiscoveryReport>> DiscoverRecoveryAsync(CancellationToken cancellationToken)
    {
        if (_applicationPaths is null)
        {
            return Task.FromResult(DomainResult<RecoveryDiscoveryReport>.Failure(new DomainError(
                "RECOVERY_PATHS_UNAVAILABLE",
                DomainErrorSeverity.Warning,
                "Errors.ApplicationPathsUnavailable",
                "Application paths were not supplied to the manual runtime.",
                Recoverable: true,
                "restart_application")));
        }

        var recovery = new ProjectRecoveryService(_projectFileStore);
        return CurrentProjectPath is null
            ? recovery.DiscoverAsync(_applicationPaths.AutosaveRoot, CurrentProject, null, cancellationToken)
            : recovery.DiscoverForProjectPathAsync(_applicationPaths.AutosaveRoot, CurrentProjectPath, cancellationToken);
    }

    public async Task<DomainResult<ProjectSaveReceipt>> RecoverLatestToNewFileAsync(
        string destinationPath,
        CancellationToken cancellationToken)
    {
        if (_applicationPaths is null)
        {
            return DomainResult<ProjectSaveReceipt>.Failure(new DomainError(
                "RECOVERY_PATHS_UNAVAILABLE",
                DomainErrorSeverity.Warning,
                "Errors.ApplicationPathsUnavailable",
                "Application paths were not supplied to the manual runtime.",
                Recoverable: true,
                "restart_application"));
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(destinationPath);
        DomainResult<RecoveryDiscoveryReport> discovery = await DiscoverRecoveryAsync(cancellationToken)
            .ConfigureAwait(false);
        if (!discovery.IsSuccess || discovery.Value is null)
        {
            return DomainResult<ProjectSaveReceipt>.Failure(discovery.Errors);
        }

        RecoveryCandidate? candidate = discovery.Value.Candidates
            .FirstOrDefault(static item => item.Recommendation == RecoveryRecommendation.RestoreRecommended);
        if (candidate is null && discovery.Value.Candidates.Count > 0)
        {
            candidate = discovery.Value.Candidates[0];
        }
        if (candidate is null)
        {
            return DomainResult<ProjectSaveReceipt>.Failure(new DomainError(
                "RECOVERY_CANDIDATE_NOT_FOUND",
                DomainErrorSeverity.Warning,
                "Errors.RecoveryCandidateNotFound",
                "No autosave for the current project is available to recover.",
                Recoverable: true,
                "continue_editing"));
        }

        var recovery = new ProjectRecoveryService(_projectFileStore);
        return await recovery.RecoverToNewFileAsync(
            candidate.AutosavePath,
            destinationPath,
            cancellationToken).ConfigureAwait(false);
    }

    public async Task<ExportResult> ExportAsync(
        string tabId,
        string outputDirectory,
        CancellationToken cancellationToken)
    {
        WorkspaceTabViewModel tab = RequireTab(tabId);
        if (tab.Calibration is null)
        {
            throw new InvalidOperationException("Manual three-anchor calibration is required before export.");
        }

        SynchronizeProject(DomainEventKind.ExportSettingsChanged, Guid.Parse(tab.PanelId!), entityId: null, "Manual CSV export");
        PanelRecord panel = CurrentProject.Panels.Single(item => item.PanelId.Value == Guid.Parse(tab.PanelId!));
        SeriesRecord[] interventions = panel.Series
            .Where(static series => series.SemanticRole == SemanticRole.Intervention)
            .ToArray();
        if (interventions.Length == 0)
        {
            throw new InvalidOperationException("Create at least one intervention series before export.");
        }

        var request = new ExportRequest(
            Guid.NewGuid(),
            CurrentProject.ProjectId.Value,
            panel.PanelId.Value,
            outputDirectory,
            panel.Participant ?? Path.GetFileNameWithoutExtension(tab.DisplayName),
            ExportMode.ObservationOrder,
            ExportAuditMode.ExtendedCsvAndJson,
            ExportOperation.WriteFiles,
            new ExportCalibration(
                ExportCalibrationStatus.Valid,
                hasYCalibration: true,
                hasPrintedSessionCalibration: false,
                hasAbsoluteSessionOrigin: true,
                firstObservedSession: null,
                panel.Calibration!.Confidence),
            new ExportSessionOriginPolicy(
                RequireFirstObservedSessionOne: false,
                InvalidSessionOriginBehavior.Block),
            panel.Phases.Select(ToExportPhase),
            panel.Series.Select(ToExportSeries),
            panel.Points.Select(ToExportPoint),
            interventions.Select(series => new ExportSeriesRelation(
                series.SeriesId.Value,
                series.SharedBaselineSeriesId?.Value,
                series.ApplicableProbeSeriesIds.Select(static id => id.Value))),
            interventions.Select(static series => series.SeriesId.Value));
        return await _exportService.ExportAsync(request, cancellationToken).ConfigureAwait(false);
    }

    public Task RunStageAsync(WorkflowStage stage, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        AutomaticStageStatus? unavailable = stage switch
        {
            WorkflowStage.Prepare => AutomaticStages.First(status => status.Stage == "enhancement"),
            WorkflowStage.Detect => AutomaticStages.First(status => status.Stage == "markers"),
            _ => null,
        };

        return unavailable is null
            ? Task.CompletedTask
            : Task.FromException(new InvalidOperationException(unavailable.Explanation));
    }

    private static string GetApplicationVersion()
    {
        Version version = typeof(ManualPreviewWorkspaceService).Assembly.GetName().Version ?? new Version(0, 0, 19);
        return $"{Math.Max(0, version.Major)}.{Math.Max(0, version.Minor)}.{Math.Max(0, version.Build)}";
    }

    private static WorkspaceTabViewModel CreateEmptyImageTab(
        PanelId panelId,
        SourceReference source,
        ImportedImage image)
    {
        var points = new ObservableCollection<AppGraphPoint>();
        var series = new ObservableCollection<SeriesCardViewModel>();
        var dividers = new ObservableCollection<EditablePhaseDivider>();
        return new WorkspaceTabViewModel(
            panelId.Value.ToString("D"),
            source.DisplayName,
            points,
            series,
            CreateBitmap(image.OriginalBytes),
            enhancedImageSource: null,
            phaseOverlayContent: null,
            panelId.Value.ToString("D"),
            source.SourceId.Value.ToString("D"),
            source.LocalPath,
            source.Sha256,
            image.Metadata.Width,
            image.Metadata.Height,
            calibration: null,
            dividers);
    }

    private static WorkspaceTabViewModel CreateTabFromProject(
        PanelRecord panel,
        SourceReference source,
        ImportedImage image)
    {
        var points = new ObservableCollection<AppGraphPoint>(panel.Points.Select(point => new AppGraphPoint(
            point.PointId.Value.ToString("D"),
            point.SeriesId?.Value.ToString("D") ?? string.Empty,
            point.OriginalPixel.X,
            point.OriginalPixel.Y,
            point.GraphX ?? 0,
            point.GraphY ?? 0,
            panel.Phases.FirstOrDefault(phase => phase.PhaseId == point.PhaseId)?.Code ?? "unknown",
            point.PhaseId?.Value.ToString("D"),
            point.ObservationIndex)));
        var series = new ObservableCollection<SeriesCardViewModel>(panel.Series.Select(item =>
            new SeriesCardViewModel(
                item.SeriesId.Value.ToString("D"),
                item.Symbol,
                $"{item.Fill} {item.Shape}",
                item.DisplayName,
                item.Confidence,
                points,
                item.Shape,
                item.Fill,
                item.SemanticRole)));
        var dividers = new ObservableCollection<EditablePhaseDivider>(panel.Phases
            .Where(static phase => phase.Order > 1 && phase.BoundaryLeftId is not null)
            .Select(phase => new EditablePhaseDivider(
                phase.BoundaryLeftId!.Value.Value.ToString("D"),
                phase.ScreenXMin,
                phase.Code,
                phase.LabelText ?? phase.Code)));
        ManualCalibrationState? calibration = FromDomainCalibration(panel.Calibration);
        return new WorkspaceTabViewModel(
            panel.PanelId.Value.ToString("D"),
            panel.DisplayName,
            points,
            series,
            CreateBitmap(image.OriginalBytes),
            enhancedImageSource: null,
            phaseOverlayContent: null,
            panel.PanelId.Value.ToString("D"),
            panel.SourceId.Value.ToString("D"),
            source.LocalPath,
            source.Sha256,
            image.Metadata.Width,
            image.Metadata.Height,
            calibration,
            dividers);
    }

    private static BitmapImage CreateBitmap(ImmutableImageBytes originalBytes)
    {
        using Stream stream = originalBytes.OpenRead();
        var bitmap = new BitmapImage();
        bitmap.BeginInit();
        bitmap.CacheOption = BitmapCacheOption.OnLoad;
        bitmap.StreamSource = stream;
        bitmap.EndInit();
        bitmap.Freeze();
        return bitmap;
    }

    private static ManualCalibrationState? FromDomainCalibration(CalibrationRecord? calibration)
    {
        if (calibration is null)
        {
            return null;
        }

        DomainCalibrationAnchor? y0 = calibration.Anchors.FirstOrDefault(
            static anchor => anchor.Kind == DomainCalibrationAnchorKind.Session1Y0);
        DomainCalibrationAnchor? yMax = calibration.Anchors.FirstOrDefault(
            static anchor => anchor.Kind == DomainCalibrationAnchorKind.Session1Ymax);
        DomainCalibrationAnchor? xMax = calibration.Anchors.FirstOrDefault(
            static anchor => anchor.Kind == DomainCalibrationAnchorKind.SessionmaxY0);
        if (y0 is null || yMax is null || xMax is null)
        {
            return null;
        }

        double xMaximum = xMax.Graph.X;
        double yMaximum = yMax.Graph.Y;
        var xTransform = FitTransform(y0.Screen.X, 1, xMax.Screen.X, xMaximum);
        var yTransform = FitTransform(y0.Screen.Y, 0, yMax.Screen.Y, yMaximum);
        return new ManualCalibrationState(
            new AxisPixelPoint(y0.Screen.X, y0.Screen.Y),
            new AxisPixelPoint(yMax.Screen.X, yMax.Screen.Y),
            new AxisPixelPoint(xMax.Screen.X, xMax.Screen.Y),
            yMaximum,
            xMaximum,
            xTransform,
            yTransform,
            calibration.Confidence);
    }

    private void SynchronizeProject(
        DomainEventKind eventKind,
        Guid? panelId,
        string? entityId,
        string note)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        var auditEvent = new AuditEvent(
            AuditEventId.New(),
            now,
            eventKind,
            panelId is null ? null : PanelId.FromGuid(panelId.Value),
            entityId,
            note,
            Details: null);
        CurrentProject = CurrentProject with
        {
            ModifiedUtc = now,
            Panels = _tabs.Select(ToPanelRecord).ToArray(),
            Audit = CurrentProject.Audit with
            {
                Events = CurrentProject.Audit.Events.Append(auditEvent).ToArray(),
            },
        };
    }

    private PanelRecord ToPanelRecord(WorkspaceTabViewModel tab)
    {
        PanelId panelId = PanelId.FromGuid(Guid.Parse(tab.PanelId!));
        SourceId sourceId = SourceId.FromGuid(Guid.Parse(tab.SourceId!));
        PhaseRecord[] regionPhases = BuildRegionPhases(tab);
        PhaseRecord[] semanticProbePhases = BuildSemanticProbePhases(tab, regionPhases.Length);
        PhaseRecord[] phases = [.. regionPhases, .. semanticProbePhases];
        Dictionary<SeriesId, SeriesRecord> existingSeries = CurrentProject.Panels
            .FirstOrDefault(panel => panel.PanelId == panelId)?
            .Series.ToDictionary(static series => series.SeriesId)
            ?? new Dictionary<SeriesId, SeriesRecord>();
        Dictionary<string, PhaseRecord> pointPhases = tab.Points.ToDictionary(
            static point => point.PointId,
            point => ResolvePointPhase(tab, point, regionPhases, semanticProbePhases));
        HashSet<SeriesId> validBaselineIds = tab.SeriesCards
            .Where(static item => item.SemanticRole == SemanticRole.Baseline)
            .Select(item => SeriesId.FromGuid(Guid.Parse(item.SeriesId)))
            .ToHashSet();
        HashSet<SeriesId> validProbeIds = tab.SeriesCards
            .Where(static item => item.SemanticRole is SemanticRole.Maintenance or SemanticRole.Generalization)
            .Select(item => SeriesId.FromGuid(Guid.Parse(item.SeriesId)))
            .ToHashSet();
        SeriesRecord[] series = tab.SeriesCards.Select(item =>
        {
            SeriesId seriesId = SeriesId.FromGuid(Guid.Parse(item.SeriesId));
            _ = existingSeries.TryGetValue(seriesId, out SeriesRecord? prior);
            return new SeriesRecord(
                seriesId,
                item.Symbol,
                item.Shape,
                item.Fill,
                item.Label,
                item.SemanticRole,
                prior?.LegendText,
                tab.Points.Where(point => point.SeriesId == item.SeriesId)
                    .Select(point => PointId.FromGuid(Guid.Parse(point.PointId)))
                    .ToArray(),
                item.Confidence,
                item.SemanticRole == SemanticRole.Intervention &&
                    prior?.SharedBaselineSeriesId is { } baselineId &&
                    validBaselineIds.Contains(baselineId)
                        ? baselineId
                        : null,
                item.SemanticRole == SemanticRole.Intervention
                    ? prior?.ApplicableProbeSeriesIds.Where(validProbeIds.Contains).ToArray() ?? []
                    : [],
                UserConfirmedName: true);
        }).ToArray();
        PointRecord[] points = tab.Points.Select(point =>
        {
            PhaseRecord phase = pointPhases[point.PointId];
            point.PhaseId = phase.PhaseId.Value.ToString("D");
            point.PhaseCode = phase.Code;
            ManualPointXState xState = GetPointXState(tab, point);
            return new PointRecord(
                PointId.FromGuid(Guid.Parse(point.PointId)),
                MarkerId: null,
                SeriesId.FromGuid(Guid.Parse(point.SeriesId)),
                phase.PhaseId,
                new DomainPixelPoint(point.PixelX, point.PixelY),
                xState.HasGraphX ? point.GraphX : null,
                tab.Calibration is null ? null : point.GraphY,
                point.ObservationIndex,
                xState.PrintedXValue,
                xState.EstimatedXValue,
                xState.Source,
                xState.Confidence,
                tab.Calibration?.Confidence ?? 0,
                1,
                "manual",
                ModelVersion: null,
                ReviewStatus.Corrected,
                ModificationHistory: GetPointModificationHistory(point.PointId));
        }).ToArray();
        return new PanelRecord(
            panelId,
            sourceId,
            PageNumber: null,
            tab.DisplayName,
            Participant: null,
            new CropRectangle(0, 0, tab.PixelWidth, tab.PixelHeight),
            Transforms: [],
            Enhancement: _enhancementByTab.TryGetValue(tab.TabId, out JsonElement enhancement)
                ? enhancement.Clone()
                : null,
            ToDomainCalibration(tab.Calibration),
            OcrRegions: [],
            Markers: [],
            series,
            points,
            phases,
            new ExportSettingsRecord(
                "observation_order",
                IncludeAuditSidecar: true,
                series.Where(static item => item.SemanticRole == SemanticRole.Intervention)
                    .Select(static item => item.SeriesId)
                    .ToArray()),
            Validation: null);
    }

    private static CalibrationRecord? ToDomainCalibration(ManualCalibrationState? calibration)
    {
        if (calibration is null)
        {
            return null;
        }

        return new CalibrationRecord(
            CalibrationId.New(),
            DomainCalibrationStatus.Valid,
            [
                new DomainCalibrationAnchor(
                    DomainCalibrationAnchorKind.Session1Y0,
                    new DomainPixelPoint(calibration.Session1Y0.X, calibration.Session1Y0.Y),
                    new DomainGraphPoint(1, 0),
                    calibration.Confidence,
                    EvidenceRegionId: null),
                new DomainCalibrationAnchor(
                    DomainCalibrationAnchorKind.Session1Ymax,
                    new DomainPixelPoint(calibration.Session1YMaximum.X, calibration.Session1YMaximum.Y),
                    new DomainGraphPoint(1, calibration.YMaximum),
                    calibration.Confidence,
                    EvidenceRegionId: null),
                new DomainCalibrationAnchor(
                    DomainCalibrationAnchorKind.SessionmaxY0,
                    new DomainPixelPoint(calibration.SessionMaximumY0.X, calibration.SessionMaximumY0.Y),
                    new DomainGraphPoint(calibration.XMaximum, 0),
                    calibration.Confidence,
                    EvidenceRegionId: null),
            ],
            new SessionLatticeRecord(
                calibration.Session1Y0.X,
                (calibration.SessionMaximumY0.X - calibration.Session1Y0.X) / (calibration.XMaximum - 1),
                PrintedMin: null,
                PrintedMax: null,
                calibration.Confidence,
                "manual_three_anchor"),
            UserConfirmed: true,
            calibration.Confidence,
            Reasons: []);
    }

    private void UpdateEnhancementProvenance(
        WorkspaceTabViewModel tab,
        EnhancementEnvelope envelope)
    {
        _enhancementByTab[tab.TabId] = JsonSerializer.SerializeToElement(new
        {
            selected_preview = tab.EnhancementPreviewMode.ToString().ToLowerInvariant(),
            original_immutable = true,
            enhancement = envelope,
        });
    }

    private static PhaseRecord[] BuildRegionPhases(WorkspaceTabViewModel tab)
    {
        EditablePhaseDivider[] dividers = tab.PhaseDividers.OrderBy(static item => item.OriginalX).ToArray();
        var phases = new List<PhaseRecord>(dividers.Length + 1);
        double left = 0;
        for (int index = 0; index <= dividers.Length; index++)
        {
            EditablePhaseDivider? preceding = index == 0 ? null : dividers[index - 1];
            EditablePhaseDivider? following = index == dividers.Length ? null : dividers[index];
            double right = following?.OriginalX ?? tab.PixelWidth;
            string code = preceding?.Code ?? "a";
            string label = preceding?.Label ?? "Baseline";
            var phaseId = PhaseId.FromGuid(Guid.Parse(preceding?.DividerId ?? tab.PanelId!));
            phases.Add(new PhaseRecord(
                phaseId,
                index + 1,
                code,
                MapDomainPhaseType(code),
                label,
                left,
                right,
                preceding is null ? null : PhaseId.FromGuid(Guid.Parse(preceding.DividerId)),
                following is null ? null : PhaseId.FromGuid(Guid.Parse(following.DividerId)),
                1,
                PhaseSource.Manual,
                UserConfirmed: true));
            left = right;
        }

        return phases.ToArray();
    }

    private static PhaseRecord[] BuildSemanticProbePhases(WorkspaceTabViewModel tab, int firstOrder)
    {
        return tab.SeriesCards
            .Where(series => ProbePhaseCode(series.SemanticRole) is not null)
            .Where(series => tab.Points.Any(point => string.Equals(
                point.SeriesId,
                series.SeriesId,
                StringComparison.Ordinal)))
            .Select((series, index) =>
            {
                string code = ProbePhaseCode(series.SemanticRole)!;
                return new PhaseRecord(
                    PhaseId.FromGuid(Guid.Parse(series.SeriesId)),
                    firstOrder + index + 1,
                    code,
                    MapDomainPhaseType(code),
                    series.Label,
                    0,
                    tab.PixelWidth,
                    BoundaryLeftId: null,
                    BoundaryRightId: null,
                    Confidence: 1,
                    PhaseSource.Manual,
                    UserConfirmed: true);
            })
            .ToArray();
    }

    private static PhaseRecord ResolvePointPhase(
        WorkspaceTabViewModel tab,
        AppGraphPoint point,
        IReadOnlyList<PhaseRecord> regionPhases,
        IReadOnlyList<PhaseRecord> semanticProbePhases)
    {
        SeriesCardViewModel series = RequireSeries(tab, point.SeriesId);
        if (ProbePhaseCode(series.SemanticRole) is not null)
        {
            PhaseId semanticPhaseId = PhaseId.FromGuid(Guid.Parse(series.SeriesId));
            return semanticProbePhases.Single(phase => phase.PhaseId == semanticPhaseId);
        }

        return regionPhases.Last(phase =>
            point.PixelX >= phase.ScreenXMin && point.PixelX <= phase.ScreenXMax);
    }

    private static void UpdatePointCoordinates(WorkspaceTabViewModel tab, AppGraphPoint point)
    {
        if (tab.Calibration is { } calibration)
        {
            point.GraphX = calibration.XTransform.PixelToGraph(point.PixelX);
            point.GraphY = calibration.YTransform.PixelToGraph(point.PixelY);
        }

        SeriesCardViewModel series = RequireSeries(tab, point.SeriesId);
        string? probePhaseCode = ProbePhaseCode(series.SemanticRole);
        if (probePhaseCode is not null)
        {
            // A confirmed probe role is point semantics, not a horizontal phase boundary.
            point.PhaseId = series.SeriesId;
            point.PhaseCode = probePhaseCode;
            return;
        }

        EditablePhaseDivider? preceding = tab.PhaseDividers
            .Where(divider => divider.OriginalX <= point.PixelX)
            .OrderByDescending(static divider => divider.OriginalX)
            .FirstOrDefault();
        point.PhaseId = preceding?.DividerId ?? tab.PanelId;
        point.PhaseCode = preceding?.Code ?? "a";
    }

    private static string? ProbePhaseCode(SemanticRole semanticRole) => semanticRole switch
    {
        SemanticRole.Maintenance => "m",
        SemanticRole.Generalization => "g",
        _ => null,
    };

    private static void UpdateAllPointPhases(WorkspaceTabViewModel tab)
    {
        foreach (AppGraphPoint point in tab.Points)
        {
            UpdatePointCoordinates(tab, point);
        }
    }

    private static void ReindexAllSeries(WorkspaceTabViewModel tab)
    {
        foreach (SeriesCardViewModel series in tab.SeriesCards)
        {
            ReindexSeries(tab, series.SeriesId);
        }
    }

    private static void ReindexSeries(WorkspaceTabViewModel tab, string seriesId)
    {
        AppGraphPoint[] ordered = tab.Points
            .Where(point => string.Equals(point.SeriesId, seriesId, StringComparison.Ordinal))
            .OrderBy(static point => point.PixelX)
            .ThenBy(static point => point.PixelY)
            .ThenBy(static point => point.PointId, StringComparer.Ordinal)
            .ToArray();
        for (int index = 0; index < ordered.Length; index++)
        {
            ordered[index].ObservationIndex = index + 1;
        }
    }

    private static DomainGraphPoint? GetPreviousGraph(WorkspaceTabViewModel tab, AppGraphPoint point) =>
        tab.Calibration is null ? null : new DomainGraphPoint(point.GraphX, point.GraphY);

    private void AppendPointModification(
        AppGraphPoint point,
        DomainPixelPoint previousPixel,
        DomainGraphPoint? previousGraph,
        string reason)
    {
        if (!_pointModificationHistories.TryGetValue(point.PointId, out List<PointModification>? history))
        {
            history = [];
            _pointModificationHistories[point.PointId] = history;
        }

        history.Add(new PointModification(
            AuditEventId.New(),
            DateTimeOffset.UtcNow,
            previousPixel,
            previousGraph,
            reason));
    }

    private PointModification[] GetPointModificationHistory(string pointId) =>
        _pointModificationHistories.TryGetValue(pointId, out List<PointModification>? history)
            ? history.ToArray()
            : [];

    private void MarkPointXAsManualEstimate(WorkspaceTabViewModel tab, AppGraphPoint point)
    {
        _pointXStates[point.PointId] = tab.Calibration is null
            ? new ManualPointXState(
                PrintedXValue: null,
                EstimatedXValue: null,
                Source: PointXSource.Unknown,
                Confidence: 0,
                HasGraphX: false)
            : new ManualPointXState(
                PrintedXValue: null,
                EstimatedXValue: point.GraphX,
                Source: PointXSource.Estimated,
                Confidence: 0,
                HasGraphX: true);
    }

    private ManualPointXState GetPointXState(WorkspaceTabViewModel tab, AppGraphPoint point)
    {
        if (_pointXStates.TryGetValue(point.PointId, out ManualPointXState? state))
        {
            return state;
        }

        return tab.Calibration is null
            ? new ManualPointXState(null, null, PointXSource.Unknown, 0, HasGraphX: false)
            : new ManualPointXState(null, point.GraphX, PointXSource.Estimated, 0, HasGraphX: true);
    }

    private static ExportPhase ToExportPhase(PhaseRecord phase) => new(
        phase.PhaseId.Value,
        phase.Order,
        phase.Code,
        phase.NormalizedType switch
        {
            DomainPhaseNormalizedType.Baseline => ExportPhaseType.Baseline,
            DomainPhaseNormalizedType.Intervention => ExportPhaseType.Intervention,
            DomainPhaseNormalizedType.Maintenance => ExportPhaseType.Maintenance,
            DomainPhaseNormalizedType.Generalization => ExportPhaseType.Generalization,
            _ => ExportPhaseType.Unknown,
        },
        phase.LabelText,
        phase.ScreenXMin,
        phase.ScreenXMax,
        phase.Confidence);

    private static ExportSeries ToExportSeries(SeriesRecord series) => new(
        series.SeriesId.Value,
        series.Symbol,
        series.DisplayName,
        series.SemanticRole switch
        {
            SemanticRole.Baseline => ExportSeriesRole.Baseline,
            SemanticRole.Intervention => ExportSeriesRole.Intervention,
            SemanticRole.Maintenance => ExportSeriesRole.Maintenance,
            SemanticRole.Generalization => ExportSeriesRole.Generalization,
            _ => ExportSeriesRole.Unknown,
        },
        series.PointIds.Select(static id => id.Value),
        series.Confidence,
        series.LegendText);

    private static ExportPoint ToExportPoint(PointRecord point) => new(
        point.PointId.Value,
        point.MarkerId?.Value,
        point.SeriesId?.Value,
        point.PhaseId?.Value,
        new ExportPixelPoint(point.OriginalPixel.X, point.OriginalPixel.Y),
        point.GraphX,
        point.GraphY,
        point.ObservationIndex,
        point.PrintedXValue,
        point.EstimatedXValue,
        point.XSource switch
        {
            PointXSource.Printed => ExportXValueSource.Printed,
            PointXSource.Estimated => ExportXValueSource.Estimated,
            PointXSource.ObservationOrder => ExportXValueSource.ObservationOrder,
            _ => ExportXValueSource.Unknown,
        },
        point.XConfidence,
        point.YConfidence,
        point.PointConfidence,
        point.ReviewStatus switch
        {
            ReviewStatus.Accepted => ExportReviewStatus.Accepted,
            ReviewStatus.Corrected => ExportReviewStatus.Corrected,
            ReviewStatus.Rejected => ExportReviewStatus.Rejected,
            _ => ExportReviewStatus.Unreviewed,
        },
        point.SourceStage,
        point.ModelVersion);

    private static LinearAxisTransform FitTransform(double pixel1, double graph1, double pixel2, double graph2)
    {
        double slope = (graph2 - graph1) / (pixel2 - pixel1);
        return new LinearAxisTransform(slope, graph1 - (slope * pixel1));
    }

    private static void ValidateCalibrationRequest(ManualCalibrationRequest request)
    {
        if (!request.Session1Y0.IsFinite || !request.Session1YMaximum.IsFinite ||
            !request.SessionMaximumY0.IsFinite ||
            request.YMaximum <= 0 || request.XMaximum <= 1 ||
            Math.Abs(request.Session1Y0.X - request.Session1YMaximum.X) > 2 ||
            Math.Abs(request.Session1Y0.Y - request.SessionMaximumY0.Y) > 2)
        {
            throw new ArgumentException("Manual calibration requires aligned (1,0), (1,yMax), and (xMax,0) anchors with positive maxima.", nameof(request));
        }
    }

    private static void EnsureFinitePoint(double x, double y)
    {
        if (!double.IsFinite(x) || !double.IsFinite(y))
        {
            throw new ArgumentOutOfRangeException(nameof(x), "Point coordinates must be finite original pixels.");
        }
    }

    private static void ValidatePhaseLabel(string code, string label)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(code);
        ArgumentException.ThrowIfNullOrWhiteSpace(label);
    }

    private static PhaseNormalizedType MapPhaseType(string code) => code.Trim().ToLowerInvariant() switch
    {
        "a" or "a1" or "a2" => PhaseNormalizedType.Baseline,
        "b" or "b1" or "b2" => PhaseNormalizedType.Intervention,
        "m" or "m1" or "m2" => PhaseNormalizedType.Maintenance,
        "g" or "g1" or "g2" => PhaseNormalizedType.Generalization,
        _ => PhaseNormalizedType.Unknown,
    };

    private static DomainPhaseNormalizedType MapDomainPhaseType(string code) => MapPhaseType(code) switch
    {
        PhaseNormalizedType.Baseline => DomainPhaseNormalizedType.Baseline,
        PhaseNormalizedType.Intervention => DomainPhaseNormalizedType.Intervention,
        PhaseNormalizedType.Maintenance => DomainPhaseNormalizedType.Maintenance,
        PhaseNormalizedType.Generalization => DomainPhaseNormalizedType.Generalization,
        _ => DomainPhaseNormalizedType.Unknown,
    };

    private static PhaseRectangle PlotBounds(WorkspaceTabViewModel tab) =>
        new(0, 0, tab.PixelWidth, tab.PixelHeight);

    private static void RequirePhaseSuccess(PhaseEditResult result)
    {
        if (!result.Succeeded)
        {
            throw new InvalidOperationException(result.Failure!.TechnicalMessage);
        }
    }

    private static void SortDividers(WorkspaceTabViewModel tab)
    {
        EditablePhaseDivider[] ordered = tab.PhaseDividers.OrderBy(static divider => divider.OriginalX).ToArray();
        tab.PhaseDividers.Clear();
        foreach (EditablePhaseDivider divider in ordered)
        {
            tab.PhaseDividers.Add(divider);
        }
    }

    private PhaseManualOverrides GetOverrides(WorkspaceTabViewModel tab) =>
        _phaseOverrides.TryGetValue(tab.TabId, out PhaseManualOverrides? overrides)
            ? overrides
            : new PhaseManualOverrides();

    private static PhaseManualOverrides CreatePhaseOverrides(WorkspaceTabViewModel tab) =>
        new(tab.PhaseDividers.Select(divider => new PhaseManualDivider(
            divider.DividerId,
            divider.OriginalX,
            PhaseDividerStyle.Dashed)));

    private WorkspaceTabViewModel RequireTab(string tabId) =>
        _tabs.SingleOrDefault(tab => string.Equals(tab.TabId, tabId, StringComparison.Ordinal))
        ?? throw new KeyNotFoundException($"Workspace tab '{tabId}' does not exist.");

    private static SeriesCardViewModel RequireSeries(WorkspaceTabViewModel tab, string seriesId) =>
        tab.SeriesCards.SingleOrDefault(series => string.Equals(series.SeriesId, seriesId, StringComparison.Ordinal))
        ?? throw new KeyNotFoundException($"Series '{seriesId}' does not exist.");

    private static AppGraphPoint RequirePoint(WorkspaceTabViewModel tab, string pointId) =>
        tab.Points.SingleOrDefault(point => string.Equals(point.PointId, pointId, StringComparison.Ordinal))
        ?? throw new KeyNotFoundException($"Point '{pointId}' does not exist.");

    private static EditablePhaseDivider RequireDivider(WorkspaceTabViewModel tab, string dividerId) =>
        tab.PhaseDividers.SingleOrDefault(divider => string.Equals(divider.DividerId, dividerId, StringComparison.Ordinal))
        ?? throw new KeyNotFoundException($"Phase divider '{dividerId}' does not exist.");

    private static string FormatErrors(IEnumerable<DomainError> errors) =>
        string.Join(" | ", errors.Select(static error => $"{error.Code}: {error.TechnicalMessage}"));

    private sealed record ManualPointXState(
        double? PrintedXValue,
        double? EstimatedXValue,
        PointXSource Source,
        double Confidence,
        bool HasGraphX);
}
