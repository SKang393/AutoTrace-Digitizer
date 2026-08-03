// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using GraphReader.App.Models;

namespace GraphReader.App.ViewModels;

public enum WorkflowStage
{
    Import,
    Prepare,
    Detect,
    Review,
    Export,
}

public sealed class MagnifierViewModel : ObservableObject
{
    private bool _isEnhanced;
    private bool _isCrosshairVisible = true;
    private Point _crosshairPosition = new(0.5, 0.5);
    private Point _pixelPosition = new(282, 128);
    private Point? _graphPosition = new Point(4, 58);
    private string? _nearestDetectionName = "Filled circle";
    private double? _nearestDetectionConfidence = 0.96;
    private double _zoomLevel = 3;

    public bool IsEnhanced
    {
        get => _isEnhanced;
        set => SetProperty(ref _isEnhanced, value);
    }

    public bool IsCrosshairVisible
    {
        get => _isCrosshairVisible;
        set => SetProperty(ref _isCrosshairVisible, value);
    }

    public Point CrosshairPosition
    {
        get => _crosshairPosition;
        set => SetProperty(ref _crosshairPosition, value);
    }

    public Point PixelPosition
    {
        get => _pixelPosition;
        set => SetProperty(ref _pixelPosition, value);
    }

    public Point? GraphPosition
    {
        get => _graphPosition;
        set => SetProperty(ref _graphPosition, value);
    }

    public string? NearestDetectionName
    {
        get => _nearestDetectionName;
        set => SetProperty(ref _nearestDetectionName, value);
    }

    public double? NearestDetectionConfidence
    {
        get => _nearestDetectionConfidence;
        set => SetProperty(ref _nearestDetectionConfidence, value);
    }

    public double ZoomLevel
    {
        get => _zoomLevel;
        set
        {
            if (!double.IsFinite(value) || value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value));
            }

            SetProperty(ref _zoomLevel, value);
        }
    }
}

public sealed class SeriesCardViewModel : ObservableObject
{
    private readonly ObservableCollection<GraphPoint> _points;
    private bool _isVisible = true;

    public SeriesCardViewModel(
        string seriesId,
        string symbol,
        string accessibleName,
        string label,
        double confidence,
        ObservableCollection<GraphPoint> points)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(seriesId);
        ArgumentException.ThrowIfNullOrWhiteSpace(symbol);
        ArgumentException.ThrowIfNullOrWhiteSpace(accessibleName);
        ArgumentException.ThrowIfNullOrWhiteSpace(label);
        if (!double.IsFinite(confidence) || confidence is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(confidence));
        }

        SeriesId = seriesId;
        Symbol = symbol;
        AccessibleName = accessibleName;
        Label = label;
        Confidence = confidence;
        _points = points ?? throw new ArgumentNullException(nameof(points));
        SelectCommand = new RelayCommand(_ => { });
    }

    public string SeriesId { get; }

    public string Symbol { get; }

    public string AccessibleName { get; }

    public string Label { get; }

    public int Count => _points.Count(point => point.SeriesId == SeriesId);

    public double Confidence { get; }

    public bool IsVisible
    {
        get => _isVisible;
        set => SetProperty(ref _isVisible, value);
    }

    public ICommand SelectCommand { get; }

    internal void NotifyCountChanged() => OnPropertyChanged(nameof(Count));
}

public sealed class WorkspaceTabViewModel : ObservableObject
{
    private readonly object? _overlayContent;
    private double _zoomLevel = 1;

    public WorkspaceTabViewModel(
        string tabId,
        string displayName,
        ObservableCollection<GraphPoint> points,
        ObservableCollection<SeriesCardViewModel> seriesCards,
        ImageSource? imageSource,
        ImageSource? enhancedImageSource,
        object? phaseOverlayContent)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tabId);
        ArgumentException.ThrowIfNullOrWhiteSpace(displayName);
        TabId = tabId;
        DisplayName = displayName;
        Points = points ?? throw new ArgumentNullException(nameof(points));
        SeriesCards = seriesCards ?? throw new ArgumentNullException(nameof(seriesCards));
        ImageSource = imageSource;
        EnhancedImageSource = enhancedImageSource;
        _overlayContent = null;
        PhaseOverlayContent = phaseOverlayContent;
    }

    public string TabId { get; }

    public string DisplayName { get; }

    public ObservableCollection<GraphPoint> Points { get; }

    public ObservableCollection<SeriesCardViewModel> SeriesCards { get; }

    public ImageSource? ImageSource { get; }

    public ImageSource? EnhancedImageSource { get; }

    public object? OverlayContent => _overlayContent;

    public object? PhaseOverlayContent { get; }

    public double ZoomLevel
    {
        get => _zoomLevel;
        set
        {
            if (!double.IsFinite(value) || value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value));
            }

            SetProperty(ref _zoomLevel, value);
        }
    }
}

public sealed record MovePointRequest(string PointId, double PixelX, double PixelY);

public sealed record SeriesPairRequest(string SourceSeriesId, string TargetSeriesId);

public sealed record SplitSeriesRequest(string SourceSeriesId, IReadOnlyCollection<string> PointIds);

public sealed record ReassignPointRequest(string PointId, string TargetSeriesId);
