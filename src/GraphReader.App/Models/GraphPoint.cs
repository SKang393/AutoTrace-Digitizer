// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace GraphReader.App.Models;

public sealed class GraphPoint : INotifyPropertyChanged
{
    private string _seriesId;
    private double _pixelX;
    private double _pixelY;
    private double _graphX;
    private double _graphY;
    private string _phaseCode;
    private string? _phaseId;
    private int _observationIndex;

    public GraphPoint(
        string pointId,
        string seriesId,
        double pixelX,
        double pixelY,
        double graphX,
        double graphY,
        string phaseCode,
        string? phaseId = null,
        int observationIndex = 1)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pointId);
        ArgumentException.ThrowIfNullOrWhiteSpace(seriesId);
        ArgumentException.ThrowIfNullOrWhiteSpace(phaseCode);

        PointId = pointId;
        _seriesId = seriesId;
        _pixelX = pixelX;
        _pixelY = pixelY;
        _graphX = graphX;
        _graphY = graphY;
        _phaseCode = phaseCode;
        _phaseId = phaseId;
        ObservationIndex = observationIndex;
    }

    public string PointId { get; }

    public string SeriesId
    {
        get => _seriesId;
        internal set => SetProperty(ref _seriesId, value);
    }

    public double PixelX
    {
        get => _pixelX;
        internal set => SetProperty(ref _pixelX, value);
    }

    public double PixelY
    {
        get => _pixelY;
        internal set => SetProperty(ref _pixelY, value);
    }

    public double GraphX
    {
        get => _graphX;
        internal set => SetProperty(ref _graphX, value);
    }

    public double GraphY
    {
        get => _graphY;
        internal set => SetProperty(ref _graphY, value);
    }

    public string PhaseCode
    {
        get => _phaseCode;
        internal set => SetProperty(ref _phaseCode, value);
    }

    public string? PhaseId
    {
        get => _phaseId;
        internal set => SetProperty(ref _phaseId, value);
    }

    public int ObservationIndex
    {
        get => _observationIndex;
        internal set
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(value, 1);
            SetProperty(ref _observationIndex, value);
        }
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private void SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return;
        }

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
