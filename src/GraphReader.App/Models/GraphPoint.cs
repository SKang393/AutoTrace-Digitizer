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

    public GraphPoint(
        string pointId,
        string seriesId,
        double pixelX,
        double pixelY,
        double graphX,
        double graphY,
        string phaseCode)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pointId);
        ArgumentException.ThrowIfNullOrWhiteSpace(seriesId);
        ArgumentException.ThrowIfNullOrWhiteSpace(phaseCode);

        PointId = pointId;
        _seriesId = seriesId;
        _pixelX = pixelX;
        _pixelY = pixelY;
        GraphX = graphX;
        GraphY = graphY;
        PhaseCode = phaseCode;
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

    public double GraphX { get; }

    public double GraphY { get; }

    public string PhaseCode { get; }

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
