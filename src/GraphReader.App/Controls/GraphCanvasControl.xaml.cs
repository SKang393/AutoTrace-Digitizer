// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace GraphReader.App.Controls;

public partial class GraphCanvasControl : UserControl
{
    public static readonly DependencyProperty ImageSourceProperty = DependencyProperty.Register(
        nameof(ImageSource),
        typeof(ImageSource),
        typeof(GraphCanvasControl),
        new PropertyMetadata(null, OnPresentationChanged));

    public static readonly DependencyProperty OverlayContentProperty = DependencyProperty.Register(
        nameof(OverlayContent),
        typeof(object),
        typeof(GraphCanvasControl));

    public static readonly DependencyProperty PhaseOverlayContentProperty = DependencyProperty.Register(
        nameof(PhaseOverlayContent),
        typeof(object),
        typeof(GraphCanvasControl));

    public static readonly DependencyProperty PhaseOverlayVisibleProperty = DependencyProperty.Register(
        nameof(PhaseOverlayVisible),
        typeof(bool),
        typeof(GraphCanvasControl),
        new FrameworkPropertyMetadata(
            true,
            FrameworkPropertyMetadataOptions.BindsTwoWayByDefault,
            OnPresentationChanged));

    public static readonly DependencyProperty ShowCrosshairProperty = DependencyProperty.Register(
        nameof(ShowCrosshair),
        typeof(bool),
        typeof(GraphCanvasControl),
        new PropertyMetadata(false, OnPresentationChanged));

    public static readonly DependencyProperty CrosshairPositionProperty = DependencyProperty.Register(
        nameof(CrosshairPosition),
        typeof(Point),
        typeof(GraphCanvasControl),
        new PropertyMetadata(new Point(0.5, 0.5)));

    public static readonly DependencyProperty ZoomLevelProperty = DependencyProperty.Register(
        nameof(ZoomLevel),
        typeof(double),
        typeof(GraphCanvasControl),
        new FrameworkPropertyMetadata(1d, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault),
        IsValidZoomLevel);

    public GraphCanvasControl()
    {
        InitializeComponent();
        Loaded += OnLoaded;
        ImageCoordinateSurface.MouseLeftButtonDown += OnImageMouseLeftButtonDown;
    }

    public event EventHandler<GraphImagePointEventArgs>? ImagePointInvoked;

    public ImageSource? ImageSource
    {
        get => (ImageSource?)GetValue(ImageSourceProperty);
        set => SetValue(ImageSourceProperty, value);
    }

    public object? OverlayContent
    {
        get => GetValue(OverlayContentProperty);
        set => SetValue(OverlayContentProperty, value);
    }

    public object? PhaseOverlayContent
    {
        get => GetValue(PhaseOverlayContentProperty);
        set => SetValue(PhaseOverlayContentProperty, value);
    }

    public bool PhaseOverlayVisible
    {
        get => (bool)GetValue(PhaseOverlayVisibleProperty);
        set => SetValue(PhaseOverlayVisibleProperty, value);
    }

    public bool ShowCrosshair
    {
        get => (bool)GetValue(ShowCrosshairProperty);
        set => SetValue(ShowCrosshairProperty, value);
    }

    /// <summary>Gets or sets the normalized canvas location of the crosshair.</summary>
    public Point CrosshairPosition
    {
        get => (Point)GetValue(CrosshairPositionProperty);
        set => SetValue(CrosshairPositionProperty, value);
    }

    public double ZoomLevel
    {
        get => (double)GetValue(ZoomLevelProperty);
        set => SetValue(ZoomLevelProperty, value);
    }

    private static bool IsValidZoomLevel(object value) =>
        value is double zoom && double.IsFinite(zoom) && zoom > 0;

    private static void OnPresentationChanged(DependencyObject dependencyObject, DependencyPropertyChangedEventArgs args)
    {
        _ = args;
        ((GraphCanvasControl)dependencyObject).RefreshPresentation();
    }

    private void OnLoaded(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        RefreshPresentation();
    }

    private void RefreshPresentation()
    {
        if (!IsInitialized)
        {
            return;
        }

        EmptyStateText.Visibility = ImageSource is null ? Visibility.Visible : Visibility.Collapsed;
        PhaseOverlayPresenter.Visibility = PhaseOverlayVisible ? Visibility.Visible : Visibility.Collapsed;
        Crosshair.Visibility = ShowCrosshair ? Visibility.Visible : Visibility.Collapsed;
    }

    private void OnImageMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        _ = sender;
        if (ImageSource is null)
        {
            return;
        }

        Point point = e.GetPosition(GraphImage);
        if (point.X < 0 || point.Y < 0 || point.X > GraphImage.ActualWidth || point.Y > GraphImage.ActualHeight)
        {
            return;
        }

        ImagePointInvoked?.Invoke(this, new GraphImagePointEventArgs(point));
        e.Handled = true;
    }
}

public sealed class GraphImagePointEventArgs(Point imagePoint) : EventArgs
{
    public Point ImagePoint { get; } = imagePoint;
}
