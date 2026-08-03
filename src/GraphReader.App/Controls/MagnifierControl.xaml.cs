// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace GraphReader.App.Controls;

public partial class MagnifierControl : UserControl
{
    public static readonly DependencyProperty OriginalImageSourceProperty = DependencyProperty.Register(
        nameof(OriginalImageSource),
        typeof(ImageSource),
        typeof(MagnifierControl),
        new PropertyMetadata(null, OnImageSourceChanged));

    public static readonly DependencyProperty EnhancedImageSourceProperty = DependencyProperty.Register(
        nameof(EnhancedImageSource),
        typeof(ImageSource),
        typeof(MagnifierControl),
        new PropertyMetadata(null, OnImageSourceChanged));

    public static readonly DependencyProperty ShowEnhancedProperty = DependencyProperty.Register(
        nameof(ShowEnhanced),
        typeof(bool),
        typeof(MagnifierControl),
        new FrameworkPropertyMetadata(
            false,
            FrameworkPropertyMetadataOptions.BindsTwoWayByDefault,
            OnDisplayPropertyChanged));

    public static readonly DependencyProperty CrosshairPositionProperty = DependencyProperty.Register(
        nameof(CrosshairPosition),
        typeof(Point),
        typeof(MagnifierControl),
        new PropertyMetadata(new Point(0.5, 0.5)));

    public static readonly DependencyProperty PixelPositionProperty = DependencyProperty.Register(
        nameof(PixelPosition),
        typeof(Point),
        typeof(MagnifierControl),
        new PropertyMetadata(default(Point), OnDisplayPropertyChanged));

    public static readonly DependencyProperty GraphPositionProperty = DependencyProperty.Register(
        nameof(GraphPosition),
        typeof(Point?),
        typeof(MagnifierControl),
        new PropertyMetadata(null, OnDisplayPropertyChanged));

    public static readonly DependencyProperty NearestDetectionNameProperty = DependencyProperty.Register(
        nameof(NearestDetectionName),
        typeof(string),
        typeof(MagnifierControl),
        new PropertyMetadata(null, OnDisplayPropertyChanged));

    public static readonly DependencyProperty NearestDetectionConfidenceProperty = DependencyProperty.Register(
        nameof(NearestDetectionConfidence),
        typeof(double?),
        typeof(MagnifierControl),
        new PropertyMetadata(null, OnDisplayPropertyChanged));

    public static readonly DependencyProperty ZoomLevelProperty = DependencyProperty.Register(
        nameof(ZoomLevel),
        typeof(double),
        typeof(MagnifierControl),
        new PropertyMetadata(2d, OnDisplayPropertyChanged), IsValidZoomLevel);

    public MagnifierControl()
    {
        InitializeComponent();
        Loaded += OnLoaded;
    }

    public ImageSource? OriginalImageSource
    {
        get => (ImageSource?)GetValue(OriginalImageSourceProperty);
        set => SetValue(OriginalImageSourceProperty, value);
    }

    public ImageSource? EnhancedImageSource
    {
        get => (ImageSource?)GetValue(EnhancedImageSourceProperty);
        set => SetValue(EnhancedImageSourceProperty, value);
    }

    public bool ShowEnhanced
    {
        get => (bool)GetValue(ShowEnhancedProperty);
        set => SetValue(ShowEnhancedProperty, value);
    }

    /// <summary>Gets or sets the normalized viewport location of the crosshair.</summary>
    public Point CrosshairPosition
    {
        get => (Point)GetValue(CrosshairPositionProperty);
        set => SetValue(CrosshairPositionProperty, value);
    }

    public Point PixelPosition
    {
        get => (Point)GetValue(PixelPositionProperty);
        set => SetValue(PixelPositionProperty, value);
    }

    public Point? GraphPosition
    {
        get => (Point?)GetValue(GraphPositionProperty);
        set => SetValue(GraphPositionProperty, value);
    }

    public string? NearestDetectionName
    {
        get => (string?)GetValue(NearestDetectionNameProperty);
        set => SetValue(NearestDetectionNameProperty, value);
    }

    public double? NearestDetectionConfidence
    {
        get => (double?)GetValue(NearestDetectionConfidenceProperty);
        set => SetValue(NearestDetectionConfidenceProperty, value);
    }

    public double ZoomLevel
    {
        get => (double)GetValue(ZoomLevelProperty);
        set => SetValue(ZoomLevelProperty, value);
    }

    private static bool IsValidZoomLevel(object value) =>
        value is double zoom && double.IsFinite(zoom) && zoom > 0;

    private static void OnImageSourceChanged(DependencyObject dependencyObject, DependencyPropertyChangedEventArgs args)
    {
        _ = args;
        ((MagnifierControl)dependencyObject).RefreshImage();
    }

    private static void OnDisplayPropertyChanged(DependencyObject dependencyObject, DependencyPropertyChangedEventArgs args)
    {
        _ = args;
        var control = (MagnifierControl)dependencyObject;
        control.RefreshImage();
        control.RefreshDisplayValues();
    }

    private void OnLoaded(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        RefreshImage();
        RefreshDisplayValues();
    }

    private void OnOriginalModeChecked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        ShowEnhanced = false;
    }

    private void OnEnhancedModeChecked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        ShowEnhanced = true;
    }

    private void RefreshImage()
    {
        if (!IsInitialized)
        {
            return;
        }

        var selectedSource = ShowEnhanced && EnhancedImageSource is not null
            ? EnhancedImageSource
            : OriginalImageSource;

        ViewportImage.Source = selectedSource;
        EmptyImageText.Visibility = selectedSource is null ? Visibility.Visible : Visibility.Collapsed;
        EnhancedModeButton.IsEnabled = EnhancedImageSource is not null;
        OriginalModeButton.IsChecked = !ShowEnhanced;
        EnhancedModeButton.IsChecked = ShowEnhanced;
    }

    private void RefreshDisplayValues()
    {
        if (!IsInitialized)
        {
            return;
        }

        PixelPositionText.Text = FormatResource("Magnifier.CoordinateFormat", PixelPosition.X, PixelPosition.Y);
        GraphPositionText.Text = GraphPosition is Point graphPosition
            ? FormatResource("Magnifier.CoordinateFormat", graphPosition.X, graphPosition.Y)
            : ResourceText("Magnifier.NearestDetection.None");

        var hasDetection = !string.IsNullOrWhiteSpace(NearestDetectionName);
        NearestDetectionText.Text = hasDetection ? NearestDetectionName : null;
        NearestDetectionText.Visibility = hasDetection ? Visibility.Visible : Visibility.Collapsed;
        NoDetectionText.Visibility = hasDetection ? Visibility.Collapsed : Visibility.Visible;

        ConfidenceText.Text = NearestDetectionConfidence is double confidence
            ? FormatResource("Magnifier.ConfidenceFormat", confidence)
            : ResourceText("Magnifier.NearestDetection.None");
        ZoomText.Text = FormatResource("Magnifier.ZoomFormat", ZoomLevel);
    }

    private string FormatResource(string key, params object[] arguments)
    {
        var format = ResourceText(key);
        return string.IsNullOrEmpty(format)
            ? string.Empty
            : string.Format(CultureInfo.CurrentCulture, format, arguments);
    }

    private string ResourceText(string key) => TryFindResource(key) as string ?? string.Empty;
}
