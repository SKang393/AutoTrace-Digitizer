// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace GraphReader.App.Controls;

public partial class SeriesCardControl : UserControl
{
    public static readonly DependencyProperty SymbolGlyphProperty = DependencyProperty.Register(
        nameof(SymbolGlyph),
        typeof(string),
        typeof(SeriesCardControl),
        new PropertyMetadata(string.Empty));

    public static readonly DependencyProperty AccessibleSymbolNameProperty = DependencyProperty.Register(
        nameof(AccessibleSymbolName),
        typeof(string),
        typeof(SeriesCardControl),
        new PropertyMetadata(string.Empty));

    public static readonly DependencyProperty InferredLabelProperty = DependencyProperty.Register(
        nameof(InferredLabel),
        typeof(string),
        typeof(SeriesCardControl),
        new PropertyMetadata(string.Empty));

    public static readonly DependencyProperty MarkerCountProperty = DependencyProperty.Register(
        nameof(MarkerCount),
        typeof(int),
        typeof(SeriesCardControl),
        new PropertyMetadata(0));

    public static readonly DependencyProperty ConfidenceProperty = DependencyProperty.Register(
        nameof(Confidence),
        typeof(double),
        typeof(SeriesCardControl),
        new PropertyMetadata(0d, OnConfidenceChanged), IsValidConfidence);

    public static readonly DependencyProperty IsSeriesVisibleProperty = DependencyProperty.Register(
        nameof(IsSeriesVisible),
        typeof(bool),
        typeof(SeriesCardControl),
        new FrameworkPropertyMetadata(true, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));

    public static readonly DependencyProperty SelectCommandProperty = DependencyProperty.Register(
        nameof(SelectCommand),
        typeof(ICommand),
        typeof(SeriesCardControl));

    public static readonly DependencyProperty ReassignCommandProperty = DependencyProperty.Register(
        nameof(ReassignCommand),
        typeof(ICommand),
        typeof(SeriesCardControl));

    public static readonly DependencyProperty CommandParameterProperty = DependencyProperty.Register(
        nameof(CommandParameter),
        typeof(object),
        typeof(SeriesCardControl));

    public SeriesCardControl()
    {
        InitializeComponent();
        Loaded += OnLoaded;
    }

    public string SymbolGlyph
    {
        get => (string)GetValue(SymbolGlyphProperty);
        set => SetValue(SymbolGlyphProperty, value);
    }

    public string AccessibleSymbolName
    {
        get => (string)GetValue(AccessibleSymbolNameProperty);
        set => SetValue(AccessibleSymbolNameProperty, value);
    }

    public string InferredLabel
    {
        get => (string)GetValue(InferredLabelProperty);
        set => SetValue(InferredLabelProperty, value);
    }

    public int MarkerCount
    {
        get => (int)GetValue(MarkerCountProperty);
        set => SetValue(MarkerCountProperty, value);
    }

    public double Confidence
    {
        get => (double)GetValue(ConfidenceProperty);
        set => SetValue(ConfidenceProperty, value);
    }

    public bool IsSeriesVisible
    {
        get => (bool)GetValue(IsSeriesVisibleProperty);
        set => SetValue(IsSeriesVisibleProperty, value);
    }

    public ICommand? SelectCommand
    {
        get => (ICommand?)GetValue(SelectCommandProperty);
        set => SetValue(SelectCommandProperty, value);
    }

    public ICommand? ReassignCommand
    {
        get => (ICommand?)GetValue(ReassignCommandProperty);
        set => SetValue(ReassignCommandProperty, value);
    }

    public object? CommandParameter
    {
        get => GetValue(CommandParameterProperty);
        set => SetValue(CommandParameterProperty, value);
    }

    private static bool IsValidConfidence(object value) =>
        value is double confidence && double.IsFinite(confidence) && confidence is >= 0 and <= 1;

    private static void OnConfidenceChanged(DependencyObject dependencyObject, DependencyPropertyChangedEventArgs args)
    {
        _ = args;
        ((SeriesCardControl)dependencyObject).RefreshConfidence();
    }

    private void OnLoaded(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        RefreshConfidence();
    }

    private void RefreshConfidence()
    {
        if (!IsInitialized)
        {
            return;
        }

        var format = TryFindResource("SeriesCard.ConfidenceFormat") as string;
        ConfidenceText.Text = string.IsNullOrEmpty(format)
            ? string.Empty
            : string.Format(CultureInfo.CurrentCulture, format, Confidence);
    }
}
