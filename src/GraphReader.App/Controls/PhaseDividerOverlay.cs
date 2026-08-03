// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Windows;
using System.Windows.Media;

namespace GraphReader.App.Controls;

public sealed class PhaseDividerOverlay : FrameworkElement
{
    public static readonly DependencyProperty NormalizedXProperty = DependencyProperty.Register(
        nameof(NormalizedX),
        typeof(double),
        typeof(PhaseDividerOverlay),
        new FrameworkPropertyMetadata(
            0.5d,
            FrameworkPropertyMetadataOptions.AffectsRender),
        IsNormalizedValue);

    public static readonly DependencyProperty NormalizedTopProperty = DependencyProperty.Register(
        nameof(NormalizedTop),
        typeof(double),
        typeof(PhaseDividerOverlay),
        new FrameworkPropertyMetadata(
            0d,
            FrameworkPropertyMetadataOptions.AffectsRender),
        IsNormalizedValue);

    public static readonly DependencyProperty NormalizedBottomProperty = DependencyProperty.Register(
        nameof(NormalizedBottom),
        typeof(double),
        typeof(PhaseDividerOverlay),
        new FrameworkPropertyMetadata(
            1d,
            FrameworkPropertyMetadataOptions.AffectsRender),
        IsNormalizedValue);

    public double NormalizedX
    {
        get => (double)GetValue(NormalizedXProperty);
        set => SetValue(NormalizedXProperty, value);
    }

    public double NormalizedTop
    {
        get => (double)GetValue(NormalizedTopProperty);
        set => SetValue(NormalizedTopProperty, value);
    }

    public double NormalizedBottom
    {
        get => (double)GetValue(NormalizedBottomProperty);
        set => SetValue(NormalizedBottomProperty, value);
    }

    protected override void OnRender(DrawingContext drawingContext)
    {
        base.OnRender(drawingContext);
        if (ActualWidth <= 0 || ActualHeight <= 0 || NormalizedBottom <= NormalizedTop)
        {
            return;
        }

        var brush = TryFindResource("App.Brush.TextMuted") as Brush ?? SystemColors.GrayTextBrush;
        Pen pen = new(brush, 1)
        {
            DashStyle = DashStyles.Dash,
        };
        if (pen.CanFreeze)
        {
            pen.Freeze();
        }

        double x = NormalizedX * ActualWidth;
        drawingContext.DrawLine(
            pen,
            new Point(x, NormalizedTop * ActualHeight),
            new Point(x, NormalizedBottom * ActualHeight));
    }

    private static bool IsNormalizedValue(object value) =>
        value is double number && double.IsFinite(number) && number is >= 0 and <= 1;
}
