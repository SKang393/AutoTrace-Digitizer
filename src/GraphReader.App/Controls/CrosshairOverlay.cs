// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Windows;
using System.Windows.Media;

namespace GraphReader.App.Controls;

public sealed class CrosshairOverlay : FrameworkElement
{
    public static readonly DependencyProperty PositionProperty = DependencyProperty.Register(
        nameof(Position),
        typeof(Point),
        typeof(CrosshairOverlay),
        new FrameworkPropertyMetadata(
            new Point(0.5, 0.5),
            FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty StrokeProperty = DependencyProperty.Register(
        nameof(Stroke),
        typeof(Brush),
        typeof(CrosshairOverlay),
        new FrameworkPropertyMetadata(
            SystemColors.HighlightBrush,
            FrameworkPropertyMetadataOptions.AffectsRender));

    public Point Position
    {
        get => (Point)GetValue(PositionProperty);
        set => SetValue(PositionProperty, value);
    }

    public Brush Stroke
    {
        get => (Brush)GetValue(StrokeProperty);
        set => SetValue(StrokeProperty, value);
    }

    protected override void OnRender(DrawingContext drawingContext)
    {
        base.OnRender(drawingContext);

        if (ActualWidth <= 0 || ActualHeight <= 0 ||
            !double.IsFinite(Position.X) || !double.IsFinite(Position.Y))
        {
            return;
        }

        var x = Math.Clamp(Position.X, 0, 1) * ActualWidth;
        var y = Math.Clamp(Position.Y, 0, 1) * ActualHeight;
        var pen = new Pen(Stroke, 1);
        if (pen.CanFreeze)
        {
            pen.Freeze();
        }

        drawingContext.DrawLine(pen, new Point(x, 0), new Point(x, ActualHeight));
        drawingContext.DrawLine(pen, new Point(0, y), new Point(ActualWidth, y));
    }
}
