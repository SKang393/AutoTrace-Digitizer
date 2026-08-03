// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Axis.Tests;

internal sealed class AxisFixtureBuilder
{
    private readonly List<GeometryLineCandidate> _candidates = [];
    private int _nextId;

    public AxisFixtureBuilder(int width = 800, int height = 400)
    {
        Width = width;
        Height = height;
    }

    public int Width { get; }

    public int Height { get; }

    public AxisFixtureBuilder Line(
        double x1,
        double y1,
        double x2,
        double y2,
        LinePatternHint pattern = LinePatternHint.Solid,
        double strength = 1d,
        double strokeWidth = 1d,
        string? id = null,
        LineCandidateSource source = LineCandidateSource.RecordedFixture)
    {
        _candidates.Add(new GeometryLineCandidate(
            id ?? $"line-{++_nextId}",
            new GeometryLineSegment(new PixelPoint(x1, y1), new PixelPoint(x2, y2)),
            source,
            strength,
            strokeWidth,
            pattern));
        return this;
    }

    public AxisFixtureBuilder CleanAxes(
        double left = 100d,
        double top = 50d,
        double right = 700d,
        double bottom = 300d)
    {
        return Line(left, bottom, right, bottom, id: "x-axis")
            .Line(left, bottom, left, top, id: "y-axis");
    }

    public AxisFixtureBuilder XTick(double x, double y = 300d, double halfLength = 5d) =>
        Line(x, y - halfLength, x, y + halfLength, id: $"x-tick-{x:0.###}");

    public AxisFixtureBuilder YTick(double y, double x = 100d, double halfLength = 5d) =>
        Line(x - halfLength, y, x + halfLength, y, id: $"y-tick-{y:0.###}");

    public AxisFixtureBuilder DottedDivider(
        double x,
        double top = 50d,
        double bottom = 300d,
        double dash = 10d,
        double gap = 10d,
        LinePatternHint pattern = LinePatternHint.Dotted)
    {
        var index = 0;
        for (double y = top; y < bottom; y += dash + gap)
        {
            Line(
                x,
                y,
                x,
                Math.Min(bottom, y + dash),
                pattern,
                id: $"divider-{x:0.###}-{index++}");
        }

        return this;
    }

    public AxisFixtureBuilder VerticalGrid(double x, double top = 50d, double bottom = 300d) =>
        Line(x, top, x, bottom, LinePatternHint.Solid, strength: 0.6d, id: $"grid-v-{x:0.###}");

    public AxisFixtureBuilder HorizontalGrid(double y, double left = 100d, double right = 700d) =>
        Line(left, y, right, y, LinePatternHint.Solid, strength: 0.6d, id: $"grid-h-{y:0.###}");

    public AxisGeometryRequest Build(AxisGeometryOptions? options = null) =>
        new(Width, Height, _candidates.AsReadOnly(), options);
}
