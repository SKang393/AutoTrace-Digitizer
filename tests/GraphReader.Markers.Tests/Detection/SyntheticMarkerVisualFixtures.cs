// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Markers.Detection;

namespace GraphReader.Markers.Tests.Detection;

internal sealed record SyntheticMarkerVisualScene(
    string Name,
    MarkerImageFrame Frame,
    MarkerPolygon Plot,
    IReadOnlyList<MarkerPoint> GoldenCenters,
    IReadOnlyList<MarkerPoint> HardNegativeCenters,
    bool DuplicateProposal = false);

internal static class SyntheticMarkerVisualFixtures
{
    private const int Size = MarkerDetectionTestSupport.FrameSize;

    internal static IReadOnlyList<SyntheticMarkerVisualScene> OriginalScenes() =>
    [
        Dense(),
        OpenTouched(),
        Filled(),
        Arrow(),
        Axes(),
        LegendGlyph(),
        Text(),
        DottedDivider(),
        SameColumn(),
        Duplicate(),
    ];

    internal static SyntheticMarkerVisualScene Transform()
    {
        var canvas = new VisualCanvas();
        canvas.FillCircle(20, 28, 3);
        return canvas.Scene(
            "2x transform",
            [new MarkerPoint(10, 14)],
            [],
            MarkerSourceImage.Enhanced,
            new MarkerAffineTransform(2, 0, 0, 0, 2, 0),
            new MarkerRectangle(0, 0, 32, 32));
    }

    private static SyntheticMarkerVisualScene Dense()
    {
        MarkerPoint[] centers =
        [
            new(8, 10),
            new(16, 14),
            new(24, 18),
            new(32, 22),
            new(40, 26),
            new(48, 30),
            new(12, 42),
            new(28, 46),
            new(44, 50),
        ];
        var canvas = new VisualCanvas();
        foreach (MarkerPoint center in centers)
        {
            canvas.FillCircle((int)center.X, (int)center.Y, 2);
        }

        return canvas.Scene("dense filled markers", centers, []);
    }

    private static SyntheticMarkerVisualScene OpenTouched()
    {
        MarkerPoint[] centers = [new(14, 18), new(32, 30), new(50, 42)];
        var canvas = new VisualCanvas();
        canvas.Line(14, 18, 32, 30);
        canvas.Line(32, 30, 50, 42);
        foreach (MarkerPoint center in centers)
        {
            canvas.StrokeCircle((int)center.X, (int)center.Y, 3);
        }

        return canvas.Scene("open markers touched by lines", centers, []);
    }

    private static SyntheticMarkerVisualScene Filled()
    {
        MarkerPoint[] centers = [new(12, 16), new(30, 32), new(48, 46)];
        var canvas = new VisualCanvas();
        foreach (MarkerPoint center in centers)
        {
            canvas.FillCircle((int)center.X, (int)center.Y, 3);
        }

        return canvas.Scene("filled markers", centers, []);
    }

    private static SyntheticMarkerVisualScene Arrow()
    {
        var tip = new MarkerPoint(34, 28);
        var canvas = new VisualCanvas();
        canvas.Line(12, 48, 34, 28);
        canvas.Line(34, 28, 27, 30);
        canvas.Line(34, 28, 32, 35);
        canvas.MaskArtifactRectangle(26, 26, 10, 11);
        return canvas.Scene("annotation arrow", [], [tip]);
    }

    private static SyntheticMarkerVisualScene Axes()
    {
        var crossing = new MarkerPoint(10, 52);
        var canvas = new VisualCanvas();
        canvas.Line(10, 8, 10, 52);
        canvas.Line(10, 52, 58, 52);
        for (var x = 18; x <= 50; x += 8)
        {
            canvas.Line(x, 50, x, 54);
        }

        canvas.MaskArtifactRectangle(8, 6, 52, 50);
        return canvas.Scene("axes and ticks", [], [crossing]);
    }

    private static SyntheticMarkerVisualScene LegendGlyph()
    {
        var glyph = new MarkerPoint(46, 12);
        var canvas = new VisualCanvas();
        canvas.Rectangle(40, 6, 18, 12);
        canvas.StrokeCircle(46, 12, 3);
        canvas.Line(50, 12, 55, 12);
        canvas.MaskArtifactRectangle(39, 5, 20, 14);
        return canvas.Scene("legend glyph", [], [glyph]);
    }

    private static SyntheticMarkerVisualScene Text()
    {
        var textCenter = new MarkerPoint(28, 14);
        var canvas = new VisualCanvas();
        canvas.Rectangle(18, 9, 3, 10);
        canvas.Rectangle(24, 9, 3, 10);
        canvas.Rectangle(30, 9, 3, 10);
        canvas.Rectangle(36, 9, 3, 10);
        canvas.MaskOcrRectangle(16, 7, 25, 14);
        return canvas.Scene("text", [], [textCenter]);
    }

    private static SyntheticMarkerVisualScene DottedDivider()
    {
        var dividerCenter = new MarkerPoint(32, 30);
        var canvas = new VisualCanvas();
        for (var y = 8; y <= 52; y += 6)
        {
            canvas.Rectangle(31, y, 3, 3);
        }

        canvas.MaskArtifactRectangle(30, 6, 5, 50);
        return canvas.Scene("dotted divider", [], [dividerCenter]);
    }

    private static SyntheticMarkerVisualScene SameColumn()
    {
        MarkerPoint[] centers = [new(32, 10), new(32, 26), new(32, 42), new(32, 56)];
        var canvas = new VisualCanvas();
        foreach (MarkerPoint center in centers)
        {
            canvas.StrokeCircle((int)center.X, (int)center.Y, 2);
        }

        return canvas.Scene("same-column probes", centers, []);
    }

    private static SyntheticMarkerVisualScene Duplicate()
    {
        MarkerPoint center = new(28, 30);
        var canvas = new VisualCanvas();
        canvas.FillCircle((int)center.X, (int)center.Y, 3);
        return canvas.Scene("duplicate proposals", [center], [], duplicateProposal: true);
    }

    private sealed class VisualCanvas
    {
        private readonly float[] _pixels = Enumerable.Repeat(1f, Size * Size).ToArray();
        private readonly float[] _ocrMask = new float[Size * Size];
        private readonly float[] _artifactMask = new float[Size * Size];

        internal void FillCircle(int centerX, int centerY, int radius)
        {
            for (var y = centerY - radius; y <= centerY + radius; y++)
            {
                for (var x = centerX - radius; x <= centerX + radius; x++)
                {
                    if (((x - centerX) * (x - centerX)) + ((y - centerY) * (y - centerY)) <= radius * radius)
                    {
                        Ink(x, y);
                    }
                }
            }
        }

        internal void StrokeCircle(int centerX, int centerY, int radius)
        {
            for (var degrees = 0; degrees < 360; degrees += 5)
            {
                var radians = Math.PI * degrees / 180;
                Ink(
                    (int)Math.Round(centerX + (radius * Math.Cos(radians))),
                    (int)Math.Round(centerY + (radius * Math.Sin(radians))));
            }
        }

        internal void Line(int x1, int y1, int x2, int y2)
        {
            var steps = Math.Max(Math.Abs(x2 - x1), Math.Abs(y2 - y1));
            for (var step = 0; step <= steps; step++)
            {
                var fraction = (double)step / steps;
                Ink(
                    (int)Math.Round(x1 + ((x2 - x1) * fraction)),
                    (int)Math.Round(y1 + ((y2 - y1) * fraction)));
            }
        }

        internal void Rectangle(int x, int y, int width, int height)
        {
            for (var row = y; row < y + height; row++)
            {
                for (var column = x; column < x + width; column++)
                {
                    Ink(column, row);
                }
            }
        }

        internal void MaskOcrRectangle(int x, int y, int width, int height) =>
            MaskRectangle(_ocrMask, x, y, width, height);

        internal void MaskArtifactRectangle(int x, int y, int width, int height) =>
            MaskRectangle(_artifactMask, x, y, width, height);

        internal SyntheticMarkerVisualScene Scene(
            string name,
            IReadOnlyList<MarkerPoint> goldenCenters,
            IReadOnlyList<MarkerPoint> hardNegativeCenters,
            MarkerSourceImage source = MarkerSourceImage.Original,
            MarkerAffineTransform? transform = null,
            MarkerRectangle? plot = null,
            bool duplicateProposal = false) =>
            new(
                name,
                new MarkerImageFrame(
                    Size,
                    Size,
                    1,
                    _pixels,
                    source,
                    transform ?? MarkerAffineTransform.Identity,
                    new MarkerMask(Size, Size, _ocrMask),
                    new MarkerMask(Size, Size, _artifactMask)),
                MarkerPolygon.FromRectangle(plot ?? new MarkerRectangle(0, 0, Size, Size)),
                goldenCenters,
                hardNegativeCenters,
                duplicateProposal);

        private void Ink(int x, int y)
        {
            if (x >= 0 && x < Size && y >= 0 && y < Size)
            {
                _pixels[(y * Size) + x] = 0;
            }
        }

        private static void MaskRectangle(float[] mask, int x, int y, int width, int height)
        {
            for (var row = y; row < y + height; row++)
            {
                for (var column = x; column < x + width; column++)
                {
                    if (column >= 0 && column < Size && row >= 0 && row < Size)
                    {
                        mask[(row * Size) + column] = 1;
                    }
                }
            }
        }
    }
}
