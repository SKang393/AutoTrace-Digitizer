// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using GraphReader.Axis;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Integration.Tests.IntegrationSmoke;

[TestClass]
public sealed class OpenCvSourceRuntimeParityTests
{
    private const string EvidenceOutputVariable = "GRAPHREADER_OPENCV_PARITY_OUTPUT";
    private static readonly JsonSerializerOptions EvidenceJsonOptions = new() { WriteIndented = true };

    [TestMethod]
    public async Task PublicOpenCvAxisBenchmarkMatchesFixedMetricBudget()
    {
        OpenCvAxisFixture[] fixtures =
        [
            CreateCleanAxes(),
            CreateAxesWithTicksAndDivider(),
            CreateNoisyPaddedAxes(),
            CreateAxesWithNonDataStrokes(),
        ];

        var results = new List<OpenCvAxisFixtureResult>(fixtures.Length);
        foreach (OpenCvAxisFixture fixture in fixtures)
        {
            var provider = new OpenCvLineCandidateProvider(new OpenCvLineCandidateOptions
            {
                HoughThreshold = 12,
                HoughMinimumLineLengthPixels = 20,
                HoughMaximumLineGapPixels = 3,
            });

            IReadOnlyList<GeometryLineCandidate> candidates =
                await provider.DetectLinesAsync(fixture.Frame, CancellationToken.None);
            AxisGeometryResult geometry = await new AxisGeometryDetector().DetectAsync(
                fixture.Frame,
                provider,
                cancellationToken: CancellationToken.None);

            double originError = Distance(geometry.PlotPolygon.BottomLeft, fixture.ExpectedOrigin);
            int lsdCount = candidates.Count(static candidate => candidate.Source == LineCandidateSource.OpenCvLsd);
            int houghCount = candidates.Count(static candidate => candidate.Source == LineCandidateSource.OpenCvHough);

            Assert.IsTrue(lsdCount > 0, $"{fixture.Id}: LSD returned no candidates.");
            Assert.IsTrue(houghCount > 0, $"{fixture.Id}: Hough returned no candidates.");
            Assert.IsTrue(originError <= 5d, $"{fixture.Id}: axis-origin error {originError:F6}px exceeded 5px.");
            Assert.AreEqual(AxisGeometryCoordinateSpaces.OriginalPixels, geometry.CoordinateSpace);
            Assert.IsTrue(geometry.Diagnostics.AcceptedCandidateCount > 0);

            results.Add(new OpenCvAxisFixtureResult(
                fixture.Id,
                fixture.Frame.Width,
                fixture.Frame.Height,
                Round(fixture.ExpectedOrigin.X),
                Round(fixture.ExpectedOrigin.Y),
                Round(geometry.PlotPolygon.BottomLeft.X),
                Round(geometry.PlotPolygon.BottomLeft.Y),
                Round(originError),
                lsdCount,
                houghCount,
                geometry.Diagnostics.AcceptedCandidateCount,
                CandidateFingerprint(candidates)));
        }

        var evidence = new OpenCvAxisParityEvidence(
            "graphreader.opencv-public-axis-parity.v1",
            AxisGeometryCoordinateSpaces.OriginalPixels,
            5d,
            results);
        WriteOptionalEvidence(evidence);
    }

    private static OpenCvAxisFixture CreateCleanAxes() =>
        BuildFixture("clean-axes", 256, 160, 256, 30, 130, static (_, _, _, _) => { });

    private static OpenCvAxisFixture CreateAxesWithTicksAndDivider() =>
        BuildFixture("ticks-and-divider", 320, 210, 320, 42, 172, static (pixels, width, stride, axisY) =>
        {
            for (int x = 74; x <= 298; x += 32)
            {
                DrawVertical(pixels, stride, x, axisY - 4, axisY + 4, 0);
            }

            for (int y = 47; y <= 147; y += 25)
            {
                DrawHorizontal(pixels, stride, 38, 46, y, 0);
            }

            for (int y = 48; y <= axisY - 10; y += 14)
            {
                DrawVertical(pixels, stride, 182, y, Math.Min(y + 6, axisY - 10), 0);
            }

            DrawHorizontal(pixels, stride, 82, 142, 24, 0);
        });

    private static OpenCvAxisFixture CreateNoisyPaddedAxes() =>
        BuildFixture("noisy-padded-axes", 280, 190, 288, 36, 156, static (pixels, width, stride, axisY) =>
        {
            for (int y = 0; y < axisY; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (((x * 31) + (y * 17)) % 211 == 0)
                    {
                        pixels[(y * stride) + x] = 205;
                    }
                }
            }

            DrawVertical(pixels, stride, 196, 45, axisY - 8, 64);
        });

    private static OpenCvAxisFixture CreateAxesWithNonDataStrokes() =>
        BuildFixture("axes-with-nondata-strokes", 300, 180, 300, 34, 148, static (pixels, _, stride, axisY) =>
        {
            DrawHorizontal(pixels, stride, 78, 126, 22, 0);
            DrawVertical(pixels, stride, 78, 22, 34, 0);
            DrawHorizontal(pixels, stride, 176, 246, 30, 0);
            DrawVertical(pixels, stride, 176, 30, 42, 0);
            DrawLine(pixels, stride, 218, axisY - 42, 231, axisY - 56, 0);
            DrawLine(pixels, stride, 231, axisY - 56, 228, axisY - 48, 0);
        });

    private static OpenCvAxisFixture BuildFixture(
        string id,
        int width,
        int height,
        int stride,
        int axisX,
        int axisY,
        Action<byte[], int, int, int> decorate)
    {
        var pixels = new byte[checked(stride * height)];
        Array.Fill(pixels, byte.MaxValue);
        DrawHorizontal(pixels, stride, axisX, width - 24, axisY, 0);
        DrawHorizontal(pixels, stride, axisX, width - 24, axisY + 1, 0);
        DrawVertical(pixels, stride, axisX, 18, axisY + 1, 0);
        DrawVertical(pixels, stride, axisX + 1, 18, axisY + 1, 0);
        decorate(pixels, width, stride, axisY);
        return new OpenCvAxisFixture(
            id,
            new GrayscaleLineCandidateFrame(width, height, stride, pixels),
            new PixelPoint(axisX + 0.5d, axisY + 0.5d));
    }

    private static void DrawHorizontal(byte[] pixels, int stride, int x1, int x2, int y, byte value)
    {
        for (int x = x1; x <= x2; x++)
        {
            pixels[(y * stride) + x] = value;
        }
    }

    private static void DrawVertical(byte[] pixels, int stride, int x, int y1, int y2, byte value)
    {
        for (int y = y1; y <= y2; y++)
        {
            pixels[(y * stride) + x] = value;
        }
    }

    private static void DrawLine(
        byte[] pixels,
        int stride,
        int x1,
        int y1,
        int x2,
        int y2,
        byte value)
    {
        int dx = Math.Abs(x2 - x1);
        int sx = x1 < x2 ? 1 : -1;
        int dy = -Math.Abs(y2 - y1);
        int sy = y1 < y2 ? 1 : -1;
        int error = dx + dy;
        while (true)
        {
            pixels[(y1 * stride) + x1] = value;
            if (x1 == x2 && y1 == y2)
            {
                return;
            }

            int doubled = 2 * error;
            if (doubled >= dy)
            {
                error += dy;
                x1 += sx;
            }

            if (doubled <= dx)
            {
                error += dx;
                y1 += sy;
            }
        }
    }

    private static string CandidateFingerprint(IReadOnlyList<GeometryLineCandidate> candidates)
    {
        string canonical = string.Join(
            '\n',
            candidates
                .Select(static candidate => CanonicalCandidate(candidate))
                .Order(StringComparer.Ordinal));
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(canonical))).ToLowerInvariant();
    }

    private static string CanonicalCandidate(GeometryLineCandidate candidate)
    {
        PixelPoint first = candidate.Segment.Start;
        PixelPoint second = candidate.Segment.End;
        if (Compare(first, second) > 0)
        {
            (first, second) = (second, first);
        }

        return FormattableString.Invariant(
            $"{candidate.Source}|{Round(first.X):F6}|{Round(first.Y):F6}|{Round(second.X):F6}|{Round(second.Y):F6}|{Round(candidate.StrokeWidthPixels):F6}");
    }

    private static int Compare(PixelPoint left, PixelPoint right)
    {
        int x = left.X.CompareTo(right.X);
        return x != 0 ? x : left.Y.CompareTo(right.Y);
    }

    private static double Distance(PixelPoint left, PixelPoint right)
    {
        double dx = left.X - right.X;
        double dy = left.Y - right.Y;
        return Math.Sqrt((dx * dx) + (dy * dy));
    }

    private static double Round(double value) => Math.Round(value, 6, MidpointRounding.AwayFromZero);

    private static void WriteOptionalEvidence(OpenCvAxisParityEvidence evidence)
    {
        string? outputPath = Environment.GetEnvironmentVariable(EvidenceOutputVariable);
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            return;
        }

        string fullPath = Path.GetFullPath(outputPath);
        string? parent = Path.GetDirectoryName(fullPath);
        if (string.IsNullOrWhiteSpace(parent))
        {
            throw new InvalidOperationException("OpenCV parity evidence output must have a parent directory.");
        }

        Directory.CreateDirectory(parent);
        File.WriteAllText(
            fullPath,
            JsonSerializer.Serialize(evidence, EvidenceJsonOptions) + Environment.NewLine,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }

    private sealed record OpenCvAxisFixture(
        string Id,
        GrayscaleLineCandidateFrame Frame,
        PixelPoint ExpectedOrigin);

    private sealed record OpenCvAxisFixtureResult(
        string FixtureId,
        int Width,
        int Height,
        double ExpectedOriginX,
        double ExpectedOriginY,
        double ActualOriginX,
        double ActualOriginY,
        double OriginErrorPixels,
        int LsdCandidateCount,
        int HoughCandidateCount,
        int AcceptedCandidateCount,
        string CandidateFingerprintSha256);

    private sealed record OpenCvAxisParityEvidence(
        string Schema,
        string CoordinateSpace,
        double MaximumOriginErrorPixels,
        IReadOnlyList<OpenCvAxisFixtureResult> Fixtures);
}
