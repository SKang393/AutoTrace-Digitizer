// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace GraphReader.Axis;

public sealed record OpenCvLineCandidateOptions
{
    public bool UseLineSegmentDetector { get; init; } = true;

    public bool UseProbabilisticHough { get; init; } = true;

    public double CannyLowThreshold { get; init; } = 40d;

    public double CannyHighThreshold { get; init; } = 120d;

    public int HoughThreshold { get; init; } = 24;

    public double HoughMinimumLineLengthPixels { get; init; } = 12d;

    public double HoughMaximumLineGapPixels { get; init; } = 4d;

    public LineSegmentDetectorModes LsdRefinement { get; init; } = LineSegmentDetectorModes.RefineStd;
}

/// <summary>
/// Extracts detector-neutral line candidates from an original-pixel grayscale
/// frame using OpenCV LSD and probabilistic Hough. It does not classify or emit
/// plotted markers.
/// </summary>
public sealed class OpenCvLineCandidateProvider : ILineCandidateProvider
{
    private readonly OpenCvLineCandidateOptions _options;

    public OpenCvLineCandidateProvider(OpenCvLineCandidateOptions? options = null)
    {
        _options = options ?? new OpenCvLineCandidateOptions();
        ValidateOptions(_options);
    }

    public async ValueTask<IReadOnlyList<GeometryLineCandidate>> DetectLinesAsync(
        GrayscaleLineCandidateFrame frame,
        CancellationToken cancellationToken)
    {
        ValidateFrame(frame);
        cancellationToken.ThrowIfCancellationRequested();
        IReadOnlyList<GeometryLineCandidate> candidates = await Task.Run(
            () => Detect(frame, cancellationToken),
            cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        return candidates;
    }

    private ReadOnlyCollection<GeometryLineCandidate> Detect(
        GrayscaleLineCandidateFrame frame,
        CancellationToken cancellationToken)
    {
        byte[] pixels = frame.Pixels.ToArray();
        using var gray = new Mat(frame.Height, frame.Width, MatType.CV_8UC1);
        long destinationStride = checked((long)gray.Step());
        for (int row = 0; row < frame.Height; row++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Marshal.Copy(
                pixels,
                checked(row * frame.Stride),
                IntPtr.Add(gray.Data, checked((int)(row * destinationStride))),
                frame.Width);
        }

        var candidates = new List<GeometryLineCandidate>();

        if (_options.UseLineSegmentDetector)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using LineSegmentDetector detector = LineSegmentDetector.Create(_options.LsdRefinement);
            detector.Detect(
                gray,
                out Vec4f[] lines,
                out double[] widths,
                out _,
                out _);
            cancellationToken.ThrowIfCancellationRequested();
            for (int index = 0; index < lines.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Vec4f line = lines[index];
                double width = index < widths.Length && double.IsFinite(widths[index]) && widths[index] > 0
                    ? widths[index]
                    : 1d;
                candidates.Add(new GeometryLineCandidate(
                    $"opencv-lsd-{index:D6}",
                    new GeometryLineSegment(
                        new PixelPoint(line.Item0, line.Item1),
                        new PixelPoint(line.Item2, line.Item3)),
                    LineCandidateSource.OpenCvLsd,
                    Strength: 1d,
                    StrokeWidthPixels: width));
            }
        }

        if (_options.UseProbabilisticHough)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var edges = new Mat();
            Cv2.Canny(
                gray,
                edges,
                _options.CannyLowThreshold,
                _options.CannyHighThreshold,
                apertureSize: 3,
                L2gradient: true);
            LineSegmentPoint[] lines = Cv2.HoughLinesP(
                edges,
                rho: 1d,
                theta: Math.PI / 180d,
                threshold: _options.HoughThreshold,
                minLineLength: _options.HoughMinimumLineLengthPixels,
                maxLineGap: _options.HoughMaximumLineGapPixels);
            cancellationToken.ThrowIfCancellationRequested();
            for (int index = 0; index < lines.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                LineSegmentPoint line = lines[index];
                candidates.Add(new GeometryLineCandidate(
                    $"opencv-hough-{index:D6}",
                    new GeometryLineSegment(
                        new PixelPoint(line.P1.X, line.P1.Y),
                        new PixelPoint(line.P2.X, line.P2.Y)),
                    LineCandidateSource.OpenCvHough));
            }
        }

        return candidates.AsReadOnly();
    }

    private static void ValidateFrame(GrayscaleLineCandidateFrame frame)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (!string.Equals(
                frame.CoordinateSpace,
                AxisGeometryCoordinateSpaces.OriginalPixels,
                StringComparison.Ordinal))
        {
            throw new ArgumentException(
                "OpenCV line detection accepts only original-pixel frames.",
                nameof(frame));
        }

        if (frame.Width <= 0 || frame.Height <= 0 || frame.Stride < frame.Width)
        {
            throw new ArgumentException("The grayscale frame dimensions and stride are invalid.", nameof(frame));
        }

        long requiredBytes = checked((long)frame.Stride * frame.Height);
        if (requiredBytes > frame.Pixels.Length)
        {
            throw new ArgumentException("The grayscale frame buffer is shorter than its declared stride and height.", nameof(frame));
        }
    }

    private static void ValidateOptions(OpenCvLineCandidateOptions options)
    {
        if (!options.UseLineSegmentDetector && !options.UseProbabilisticHough)
        {
            throw new ArgumentException("At least one OpenCV line detector must be enabled.", nameof(options));
        }

        if (!double.IsFinite(options.CannyLowThreshold) || options.CannyLowThreshold < 0 ||
            !double.IsFinite(options.CannyHighThreshold) ||
            options.CannyHighThreshold <= options.CannyLowThreshold ||
            options.HoughThreshold <= 0 ||
            !double.IsFinite(options.HoughMinimumLineLengthPixels) ||
            options.HoughMinimumLineLengthPixels <= 0 ||
            !double.IsFinite(options.HoughMaximumLineGapPixels) ||
            options.HoughMaximumLineGapPixels < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "OpenCV line detector parameters are invalid.");
        }
    }
}
