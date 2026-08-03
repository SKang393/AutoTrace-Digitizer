// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Axis;

/// <summary>
/// A location in the immutable original image, measured in pixels.
/// </summary>
public readonly record struct PixelPoint(double X, double Y)
{
    public bool IsFinite => double.IsFinite(X) && double.IsFinite(Y);
}

/// <summary>
/// A finite line segment in original-image pixels.
/// </summary>
public readonly record struct GeometryLineSegment(PixelPoint Start, PixelPoint End)
{
    public double Length
    {
        get
        {
            var deltaX = End.X - Start.X;
            var deltaY = End.Y - Start.Y;
            return Math.Sqrt((deltaX * deltaX) + (deltaY * deltaY));
        }
    }

    public PixelPoint Midpoint => new((Start.X + End.X) / 2d, (Start.Y + End.Y) / 2d);
}

public enum LineCandidateSource
{
    OpenCvLsd,
    OpenCvHough,
    RecordedFixture,
    Other,
}

public enum LinePatternHint
{
    Unknown,
    Solid,
    Dashed,
    Dotted,
}

/// <summary>
/// A detector-neutral line candidate. OpenCV adapters should convert LSD or
/// Hough results to this contract before geometry fitting.
/// </summary>
public sealed record GeometryLineCandidate(
    string CandidateId,
    GeometryLineSegment Segment,
    LineCandidateSource Source,
    double Strength = 1d,
    double StrokeWidthPixels = 1d,
    LinePatternHint PatternHint = LinePatternHint.Unknown);

/// <summary>
/// An 8-bit grayscale original-image frame for a native line-candidate adapter.
/// </summary>
public sealed record GrayscaleLineCandidateFrame(
    int Width,
    int Height,
    int Stride,
    ReadOnlyMemory<byte> Pixels,
    string CoordinateSpace = AxisGeometryCoordinateSpaces.OriginalPixels);

public interface ILineCandidateProvider
{
    ValueTask<IReadOnlyList<GeometryLineCandidate>> DetectLinesAsync(
        GrayscaleLineCandidateFrame frame,
        CancellationToken cancellationToken);
}

public static class AxisGeometryCoordinateSpaces
{
    public const string OriginalPixels = "original_pixels";
}

public sealed record AxisGeometryOptions
{
    public double MaximumAxisDeviationDegrees { get; init; } = 15d;

    public double MergeAngleToleranceDegrees { get; init; } = 4d;

    public double MergeDistancePixels { get; init; } = 3d;

    public double MinimumCandidateLengthPixels { get; init; } = 1d;

    public double MinimumAxisSpanFraction { get; init; } = 0.25d;

    public double TickMaximumLengthFraction { get; init; } = 0.08d;

    public double TickAxisDistancePixels { get; init; } = 4d;

    public double DividerMinimumSpanFraction { get; init; } = 0.45d;

    public double DividerMinimumCoverageFraction { get; init; } = 0.06d;

    public int DottedDividerMinimumSegments { get; init; } = 3;

    public double DottedDividerMaximumSegmentFraction { get; init; } = 0.18d;

    public double PlotEdgeExclusionFraction { get; init; } = 0.025d;

    public double GridSpacingCoefficientOfVariation { get; init; } = 0.12d;

    public int GridMinimumAlignedLines { get; init; } = 3;

    public double NeedsReviewConfidenceThreshold { get; init; } = 0.55d;
}

public sealed record AxisGeometryRequest(
    int ImageWidth,
    int ImageHeight,
    IReadOnlyList<GeometryLineCandidate> LineCandidates,
    AxisGeometryOptions? Options = null,
    string CoordinateSpace = AxisGeometryCoordinateSpaces.OriginalPixels);

public sealed record AxisLineFit(
    GeometryLineSegment Line,
    double Confidence,
    double RootMeanSquareErrorPixels,
    double CoverageFraction,
    IReadOnlyList<string> SupportingCandidateIds);

public sealed record PlotPolygon(
    PixelPoint BottomLeft,
    PixelPoint BottomRight,
    PixelPoint TopRight,
    PixelPoint TopLeft)
{
    public IReadOnlyList<PixelPoint> Points =>
        Array.AsReadOnly([BottomLeft, BottomRight, TopRight, TopLeft]);
}

public enum TickAxis
{
    XAxis,
    YAxis,
}

public sealed record AxisTickGeometry(
    string TickId,
    TickAxis Axis,
    PixelPoint Center,
    GeometryLineSegment Line,
    double Confidence,
    IReadOnlyList<string> SupportingCandidateIds);

public enum DividerStyle
{
    Unknown,
    Solid,
    Dashed,
    Dotted,
}

public sealed record PhaseDividerGeometry(
    string DividerId,
    GeometryLineSegment Line,
    DividerStyle Style,
    double Confidence,
    double PlotSpanFraction,
    double CoverageFraction,
    IReadOnlyList<string> SupportingCandidateIds);

/// <summary>
/// A geometrically reliable vertical family whose semantics cannot be resolved
/// without OCR, cross-panel evidence, or user confirmation.
/// </summary>
public sealed record AmbiguousGridOrDividerGeometry(
    string AmbiguityId,
    GeometryLineSegment Line,
    double GeometryConfidence,
    double PlotSpanFraction,
    double CoverageFraction,
    IReadOnlyList<string> SupportingCandidateIds);

public sealed record AxisGeometryUncertainty(
    double XAxisRootMeanSquareErrorPixels,
    double YAxisRootMeanSquareErrorPixels,
    double BestAlternativeScoreMargin,
    bool NeedsReview,
    IReadOnlyList<string> Reasons);

public sealed record AxisGeometryDiagnostics(
    int InputCandidateCount,
    int AcceptedCandidateCount,
    int RejectedCandidateCount,
    int HorizontalCandidateCount,
    int VerticalCandidateCount,
    int TickCount,
    int DividerCount,
    int GridOrFrameExclusionCount,
    TimeSpan Elapsed,
    IReadOnlyList<string> Warnings);

/// <summary>
/// Geometry-only axis result. It deliberately has no marker collection.
/// </summary>
public sealed record AxisGeometryResult(
    string CoordinateSpace,
    PlotPolygon PlotPolygon,
    AxisLineFit XAxis,
    AxisLineFit YAxis,
    IReadOnlyList<AxisTickGeometry> Ticks,
    IReadOnlyList<PhaseDividerGeometry> PhaseDividers,
    IReadOnlyList<AmbiguousGridOrDividerGeometry> AmbiguousGridOrDividers,
    double Confidence,
    AxisGeometryUncertainty Uncertainty,
    AxisGeometryDiagnostics Diagnostics);

public interface IAxisGeometryDetector
{
    ValueTask<AxisGeometryResult> DetectAsync(
        AxisGeometryRequest request,
        CancellationToken cancellationToken = default);

    ValueTask<AxisGeometryResult> DetectAsync(
        GrayscaleLineCandidateFrame frame,
        ILineCandidateProvider candidateProvider,
        AxisGeometryOptions? options = null,
        CancellationToken cancellationToken = default);
}

public sealed class AxisGeometryDetectionException : InvalidOperationException
{
    public AxisGeometryDetectionException(string technicalMessage)
        : base(technicalMessage)
    {
        Code = "AXIS_GEOMETRY_NOT_FOUND";
        UserMessageKey = "Errors.AxisGeometryNotFound";
        Recoverable = true;
        SuggestedAction = "select_manual_calibration";
    }

    public string Code { get; }

    public string UserMessageKey { get; }

    public bool Recoverable { get; }

    public string SuggestedAction { get; }
}
