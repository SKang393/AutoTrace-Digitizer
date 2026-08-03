// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections;
using GraphReader.Inference;

namespace GraphReader.Markers.Detection;

public static class MarkerContract
{
    public const int Version = 1;
    public const string Stage = "markers";
    public const string CoordinateSpace = "original_pixels";
    public const string RuntimeRevision = "marker-center-runtime-v2";
    public const string InputChannelOrder = "ink_probability,text_mask,artifact_mask";
    public const string OutputChannelOrder = "center_probability,radius_pixels,artifact_probability";
    public const string GridAlignment = "half_pixel_centers";
}

public enum MarkerSourceImage
{
    Original,
    Enhanced,
    Consensus,
}

public enum MarkerTensorLayout
{
    ChannelsFirst,
    ChannelsLast,
}

public enum MarkerHeadActivation
{
    Identity,
    Sigmoid,
}

public enum MarkerReviewState
{
    Unreviewed,
    NeedsReview,
}

public enum MarkerDisagreementKind
{
    None,
    OriginalOnly,
    EnhancedOnly,
}

public readonly record struct MarkerPoint(double X, double Y)
{
    public bool IsFinite => double.IsFinite(X) && double.IsFinite(Y);
}

public readonly record struct MarkerRectangle(double X, double Y, double Width, double Height)
{
    public double Left => X;

    public double Top => Y;

    public double Right => X + Width;

    public double Bottom => Y + Height;

    public bool IsValid =>
        double.IsFinite(X) && double.IsFinite(Y) &&
        double.IsFinite(Width) && double.IsFinite(Height) &&
        Width > 0 && Height > 0;
}

/// <summary>
/// Maps immutable original-image pixels into a detector source frame.
/// </summary>
public readonly record struct MarkerAffineTransform(
    double M11,
    double M12,
    double M13,
    double M21,
    double M22,
    double M23)
{
    public static MarkerAffineTransform Identity { get; } = new(1, 0, 0, 0, 1, 0);

    public double Determinant => (M11 * M22) - (M12 * M21);

    public bool IsInvertible =>
        double.IsFinite(M11) && double.IsFinite(M12) && double.IsFinite(M13) &&
        double.IsFinite(M21) && double.IsFinite(M22) && double.IsFinite(M23) &&
        double.IsFinite(Determinant) && Math.Abs(Determinant) > 1e-12;

    public MarkerPoint MapFromOriginal(MarkerPoint point) =>
        new(
            (M11 * point.X) + (M12 * point.Y) + M13,
            (M21 * point.X) + (M22 * point.Y) + M23);

    public MarkerPoint MapToOriginal(MarkerPoint point)
    {
        if (!IsInvertible)
        {
            throw new InvalidOperationException("Marker frame transform is not invertible.");
        }

        var translatedX = point.X - M13;
        var translatedY = point.Y - M23;
        return new MarkerPoint(
            ((M22 * translatedX) - (M12 * translatedY)) / Determinant,
            ((-M21 * translatedX) + (M11 * translatedY)) / Determinant);
    }

    public double MapFrameRadiusToOriginal(double radius)
    {
        if (!IsInvertible || !double.IsFinite(radius) || radius < 0)
        {
            throw new InvalidOperationException("A finite radius and invertible marker transform are required.");
        }

        var inverseScaleX = Math.Sqrt((M22 * M22) + (M21 * M21)) / Math.Abs(Determinant);
        var inverseScaleY = Math.Sqrt((M12 * M12) + (M11 * M11)) / Math.Abs(Determinant);
        return radius * Math.Sqrt(inverseScaleX * inverseScaleY);
    }

    public string ToCacheMaterial() => string.Join(
        ',',
        M11.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        M12.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        M13.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        M21.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        M22.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        M23.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
}

public sealed record MarkerPolygon
{
    public MarkerPolygon(IReadOnlyList<MarkerPoint> points)
    {
        ArgumentNullException.ThrowIfNull(points);
        if (points.Count < 3 || points.Any(static point => !point.IsFinite))
        {
            throw new ArgumentException("A marker polygon requires at least three finite points.", nameof(points));
        }

        Points = MarkerCollections.Freeze(points);
    }

    public IReadOnlyList<MarkerPoint> Points { get; }

    public MarkerRectangle Bounds
    {
        get
        {
            var minimumX = Points.Min(static point => point.X);
            var maximumX = Points.Max(static point => point.X);
            var minimumY = Points.Min(static point => point.Y);
            var maximumY = Points.Max(static point => point.Y);
            return new MarkerRectangle(minimumX, minimumY, maximumX - minimumX, maximumY - minimumY);
        }
    }

    public bool Contains(MarkerPoint point)
    {
        if (!point.IsFinite)
        {
            return false;
        }

        var inside = false;
        for (var current = 0; current < Points.Count; current++)
        {
            var previous = current == 0 ? Points.Count - 1 : current - 1;
            var a = Points[current];
            var b = Points[previous];
            var crosses = (a.Y > point.Y) != (b.Y > point.Y);
            if (crosses && point.X < ((b.X - a.X) * (point.Y - a.Y) / (b.Y - a.Y)) + a.X)
            {
                inside = !inside;
            }
        }

        return inside;
    }

    public static MarkerPolygon FromRectangle(MarkerRectangle rectangle)
    {
        if (!rectangle.IsValid)
        {
            throw new ArgumentException("Marker rectangle must be finite and have positive dimensions.", nameof(rectangle));
        }

        return new MarkerPolygon([
            new MarkerPoint(rectangle.Left, rectangle.Top),
            new MarkerPoint(rectangle.Right, rectangle.Top),
            new MarkerPoint(rectangle.Right, rectangle.Bottom),
            new MarkerPoint(rectangle.Left, rectangle.Bottom),
        ]);
    }
}

public sealed record MarkerMask(
    int Width,
    int Height,
    ReadOnlyMemory<float> Values)
{
    public static MarkerMask Empty(int width, int height) =>
        new(width, height, new float[checked(width * height)]);
}

public sealed record MarkerImageFrame(
    int Width,
    int Height,
    int ChannelCount,
    ReadOnlyMemory<float> ChannelsFirstPixels,
    MarkerSourceImage SourceImage,
    MarkerAffineTransform OriginalToFrame,
    MarkerMask OcrMask,
    MarkerMask ArtifactMask);

/// <summary>
/// Explicit model tensor metadata. Runtime v2 accepts only one NCHW input
/// [1,3,H,W] and one flattened NCHW output [1,3,H,W]. Radius values are pixels.
/// </summary>
public sealed record MarkerModelTensorContract(
    string InputName,
    string OutputName,
    int InputWidth,
    int InputHeight,
    int InputChannelCount,
    MarkerTensorLayout InputLayout,
    int OutputWidth,
    int OutputHeight,
    int OutputChannelCount,
    MarkerTensorLayout OutputLayout,
    int CenterChannelIndex,
    int RadiusChannelIndex,
    int ArtifactChannelIndex,
    MarkerHeadActivation CenterActivation,
    MarkerHeadActivation ArtifactActivation,
    float RadiusScale,
    float NormalizeMean,
    float NormalizeScale);

public sealed record MarkerDetectionOptions(MarkerModelTensorContract TensorContract)
{
    public float CenterThreshold { get; init; } = 0.36f;

    public float ArtifactThreshold { get; init; } = 0.35f;

    public float MaskThreshold { get; init; } = 0.5f;

    public int LocalMaximumWindow { get; init; } = 9;

    public double MinimumRadiusGridPixels { get; init; } = 2.5;

    public double MinimumSuppressionDistanceGridPixels { get; init; } = 5;

    public double RadiusSuppressionScale { get; init; } = 1.25;

    public double ConsensusToleranceOriginalPixels { get; init; } = 5;

    public double UnmatchedSourceConfidenceScale { get; init; } = 0.75;

    public string StageVersion { get; init; } = "0.2.0";

    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);
}

public sealed record MarkerDetectionRequest(
    string ProjectId,
    string PanelId,
    string InputSha256,
    ModelIdentity Model,
    MarkerImageFrame OriginalImage,
    MarkerPolygon PlotPolygon,
    MarkerDetectionOptions Options,
    MarkerImageFrame? EnhancedImage = null,
    int ContractVersion = MarkerContract.Version,
    string TransformChain = "identity");

public sealed record MarkerCenter(
    string MarkerId,
    MarkerPoint Center,
    double Radius,
    double ArtifactProbability,
    double CenterConfidence,
    MarkerSourceImage SourceImage,
    string CoordinateSpace = MarkerContract.CoordinateSpace,
    MarkerReviewState ReviewState = MarkerReviewState.Unreviewed,
    MarkerDisagreementKind Disagreement = MarkerDisagreementKind.None);

public sealed record MarkerDetectionTiming(
    double PreprocessMilliseconds,
    double InferenceMilliseconds,
    double PostprocessMilliseconds,
    double TotalMilliseconds);

public sealed record MarkerDetectionFailure(
    string Code,
    string Severity,
    string UserMessageKey,
    string TechnicalMessage,
    bool Recoverable,
    string SuggestedAction);

public sealed record MarkerFrameReport(
    MarkerSourceImage SourceImage,
    string CacheKey,
    InferenceProvider? Provider,
    IReadOnlyList<ProviderAttempt> ProviderAttempts,
    MarkerDetectionTiming Timing,
    int RawCandidateCount,
    int AcceptedCandidateCount,
    bool CacheHit,
    MarkerDetectionFailure? Failure);

public sealed record MarkerModelReport(
    string ModelId,
    string Version,
    string Sha256,
    InferenceProvider? Provider);

public sealed record MarkerDetectionResult(
    int ContractVersion,
    string RunId,
    string ProjectId,
    string PanelId,
    string Stage,
    string StageVersion,
    string InputSha256,
    string CoordinateSpace,
    IReadOnlyList<MarkerCenter> Markers,
    MarkerDetectionTiming Timing,
    double Confidence,
    IReadOnlyList<string> Warnings,
    IReadOnlyList<MarkerFrameReport> Frames,
    MarkerModelReport Model,
    MarkerDetectionFailure? Failure)
{
    public bool Succeeded => Failure is null;
}

public interface IMarkerDetectionService
{
    ValueTask<MarkerDetectionResult> DetectAsync(
        MarkerDetectionRequest request,
        CancellationToken cancellationToken);
}

public interface IMarkerInferenceRunner
{
    ValueTask<InferenceResponse> RunAsync(
        InferenceRequest request,
        CancellationToken cancellationToken);
}

internal static class MarkerCollections
{
    public static IReadOnlyList<T> Freeze<T>(IEnumerable<T> values) => new FrozenList<T>(values);

    private sealed class FrozenList<T> : IReadOnlyList<T>, IEquatable<FrozenList<T>>
    {
        private readonly T[] _items;

        public FrozenList(IEnumerable<T> items) => _items = items.ToArray();

        public int Count => _items.Length;

        public T this[int index] => _items[index];

        public bool Equals(FrozenList<T>? other) =>
            other is not null && _items.SequenceEqual(other._items);

        public override bool Equals(object? obj) =>
            obj is IEnumerable<T> items && _items.SequenceEqual(items);

        public override int GetHashCode()
        {
            var hash = new HashCode();
            foreach (var item in _items)
            {
                hash.Add(item);
            }

            return hash.ToHashCode();
        }

        public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)_items).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => _items.GetEnumerator();
    }
}
