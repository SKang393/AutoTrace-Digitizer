// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;

namespace GraphReader.Pdf;

public static class PdfImportContract
{
    public const int Version = 1;
    public const string PagePointSpace = "page_points";
    public const string PagePixelSpace = "page_pixels";
    public const string PanelPixelSpace = "panel_pixels";
}

public enum PdfFailureSeverity { Warning, Error }

public enum PdfTextRole { Unknown, Body, Caption, ParticipantLabel, AxisTitle, Legend, PhaseHeader }

public enum PdfFigureSourceKind { EmbeddedImage, RenderedPage, VectorPageRegion }

public enum PdfSemanticField { IndependentVariable, DependentVariable }

public enum PdfSuggestionReviewState { Suggested, ConfirmedByUser, RejectedByUser }

public enum PdfPanelEvidenceKind
{
    EmbeddedImage,
    CaptionProximity,
    DenseLineStructure,
    HorizontalAxis,
    VerticalAxis,
    WhitespaceValley,
    RepeatedAxes,
    AlignedPlotWidths,
    AlignedYAxisColumns,
    ParticipantLabel,
    SharedDivider,
    Manual,
}

public readonly record struct PdfPointD(double X, double Y)
{
    public bool IsFinite => double.IsFinite(X) && double.IsFinite(Y);
}

public readonly record struct PdfRectD(double X, double Y, double Width, double Height)
{
    public double Right => X + Width;
    public double Bottom => Y + Height;
    public double Area => Width * Height;
    public bool IsValid => double.IsFinite(X) && double.IsFinite(Y) &&
        double.IsFinite(Width) && double.IsFinite(Height) && Width > 0d && Height > 0d;
    public bool Contains(PdfRectD other) => other.X >= X && other.Y >= Y &&
        other.Right <= Right && other.Bottom <= Bottom;
}

public readonly record struct PdfQuadrilateralD(
    PdfPointD TopLeft,
    PdfPointD TopRight,
    PdfPointD BottomRight,
    PdfPointD BottomLeft)
{
    public PdfRectD Bounds
    {
        get
        {
            double minimumX = Math.Min(Math.Min(TopLeft.X, TopRight.X), Math.Min(BottomRight.X, BottomLeft.X));
            double minimumY = Math.Min(Math.Min(TopLeft.Y, TopRight.Y), Math.Min(BottomRight.Y, BottomLeft.Y));
            double maximumX = Math.Max(Math.Max(TopLeft.X, TopRight.X), Math.Max(BottomRight.X, BottomLeft.X));
            double maximumY = Math.Max(Math.Max(TopLeft.Y, TopRight.Y), Math.Max(BottomRight.Y, BottomLeft.Y));
            return new PdfRectD(minimumX, minimumY, maximumX - minimumX, maximumY - minimumY);
        }
    }

    public bool IsFinite =>
        TopLeft.IsFinite && TopRight.IsFinite && BottomRight.IsFinite && BottomLeft.IsFinite;
}

/// <summary>Immutable PDF-order affine transform with an explicit inverse.</summary>
public readonly record struct PdfAffineTransform(
    double A,
    double B,
    double C,
    double D,
    double E,
    double F)
{
    public static PdfAffineTransform Identity { get; } = new(1d, 0d, 0d, 1d, 0d, 0d);

    public double Determinant => (A * D) - (B * C);
    public bool IsInvertible =>
        double.IsFinite(A) && double.IsFinite(B) && double.IsFinite(C) &&
        double.IsFinite(D) && double.IsFinite(E) && double.IsFinite(F) &&
        Math.Abs(Determinant) > 1e-12d;

    public PdfPointD Transform(PdfPointD value)
    {
        if (!value.IsFinite || !IsInvertible)
        {
            throw new InvalidOperationException("The affine transform and point must be finite and invertible.");
        }

        return new PdfPointD(
            (A * value.X) + (C * value.Y) + E,
            (B * value.X) + (D * value.Y) + F);
    }

    public PdfQuadrilateralD Transform(PdfRectD value)
    {
        if (!value.IsValid)
        {
            throw new InvalidOperationException("The rectangle must be finite and non-empty.");
        }

        return new PdfQuadrilateralD(
            Transform(new PdfPointD(value.X, value.Y)),
            Transform(new PdfPointD(value.Right, value.Y)),
            Transform(new PdfPointD(value.Right, value.Bottom)),
            Transform(new PdfPointD(value.X, value.Bottom)));
    }

    public PdfQuadrilateralD Transform(PdfQuadrilateralD value)
    {
        if (!value.IsFinite)
        {
            throw new InvalidOperationException("The quadrilateral must be finite.");
        }

        return new PdfQuadrilateralD(
            Transform(value.TopLeft),
            Transform(value.TopRight),
            Transform(value.BottomRight),
            Transform(value.BottomLeft));
    }

    public PdfAffineTransform Inverse()
    {
        if (!IsInvertible)
        {
            throw new InvalidOperationException("The affine transform must be finite and invertible.");
        }

        double inverseDeterminant = 1d / Determinant;
        return new PdfAffineTransform(
            D * inverseDeterminant,
            -B * inverseDeterminant,
            -C * inverseDeterminant,
            A * inverseDeterminant,
            ((C * F) - (D * E)) * inverseDeterminant,
            ((B * E) - (A * F)) * inverseDeterminant);
    }
}

/// <summary>Maps original lower-left PDF points to rendered top-left page pixels and back.</summary>
public sealed class PdfPageCoordinateTransform
{
    public PdfPageCoordinateTransform(
        double PageWidthPoints,
        double PageHeightPoints,
        int PixelWidth,
        int PixelHeight)
        : this(new PdfRectD(0d, 0d, PageWidthPoints, PageHeightPoints), 0, PixelWidth, PixelHeight)
    {
    }

    public PdfPageCoordinateTransform(
        PdfRectD visiblePageBoundsPoints,
        int rotationDegrees,
        int pixelWidth,
        int pixelHeight)
    {
        if (!visiblePageBoundsPoints.IsValid || pixelWidth <= 0 || pixelHeight <= 0 ||
            rotationDegrees is not (0 or 90 or 180 or 270))
        {
            throw new ArgumentOutOfRangeException(nameof(visiblePageBoundsPoints));
        }

        VisiblePageBoundsPoints = visiblePageBoundsPoints;
        RotationDegrees = rotationDegrees;
        PixelWidth = pixelWidth;
        PixelHeight = pixelHeight;
        PageWidthPoints = rotationDegrees is 90 or 270
            ? visiblePageBoundsPoints.Height
            : visiblePageBoundsPoints.Width;
        PageHeightPoints = rotationDegrees is 90 or 270
            ? visiblePageBoundsPoints.Width
            : visiblePageBoundsPoints.Height;
        PagePointsToPagePixelsMatrix = CreateMatrix(
            visiblePageBoundsPoints,
            rotationDegrees,
            pixelWidth,
            pixelHeight);
        PagePixelsToPagePointsMatrix = PagePointsToPagePixelsMatrix.Inverse();
    }

    public PdfPageCoordinateTransform(
        PdfRectD visiblePageBoundsPoints,
        int rotationDegrees,
        int pixelWidth,
        int pixelHeight,
        PdfAffineTransform pagePointsToPagePixelsMatrix)
    {
        if (!visiblePageBoundsPoints.IsValid || pixelWidth <= 0 || pixelHeight <= 0 ||
            !pagePointsToPagePixelsMatrix.IsInvertible)
        {
            throw new ArgumentOutOfRangeException(nameof(visiblePageBoundsPoints));
        }

        VisiblePageBoundsPoints = visiblePageBoundsPoints;
        RotationDegrees = rotationDegrees;
        PixelWidth = pixelWidth;
        PixelHeight = pixelHeight;
        PageWidthPoints = visiblePageBoundsPoints.Width;
        PageHeightPoints = visiblePageBoundsPoints.Height;
        PagePointsToPagePixelsMatrix = pagePointsToPagePixelsMatrix;
        PagePixelsToPagePointsMatrix = pagePointsToPagePixelsMatrix.Inverse();
    }

    public double PageWidthPoints { get; }
    public double PageHeightPoints { get; }
    public int PixelWidth { get; }
    public int PixelHeight { get; }
    public PdfRectD VisiblePageBoundsPoints { get; }
    public int RotationDegrees { get; }
    public PdfAffineTransform PagePointsToPagePixelsMatrix { get; }
    public PdfAffineTransform PagePixelsToPagePointsMatrix { get; }
    public double ScaleX => Math.Sqrt(
        (PagePointsToPagePixelsMatrix.A * PagePointsToPagePixelsMatrix.A) +
        (PagePointsToPagePixelsMatrix.B * PagePointsToPagePixelsMatrix.B));
    public double ScaleY => Math.Sqrt(
        (PagePointsToPagePixelsMatrix.C * PagePointsToPagePixelsMatrix.C) +
        (PagePointsToPagePixelsMatrix.D * PagePointsToPagePixelsMatrix.D));
    public bool IsValid =>
        VisiblePageBoundsPoints.IsValid && PixelWidth > 0 && PixelHeight > 0 &&
        PagePointsToPagePixelsMatrix.IsInvertible;

    public PdfRectD PagePointsToPixels(PdfRectD value)
    {
        EnsureValid(value);
        return PagePointsToPixelQuadrilateral(value).Bounds;
    }

    public PdfRectD PagePixelsToPoints(PdfRectD value)
    {
        EnsureValid(value);
        return PagePixelsToPointQuadrilateral(value).Bounds;
    }

    public PdfQuadrilateralD PagePointsToPixelQuadrilateral(PdfRectD value)
    {
        EnsureValid(value);
        return PagePointsToPagePixelsMatrix.Transform(new PdfQuadrilateralD(
            new PdfPointD(value.X, value.Bottom),
            new PdfPointD(value.Right, value.Bottom),
            new PdfPointD(value.Right, value.Y),
            new PdfPointD(value.X, value.Y)));
    }

    public PdfQuadrilateralD PagePixelsToPointQuadrilateral(PdfRectD value) =>
        PagePixelsToPagePointsMatrix.Transform(value);

    public PdfQuadrilateralD PagePointsToPixels(PdfQuadrilateralD value) =>
        PagePointsToPagePixelsMatrix.Transform(value);

    public PdfQuadrilateralD PagePixelsToPoints(PdfQuadrilateralD value) =>
        PagePixelsToPagePointsMatrix.Transform(value);

    private void EnsureValid(PdfRectD value)
    {
        if (!IsValid || !value.IsValid)
        {
            throw new InvalidOperationException("The page transform and rectangle must be valid.");
        }
    }

    private static PdfAffineTransform CreateMatrix(
        PdfRectD bounds,
        int rotationDegrees,
        int pixelWidth,
        int pixelHeight)
    {
        double xScale = pixelWidth /
            (rotationDegrees is 90 or 270 ? bounds.Height : bounds.Width);
        double yScale = pixelHeight /
            (rotationDegrees is 90 or 270 ? bounds.Width : bounds.Height);
        return rotationDegrees switch
        {
            0 => new PdfAffineTransform(
                xScale, 0d, 0d, -yScale,
                -bounds.X * xScale,
                bounds.Bottom * yScale),
            90 => new PdfAffineTransform(
                0d, yScale, xScale, 0d,
                -bounds.Y * xScale,
                -bounds.X * yScale),
            180 => new PdfAffineTransform(
                -xScale, 0d, 0d, yScale,
                bounds.Right * xScale,
                -bounds.Y * yScale),
            270 => new PdfAffineTransform(
                0d, -yScale, -xScale, 0d,
                bounds.Bottom * xScale,
                bounds.Right * yScale),
            _ => throw new ArgumentOutOfRangeException(nameof(rotationDegrees)),
        };
    }
}

public sealed record PdfDocumentMetadata(
    string? Title,
    string? Author,
    string? Subject,
    string? Keywords,
    DateTimeOffset? CreatedAt,
    DateTimeOffset? ModifiedAt);

public sealed record PdfTextBlock(
    Guid BlockId,
    string Text,
    PdfRectD BoundsPagePoints,
    PdfTextRole Role,
    double Confidence);

public sealed record PdfVectorLine(PdfPointD StartPagePoints, PdfPointD EndPagePoints, double WidthPoints)
{
    public bool IsHorizontal(double tolerancePoints = 1d) =>
        Math.Abs(StartPagePoints.Y - EndPagePoints.Y) <= tolerancePoints;
    public bool IsVertical(double tolerancePoints = 1d) =>
        Math.Abs(StartPagePoints.X - EndPagePoints.X) <= tolerancePoints;
}

public sealed class PdfEmbeddedImage
{
    public PdfEmbeddedImage(
        Guid imageId,
        PdfRectD boundsPagePoints,
        int pixelWidth,
        int pixelHeight,
        string mediaType,
        ImmutableByteBuffer encodedBytes,
        string sha256,
        PdfAffineTransform? sourcePixelsToPagePoints = null)
    {
        ImageId = imageId;
        BoundsPagePoints = boundsPagePoints;
        PixelWidth = pixelWidth;
        PixelHeight = pixelHeight;
        MediaType = mediaType ?? throw new ArgumentNullException(nameof(mediaType));
        EncodedBytes = encodedBytes ?? throw new ArgumentNullException(nameof(encodedBytes));
        Sha256 = sha256 ?? throw new ArgumentNullException(nameof(sha256));
        SourcePixelsToPagePoints = sourcePixelsToPagePoints ?? CreateAxisAlignedImageTransform(
            boundsPagePoints,
            pixelWidth,
            pixelHeight);
        if (!SourcePixelsToPagePoints.IsInvertible)
        {
            throw new ArgumentException("The image placement transform must be invertible.", nameof(sourcePixelsToPagePoints));
        }
    }

    public Guid ImageId { get; }
    public PdfRectD BoundsPagePoints { get; }
    public int PixelWidth { get; }
    public int PixelHeight { get; }
    public string MediaType { get; }
    public ImmutableByteBuffer EncodedBytes { get; }
    public string Sha256 { get; }
    public PdfAffineTransform SourcePixelsToPagePoints { get; }
    public PdfAffineTransform PagePointsToSourcePixels => SourcePixelsToPagePoints.Inverse();

    private static PdfAffineTransform CreateAxisAlignedImageTransform(
        PdfRectD bounds,
        int pixelWidth,
        int pixelHeight)
    {
        if (!bounds.IsValid || pixelWidth <= 0 || pixelHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(bounds));
        }

        return new PdfAffineTransform(
            bounds.Width / pixelWidth,
            0d,
            0d,
            -bounds.Height / pixelHeight,
            bounds.X,
            bounds.Bottom);
    }
}

public sealed class PdfPageSnapshot
{
    public PdfPageSnapshot(
        int pageNumber,
        double widthPoints,
        double heightPoints,
        IEnumerable<PdfTextBlock> textBlocks,
        IEnumerable<PdfEmbeddedImage> embeddedImages,
        IEnumerable<PdfVectorLine> vectorLines,
        PdfRectD? originalVisibleBoundsPoints = null,
        int rotationDegrees = 0,
        PdfAffineTransform? normalizedToOriginalPagePoints = null)
    {
        PageNumber = pageNumber;
        WidthPoints = widthPoints;
        HeightPoints = heightPoints;
        TextBlocks = PdfCollections.Freeze(textBlocks);
        EmbeddedImages = PdfCollections.Freeze(embeddedImages);
        VectorLines = PdfCollections.Freeze(vectorLines);
        OriginalVisibleBoundsPoints = originalVisibleBoundsPoints ?? new PdfRectD(0d, 0d, widthPoints, heightPoints);
        RotationDegrees = rotationDegrees;
        NormalizedToOriginalPagePoints = normalizedToOriginalPagePoints ?? PdfAffineTransform.Identity;
        if (!OriginalVisibleBoundsPoints.IsValid || rotationDegrees is not (0 or 90 or 180 or 270) ||
            !NormalizedToOriginalPagePoints.IsInvertible)
        {
            throw new ArgumentException("The page coordinate geometry must be finite and invertible.");
        }
    }

    public int PageNumber { get; }
    public double WidthPoints { get; }
    public double HeightPoints { get; }
    public IReadOnlyList<PdfTextBlock> TextBlocks { get; }
    public IReadOnlyList<PdfEmbeddedImage> EmbeddedImages { get; }
    public IReadOnlyList<PdfVectorLine> VectorLines { get; }
    public PdfRectD OriginalVisibleBoundsPoints { get; }
    public int RotationDegrees { get; }
    public PdfAffineTransform NormalizedToOriginalPagePoints { get; }
    public PdfAffineTransform OriginalToNormalizedPagePoints => NormalizedToOriginalPagePoints.Inverse();
    public bool HasExtractableText => TextBlocks.Count > 0;
}

public sealed class PdfDocumentSnapshot
{
    public PdfDocumentSnapshot(
        string documentSha256,
        PdfDocumentMetadata metadata,
        IEnumerable<PdfPageSnapshot> pages)
    {
        DocumentSha256 = documentSha256 ?? throw new ArgumentNullException(nameof(documentSha256));
        Metadata = metadata ?? throw new ArgumentNullException(nameof(metadata));
        Pages = PdfCollections.Freeze(pages);
    }

    public string DocumentSha256 { get; }
    public PdfDocumentMetadata Metadata { get; }
    public IReadOnlyList<PdfPageSnapshot> Pages { get; }
}

public sealed record PdfFailure(
    string Code,
    PdfFailureSeverity Severity,
    string UserMessageKey,
    string TechnicalMessage,
    bool Recoverable,
    string SuggestedAction,
    int? PageNumber = null);

public sealed record PdfInspectionTiming(double OpenMilliseconds, double ExtractMilliseconds, double TotalMilliseconds);

public sealed record PdfInspectionRequest(
    ImmutableByteBuffer PdfBytes,
    string SourceDisplayName,
    string? Password = null,
    int ContractVersion = PdfImportContract.Version);

public sealed class PdfInspectionResult
{
    public PdfInspectionResult(
        PdfDocumentSnapshot? document,
        IEnumerable<PdfFailure> failures,
        PdfInspectionTiming timing)
    {
        Document = document;
        Failures = PdfCollections.Freeze(failures);
        Timing = timing;
    }

    public PdfDocumentSnapshot? Document { get; }
    public IReadOnlyList<PdfFailure> Failures { get; }
    public PdfInspectionTiming Timing { get; }
    public bool Succeeded => Document is not null &&
        Failures.All(static failure => failure.Severity != PdfFailureSeverity.Error);
}

public interface IPdfDocumentInspector
{
    Task<PdfInspectionResult> InspectAsync(PdfInspectionRequest request, CancellationToken cancellationToken);
}

public sealed record PdfPanelEvidence(PdfPanelEvidenceKind Kind, double Weight, string Detail);

public sealed record PdfSemanticSuggestion(
    PdfSemanticField Field,
    string Value,
    string SourceText,
    double Confidence,
    PdfSuggestionReviewState ReviewState = PdfSuggestionReviewState.Suggested);

public sealed class PdfFigureCandidate
{
    public PdfFigureCandidate(
        Guid figureId,
        int pageNumber,
        PdfFigureSourceKind sourceKind,
        Guid? embeddedImageId,
        PdfRectD boundsPagePixels,
        PdfRectD boundsPagePoints,
        int sourcePixelWidth,
        int sourcePixelHeight,
        ImmutableByteBuffer? encodedSource,
        string? mediaType,
        string? caption,
        IEnumerable<PdfPanelEvidence> evidence,
        double confidence,
        PdfAffineTransform? pagePointsToPagePixels = null,
        PdfAffineTransform? sourcePixelsToPagePoints = null)
    {
        FigureId = figureId;
        PageNumber = pageNumber;
        SourceKind = sourceKind;
        EmbeddedImageId = embeddedImageId;
        BoundsPagePixels = boundsPagePixels;
        BoundsPagePoints = boundsPagePoints;
        SourcePixelWidth = sourcePixelWidth;
        SourcePixelHeight = sourcePixelHeight;
        EncodedSource = encodedSource;
        MediaType = mediaType;
        Caption = caption;
        Evidence = PdfCollections.Freeze(evidence);
        Confidence = confidence;
        PagePointsToPagePixels = pagePointsToPagePixels ?? CreateAxisAlignedTransform(
            boundsPagePoints,
            boundsPagePixels);
        SourcePixelsToPagePoints = sourcePixelsToPagePoints ?? CreateAxisAlignedSourceTransform(
            boundsPagePoints,
            sourcePixelWidth,
            sourcePixelHeight);
        if (!PagePointsToPagePixels.IsInvertible || !SourcePixelsToPagePoints.IsInvertible)
        {
            throw new ArgumentException("Figure coordinate transforms must be finite and invertible.");
        }
    }

    public Guid FigureId { get; }
    public int PageNumber { get; }
    public PdfFigureSourceKind SourceKind { get; }
    public Guid? EmbeddedImageId { get; }
    public PdfRectD BoundsPagePixels { get; }
    public PdfRectD BoundsPagePoints { get; }
    public int SourcePixelWidth { get; }
    public int SourcePixelHeight { get; }
    public ImmutableByteBuffer? EncodedSource { get; }
    public string? MediaType { get; }
    public string? Caption { get; }
    public IReadOnlyList<PdfPanelEvidence> Evidence { get; }
    public double Confidence { get; }
    public PdfAffineTransform PagePointsToPagePixels { get; }
    public PdfAffineTransform PagePixelsToPagePoints => PagePointsToPagePixels.Inverse();
    public PdfAffineTransform SourcePixelsToPagePoints { get; }
    public PdfAffineTransform PagePointsToSourcePixels => SourcePixelsToPagePoints.Inverse();

    private static PdfAffineTransform CreateAxisAlignedTransform(PdfRectD source, PdfRectD target)
    {
        if (!source.IsValid || !target.IsValid)
        {
            throw new ArgumentOutOfRangeException(nameof(source));
        }

        double xScale = target.Width / source.Width;
        double yScale = target.Height / source.Height;
        return new PdfAffineTransform(
            xScale,
            0d,
            0d,
            -yScale,
            target.X - (source.X * xScale),
            target.Bottom + (source.Y * yScale));
    }

    private static PdfAffineTransform CreateAxisAlignedSourceTransform(
        PdfRectD bounds,
        int pixelWidth,
        int pixelHeight)
    {
        if (!bounds.IsValid || pixelWidth <= 0 || pixelHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(bounds));
        }

        return new PdfAffineTransform(
            bounds.Width / pixelWidth,
            0d,
            0d,
            -bounds.Height / pixelHeight,
            bounds.X,
            bounds.Bottom);
    }
}

public sealed class PdfPanelRecord
{
    public PdfPanelRecord(
        Guid panelId,
        Guid figureId,
        int pageNumber,
        int order,
        PdfRectD cropInSourcePixels,
        PdfRectD boundsPagePixels,
        PdfRectD boundsPagePoints,
        string? participantLabel,
        string? caption,
        IEnumerable<PdfSemanticSuggestion> semanticSuggestions,
        IEnumerable<PdfPanelEvidence> evidence,
        double confidence,
        PdfQuadrilateralD? cropInSourcePixelsQuadrilateral = null)
    {
        PanelId = panelId;
        FigureId = figureId;
        PageNumber = pageNumber;
        Order = order;
        CropInSourcePixels = cropInSourcePixels;
        BoundsPagePixels = boundsPagePixels;
        BoundsPagePoints = boundsPagePoints;
        ParticipantLabel = participantLabel;
        Caption = caption;
        SemanticSuggestions = PdfCollections.Freeze(semanticSuggestions);
        Evidence = PdfCollections.Freeze(evidence);
        Confidence = confidence;
        CropInSourcePixelsQuadrilateral = cropInSourcePixelsQuadrilateral ?? new PdfQuadrilateralD(
            new PdfPointD(cropInSourcePixels.X, cropInSourcePixels.Y),
            new PdfPointD(cropInSourcePixels.Right, cropInSourcePixels.Y),
            new PdfPointD(cropInSourcePixels.Right, cropInSourcePixels.Bottom),
            new PdfPointD(cropInSourcePixels.X, cropInSourcePixels.Bottom));
    }

    public Guid PanelId { get; }
    public Guid FigureId { get; }
    public int PageNumber { get; }
    public int Order { get; }
    public PdfRectD CropInSourcePixels { get; }
    public PdfRectD BoundsPagePixels { get; }
    public PdfRectD BoundsPagePoints { get; }
    public string? ParticipantLabel { get; }
    public string? Caption { get; }
    public IReadOnlyList<PdfSemanticSuggestion> SemanticSuggestions { get; }
    public IReadOnlyList<PdfPanelEvidence> Evidence { get; }
    public double Confidence { get; }
    public PdfQuadrilateralD CropInSourcePixelsQuadrilateral { get; }
}

public sealed record PdfPanelizationOptions(
    int RenderDpi = 144,
    double MinimumFigureConfidence = 0.55,
    double MinimumWhitespaceFraction = 0.035,
    int MaximumPanelsPerFigure = 12);

public sealed record PdfPanelizationInput(
    string DocumentSha256,
    PdfPageSnapshot Page,
    PdfRenderedPage? RenderedPage,
    PdfPanelizationOptions Options);

public sealed class PdfPanelizationResult
{
    public PdfPanelizationResult(
        IEnumerable<PdfFigureCandidate> figures,
        IEnumerable<PdfPanelRecord> panels,
        IEnumerable<PdfFailure>? failures = null,
        IEnumerable<string>? warnings = null,
        double elapsedMilliseconds = 0d)
    {
        Figures = PdfCollections.Freeze(figures);
        Panels = PdfCollections.Freeze(panels);
        Failures = PdfCollections.Freeze(failures ?? []);
        Warnings = PdfCollections.Freeze(warnings ?? []);
        ElapsedMilliseconds = elapsedMilliseconds;
    }

    public IReadOnlyList<PdfFigureCandidate> Figures { get; }
    public IReadOnlyList<PdfPanelRecord> Panels { get; }
    public IReadOnlyList<PdfFailure> Failures { get; }
    public IReadOnlyList<string> Warnings { get; }
    public double ElapsedMilliseconds { get; }
    public bool Succeeded => Failures.All(static failure => failure.Severity != PdfFailureSeverity.Error);
}

public sealed class PdfManualSplitCommand
{
    public PdfManualSplitCommand(Guid figureId, IEnumerable<double> horizontalBoundariesPagePixels)
    {
        FigureId = figureId;
        HorizontalBoundariesPagePixels = PdfCollections.Freeze(horizontalBoundariesPagePixels);
    }

    public Guid FigureId { get; }
    public IReadOnlyList<double> HorizontalBoundariesPagePixels { get; }
}

public sealed class PdfManualMergeCommand
{
    public PdfManualMergeCommand(IEnumerable<Guid> panelIds) =>
        PanelIds = PdfCollections.Freeze(panelIds);

    public IReadOnlyList<Guid> PanelIds { get; }
}

public interface IPdfPanelizationEngine
{
    Task<PdfPanelizationResult> ProposeAsync(
        PdfPanelizationInput input,
        CancellationToken cancellationToken);
    PdfPanelizationResult ApplySplit(PdfPanelizationResult current, PdfManualSplitCommand command);
    PdfPanelizationResult ApplyMerge(PdfPanelizationResult current, PdfManualMergeCommand command);
}

public sealed record PdfImportRequest(
    Guid RunId,
    Guid ProjectId,
    ImmutableByteBuffer PdfBytes,
    string SourceDisplayName,
    string? Password,
    PdfPanelizationOptions PanelizationOptions,
    int ContractVersion = PdfImportContract.Version);

public sealed record PdfImportTiming(
    double InspectionMilliseconds,
    double RenderingMilliseconds,
    double PanelizationMilliseconds,
    double TotalMilliseconds);

public sealed class PdfImportResult
{
    public PdfImportResult(
        Guid runId,
        Guid projectId,
        PdfDocumentSnapshot? document,
        IEnumerable<PdfFigureCandidate> figures,
        IEnumerable<PdfPanelRecord> panels,
        IEnumerable<PdfFailure> failures,
        IEnumerable<string> warnings,
        PdfImportTiming timing)
    {
        RunId = runId;
        ProjectId = projectId;
        Document = document;
        Figures = PdfCollections.Freeze(figures);
        Panels = PdfCollections.Freeze(panels);
        Failures = PdfCollections.Freeze(failures);
        Warnings = PdfCollections.Freeze(warnings);
        Timing = timing;
    }

    public Guid RunId { get; }
    public Guid ProjectId { get; }
    public PdfDocumentSnapshot? Document { get; }
    public IReadOnlyList<PdfFigureCandidate> Figures { get; }
    public IReadOnlyList<PdfPanelRecord> Panels { get; }
    public IReadOnlyList<PdfFailure> Failures { get; }
    public IReadOnlyList<string> Warnings { get; }
    public PdfImportTiming Timing { get; }
    public bool Succeeded => Document is not null &&
        Failures.All(static failure => failure.Severity != PdfFailureSeverity.Error);
}

public interface IPdfImportService
{
    Task<PdfImportResult> ImportAsync(PdfImportRequest request, CancellationToken cancellationToken);
}

internal static class PdfCollections
{
    public static IReadOnlyList<T> Freeze<T>(IEnumerable<T> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return new ReadOnlyCollection<T>(values.ToArray());
    }
}
