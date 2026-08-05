// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Security.Cryptography;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using GraphReader.Axis;
using GraphReader.Markers.Detection;
using GraphReader.Ocr;

namespace GraphReader.App.Integration.Workflow;

public interface IProductionRasterFrameDecoder
{
    ProductionDecodedRaster Decode(
        ProductionWorkflowDetectionRequest request,
        CancellationToken cancellationToken);
}

/// <summary>
/// Owns a checksum-verified grayscale decode and creates isolated stage inputs.
/// Marker masks are always supplied by upstream evidence and are never inferred
/// or silently replaced by this boundary.
/// </summary>
public sealed class ProductionDecodedRaster
{
    private readonly byte[] grayscalePixels;
    private readonly float[] normalizedLuminance;

    internal ProductionDecodedRaster(
        int width,
        int height,
        string inputSha256,
        WorkflowImageVariant variant,
        MarkerAffineTransform originalToFrame,
        OcrFrameTransform originalToImage,
        int canonicalOriginalWidth,
        int canonicalOriginalHeight,
        byte[] grayscalePixels,
        float[] normalizedLuminance)
    {
        Width = width;
        Height = height;
        InputSha256 = inputSha256;
        Variant = variant;
        OriginalToFrame = originalToFrame;
        OriginalToImage = originalToImage;
        CanonicalOriginalWidth = canonicalOriginalWidth;
        CanonicalOriginalHeight = canonicalOriginalHeight;
        this.grayscalePixels = (byte[])grayscalePixels.Clone();
        this.normalizedLuminance = (float[])normalizedLuminance.Clone();
    }

    public int Width { get; }

    public int Height { get; }

    public string InputSha256 { get; }

    public WorkflowImageVariant Variant { get; }

    public MarkerAffineTransform OriginalToFrame { get; }

    public OcrFrameTransform OriginalToImage { get; }

    public int CanonicalOriginalWidth { get; }

    public int CanonicalOriginalHeight { get; }

    public GrayscaleLineCandidateFrame CreateAxisFrame()
    {
        if (Variant != WorkflowImageVariant.Original ||
            OriginalToFrame != MarkerAffineTransform.Identity)
        {
            throw Failure(
                "Axis input must be the immutable original raster with an identity transform.");
        }

        return new GrayscaleLineCandidateFrame(
            Width,
            Height,
            Width,
            (byte[])grayscalePixels.Clone());
    }

    public OcrImage CreateOcrImage() => new(
        Width,
        Height,
        Width,
        (byte[])grayscalePixels.Clone(),
        Variant == WorkflowImageVariant.Original
            ? OcrSourceImage.Original
            : OcrSourceImage.Enhanced,
        OriginalToImage,
        OcrContract.CoordinateSpace,
        CanonicalOriginalWidth,
        CanonicalOriginalHeight);

    public MarkerImageFrame CreateMarkerFrame(
        MarkerMask ocrMask,
        MarkerMask artifactMask)
    {
        ArgumentNullException.ThrowIfNull(ocrMask);
        ArgumentNullException.ThrowIfNull(artifactMask);
        ValidateMask(ocrMask, nameof(ocrMask));
        ValidateMask(artifactMask, nameof(artifactMask));
        return new MarkerImageFrame(
            Width,
            Height,
            ChannelCount: 1,
            (float[])normalizedLuminance.Clone(),
            Variant == WorkflowImageVariant.Original
                ? MarkerSourceImage.Original
                : MarkerSourceImage.Enhanced,
            OriginalToFrame,
            new MarkerMask(Width, Height, ocrMask.Values.ToArray()),
            new MarkerMask(Width, Height, artifactMask.Values.ToArray()));
    }

    private void ValidateMask(MarkerMask mask, string parameterName)
    {
        int pixelCount = checked(Width * Height);
        if (mask.Width != Width || mask.Height != Height || mask.Values.Length != pixelCount ||
            !IsNormalized(mask.Values.Span))
        {
            throw Failure(
                $"Marker mask '{parameterName}' does not match the decoded raster or contains values outside [0,1].");
        }
    }

    private static bool IsNormalized(ReadOnlySpan<float> values)
    {
        foreach (float value in values)
        {
            if (!float.IsFinite(value) || value is < 0 or > 1)
            {
                return false;
            }
        }

        return true;
    }

    private static ProductionWorkflowStageException Failure(string technicalMessage) =>
        new(new ProductionWorkflowFailure(
            ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
            "Errors.DetectionEvidenceRejected",
            technicalMessage,
            Recoverable: true,
            "Regenerate checksum-bound raster and mask evidence from the retained source."));
}

public sealed class ProductionRasterFrameDecoder : IProductionRasterFrameDecoder
{
    private const double MatrixTolerance = 1e-9;

    public ProductionDecodedRaster Decode(
        ProductionWorkflowDetectionRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        ValidateIdentity(request);

        byte[] encodedBytes = request.CopyImageBytes();
        string actualSha256 = Convert.ToHexStringLower(SHA256.HashData(encodedBytes));
        if (!string.Equals(actualSha256, request.Image.Sha256, StringComparison.Ordinal))
        {
            throw Failure(
                "Decoded raster input bytes do not match the retained immutable checksum.",
                "Errors.SourceChanged");
        }

        byte[] grayscalePixels;
        try
        {
            grayscalePixels = DecodeGray8(
                encodedBytes,
                request.Image.Width,
                request.Image.Height);
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidDataException or
            IOException or NotSupportedException or FileFormatException)
        {
            throw Failure($"Raster decoding failed: {exception.Message}");
        }

        cancellationToken.ThrowIfCancellationRequested();
        var normalizedLuminance = new float[grayscalePixels.Length];
        for (int index = 0; index < grayscalePixels.Length; index++)
        {
            if ((index & 0x3fff) == 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }

            normalizedLuminance[index] = grayscalePixels[index] / 255f;
        }

        (MarkerAffineTransform markerTransform, OcrFrameTransform ocrTransform) =
            ResolveTransforms(request);
        return new ProductionDecodedRaster(
            request.Image.Width,
            request.Image.Height,
            request.Image.Sha256,
            request.ImageVariant,
            markerTransform,
            ocrTransform,
            request.Panel.Original.Width,
            request.Panel.Original.Height,
            grayscalePixels,
            normalizedLuminance);
    }

    private static void ValidateIdentity(ProductionWorkflowDetectionRequest request)
    {
        if (request.RunId == Guid.Empty || request.ProjectId == Guid.Empty ||
            request.Image.Width <= 0 || request.Image.Height <= 0 ||
            request.ImageVariant != request.Image.Variant)
        {
            throw Failure(
                "Raster evidence lacks a valid run, project, image variant, or dimensions.");
        }

        WorkflowImageEvidence? expected = request.ImageVariant switch
        {
            WorkflowImageVariant.Original => request.Panel.Original,
            WorkflowImageVariant.Enhanced => request.Panel.Enhanced,
            _ => null,
        };
        if (expected is null || expected.Width != request.Image.Width ||
            expected.Height != request.Image.Height ||
            !string.Equals(expected.Sha256, request.Image.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            throw Failure(
                "Raster evidence does not match the prepared panel's retained image identity.");
        }

        if ((request.ImageVariant == WorkflowImageVariant.Original && request.Transforms.Count != 0) ||
            (request.ImageVariant == WorkflowImageVariant.Enhanced && request.Transforms.Count == 0))
        {
            throw Failure(
                "Original raster evidence must be untransformed and enhanced evidence must carry a complete transform chain.");
        }
    }

    private static byte[] DecodeGray8(
        byte[] encodedBytes,
        int expectedWidth,
        int expectedHeight)
    {
        using var stream = new MemoryStream(encodedBytes, writable: false);
        BitmapDecoder decoder = BitmapDecoder.Create(
            stream,
            BitmapCreateOptions.PreservePixelFormat,
            BitmapCacheOption.OnLoad);
        if (decoder.Frames.Count == 0)
        {
            throw new InvalidDataException("The image contains no decodable frame.");
        }

        BitmapSource source = decoder.Frames[0];
        if (source.PixelWidth != expectedWidth || source.PixelHeight != expectedHeight)
        {
            throw new InvalidDataException(
                "Decoded image dimensions do not match retained image evidence.");
        }

        var grayscale = new FormatConvertedBitmap(source, PixelFormats.Gray8, null, 0);
        grayscale.Freeze();
        var pixels = new byte[checked(grayscale.PixelWidth * grayscale.PixelHeight)];
        grayscale.CopyPixels(pixels, grayscale.PixelWidth, 0);
        return pixels;
    }

    private static (MarkerAffineTransform Marker, OcrFrameTransform Ocr) ResolveTransforms(
        ProductionWorkflowDetectionRequest request)
    {
        if (request.ImageVariant == WorkflowImageVariant.Original)
        {
            return (MarkerAffineTransform.Identity, OcrFrameTransform.Identity);
        }

        double[] combined = IdentityMatrix();
        string expectedInputSpace = "original_pixels";
        foreach (WorkflowTransformProvenance transform in request.Transforms)
        {
            if (transform.Lossy || transform.OutputToInputMatrix is null ||
                !string.Equals(
                    transform.InputCoordinateSpace,
                    expectedInputSpace,
                    StringComparison.Ordinal) ||
                !IsAffine(transform.InputToOutputMatrix) ||
                !IsAffine(transform.OutputToInputMatrix) ||
                !AreInverse(transform.InputToOutputMatrix, transform.OutputToInputMatrix))
            {
                throw Failure(
                    "Enhanced raster evidence requires one continuous, reversible affine transform chain.");
            }

            combined = Multiply(transform.InputToOutputMatrix, combined);
            expectedInputSpace = transform.OutputCoordinateSpace;
        }

        if (!string.Equals(expectedInputSpace, "enhanced_pixels", StringComparison.Ordinal) ||
            Math.Abs(combined[1]) > MatrixTolerance ||
            Math.Abs(combined[3]) > MatrixTolerance ||
            combined[0] <= 0 || combined[4] <= 0 ||
            !NearlyEqual(combined[2], 0) || !NearlyEqual(combined[5], 0) ||
            !NearlyEqual(combined[0] * request.Panel.Original.Width, request.Image.Width) ||
            !NearlyEqual(combined[4] * request.Panel.Original.Height, request.Image.Height))
        {
            throw Failure(
                "The enhanced raster transform must be an exact positive axis-aligned original-to-enhanced mapping.");
        }

        var marker = new MarkerAffineTransform(
            combined[0], combined[1], combined[2],
            combined[3], combined[4], combined[5]);
        var ocr = new OcrFrameTransform(
            combined[0],
            combined[4],
            combined[2],
            combined[5]);
        if (!marker.IsInvertible || !ocr.IsInvertible)
        {
            throw Failure("The enhanced raster transform is not invertible.");
        }

        return (marker, ocr);
    }

    private static bool IsAffine(IReadOnlyList<double> matrix) =>
        matrix.Count == 9 &&
        matrix.All(double.IsFinite) &&
        Math.Abs(matrix[6]) <= MatrixTolerance &&
        Math.Abs(matrix[7]) <= MatrixTolerance &&
        Math.Abs(matrix[8] - 1) <= MatrixTolerance;

    private static bool AreInverse(
        IReadOnlyList<double> forward,
        IReadOnlyList<double> inverse)
    {
        double[] product = Multiply(forward, inverse);
        double[] identity = IdentityMatrix();
        return product.Zip(identity, static (actual, expected) => Math.Abs(actual - expected))
            .All(static difference => difference <= MatrixTolerance);
    }

    private static double[] Multiply(
        IReadOnlyList<double> left,
        IReadOnlyList<double> right)
    {
        var result = new double[9];
        for (int row = 0; row < 3; row++)
        {
            for (int column = 0; column < 3; column++)
            {
                result[(row * 3) + column] =
                    (left[row * 3] * right[column]) +
                    (left[(row * 3) + 1] * right[3 + column]) +
                    (left[(row * 3) + 2] * right[6 + column]);
            }
        }

        return result;
    }

    private static double[] IdentityMatrix() =>
        [1, 0, 0, 0, 1, 0, 0, 0, 1];

    private static bool NearlyEqual(double left, double right)
    {
        double scale = Math.Max(1, Math.Max(Math.Abs(left), Math.Abs(right)));
        return Math.Abs(left - right) <= MatrixTolerance * scale;
    }

    private static ProductionWorkflowStageException Failure(
        string technicalMessage,
        string userMessageKey = "Errors.DetectionEvidenceRejected") =>
        new(new ProductionWorkflowFailure(
            ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
            userMessageKey,
            technicalMessage,
            Recoverable: true,
            "Re-import the source and regenerate aligned production evidence."));
}
