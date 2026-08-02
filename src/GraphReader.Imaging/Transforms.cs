// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;

namespace GraphReader.Imaging;

public readonly record struct PixelPoint(double X, double Y);

public readonly record struct PixelRect
{
    public PixelRect(double x, double y, double width, double height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);

        X = x;
        Y = y;
        Width = width;
        Height = height;
    }

    public double X { get; }

    public double Y { get; }

    public double Width { get; }

    public double Height { get; }
}

public readonly record struct Matrix3x3(
    double M11,
    double M12,
    double M13,
    double M21,
    double M22,
    double M23,
    double M31,
    double M32,
    double M33)
{
    private const double SingularTolerance = 1e-12;

    public static Matrix3x3 Identity { get; } = new(1, 0, 0, 0, 1, 0, 0, 0, 1);

    public PixelPoint Transform(PixelPoint point)
    {
        double denominator = (M31 * point.X) + (M32 * point.Y) + M33;
        if (Math.Abs(denominator) < SingularTolerance)
        {
            throw new InvalidOperationException("The point maps to infinity for this transform.");
        }

        return new PixelPoint(
            ((M11 * point.X) + (M12 * point.Y) + M13) / denominator,
            ((M21 * point.X) + (M22 * point.Y) + M23) / denominator);
    }

    public Matrix3x3 Multiply(Matrix3x3 right) => new(
        (M11 * right.M11) + (M12 * right.M21) + (M13 * right.M31),
        (M11 * right.M12) + (M12 * right.M22) + (M13 * right.M32),
        (M11 * right.M13) + (M12 * right.M23) + (M13 * right.M33),
        (M21 * right.M11) + (M22 * right.M21) + (M23 * right.M31),
        (M21 * right.M12) + (M22 * right.M22) + (M23 * right.M32),
        (M21 * right.M13) + (M22 * right.M23) + (M23 * right.M33),
        (M31 * right.M11) + (M32 * right.M21) + (M33 * right.M31),
        (M31 * right.M12) + (M32 * right.M22) + (M33 * right.M32),
        (M31 * right.M13) + (M32 * right.M23) + (M33 * right.M33));

    public bool TryInvert(out Matrix3x3 inverse)
    {
        double determinant =
            (M11 * ((M22 * M33) - (M23 * M32))) -
            (M12 * ((M21 * M33) - (M23 * M31))) +
            (M13 * ((M21 * M32) - (M22 * M31)));

        if (Math.Abs(determinant) < SingularTolerance || !double.IsFinite(determinant))
        {
            inverse = default;
            return false;
        }

        double reciprocal = 1 / determinant;
        inverse = new Matrix3x3(
            ((M22 * M33) - (M23 * M32)) * reciprocal,
            ((M13 * M32) - (M12 * M33)) * reciprocal,
            ((M12 * M23) - (M13 * M22)) * reciprocal,
            ((M23 * M31) - (M21 * M33)) * reciprocal,
            ((M11 * M33) - (M13 * M31)) * reciprocal,
            ((M13 * M21) - (M11 * M23)) * reciprocal,
            ((M21 * M32) - (M22 * M31)) * reciprocal,
            ((M12 * M31) - (M11 * M32)) * reciprocal,
            ((M11 * M22) - (M12 * M21)) * reciprocal);
        return true;
    }

    public IReadOnlyList<double> ToValues() => Array.AsReadOnly(new[]
    {
        M11, M12, M13, M21, M22, M23, M31, M32, M33
    });
}

public enum ImageTransformKind
{
    Crop,
    Scale,
    Affine,
    Perspective,
    Rotation
}

public enum ImageCoordinateSpace
{
    PagePixels,
    OriginalPixels,
    PanelPixels,
    DeskewedPixels,
    EnhancedPixels,
    ModelTensor,
    GraphUnits
}

public sealed record ImageTransform(
    Guid TransformId,
    ImageTransformKind Kind,
    ImageCoordinateSpace SourceSpace,
    ImageCoordinateSpace TargetSpace,
    Matrix3x3 Matrix,
    Matrix3x3? InverseMatrix,
    IReadOnlyDictionary<string, double> Parameters,
    bool Lossy)
{
    public static ImageTransform Create(
        ImageTransformKind kind,
        ImageCoordinateSpace source,
        ImageCoordinateSpace target,
        Matrix3x3 matrix,
        IEnumerable<KeyValuePair<string, double>>? parameters = null)
    {
        bool invertible = matrix.TryInvert(out Matrix3x3 inverse);
        var values = new SortedDictionary<string, double>(StringComparer.Ordinal);
        if (parameters is not null)
        {
            foreach (KeyValuePair<string, double> parameter in parameters)
            {
                values.Add(parameter.Key, parameter.Value);
            }
        }

        return new ImageTransform(
            Guid.NewGuid(),
            kind,
            source,
            target,
            matrix,
            invertible ? inverse : null,
            new ReadOnlyDictionary<string, double>(values),
            !invertible);
    }
}

public sealed class TransformChain
{
    public TransformChain(IEnumerable<ImageTransform> transforms)
    {
        ImageTransform[] materialized = transforms.ToArray();
        for (int index = 1; index < materialized.Length; index++)
        {
            if (materialized[index - 1].TargetSpace != materialized[index].SourceSpace)
            {
                throw new ArgumentException("Adjacent transform coordinate spaces do not match.", nameof(transforms));
            }
        }

        Transforms = new ReadOnlyCollection<ImageTransform>(materialized);
        Matrix3x3 composite = Matrix3x3.Identity;
        foreach (ImageTransform transform in materialized)
        {
            composite = transform.Matrix.Multiply(composite);
        }

        OriginalToDerived = composite;
        DerivedToOriginal = composite.TryInvert(out Matrix3x3 inverse) ? inverse : null;
    }

    public IReadOnlyList<ImageTransform> Transforms { get; }

    public Matrix3x3 OriginalToDerived { get; }

    public Matrix3x3? DerivedToOriginal { get; }

    public PixelPoint ToDerived(PixelPoint point) => OriginalToDerived.Transform(point);

    public PixelPoint ToOriginal(PixelPoint point) =>
        (DerivedToOriginal ?? throw new InvalidOperationException("The transform chain has no inverse.")).Transform(point);
}

public enum DerivedImageOperation
{
    Crop,
    Rotation,
    Deskew,
    Perspective,
    Scale,
    Display
}

public sealed record DerivedImageHandle(
    Guid HandleId,
    string OriginalSha256,
    DerivedImageOperation Operation,
    int Width,
    int Height,
    TransformChain TransformChain);

public static class DerivedImageHandles
{
    public static DerivedImageHandle Crop(ImportedImage image, PixelRect crop)
    {
        var matrix = new Matrix3x3(1, 0, -crop.X, 0, 1, -crop.Y, 0, 0, 1);
        ImageTransform transform = ImageTransform.Create(
            ImageTransformKind.Crop,
            ImageCoordinateSpace.OriginalPixels,
            ImageCoordinateSpace.PanelPixels,
            matrix,
            new Dictionary<string, double>
            {
                ["x"] = crop.X,
                ["y"] = crop.Y,
                ["width"] = crop.Width,
                ["height"] = crop.Height
            });
        return Create(image, DerivedImageOperation.Crop, checked((int)Math.Ceiling(crop.Width)), checked((int)Math.Ceiling(crop.Height)), transform);
    }

    public static DerivedImageHandle Rotate(ImportedImage image, double degrees, PixelPoint center) =>
        Rotation(image, degrees, center, DerivedImageOperation.Rotation, ImageCoordinateSpace.OriginalPixels, ImageCoordinateSpace.DeskewedPixels);

    public static DerivedImageHandle Deskew(ImportedImage image, double degrees, PixelPoint center) =>
        Rotation(image, -degrees, center, DerivedImageOperation.Deskew, ImageCoordinateSpace.OriginalPixels, ImageCoordinateSpace.DeskewedPixels);

    public static DerivedImageHandle Perspective(ImportedImage image, Matrix3x3 matrix, int width, int height)
    {
        ImageTransform transform = ImageTransform.Create(
            ImageTransformKind.Perspective,
            ImageCoordinateSpace.OriginalPixels,
            ImageCoordinateSpace.DeskewedPixels,
            matrix);
        return Create(image, DerivedImageOperation.Perspective, width, height, transform);
    }

    public static DerivedImageHandle Scale(ImportedImage image, double scaleX, double scaleY)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(scaleX);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(scaleY);

        var matrix = new Matrix3x3(scaleX, 0, 0, 0, scaleY, 0, 0, 0, 1);
        ImageTransform transform = ImageTransform.Create(
            ImageTransformKind.Scale,
            ImageCoordinateSpace.OriginalPixels,
            ImageCoordinateSpace.EnhancedPixels,
            matrix,
            new Dictionary<string, double> { ["scale_x"] = scaleX, ["scale_y"] = scaleY });
        return Create(
            image,
            DerivedImageOperation.Scale,
            checked((int)Math.Ceiling(image.Metadata.Width * scaleX)),
            checked((int)Math.Ceiling(image.Metadata.Height * scaleY)),
            transform);
    }

    public static DerivedImageHandle Display(ImportedImage image, double zoom, double offsetX, double offsetY)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(zoom);

        ImageTransform transform = ImageTransform.Create(
            ImageTransformKind.Affine,
            ImageCoordinateSpace.OriginalPixels,
            ImageCoordinateSpace.OriginalPixels,
            Matrix3x3.Identity,
            new Dictionary<string, double> { ["zoom"] = zoom, ["offset_x"] = offsetX, ["offset_y"] = offsetY });
        return Create(image, DerivedImageOperation.Display, image.Metadata.Width, image.Metadata.Height, transform);
    }

    private static DerivedImageHandle Rotation(
        ImportedImage image,
        double degrees,
        PixelPoint center,
        DerivedImageOperation operation,
        ImageCoordinateSpace source,
        ImageCoordinateSpace target)
    {
        double radians = degrees * Math.PI / 180;
        double cosine = Math.Cos(radians);
        double sine = Math.Sin(radians);
        double translationX = center.X - (cosine * center.X) + (sine * center.Y);
        double translationY = center.Y - (sine * center.X) - (cosine * center.Y);
        var matrix = new Matrix3x3(cosine, -sine, translationX, sine, cosine, translationY, 0, 0, 1);
        ImageTransform transform = ImageTransform.Create(
            operation == DerivedImageOperation.Rotation ? ImageTransformKind.Rotation : ImageTransformKind.Affine,
            source,
            target,
            matrix,
            new Dictionary<string, double> { ["degrees"] = degrees, ["center_x"] = center.X, ["center_y"] = center.Y });
        return Create(image, operation, image.Metadata.Width, image.Metadata.Height, transform);
    }

    private static DerivedImageHandle Create(
        ImportedImage image,
        DerivedImageOperation operation,
        int width,
        int height,
        ImageTransform transform) =>
        new(Guid.NewGuid(), image.Sha256, operation, width, height, new TransformChain(new[] { transform }));
}
