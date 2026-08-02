// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Imaging.Tests;

[TestClass]
public sealed class TransformAndCanvasTests
{
    [TestMethod]
    public void TransformChainOriginalDerivedRoundTripIsWithinQuarterPixel()
    {
        ImageTransform crop = ImageTransform.Create(
            ImageTransformKind.Crop,
            ImageCoordinateSpace.OriginalPixels,
            ImageCoordinateSpace.PanelPixels,
            new Matrix3x3(1, 0, -31.5, 0, 1, -17.25, 0, 0, 1));
        ImageTransform perspective = ImageTransform.Create(
            ImageTransformKind.Perspective,
            ImageCoordinateSpace.PanelPixels,
            ImageCoordinateSpace.DeskewedPixels,
            new Matrix3x3(1.01, 0.02, 2, -0.01, 0.99, 3, 0.00002, -0.00001, 1));
        ImageTransform scale = ImageTransform.Create(
            ImageTransformKind.Scale,
            ImageCoordinateSpace.DeskewedPixels,
            ImageCoordinateSpace.EnhancedPixels,
            new Matrix3x3(2, 0, 0, 0, 2, 0, 0, 0, 1));
        var chain = new TransformChain(new[] { crop, perspective, scale });
        var original = new PixelPoint(477.125, 293.75);

        PixelPoint restored = chain.ToOriginal(chain.ToDerived(original));
        double error = Math.Sqrt(Math.Pow(restored.X - original.X, 2) + Math.Pow(restored.Y - original.Y, 2));

        Assert.IsLessThanOrEqualTo(0.25, error);
        Assert.IsLessThan(1e-8, error);
    }

    [TestMethod]
    public void DerivedHandleFactoryExposesAllRequiredOperationsAndInverses()
    {
        ImportedImage image = TestImageFixtures.FakeImportedImage("fake.png", 1);
        DerivedImageHandle[] handles =
        [
            DerivedImageHandles.Crop(image, new PixelRect(1, 1, 4, 3)),
            DerivedImageHandles.Rotate(image, 12, new PixelPoint(3.5, 2.5)),
            DerivedImageHandles.Deskew(image, 1.5, new PixelPoint(3.5, 2.5)),
            DerivedImageHandles.Perspective(image, new Matrix3x3(1, 0.01, 0, 0.01, 1, 0, 0.0001, 0, 1), 7, 5),
            DerivedImageHandles.Scale(image, 2, 2),
            DerivedImageHandles.Display(image, 1.5, 12, 24)
        ];

        CollectionAssert.AreEquivalent(
            Enum.GetValues<DerivedImageOperation>(),
            handles.Select(static handle => handle.Operation).ToArray());
        Assert.IsTrue(handles.All(static handle => handle.TransformChain.DerivedToOriginal.HasValue));
        ImageTransform displayTransform = handles.Single(static handle => handle.Operation == DerivedImageOperation.Display).TransformChain.Transforms[0];
        Assert.AreEqual(ImageCoordinateSpace.OriginalPixels, displayTransform.SourceSpace);
        Assert.AreEqual(ImageCoordinateSpace.OriginalPixels, displayTransform.TargetSpace);
    }

    [TestMethod]
    public void CoordinateSpaceEnumMatchesFrozenContract()
    {
        CollectionAssert.AreEquivalent(
            new[]
            {
                ImageCoordinateSpace.PagePixels,
                ImageCoordinateSpace.OriginalPixels,
                ImageCoordinateSpace.PanelPixels,
                ImageCoordinateSpace.DeskewedPixels,
                ImageCoordinateSpace.EnhancedPixels,
                ImageCoordinateSpace.ModelTensor,
                ImageCoordinateSpace.GraphUnits
            },
            Enum.GetValues<ImageCoordinateSpace>());
    }

    [TestMethod]
    public void CoordinateMapperAccountsForHighDpiAndRoundTrips()
    {
        var view = new CanvasViewState(zoom: 2.25, offsetXDip: 17, offsetYDip: -8, dpiScaleX: 1.5, dpiScaleY: 2);
        var image = new PixelPoint(120.5, 80.25);

        PixelPoint device = CanvasCoordinateMapper.ImageToDevicePixels(image, view);
        PixelPoint restored = CanvasCoordinateMapper.DevicePixelsToImage(device, view);

        Assert.AreEqual(image.X, restored.X, 1e-9);
        Assert.AreEqual(image.Y, restored.Y, 1e-9);
        Assert.AreEqual(((image.X * view.Zoom) + view.OffsetXDip) * 1.5, device.X, 1e-9);
    }

    [TestMethod]
    public async Task TiledSourceReturnsClippedTilesInRowMajorOrder()
    {
        var source = new TiledCanvasImageSource(550, 390, 256, new FakeCanvasTileProvider());

        IReadOnlyList<CanvasTile> tiles = source.GetVisibleTiles(new PixelRect(250, 250, 400, 300));

        CollectionAssert.AreEqual(
            new[] { (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1) },
            tiles.Select(static tile => (tile.Column, tile.Row)).ToArray());
        Assert.AreEqual(38, tiles[^1].Width);
        Assert.AreEqual(134, tiles[^1].Height);
        ReadOnlyMemory<byte> content = await source.GetTileAsync(tiles[^1], CancellationToken.None);
        CollectionAssert.AreEqual(new byte[] { 2, 1 }, content.ToArray());

        Assert.IsEmpty(source.GetVisibleTiles(new PixelRect(600, 20, 30, 30)));
        Assert.HasCount(1, source.GetVisibleTiles(new PixelRect(0, 0, 256, 256)));
    }

    [TestMethod]
    public async Task TiledSourceRejectsForgedOrOutOfRangeTiles()
    {
        var source = new TiledCanvasImageSource(550, 390, 256, new FakeCanvasTileProvider());

        await Assert.ThrowsExactlyAsync<ArgumentException>(async () =>
        {
            await source.GetTileAsync(new CanvasTile(1, 0, 0, 0, 256, 256), CancellationToken.None);
        });
        await Assert.ThrowsExactlyAsync<ArgumentOutOfRangeException>(async () =>
        {
            await source.GetTileAsync(new CanvasTile(3, 0, 768, 0, 1, 256), CancellationToken.None);
        });
        await Assert.ThrowsExactlyAsync<ArgumentException>(async () =>
        {
            await source.GetTileAsync(new CanvasTile(2, 1, 512, 256, 256, 256), CancellationToken.None);
        });
    }
}
