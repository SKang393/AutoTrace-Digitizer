// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class TextRegionDetectorTests
{
    [TestMethod]
    public async Task NearbyGlyphComponentsAreGroupedIntoOneDeterministicTextRegion()
    {
        OcrImage image = GlyphImage(
            OcrSourceImage.Original,
            OcrFrameTransform.Identity,
            (20, 30, 2, 6),
            (25, 30, 3, 6),
            (31, 30, 2, 6));
        var detector = new ConnectedComponentTextRegionDetector(
            new ConnectedComponentTextRegionDetectorOptions { ForegroundThreshold = 128 });

        IReadOnlyList<OcrDetectedRegion> first = await detector.DetectAsync(image, CancellationToken.None);
        IReadOnlyList<OcrDetectedRegion> second = await detector.DetectAsync(image, CancellationToken.None);

        Assert.HasCount(1, first);
        Assert.AreEqual(new OcrRectangle(20, 30, 13, 6), first[0].Polygon.Bounds);
        Assert.AreEqual(OcrContract.CoordinateSpace, first[0].CoordinateSpace);
        Assert.AreEqual(first[0].RegionId, second[0].RegionId);
        Assert.IsTrue(Guid.TryParse(first[0].RegionId, out _));
    }

    [TestMethod]
    public async Task EnhancedFrameDetectionMapsBackToOriginalPixels()
    {
        OcrImage enhanced = GlyphImage(
            OcrSourceImage.Enhanced,
            new OcrFrameTransform(2, 2, 0, 0),
            (40, 60, 4, 12),
            (50, 60, 6, 12));
        var detector = new ConnectedComponentTextRegionDetector(
            new ConnectedComponentTextRegionDetectorOptions { ForegroundThreshold = 128 });

        IReadOnlyList<OcrDetectedRegion> regions = await detector.DetectAsync(enhanced, CancellationToken.None);

        Assert.HasCount(1, regions);
        Assert.AreEqual(new OcrRectangle(20, 30, 8, 6), regions[0].Polygon.Bounds);
        Assert.AreEqual(OcrContract.CoordinateSpace, regions[0].CoordinateSpace);
    }

    [TestMethod]
    public async Task DetectorHonorsCancellationBeforeScanningPixels()
    {
        var detector = new ConnectedComponentTextRegionDetector();
        var cancellation = new CancellationToken(canceled: true);

        await Assert.ThrowsExactlyAsync<OperationCanceledException>(
            () => detector.DetectAsync(OcrTestFixtures.Image(), cancellation).AsTask());
    }

    [TestMethod]
    public async Task DetectorObservesCancellationWhileTraversingOneLargeComponent()
    {
        const int width = 5000;
        const int height = 5000;
        var image = new OcrImage(
            width,
            height,
            width,
            new byte[width * height],
            OcrSourceImage.Original,
            OcrFrameTransform.Identity);
        var detector = new ConnectedComponentTextRegionDetector(
            new ConnectedComponentTextRegionDetectorOptions { ForegroundThreshold = 128 });
        using var cancellation = new CancellationTokenSource();
        using var started = new ManualResetEventSlim();
        Task detection = Task.Run(async () =>
        {
            started.Set();
            await detector.DetectAsync(image, cancellation.Token);
        });

        Assert.IsTrue(started.Wait(TimeSpan.FromSeconds(5)));
        await Task.Delay(TimeSpan.FromMilliseconds(25));
        Assert.IsFalse(detection.IsCompleted, "The fixed fixture did not reach the long component traversal.");
        cancellation.Cancel();

        await Assert.ThrowsExactlyAsync<OperationCanceledException>(() =>
            detection.WaitAsync(TimeSpan.FromSeconds(10)));
    }

    private static OcrImage GlyphImage(
        OcrSourceImage source,
        OcrFrameTransform transform,
        params (int X, int Y, int Width, int Height)[] components)
    {
        const int width = 160;
        const int height = 100;
        byte[] pixels = Enumerable.Repeat((byte)255, width * height).ToArray();
        foreach ((int x, int y, int componentWidth, int componentHeight) in components)
        {
            for (var row = y; row < y + componentHeight; row++)
            {
                for (var column = x; column < x + componentWidth; column++)
                {
                    pixels[(row * width) + column] = 0;
                }
            }
        }

        return new OcrImage(width, height, width, pixels, source, transform);
    }
}
