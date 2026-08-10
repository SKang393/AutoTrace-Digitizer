// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class CropAndMaskTests
{
    private static readonly float[] ExpectedTwoPixelCrop = [0f, 1f];

    [TestMethod]
    public void TinyRegionIsNormalizedWithoutLosingItsOriginalPolygon()
    {
        OcrDetectedRegion tiny = OcrTestFixtures.Region("tiny", 20, 30, 2, 3);

        IReadOnlyList<IReadOnlyList<OcrCrop>> batches = OcrCropBatcher.CreateBatches(
            OcrTestFixtures.Image(),
            [tiny],
            new OcrCropBatcherOptions { TargetWidth = 64, TargetHeight = 24, BatchSize = 8 });

        Assert.HasCount(1, batches);
        Assert.HasCount(1, batches[0]);
        OcrCrop crop = batches[0][0];
        Assert.AreEqual(64, crop.Width);
        Assert.AreEqual(24, crop.Height);
        Assert.AreEqual(64 * 24, crop.Pixels.Length);
        Assert.AreEqual(64, crop.CropSha256.Length);
        CollectionAssert.AreEqual(tiny.Polygon.Points.ToArray(), crop.OriginalPolygon.Points.ToArray());
        Assert.IsTrue(crop.Pixels.Span.ToArray().All(float.IsFinite));
    }

    [TestMethod]
    public void DefaultResizePreservesAspectRatioAndPadsWithNormalizedZeroSourceValue()
    {
        var pixels = Enumerable.Repeat((byte)255, 20 * 10).ToArray();
        var image = new OcrImage(
            20,
            10,
            20,
            pixels,
            OcrSourceImage.Original,
            OcrFrameTransform.Identity);
        OcrDetectedRegion region = OcrTestFixtures.Region("wide", 0, 0, 20, 10);

        OcrCrop crop = OcrCropBatcher.CreateBatches(
            image,
            [region],
            new OcrCropBatcherOptions
            {
                TargetWidth = 40,
                TargetHeight = 10,
                PaddingPixels = 0,
            })[0][0];

        for (var y = 0; y < crop.Height; y++)
        {
            ReadOnlySpan<float> row = crop.Pixels.Span.Slice(y * crop.Width, crop.Width);
            Assert.IsTrue(row[..20].ToArray().All(static value => value == 1f));
            Assert.IsTrue(row[20..].ToArray().All(static value => value == 0.5f));
        }
    }

    [TestMethod]
    public void StretchModeRemainsAvailableForModelsWithFixedWarpedTrainingInputs()
    {
        var image = new OcrImage(
            20,
            10,
            20,
            Enumerable.Repeat((byte)255, 20 * 10).ToArray(),
            OcrSourceImage.Original,
            OcrFrameTransform.Identity);
        OcrDetectedRegion region = OcrTestFixtures.Region("wide", 0, 0, 20, 10);

        OcrCrop crop = OcrCropBatcher.CreateBatches(
            image,
            [region],
            new OcrCropBatcherOptions
            {
                TargetWidth = 40,
                TargetHeight = 10,
                PaddingPixels = 0,
                ResizeMode = OcrCropResizeMode.Stretch,
            })[0][0];

        Assert.IsTrue(crop.Pixels.Span.ToArray().All(static value => value == 1f));
    }

    [TestMethod]
    public void EqualSizeResizeKeepsSourcePixelCenters()
    {
        var image = new OcrImage(
            2,
            1,
            2,
            new byte[] { 0, 255 },
            OcrSourceImage.Original,
            OcrFrameTransform.Identity);
        OcrDetectedRegion region = OcrTestFixtures.Region("two-pixels", 0, 0, 2, 1);

        OcrCrop crop = OcrCropBatcher.CreateBatches(
            image,
            [region],
            new OcrCropBatcherOptions
            {
                TargetWidth = 2,
                TargetHeight = 1,
                PaddingPixels = 0,
                ResizeMode = OcrCropResizeMode.Stretch,
            })[0][0];

        CollectionAssert.AreEqual(ExpectedTwoPixelCrop, crop.Pixels.ToArray());
    }

    [TestMethod]
    public void EnhancedCropMapsOriginalPolygonThroughDeclaredTransform()
    {
        OcrDetectedRegion region = OcrTestFixtures.Region("scaled", 20, 30, 12, 8);
        OcrImage enhanced = OcrTestFixtures.Image(
            OcrSourceImage.Enhanced,
            width: 320,
            height: 200,
            transform: new OcrFrameTransform(2, 2, 0, 0));

        OcrCrop crop = OcrCropBatcher.CreateBatches(enhanced, [region])[0][0];

        Assert.AreEqual(OcrSourceImage.Enhanced, crop.SourceImage);
        CollectionAssert.AreEqual(region.Polygon.Points.ToArray(), crop.OriginalPolygon.Points.ToArray());
        Assert.IsNotEmpty(crop.Pixels.Span.ToArray());
    }

    [TestMethod]
    public void CropBatcherHonorsPreCanceledRequest()
    {
        var cancellation = new CancellationToken(canceled: true);

        Assert.ThrowsExactly<OperationCanceledException>(() =>
            OcrCropBatcher.CreateBatches(
                OcrTestFixtures.Image(),
                [OcrTestFixtures.Region("cancel", 20, 30, 12, 8)],
                cancellationToken: cancellation));
    }

    [TestMethod]
    public void MarkerExclusionMasksArePaddedClippedAndOriginalPixelBased()
    {
        OcrDetectedRegion[] textRegions =
        [
            OcrTestFixtures.Region("left-edge", 0, 10, 8, 6, confidence: 0.81),
            OcrTestFixtures.Region("generalization", 70, 40, 28, 8, confidence: 0.94),
        ];

        IReadOnlyList<OcrMask> masks = OcrMaskBuilder.Build(
            textRegions,
            originalWidth: 100,
            originalHeight: 80,
            paddingPixels: 2);

        Assert.HasCount(2, masks);
        Assert.AreEqual(new OcrRectangle(0, 8, 10, 10), masks[0].Polygon.Bounds);
        Assert.AreEqual(new OcrRectangle(68, 38, 32, 12), masks[1].Polygon.Bounds);
        Assert.IsTrue(masks.All(static mask => mask.CoordinateSpace == OcrContract.CoordinateSpace));
        Assert.AreEqual(0.94d, masks[1].Confidence, 1e-9);
    }

    [TestMethod]
    public void NonOriginalRegionCannotBecomeAMarkerMask()
    {
        OcrDetectedRegion region = OcrTestFixtures.Region("derived", 20, 30, 12, 8) with
        {
            CoordinateSpace = "enhanced_pixels",
        };

        Assert.Throws<ArgumentException>(() => OcrMaskBuilder.Build([region], 100, 80));
    }

    [TestMethod]
    public async Task OffsetAndScaledFrameKeepsMaskInCanonicalOriginalPixels()
    {
        OcrDetectedRegion region = OcrTestFixtures.Region("offset", 520, 120, 20, 8);
        OcrImage offsetFrame = OcrTestFixtures.Image(
            OcrSourceImage.Original,
            width: 200,
            height: 100,
            transform: new OcrFrameTransform(2, 2, -1000, -220)) with
        {
            CanonicalOriginalWidth = 1000,
            CanonicalOriginalHeight = 800,
        };
        var alternatives = new Dictionary<
            (string RegionId, OcrSourceImage Source),
            IReadOnlyList<OcrRecognitionAlternative>>
        {
            [("offset", OcrSourceImage.Original)] =
                [new OcrRecognitionAlternative("100", 0.95, OcrSourceImage.Original)],
        };
        var pipeline = new OcrPipeline(
            new StubTextRegionDetector([]),
            new StubTextRecognizer(alternatives),
            new InMemoryOcrResultCache(),
            batchSize: 4);
        var request = new OcrRequest(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            offsetFrame,
            new OcrRectangle(500, 100, 100, 50),
            DetectedRegions: [region],
            TransformChain: "crop:500,110;scale:2");

        OcrResult result = await pipeline.RecognizeAsync(request, CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Masks);
        Assert.AreEqual(new OcrRectangle(519, 119, 22, 10), result.Masks[0].Polygon.Bounds);
        Assert.AreEqual(OcrContract.CoordinateSpace, result.Masks[0].CoordinateSpace);
    }
}
