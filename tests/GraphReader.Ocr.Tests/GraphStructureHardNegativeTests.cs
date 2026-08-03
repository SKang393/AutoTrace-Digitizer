// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class GraphStructureHardNegativeTests
{
    [TestMethod]
    public async Task TwoNearbyMarkerComponentsNeverBecomeMarkerExclusionMask()
    {
        const int width = 120;
        const int height = 80;
        byte[] pixels = Enumerable.Repeat((byte)255, width * height).ToArray();
        FillRectangle(pixels, width, 50, 35, 5, 5);
        FillRectangle(pixels, width, 60, 35, 5, 5);
        var image = new OcrImage(
            width,
            height,
            width,
            pixels,
            OcrSourceImage.Original,
            OcrFrameTransform.Identity);
        var detector = Detector();
        var pipeline = Pipeline(detector, "88");

        OcrResult result = await pipeline.RecognizeAsync(
            Request(image),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.IsEmpty(
            result.Masks,
            "Two nearby square marker components were grouped and emitted as an OCR exclusion mask.");
    }

    [TestMethod]
    public async Task TwoNearbyHollowSquareMarkersNeverBecomeMarkerExclusionMask()
    {
        const int width = 120;
        const int height = 80;
        byte[] pixels = Enumerable.Repeat((byte)255, width * height).ToArray();
        StrokeSquare(pixels, width, 48, 34, 7);
        StrokeSquare(pixels, width, 60, 34, 7);
        var image = new OcrImage(
            width,
            height,
            width,
            pixels,
            OcrSourceImage.Original,
            OcrFrameTransform.Identity);
        var detector = Detector();

        IReadOnlyList<OcrDetectedRegion> detected = await detector.DetectAsync(
            image,
            CancellationToken.None);
        OcrResult result = await Pipeline(detector, "88").RecognizeAsync(
            Request(image),
            CancellationToken.None);

        Assert.IsNotEmpty(detected, "The adversarial hollow-marker fixture did not enter the OCR detector path.");
        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.IsEmpty(
            result.Masks,
            "Two nearby hollow square markers were grouped and emitted as an OCR exclusion mask.");
    }

    [TestMethod]
    public async Task StandaloneTinyNumericOneRemainsDetectableAndEmittable()
    {
        const int width = 120;
        const int height = 80;
        byte[] pixels = Enumerable.Repeat((byte)255, width * height).ToArray();
        FillRectangle(pixels, width, 50, 63, 2, 6);
        var image = new OcrImage(
            width,
            height,
            width,
            pixels,
            OcrSourceImage.Original,
            OcrFrameTransform.Identity);
        var detector = Detector();

        IReadOnlyList<OcrDetectedRegion> detected = await detector.DetectAsync(
            image,
            CancellationToken.None);
        OcrResult result = await Pipeline(detector, "1").RecognizeAsync(
            Request(image),
            CancellationToken.None);

        Assert.HasCount(1, detected);
        Assert.AreEqual(1, detected[0].Evidence?.ComponentCount);
        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Regions);
        Assert.AreEqual("1", result.Regions[0].Text);
        Assert.AreEqual(OcrTextRole.XTick, result.Regions[0].Role);
        Assert.HasCount(1, result.Masks);
        Assert.AreEqual(result.Regions[0].RegionId, result.Masks[0].RegionId);
    }

    [TestMethod]
    public async Task GraphStructuresNeverBecomeMarkerExclusionOcrMasks()
    {
        foreach (GraphStructureKind kind in Enum.GetValues<GraphStructureKind>())
        {
            OcrImage image = Draw(kind);
            var recognizer = new StubTextRecognizer((crops, _) =>
                ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(crops.Select(crop =>
                    new OcrRecognition(
                        crop.RegionId,
                        crop.SourceImage,
                        [new OcrRecognitionAlternative("8", 0.99, crop.SourceImage)],
                        0.1)).ToArray()));
            var pipeline = new OcrPipeline(
                new ConnectedComponentTextRegionDetector(
                    new ConnectedComponentTextRegionDetectorOptions { ForegroundThreshold = 128 }),
                recognizer,
                new MemoryOcrResultCache(),
                batchSize: 4);
            OcrRequest request = OcrTestFixtures.Request() with
            {
                OriginalImage = image,
                DetectedRegions = null,
            };

            OcrResult result = await pipeline.RecognizeAsync(request, CancellationToken.None);

            Assert.IsTrue(result.Succeeded, $"{kind}: {result.Failure?.TechnicalMessage}");
            Assert.IsEmpty(result.Masks, $"{kind} became a marker-exclusion OCR mask.");
        }
    }

    private static OcrImage Draw(GraphStructureKind kind)
    {
        const int width = 120;
        const int height = 80;
        byte[] pixels = Enumerable.Repeat((byte)255, width * height).ToArray();
        switch (kind)
        {
            case GraphStructureKind.Marker:
                FillRectangle(pixels, width, 55, 38, 5, 5);
                break;
            case GraphStructureKind.Tick:
                FillRectangle(pixels, width, 59, 36, 2, 8);
                break;
            case GraphStructureKind.Axis:
                FillRectangle(pixels, width, 10, 40, 100, 1);
                break;
            case GraphStructureKind.Divider:
                for (var y = 8; y < 72; y += 10)
                {
                    FillRectangle(pixels, width, 60, y, 1, 5);
                }

                break;
            case GraphStructureKind.Arrowhead:
                for (var row = 0; row < 7; row++)
                {
                    FillRectangle(pixels, width, 56 + row, 37 - row, 1, 1 + (2 * row));
                }

                break;
            case GraphStructureKind.Bracket:
                FillRectangle(pixels, width, 52, 34, 17, 1);
                FillRectangle(pixels, width, 52, 34, 1, 7);
                FillRectangle(pixels, width, 68, 34, 1, 7);
                break;
            case GraphStructureKind.Intersection:
                FillRectangle(pixels, width, 54, 40, 13, 1);
                FillRectangle(pixels, width, 60, 34, 1, 13);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(kind));
        }

        return new OcrImage(
            width,
            height,
            width,
            pixels,
            OcrSourceImage.Original,
            OcrFrameTransform.Identity);
    }

    private static ConnectedComponentTextRegionDetector Detector() =>
        new(new ConnectedComponentTextRegionDetectorOptions { ForegroundThreshold = 128 });

    private static OcrPipeline Pipeline(ITextRegionDetector detector, string text)
    {
        var recognizer = new StubTextRecognizer((crops, _) =>
            ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(crops.Select(crop =>
                new OcrRecognition(
                    crop.RegionId,
                    crop.SourceImage,
                    [new OcrRecognitionAlternative(text, 0.99, crop.SourceImage)],
                    0.1)).ToArray()));
        return new OcrPipeline(detector, recognizer, new MemoryOcrResultCache(), batchSize: 4);
    }

    private static OcrRequest Request(OcrImage image) =>
        new(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            image,
            new OcrRectangle(20, 15, 90, 45));

    private static void FillRectangle(
        byte[] pixels,
        int stride,
        int x,
        int y,
        int width,
        int height)
    {
        for (var row = y; row < y + height; row++)
        {
            for (var column = x; column < x + width; column++)
            {
                pixels[(row * stride) + column] = 0;
            }
        }
    }

    private static void StrokeSquare(byte[] pixels, int stride, int x, int y, int size)
    {
        FillRectangle(pixels, stride, x, y, size, 1);
        FillRectangle(pixels, stride, x, y + size - 1, size, 1);
        FillRectangle(pixels, stride, x, y + 1, 1, size - 2);
        FillRectangle(pixels, stride, x + size - 1, y + 1, 1, size - 2);
    }

    private enum GraphStructureKind
    {
        Marker,
        Tick,
        Axis,
        Divider,
        Arrowhead,
        Bracket,
        Intersection,
    }
}
