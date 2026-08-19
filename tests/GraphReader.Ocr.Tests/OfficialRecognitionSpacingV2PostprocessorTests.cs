// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OfficialRecognitionSpacingV2PostprocessorTests
{
    [TestMethod]
    public void RestoresOnlySourceEvidencedSpacesWithFrozenWidthPartition()
    {
        OcrV8SourceCrop crop = Crop(
            (2, 9, false),
            (16, 19, false),
            (26, 29, false));

        Assert.AreEqual(
            "AB C D",
            OfficialRecognitionSpacingV2Postprocessor.Restore(crop, "ABCD"));
        Assert.AreEqual(
            "AB C D",
            OfficialRecognitionSpacingV2Postprocessor.Restore(crop, "AB C D"));
    }

    [TestMethod]
    public void SourceSerifsDistinguishIsolatedCapitalIFromLowercaseL()
    {
        OcrV8SourceCrop crop = Crop(
            (2, 5, false),
            (12, 15, false),
            (22, 26, true));

        Assert.AreEqual(
            "A b I",
            OfficialRecognitionSpacingV2Postprocessor.Restore(crop, "Abl"));
    }

    [TestMethod]
    public void JoinedInkAndShortPredictionsRemainUnchanged()
    {
        OcrV8SourceCrop crop = Crop((2, 5, false), (8, 11, false));

        Assert.AreEqual(
            "AB",
            OfficialRecognitionSpacingV2Postprocessor.Restore(crop, "AB"));
        Assert.AreEqual(
            "7",
            OfficialRecognitionSpacingV2Postprocessor.Restore(crop, "7"));
    }

    [TestMethod]
    public void TamperedSourceCropFailsClosed()
    {
        OcrV8SourceCrop crop = Crop(
            (2, 5, false),
            (12, 15, false),
            (22, 25, false));
        var tampered = crop with { PixelSha256 = new string('0', 64) };

        Assert.ThrowsExactly<ArgumentException>(() =>
            OfficialRecognitionSpacingV2Postprocessor.Restore(tampered, "ABC"));
    }

    [TestMethod]
    public async Task DecoratorPreservesIdentityGeometryConfidenceAndFailure()
    {
        OcrV8SourceCrop source = Crop(
            (2, 5, false),
            (12, 15, false),
            (22, 26, true));
        var polygon = OcrPolygon.FromRectangle(new OcrRectangle(3, 4, 30, 18));
        var crop = new OcrCrop(
            "region",
            OcrSourceImage.Original,
            320,
            48,
            new float[320 * 48],
            new string('a', 64),
            polygon,
            SourceCrop: source);
        var inner = new StubTextRecognizer((crops, cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(
            [
                new OcrRecognition(
                    crops[0].RegionId,
                    crops[0].SourceImage,
                    [new OcrRecognitionAlternative("Abl", 0.9375, crops[0].SourceImage)],
                    2.5),
            ]);
        });
        var decorator = new OfficialRecognitionSpacingV2TextRecognizer(inner);

        OcrRecognition result = (await decorator.RecognizeBatchAsync(
            [crop],
            CancellationToken.None)).Single();

        Assert.AreEqual(inner.ModelId, decorator.ModelId);
        Assert.AreEqual(inner.ModelVersion, decorator.ModelVersion);
        Assert.AreEqual(inner.ModelSha256, decorator.ModelSha256);
        Assert.AreEqual("region", result.RegionId);
        Assert.AreEqual("A b I", result.Alternatives.Single().Text);
        Assert.AreEqual(0.9375, result.Alternatives.Single().Confidence, 0.0);
        Assert.AreEqual(2.5, result.InferenceMilliseconds, 0.0);
        Assert.AreEqual(64, decorator.ConfigurationFingerprint.Length);
    }

    private static OcrV8SourceCrop Crop(params (int Left, int Right, bool SerifI)[] groups)
    {
        const int width = 40;
        const int height = 20;
        var pixels = Enumerable.Repeat((byte)255, width * height).ToArray();
        foreach ((int left, int right, bool serifI) in groups)
        {
            if (serifI)
            {
                for (var x = left; x <= right; x++)
                {
                    pixels[(4 * width) + x] = 0;
                    pixels[(15 * width) + x] = 0;
                }
                int center = (left + right) / 2;
                for (var y = 4; y <= 15; y++)
                {
                    pixels[(y * width) + center] = 0;
                }
            }
            else
            {
                for (var y = 4; y <= 15; y++)
                {
                    for (var x = left; x <= right; x++)
                    {
                        pixels[(y * width) + x] = 0;
                    }
                }
            }
        }

        return new OcrV8SourceCrop(
            width,
            height,
            pixels,
            Convert.ToHexStringLower(SHA256.HashData(pixels)),
            OcrPolygon.FromRectangle(new OcrRectangle(0, 0, width, height)));
    }
}
