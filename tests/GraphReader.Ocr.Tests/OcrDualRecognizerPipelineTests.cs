// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrDualRecognizerPipelineTests
{
    [TestMethod]
    public async Task DetectsOnceUsesModelSpecificCropsAndRoutesOnlyNumericTicks()
    {
        OcrDetectedRegion[] detected =
        [
            OcrTestFixtures.Region("x-tick", 46, 89, 10, 6),
            OcrTestFixtures.Region(
                "phase-word",
                66,
                4,
                42,
                8,
                context: new OcrRegionContext(ExplicitRoleHint: OcrTextRole.PhaseHeading)),
        ];
        var detector = new StubTextRegionDetector(detected);
        var general = new StubTextRecognizer((crops, cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            Assert.IsTrue(crops.All(static crop => crop.Width == 320 && crop.Height == 48));
            return ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(
                crops.Select(crop => Recognition(
                    crop,
                    crop.RegionId == "x-tick" ? "O" : "Generalization",
                    0.96)).ToArray());
        });
        var numeric = new StubTextRecognizer((crops, cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            Assert.IsTrue(crops.All(static crop => crop.Width == 128 && crop.Height == 32));
            return ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(
                crops.Select(crop => Recognition(
                    crop,
                    crop.RegionId == "x-tick" ? "0" : "4",
                    0.98)).ToArray());
        });
        OcrDualRecognizerPipeline pipeline = Create(detector, general, numeric);

        OcrResult result = await pipeline.RecognizeAsync(OcrTestFixtures.Request(), CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.AreEqual(1, detector.CallCount);
        Assert.AreEqual("0", result.Regions.Single(region => region.RegionId == "x-tick").Text);
        Assert.AreEqual(OcrTextRole.XTick, result.Regions.Single(region => region.RegionId == "x-tick").Role);
        Assert.AreEqual(
            "Generalization",
            result.Regions.Single(region => region.RegionId == "phase-word").Text);
        Assert.AreEqual(
            OcrTextRole.PhaseHeading,
            result.Regions.Single(region => region.RegionId == "phase-word").Role);
        CollectionAssert.Contains(result.Warnings.ToArray(), "ocr_numeric_specialist_selected:x-tick");
        Assert.AreEqual(4, result.Cache.CropCount);
        Assert.AreEqual("0.1.0-dev", result.StageVersion);
    }

    [TestMethod]
    public async Task ExplicitParticipantRoleCannotBeReplacedByNumericNormalization()
    {
        OcrDetectedRegion detected = OcrTestFixtures.Region(
            "participant",
            143,
            70,
            14,
            8,
            context: new OcrRegionContext(
                InParticipantBand: true,
                ExplicitRoleHint: OcrTextRole.Participant));
        var detector = new StubTextRegionDetector([detected]);
        var general = Recognizer(("participant", "Chandler", 0.91));
        var numeric = Recognizer(("participant", "10", 0.99));
        OcrDualRecognizerPipeline pipeline = Create(detector, general, numeric);

        OcrResult result = await pipeline.RecognizeAsync(OcrTestFixtures.Request(), CancellationToken.None);

        Assert.AreEqual("Chandler", result.Regions.Single().Text);
        Assert.AreEqual(OcrTextRole.Participant, result.Regions.Single().Role);
        Assert.IsFalse(result.Warnings.Any(static warning =>
            warning.StartsWith("ocr_numeric_specialist_selected:", StringComparison.Ordinal)));
    }

    [TestMethod]
    public async Task GeneralResultSurvivesNumericRegionFailureWithDirectWarning()
    {
        OcrDetectedRegion detected = OcrTestFixtures.Region(
            "annotation",
            60,
            40,
            30,
            8,
            context: new OcrRegionContext(ExplicitRoleHint: OcrTextRole.Annotation));
        var detector = new StubTextRegionDetector([detected]);
        var general = Recognizer(("annotation", "Review", 0.90));
        var numeric = new StubTextRecognizer((crops, _) =>
            ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(
                crops.Select(crop => new OcrRecognition(
                    crop.RegionId,
                    crop.SourceImage,
                    Array.Empty<OcrRecognitionAlternative>(),
                    0,
                    Failure("OCR_NUMERIC_TEST_FAILURE"))).ToArray()));
        OcrDualRecognizerPipeline pipeline = Create(detector, general, numeric);

        OcrResult result = await pipeline.RecognizeAsync(OcrTestFixtures.Request(), CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.AreEqual("Review", result.Regions.Single().Text);
        Assert.HasCount(1, result.RegionFailures!);
        Assert.AreEqual("OCR_NUMERIC_TEST_FAILURE", result.RegionFailures![0].Failure.Code);
        CollectionAssert.Contains(
            result.Warnings.ToArray(),
            "ocr_region_failure:annotation:Original:OCR_NUMERIC_TEST_FAILURE");
    }

    [TestMethod]
    public async Task SuppliedRegionsBypassDetectorForFrozenFixtureExecution()
    {
        OcrDetectedRegion detected = OcrTestFixtures.Region(
            "numeric",
            46,
            89,
            10,
            6,
            context: new OcrRegionContext(NumericExpected: true));
        var detector = new StubTextRegionDetector((_, _) =>
            throw new AssertFailedException("Detector must not run when frozen regions are supplied."));
        OcrDualRecognizerPipeline pipeline = Create(
            detector,
            Recognizer(("numeric", "O", 0.8)),
            Recognizer(("numeric", "0", 0.9)));

        OcrResult result = await pipeline.RecognizeAsync(
            OcrTestFixtures.Request([detected]),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.AreEqual(0, detector.CallCount);
        Assert.AreEqual("0", result.Regions.Single().Text);
    }

    [TestMethod]
    public async Task DetectionFailureIsStructuredAndRecognizersDoNotRun()
    {
        var detector = new StubTextRegionDetector((_, _) =>
            throw new InvalidDataException("fixture detector failed"));
        var general = new StubTextRecognizer((_, _) =>
            throw new AssertFailedException("General recognizer must not run."));
        var numeric = new StubTextRecognizer((_, _) =>
            throw new AssertFailedException("Numeric recognizer must not run."));
        OcrDualRecognizerPipeline pipeline = Create(detector, general, numeric);

        OcrResult result = await pipeline.RecognizeAsync(OcrTestFixtures.Request(), CancellationToken.None);

        Assert.AreEqual("OCR_REGION_DETECTION_FAILED", result.Failure?.Code);
        Assert.AreEqual(0, general.CallCount);
        Assert.AreEqual(0, numeric.CallCount);
    }

    [TestMethod]
    public void ConfigurationFingerprintBindsBothRecognizersAndReviewedComposition()
    {
        OcrDualRecognizerPipeline pipeline = Create(
            new StubTextRegionDetector([]),
            Recognizer(("unused", "word", 0.9)),
            Recognizer(("unused", "1", 0.9)));

        Assert.AreEqual(
            OcrDualRecognizerPipelineOptions.ReviewedCompositionId,
            pipeline.CompositionId);
        Assert.AreEqual(64, pipeline.ConfigurationFingerprint.Length);
        Assert.IsTrue(pipeline.ConfigurationFingerprint.All(Uri.IsHexDigit));
        Assert.Throws<ArgumentException>(() => Create(
            new StubTextRegionDetector([]),
            Recognizer(("unused", "word", 0.9)),
            Recognizer(("unused", "1", 0.9)),
            new OcrDualRecognizerPipelineOptions { CompositionId = "unreviewed" }));
    }

    private static OcrDualRecognizerPipeline Create(
        ITextRegionDetector detector,
        ITextRecognizer general,
        ITextRecognizer numeric,
        OcrDualRecognizerPipelineOptions? options = null) =>
        new(
            detector,
            general,
            new InMemoryOcrResultCache(),
            new OcrPipelineOptions
            {
                StageVersion = "official-english-test",
                CropWidth = 320,
                CropHeight = 48,
                CropPaddingPixels = 0,
                CropPaddingValue = 0.5f,
            },
            numeric,
            new InMemoryOcrResultCache(),
            new OcrPipelineOptions
            {
                StageVersion = "numeric-specialist-test",
                CropWidth = 128,
                CropHeight = 32,
                CropPaddingPixels = 1,
                CropVerticalContentPaddingRatio = 0.25,
                CropPaddingValue = 1f,
            },
            options);

    private static StubTextRecognizer Recognizer(params (string RegionId, string Text, double Confidence)[] values)
    {
        Dictionary<string, (string Text, double Confidence)> byId = values.ToDictionary(
            static value => value.RegionId,
            static value => (value.Text, value.Confidence),
            StringComparer.Ordinal);
        return new StubTextRecognizer((crops, cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(
                crops.Select(crop =>
                {
                    (string text, double confidence) = byId[crop.RegionId];
                    return Recognition(crop, text, confidence);
                }).ToArray());
        });
    }

    private static OcrRecognition Recognition(OcrCrop crop, string text, double confidence) =>
        new(
            crop.RegionId,
            crop.SourceImage,
            [new OcrRecognitionAlternative(text, confidence, crop.SourceImage)],
            0.25);

    private static OcrFailure Failure(string code) =>
        new(code, "error", "Errors." + code, "fixture failure", true, "retry");
}
