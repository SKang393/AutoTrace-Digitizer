// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV8ProductionCompositionPipelineTests
{
    private static readonly string[] ExpectedAcceptedRegionIds =
        ["direct", "official", "consensus", "zero"];

    [TestMethod]
    public async Task AppliesAllFourReviewedAcceptanceRoutesAndRejectsEverythingElse()
    {
        OcrDetectedRegion[] proposals =
        [
            Tick("direct", 42, 0.96),
            Tick("official", 55, 0.92),
            Tick("consensus", 68, 0.87),
            Tick("zero", 81, 0.83),
            Tick("nonzero-low", 94, 0.83),
            OcrTestFixtures.Region("word-low", 60, 30, 24, 8, confidence: 0.90),
        ];
        var detector = new StubProposalDetector(proposals);
        var official = Recognizer(
            ("direct", "Generalization", 0.90),
            ("official", "1", 0.91),
            ("consensus", "2", 0.89),
            ("zero", "0", 0.92),
            ("nonzero-low", "3", 0.90),
            ("word-low", "Note", 0.93));
        var numeric = Recognizer(
            ("direct", "", 0),
            ("official", "", 0),
            ("consensus", "2", 0.95),
            ("zero", "0", 0.99),
            ("nonzero-low", "3", 0.97),
            ("word-low", "", 0));
        OcrV8ProductionCompositionPipeline pipeline = Create(detector, official, numeric);

        OcrResult result = await pipeline.RecognizeAsync(Request(), CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        CollectionAssert.AreEquivalent(
            ExpectedAcceptedRegionIds,
            result.Regions.Select(static region => region.RegionId).ToArray());
        Assert.AreEqual("Generalization", result.Regions.Single(region => region.RegionId == "direct").Text);
        Assert.AreEqual("1", result.Regions.Single(region => region.RegionId == "official").Text);
        Assert.AreEqual("2", result.Regions.Single(region => region.RegionId == "consensus").Text);
        Assert.AreEqual("0", result.Regions.Single(region => region.RegionId == "zero").Text);
        Assert.HasCount(4, result.Masks);
        CollectionAssert.Contains(result.Warnings.ToArray(), "ocr_v8_acceptance_route:direct:detector");
        CollectionAssert.Contains(result.Warnings.ToArray(), "ocr_v8_acceptance_route:official:official_tick_rescue");
        CollectionAssert.Contains(result.Warnings.ToArray(), "ocr_v8_acceptance_route:consensus:official_numeric_consensus_rescue");
        CollectionAssert.Contains(result.Warnings.ToArray(), "ocr_v8_acceptance_route:zero:zero_numeric_consensus_rescue");
        CollectionAssert.Contains(result.Warnings.ToArray(), "ocr_v8_proposal_rejected:nonzero-low");
        CollectionAssert.Contains(result.Warnings.ToArray(), "ocr_v8_proposal_rejected:word-low");
        Assert.AreEqual(1, detector.ProposalCallCount);
        Assert.AreEqual(0, detector.NormalCallCount);
        Assert.AreEqual(12, result.Cache.CropCount);
        Assert.AreEqual(OcrV8ProductionCompositionOptions.ReviewedCompositionId, pipeline.CompositionId);
        Assert.AreEqual(64, pipeline.ConfigurationFingerprint.Length);
    }

    [TestMethod]
    public async Task SuppliedProposalsAreAcceptedOnlyByTheFrozenRulesWithoutDetectorExecution()
    {
        OcrDetectedRegion supplied = Tick("supplied", 64, 0.87);
        var detector = new StubProposalDetector((_, _) =>
            throw new AssertFailedException("Proposal detector must not run for supplied fixture regions."));
        OcrV8ProductionCompositionPipeline pipeline = Create(
            detector,
            Recognizer(("supplied", "4", 0.90)),
            Recognizer(("supplied", "5", 0.99)));

        OcrResult result = await pipeline.RecognizeAsync(
            Request([supplied]),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded);
        Assert.IsEmpty(result.Regions);
        Assert.AreEqual(0, detector.ProposalCallCount);
        CollectionAssert.Contains(result.Warnings.ToArray(), "ocr_v8_proposal_rejected:supplied");
    }

    [TestMethod]
    public void DriftedAcceptanceThresholdIsRejectedBeforeExecution()
    {
        var detector = new StubProposalDetector(Array.Empty<OcrDetectedRegion>());
        var official = Recognizer();
        var numeric = Recognizer();

        Assert.Throws<ArgumentException>(() => Create(
            detector,
            official,
            numeric,
            new OcrV8ProductionCompositionOptions { ZeroConsensusRescueThreshold = 0.81 }));
        Assert.Throws<ArgumentException>(() => Create(
            detector,
            official,
            numeric,
            new OcrV8ProductionCompositionOptions { CompositionId = "unreviewed" }));
    }

    [TestMethod]
    public async Task ProposalBelowFrozenFloorFailsClosedBeforeRecognition()
    {
        var detector = new StubProposalDetector([Tick("below", 50, 0.819)]);
        var official = Recognizer(("below", "0", 0.99));
        var numeric = Recognizer(("below", "0", 0.99));
        OcrV8ProductionCompositionPipeline pipeline = Create(detector, official, numeric);

        OcrResult result = await pipeline.RecognizeAsync(Request(), CancellationToken.None);

        Assert.AreEqual("OCR_REGION_DETECTION_FAILED", result.Failure?.Code);
        Assert.AreEqual(0, official.CallCount);
        Assert.AreEqual(0, numeric.CallCount);
    }

    private static OcrV8ProductionCompositionPipeline Create(
        ITextRegionProposalDetector detector,
        ITextRecognizer official,
        ITextRecognizer numeric,
        OcrV8ProductionCompositionOptions? options = null) =>
        new(
            detector,
            official,
            new InMemoryOcrResultCache(),
            new OcrPipelineOptions
            {
                StageVersion = "official-v8-test",
                CropWidth = 320,
                CropHeight = 48,
                CropHorizontalPaddingPixels = 8,
                CropVerticalPaddingPixels = 2,
                CropPaddingValue = 0.5f,
            },
            numeric,
            new InMemoryOcrResultCache(),
            new OcrPipelineOptions
            {
                StageVersion = "numeric-v5-test",
                CropWidth = 128,
                CropHeight = 32,
                CropHorizontalPaddingPixels = 12,
                CropVerticalPaddingPixels = 1,
                CropVerticalContentPaddingRatio = 0.25,
                CropPaddingValue = 1f,
            },
            options);

    private static OcrRequest Request(IReadOnlyList<OcrDetectedRegion>? supplied = null)
    {
        const int width = 160;
        const int height = 120;
        var image = new OcrImage(
            width,
            height,
            width,
            Enumerable.Repeat((byte)255, width * height).ToArray(),
            OcrSourceImage.Original,
            OcrFrameTransform.Identity,
            CanonicalOriginalWidth: width,
            CanonicalOriginalHeight: height);
        return new OcrRequest(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            new string('a', 64),
            image,
            new OcrRectangle(30, 15, 110, 70),
            DetectedRegions: supplied);
    }

    private static OcrDetectedRegion Tick(string id, double x, double confidence) =>
        OcrTestFixtures.Region(id, x, 92, 8, 5, confidence: confidence);

    private static StubTextRecognizer Recognizer(
        params (string RegionId, string Text, double Confidence)[] values)
    {
        Dictionary<string, (string Text, double Confidence)> byId = values.ToDictionary(
            static value => value.RegionId,
            static value => (value.Text, value.Confidence),
            StringComparer.Ordinal);
        return new StubTextRecognizer((crops, cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(crops.Select(crop =>
            {
                if (!byId.TryGetValue(crop.RegionId, out (string Text, double Confidence) value) ||
                    string.IsNullOrEmpty(value.Text))
                {
                    return new OcrRecognition(
                        crop.RegionId,
                        crop.SourceImage,
                        Array.Empty<OcrRecognitionAlternative>(),
                        0.1);
                }

                return new OcrRecognition(
                    crop.RegionId,
                    crop.SourceImage,
                    [new OcrRecognitionAlternative(value.Text, value.Confidence, crop.SourceImage)],
                    0.1);
            }).ToArray());
        });
    }

    private sealed class StubProposalDetector : ITextRegionProposalDetector
    {
        private readonly Func<OcrImage, CancellationToken, ValueTask<IReadOnlyList<OcrDetectedRegion>>> detect;
        private int proposalCallCount;
        private int normalCallCount;

        public StubProposalDetector(IReadOnlyList<OcrDetectedRegion> regions)
            : this((_, _) => ValueTask.FromResult(regions))
        {
        }

        public StubProposalDetector(
            Func<OcrImage, CancellationToken, ValueTask<IReadOnlyList<OcrDetectedRegion>>> detect) =>
            this.detect = detect;

        public int ProposalCallCount => Volatile.Read(ref proposalCallCount);

        public int NormalCallCount => Volatile.Read(ref normalCallCount);

        public string ConfigurationFingerprint => "v8-proposal-test";

        public ValueTask<IReadOnlyList<OcrDetectedRegion>> DetectProposalsAsync(
            OcrImage image,
            CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref proposalCallCount);
            return detect(image, cancellationToken);
        }

        public ValueTask<IReadOnlyList<OcrDetectedRegion>> DetectAsync(
            OcrImage image,
            CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref normalCallCount);
            throw new AssertFailedException("V8 composition must use the proposal seam.");
        }
    }
}
