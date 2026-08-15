// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV9P2CandidateCompositionPipelineTests
{
    private static readonly string[] ExpectedRetainedRegionIds = ["rescue", "strong-direct"];

    [TestMethod]
    public async Task P2FiltersOnlyWeakSelectedTextOnTheHighDetectorRoute()
    {
        OcrDetectedRegion[] proposals =
        [
            Tick("strong-direct", 42, 0.96),
            Tick("weak-direct", 55, 0.98),
            Tick("rescue", 68, 0.92),
        ];
        var inner = CreateP1(
            proposals,
            Recognizer(
                ("strong-direct", "6", 0.80),
                ("weak-direct", "8", 0.70),
                ("rescue", "10", 0.70)),
            Recognizer());
        var candidate = new OcrV9P2CandidateCompositionPipeline(inner);

        OcrResult result = await candidate.RecognizeAsync(Request(), CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        CollectionAssert.AreEqual(
            ExpectedRetainedRegionIds,
            result.Regions.Select(static region => region.RegionId).Order(StringComparer.Ordinal).ToArray());
        CollectionAssert.AreEqual(
            ExpectedRetainedRegionIds,
            result.Masks.Select(static mask => mask.RegionId).Order(StringComparer.Ordinal).ToArray());
        CollectionAssert.Contains(
            result.Warnings.ToArray(),
            "ocr_v9_p2_candidate_selected_confidence_rejected:weak-direct");
        CollectionAssert.Contains(
            result.Warnings.ToArray(),
            "ocr_v9_p2_candidate_non_direct_route_retained:rescue");
        Assert.AreEqual(OcrV9P2CandidateCompositionPipeline.StageVersion, result.StageVersion);
        Assert.AreEqual(64, candidate.ConfigurationFingerprint.Length);
    }

    [TestMethod]
    public void P2RejectsAnyUnderlyingCompositionOtherThanTheImmutableP1Candidate()
    {
        OcrV8ProductionCompositionPipeline reviewed = Create(
            Array.Empty<OcrDetectedRegion>(),
            Recognizer(),
            Recognizer(),
            new OcrV8ProductionCompositionOptions());

        Assert.Throws<ArgumentException>(() => new OcrV9P2CandidateCompositionPipeline(reviewed));
    }

    private static OcrV8ProductionCompositionPipeline CreateP1(
        IReadOnlyList<OcrDetectedRegion> proposals,
        ITextRecognizer official,
        ITextRecognizer numeric) =>
        Create(
            proposals,
            official,
            numeric,
            new OcrV8ProductionCompositionOptions
            {
                CompositionId = OcrV8ProductionCompositionOptions.CandidateV9CompositionId,
                StageVersion = "0.0.21-v9-p1",
                OfficialMinimumConfidence = 0.65,
            });

    private static OcrV8ProductionCompositionPipeline Create(
        IReadOnlyList<OcrDetectedRegion> proposals,
        ITextRecognizer official,
        ITextRecognizer numeric,
        OcrV8ProductionCompositionOptions options) =>
        new(
            new StubProposalDetector(proposals),
            official,
            new InMemoryOcrResultCache(),
            PipelineOptions("p2-official", 320, 48, 8, 2, 0.5f),
            numeric,
            new InMemoryOcrResultCache(),
            PipelineOptions("p2-numeric", 128, 32, 12, 1, 1f),
            options);

    private static OcrPipelineOptions PipelineOptions(
        string version,
        int width,
        int height,
        int horizontalPadding,
        int verticalPadding,
        float paddingValue) =>
        new()
        {
            StageVersion = version,
            CropWidth = width,
            CropHeight = height,
            CropHorizontalPaddingPixels = horizontalPadding,
            CropVerticalPaddingPixels = verticalPadding,
            CropPaddingValue = paddingValue,
        };

    private static OcrRequest Request()
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
            new OcrRectangle(30, 15, 110, 70));
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
                if (!byId.TryGetValue(crop.RegionId, out (string Text, double Confidence) value))
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
        private readonly IReadOnlyList<OcrDetectedRegion> proposals;

        public StubProposalDetector(IReadOnlyList<OcrDetectedRegion> proposals) =>
            this.proposals = proposals;

        public string ConfigurationFingerprint => "v9-p2-proposal-test";

        public ValueTask<IReadOnlyList<OcrDetectedRegion>> DetectProposalsAsync(
            OcrImage image,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult(proposals);
        }

        public ValueTask<IReadOnlyList<OcrDetectedRegion>> DetectAsync(
            OcrImage image,
            CancellationToken cancellationToken) =>
            throw new AssertFailedException("P2 must retain the P1 proposal seam.");
    }
}
