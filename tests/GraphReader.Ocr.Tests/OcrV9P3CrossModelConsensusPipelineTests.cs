// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV9P3CrossModelConsensusPipelineTests
{
    private static readonly string[] ExpectedRetainedRegionIds = ["consensus", "strong"];

    [TestMethod]
    public async Task P3UsesV11RoleEvidenceToRescueOnlyBoundedWeakDirectText()
    {
        OcrDetectedRegion[] proposals =
        [
            Region("strong", 40, 0.98),
            Region("consensus", 60, 0.98),
            Region("weak", 80, 0.98),
        ];
        var inner = CreateP1(
            proposals,
            Recognizer(("strong", "Baseline", 0.80), ("consensus", "Learner", 0.70), ("weak", "noise", 0.70)),
            Recognizer());
        var roles = new StubRoleDetector(
        [
            RoleRegion("v11-strong", 40, OcrTextRole.PhaseHeading),
            RoleRegion("v11-consensus", 60, OcrTextRole.Participant),
        ]);
        var candidate = new OcrV9P3CrossModelConsensusPipeline(inner, roles);

        OcrResult result = await candidate.RecognizeAsync(Request(), CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        CollectionAssert.AreEqual(
            ExpectedRetainedRegionIds,
            result.Regions.Select(static item => item.RegionId).Order(StringComparer.Ordinal).ToArray());
        Assert.AreEqual(OcrTextRole.PhaseHeading, result.Regions.Single(item => item.RegionId == "strong").Role);
        Assert.AreEqual(OcrTextRole.Participant, result.Regions.Single(item => item.RegionId == "consensus").Role);
        CollectionAssert.AreEqual(
            ExpectedRetainedRegionIds,
            result.Masks.Select(static item => item.RegionId).Order(StringComparer.Ordinal).ToArray());
        CollectionAssert.Contains(result.Warnings.ToArray(), "ocr_v9_p3_cross_model_rejected:weak");
        Assert.AreEqual(OcrV9P3CrossModelConsensusPipeline.StageVersion, result.StageVersion);
        Assert.AreEqual(64, candidate.ConfigurationFingerprint.Length);
        Assert.AreEqual(1, roles.CallCount);
    }

    [TestMethod]
    public async Task P3RetainsNonDirectRouteWithoutInventingV11Role()
    {
        OcrDetectedRegion proposal = Region("rescue", 50, 0.90);
        var inner = CreateP1([proposal], Recognizer(("rescue", "10", 0.50)), Recognizer());
        var candidate = new OcrV9P3CrossModelConsensusPipeline(inner, new StubRoleDetector([]));

        OcrResult result = await candidate.RecognizeAsync(Request(), CancellationToken.None);

        Assert.AreEqual(1, result.Regions.Count);
        Assert.AreEqual("rescue", result.Regions[0].RegionId);
        CollectionAssert.Contains(result.Warnings.ToArray(), "ocr_v9_p3_non_direct_route_retained:rescue");
    }

    [TestMethod]
    public void P3RejectsAnyUnderlyingCompositionOtherThanImmutableP1()
    {
        OcrV8ProductionCompositionPipeline reviewed = Create(
            [], Recognizer(), Recognizer(), new OcrV8ProductionCompositionOptions());

        Assert.Throws<ArgumentException>(() =>
            new OcrV9P3CrossModelConsensusPipeline(reviewed, new StubRoleDetector([])));
    }

    private static OcrV8ProductionCompositionPipeline CreateP1(
        IReadOnlyList<OcrDetectedRegion> proposals,
        ITextRecognizer official,
        ITextRecognizer numeric) =>
        Create(proposals, official, numeric, new OcrV8ProductionCompositionOptions
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
            PipelineOptions("p3-official", 320, 48, 8, 2, 0.5f),
            numeric,
            new InMemoryOcrResultCache(),
            PipelineOptions("p3-numeric", 128, 32, 12, 1, 1f),
            options);

    private static OcrPipelineOptions PipelineOptions(
        string version, int width, int height, int horizontalPadding, int verticalPadding, float paddingValue) =>
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
            width, height, width, Enumerable.Repeat((byte)255, width * height).ToArray(),
            OcrSourceImage.Original, OcrFrameTransform.Identity,
            CanonicalOriginalWidth: width, CanonicalOriginalHeight: height);
        return new OcrRequest(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            new string('a', 64), image, new OcrRectangle(30, 15, 110, 70));
    }

    private static OcrDetectedRegion Region(string id, double x, double confidence) =>
        OcrTestFixtures.Region(id, x, 92, 8, 5, confidence: confidence);

    private static OcrDetectedRegion RoleRegion(string id, double x, OcrTextRole role) =>
        Region(id, x, 0.99) with { Context = new OcrRegionContext(ExplicitRoleHint: role) };

    private static StubTextRecognizer Recognizer(params (string Id, string Text, double Confidence)[] values)
    {
        Dictionary<string, (string Text, double Confidence)> byId = values.ToDictionary(
            static item => item.Id,
            static item => (item.Text, item.Confidence),
            StringComparer.Ordinal);
        return new StubTextRecognizer((crops, token) =>
        {
            token.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(crops.Select(crop =>
            {
                if (!byId.TryGetValue(crop.RegionId, out (string Text, double Confidence) value))
                {
                    return new OcrRecognition(crop.RegionId, crop.SourceImage, [], 0.1);
                }
                return new OcrRecognition(
                    crop.RegionId, crop.SourceImage,
                    [new OcrRecognitionAlternative(value.Text, value.Confidence, crop.SourceImage)], 0.1);
            }).ToArray());
        });
    }

    private sealed class StubProposalDetector(IReadOnlyList<OcrDetectedRegion> proposals)
        : ITextRegionProposalDetector
    {
        public string ConfigurationFingerprint => "v9-p3-primary-test";
        public ValueTask<IReadOnlyList<OcrDetectedRegion>> DetectProposalsAsync(
            OcrImage image, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult(proposals);
        }
        public ValueTask<IReadOnlyList<OcrDetectedRegion>> DetectAsync(
            OcrImage image, CancellationToken cancellationToken) =>
            throw new AssertFailedException("P3 must retain the primary proposal seam.");
    }

    private sealed class StubRoleDetector(IReadOnlyList<OcrDetectedRegion> regions) : ITextRegionDetector
    {
        public int CallCount { get; private set; }
        public string ConfigurationFingerprint => "v9-p3-role-test";
        public ValueTask<IReadOnlyList<OcrDetectedRegion>> DetectAsync(
            OcrImage image, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            CallCount++;
            return ValueTask.FromResult(regions);
        }
    }
}
