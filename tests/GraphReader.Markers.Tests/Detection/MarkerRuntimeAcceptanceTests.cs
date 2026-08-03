// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using GraphReader.Markers.Detection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Detection;

[TestClass]
/// <summary>
/// Unit tests with scripted inference responses. Real ONNX provider execution is covered separately.
/// </summary>
public sealed class MarkerRuntimeFakeRunnerUnitTests
{
    [TestMethod]
    public async Task EnhancedOnlyDetectionMapsBackToOriginalPixels()
    {
        HeatmapPeak peak = new(20, 28, 0.96f);
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success([]),
            MarkerDetectionTestSupport.Success([peak]));
        MarkerImageFrame enhanced = MarkerDetectionTestSupport.Frame(
            MarkerSourceImage.Enhanced,
            new MarkerAffineTransform(2, 0, 0, 0, 2, 0));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(
                plot: MarkerPolygon.FromRectangle(new MarkerRectangle(0, 0, 32, 32)),
                enhanced: enhanced),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Markers);
        MarkerDetectionTestSupport.AssertNear(new MarkerPoint(10, 14), result.Markers[0].Center);
        Assert.AreEqual(MarkerContract.CoordinateSpace, result.Markers[0].CoordinateSpace);
        Assert.AreEqual(MarkerSourceImage.Enhanced, result.Markers[0].SourceImage);
        Assert.AreEqual(MarkerReviewState.NeedsReview, result.Markers[0].ReviewState);
        Assert.AreEqual(MarkerDisagreementKind.EnhancedOnly, result.Markers[0].Disagreement);
        Assert.AreEqual(0.96 * 0.75, result.Markers[0].CenterConfidence, 0.0001);
        CollectionAssert.Contains(result.Warnings.ToArray(), "original_enhanced_disagreement_requires_review");
    }

    [TestMethod]
    public async Task OriginalEnhancedAgreementProducesOneConsensusCenter()
    {
        HeatmapPeak originalPeak = new(21, 29, 0.94f);
        HeatmapPeak enhancedPeak = new(20, 28, 0.90f);
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success([originalPeak], provider: InferenceProvider.Cpu),
            MarkerDetectionTestSupport.Success([enhancedPeak], provider: InferenceProvider.Cpu));
        MarkerImageFrame enhanced = MarkerDetectionTestSupport.Frame(
            MarkerSourceImage.Enhanced,
            new MarkerAffineTransform(2, 0, 0, 0, 2, 0));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(
                plot: MarkerPolygon.FromRectangle(new MarkerRectangle(0, 0, 32, 32)),
                enhanced: enhanced),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Markers, "Original and enhanced evidence must not duplicate a marker.");
        double expected = ((10.25 * 0.94) + (10 * 0.90)) / (0.94 + 0.90);
        double expectedY = ((14.25 * 0.94) + (14 * 0.90)) / (0.94 + 0.90);
        MarkerDetectionTestSupport.AssertNear(new MarkerPoint(expected, expectedY), result.Markers[0].Center);
        Assert.AreEqual(MarkerSourceImage.Consensus, result.Markers[0].SourceImage);
        Assert.AreEqual(MarkerReviewState.Unreviewed, result.Markers[0].ReviewState);
        Assert.AreEqual(MarkerDisagreementKind.None, result.Markers[0].Disagreement);
        double disagreementDistance = Math.Sqrt((0.25 * 0.25) + (0.25 * 0.25));
        double expectedConfidence = ((0.94 + 0.90) / 2) *
            (1 - (0.2 * (disagreementDistance / 5)));
        Assert.AreEqual(expectedConfidence, result.Markers[0].CenterConfidence, 0.0001);
        Assert.IsEmpty(result.Warnings);
        Assert.AreEqual(InferenceProvider.Cpu, result.Model.Provider);
    }

    [TestMethod]
    public async Task EnhancedEvidenceSurvivesRecoverableOriginalFrameFailureWithWarning()
    {
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Failure("ORIGINAL_PROVIDER_FAILED"),
            MarkerDetectionTestSupport.Success([new HeatmapPeak(20, 28)], provider: InferenceProvider.Cpu));
        MarkerImageFrame enhanced = MarkerDetectionTestSupport.Frame(
            MarkerSourceImage.Enhanced,
            new MarkerAffineTransform(2, 0, 0, 0, 2, 0));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(
                plot: MarkerPolygon.FromRectangle(new MarkerRectangle(0, 0, 32, 32)),
                enhanced: enhanced),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Markers);
        Assert.AreEqual(MarkerReviewState.NeedsReview, result.Markers[0].ReviewState);
        Assert.AreEqual(MarkerDisagreementKind.EnhancedOnly, result.Markers[0].Disagreement);
        Assert.HasCount(2, result.Frames);
        Assert.AreEqual("ORIGINAL_PROVIDER_FAILED", result.Frames[0].Failure?.Code);
        CollectionAssert.Contains(result.Warnings.ToArray(), "original_frame_failed:ORIGINAL_PROVIDER_FAILED");
        CollectionAssert.Contains(result.Warnings.ToArray(), "enhanced_only_evidence_requires_review");
    }

    [TestMethod]
    public async Task CancellationPropagatesAndStopsInference()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var runner = new MarkerInferenceRunnerStub(async (_, cancellationToken) =>
        {
            entered.SetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            return MarkerDetectionTestSupport.Success([]);
        });
        var detector = new MarkerCenterDetector(runner);
        using var cancellation = new CancellationTokenSource();

        Task<MarkerDetectionResult> detection = detector
            .DetectAsync(MarkerDetectionTestSupport.Request(), cancellation.Token)
            .AsTask();
        await entered.Task;
        await cancellation.CancelAsync();

        await Assert.ThrowsAsync<OperationCanceledException>(() => detection);
        Assert.HasCount(1, runner.Requests);
    }

    [TestMethod]
    public async Task InvalidFrameReturnsStructuredFailureWithoutInference()
    {
        var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Success([]));
        MarkerImageFrame valid = MarkerDetectionTestSupport.Frame(MarkerSourceImage.Original);
        MarkerImageFrame invalid = valid with
        {
            OcrMask = new MarkerMask(64, 64, new float[3]),
        };
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(original: invalid),
            CancellationToken.None);

        Assert.IsFalse(result.Succeeded);
        Assert.AreEqual("MARKER_REQUEST_INVALID", result.Failure?.Code);
        Assert.IsFalse(result.Failure?.Recoverable);
        Assert.IsEmpty(result.Markers);
        Assert.IsEmpty(runner.Requests, "Invalid input must fail before model inference.");
    }

    [TestMethod]
    public async Task InferenceFailureIsReturnedThroughStructuredErrorContract()
    {
        var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Failure("MODEL_NOT_FOUND"));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(),
            CancellationToken.None);

        Assert.IsFalse(result.Succeeded);
        Assert.AreEqual("MODEL_NOT_FOUND", result.Failure?.Code);
        Assert.IsTrue(result.Failure?.Recoverable);
        Assert.AreEqual("retry", result.Failure?.SuggestedAction);
        Assert.HasCount(1, result.Frames);
        Assert.AreEqual("MODEL_NOT_FOUND", result.Frames[0].Failure?.Code);
        Assert.IsEmpty(result.Markers);
    }

    [TestMethod]
    public async Task StrideOneFlatOutputPreservesExactIntegerCoordinates()
    {
        HeatmapPeak[] peaks = [new(3, 4, 0.91f, 1.25f), new(11, 9, 0.87f, 1.75f)];
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success(peaks, MarkerTensorLayout.ChannelsFirst));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(options: MarkerDetectionTestSupport.Options()),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(2, result.Markers);
        MarkerDetectionTestSupport.AssertNear(new MarkerPoint(3, 4), result.Markers[0].Center);
        MarkerDetectionTestSupport.AssertNear(new MarkerPoint(11, 9), result.Markers[1].Center);
        Assert.IsTrue(result.Markers.All(marker => marker.Radius >= 2.5));
    }

    [TestMethod]
    public async Task FlatOutputLengthMismatchIsRejectedAsModelLayoutFailure()
    {
        var execution = new InferenceExecution(
            new float[17],
            InferenceProvider.Cpu,
            new StageTiming(0, 0.1, 0, 0.1, 0, false, false),
            new MemoryDiagnostics(0, 0, 0, 0, 0));
        var response = new InferenceResponse(
            true,
            execution,
            null,
            [new ProviderAttempt(InferenceProvider.Cpu, true, null)]);
        var detector = new MarkerCenterDetector(new MarkerInferenceRunnerStub(response));

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(),
            CancellationToken.None);

        Assert.IsFalse(result.Succeeded);
        Assert.AreEqual("MARKER_MODEL_OUTPUT_SHAPE_MISMATCH", result.Failure?.Code);
        Assert.IsFalse(result.Failure?.Recoverable);
        Assert.AreEqual("select_compatible_model", result.Failure?.SuggestedAction);
    }

    [TestMethod]
    public async Task UndeclaredTensorLayoutIsRejectedBeforeInference()
    {
        MarkerModelTensorContract invalidContract = MarkerDetectionTestSupport.Contract() with
        {
            OutputLayout = (MarkerTensorLayout)999,
        };
        var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Success([]));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(options: new MarkerDetectionOptions(invalidContract)),
            CancellationToken.None);

        Assert.IsFalse(result.Succeeded);
        Assert.AreEqual("MARKER_REQUEST_INVALID", result.Failure?.Code);
        StringAssert.Contains(result.Failure?.TechnicalMessage, "NCHW");
        Assert.IsEmpty(runner.Requests);
    }

    [TestMethod]
    public async Task CpuFallbackAndDirectMlResultsAreProviderIndependentWithinTolerance()
    {
        HeatmapPeak[] peaks = [new(4, 6, 0.93f), new(20, 18, 0.89f)];
        ProviderAttempt[] fallbackAttempts =
        [
            new(InferenceProvider.DirectMl, false, "No compatible GPU adapter."),
            new(InferenceProvider.Cpu, true, null),
        ];
        var directMlDetector = new MarkerCenterDetector(new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success(peaks, provider: InferenceProvider.DirectMl)));
        var fallbackDetector = new MarkerCenterDetector(new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success(
                peaks,
                provider: InferenceProvider.Cpu,
                attempts: fallbackAttempts)));

        MarkerDetectionResult directMl = await directMlDetector.DetectAsync(
            MarkerDetectionTestSupport.Request(),
            CancellationToken.None);
        MarkerDetectionResult fallback = await fallbackDetector.DetectAsync(
            MarkerDetectionTestSupport.Request(),
            CancellationToken.None);

        Assert.IsTrue(directMl.Succeeded, directMl.Failure?.TechnicalMessage);
        Assert.IsTrue(fallback.Succeeded, fallback.Failure?.TechnicalMessage);
        Assert.AreEqual(InferenceProvider.DirectMl, directMl.Model.Provider);
        Assert.AreEqual(InferenceProvider.Cpu, fallback.Model.Provider);
        CollectionAssert.AreEqual(fallbackAttempts, fallback.Frames.Single().ProviderAttempts.ToArray());
        Assert.HasCount(directMl.Markers.Count, fallback.Markers);
        for (var index = 0; index < directMl.Markers.Count; index++)
        {
            MarkerPoint expected = directMl.Markers[index].Center;
            MarkerPoint actual = fallback.Markers[index].Center;
            var distance = Math.Sqrt(
                Math.Pow(expected.X - actual.X, 2) +
                Math.Pow(expected.Y - actual.Y, 2));
            Assert.IsLessThanOrEqualTo(0.01, distance, "Provider output exceeded center tolerance.");
        }
    }

    [TestMethod]
    public async Task CacheIdentityIsStableAndInvalidatesOnModelTransformOrMaskChange()
    {
        InferenceResponse[] responses = Enumerable.Range(0, 5)
            .Select(index => MarkerDetectionTestSupport.Success(
                [new HeatmapPeak(4, 4)],
                cacheHit: index == 1))
            .ToArray();
        var runner = new MarkerInferenceRunnerStub(responses);
        var detector = new MarkerCenterDetector(runner);
        MarkerDetectionRequest baseline = MarkerDetectionTestSupport.Request();
        ModelIdentity revisedModel = MarkerDetectionTestSupport.Model with { Sha256 = new string('c', 64) };
        MarkerPoint maskedPoint = MarkerDetectionTestSupport.ExpectedCenter(2, 2);
        MarkerImageFrame changedMask = MarkerDetectionTestSupport.Frame(
            MarkerSourceImage.Original,
            ocrMask: MarkerDetectionTestSupport.Mask(maskedPoint));

        MarkerDetectionResult cold = await detector.DetectAsync(baseline, CancellationToken.None);
        MarkerDetectionResult warm = await detector.DetectAsync(baseline, CancellationToken.None);
        MarkerDetectionResult changedModel = await detector.DetectAsync(
            baseline with { Model = revisedModel },
            CancellationToken.None);
        MarkerDetectionResult changedTransform = await detector.DetectAsync(
            baseline with { TransformChain = "deskew:1.25" },
            CancellationToken.None);
        MarkerDetectionResult changedMaskResult = await detector.DetectAsync(
            baseline with { OriginalImage = changedMask },
            CancellationToken.None);

        string coldKey = cold.Frames.Single().CacheKey;
        Assert.AreEqual(coldKey, warm.Frames.Single().CacheKey);
        Assert.IsFalse(cold.Frames.Single().CacheHit);
        Assert.IsTrue(warm.Frames.Single().CacheHit);
        Assert.AreNotEqual(coldKey, changedModel.Frames.Single().CacheKey);
        Assert.AreNotEqual(coldKey, changedTransform.Frames.Single().CacheKey);
        Assert.AreNotEqual(coldKey, changedMaskResult.Frames.Single().CacheKey);
        Assert.IsTrue(new[] { cold, warm, changedModel, changedTransform, changedMaskResult }
            .All(result => result.Succeeded));
    }
}
