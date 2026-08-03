// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Markers.Detection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Detection;

[TestClass]
public sealed class MarkerRuntimeRepairRegressionTests
{
    [TestMethod]
    public async Task PythonEquivalentNineByNineLocalMaximumWindowUsesFourPixelRadius()
    {
        HeatmapPeak preferred = new(20, 20, 0.95f);
        HeatmapPeak suppressedWithinWindow = new(24, 24, 0.80f);
        HeatmapPeak retainedOutsideWindow = new(29, 25, 0.75f);
        var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Success(
            [preferred, suppressedWithinWindow, retainedOutsideWindow]));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.AreEqual(3, result.Frames.Single().RawCandidateCount);
        Assert.HasCount(2, result.Markers);
        Assert.IsTrue(result.Markers.Any(marker => marker.Center == new MarkerPoint(20, 20)));
        Assert.IsTrue(result.Markers.Any(marker => marker.Center == new MarkerPoint(29, 25)));
        Assert.IsFalse(result.Markers.Any(marker => marker.Center == new MarkerPoint(24, 24)));
    }

    [TestMethod]
    public async Task PythonEquivalentRadiusAwareNmsUsesStrictSuppressionBoundary()
    {
        HeatmapPeak preferred = new(10, 40, 0.95f, 5);
        HeatmapPeak insideRadiusBoundary = new(20, 40, 0.85f, 5);
        HeatmapPeak outsideRadiusBoundary = new(23, 45, 0.75f, 5);
        var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Success(
            [preferred, insideRadiusBoundary, outsideRadiusBoundary]));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(2, result.Markers);
        Assert.IsTrue(result.Markers.Any(marker => marker.Center == new MarkerPoint(10, 40)));
        Assert.IsTrue(result.Markers.Any(marker => marker.Center == new MarkerPoint(23, 45)));
        Assert.IsFalse(result.Markers.Any(marker => marker.Center == new MarkerPoint(20, 40)));
    }

    [TestMethod]
    public async Task AdversarialTwoByTwoConsensusFindsMaximumCardinalityMatching()
    {
        MarkerDetectionOptions options = MarkerDetectionTestSupport.Options() with
        {
            ConsensusToleranceOriginalPixels = 6,
        };
        HeatmapPeak[] original = [new(10, 10, 0.99f), new(20, 10, 0.98f)];
        HeatmapPeak[] enhanced = [new(14, 10, 0.97f), new(5, 10, 0.96f)];
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success(original),
            MarkerDetectionTestSupport.Success(enhanced));
        MarkerImageFrame enhancedFrame = MarkerDetectionTestSupport.Frame(MarkerSourceImage.Enhanced);
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(options: options, enhanced: enhancedFrame),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(2, result.Markers, "The two-by-two graph has a matching of cardinality two.");
        Assert.IsTrue(result.Markers.All(marker => marker.SourceImage == MarkerSourceImage.Consensus));
        Assert.IsTrue(result.Markers.All(marker => marker.ReviewState == MarkerReviewState.Unreviewed));
        Assert.IsTrue(result.Markers.All(marker => marker.Disagreement == MarkerDisagreementKind.None));
        Assert.IsEmpty(result.Warnings);
        Assert.IsTrue(result.Markers.Any(marker => marker.Center.X < 10));
        Assert.IsTrue(result.Markers.Any(marker => marker.Center.X > 15));
    }

    [TestMethod]
    public async Task FullCardinalityConsensusChoosesMinimumTotalEuclideanCostDeterministically()
    {
        MarkerDetectionOptions options = MarkerDetectionTestSupport.Options() with
        {
            ConsensusToleranceOriginalPixels = 25,
        };
        HeatmapPeak[] original = [new(10, 10, 0.90f), new(30, 10, 0.90f)];
        HeatmapPeak[] enhanced = [new(12, 10, 0.90f), new(28, 10, 0.90f)];
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success(original),
            MarkerDetectionTestSupport.Success(enhanced),
            MarkerDetectionTestSupport.Success(original),
            MarkerDetectionTestSupport.Success(enhanced));
        MarkerImageFrame enhancedFrame = MarkerDetectionTestSupport.Frame(MarkerSourceImage.Enhanced);
        var detector = new MarkerCenterDetector(runner);
        MarkerDetectionRequest request = MarkerDetectionTestSupport.Request(
            options: options,
            enhanced: enhancedFrame);

        MarkerDetectionResult first = await detector.DetectAsync(request, CancellationToken.None);
        MarkerDetectionResult second = await detector.DetectAsync(request, CancellationToken.None);

        Assert.IsTrue(first.Succeeded, first.Failure?.TechnicalMessage);
        Assert.IsTrue(second.Succeeded, second.Failure?.TechnicalMessage);
        CollectionAssert.AreEqual(
            new[] { new MarkerPoint(11, 10), new MarkerPoint(29, 10) },
            first.Markers.Select(marker => marker.Center).ToArray());
        CollectionAssert.AreEqual(
            first.Markers.Select(Signature).ToArray(),
            second.Markers.Select(Signature).ToArray(),
            "Minimum-cost matching must be deterministic across identical runs.");
    }

    [TestMethod]
    public async Task ConsensusDisagreementSetsPerMarkerReviewStateAndKind()
    {
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success([new HeatmapPeak(10, 10)]),
            MarkerDetectionTestSupport.Success([new HeatmapPeak(40, 40)]));
        MarkerImageFrame enhancedFrame = MarkerDetectionTestSupport.Frame(MarkerSourceImage.Enhanced);
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(enhanced: enhancedFrame),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(2, result.Markers);
        Assert.IsTrue(result.Markers.All(marker => marker.ReviewState == MarkerReviewState.NeedsReview));
        CollectionAssert.AreEquivalent(
            new[] { MarkerDisagreementKind.OriginalOnly, MarkerDisagreementKind.EnhancedOnly },
            result.Markers.Select(marker => marker.Disagreement).ToArray());
        CollectionAssert.Contains(result.Warnings.ToArray(), "original_enhanced_disagreement_requires_review");
    }

    [TestMethod]
    public async Task SingleOriginalEvidenceRemainsUnreviewedWithoutDisagreement()
    {
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success([new HeatmapPeak(18, 26)]));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Markers);
        Assert.AreEqual(MarkerReviewState.Unreviewed, result.Markers[0].ReviewState);
        Assert.AreEqual(MarkerDisagreementKind.None, result.Markers[0].Disagreement);
    }

    private static string Signature(MarkerCenter marker) => string.Join(
        "|",
        marker.Center.X.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        marker.Center.Y.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        marker.Radius.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        marker.CenterConfidence.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        marker.SourceImage,
        marker.ReviewState,
        marker.Disagreement);
}
