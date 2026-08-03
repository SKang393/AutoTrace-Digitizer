// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using GraphReader.Markers.Detection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Detection;

[TestClass]
/// <summary>
/// Real raster preprocessing fixtures with scripted model outputs. These verify that visual
/// inputs, masks, transforms, and postprocessing compose correctly, not model visual accuracy.
/// </summary>
public sealed class SyntheticRasterFakeRunnerUnitTests
{
    [TestMethod]
    public async Task FakeInferenceUnitRealSyntheticRastersExerciseEveryRequiredVisualFamily()
    {
        foreach (SyntheticMarkerVisualScene scene in SyntheticMarkerVisualFixtures.OriginalScenes())
        {
            HeatmapPeak[] proposals =
            [
                .. scene.GoldenCenters.Select(center => new HeatmapPeak(
                    (int)center.X,
                    (int)center.Y,
                    0.95f,
                    3)),
                .. scene.HardNegativeCenters.Select(center => new HeatmapPeak(
                    (int)center.X,
                    (int)center.Y,
                    0.90f,
                    3,
                    0.05f)),
            ];
            if (scene.DuplicateProposal)
            {
                MarkerPoint center = scene.GoldenCenters.Single();
                proposals =
                [
                    .. proposals,
                    new HeatmapPeak((int)center.X + 5, (int)center.Y, 0.80f, 3),
                ];
            }

            var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Success(proposals));
            var detector = new MarkerCenterDetector(runner);
            MarkerDetectionResult result = await detector.DetectAsync(
                MarkerDetectionTestSupport.Request(original: scene.Frame, plot: scene.Plot),
                CancellationToken.None);

            Assert.IsTrue(result.Succeeded, $"{scene.Name}: {result.Failure?.TechnicalMessage}");
            Assert.HasCount(
                scene.GoldenCenters.Count,
                result.Markers,
                $"{scene.Name}: scripted model proposals did not survive production postprocessing exactly once.");
            foreach (MarkerPoint golden in scene.GoldenCenters)
            {
                Assert.AreEqual(
                    1,
                    result.Markers.Count(marker => marker.Center == golden),
                    $"{scene.Name}: expected exactly one center at {golden}.");
            }

            Assert.HasCount(1, runner.Requests);
            ReadOnlySpan<float> framePixels = scene.Frame.ChannelsFirstPixels.Span;
            Assert.IsTrue(framePixels.Contains(0f), $"{scene.Name}: fixture has no rendered ink.");
            Assert.IsTrue(framePixels.Contains(1f), $"{scene.Name}: fixture has no white background.");
            float[] inkPlane = runner.Requests[0].Input.Values.Span[..(64 * 64)].ToArray();
            Assert.IsTrue(inkPlane.Contains(1f), $"{scene.Name}: rendered ink did not enter the model tensor.");
            Assert.IsTrue(inkPlane.Contains(0f), $"{scene.Name}: model tensor lost the fixture background.");
        }
    }

    [TestMethod]
    public async Task FakeInferenceUnitRealTwoTimesRasterMapsToOriginalAcceptanceCenter()
    {
        SyntheticMarkerVisualScene scene = SyntheticMarkerVisualFixtures.Transform();
        MarkerPoint originalCenter = scene.GoldenCenters.Single();
        MarkerPoint frameCenter = scene.Frame.OriginalToFrame.MapFromOriginal(originalCenter);
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success([]),
            MarkerDetectionTestSupport.Success(
                [new HeatmapPeak((int)frameCenter.X, (int)frameCenter.Y, 0.95f, 3)]));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(
                original: MarkerDetectionTestSupport.Frame(MarkerSourceImage.Original),
                plot: scene.Plot,
                enhanced: scene.Frame),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Markers);
        MarkerDetectionTestSupport.AssertNear(originalCenter, result.Markers[0].Center);
        Assert.AreEqual(MarkerReviewState.NeedsReview, result.Markers[0].ReviewState);
        Assert.AreEqual(MarkerDisagreementKind.EnhancedOnly, result.Markers[0].Disagreement);
        Assert.AreEqual(InferenceProvider.Fake, result.Frames[1].Provider);
    }
}
