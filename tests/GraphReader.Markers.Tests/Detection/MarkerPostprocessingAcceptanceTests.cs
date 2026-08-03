// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using GraphReader.Markers.Detection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Detection;

[TestClass]
/// <summary>
/// Unit tests with scripted inference tensors. These verify runtime postprocessing,
/// not trained-model visual accuracy.
/// </summary>
public sealed class MarkerPostprocessingFakeRunnerUnitTests
{
    [TestMethod]
    public async Task ScriptedDenseHeatmapEmitsExactlyOneCenterPerGoldenPeak()
    {
        HeatmapPeak[] peaks =
        [
            new(6, 6),
            new(18, 6),
            new(30, 6),
            new(42, 6),
            new(54, 6),
            new(10, 22),
            new(26, 22),
            new(42, 22),
            new(14, 40),
            new(34, 40),
        ];
        var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Success(peaks));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(peaks.Length, result.Markers, "Golden markers must have one and only one center.");
        Assert.AreEqual(peaks.Length, result.Markers.Select(marker => marker.MarkerId).Distinct().Count());
        foreach (HeatmapPeak peak in peaks)
        {
            MarkerPoint expected = MarkerDetectionTestSupport.ExpectedCenter(peak.X, peak.Y);
            Assert.AreEqual(
                1,
                result.Markers.Count(marker => Distance(marker.Center, expected) <= 0.01),
                $"Golden marker at {expected} did not map to exactly one output center.");
        }

        Assert.AreEqual(MarkerContract.CoordinateSpace, result.CoordinateSpace);
        Assert.IsTrue(result.Markers.All(marker => marker.CoordinateSpace == MarkerContract.CoordinateSpace));
    }

    [TestMethod]
    public async Task ScriptedOpenTouchedAndFilledPeaksEachEmitOneCenter()
    {
        HeatmapPeak[] peaks = [new(14, 22), new(34, 34), new(50, 46)];
        MarkerPoint[] ink =
        [
            .. Line(10, 22, 34, 34),
            .. Line(34, 34, 50, 46),
            new(14, 22),
            new(34, 34),
            new(50, 46),
        ];
        MarkerImageFrame frame = MarkerDetectionTestSupport.Frame(
            MarkerSourceImage.Original,
            darkPixels: ink);
        var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Success(peaks));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(original: frame),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(3, result.Markers);
        foreach (HeatmapPeak peak in peaks)
        {
            MarkerPoint expected = MarkerDetectionTestSupport.ExpectedCenter(peak.X, peak.Y);
            Assert.AreEqual(1, result.Markers.Count(marker => Distance(marker.Center, expected) <= 0.01));
        }
    }

    [TestMethod]
    public async Task ScriptedArtifactHeadRejectsNamedHardNegativeClasses()
    {
        string[] kinds =
        [
            "arrowhead",
            "legend glyph",
            "text",
            "axis",
            "tick",
            "dotted divider",
            "bracket",
            "line intersection",
        ];
        HeatmapPeak[] hardNegatives = kinds
            .Select((_, index) => new HeatmapPeak(
                1 + ((index % 4) * 4),
                1 + ((index / 4) * 8),
                ArtifactProbability: 0.99f))
            .ToArray();
        var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Success(hardNegatives));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.IsEmpty(result.Markers, $"Hard negatives escaped artifact rejection: {string.Join(", ", kinds)}.");
        Assert.AreEqual(kinds.Length, result.Frames.Single().RawCandidateCount);
        Assert.AreEqual(0, result.Frames.Single().AcceptedCandidateCount);
    }

    [TestMethod]
    public async Task OcrAndArtifactMasksRejectTextAndGraphStructuresBeforeNms()
    {
        HeatmapPeak[] peaks =
        [
            new(2, 2),
            new(4, 4),
            new(6, 6),
            new(8, 8),
            new(10, 10),
            new(12, 12),
        ];
        MarkerPoint text = MarkerDetectionTestSupport.ExpectedCenter(2, 2);
        MarkerPoint[] graphStructures = peaks
            .Skip(1)
            .Select(peak => MarkerDetectionTestSupport.ExpectedCenter(peak.X, peak.Y))
            .ToArray();
        MarkerImageFrame frame = MarkerDetectionTestSupport.Frame(
            MarkerSourceImage.Original,
            ocrMask: MarkerDetectionTestSupport.Mask(text),
            artifactMask: MarkerDetectionTestSupport.Mask(graphStructures));
        var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Success(peaks));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(original: frame),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.IsEmpty(result.Markers, "OCR and artifact masks must reject every masked false positive.");
        Assert.HasCount(1, runner.Requests);
        CollectionAssert.AreEqual(
            new long[] { 1, 3, 64, 64 },
            runner.Requests[0].Input.Shape.ToArray(),
            "The model input must include image, OCR-mask, and artifact-mask channels.");
        Assert.IsTrue(runner.Requests[0].Input.Values.Span.Slice(64 * 64, 64 * 64).ToArray().Any(value => value > 0));
        Assert.IsTrue(runner.Requests[0].Input.Values.Span.Slice(2 * 64 * 64, 64 * 64).ToArray().Any(value => value > 0));
    }

    [TestMethod]
    public async Task ProductionArtifactThresholdRejectsPointFourRawOcrOrArtifactMaskEvidence()
    {
        HeatmapPeak identicalHead = new(30, 30, 0.95f, 3, 0.05f);
        InferenceResponse response = MarkerDetectionTestSupport.Success([identicalHead]);
        var runner = new MarkerInferenceRunnerStub(response, response, response);
        var detector = new MarkerCenterDetector(runner);
        MarkerDetectionOptions productionOptions = MarkerDetectionTestSupport.Options() with
        {
            CenterThreshold = 0.36f,
            ArtifactThreshold = 0.35f,
            MaskThreshold = 0.5f,
        };
        MarkerMask pointFourMask = MaskAt(new MarkerPoint(30, 30), 0.4f);
        MarkerImageFrame ocrMasked = MarkerDetectionTestSupport.Frame(
            MarkerSourceImage.Original,
            ocrMask: pointFourMask);
        MarkerImageFrame artifactMasked = MarkerDetectionTestSupport.Frame(
            MarkerSourceImage.Original,
            artifactMask: pointFourMask);

        MarkerDetectionResult control = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(options: productionOptions),
            CancellationToken.None);
        MarkerDetectionResult ocr = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(options: productionOptions, original: ocrMasked),
            CancellationToken.None);
        MarkerDetectionResult artifact = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(options: productionOptions, original: artifactMasked),
            CancellationToken.None);

        Assert.IsTrue(control.Succeeded, control.Failure?.TechnicalMessage);
        Assert.IsTrue(ocr.Succeeded, ocr.Failure?.TechnicalMessage);
        Assert.IsTrue(artifact.Succeeded, artifact.Failure?.TechnicalMessage);
        Assert.HasCount(1, control.Markers, "The otherwise identical unmasked candidate must be valid.");
        Assert.IsEmpty(ocr.Markers, "Raw OCR evidence 0.4 must exceed the 0.35 artifact threshold.");
        Assert.IsEmpty(artifact.Markers, "Raw artifact evidence 0.4 must exceed the 0.35 artifact threshold.");
        Assert.AreEqual(1, control.Frames.Single().RawCandidateCount);
        Assert.AreEqual(1, ocr.Frames.Single().RawCandidateCount);
        Assert.AreEqual(1, artifact.Frames.Single().RawCandidateCount);
        Assert.AreEqual(1, control.Frames.Single().AcceptedCandidateCount);
        Assert.AreEqual(0, ocr.Frames.Single().AcceptedCandidateCount);
        Assert.AreEqual(0, artifact.Frames.Single().AcceptedCandidateCount);
    }

    [TestMethod]
    public async Task NonRectangularPlotCropRejectsBoundingBoxFalsePositive()
    {
        var plot = new MarkerPolygon(
        [
            new MarkerPoint(8, 8),
            new MarkerPoint(56, 8),
            new MarkerPoint(8, 56),
        ]);
        HeatmapPeak inside = new(8, 8);
        HeatmapPeak outsideTriangle = new(60, 60);
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success([inside, outsideTriangle]));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(plot: plot),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Markers, "A candidate outside the plot polygon escaped plot masking.");
        MarkerDetectionTestSupport.AssertNear(new MarkerPoint(13.875, 13.875), result.Markers[0].Center);
        Assert.AreEqual("8,8;56,8;8,56", runner.Requests.Single().CacheMaterial.PanelCrop);
    }

    [TestMethod]
    public async Task SameColumnProbesRemainDistinctCenters()
    {
        HeatmapPeak[] sameColumn = [new(7, 8), new(7, 24), new(7, 40)];
        var runner = new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Success(sameColumn));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(3, result.Markers);
        Assert.AreEqual(1, result.Markers.Select(marker => marker.Center.X).Distinct().Count());
        Assert.AreEqual(3, result.Markers.Select(marker => marker.Center.Y).Distinct().Count());
    }

    [TestMethod]
    public async Task RadiusAwareNmsSuppressesLowerConfidenceDuplicate()
    {
        MarkerDetectionOptions options = MarkerDetectionTestSupport.Options();
        HeatmapPeak preferred = new(10, 10, 0.97f, 5);
        HeatmapPeak duplicate = new(20, 10, 0.81f, 5);
        var runner = new MarkerInferenceRunnerStub(
            MarkerDetectionTestSupport.Success([duplicate, preferred]));
        var detector = new MarkerCenterDetector(runner);

        MarkerDetectionResult result = await detector.DetectAsync(
            MarkerDetectionTestSupport.Request(options: options),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Markers, "NMS must emit exactly one center for a duplicate cluster.");
        MarkerDetectionTestSupport.AssertNear(
            MarkerDetectionTestSupport.ExpectedCenter(preferred.X, preferred.Y),
            result.Markers[0].Center);
        Assert.AreEqual(preferred.Confidence, result.Markers[0].CenterConfidence, 0.0001);
        Assert.AreEqual(2, result.Frames.Single().RawCandidateCount);
        Assert.AreEqual(1, result.Frames.Single().AcceptedCandidateCount);
    }

    private static double Distance(MarkerPoint left, MarkerPoint right)
    {
        var x = left.X - right.X;
        var y = left.Y - right.Y;
        return Math.Sqrt((x * x) + (y * y));
    }

    private static IEnumerable<MarkerPoint> Line(int x1, int y1, int x2, int y2)
    {
        var steps = Math.Max(Math.Abs(x2 - x1), Math.Abs(y2 - y1));
        for (var step = 0; step <= steps; step++)
        {
            var fraction = (double)step / steps;
            yield return new MarkerPoint(
                x1 + ((x2 - x1) * fraction),
                y1 + ((y2 - y1) * fraction));
        }
    }

    private static MarkerMask MaskAt(MarkerPoint point, float value)
    {
        var values = new float[64 * 64];
        values[((int)point.Y * 64) + (int)point.X] = value;
        return new MarkerMask(64, 64, values);
    }
}
