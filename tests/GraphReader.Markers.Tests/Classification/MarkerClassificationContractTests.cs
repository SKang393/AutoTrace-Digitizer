// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using GraphReader.Markers.Classification;
using GraphReader.Markers.Detection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Classification;

[TestClass]
public sealed class MarkerClassificationContractTests
{
    private static readonly string[] ExpectedFillOutputOrder = ["filled", "open", "unknown"];

    [TestMethod]
    public void ShapeAndFillContractsExposeEveryRequiredIndependentClass()
    {
        CollectionAssert.AreEqual(
            new[]
            {
                nameof(MarkerShape.Circle),
                nameof(MarkerShape.Square),
                nameof(MarkerShape.TriangleUp),
                nameof(MarkerShape.TriangleDown),
                nameof(MarkerShape.Diamond),
                nameof(MarkerShape.Star),
                nameof(MarkerShape.Asterisk),
                nameof(MarkerShape.Cross),
                nameof(MarkerShape.Other),
            },
            Enum.GetNames<MarkerShape>());
        CollectionAssert.AreEqual(
            new[]
            {
                nameof(MarkerFill.Filled),
                nameof(MarkerFill.Open),
                nameof(MarkerFill.Unknown),
            },
            Enum.GetNames<MarkerFill>());
        Assert.AreEqual(
            MarkerClassificationContract.ShapeClassCount,
            Enum.GetValues<MarkerShape>().Length);
        Assert.AreEqual(
            MarkerClassificationContract.FillClassCount,
            Enum.GetValues<MarkerFill>().Length);
    }

    [TestMethod]
    public void TensorContractKeepsShapeFillArtifactAndEmbeddingOutputsDisjoint()
    {
        MarkerClassifierTensorContract contract = ClassificationTestSupport.TensorContract();

        Assert.AreEqual(0, MarkerClassifierTensorContract.ShapeOffset);
        Assert.AreEqual(9, MarkerClassifierTensorContract.FillOffset);
        Assert.AreEqual(12, MarkerClassifierTensorContract.ArtifactOffset);
        Assert.AreEqual(13, MarkerClassifierTensorContract.EmbeddingOffset);
        Assert.AreEqual(21, contract.ValuesPerMarker);
        Assert.AreEqual(1, contract.InputChannelCount);
        Assert.AreEqual(0f, contract.NormalizeMean);
        Assert.AreEqual(1f, contract.NormalizeScale);
        Assert.AreEqual(MarkerClassifierOutputEncoding.Logits, contract.OutputEncoding);
        StringAssert.StartsWith(MarkerClassificationContract.ShapeOutputOrder, "circle,square,");
        CollectionAssert.AreEqual(
            ExpectedFillOutputOrder,
            MarkerClassificationContract.FillOutputOrder.Split(','));
    }

    [TestMethod]
    public void EveryShapeAndFillCombinationHasDistinctVisibleAndAccessibleIdentity()
    {
        var symbols = new HashSet<string>(StringComparer.Ordinal);
        foreach (MarkerShape shape in Enum.GetValues<MarkerShape>())
        {
            foreach (MarkerFill fill in Enum.GetValues<MarkerFill>())
            {
                MarkerSymbolDescriptor descriptor = MarkerSymbolMap.Describe(shape, fill);
                string accessibleName = descriptor.AccessibleName.ToLowerInvariant();

                Assert.IsFalse(string.IsNullOrWhiteSpace(descriptor.Symbol));
                Assert.IsFalse(string.IsNullOrWhiteSpace(descriptor.AccessibleName));
                StringAssert.Contains(accessibleName, ShapeWord(shape));
                StringAssert.Contains(accessibleName, FillWord(fill));
                Assert.IsTrue(
                    symbols.Add(descriptor.Symbol),
                    $"Symbol '{descriptor.Symbol}' is reused and cannot independently identify {shape}/{fill}.");
                Assert.AreEqual(descriptor.Symbol, MarkerSymbolMap.GetSymbol(shape, fill));
                Assert.AreEqual(descriptor.AccessibleName, MarkerSymbolMap.GetAccessibleName(shape, fill));
            }
        }

        Assert.HasCount(27, symbols);
        Assert.AreEqual("●", MarkerSymbolMap.GetSymbol(MarkerShape.Circle, MarkerFill.Filled));
        Assert.AreEqual("○", MarkerSymbolMap.GetSymbol(MarkerShape.Circle, MarkerFill.Open));
        Assert.AreEqual("■", MarkerSymbolMap.GetSymbol(MarkerShape.Square, MarkerFill.Filled));
        Assert.AreEqual("□", MarkerSymbolMap.GetSymbol(MarkerShape.Square, MarkerFill.Open));
        Assert.AreEqual("▲", MarkerSymbolMap.GetSymbol(MarkerShape.TriangleUp, MarkerFill.Filled));
        Assert.AreEqual("▼", MarkerSymbolMap.GetSymbol(MarkerShape.TriangleDown, MarkerFill.Filled));
        Assert.AreEqual("◆", MarkerSymbolMap.GetSymbol(MarkerShape.Diamond, MarkerFill.Filled));
        Assert.AreEqual("★", MarkerSymbolMap.GetSymbol(MarkerShape.Star, MarkerFill.Filled));
        Assert.AreEqual("✱", MarkerSymbolMap.GetSymbol(MarkerShape.Asterisk, MarkerFill.Filled));
    }

    [TestMethod]
    public void SymbolMapRejectsUndefinedShapeAndFillValues()
    {
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(
            () => MarkerSymbolMap.Describe((MarkerShape)999, MarkerFill.Filled));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(
            () => MarkerSymbolMap.Describe(MarkerShape.Circle, (MarkerFill)999));
    }

    [TestMethod]
    public void RequestDefensivelyCopiesPixelsMasksAndMarkerSequence()
    {
        float[] pixels = Enumerable.Repeat(1f, ClassificationTestSupport.FrameSizeSquared).ToArray();
        float[] ocr = new float[ClassificationTestSupport.FrameSizeSquared];
        float[] artifacts = new float[ClassificationTestSupport.FrameSizeSquared];
        var markers = new List<MarkerCenter> { ClassificationTestSupport.Marker("m1", 8, 9) };
        MarkerImageFrame frame = ClassificationTestSupport.Frame(pixels, ocr, artifacts);

        MarkerClassificationRequest request = ClassificationTestSupport.Request(frame, markers);
        pixels[0] = 0;
        ocr[1] = 1;
        artifacts[2] = 1;
        markers.Add(ClassificationTestSupport.Marker("m2", 15, 16));

        Assert.AreEqual(1f, request.Image.ChannelsFirstPixels.Span[0]);
        Assert.AreEqual(0f, request.Image.OcrMask.Values.Span[1]);
        Assert.AreEqual(0f, request.Image.ArtifactMask.Values.Span[2]);
        Assert.HasCount(1, request.Markers);
        Assert.AreEqual("m1", request.Markers[0].MarkerId);
    }

    [TestMethod]
    public void ClassifiedMarkerDefensivelyCopiesNormalizedEmbedding()
    {
        float normalization = 1f / MathF.Sqrt(2f);
        var embedding = new List<float> { normalization, normalization };
        var classified = new ClassifiedMarker(
            ClassificationTestSupport.Marker("m1", 8, 9),
            MarkerShape.Circle,
            MarkerFill.Open,
            "○",
            "Open circle",
            0.01,
            0.92,
            0.96,
            embedding);

        embedding[0] = 99;

        Assert.AreEqual(normalization, classified.Embedding[0], 1e-6);
        Assert.AreEqual(1, Math.Sqrt(classified.Embedding.Sum(value => value * value)), 1e-6);
        Assert.AreEqual(0.92, classified.Confidence, 1e-12);
        Assert.AreEqual(MarkerShape.Circle, classified.Shape);
        Assert.AreEqual(MarkerFill.Open, classified.Fill);
    }

    [TestMethod]
    public void ResultDefensivelyCopiesMarkersWarningsAndBatchReports()
    {
        var marker = new ClassifiedMarker(
            ClassificationTestSupport.Marker("m1", 8, 9),
            MarkerShape.Square,
            MarkerFill.Filled,
            "■",
            "Filled square",
            0.01,
            0.9,
            0.8,
            [1f, 0f]);
        var markers = new List<ClassifiedMarker> { marker };
        var warnings = new List<string> { "NeedsReview" };
        var batches = new List<MarkerClassificationBatchReport>
        {
            new(
                0,
                1,
                InferenceProvider.Fake,
                [new ProviderAttempt(InferenceProvider.Fake, true, null)],
                new MarkerClassificationTiming(1, 2, 1, 4),
                false,
                null),
        };
        var result = new MarkerClassificationResult(
            MarkerClassificationContract.Version,
            Guid.NewGuid().ToString(),
            "project",
            "panel",
            MarkerClassificationContract.Stage,
            "0.1.0-test",
            new string('a', 64),
            MarkerClassificationContract.CoordinateSpace,
            markers,
            new MarkerClassificationTiming(1, 2, 1, 4),
            0.8,
            warnings,
            batches,
            new MarkerClassificationModelReport("classifier", "0.1.0", new string('b', 64), InferenceProvider.Fake),
            null);

        markers.Clear();
        warnings.Clear();
        batches.Clear();

        Assert.HasCount(1, result.Markers);
        Assert.HasCount(1, result.Warnings);
        Assert.HasCount(1, result.Batches);
        Assert.IsTrue(result.Succeeded);
        Assert.AreEqual(MarkerContract.CoordinateSpace, result.CoordinateSpace);
    }

    private static string ShapeWord(MarkerShape shape) => shape switch
    {
        MarkerShape.TriangleUp or MarkerShape.TriangleDown => "triangle",
        _ => shape.ToString().ToLowerInvariant(),
    };

    private static string FillWord(MarkerFill fill) => fill switch
    {
        MarkerFill.Unknown => "unknown fill",
        _ => fill.ToString().ToLowerInvariant(),
    };
}

internal static class ClassificationTestSupport
{
    internal const int FrameSize = 32;
    internal const int FrameSizeSquared = FrameSize * FrameSize;

    internal static readonly ModelIdentity Model = new(
        "graph-marker-classifier",
        "0.1.0",
        new string('c', 64),
        "marker-classifier.onnx");

    internal static MarkerClassifierTensorContract TensorContract() =>
        new("patches", "predictions", 24, 24, 1, 8);

    internal static MarkerClassificationOptions Options() =>
        new(TensorContract())
        {
            BatchSize = 4,
            Timeout = TimeSpan.FromSeconds(2),
            StageVersion = "0.1.0-test",
        };

    internal static MarkerClassificationRequest Request(
        MarkerImageFrame? frame = null,
        IEnumerable<MarkerCenter>? markers = null,
        MarkerClassificationOptions? options = null) =>
        new(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            new string('d', 64),
            Model,
            frame ?? Frame(),
            markers ?? [Marker("m1", 8, 9)],
            options ?? Options());

    internal static MarkerCenter Marker(string id, double x, double y, double radius = 3) =>
        new(id, new MarkerPoint(x, y), radius, 0.01, 0.95, MarkerSourceImage.Original);

    internal static MarkerImageFrame Frame(
        float[]? pixels = null,
        float[]? ocr = null,
        float[]? artifacts = null) =>
        new(
            FrameSize,
            FrameSize,
            1,
            pixels ?? Enumerable.Repeat(1f, FrameSizeSquared).ToArray(),
            MarkerSourceImage.Original,
            MarkerAffineTransform.Identity,
            new MarkerMask(FrameSize, FrameSize, ocr ?? new float[FrameSizeSquared]),
            new MarkerMask(FrameSize, FrameSize, artifacts ?? new float[FrameSizeSquared]));

    internal static MarkerImageFrame Frame(int channelCount, float[] pixels) =>
        new(
            FrameSize,
            FrameSize,
            channelCount,
            pixels,
            MarkerSourceImage.Original,
            MarkerAffineTransform.Identity,
            MarkerMask.Empty(FrameSize, FrameSize),
            MarkerMask.Empty(FrameSize, FrameSize));
}
