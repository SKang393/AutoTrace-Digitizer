// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Globalization;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class NumericTickConflictTests
{
    private static readonly string[] ExpectedResolvedTickTexts = ["1", "6", "11"];

    [TestMethod]
    public async Task LowerConfidenceAlternativeWinsWhenItCompletesRegularMonotonicTickSequence()
    {
        OcrDetectedRegion[] ticks = XTicks();
        var recognizer = Recognizer(
            ("x1", OcrSourceImage.Original, "1", 0.90),
            ("x1", OcrSourceImage.Enhanced, "1", 0.96),
            ("x2", OcrSourceImage.Original, "6", 0.82),
            ("x2", OcrSourceImage.Enhanced, "60", 0.99),
            ("x3", OcrSourceImage.Original, "11", 0.90),
            ("x3", OcrSourceImage.Enhanced, "11", 0.96));
        var pipeline = Pipeline(recognizer);

        OcrResult result = await pipeline.RecognizeAsync(
            OcrTestFixtures.Request(ticks, EnhancedImage()),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded);
        CollectionAssert.AreEqual(
            ExpectedResolvedTickTexts,
            result.Regions.Select(static region => region.Text).ToArray());
        OcrRegion resolved = result.Regions.Single(region => region.RegionId == "x2");
        Assert.AreEqual(OcrSourceImage.Original, resolved.SourceImage);
        Assert.IsTrue(resolved.Alternatives.Any(alternative => alternative.Text == "60"));
        Assert.IsFalse(result.Warnings.Any(warning =>
            warning.Contains("needs_review", StringComparison.OrdinalIgnoreCase)));
    }

    [TestMethod]
    public async Task EquallyRegularConflictingTickSequencesEmitReviewWarning()
    {
        OcrDetectedRegion[] ticks = XTicks();
        var recognizer = Recognizer(
            ("x1", OcrSourceImage.Original, "1", 0.90),
            ("x1", OcrSourceImage.Enhanced, "2", 0.90),
            ("x2", OcrSourceImage.Original, "6", 0.90),
            ("x2", OcrSourceImage.Enhanced, "7", 0.90),
            ("x3", OcrSourceImage.Original, "11", 0.90),
            ("x3", OcrSourceImage.Enhanced, "12", 0.90));
        var pipeline = Pipeline(recognizer);

        OcrResult result = await pipeline.RecognizeAsync(
            OcrTestFixtures.Request(ticks, EnhancedImage()),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded);
        Assert.IsTrue(result.Warnings.Any(warning =>
            warning.Contains("tick", StringComparison.OrdinalIgnoreCase) &&
            warning.Contains("review", StringComparison.OrdinalIgnoreCase)));
        Assert.IsTrue(result.Regions.All(static region => region.Role == OcrTextRole.XTick));
    }

    [TestMethod]
    public async Task MoreThanMaximumCombinationsCannotSilentlyReplaceHighConfidenceTickValues()
    {
        OcrDetectedRegion[] ticks = Enumerable.Range(0, 8)
            .Select(index => OcrTestFixtures.Region($"x{index}", 36 + (index * 14), 89, 8, 6))
            .ToArray();
        var alternatives = ticks.ToDictionary(
            static tick => (tick.RegionId, OcrSourceImage.Original),
            static tick =>
            {
                var index = int.Parse(tick.RegionId.AsSpan(1), CultureInfo.InvariantCulture);
                var target = 2 + (10 * index);
                return (IReadOnlyList<OcrRecognitionAlternative>)[
                    new((target - 2).ToString(CultureInfo.InvariantCulture), 0.10, OcrSourceImage.Original),
                    new((target - 1).ToString(CultureInfo.InvariantCulture), 0.20, OcrSourceImage.Original),
                    new(target.ToString(CultureInfo.InvariantCulture), 0.99, OcrSourceImage.Original),
                ];
            });
        var recognizer = new StubTextRecognizer(alternatives);
        var pipeline = new OcrPipeline(
            new StubTextRegionDetector([]),
            recognizer,
            new InMemoryOcrResultCache(),
            new OcrPipelineOptions { BatchSize = 8, MaximumTickCombinationEvaluations = 4096 });
        string[] expected = Enumerable.Range(0, 8)
            .Select(index => (2 + (10 * index)).ToString(CultureInfo.InvariantCulture))
            .ToArray();

        OcrResult result = await pipeline.RecognizeAsync(
            OcrTestFixtures.Request(ticks),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        CollectionAssert.AreEqual(expected, result.Regions.Select(static region => region.Text).ToArray());
        Assert.IsTrue(result.Regions.All(static region => region.Role == OcrTextRole.XTick));
        CollectionAssert.Contains(
            result.Warnings.ToArray(),
            "ocr_tick_sequence_needs_review:XTick:combination_search_incomplete");
        Assert.IsFalse(result.Warnings.Any(static warning =>
            warning.Contains("resolved_by_monotonic_spacing", StringComparison.Ordinal)));
    }

    private static OcrDetectedRegion[] XTicks() =>
    [
        OcrTestFixtures.Region("x1", 46, 89, 8, 6),
        OcrTestFixtures.Region("x2", 76, 89, 8, 6),
        OcrTestFixtures.Region("x3", 106, 89, 10, 6),
    ];

    private static OcrImage EnhancedImage() => OcrTestFixtures.Image(
        OcrSourceImage.Enhanced,
        width: 320,
        height: 200,
        transform: new OcrFrameTransform(2, 2, 0, 0));

    private static OcrPipeline Pipeline(StubTextRecognizer recognizer) =>
        new(
            new StubTextRegionDetector([]),
            recognizer,
            new InMemoryOcrResultCache(),
            batchSize: 8);

    private static StubTextRecognizer Recognizer(
        params (string RegionId, OcrSourceImage Source, string Text, double Confidence)[] results)
    {
        var alternatives = results.ToDictionary(
            static result => (result.RegionId, result.Source),
            static result => (IReadOnlyList<OcrRecognitionAlternative>)[
                new OcrRecognitionAlternative(result.Text, result.Confidence, result.Source),
            ]);
        return new StubTextRecognizer(alternatives);
    }
}
