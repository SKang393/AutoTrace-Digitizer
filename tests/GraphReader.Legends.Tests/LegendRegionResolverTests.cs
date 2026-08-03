// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Markers.Classification;
using GraphReader.Ocr;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Globalization;

namespace GraphReader.Legends.Tests;

[TestClass]
public sealed class LegendRegionResolverTests
{
    [TestMethod]
    public void InsidePlotLegendMapsGlyphToTextAndSeries()
    {
        var request = LegendTestFixtures.Request(
            textRegions: [LegendTestFixtures.Text("text-a", 105, 51, "Treatment")],
            glyphs: [LegendTestFixtures.Glyph("glyph-a", 85, 52, MarkerShape.Circle, MarkerFill.Filled, [1f, 0f, 0f])]);

        var regions = new LegendRegionResolver().Resolve(request, CancellationToken.None);

        Assert.HasCount(1, regions);
        Assert.AreEqual(LegendRegionLocation.InsidePlot, regions[0].Location);
        Assert.HasCount(1, regions[0].Entries);
        Assert.AreEqual("Treatment", regions[0].Entries[0].Text);
        Assert.IsNull(regions[0].Entries[0].NormalizedSeriesId);
    }

    [TestMethod]
    public void OutsidePlotLegendIsRetainedAndClassified()
    {
        var request = LegendTestFixtures.Request(
            textRegions: [LegendTestFixtures.Text("text-a", 410, 52, "Intervention")],
            glyphs: [LegendTestFixtures.Glyph("glyph-a", 390, 53, MarkerShape.Circle, MarkerFill.Filled, [1f, 0f, 0f])]);

        var regions = new LegendRegionResolver().Resolve(request, CancellationToken.None);

        Assert.HasCount(1, regions);
        Assert.AreEqual(LegendRegionLocation.OutsidePlot, regions[0].Location);
        Assert.IsNull(regions[0].Entries[0].NormalizedSeriesId);
    }

    [TestMethod]
    public void TwoLegendEntriesPairByGeometryWithoutCrossing()
    {
        var request = LegendTestFixtures.Request(
            textRegions:
            [
                LegendTestFixtures.Text("text-filled", 105, 52, "Treatment"),
                LegendTestFixtures.Text("text-open", 105, 82, "Generalization"),
            ],
            glyphs:
            [
                LegendTestFixtures.Glyph("glyph-filled", 85, 53, MarkerShape.Circle, MarkerFill.Filled, [1f, 0f, 0f]),
                LegendTestFixtures.Glyph("glyph-open", 85, 83, MarkerShape.Circle, MarkerFill.Open, [0f, 1f, 0f]),
            ]);

        var entries = new LegendRegionResolver().Resolve(request, CancellationToken.None)
            .SelectMany(static region => region.Entries)
            .OrderBy(static entry => entry.GlyphId, StringComparer.Ordinal)
            .ToArray();

        Assert.HasCount(2, entries);
        Assert.AreEqual("Treatment", entries[0].Text);
        Assert.AreEqual("Generalization", entries[1].Text);
        Assert.AreEqual(LegendSemanticHint.Generalization, entries[1].Semantic.Hint);
    }

    [TestMethod]
    public void AmbiguousNearbyAnnotationTextIsNotConsumedAsLegendText()
    {
        var request = LegendTestFixtures.Request(
            textRegions:
            [
                LegendTestFixtures.Text("annotation", 103, 51, "Change here", OcrTextRole.Annotation),
                LegendTestFixtures.Text("legend", 105, 58, "Treatment"),
            ],
            glyphs: [LegendTestFixtures.Glyph("glyph", 85, 54, MarkerShape.Circle, MarkerFill.Filled, [1f, 0f, 0f])]);

        var entry = new LegendRegionResolver().Resolve(request, CancellationToken.None)
            .SelectMany(static region => region.Entries)
            .Single();

        Assert.AreEqual("legend", entry.TextRegionId);
        Assert.AreEqual("Treatment", entry.Text);
    }

    [TestMethod]
    public void MissingGlyphOrReliableLegendTextReturnsNoRegion()
    {
        var request = LegendTestFixtures.Request(
            textRegions: [LegendTestFixtures.Text("other", 105, 52, "Figure note", OcrTextRole.Other)],
            glyphs: []);

        var regions = new LegendRegionResolver().Resolve(request, CancellationToken.None);

        Assert.IsEmpty(regions);
    }

    [TestMethod]
    public void RejectedLegendTextIsExcludedAndEachTextIsUsedAtMostOnce()
    {
        var rejected = new LegendTextRegion(
            "rejected",
            new LegendRectangle(105, 51, 70, 14),
            "Rejected name",
            OcrTextRole.LegendText,
            0.99,
            OcrReviewStatus.Rejected);
        var accepted = LegendTestFixtures.Text("accepted", 105, 58, "Treatment");
        var request = LegendTestFixtures.Request(
            textRegions: [rejected, accepted],
            glyphs:
            [
                LegendTestFixtures.Glyph("glyph-a", 85, 53, MarkerShape.Circle, MarkerFill.Filled, [1f, 0f, 0f]),
                LegendTestFixtures.Glyph("glyph-b", 85, 59, MarkerShape.Circle, MarkerFill.Filled, [1f, 0f, 0f]),
            ]);

        var entries = new LegendRegionResolver().Resolve(request, CancellationToken.None)
            .SelectMany(static region => region.Entries)
            .ToArray();

        Assert.HasCount(1, entries);
        Assert.AreEqual("accepted", entries[0].TextRegionId);
    }

    [TestMethod]
    public void SemanticNormalizationHandlesPunctuationAndProbeWordingDeterministically()
    {
        var request = LegendTestFixtures.Request(
            textRegions: [LegendTestFixtures.Text("text", 105, 51, "GENERALIZATION probes!")],
            glyphs: [LegendTestFixtures.Glyph("glyph", 85, 52, MarkerShape.Circle, MarkerFill.Open, [0f, 1f, 0f])]);
        var resolver = new LegendRegionResolver();

        var first = resolver.Resolve(request, CancellationToken.None).Single().Entries.Single();
        var second = resolver.Resolve(request, CancellationToken.None).Single().Entries.Single();

        Assert.AreEqual(LegendSemanticHint.Generalization, first.Semantic.Hint);
        Assert.AreEqual("generalization", first.Semantic.NormalizedText);
        Assert.AreEqual(first.EntryId, second.EntryId);
        Assert.AreEqual(first.Confidence, second.Confidence);
    }

    [TestMethod]
    public void ResolveHonorsPreCanceledToken()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        Assert.ThrowsExactly<OperationCanceledException>(() =>
            new LegendRegionResolver().Resolve(LegendTestFixtures.Request(), cancellation.Token));
    }

    [TestMethod]
    public async Task ExternalHeldoutLegendMappingAccuracyMeetsPointNineGate()
    {
        IReadOnlyList<HeldoutCase> cases = LoadHeldoutCases();
        var correct = 0;

        foreach (var fixture in cases)
        {
            var result = await new LegendReasoningService().ResolveAsync(fixture.Request, CancellationToken.None);
            var entry = result.Regions
                .SelectMany(static region => region.Entries)
                .SingleOrDefault();
            if (entry?.NormalizedSeriesId == fixture.ExpectedSeriesId && entry.Text == fixture.ExpectedText)
            {
                correct++;
            }
        }

        var accuracy = correct / (double)cases.Count;
        Assert.HasCount(24, cases);
        Assert.IsGreaterThanOrEqualTo(4, cases.Select(static item => (item.Shape, item.Fill)).Distinct().Count());
        Assert.IsTrue(cases.Any(static item => item.GlyphX < LegendTestFixtures.PlotBounds.Right));
        Assert.IsTrue(cases.Any(static item => item.GlyphX > LegendTestFixtures.PlotBounds.Right));
        Assert.IsGreaterThanOrEqualTo(0.90, accuracy, $"Held-out mapping accuracy was {accuracy:F3} ({correct}/{cases.Count}).");
    }

    private static List<HeldoutCase> LoadHeldoutCases()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "heldout-legend-mappings.csv");
        string[] lines = File.ReadAllLines(path)
            .Where(static line => !string.IsNullOrWhiteSpace(line) && !line.StartsWith('#'))
            .ToArray();
        Assert.IsGreaterThan(1, lines.Length);
        var cases = new List<HeldoutCase>(lines.Length - 1);
        foreach (string line in lines.Skip(1))
        {
            string[] fields = line.Split(',');
            Assert.HasCount(17, fields);
            string caseId = fields[0];
            double glyphX = ParseDouble(fields[1]);
            double glyphY = ParseDouble(fields[2]);
            double textX = ParseDouble(fields[3]);
            double textY = ParseDouble(fields[4]);
            MarkerShape shape = Enum.Parse<MarkerShape>(fields[5]);
            MarkerFill fill = Enum.Parse<MarkerFill>(fields[6]);
            float[] embedding = fields[7..15].Select(ParseFloat).ToArray();
            string text = fields[15];
            string expected = fields[16];
            var request = LegendTestFixtures.Request(
                textRegions: [LegendTestFixtures.Text($"text-{caseId}", textX, textY, text)],
                glyphs: [LegendTestFixtures.Glyph($"glyph-{caseId}", glyphX, glyphY, shape, fill, embedding)],
                series: HeldoutSeries(),
                markers: HeldoutMarkers());
            cases.Add(new HeldoutCase(request, expected, text, shape, fill, glyphX));
        }

        return cases;
    }

    private static LegendSeriesCandidate[] HeldoutSeries() =>
        Enumerable.Range(1, 8)
            .Select(index =>
            {
                (MarkerShape shape, MarkerFill fill) = index switch
                {
                    1 or 2 => (MarkerShape.Circle, MarkerFill.Filled),
                    3 or 4 => (MarkerShape.Circle, MarkerFill.Open),
                    5 or 6 => (MarkerShape.Square, MarkerFill.Filled),
                    _ => (MarkerShape.Diamond, MarkerFill.Open),
                };
                var embedding = new float[8];
                embedding[index - 1] = 1;
                return new LegendSeriesCandidate(
                    $"00000000-0000-0000-0000-{100 + index:D12}",
                    shape,
                    fill,
                    MarkerSymbolMap.GetSymbol(shape, fill),
                    MarkerSymbolMap.GetAccessibleName(shape, fill),
                    [$"00000000-0000-0000-0000-{200 + index:D12}"],
                    embedding);
            })
            .ToArray();

    private static LegendPlotMarker[] HeldoutMarkers() =>
        HeldoutSeries()
            .Select((series, index) => new LegendPlotMarker(
                series.MarkerIds[0],
                series.SeriesId,
                new LegendPoint(220 + (index * 5), 55 + (index * 25)),
                series.Shape,
                series.Fill))
            .ToArray();

    private static double ParseDouble(string value) =>
        double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);

    private static float ParseFloat(string value) =>
        float.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);

    private sealed record HeldoutCase(
        LegendReasoningRequest Request,
        string ExpectedSeriesId,
        string ExpectedText,
        MarkerShape Shape,
        MarkerFill Fill,
        double GlyphX);
}
