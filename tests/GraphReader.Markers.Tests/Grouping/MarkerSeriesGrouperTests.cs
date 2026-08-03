// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Markers.Classification;
using GraphReader.Markers.Grouping;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Grouping;

[TestClass]
public sealed class MarkerSeriesGrouperTests
{
    private static readonly string[] ExpectedBaselineIds = ["baseline-1", "baseline-2"];

    [TestMethod]
    public async Task FilledAndOpenCirclesNeverMergeSolelyBecauseTheyTouchTheSameLine()
    {
        MarkerGroupingEvidence[] markers =
        [
            MarkerGroupingTestSupport.Evidence("filled-1", 1, fill: MarkerFill.Filled),
            MarkerGroupingTestSupport.Evidence("open-1", 2, fill: MarkerFill.Open),
            MarkerGroupingTestSupport.Evidence("filled-2", 3, fill: MarkerFill.Filled),
            MarkerGroupingTestSupport.Evidence("open-2", 4, fill: MarkerFill.Open),
        ];
        MarkerConnection[] connections =
        [
            new("filled-1", "open-1", 0.99, MarkerConnectionStyle.Solid),
            new("filled-1", "filled-2", 0.9, MarkerConnectionStyle.Solid),
            new("open-1", "open-2", 0.9, MarkerConnectionStyle.Solid),
        ];

        MarkerGroupingResult result = await new MarkerSeriesGrouper().GroupAsync(
            Request(markers, connections),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.IsNotNull(result.State);
        Assert.AreEqual(2, result.State.SeriesCount);
        Assert.HasCount(2, result.State.Series.Single(series => series.Fill == MarkerFill.Filled).MarkerIds);
        Assert.HasCount(2, result.State.Series.Single(series => series.Fill == MarkerFill.Open).MarkerIds);
        AssertUniqueCompleteAssignment(result.State);
    }

    [TestMethod]
    public async Task UnknownFillCannotTransitivelyBridgeOpenAndFilledSeries()
    {
        MarkerGroupingEvidence[] markers =
        [
            MarkerGroupingTestSupport.Evidence("open", 1, fill: MarkerFill.Open),
            MarkerGroupingTestSupport.Evidence("unknown", 2, fill: MarkerFill.Unknown),
            MarkerGroupingTestSupport.Evidence("filled", 3, fill: MarkerFill.Filled),
        ];
        MarkerConnection[] connections =
        [
            new("open", "unknown", 1, MarkerConnectionStyle.Solid),
            new("unknown", "filled", 1, MarkerConnectionStyle.Solid),
        ];

        MarkerGroupingResult result = await new MarkerSeriesGrouper().GroupAsync(
            Request(markers, connections),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        MarkerGroupingState state = result.State ?? throw new AssertFailedException("Missing grouping state.");
        Assert.AreEqual(2, state.SeriesCount);
        Assert.IsFalse(state.Series.Any(series =>
            series.MarkerIds.Contains("open", StringComparer.Ordinal) &&
            series.MarkerIds.Contains("filled", StringComparer.Ordinal)));
        AssertUniqueCompleteAssignment(state);
    }

    [TestMethod]
    public async Task SharedBaselineAttachesToTwoInterventionsWithoutMarkerDuplication()
    {
        MarkerGroupingEvidence[] markers =
        [
            MarkerGroupingTestSupport.Evidence("baseline-1", 1, phase: "baseline"),
            MarkerGroupingTestSupport.Evidence("baseline-2", 2, phase: "baseline"),
            MarkerGroupingTestSupport.Evidence(
                "treatment-a-1", 3, MarkerShape.Square, MarkerFill.Filled, phase: "intervention"),
            MarkerGroupingTestSupport.Evidence(
                "treatment-b-1", 4, MarkerShape.TriangleUp, MarkerFill.Open, phase: "intervention"),
            MarkerGroupingTestSupport.Evidence(
                "treatment-a-2", 5, MarkerShape.Square, MarkerFill.Filled, phase: "intervention"),
            MarkerGroupingTestSupport.Evidence(
                "treatment-b-2", 6, MarkerShape.TriangleUp, MarkerFill.Open, phase: "intervention"),
        ];
        MarkerConnection[] connections =
        [
            new("baseline-1", "baseline-2", 0.95, MarkerConnectionStyle.Solid),
            new("baseline-2", "treatment-a-1", 0.9, MarkerConnectionStyle.Solid),
            new("baseline-2", "treatment-b-1", 0.9, MarkerConnectionStyle.Solid),
            new("treatment-a-1", "treatment-a-2", 0.95, MarkerConnectionStyle.Solid),
            new("treatment-b-1", "treatment-b-2", 0.95, MarkerConnectionStyle.Solid),
        ];

        MarkerGroupingResult result = await new MarkerSeriesGrouper().GroupAsync(
            Request(markers, connections),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        MarkerGroupingState state = result.State ?? throw new AssertFailedException("Missing grouping state.");
        Assert.AreEqual(3, state.SeriesCount);
        MarkerSeries baseline = state.Series.Single(series => series.SemanticRole == MarkerSeriesRole.Baseline);
        MarkerSeries[] interventions = state.Series
            .Where(series => series.SemanticRole == MarkerSeriesRole.Intervention)
            .ToArray();
        Assert.HasCount(2, interventions);
        Assert.IsTrue(interventions.All(series => series.SharedBaselineSeriesId == baseline.SeriesId));
        CollectionAssert.AreEquivalent(
            ExpectedBaselineIds,
            baseline.MarkerIds.ToArray());
        Assert.AreEqual(markers.Length, state.UniqueMarkerCount);
        AssertUniqueCompleteAssignment(state);
    }

    [TestMethod]
    public async Task IsolatedMinorityProbeRemainsDistinctFromPrimarySeries()
    {
        MarkerGroupingEvidence[] markers =
        [
            MarkerGroupingTestSupport.Evidence("primary-1", 1, phase: "intervention"),
            MarkerGroupingTestSupport.Evidence("primary-2", 2, phase: "intervention"),
            MarkerGroupingTestSupport.Evidence("primary-3", 3, phase: "intervention"),
            MarkerGroupingTestSupport.Evidence(
                "probe",
                4,
                MarkerShape.Diamond,
                MarkerFill.Open,
                phase: "generalization"),
        ];
        MarkerLegendEvidence[] legend =
        [
            new(
                MarkerShape.Diamond,
                MarkerFill.Open,
                "Generalization",
                MarkerTextEvidenceSource.UserConfirmed,
                1,
                true),
        ];

        MarkerGroupingResult result = await new MarkerSeriesGrouper().GroupAsync(
            Request(markers, Array.Empty<MarkerConnection>(), legend),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        MarkerGroupingState state = result.State ?? throw new AssertFailedException("Missing grouping state.");
        Assert.AreEqual(2, state.SeriesCount);
        MarkerSeries probe = state.Series.Single(series => series.MarkerIds.Contains("probe", StringComparer.Ordinal));
        Assert.HasCount(1, probe.MarkerIds);
        Assert.AreEqual(MarkerShape.Diamond, probe.Shape);
        Assert.AreEqual(MarkerFill.Open, probe.Fill);
        Assert.AreEqual("Generalization", probe.DisplayName);
        AssertUniqueCompleteAssignment(state);
    }

    [TestMethod]
    public async Task ParticipantAndAnnotationTextAreRejectedAsSeriesNamesButLegendTextIsAllowed()
    {
        MarkerGroupingEvidence[] markers =
        [
            MarkerGroupingTestSupport.Evidence("participant-shape", 1),
            MarkerGroupingTestSupport.Evidence(
                "annotation-shape", 2, MarkerShape.Diamond, MarkerFill.Open),
            MarkerGroupingTestSupport.Evidence(
                "legend-shape", 3, MarkerShape.Square, MarkerFill.Filled),
        ];
        MarkerLegendEvidence[] textEvidence =
        [
            new(
                MarkerShape.Circle,
                MarkerFill.Filled,
                "Chandler",
                MarkerTextEvidenceSource.Participant,
                0.99),
            new(
                MarkerShape.Diamond,
                MarkerFill.Open,
                "Generalization",
                MarkerTextEvidenceSource.Annotation,
                0.99),
            new(
                MarkerShape.Square,
                MarkerFill.Filled,
                "Treatment",
                MarkerTextEvidenceSource.Legend,
                0.95),
        ];

        MarkerGroupingResult result = await new MarkerSeriesGrouper().GroupAsync(
            Request(markers, Array.Empty<MarkerConnection>(), textEvidence),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        MarkerGroupingState state = result.State ?? throw new AssertFailedException("Missing grouping state.");
        Assert.IsFalse(state.Series.Any(series => series.DisplayName == "Chandler"));
        Assert.IsFalse(state.Series.Any(series => series.DisplayName == "Generalization"));
        MarkerSeries named = state.Series.Single(series => series.DisplayName == "Treatment");
        Assert.AreEqual(MarkerShape.Square, named.Shape);
        Assert.AreEqual("Treatment", named.LegendText);
    }

    [TestMethod]
    public async Task ExplicitUserConfirmationAllowsAnnotationTextAsTheSeriesName()
    {
        MarkerGroupingEvidence marker = MarkerGroupingTestSupport.Evidence(
            "confirmed-probe",
            1,
            MarkerShape.Diamond,
            MarkerFill.Open);
        MarkerLegendEvidence confirmed = new(
            MarkerShape.Diamond,
            MarkerFill.Open,
            "Generalization",
            MarkerTextEvidenceSource.Annotation,
            1,
            ExplicitlyConfirmed: true);

        MarkerGroupingResult result = await new MarkerSeriesGrouper().GroupAsync(
            Request([marker], Array.Empty<MarkerConnection>(), [confirmed]),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.AreEqual("Generalization", result.State?.Series.Single().DisplayName);
        Assert.AreEqual("Generalization", result.State?.Series.Single().LegendText);
    }

    [TestMethod]
    public async Task GroupingIsDeterministicAcrossReorderedEvidenceAndConnections()
    {
        MarkerGroupingEvidence[] markers =
        [
            MarkerGroupingTestSupport.Evidence("a1", 1),
            MarkerGroupingTestSupport.Evidence("b1", 2, MarkerShape.Square, MarkerFill.Open),
            MarkerGroupingTestSupport.Evidence("a2", 3),
            MarkerGroupingTestSupport.Evidence("b2", 4, MarkerShape.Square, MarkerFill.Open),
        ];
        MarkerConnection[] connections =
        [
            new("a1", "a2", 0.9, MarkerConnectionStyle.Solid),
            new("b1", "b2", 0.9, MarkerConnectionStyle.Dashed),
        ];
        var grouper = new MarkerSeriesGrouper();

        MarkerGroupingResult first = await grouper.GroupAsync(
            Request(markers, connections),
            CancellationToken.None);
        MarkerGroupingResult reordered = await grouper.GroupAsync(
            Request(markers.Reverse(), connections.Reverse()),
            CancellationToken.None);

        Assert.IsTrue(first.Succeeded, first.Failure?.TechnicalMessage);
        Assert.IsTrue(reordered.Succeeded, reordered.Failure?.TechnicalMessage);
        CollectionAssert.AreEqual(
            first.State!.Series.Select(SeriesMaterial).ToArray(),
            reordered.State!.Series.Select(SeriesMaterial).ToArray());
        Assert.AreEqual(first.Confidence, reordered.Confidence, 1e-12);
    }

    [TestMethod]
    public async Task DuplicateMarkersAndDanglingConnectionsReturnStructuredFailures()
    {
        MarkerGroupingEvidence marker = MarkerGroupingTestSupport.Evidence("duplicate", 1);
        var grouper = new MarkerSeriesGrouper();

        MarkerGroupingResult duplicates = await grouper.GroupAsync(
            Request([marker, marker], Array.Empty<MarkerConnection>()),
            CancellationToken.None);
        MarkerGroupingResult dangling = await grouper.GroupAsync(
            Request(
                [marker],
                [new MarkerConnection("duplicate", "missing", 0.9, MarkerConnectionStyle.Solid)]),
            CancellationToken.None);

        Assert.IsFalse(duplicates.Succeeded);
        Assert.IsNull(duplicates.State);
        Assert.AreEqual("MARKER_GROUPING_INVALID_REQUEST", duplicates.Failure?.Code);
        Assert.IsFalse(dangling.Succeeded);
        Assert.IsNull(dangling.State);
        Assert.AreEqual("MARKER_GROUPING_INVALID_CONNECTION", dangling.Failure?.Code);
    }

    [TestMethod]
    public async Task CancellationPropagatesWithoutReturningPartialGrouping()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsExactlyAsync<OperationCanceledException>(
            async () => await new MarkerSeriesGrouper().GroupAsync(
                Request(
                    [MarkerGroupingTestSupport.Evidence("m1", 1)],
                    Array.Empty<MarkerConnection>()),
                cancellation.Token));
    }

    private static MarkerGroupingRequest Request(
        IEnumerable<MarkerGroupingEvidence> markers,
        IEnumerable<MarkerConnection> connections,
        IEnumerable<MarkerLegendEvidence>? legends = null) =>
        new(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            markers,
            connections,
            legends,
            new MarkerGroupingOptions());

    private static string SeriesMaterial(MarkerSeries series) => string.Join(
        '|',
        series.SeriesId,
        series.Shape,
        series.Fill,
        series.SemanticRole,
        series.DisplayName,
        series.SharedBaselineSeriesId,
        string.Join(',', series.MarkerIds),
        string.Join(',', series.ApplicableProbeSeriesIds));

    private static void AssertUniqueCompleteAssignment(MarkerGroupingState state)
    {
        string[] assigned = state.Series.SelectMany(series => series.MarkerIds).ToArray();
        Assert.AreEqual(state.UniqueMarkerCount, assigned.Length);
        Assert.AreEqual(state.UniqueMarkerCount, assigned.Distinct(StringComparer.Ordinal).Count());
        CollectionAssert.AreEquivalent(
            state.Markers.Select(marker => marker.Marker.Marker.MarkerId).ToArray(),
            assigned);
    }
}
