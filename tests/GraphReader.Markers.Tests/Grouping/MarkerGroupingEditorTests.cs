// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Markers.Classification;
using GraphReader.Markers.Grouping;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Grouping;

[TestClass]
public sealed class MarkerGroupingEditorTests
{
    private static readonly DateTimeOffset FixedTimestamp =
        new(2026, 8, 3, 6, 30, 0, TimeSpan.Zero);
    private static readonly string[] ExpectedMergedIds = ["m1", "m2", "m3", "m4"];
    private static readonly string[] ExpectedRetainedIds = ["m1", "m3"];
    private static readonly string[] ExpectedSeparatedIds = ["m2", "m4"];
    private static readonly string[] ExpectedLeftAfterReassign = ["m1"];
    private static readonly string[] ExpectedRightAfterReassign = ["m2", "m3", "m4"];

    [TestMethod]
    public void MergeCreatesOneSeriesWithoutDuplicatingMarkerObjectsOrIds()
    {
        MarkerGroupingState original = EditableState();
        var command = new MarkerGroupingEditCommand(
            MarkerGroupingCommandKind.MergeSeries,
            Array.Empty<string>(),
            sourceSeriesId: "left",
            secondarySeriesId: "right",
            targetSeriesId: "merged",
            reason: "confirmed_same_series",
            timestampUtc: FixedTimestamp);

        MarkerGroupingEditResult result = new MarkerGroupingEditor().Apply(
            original,
            command,
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.State.Series);
        CollectionAssert.AreEqual(
            ExpectedMergedIds,
            result.State.Series.Single().MarkerIds.ToArray());
        Assert.AreEqual(4, result.State.UniqueMarkerCount);
        Assert.AreEqual(4, result.State.Series.SelectMany(series => series.MarkerIds).Distinct().Count());
        Assert.AreSame(original.Markers[0].Marker, result.State.Markers[0].Marker);
        Assert.HasCount(1, result.State.AuditEvents);
        Assert.AreEqual(result.AuditEvent, result.State.AuditEvents.Single());
        Assert.AreEqual(MarkerGroupingCommandKind.MergeSeries, result.AuditEvent?.Kind);
    }

    [TestMethod]
    public void SplitPreservesCountsAndAssignsEveryMarkerExactlyOnce()
    {
        MarkerGroupingState original = SingleSeriesState();
        var command = new MarkerGroupingEditCommand(
            MarkerGroupingCommandKind.SplitSeries,
            ["m2", "m4"],
            sourceSeriesId: "all",
            newSeriesId: "minority",
            reason: "visible_identity_difference",
            timestampUtc: FixedTimestamp);

        MarkerGroupingEditResult result = new MarkerGroupingEditor().Apply(
            original,
            command,
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.AreEqual(2, result.State.SeriesCount);
        CollectionAssert.AreEqual(
            ExpectedRetainedIds,
            result.State.Series.Single(series => series.SeriesId == "all").MarkerIds.ToArray());
        CollectionAssert.AreEqual(
            ExpectedSeparatedIds,
            result.State.Series.Single(series => series.SeriesId == "minority").MarkerIds.ToArray());
        AssertUniqueCompleteAssignment(result.State);
    }

    [TestMethod]
    public void ReassignMovesMarkersAndUpdatesCountsDeterministically()
    {
        MarkerGroupingState original = EditableState();
        var command = new MarkerGroupingEditCommand(
            MarkerGroupingCommandKind.ReassignMarkers,
            ["m2"],
            sourceSeriesId: "left",
            targetSeriesId: "right",
            reason: "manual_reassignment",
            timestampUtc: FixedTimestamp);
        var editor = new MarkerGroupingEditor();

        MarkerGroupingEditResult first = editor.Apply(original, command, CancellationToken.None);
        MarkerGroupingEditResult repeated = editor.Apply(first.State, command, CancellationToken.None);

        Assert.IsTrue(first.Succeeded, first.Failure?.TechnicalMessage);
        CollectionAssert.AreEqual(
            ExpectedLeftAfterReassign,
            first.State.Series.Single(series => series.SeriesId == "left").MarkerIds.ToArray());
        CollectionAssert.AreEqual(
            ExpectedRightAfterReassign,
            first.State.Series.Single(series => series.SeriesId == "right").MarkerIds.ToArray());
        AssertUniqueCompleteAssignment(first.State);
        Assert.IsFalse(repeated.Succeeded, "Repeating the same edit must be a stable no-op failure.");
        Assert.AreEqual("MARKER_GROUPING_INVALID_COMMAND", repeated.Failure?.Code);
        Assert.AreSame(first.State, repeated.State);
        Assert.HasCount(1, repeated.State.AuditEvents);
    }

    [TestMethod]
    public void MergeAndReassignRejectIncompatibleMarkerIdentities()
    {
        MarkerGroupingEvidence circle = MarkerGroupingTestSupport.Evidence("circle", 1);
        MarkerGroupingEvidence square = MarkerGroupingTestSupport.Evidence(
            "square", 2, MarkerShape.Square, MarkerFill.Filled);
        var state = new MarkerGroupingState(
            [circle, square],
            Array.Empty<MarkerConnection>(),
            [
                MarkerGroupingTestSupport.Series(
                    "circles", MarkerShape.Circle, MarkerFill.Filled, MarkerSeriesRole.Intervention, ["circle"]),
                MarkerGroupingTestSupport.Series(
                    "squares", MarkerShape.Square, MarkerFill.Filled, MarkerSeriesRole.Intervention, ["square"]),
            ]);
        var editor = new MarkerGroupingEditor();

        MarkerGroupingEditResult merge = editor.Apply(
            state,
            new MarkerGroupingEditCommand(
                MarkerGroupingCommandKind.MergeSeries,
                Array.Empty<string>(),
                sourceSeriesId: "circles",
                secondarySeriesId: "squares",
                timestampUtc: FixedTimestamp),
            CancellationToken.None);
        MarkerGroupingEditResult reassign = editor.Apply(
            state,
            new MarkerGroupingEditCommand(
                MarkerGroupingCommandKind.ReassignMarkers,
                ["square"],
                sourceSeriesId: "squares",
                targetSeriesId: "circles",
                timestampUtc: FixedTimestamp),
            CancellationToken.None);

        Assert.IsFalse(merge.Succeeded);
        Assert.AreEqual("MARKER_GROUPING_INCOMPATIBLE_IDENTITY", merge.Failure?.Code);
        Assert.AreSame(state, merge.State);
        Assert.IsFalse(reassign.Succeeded);
        Assert.AreEqual("MARKER_GROUPING_INCOMPATIBLE_IDENTITY", reassign.Failure?.Code);
        Assert.AreSame(state, reassign.State);
    }

    [TestMethod]
    public void MergeResolvesUnknownFillSymbolAndAccessibleNameConsistently()
    {
        MarkerGroupingEvidence unknown = MarkerGroupingTestSupport.Evidence(
            "unknown", 1, fill: MarkerFill.Unknown);
        MarkerGroupingEvidence filled = MarkerGroupingTestSupport.Evidence(
            "filled", 2, fill: MarkerFill.Filled);
        var state = new MarkerGroupingState(
            [unknown, filled],
            Array.Empty<MarkerConnection>(),
            [
                MarkerGroupingTestSupport.Series(
                    "unknown-series",
                    MarkerShape.Circle,
                    MarkerFill.Unknown,
                    MarkerSeriesRole.Intervention,
                    ["unknown"]),
                MarkerGroupingTestSupport.Series(
                    "filled-series",
                    MarkerShape.Circle,
                    MarkerFill.Filled,
                    MarkerSeriesRole.Intervention,
                    ["filled"]),
            ]);

        MarkerGroupingEditResult result = new MarkerGroupingEditor().Apply(
            state,
            new MarkerGroupingEditCommand(
                MarkerGroupingCommandKind.MergeSeries,
                Array.Empty<string>(),
                sourceSeriesId: "unknown-series",
                secondarySeriesId: "filled-series",
                targetSeriesId: "merged",
                timestampUtc: FixedTimestamp),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        MarkerSeries merged = result.State.Series.Single();
        MarkerSymbolDescriptor expected = MarkerSymbolMap.Describe(MarkerShape.Circle, MarkerFill.Filled);
        Assert.AreEqual(MarkerFill.Filled, merged.Fill);
        Assert.AreEqual(expected.Symbol, merged.Symbol);
        Assert.AreEqual(expected.AccessibleName, merged.DisplayName);
    }

    [TestMethod]
    public void InvalidSplitReturnsOriginalImmutableStateWithoutAudit()
    {
        MarkerGroupingState original = SingleSeriesState();
        var command = new MarkerGroupingEditCommand(
            MarkerGroupingCommandKind.SplitSeries,
            ["missing"],
            sourceSeriesId: "all",
            newSeriesId: "new",
            timestampUtc: FixedTimestamp);

        MarkerGroupingEditResult result = new MarkerGroupingEditor().Apply(
            original,
            command,
            CancellationToken.None);

        Assert.IsFalse(result.Succeeded);
        Assert.AreEqual("MARKER_GROUPING_INVALID_COMMAND", result.Failure?.Code);
        Assert.AreSame(original, result.State);
        Assert.IsNull(result.AuditEvent);
        Assert.IsEmpty(original.AuditEvents);
        AssertUniqueCompleteAssignment(original);
    }

    [TestMethod]
    public void EditRejectsStateWhenAnyKnownMarkerIsUnassigned()
    {
        MarkerGroupingEvidence[] markers =
        [
            MarkerGroupingTestSupport.Evidence("m1", 1),
            MarkerGroupingTestSupport.Evidence("m2", 2),
        ];
        var invalid = new MarkerGroupingState(
            markers,
            Array.Empty<MarkerConnection>(),
            [
                MarkerGroupingTestSupport.Series(
                    "partial",
                    MarkerShape.Circle,
                    MarkerFill.Filled,
                    MarkerSeriesRole.Intervention,
                    ["m1"]),
            ]);
        var command = new MarkerGroupingEditCommand(
            MarkerGroupingCommandKind.SplitSeries,
            ["m1"],
            sourceSeriesId: "partial",
            newSeriesId: "new",
            timestampUtc: FixedTimestamp);

        MarkerGroupingEditResult result = new MarkerGroupingEditor().Apply(
            invalid,
            command,
            CancellationToken.None);

        Assert.IsFalse(result.Succeeded);
        Assert.AreEqual("MARKER_GROUPING_INVALID_STATE", result.Failure?.Code);
        Assert.AreSame(invalid, result.State);
    }

    [TestMethod]
    public void EditAuditIdsAreDeterministicForIdenticalStateAndCommand()
    {
        MarkerGroupingState state = EditableState();
        var command = new MarkerGroupingEditCommand(
            MarkerGroupingCommandKind.ReassignMarkers,
            ["m2"],
            sourceSeriesId: "left",
            targetSeriesId: "right",
            reason: "audit_replay",
            timestampUtc: FixedTimestamp);
        var editor = new MarkerGroupingEditor();

        MarkerGroupingEditResult first = editor.Apply(state, command, CancellationToken.None);
        MarkerGroupingEditResult replay = editor.Apply(state, command, CancellationToken.None);

        Assert.IsTrue(first.Succeeded);
        Assert.IsTrue(replay.Succeeded);
        Assert.AreEqual(first.AuditEvent?.EventId, replay.AuditEvent?.EventId);
        CollectionAssert.AreEqual(
            first.State.Series.Select(SeriesMaterial).ToArray(),
            replay.State.Series.Select(SeriesMaterial).ToArray());
    }

    [TestMethod]
    public void CancellationLeavesGroupingStateUnchanged()
    {
        MarkerGroupingState state = EditableState();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        Assert.ThrowsExactly<OperationCanceledException>(
            () => new MarkerGroupingEditor().Apply(
                state,
                new MarkerGroupingEditCommand(
                    MarkerGroupingCommandKind.MergeSeries,
                    Array.Empty<string>(),
                    sourceSeriesId: "left",
                    secondarySeriesId: "right"),
                cancellation.Token));
        Assert.IsEmpty(state.AuditEvents);
        AssertUniqueCompleteAssignment(state);
    }

    private static MarkerGroupingState EditableState()
    {
        MarkerGroupingEvidence[] markers =
        [
            MarkerGroupingTestSupport.Evidence("m1", 1),
            MarkerGroupingTestSupport.Evidence("m2", 2),
            MarkerGroupingTestSupport.Evidence("m3", 3),
            MarkerGroupingTestSupport.Evidence("m4", 4),
        ];
        return new MarkerGroupingState(
            markers,
            Array.Empty<MarkerConnection>(),
            [
                MarkerGroupingTestSupport.Series(
                    "left",
                    MarkerShape.Circle,
                    MarkerFill.Filled,
                    MarkerSeriesRole.Intervention,
                    ["m1", "m2"]),
                MarkerGroupingTestSupport.Series(
                    "right",
                    MarkerShape.Circle,
                    MarkerFill.Filled,
                    MarkerSeriesRole.Intervention,
                    ["m3", "m4"]),
            ]);
    }

    private static MarkerGroupingState SingleSeriesState()
    {
        MarkerGroupingEvidence[] markers =
        [
            MarkerGroupingTestSupport.Evidence("m1", 1),
            MarkerGroupingTestSupport.Evidence("m2", 2),
            MarkerGroupingTestSupport.Evidence("m3", 3),
            MarkerGroupingTestSupport.Evidence("m4", 4),
        ];
        return new MarkerGroupingState(
            markers,
            Array.Empty<MarkerConnection>(),
            [
                MarkerGroupingTestSupport.Series(
                    "all",
                    MarkerShape.Circle,
                    MarkerFill.Filled,
                    MarkerSeriesRole.Intervention,
                    markers.Select(marker => marker.Marker.Marker.MarkerId)),
            ]);
    }

    private static void AssertUniqueCompleteAssignment(MarkerGroupingState state)
    {
        string[] assigned = state.Series.SelectMany(series => series.MarkerIds).ToArray();
        Assert.AreEqual(state.UniqueMarkerCount, assigned.Length);
        Assert.AreEqual(state.UniqueMarkerCount, assigned.Distinct(StringComparer.Ordinal).Count());
        CollectionAssert.AreEquivalent(
            state.Markers.Select(marker => marker.Marker.Marker.MarkerId).ToArray(),
            assigned);
    }

    private static string SeriesMaterial(MarkerSeries series) =>
        $"{series.SeriesId}:{string.Join(',', series.MarkerIds)}";
}
