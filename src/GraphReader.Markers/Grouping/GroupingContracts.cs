// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections;
using GraphReader.Markers.Classification;
using GraphReader.Markers.Detection;

namespace GraphReader.Markers.Grouping;

public static class MarkerGroupingContract
{
    public const int Version = 1;
    public const string CoordinateSpace = MarkerContract.CoordinateSpace;
    public const string Stage = "marker_grouping";
}

public enum MarkerConnectionStyle
{
    Solid,
    Dashed,
    Unknown,
}

public enum MarkerSeriesRole
{
    Baseline,
    Intervention,
    Maintenance,
    Generalization,
    Unknown,
}

public enum MarkerTextEvidenceSource
{
    Legend,
    Participant,
    Annotation,
    UserConfirmed,
}

public enum MarkerGroupingCommandKind
{
    MergeSeries,
    SplitSeries,
    ReassignMarkers,
}

public sealed record MarkerGroupingOptions
{
    public double MinimumEmbeddingSimilarity { get; init; } = 0.80;

    public double MinimumConnectionConfidence { get; init; } = 0.50;

    public double MinimumGroupingScore { get; init; } = 0.68;

    public double MaximumSessionGap { get; init; } = 4;

    public double ShapeFillWeight { get; init; } = 0.35;

    public double EmbeddingWeight { get; init; } = 0.25;

    public double ConnectionWeight { get; init; } = 0.20;

    public double SessionOrderWeight { get; init; } = 0.10;

    public double PhaseContinuityWeight { get; init; } = 0.05;

    public double LegendWeight { get; init; } = 0.05;

    public string StageVersion { get; init; } = "0.1.0";
}

public sealed record MarkerConnectionOptions
{
    public double MarkerExclusionRadiusScale { get; init; } = 1.20;

    public double CorridorHalfWidthPixels { get; init; } = 1.50;

    public double MinimumInkFraction { get; init; } = 0.55;

    public double MaximumMaskFraction { get; init; } = 0.20;

    public double MaximumHorizontalGapPixels { get; init; } = 250;

    public int MinimumSamples { get; init; } = 8;
}

public sealed record MarkerGroupingEvidence
{
    public MarkerGroupingEvidence(
        ClassifiedMarker marker,
        int? observationIndex = null,
        double? printedSession = null,
        string? phaseRegionId = null)
    {
        Marker = marker ?? throw new ArgumentNullException(nameof(marker));
        ObservationIndex = observationIndex;
        PrintedSession = printedSession;
        PhaseRegionId = phaseRegionId;
    }

    public ClassifiedMarker Marker { get; }

    public int? ObservationIndex { get; }

    public double? PrintedSession { get; }

    public string? PhaseRegionId { get; }
}

public sealed record MarkerConnection(
    string FromMarkerId,
    string ToMarkerId,
    double Confidence,
    MarkerConnectionStyle Style);

public sealed record MarkerLegendEvidence(
    MarkerShape Shape,
    MarkerFill Fill,
    string Text,
    MarkerTextEvidenceSource Source,
    double Confidence,
    bool ExplicitlyConfirmed = false);

public sealed class MarkerSeries
{
    public MarkerSeries(
        string seriesId,
        string symbol,
        MarkerShape shape,
        MarkerFill fill,
        string displayName,
        MarkerSeriesRole semanticRole,
        IEnumerable<string> markerIds,
        double confidence,
        string? legendText = null,
        string? sharedBaselineSeriesId = null,
        IEnumerable<string>? applicableProbeSeriesIds = null)
    {
        SeriesId = seriesId ?? throw new ArgumentNullException(nameof(seriesId));
        Symbol = symbol ?? throw new ArgumentNullException(nameof(symbol));
        Shape = shape;
        Fill = fill;
        DisplayName = displayName ?? throw new ArgumentNullException(nameof(displayName));
        SemanticRole = semanticRole;
        MarkerIds = GroupingCollections.Freeze(markerIds ?? throw new ArgumentNullException(nameof(markerIds)));
        Confidence = confidence;
        LegendText = legendText;
        SharedBaselineSeriesId = sharedBaselineSeriesId;
        ApplicableProbeSeriesIds = GroupingCollections.Freeze(
            applicableProbeSeriesIds ?? Array.Empty<string>());
    }

    public string SeriesId { get; }

    public string Symbol { get; }

    public MarkerShape Shape { get; }

    public MarkerFill Fill { get; }

    public string DisplayName { get; }

    public MarkerSeriesRole SemanticRole { get; }

    public IReadOnlyList<string> MarkerIds { get; }

    public int PointCount => MarkerIds.Count;

    public double Confidence { get; }

    public string? LegendText { get; }

    public string? SharedBaselineSeriesId { get; }

    public IReadOnlyList<string> ApplicableProbeSeriesIds { get; }
}

public sealed class MarkerGroupingAuditEvent
{
    public MarkerGroupingAuditEvent(
        string eventId,
        DateTimeOffset timestampUtc,
        MarkerGroupingCommandKind kind,
        IEnumerable<string> affectedSeriesIds,
        IEnumerable<string> affectedMarkerIds,
        string reason)
    {
        EventId = eventId ?? throw new ArgumentNullException(nameof(eventId));
        TimestampUtc = timestampUtc;
        Kind = kind;
        AffectedSeriesIds = GroupingCollections.Freeze(
            affectedSeriesIds ?? throw new ArgumentNullException(nameof(affectedSeriesIds)));
        AffectedMarkerIds = GroupingCollections.Freeze(
            affectedMarkerIds ?? throw new ArgumentNullException(nameof(affectedMarkerIds)));
        Reason = reason ?? throw new ArgumentNullException(nameof(reason));
    }

    public string EventId { get; }

    public DateTimeOffset TimestampUtc { get; }

    public MarkerGroupingCommandKind Kind { get; }

    public IReadOnlyList<string> AffectedSeriesIds { get; }

    public IReadOnlyList<string> AffectedMarkerIds { get; }

    public string Reason { get; }
}

public sealed class MarkerGroupingState
{
    public MarkerGroupingState(
        IEnumerable<MarkerGroupingEvidence> markers,
        IEnumerable<MarkerConnection> connections,
        IEnumerable<MarkerSeries> series,
        IEnumerable<MarkerGroupingAuditEvent>? auditEvents = null)
    {
        Markers = GroupingCollections.Freeze(markers ?? throw new ArgumentNullException(nameof(markers)));
        Connections = GroupingCollections.Freeze(connections ?? throw new ArgumentNullException(nameof(connections)));
        Series = GroupingCollections.Freeze(series ?? throw new ArgumentNullException(nameof(series)));
        AuditEvents = GroupingCollections.Freeze(auditEvents ?? Array.Empty<MarkerGroupingAuditEvent>());
    }

    public IReadOnlyList<MarkerGroupingEvidence> Markers { get; }

    public IReadOnlyList<MarkerConnection> Connections { get; }

    public IReadOnlyList<MarkerSeries> Series { get; }

    public IReadOnlyList<MarkerGroupingAuditEvent> AuditEvents { get; }

    public int SeriesCount => Series.Count;

    public int UniqueMarkerCount => Markers.Count;
}

public sealed class MarkerConnectionRequest
{
    public MarkerConnectionRequest(
        MarkerImageFrame image,
        IEnumerable<MarkerGroupingEvidence> markers,
        MarkerConnectionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        Image = new MarkerImageFrame(
            image.Width,
            image.Height,
            image.ChannelCount,
            image.ChannelsFirstPixels.ToArray(),
            image.SourceImage,
            image.OriginalToFrame,
            CopyMask(image.OcrMask),
            CopyMask(image.ArtifactMask));
        Markers = GroupingCollections.Freeze(markers ?? throw new ArgumentNullException(nameof(markers)));
        Options = options ?? new MarkerConnectionOptions();
    }

    public MarkerImageFrame Image { get; }

    public IReadOnlyList<MarkerGroupingEvidence> Markers { get; }

    public MarkerConnectionOptions Options { get; }

    private static MarkerMask CopyMask(MarkerMask mask) =>
        new(mask.Width, mask.Height, mask.Values.ToArray());
}

public sealed class MarkerGroupingRequest
{
    public MarkerGroupingRequest(
        string projectId,
        string panelId,
        IEnumerable<MarkerGroupingEvidence> markers,
        IEnumerable<MarkerConnection> connections,
        IEnumerable<MarkerLegendEvidence>? legendEvidence = null,
        MarkerGroupingOptions? options = null,
        int contractVersion = MarkerGroupingContract.Version)
    {
        ProjectId = projectId ?? throw new ArgumentNullException(nameof(projectId));
        PanelId = panelId ?? throw new ArgumentNullException(nameof(panelId));
        Markers = GroupingCollections.Freeze(markers ?? throw new ArgumentNullException(nameof(markers)));
        Connections = GroupingCollections.Freeze(connections ?? throw new ArgumentNullException(nameof(connections)));
        LegendEvidence = GroupingCollections.Freeze(legendEvidence ?? Array.Empty<MarkerLegendEvidence>());
        Options = options ?? new MarkerGroupingOptions();
        ContractVersion = contractVersion;
    }

    public string ProjectId { get; }

    public string PanelId { get; }

    public IReadOnlyList<MarkerGroupingEvidence> Markers { get; }

    public IReadOnlyList<MarkerConnection> Connections { get; }

    public IReadOnlyList<MarkerLegendEvidence> LegendEvidence { get; }

    public MarkerGroupingOptions Options { get; }

    public int ContractVersion { get; }
}

public sealed record MarkerGroupingTiming(
    double ConnectionMilliseconds,
    double GroupingMilliseconds,
    double TotalMilliseconds);

public sealed record MarkerGroupingFailure(
    string Code,
    string Severity,
    string UserMessageKey,
    string TechnicalMessage,
    bool Recoverable,
    string SuggestedAction);

public sealed class MarkerGroupingResult
{
    public MarkerGroupingResult(
        MarkerGroupingState? state,
        MarkerGroupingTiming timing,
        double confidence,
        IEnumerable<string> warnings,
        MarkerGroupingFailure? failure)
    {
        State = state;
        Timing = timing ?? throw new ArgumentNullException(nameof(timing));
        Confidence = confidence;
        Warnings = GroupingCollections.Freeze(warnings ?? throw new ArgumentNullException(nameof(warnings)));
        Failure = failure;
    }

    public MarkerGroupingState? State { get; }

    public MarkerGroupingTiming Timing { get; }

    public double Confidence { get; }

    public IReadOnlyList<string> Warnings { get; }

    public MarkerGroupingFailure? Failure { get; }

    public bool Succeeded => Failure is null;
}

public sealed class MarkerGroupingEditCommand
{
    public MarkerGroupingEditCommand(
        MarkerGroupingCommandKind kind,
        IEnumerable<string> markerIds,
        string? sourceSeriesId = null,
        string? secondarySeriesId = null,
        string? targetSeriesId = null,
        string? newSeriesId = null,
        string reason = "manual_review",
        DateTimeOffset? timestampUtc = null)
    {
        Kind = kind;
        MarkerIds = GroupingCollections.Freeze(markerIds ?? throw new ArgumentNullException(nameof(markerIds)));
        SourceSeriesId = sourceSeriesId;
        SecondarySeriesId = secondarySeriesId;
        TargetSeriesId = targetSeriesId;
        NewSeriesId = newSeriesId;
        Reason = reason ?? throw new ArgumentNullException(nameof(reason));
        TimestampUtc = timestampUtc ?? DateTimeOffset.UtcNow;
    }

    public MarkerGroupingCommandKind Kind { get; }

    public IReadOnlyList<string> MarkerIds { get; }

    public string? SourceSeriesId { get; }

    public string? SecondarySeriesId { get; }

    public string? TargetSeriesId { get; }

    public string? NewSeriesId { get; }

    public string Reason { get; }

    public DateTimeOffset TimestampUtc { get; }
}

public sealed record MarkerGroupingEditResult(
    MarkerGroupingState State,
    MarkerGroupingAuditEvent? AuditEvent,
    MarkerGroupingFailure? Failure)
{
    public bool Succeeded => Failure is null;
}

public interface IMarkerConnectionGraphBuilder
{
    ValueTask<IReadOnlyList<MarkerConnection>> BuildAsync(
        MarkerConnectionRequest request,
        CancellationToken cancellationToken);
}

public interface IMarkerSeriesGrouper
{
    ValueTask<MarkerGroupingResult> GroupAsync(
        MarkerGroupingRequest request,
        CancellationToken cancellationToken);
}

public interface IMarkerGroupingEditor
{
    MarkerGroupingEditResult Apply(
        MarkerGroupingState state,
        MarkerGroupingEditCommand command,
        CancellationToken cancellationToken);
}

internal static class GroupingCollections
{
    public static IReadOnlyList<T> Freeze<T>(IEnumerable<T> values) => new FrozenList<T>(values);

    private sealed class FrozenList<T> : IReadOnlyList<T>, IEquatable<FrozenList<T>>
    {
        private readonly T[] _items;

        public FrozenList(IEnumerable<T> values) => _items = values.ToArray();

        public int Count => _items.Length;

        public T this[int index] => _items[index];

        public bool Equals(FrozenList<T>? other) =>
            other is not null && _items.SequenceEqual(other._items);

        public override bool Equals(object? obj) =>
            obj is IEnumerable<T> values && _items.SequenceEqual(values);

        public override int GetHashCode()
        {
            var hash = new HashCode();
            foreach (var item in _items)
            {
                hash.Add(item);
            }

            return hash.ToHashCode();
        }

        public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)_items).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => _items.GetEnumerator();
    }
}
