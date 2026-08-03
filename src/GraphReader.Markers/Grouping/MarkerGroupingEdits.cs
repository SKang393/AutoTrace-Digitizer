// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using GraphReader.Markers.Classification;

namespace GraphReader.Markers.Grouping;

public sealed class MarkerGroupingEditor : IMarkerGroupingEditor
{
    public MarkerGroupingEditResult Apply(
        MarkerGroupingState state,
        MarkerGroupingEditCommand command,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(command);
        cancellationToken.ThrowIfCancellationRequested();

        MarkerGroupingFailure? validationFailure = ValidateState(state);
        if (validationFailure is not null)
        {
            return new MarkerGroupingEditResult(state, null, validationFailure);
        }

        if (string.IsNullOrWhiteSpace(command.Reason))
        {
            return Failure(state, "MARKER_GROUPING_INVALID_COMMAND", "An edit reason is required.");
        }

        return command.Kind switch
        {
            MarkerGroupingCommandKind.MergeSeries => Merge(state, command, cancellationToken),
            MarkerGroupingCommandKind.SplitSeries => Split(state, command, cancellationToken),
            MarkerGroupingCommandKind.ReassignMarkers => Reassign(state, command, cancellationToken),
            _ => Failure(state, "MARKER_GROUPING_INVALID_COMMAND", "The grouping command kind is unsupported."),
        };
    }

    private static MarkerGroupingEditResult Merge(
        MarkerGroupingState state,
        MarkerGroupingEditCommand command,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(command.SourceSeriesId) ||
            string.IsNullOrWhiteSpace(command.SecondarySeriesId) ||
            string.Equals(command.SourceSeriesId, command.SecondarySeriesId, StringComparison.Ordinal))
        {
            return Failure(state, "MARKER_GROUPING_INVALID_COMMAND", "Merge requires two distinct source series IDs.");
        }

        MarkerSeries? first = FindSeries(state, command.SourceSeriesId);
        MarkerSeries? second = FindSeries(state, command.SecondarySeriesId);
        if (first is null || second is null)
        {
            return Failure(state, "MARKER_GROUPING_SERIES_NOT_FOUND", "A merge source series does not exist.");
        }

        if (!IdentitiesCompatible(first.Shape, first.Fill, second.Shape, second.Fill))
        {
            return Failure(
                state,
                "MARKER_GROUPING_INCOMPATIBLE_IDENTITY",
                "Series with incompatible shape or known fill identities cannot be merged.");
        }

        string mergedId = string.IsNullOrWhiteSpace(command.TargetSeriesId)
            ? first.SeriesId
            : command.TargetSeriesId;
        if (!string.Equals(mergedId, first.SeriesId, StringComparison.Ordinal) &&
            !string.Equals(mergedId, second.SeriesId, StringComparison.Ordinal) &&
            state.Series.Any(series => string.Equals(series.SeriesId, mergedId, StringComparison.Ordinal)))
        {
            return Failure(state, "MARKER_GROUPING_DUPLICATE_SERIES", "The merge target series ID already exists.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        string[] markerIds = first.MarkerIds
            .Concat(second.MarkerIds)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(id => MarkerOrder(state, id))
            .ThenBy(static id => id, StringComparer.Ordinal)
            .ToArray();
        MarkerFill mergedFill = first.Fill == MarkerFill.Unknown ? second.Fill : first.Fill;
        MarkerSymbolDescriptor descriptor = MarkerSymbolMap.Describe(first.Shape, mergedFill);
        string? mergedLegendText = first.LegendText ?? second.LegendText;
        var merged = new MarkerSeries(
            mergedId,
            descriptor.Symbol,
            first.Shape,
            mergedFill,
            mergedLegendText ?? descriptor.AccessibleName,
            first.SemanticRole == second.SemanticRole ? first.SemanticRole : MarkerSeriesRole.Unknown,
            markerIds,
            Math.Clamp((first.Confidence + second.Confidence) / 2, 0, 1),
            mergedLegendText,
            first.SharedBaselineSeriesId ?? second.SharedBaselineSeriesId,
            first.ApplicableProbeSeriesIds.Concat(second.ApplicableProbeSeriesIds).Distinct(StringComparer.Ordinal));

        string[] removedIds = [first.SeriesId, second.SeriesId];
        MarkerSeries[] series = state.Series
            .Where(item => !removedIds.Contains(item.SeriesId, StringComparer.Ordinal))
            .Select(item => RewriteReferences(item, removedIds, mergedId))
            .Append(merged)
            .OrderBy(static item => item.SeriesId, StringComparer.Ordinal)
            .ToArray();
        return Complete(state, command, series, removedIds, markerIds);
    }

    private static MarkerGroupingEditResult Split(
        MarkerGroupingState state,
        MarkerGroupingEditCommand command,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(command.SourceSeriesId) ||
            string.IsNullOrWhiteSpace(command.NewSeriesId) ||
            command.MarkerIds.Count == 0)
        {
            return Failure(state, "MARKER_GROUPING_INVALID_COMMAND", "Split requires a source, a new series ID, and marker IDs.");
        }

        MarkerSeries? source = FindSeries(state, command.SourceSeriesId);
        if (source is null)
        {
            return Failure(state, "MARKER_GROUPING_SERIES_NOT_FOUND", "The split source series does not exist.");
        }

        if (state.Series.Any(series => string.Equals(series.SeriesId, command.NewSeriesId, StringComparison.Ordinal)))
        {
            return Failure(state, "MARKER_GROUPING_DUPLICATE_SERIES", "The split target series ID already exists.");
        }

        var selected = command.MarkerIds.ToHashSet(StringComparer.Ordinal);
        if (selected.Count != command.MarkerIds.Count ||
            selected.Any(id => !source.MarkerIds.Contains(id, StringComparer.Ordinal)) ||
            selected.Count >= source.MarkerIds.Count)
        {
            return Failure(
                state,
                "MARKER_GROUPING_INVALID_COMMAND",
                "Split markers must be a unique, non-empty proper subset of the source series.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        string[] retained = source.MarkerIds.Where(id => !selected.Contains(id)).ToArray();
        string[] separated = source.MarkerIds.Where(selected.Contains).ToArray();
        var updatedSource = CopySeries(source, markerIds: retained);
        var created = new MarkerSeries(
            command.NewSeriesId,
            source.Symbol,
            source.Shape,
            source.Fill,
            source.DisplayName,
            source.SemanticRole,
            separated,
            source.Confidence,
            source.LegendText,
            source.SharedBaselineSeriesId,
            source.ApplicableProbeSeriesIds);
        MarkerSeries[] series = state.Series
            .Where(item => !string.Equals(item.SeriesId, source.SeriesId, StringComparison.Ordinal))
            .Append(updatedSource)
            .Append(created)
            .OrderBy(static item => item.SeriesId, StringComparer.Ordinal)
            .ToArray();
        return Complete(state, command, series, [source.SeriesId, created.SeriesId], separated);
    }

    private static MarkerGroupingEditResult Reassign(
        MarkerGroupingState state,
        MarkerGroupingEditCommand command,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(command.TargetSeriesId) || command.MarkerIds.Count == 0)
        {
            return Failure(state, "MARKER_GROUPING_INVALID_COMMAND", "Reassignment requires a target series and marker IDs.");
        }

        MarkerSeries? target = FindSeries(state, command.TargetSeriesId);
        if (target is null)
        {
            return Failure(state, "MARKER_GROUPING_SERIES_NOT_FOUND", "The reassignment target series does not exist.");
        }

        var markerIds = command.MarkerIds.ToHashSet(StringComparer.Ordinal);
        if (markerIds.Count != command.MarkerIds.Count ||
            markerIds.Any(id => !state.Markers.Any(marker => string.Equals(
                marker.Marker.Marker.MarkerId,
                id,
                StringComparison.Ordinal))))
        {
            return Failure(state, "MARKER_GROUPING_MARKER_NOT_FOUND", "Reassignment markers must be unique and exist in the state.");
        }

        if (!string.IsNullOrWhiteSpace(command.SourceSeriesId))
        {
            MarkerSeries? source = FindSeries(state, command.SourceSeriesId);
            if (source is null || markerIds.Any(id => !source.MarkerIds.Contains(id, StringComparer.Ordinal)))
            {
                return Failure(state, "MARKER_GROUPING_INVALID_COMMAND", "The requested source does not contain every reassigned marker.");
            }
        }


        MarkerGroupingEvidence[] reassignedMarkers = state.Markers
            .Where(marker => markerIds.Contains(marker.Marker.Marker.MarkerId))
            .ToArray();
        if (reassignedMarkers.Any(marker => !IdentitiesCompatible(
                target.Shape,
                target.Fill,
                marker.Marker.Shape,
                marker.Marker.Fill)))
        {
            return Failure(
                state,
                "MARKER_GROUPING_INCOMPATIBLE_IDENTITY",
                "Markers with incompatible shape or known fill identities cannot be reassigned to the target series.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        var changed = false;
        var series = new List<MarkerSeries>(state.Series.Count);
        foreach (MarkerSeries item in state.Series)
        {
            cancellationToken.ThrowIfCancellationRequested();
            IEnumerable<string> ids = item.MarkerIds;
            bool isSelectedSource = string.IsNullOrWhiteSpace(command.SourceSeriesId) ||
                string.Equals(item.SeriesId, command.SourceSeriesId, StringComparison.Ordinal);
            if (isSelectedSource && !string.Equals(item.SeriesId, target.SeriesId, StringComparison.Ordinal))
            {
                string[] filtered = ids.Where(id => !markerIds.Contains(id)).ToArray();
                changed |= filtered.Length != item.MarkerIds.Count;
                ids = filtered;
            }

            if (string.Equals(item.SeriesId, target.SeriesId, StringComparison.Ordinal))
            {
                string[] combined = ids
                    .Concat(markerIds)
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(id => MarkerOrder(state, id))
                    .ThenBy(static id => id, StringComparer.Ordinal)
                    .ToArray();
                changed |= combined.Length != item.MarkerIds.Count;
                ids = combined;
            }

            series.Add(CopySeries(item, markerIds: ids));
        }

        if (!changed)
        {
            return Failure(state, "MARKER_GROUPING_NO_CHANGE", "The reassignment would not change grouping state.");
        }

        string[] affected = string.IsNullOrWhiteSpace(command.SourceSeriesId)
            ? [target.SeriesId]
            : [command.SourceSeriesId, target.SeriesId];
        return Complete(state, command, series, affected, markerIds);
    }

    private static MarkerGroupingEditResult Complete(
        MarkerGroupingState state,
        MarkerGroupingEditCommand command,
        IEnumerable<MarkerSeries> series,
        IEnumerable<string> affectedSeriesIds,
        IEnumerable<string> affectedMarkerIds)
    {
        string[] seriesIds = affectedSeriesIds.Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
        string[] markerIds = affectedMarkerIds.Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
        var auditEvent = new MarkerGroupingAuditEvent(
            CreateEventId(state, command, seriesIds, markerIds),
            command.TimestampUtc,
            command.Kind,
            GroupingCollections.Freeze(seriesIds),
            GroupingCollections.Freeze(markerIds),
            command.Reason.Trim());
        var next = new MarkerGroupingState(
            state.Markers,
            state.Connections,
            series,
            state.AuditEvents.Append(auditEvent));
        MarkerGroupingFailure? validationFailure = ValidateState(next);
        return validationFailure is null
            ? new MarkerGroupingEditResult(next, auditEvent, null)
            : new MarkerGroupingEditResult(state, null, validationFailure);
    }

    private static MarkerGroupingFailure? ValidateState(MarkerGroupingState state)
    {
        string[] markerIds = state.Markers.Select(static marker => marker.Marker.Marker.MarkerId).ToArray();
        if (markerIds.Any(string.IsNullOrWhiteSpace) || markerIds.Distinct(StringComparer.Ordinal).Count() != markerIds.Length)
        {
            return Error("MARKER_GROUPING_INVALID_STATE", "Marker IDs must be unique and non-empty.");
        }

        var knownMarkers = markerIds.ToHashSet(StringComparer.Ordinal);
        string[] seriesIds = state.Series.Select(static series => series.SeriesId).ToArray();
        if (seriesIds.Any(string.IsNullOrWhiteSpace) || seriesIds.Distinct(StringComparer.Ordinal).Count() != seriesIds.Length)
        {
            return Error("MARKER_GROUPING_INVALID_STATE", "Series IDs must be unique and non-empty.");
        }

        var knownSeries = state.Series.ToDictionary(static series => series.SeriesId, StringComparer.Ordinal);
        var assigned = new HashSet<string>(StringComparer.Ordinal);
        foreach (MarkerSeries series in state.Series)
        {
            MarkerGroupingEvidence[] members = state.Markers
                .Where(marker => series.MarkerIds.Contains(marker.Marker.Marker.MarkerId, StringComparer.Ordinal))
                .ToArray();
            MarkerFill[] knownFills = members
                .Select(static marker => marker.Marker.Fill)
                .Where(static fill => fill != MarkerFill.Unknown)
                .Distinct()
                .ToArray();
            if (series.MarkerIds.Count == 0 ||
                series.MarkerIds.Distinct(StringComparer.Ordinal).Count() != series.MarkerIds.Count ||
                series.MarkerIds.Any(id => !knownMarkers.Contains(id)) ||
                series.MarkerIds.Any(id => !assigned.Add(id)) ||
                !double.IsFinite(series.Confidence) || series.Confidence < 0 || series.Confidence > 1 ||
                members.Any(marker => marker.Marker.Shape != series.Shape) ||
                knownFills.Length > 1 ||
                (series.Fill != MarkerFill.Unknown && knownFills.Any(fill => fill != series.Fill)))
            {
                return Error(
                    "MARKER_GROUPING_INVALID_STATE",
                    "Series marker references, identities, and confidence must be valid and non-duplicated.");
            }

            if (series.SharedBaselineSeriesId is not null &&
                (!knownSeries.TryGetValue(series.SharedBaselineSeriesId, out MarkerSeries? baseline) ||
                 baseline.SemanticRole != MarkerSeriesRole.Baseline ||
                 string.Equals(baseline.SeriesId, series.SeriesId, StringComparison.Ordinal)))
            {
                return Error("MARKER_GROUPING_INVALID_STATE", "Shared baseline references must target another baseline series.");
            }

            if (series.ApplicableProbeSeriesIds.Any(id =>
                    !knownSeries.ContainsKey(id) || string.Equals(id, series.SeriesId, StringComparison.Ordinal)))
            {
                return Error("MARKER_GROUPING_INVALID_STATE", "Applicable probe references must target another existing series.");
            }
        }

        if (assigned.Count != knownMarkers.Count)
        {
            return Error("MARKER_GROUPING_INVALID_STATE", "Every marker must belong to exactly one series.");
        }

        return null;
    }

    private static bool IdentitiesCompatible(
        MarkerShape leftShape,
        MarkerFill leftFill,
        MarkerShape rightShape,
        MarkerFill rightFill) =>
        leftShape == rightShape &&
        (leftFill == rightFill || leftFill == MarkerFill.Unknown || rightFill == MarkerFill.Unknown);

    private static MarkerSeries RewriteReferences(
        MarkerSeries series,
        IReadOnlyCollection<string> removedIds,
        string mergedId)
    {
        string? sharedBaseline = series.SharedBaselineSeriesId is not null &&
            removedIds.Contains(series.SharedBaselineSeriesId, StringComparer.Ordinal)
                ? mergedId
                : series.SharedBaselineSeriesId;
        string[] probes = series.ApplicableProbeSeriesIds
            .Select(id => removedIds.Contains(id, StringComparer.Ordinal) ? mergedId : id)
            .Where(id => !string.Equals(id, series.SeriesId, StringComparison.Ordinal))
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        return CopySeries(series, sharedBaselineSeriesId: sharedBaseline, applicableProbeSeriesIds: probes);
    }

    private static MarkerSeries CopySeries(
        MarkerSeries series,
        IEnumerable<string>? markerIds = null,
        string? sharedBaselineSeriesId = null,
        IEnumerable<string>? applicableProbeSeriesIds = null) =>
        new(
            series.SeriesId,
            series.Symbol,
            series.Shape,
            series.Fill,
            series.DisplayName,
            series.SemanticRole,
            markerIds ?? series.MarkerIds,
            series.Confidence,
            series.LegendText,
            sharedBaselineSeriesId ?? series.SharedBaselineSeriesId,
            applicableProbeSeriesIds ?? series.ApplicableProbeSeriesIds);

    private static MarkerSeries? FindSeries(MarkerGroupingState state, string seriesId) =>
        state.Series.SingleOrDefault(series => string.Equals(series.SeriesId, seriesId, StringComparison.Ordinal));

    private static double MarkerOrder(MarkerGroupingState state, string markerId)
    {
        MarkerGroupingEvidence marker = state.Markers.Single(item => string.Equals(
            item.Marker.Marker.MarkerId,
            markerId,
            StringComparison.Ordinal));
        return marker.PrintedSession ?? marker.ObservationIndex ?? marker.Marker.Marker.Center.X;
    }

    private static string CreateEventId(
        MarkerGroupingState state,
        MarkerGroupingEditCommand command,
        IEnumerable<string> seriesIds,
        IEnumerable<string> markerIds)
    {
        string material = string.Join(
            '|',
            state.AuditEvents.Count.ToString(CultureInfo.InvariantCulture),
            command.Kind.ToString(),
            command.TimestampUtc.ToUniversalTime().ToString("O", CultureInfo.InvariantCulture),
            string.Join(',', seriesIds),
            string.Join(',', markerIds),
            command.Reason.Trim());
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(material));
        return new Guid(digest.AsSpan(0, 16)).ToString();
    }

    private static MarkerGroupingEditResult Failure(
        MarkerGroupingState state,
        string code,
        string technicalMessage) =>
        new(state, null, Error(code, technicalMessage));

    private static MarkerGroupingFailure Error(string code, string technicalMessage) =>
        new(code, "error", "Errors." + code, technicalMessage, true, "review_grouping");
}
