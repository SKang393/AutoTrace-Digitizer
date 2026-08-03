// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using GraphReader.Markers.Classification;

namespace GraphReader.Markers.Grouping;

public sealed class MarkerSeriesGrouper : IMarkerSeriesGrouper
{
    public ValueTask<MarkerGroupingResult> GroupAsync(
        MarkerGroupingRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        var total = Stopwatch.StartNew();
        MarkerGroupingFailure? failure = Validate(request);
        if (failure is not null)
        {
            total.Stop();
            return ValueTask.FromResult(new MarkerGroupingResult(
                null,
                new MarkerGroupingTiming(0, 0, total.Elapsed.TotalMilliseconds),
                0,
                Array.Empty<string>(),
                failure));
        }

        var grouping = Stopwatch.StartNew();
        MarkerGroupingEvidence[] markers = request.Markers
            .OrderBy(MarkerOrder)
            .ThenBy(static item => item.Marker.Marker.Center.Y)
            .ThenBy(static item => item.Marker.Marker.MarkerId, StringComparer.Ordinal)
            .ToArray();
        var indexById = markers
            .Select((marker, index) => (marker.Marker.Marker.MarkerId, index))
            .ToDictionary(static item => item.MarkerId, static item => item.index, StringComparer.Ordinal);
        var connections = request.Connections
            .Where(connection => connection.Confidence >= request.Options.MinimumConnectionConfidence)
            .ToDictionary(
                static connection => EdgeKey(connection.FromMarkerId, connection.ToMarkerId),
                static connection => connection.Confidence,
                StringComparer.Ordinal);
        var union = new UnionFind(markers.Select(static marker => marker.Marker.Fill));
        var pairScores = new Dictionary<string, double>(StringComparer.Ordinal);
        for (var left = 0; left < markers.Length; left++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (var right = left + 1; right < markers.Length; right++)
            {
                MarkerGroupingEvidence a = markers[left];
                MarkerGroupingEvidence b = markers[right];
                if (!ShapeFillCompatible(a.Marker, b.Marker))
                {
                    continue;
                }

                double gap = Math.Abs(MarkerOrder(a) - MarkerOrder(b));
                if (gap > request.Options.MaximumSessionGap)
                {
                    continue;
                }

                string edgeKey = EdgeKey(a.Marker.Marker.MarkerId, b.Marker.Marker.MarkerId);
                connections.TryGetValue(edgeKey, out double connectionConfidence);
                double score = PairScore(a, b, connectionConfidence, request);
                pairScores[edgeKey] = score;
                if (score >= request.Options.MinimumGroupingScore)
                {
                    union.TryJoin(left, right);
                }
            }
        }

        List<MarkerGroupingEvidence[]> components = Enumerable.Range(0, markers.Length)
            .GroupBy(union.Find)
            .Select(group => group.Select(index => markers[index]).ToArray())
            .OrderBy(group => group.Min(MarkerOrder))
            .ThenBy(group => group[0].Marker.Shape)
            .ThenBy(group => group[0].Marker.Fill)
            .ThenBy(group => group[0].Marker.Marker.MarkerId, StringComparer.Ordinal)
            .ToList();
        string? baselineRegion = markers
            .Where(static item => !string.IsNullOrWhiteSpace(item.PhaseRegionId))
            .OrderBy(MarkerOrder)
            .Select(static item => item.PhaseRegionId)
            .FirstOrDefault();
        var series = new List<MarkerSeries>(components.Count);
        var warnings = new SortedSet<string>(StringComparer.Ordinal);
        foreach (MarkerGroupingEvidence[] component in components)
        {
            cancellationToken.ThrowIfCancellationRequested();
            MarkerShape shape = component[0].Marker.Shape;
            MarkerFill fill = component
                .Select(static item => item.Marker.Fill)
                .FirstOrDefault(static value => value != MarkerFill.Unknown, MarkerFill.Unknown);
            MarkerSymbolDescriptor descriptor = MarkerSymbolMap.Describe(shape, fill);
            MarkerLegendEvidence? nameEvidence = SelectNameEvidence(
                request.LegendEvidence,
                shape,
                fill,
                warnings);
            string displayName = nameEvidence?.Text.Trim() ?? descriptor.AccessibleName;
            string? legendText = nameEvidence is null ? null : displayName;
            MarkerSeriesRole role = DetermineRole(component, baselineRegion, nameEvidence);
            string[] markerIds = component
                .OrderBy(MarkerOrder)
                .ThenBy(static item => item.Marker.Marker.Center.Y)
                .ThenBy(static item => item.Marker.Marker.MarkerId, StringComparer.Ordinal)
                .Select(static item => item.Marker.Marker.MarkerId)
                .ToArray();
            double confidence = ComponentConfidence(component, pairScores);
            series.Add(new MarkerSeries(
                CreateSeriesId(shape, fill, markerIds),
                descriptor.Symbol,
                shape,
                fill,
                displayName,
                role,
                markerIds,
                confidence,
                legendText));
        }

        MarkerSeries? baseline = series
            .Where(static item => item.SemanticRole == MarkerSeriesRole.Baseline)
            .OrderByDescending(static item => item.PointCount)
            .ThenBy(static item => item.SeriesId, StringComparer.Ordinal)
            .FirstOrDefault();
        if (baseline is not null && series.Any(item => item.SemanticRole == MarkerSeriesRole.Intervention))
        {
            series = series
                .Select(item => item.SemanticRole == MarkerSeriesRole.Intervention
                    ? CopySeries(item, baseline.SeriesId)
                    : item)
                .ToList();
        }

        series = series.OrderBy(static item => item.SeriesId, StringComparer.Ordinal).ToList();
        grouping.Stop();
        total.Stop();
        var state = new MarkerGroupingState(markers, request.Connections, series);
        double resultConfidence = series.Count == 0 ? 0 : series.Average(static item => item.Confidence);
        return ValueTask.FromResult(new MarkerGroupingResult(
            state,
            new MarkerGroupingTiming(0, grouping.Elapsed.TotalMilliseconds, total.Elapsed.TotalMilliseconds),
            Math.Clamp(resultConfidence, 0, 1),
            GroupingCollections.Freeze(warnings),
            null));
    }

    private static double PairScore(
        MarkerGroupingEvidence left,
        MarkerGroupingEvidence right,
        double connectionConfidence,
        MarkerGroupingRequest request)
    {
        MarkerGroupingOptions options = request.Options;
        double shapeFill = ShapeFillScore(left.Marker, right.Marker);
        double embedding = Math.Max(0, Cosine(left.Marker.Embedding, right.Marker.Embedding));
        if (embedding < options.MinimumEmbeddingSimilarity)
        {
            embedding = 0;
        }

        double gap = Math.Abs(MarkerOrder(left) - MarkerOrder(right));
        double order = Math.Clamp(1 - (gap / Math.Max(options.MaximumSessionGap, 1)), 0, 1);
        double phase = string.IsNullOrWhiteSpace(left.PhaseRegionId) || string.IsNullOrWhiteSpace(right.PhaseRegionId)
            ? 0.5
            : string.Equals(left.PhaseRegionId, right.PhaseRegionId, StringComparison.Ordinal) ? 1 : 0.25;
        double legend = HasCompatibleLegend(request.LegendEvidence, left.Marker.Shape, left.Marker.Fill) ? 1 : 0.5;
        return (shapeFill * options.ShapeFillWeight) +
            (embedding * options.EmbeddingWeight) +
            (connectionConfidence * options.ConnectionWeight) +
            (order * options.SessionOrderWeight) +
            (phase * options.PhaseContinuityWeight) +
            (legend * options.LegendWeight);
    }

    private static bool ShapeFillCompatible(ClassifiedMarker left, ClassifiedMarker right)
    {
        if (left.Shape != right.Shape)
        {
            return false;
        }

        return left.Fill == right.Fill || left.Fill == MarkerFill.Unknown || right.Fill == MarkerFill.Unknown;
    }

    private static double ShapeFillScore(ClassifiedMarker left, ClassifiedMarker right)
    {
        if (left.Shape != right.Shape)
        {
            return 0;
        }

        if (left.Fill == right.Fill)
        {
            return 1;
        }

        return left.Fill == MarkerFill.Unknown || right.Fill == MarkerFill.Unknown ? 0.65 : 0;
    }

    private static double Cosine(IReadOnlyList<float> left, IReadOnlyList<float> right)
    {
        if (left.Count != right.Count || left.Count == 0)
        {
            return -1;
        }

        var dot = 0d;
        var leftNorm = 0d;
        var rightNorm = 0d;
        for (var index = 0; index < left.Count; index++)
        {
            dot += left[index] * (double)right[index];
            leftNorm += left[index] * (double)left[index];
            rightNorm += right[index] * (double)right[index];
        }

        return leftNorm <= 0 || rightNorm <= 0
            ? -1
            : dot / Math.Sqrt(leftNorm * rightNorm);
    }

    private static bool HasCompatibleLegend(
        IReadOnlyList<MarkerLegendEvidence> evidence,
        MarkerShape shape,
        MarkerFill fill) =>
        evidence.Any(item =>
            item.Shape == shape && item.Fill == fill && IsNameEvidenceAllowed(item));

    private static MarkerLegendEvidence? SelectNameEvidence(
        IReadOnlyList<MarkerLegendEvidence> evidence,
        MarkerShape shape,
        MarkerFill fill,
        SortedSet<string> warnings)
    {
        foreach (MarkerLegendEvidence item in evidence
                     .Where(item => item.Shape == shape && item.Fill == fill)
                     .OrderByDescending(static item => item.Confidence)
                     .ThenBy(static item => item.Text, StringComparer.Ordinal))
        {
            if (IsNameEvidenceAllowed(item) && !string.IsNullOrWhiteSpace(item.Text))
            {
                return item;
            }

            if (item.Source is MarkerTextEvidenceSource.Participant or MarkerTextEvidenceSource.Annotation)
            {
                warnings.Add("non_legend_text_ignored_for_series_name");
            }
        }

        return null;
    }

    private static bool IsNameEvidenceAllowed(MarkerLegendEvidence evidence) =>
        evidence.Source is MarkerTextEvidenceSource.Legend or MarkerTextEvidenceSource.UserConfirmed ||
        evidence.ExplicitlyConfirmed;

    private static MarkerSeriesRole DetermineRole(
        IReadOnlyList<MarkerGroupingEvidence> component,
        string? baselineRegion,
        MarkerLegendEvidence? nameEvidence)
    {
        if (nameEvidence is not null && IsNameEvidenceAllowed(nameEvidence))
        {
            string normalized = nameEvidence.Text.Trim();
            if (string.Equals(normalized, "generalization", StringComparison.OrdinalIgnoreCase))
            {
                return MarkerSeriesRole.Generalization;
            }

            if (string.Equals(normalized, "maintenance", StringComparison.OrdinalIgnoreCase))
            {
                return MarkerSeriesRole.Maintenance;
            }
        }

        return component.All(item =>
                baselineRegion is not null &&
                string.Equals(item.PhaseRegionId, baselineRegion, StringComparison.Ordinal))
            ? MarkerSeriesRole.Baseline
            : MarkerSeriesRole.Intervention;
    }

    private static double ComponentConfidence(
        IReadOnlyList<MarkerGroupingEvidence> component,
        Dictionary<string, double> pairScores)
    {
        double markerConfidence = component.Average(static item => item.Marker.Confidence);
        double[] internalScores = component
            .SelectMany((left, leftIndex) => component.Skip(leftIndex + 1).Select(right =>
                pairScores.TryGetValue(
                    EdgeKey(left.Marker.Marker.MarkerId, right.Marker.Marker.MarkerId),
                    out double value)
                    ? value
                    : double.NaN))
            .Where(double.IsFinite)
            .ToArray();
        return internalScores.Length == 0
            ? Math.Clamp(markerConfidence * 0.85, 0, 1)
            : Math.Clamp((markerConfidence + internalScores.Average()) / 2, 0, 1);
    }

    private static MarkerSeries CopySeries(MarkerSeries series, string sharedBaselineSeriesId) =>
        new(
            series.SeriesId,
            series.Symbol,
            series.Shape,
            series.Fill,
            series.DisplayName,
            series.SemanticRole,
            series.MarkerIds,
            series.Confidence,
            series.LegendText,
            sharedBaselineSeriesId,
            series.ApplicableProbeSeriesIds);

    private static double MarkerOrder(MarkerGroupingEvidence marker) =>
        marker.PrintedSession ?? marker.ObservationIndex ?? marker.Marker.Marker.Center.X;

    private static string EdgeKey(string left, string right) =>
        string.CompareOrdinal(left, right) <= 0 ? left + "\n" + right : right + "\n" + left;

    private static string CreateSeriesId(MarkerShape shape, MarkerFill fill, IEnumerable<string> markerIds)
    {
        string material = string.Join('|', shape.ToString(), fill.ToString(), string.Join(',', markerIds));
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(material));
        return new Guid(digest.AsSpan(0, 16)).ToString();
    }

    private static MarkerGroupingFailure? Validate(MarkerGroupingRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.ProjectId) || string.IsNullOrWhiteSpace(request.PanelId))
        {
            return Error("MARKER_GROUPING_INVALID_REQUEST", "Project and panel IDs are required.");
        }

        if (request.ContractVersion != MarkerGroupingContract.Version)
        {
            return Error("MARKER_GROUPING_CONTRACT_UNSUPPORTED", "The grouping contract version is unsupported.");
        }

        MarkerGroupingOptions options = request.Options;
        double weight = options.ShapeFillWeight + options.EmbeddingWeight + options.ConnectionWeight +
            options.SessionOrderWeight + options.PhaseContinuityWeight + options.LegendWeight;
        if (!double.IsFinite(weight) || Math.Abs(weight - 1) > 1e-9 ||
            !double.IsFinite(options.MinimumEmbeddingSimilarity) || options.MinimumEmbeddingSimilarity is < -1 or > 1 ||
            !double.IsFinite(options.MinimumConnectionConfidence) || options.MinimumConnectionConfidence is < 0 or > 1 ||
            !double.IsFinite(options.MinimumGroupingScore) || options.MinimumGroupingScore is < 0 or > 1 ||
            !double.IsFinite(options.MaximumSessionGap) || options.MaximumSessionGap <= 0 ||
            string.IsNullOrWhiteSpace(options.StageVersion))
        {
            return Error("MARKER_GROUPING_INVALID_OPTIONS", "Grouping weights and thresholds are invalid.");
        }

        var markerIds = new HashSet<string>(StringComparer.Ordinal);
        int? embeddingLength = null;
        foreach (MarkerGroupingEvidence marker in request.Markers)
        {
            if (marker is null || marker.Marker is null ||
                string.IsNullOrWhiteSpace(marker.Marker.Marker.MarkerId) ||
                !markerIds.Add(marker.Marker.Marker.MarkerId) ||
                !marker.Marker.Marker.Center.IsFinite ||
                marker.Marker.Embedding.Count == 0 ||
                marker.Marker.Embedding.Any(static value => !float.IsFinite(value)) ||
                (marker.ObservationIndex is not null && marker.ObservationIndex <= 0) ||
                (marker.PrintedSession is not null && !double.IsFinite(marker.PrintedSession.Value)))
            {
                return Error("MARKER_GROUPING_INVALID_REQUEST", "Markers must be unique and contain finite grouping evidence.");
            }

            embeddingLength ??= marker.Marker.Embedding.Count;
            if (embeddingLength != marker.Marker.Embedding.Count)
            {
                return Error("MARKER_GROUPING_INVALID_REQUEST", "All marker embeddings must have equal length.");
            }
        }

        var edges = new HashSet<string>(StringComparer.Ordinal);
        foreach (MarkerConnection connection in request.Connections)
        {
            if (!markerIds.Contains(connection.FromMarkerId) || !markerIds.Contains(connection.ToMarkerId) ||
                string.Equals(connection.FromMarkerId, connection.ToMarkerId, StringComparison.Ordinal) ||
                !double.IsFinite(connection.Confidence) || connection.Confidence is < 0 or > 1 ||
                !edges.Add(EdgeKey(connection.FromMarkerId, connection.ToMarkerId)))
            {
                return Error("MARKER_GROUPING_INVALID_CONNECTION", "Connections must be unique and reference distinct known markers.");
            }
        }

        foreach (MarkerLegendEvidence evidence in request.LegendEvidence)
        {
            if (evidence is null || !double.IsFinite(evidence.Confidence) || evidence.Confidence is < 0 or > 1)
            {
                return Error("MARKER_GROUPING_INVALID_REQUEST", "Legend evidence confidence must be finite and normalized.");
            }
        }

        return null;
    }

    private static MarkerGroupingFailure Error(string code, string technicalMessage) =>
        new(code, "error", "Errors." + code, technicalMessage, true, "review_grouping");

    private sealed class UnionFind
    {
        private readonly int[] _parent;
        private readonly MarkerFill?[] _knownFill;

        public UnionFind(IEnumerable<MarkerFill> fills)
        {
            MarkerFill[] values = fills.ToArray();
            _parent = Enumerable.Range(0, values.Length).ToArray();
            _knownFill = values
                .Select(static fill => fill == MarkerFill.Unknown ? (MarkerFill?)null : fill)
                .ToArray();
        }

        public int Find(int value)
        {
            while (_parent[value] != value)
            {
                _parent[value] = _parent[_parent[value]];
                value = _parent[value];
            }

            return value;
        }

        public bool TryJoin(int left, int right)
        {
            int leftRoot = Find(left);
            int rightRoot = Find(right);
            if (leftRoot == rightRoot)
            {
                return true;
            }

            if (_knownFill[leftRoot] is not null &&
                _knownFill[rightRoot] is not null &&
                _knownFill[leftRoot] != _knownFill[rightRoot])
            {
                return false;
            }

            int minimum = Math.Min(leftRoot, rightRoot);
            int maximum = Math.Max(leftRoot, rightRoot);
            _parent[maximum] = minimum;
            _knownFill[minimum] ??= _knownFill[maximum];
            return true;
        }
    }
}
