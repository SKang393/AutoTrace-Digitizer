// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.App.Integration.Workflow;

public static class OriginalEnhancedConsensus
{
    private const string OriginalOnlyWarning = "ORIGINAL_ONLY_DETECTION_REQUIRES_REVIEW";
    private const string EnhancedOnlyWarning = "ENHANCED_ONLY_DETECTION_REQUIRES_REVIEW";
    private const string PositionWarning = "ORIGINAL_ENHANCED_POSITION_DISAGREEMENT";
    private const string ClassificationWarning = "ORIGINAL_ENHANCED_CLASSIFICATION_DISAGREEMENT";

    public static IReadOnlyList<WorkflowPoint> Merge(
        WorkflowDetectionBatch original,
        WorkflowDetectionBatch? enhanced,
        WorkflowConsensusOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(original);
        RequireVariant(original, WorkflowImageVariant.Original, nameof(original));
        WorkflowConsensusOptions effectiveOptions = options ?? new WorkflowConsensusOptions();
        ValidateOptions(effectiveOptions);

        if (enhanced is null)
        {
            return Freeze(original.Candidates
                .OrderBy(static candidate => candidate.OriginalPixelX)
                .ThenBy(static candidate => candidate.OriginalPixelY)
                .ThenBy(static candidate => candidate.DetectionKey, StringComparer.Ordinal)
                .Select(candidate => ToPoint(
                    candidate,
                    original.Envelope,
                    candidate.Confidence,
                    Array.Empty<string>())));
        }

        RequireVariant(enhanced, WorkflowImageVariant.Enhanced, nameof(enhanced));
        if (original.PanelId != enhanced.PanelId)
        {
            throw new ArgumentException("Original and enhanced detections must describe the same panel.", nameof(enhanced));
        }

        if (original.Envelope.RunId != enhanced.Envelope.RunId ||
            original.Envelope.ProjectId != enhanced.Envelope.ProjectId)
        {
            throw new ArgumentException("Original and enhanced detections must share run and project identity.", nameof(enhanced));
        }

        WorkflowDetectionCandidate[] orderedOriginal = original.Candidates
            .OrderBy(static candidate => candidate.OriginalPixelX)
            .ThenBy(static candidate => candidate.OriginalPixelY)
            .ThenBy(static candidate => candidate.DetectionKey, StringComparer.Ordinal)
            .ToArray();
        WorkflowDetectionCandidate[] orderedEnhanced = enhanced.Candidates
            .OrderBy(static candidate => candidate.OriginalPixelX)
            .ThenBy(static candidate => candidate.OriginalPixelY)
            .ThenBy(static candidate => candidate.DetectionKey, StringComparer.Ordinal)
            .ToArray();

        int[] enhancedByOriginal = CreateMaximumCardinalityMatching(
            orderedOriginal,
            orderedEnhanced,
            effectiveOptions.MaximumMatchDistancePixels);
        var matchedEnhanced = new bool[orderedEnhanced.Length];
        var usedPointIds = new HashSet<string>(StringComparer.Ordinal);
        var points = new List<WorkflowPoint>(orderedOriginal.Length + orderedEnhanced.Length);

        for (int originalIndex = 0; originalIndex < orderedOriginal.Length; originalIndex++)
        {
            WorkflowDetectionCandidate originalCandidate = orderedOriginal[originalIndex];
            int enhancedIndex = enhancedByOriginal[originalIndex];
            if (enhancedIndex < 0)
            {
                WorkflowPoint originalOnly = ToPoint(
                    originalCandidate,
                    original.Envelope,
                    originalCandidate.Confidence * effectiveOptions.OriginalOnlyConfidenceFactor,
                    [OriginalOnlyWarning]);
                points.Add(originalOnly);
                usedPointIds.Add(originalOnly.PointId);
                continue;
            }

            matchedEnhanced[enhancedIndex] = true;
            WorkflowDetectionCandidate enhancedCandidate = orderedEnhanced[enhancedIndex];
            double distance = Distance(originalCandidate, enhancedCandidate);
            WorkflowPoint consensus = MergePair(
                originalCandidate,
                original.Envelope,
                enhancedCandidate,
                enhanced.Envelope,
                distance,
                effectiveOptions);
            consensus = consensus with { PointId = MakeUniquePointId(consensus.PointId, usedPointIds) };
            points.Add(consensus);
            usedPointIds.Add(consensus.PointId);
        }

        for (int index = 0; index < orderedEnhanced.Length; index++)
        {
            if (matchedEnhanced[index])
            {
                continue;
            }

            WorkflowDetectionCandidate candidate = orderedEnhanced[index];
            WorkflowPoint enhancedOnly = ToPoint(
                candidate,
                enhanced.Envelope,
                candidate.Confidence * effectiveOptions.EnhancedOnlyConfidenceFactor,
                [EnhancedOnlyWarning]);
            enhancedOnly = enhancedOnly with
            {
                PointId = MakeUniquePointId(enhancedOnly.PointId, usedPointIds),
            };
            points.Add(enhancedOnly);
            usedPointIds.Add(enhancedOnly.PointId);
        }

        return Freeze(points
            .OrderBy(static point => point.OriginalPixelX)
            .ThenBy(static point => point.OriginalPixelY)
            .ThenBy(static point => point.PointId, StringComparer.Ordinal));
    }

    private static WorkflowPoint MergePair(
        WorkflowDetectionCandidate original,
        WorkflowVisionEnvelope originalEnvelope,
        WorkflowDetectionCandidate enhanced,
        WorkflowVisionEnvelope enhancedEnvelope,
        double distance,
        WorkflowConsensusOptions options)
    {
        var warnings = new List<string>(2);
        double confidence = (original.Confidence + enhanced.Confidence) / 2d;
        if (distance > options.AgreementDistancePixels)
        {
            warnings.Add(PositionWarning);
            double disagreementFraction = distance / options.MaximumMatchDistancePixels;
            confidence *= Math.Max(0.5d, 1d - (disagreementFraction * 0.5d));
        }

        bool classificationAgrees =
            string.Equals(original.Symbol, enhanced.Symbol, StringComparison.Ordinal) &&
            string.Equals(original.Shape, enhanced.Shape, StringComparison.Ordinal) &&
            string.Equals(original.Fill, enhanced.Fill, StringComparison.Ordinal);
        if (!classificationAgrees)
        {
            warnings.Add(ClassificationWarning);
            confidence *= 0.75d;
        }

        string? originalModelVersion = originalEnvelope.Model?.Version;
        string? enhancedModelVersion = enhancedEnvelope.Model?.Version;
        string? modelVersion = string.Equals(originalModelVersion, enhancedModelVersion, StringComparison.Ordinal)
            ? originalModelVersion
            : string.Join(
                "+",
                new[] { originalModelVersion, enhancedModelVersion }
                    .Where(static value => !string.IsNullOrWhiteSpace(value))
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(static value => value, StringComparer.Ordinal));

        return new WorkflowPoint(
            original.PointId,
            original.DetectionKey,
            (original.OriginalPixelX + enhanced.OriginalPixelX) / 2d,
            (original.OriginalPixelY + enhanced.OriginalPixelY) / 2d,
            Math.Clamp(confidence, 0d, 1d),
            WorkflowImageVariant.Consensus,
            WorkflowReviewStatus.Unreviewed,
            original.Symbol,
            original.Shape,
            original.Fill,
            original.SeriesId ?? enhanced.SeriesId,
            original.PhaseId ?? enhanced.PhaseId,
            original.GraphX ?? enhanced.GraphX,
            original.GraphY ?? enhanced.GraphY,
            "consensus",
            string.IsNullOrEmpty(modelVersion) ? null : modelVersion,
            isManual: false,
            warnings: warnings);
    }

    private static WorkflowPoint ToPoint(
        WorkflowDetectionCandidate candidate,
        WorkflowVisionEnvelope envelope,
        double confidence,
        IEnumerable<string> warnings) =>
        new(
            candidate.PointId,
            candidate.DetectionKey,
            candidate.OriginalPixelX,
            candidate.OriginalPixelY,
            Math.Clamp(confidence, 0d, 1d),
            candidate.SourceImage,
            WorkflowReviewStatus.Unreviewed,
            candidate.Symbol,
            candidate.Shape,
            candidate.Fill,
            candidate.SeriesId,
            candidate.PhaseId,
            candidate.GraphX,
            candidate.GraphY,
            envelope.Stage,
            envelope.Model?.Version,
            isManual: false,
            warnings: warnings);

    private static int[] CreateMaximumCardinalityMatching(
        WorkflowDetectionCandidate[] original,
        WorkflowDetectionCandidate[] enhanced,
        double maximumDistance)
    {
        int sourceNode = 0;
        int originalOffset = 1;
        int enhancedOffset = originalOffset + original.Length;
        int sinkNode = enhancedOffset + enhanced.Length;
        List<MatchingEdge>[] graph = Enumerable.Range(0, sinkNode + 1)
            .Select(static _ => new List<MatchingEdge>())
            .ToArray();

        for (int originalIndex = 0; originalIndex < original.Length; originalIndex++)
        {
            AddMatchingEdge(graph, sourceNode, originalOffset + originalIndex, cost: 0);
            for (int enhancedIndex = 0; enhancedIndex < enhanced.Length; enhancedIndex++)
            {
                double distance = Distance(original[originalIndex], enhanced[enhancedIndex]);
                if (distance <= maximumDistance)
                {
                    AddMatchingEdge(
                        graph,
                        originalOffset + originalIndex,
                        enhancedOffset + enhancedIndex,
                        distance);
                }
            }
        }

        for (int enhancedIndex = 0; enhancedIndex < enhanced.Length; enhancedIndex++)
        {
            AddMatchingEdge(graph, enhancedOffset + enhancedIndex, sinkNode, cost: 0);
        }

        while (TryFindMinimumCostAugmentingPath(graph, sourceNode, sinkNode, out int[] previousNodes, out int[] previousEdges))
        {
            int node = sinkNode;
            while (node != sourceNode)
            {
                int previousNode = previousNodes[node];
                MatchingEdge edge = graph[previousNode][previousEdges[node]];
                edge.Capacity--;
                graph[node][edge.ReverseIndex].Capacity++;
                node = previousNode;
            }
        }

        int[] enhancedByOriginal = Enumerable.Repeat(-1, original.Length).ToArray();
        for (int originalIndex = 0; originalIndex < original.Length; originalIndex++)
        {
            MatchingEdge? match = graph[originalOffset + originalIndex]
                .FirstOrDefault(edge =>
                    edge.To >= enhancedOffset &&
                    edge.To < sinkNode &&
                    edge.Capacity == 0);
            if (match is not null)
            {
                enhancedByOriginal[originalIndex] = match.To - enhancedOffset;
            }
        }

        return enhancedByOriginal;
    }

    private static void AddMatchingEdge(
        IReadOnlyList<List<MatchingEdge>> graph,
        int from,
        int to,
        double cost)
    {
        var forward = new MatchingEdge(to, graph[to].Count, capacity: 1, cost);
        var reverse = new MatchingEdge(from, graph[from].Count, capacity: 0, -cost);
        graph[from].Add(forward);
        graph[to].Add(reverse);
    }

    private static bool TryFindMinimumCostAugmentingPath(
        IReadOnlyList<List<MatchingEdge>> graph,
        int source,
        int sink,
        out int[] previousNodes,
        out int[] previousEdges)
    {
        double[] distances = Enumerable.Repeat(double.PositiveInfinity, graph.Count).ToArray();
        previousNodes = Enumerable.Repeat(-1, graph.Count).ToArray();
        previousEdges = Enumerable.Repeat(-1, graph.Count).ToArray();
        distances[source] = 0;

        for (int iteration = 0; iteration < graph.Count - 1; iteration++)
        {
            bool updated = false;
            for (int from = 0; from < graph.Count; from++)
            {
                if (double.IsPositiveInfinity(distances[from]))
                {
                    continue;
                }

                for (int edgeIndex = 0; edgeIndex < graph[from].Count; edgeIndex++)
                {
                    MatchingEdge edge = graph[from][edgeIndex];
                    if (edge.Capacity == 0)
                    {
                        continue;
                    }

                    double candidate = distances[from] + edge.Cost;
                    if (candidate >= distances[edge.To] - 1e-12)
                    {
                        continue;
                    }

                    distances[edge.To] = candidate;
                    previousNodes[edge.To] = from;
                    previousEdges[edge.To] = edgeIndex;
                    updated = true;
                }
            }

            if (!updated)
            {
                break;
            }
        }

        return previousNodes[sink] >= 0;
    }

    private static double Distance(WorkflowDetectionCandidate left, WorkflowDetectionCandidate right)
    {
        double deltaX = left.OriginalPixelX - right.OriginalPixelX;
        double deltaY = left.OriginalPixelY - right.OriginalPixelY;
        return Math.Sqrt((deltaX * deltaX) + (deltaY * deltaY));
    }

    private static string MakeUniquePointId(string requestedPointId, HashSet<string> usedPointIds)
    {
        if (!usedPointIds.Contains(requestedPointId))
        {
            return requestedPointId;
        }

        string candidate = $"{requestedPointId}:enhanced";
        int suffix = 2;
        while (usedPointIds.Contains(candidate))
        {
            candidate = $"{requestedPointId}:enhanced:{suffix}";
            suffix++;
        }

        return candidate;
    }

    private static void RequireVariant(
        WorkflowDetectionBatch batch,
        WorkflowImageVariant expected,
        string parameterName)
    {
        if (batch.SourceImage != expected)
        {
            throw new ArgumentException($"The batch must contain {expected} detections.", parameterName);
        }
    }

    private static void ValidateOptions(WorkflowConsensusOptions options)
    {
        if (!double.IsFinite(options.AgreementDistancePixels) || options.AgreementDistancePixels < 0d)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Agreement distance must be finite and non-negative.");
        }

        if (!double.IsFinite(options.MaximumMatchDistancePixels) ||
            options.MaximumMatchDistancePixels < options.AgreementDistancePixels)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Maximum match distance must cover the agreement distance.");
        }

        WorkflowContractGuards.RequireConfidence(
            options.OriginalOnlyConfidenceFactor,
            nameof(options.OriginalOnlyConfidenceFactor));
        WorkflowContractGuards.RequireConfidence(
            options.EnhancedOnlyConfidenceFactor,
            nameof(options.EnhancedOnlyConfidenceFactor));
    }

    private static IReadOnlyList<WorkflowPoint> Freeze(IEnumerable<WorkflowPoint> points) =>
        WorkflowCollections.Freeze(points);

    private sealed class MatchingEdge
    {
        public MatchingEdge(int to, int reverseIndex, int capacity, double cost)
        {
            To = to;
            ReverseIndex = reverseIndex;
            Capacity = capacity;
            Cost = cost;
        }

        public int To { get; }

        public int ReverseIndex { get; }

        public int Capacity { get; set; }

        public double Cost { get; }
    }
}
