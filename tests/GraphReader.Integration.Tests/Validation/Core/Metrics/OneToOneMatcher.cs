// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests.Validation.Core.Metrics;

public sealed record OneToOneMatch(int ExpectedIndex, int ActualIndex, double Cost);

public sealed record OneToOneMatchingResult(
    IReadOnlyList<OneToOneMatch> Matches,
    IReadOnlyList<int> UnmatchedExpectedIndices,
    IReadOnlyList<int> UnmatchedActualIndices);

public static class OneToOneMatcher
{
    private const double ComparisonEpsilon = 1e-12;

    public static OneToOneMatchingResult MatchPoints(
        IReadOnlyList<MetricPoint> expected,
        IReadOnlyList<MetricPoint> actual,
        double tolerance)
    {
        ArgumentNullException.ThrowIfNull(expected);
        ArgumentNullException.ThrowIfNull(actual);
        MetricGuard.NonNegativeFinite(tolerance, nameof(tolerance));

        return MatchByCost(
            expected.Count,
            actual.Count,
            (expectedIndex, actualIndex) =>
            {
                var distance = expected[expectedIndex].DistanceTo(actual[actualIndex]);
                return distance <= tolerance ? distance : null;
            });
    }

    public static OneToOneMatchingResult MatchByCost(
        int expectedCount,
        int actualCount,
        Func<int, int, double?> costSelector)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(expectedCount);
        ArgumentOutOfRangeException.ThrowIfNegative(actualCount);

        ArgumentNullException.ThrowIfNull(costSelector);

        var source = 0;
        var expectedOffset = 1;
        var actualOffset = expectedOffset + expectedCount;
        var sink = actualOffset + actualCount;
        var graph = Enumerable.Range(0, sink + 1)
            .Select(_ => new List<FlowEdge>())
            .ToArray();

        for (var expectedIndex = 0; expectedIndex < expectedCount; expectedIndex++)
        {
            AddEdge(graph, source, expectedOffset + expectedIndex, 1, 0);
        }

        for (var expectedIndex = 0; expectedIndex < expectedCount; expectedIndex++)
        {
            for (var actualIndex = 0; actualIndex < actualCount; actualIndex++)
            {
                var cost = costSelector(expectedIndex, actualIndex);
                if (cost is null)
                {
                    continue;
                }

                MetricGuard.NonNegativeFinite(cost.Value, nameof(costSelector));
                AddEdge(graph, expectedOffset + expectedIndex, actualOffset + actualIndex, 1, cost.Value);
            }
        }

        for (var actualIndex = 0; actualIndex < actualCount; actualIndex++)
        {
            AddEdge(graph, actualOffset + actualIndex, sink, 1, 0);
        }

        while (TryFindShortestAugmentingPath(graph, source, sink, out var previousNode, out var previousEdge))
        {
            var node = sink;
            while (node != source)
            {
                var parent = previousNode[node];
                var edge = graph[parent][previousEdge[node]];
                edge.Capacity--;
                graph[node][edge.ReverseIndex].Capacity++;
                node = parent;
            }
        }

        var matches = new List<OneToOneMatch>();
        var matchedExpected = new bool[expectedCount];
        var matchedActual = new bool[actualCount];

        for (var expectedIndex = 0; expectedIndex < expectedCount; expectedIndex++)
        {
            var node = expectedOffset + expectedIndex;
            foreach (var edge in graph[node])
            {
                if (edge.To < actualOffset || edge.To >= sink || edge.InitialCapacity != 1 || edge.Capacity != 0)
                {
                    continue;
                }

                var actualIndex = edge.To - actualOffset;
                matches.Add(new OneToOneMatch(expectedIndex, actualIndex, edge.Cost));
                matchedExpected[expectedIndex] = true;
                matchedActual[actualIndex] = true;
            }
        }

        matches.Sort(static (left, right) =>
        {
            var expectedComparison = left.ExpectedIndex.CompareTo(right.ExpectedIndex);
            return expectedComparison != 0
                ? expectedComparison
                : left.ActualIndex.CompareTo(right.ActualIndex);
        });

        return new OneToOneMatchingResult(
            matches,
            Enumerable.Range(0, expectedCount).Where(index => !matchedExpected[index]).ToArray(),
            Enumerable.Range(0, actualCount).Where(index => !matchedActual[index]).ToArray());
    }

    private static bool TryFindShortestAugmentingPath(
        IReadOnlyList<List<FlowEdge>> graph,
        int source,
        int sink,
        out int[] previousNode,
        out int[] previousEdge)
    {
        var distances = Enumerable.Repeat(double.PositiveInfinity, graph.Count).ToArray();
        previousNode = Enumerable.Repeat(-1, graph.Count).ToArray();
        previousEdge = Enumerable.Repeat(-1, graph.Count).ToArray();
        distances[source] = 0;

        for (var iteration = 0; iteration < graph.Count - 1; iteration++)
        {
            var changed = false;
            for (var node = 0; node < graph.Count; node++)
            {
                if (double.IsPositiveInfinity(distances[node]))
                {
                    continue;
                }

                for (var edgeIndex = 0; edgeIndex < graph[node].Count; edgeIndex++)
                {
                    var edge = graph[node][edgeIndex];
                    if (edge.Capacity == 0)
                    {
                        continue;
                    }

                    var candidate = distances[node] + edge.Cost;
                    if (candidate >= distances[edge.To] - ComparisonEpsilon)
                    {
                        continue;
                    }

                    distances[edge.To] = candidate;
                    previousNode[edge.To] = node;
                    previousEdge[edge.To] = edgeIndex;
                    changed = true;
                }
            }

            if (!changed)
            {
                break;
            }
        }

        return previousNode[sink] >= 0;
    }

    private static void AddEdge(List<FlowEdge>[] graph, int from, int to, int capacity, double cost)
    {
        var forward = new FlowEdge(to, graph[to].Count, capacity, cost);
        var reverse = new FlowEdge(from, graph[from].Count, 0, -cost);
        graph[from].Add(forward);
        graph[to].Add(reverse);
    }

    private sealed class FlowEdge(int to, int reverseIndex, int capacity, double cost)
    {
        public int To { get; } = to;

        public int ReverseIndex { get; } = reverseIndex;

        public int Capacity { get; set; } = capacity;

        public int InitialCapacity { get; } = capacity;

        public double Cost { get; } = cost;
    }
}
