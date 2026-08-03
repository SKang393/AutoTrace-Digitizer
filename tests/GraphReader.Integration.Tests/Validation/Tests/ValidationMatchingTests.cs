// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Integration.Tests.Validation.Core.Metrics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

#pragma warning disable CA1861 // Small collection expressions are intentional test fixtures.

namespace GraphReader.Integration.Tests.Validation.Tests;

[TestClass]
public sealed class ValidationMatchingTests
{
    [TestMethod]
    public void ValidationOneToOneMatchingMaximizesCardinalityBeforeCost()
    {
        OneToOneMatchingResult result = OneToOneMatcher.MatchByCost(
            expectedCount: 2,
            actualCount: 2,
            (expectedIndex, actualIndex) => (expectedIndex, actualIndex) switch
            {
                (0, 0) => 1,
                (0, 1) => 2,
                (1, 0) => 1,
                _ => null,
            });

        Assert.HasCount(2, result.Matches);
        CollectionAssert.AreEquivalent(
            new[] { "0:1", "1:0" },
            result.Matches.Select(match => $"{match.ExpectedIndex}:{match.ActualIndex}").ToArray());
        Assert.IsEmpty(result.UnmatchedExpectedIndices);
        Assert.IsEmpty(result.UnmatchedActualIndices);
    }

    [TestMethod]
    public void ValidationOneToOneMatchingDoesNotCountDuplicateActualPointsTwice()
    {
        OneToOneMatchingResult result = OneToOneMatcher.MatchPoints(
            [new MetricPoint(10, 10)],
            [new MetricPoint(10, 10), new MetricPoint(11, 10)],
            tolerance: 3);

        Assert.HasCount(1, result.Matches);
        Assert.IsEmpty(result.UnmatchedExpectedIndices);
        CollectionAssert.AreEqual(new[] { 1 }, result.UnmatchedActualIndices.ToArray());
    }

    [TestMethod]
    public void ValidationOneToOneMatchingUsesInclusiveToleranceAndReportsMisses()
    {
        OneToOneMatchingResult result = OneToOneMatcher.MatchPoints(
            [new MetricPoint(0, 0), new MetricPoint(20, 20)],
            [new MetricPoint(3, 4), new MetricPoint(30, 30)],
            tolerance: 5);

        Assert.HasCount(1, result.Matches);
        Assert.AreEqual(5, result.Matches[0].Cost, 1e-12);
        CollectionAssert.AreEqual(new[] { 1 }, result.UnmatchedExpectedIndices.ToArray());
        CollectionAssert.AreEqual(new[] { 1 }, result.UnmatchedActualIndices.ToArray());
    }
}
