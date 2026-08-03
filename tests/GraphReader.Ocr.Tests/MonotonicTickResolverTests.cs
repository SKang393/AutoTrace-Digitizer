// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class MonotonicTickResolverTests
{
    private static readonly double[] ExpectedIncreasingValues = [1d, 6d, 11d, 16d];

    [TestMethod]
    public void IncreasingXTickSequenceRejectsHighConfidenceOcrOutlier()
    {
        TickCandidate[] candidates =
        [
            new("x1", 40, 1, 0.98),
            new("x6", 65, 6, 0.95),
            new("bad-100", 80, 100, 0.92),
            new("x11", 90, 11, 0.96),
            new("x16", 115, 16, 0.94),
        ];

        TickResolutionResult result = MonotonicTickResolver.Resolve(
            candidates,
            TickAxisDirection.IncreasingWithPixels);

        CollectionAssert.AreEqual(
            ExpectedIncreasingValues,
            result.ResolvedTicks.Select(static tick => tick.Value).ToArray());
        Assert.HasCount(1, result.RejectedTicks);
        Assert.AreEqual("bad-100", result.RejectedTicks[0].Candidate.RegionId);
        Assert.IsFalse(result.NeedsReview);
        Assert.IsGreaterThan(0.7d, result.Confidence);
    }

    [TestMethod]
    public void YTickValuesMayDecreaseAsPixelPositionIncreases()
    {
        TickCandidate[] candidates =
        [
            new("y100", 20, 100),
            new("y75", 40, 75),
            new("y50", 60, 50),
            new("y25", 80, 25),
            new("y0", 100, 0),
        ];

        TickResolutionResult result = MonotonicTickResolver.Resolve(
            candidates,
            TickAxisDirection.DecreasingWithPixels);

        Assert.HasCount(5, result.ResolvedTicks);
        Assert.IsEmpty(result.RejectedTicks);
        Assert.IsFalse(result.NeedsReview);
        Assert.IsLessThan(0d, result.Slope!.Value);
        Assert.AreEqual(0d, result.RootMeanSquareErrorPixels, 1e-9);
    }

    [TestMethod]
    public void EquallySupportedDirectionsAreReturnedForReviewInsteadOfArbitraryCertainty()
    {
        TickCandidate[] candidates =
        [
            new("first", 10, 0),
            new("second", 20, 10),
            new("third", 30, 0),
        ];

        TickResolutionResult result = MonotonicTickResolver.Resolve(candidates);

        Assert.IsTrue(result.NeedsReview);
        Assert.IsTrue(result.Reasons.Contains("tick_direction_ambiguous"));
    }

    [TestMethod]
    public void InvalidEvidenceIsRejectedWithStructuredReason()
    {
        TickCandidate[] candidates =
        [
            new("first", 10, 0),
            new("second", 20, 10),
            new("nan", double.NaN, 20),
            new("bad-confidence", 30, 30, 1.1),
        ];

        TickResolutionResult result = MonotonicTickResolver.Resolve(
            candidates,
            TickAxisDirection.IncreasingWithPixels);

        Assert.HasCount(2, result.ResolvedTicks);
        Assert.HasCount(2, result.RejectedTicks);
        Assert.IsTrue(result.RejectedTicks.All(rejected => rejected.Reason == "invalid_tick"));
        Assert.IsTrue(result.NeedsReview);
        Assert.IsLessThan(1d, result.Confidence);
        Assert.IsTrue(result.Reasons.Contains("invalid_tick_evidence"));
    }

    [TestMethod]
    public void SingleTickCannotInventACompleteCalibration()
    {
        TickResolutionResult result = MonotonicTickResolver.Resolve([new TickCandidate("only", 40, 1)]);

        Assert.IsTrue(result.NeedsReview);
        Assert.IsTrue(result.Reasons.Contains("insufficient_monotonic_ticks"));
        Assert.IsNull(result.Slope);
        Assert.IsNull(result.Intercept);
        Assert.AreEqual(0d, result.Confidence);
    }

    [TestMethod]
    public void DifferentValuesAtSamePixelRequireReviewAndCannotRetainFullConfidence()
    {
        TickResolutionResult result = MonotonicTickResolver.Resolve(
            [new TickCandidate("zero", 50, 0), new TickCandidate("ten", 50, 10)],
            TickAxisDirection.IncreasingWithPixels);

        Assert.IsTrue(result.NeedsReview);
        Assert.IsTrue(result.Slope is null || !double.IsFinite(result.Slope.Value));
        Assert.IsFalse(double.IsFinite(result.RootMeanSquareErrorPixels));
        Assert.IsLessThan(1d, result.Confidence);
        Assert.IsTrue(result.Reasons.Any(reason =>
            reason.Contains("degenerate", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("nonfinite", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("fit", StringComparison.OrdinalIgnoreCase)));
    }

    [TestMethod]
    public void StronglyNonlinearTickSpacingRequiresReviewAndReducesConfidence()
    {
        TickResolutionResult result = MonotonicTickResolver.Resolve(
            [
                new TickCandidate("zero", 0, 0),
                new TickCandidate("ten", 1, 10),
                new TickCandidate("twenty", 100, 20),
            ],
            TickAxisDirection.IncreasingWithPixels);

        Assert.IsTrue(result.NeedsReview);
        Assert.IsGreaterThan(10d, result.RootMeanSquareErrorPixels);
        Assert.IsLessThan(1d, result.Confidence);
        Assert.IsTrue(result.Reasons.Any(reason =>
            reason.Contains("spacing", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("residual", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("fit", StringComparison.OrdinalIgnoreCase)));
    }
}
