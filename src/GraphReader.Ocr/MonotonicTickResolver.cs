// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Ocr;

public enum TickAxisDirection
{
    Auto,
    IncreasingWithPixels,
    DecreasingWithPixels,
}

public sealed record MonotonicTickResolverOptions
{
    public double MaximumAxisRelativeRootMeanSquareError { get; init; } = 0.05;

    public double MinimumAbsoluteSlope { get; init; } = 1e-9;

    public double MinimumPixelSpan { get; init; } = 1;
}

public sealed record TickCandidate(
    string RegionId,
    double PixelPosition,
    double Value,
    double Confidence = 1d);

public sealed record ResolvedTick(
    string RegionId,
    double PixelPosition,
    double Value,
    double Confidence);

public sealed record RejectedTick(TickCandidate Candidate, string Reason);

public sealed record TickResolutionResult(
    IReadOnlyList<ResolvedTick> ResolvedTicks,
    IReadOnlyList<RejectedTick> RejectedTicks,
    bool NeedsReview,
    IReadOnlyList<string> Reasons,
    double? Slope,
    double? Intercept,
    double RootMeanSquareErrorPixels,
    double Confidence);

public static class MonotonicTickResolver
{
    private const double AmbiguityMargin = 0.05;

    public static TickResolutionResult Resolve(
        IReadOnlyList<TickCandidate> candidates,
        TickAxisDirection direction = TickAxisDirection.Auto,
        MonotonicTickResolverOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        options ??= new MonotonicTickResolverOptions();
        if (options.MaximumAxisRelativeRootMeanSquareError <= 0 ||
            options.MinimumAbsoluteSlope <= 0 || options.MinimumPixelSpan <= 0 ||
            !double.IsFinite(options.MaximumAxisRelativeRootMeanSquareError) ||
            !double.IsFinite(options.MinimumAbsoluteSlope) || !double.IsFinite(options.MinimumPixelSpan))
        {
            throw new ArgumentOutOfRangeException(nameof(options));
        }

        var valid = candidates
            .Where(static candidate =>
                !string.IsNullOrWhiteSpace(candidate.RegionId) &&
                double.IsFinite(candidate.PixelPosition) &&
                double.IsFinite(candidate.Value) &&
                candidate.Confidence is >= 0 and <= 1)
            .OrderBy(static candidate => candidate.PixelPosition)
            .ThenBy(static candidate => candidate.RegionId, StringComparer.Ordinal)
            .ToArray();

        var invalid = candidates
            .Except(valid)
            .Select(static candidate => new RejectedTick(candidate, "invalid_tick"))
            .ToArray();
        if (valid.Length < 2)
        {
            var insufficientReasons = new List<string> { "insufficient_monotonic_ticks" };
            if (invalid.Length > 0)
            {
                insufficientReasons.Add("invalid_tick_evidence");
            }

            return Result(
                valid,
                invalid,
                true,
                insufficientReasons,
                null,
                null,
                double.PositiveInfinity,
                0);
        }

        var increasing = FindBestChain(valid, increasing: true);
        var decreasing = FindBestChain(valid, increasing: false);
        Chain selected;
        var reasons = new List<string>();
        var needsReview = false;

        if (direction == TickAxisDirection.IncreasingWithPixels)
        {
            selected = increasing;
        }
        else if (direction == TickAxisDirection.DecreasingWithPixels)
        {
            selected = decreasing;
        }
        else
        {
            selected = increasing.Score >= decreasing.Score ? increasing : decreasing;
            var alternative = ReferenceEquals(selected, increasing) ? decreasing : increasing;
            var denominator = Math.Max(selected.Score, double.Epsilon);
            if ((selected.Score - alternative.Score) / denominator <= AmbiguityMargin)
            {
                needsReview = true;
                reasons.Add("tick_direction_ambiguous");
            }
        }

        if (selected.Items.Length < 2)
        {
            reasons.Add("insufficient_monotonic_ticks");
            needsReview = true;
        }

        var rejected = valid
            .Where(candidate => !selected.Items.Contains(candidate))
            .Select(static candidate => new RejectedTick(candidate, "violates_monotonic_sequence"))
            .Concat(invalid)
            .ToArray();
        var fit = Fit(selected.Items);
        if (rejected.Length > 0)
        {
            reasons.Add("nonmonotonic_tick_candidates_rejected");
        }

        if (invalid.Length > 0)
        {
            reasons.Add("invalid_tick_evidence");
            needsReview = true;
        }

        var pixelSpan = selected.Items.Length == 0
            ? 0
            : selected.Items[^1].PixelPosition - selected.Items[0].PixelPosition;
        var invalidFit = !double.IsFinite(fit.Slope) || !double.IsFinite(fit.Intercept) ||
            !double.IsFinite(fit.RootMeanSquareErrorPixels) ||
            Math.Abs(fit.Slope) < options.MinimumAbsoluteSlope || pixelSpan < options.MinimumPixelSpan;
        var relativeError = invalidFit
            ? double.PositiveInfinity
            : fit.RootMeanSquareErrorPixels / pixelSpan;
        if (invalidFit)
        {
            reasons.Add("degenerate_or_nonfinite_tick_fit");
            needsReview = true;
        }
        else if (relativeError > options.MaximumAxisRelativeRootMeanSquareError)
        {
            reasons.Add("axis_relative_tick_spacing_residual_too_large");
            needsReview = true;
        }

        var supportFraction = selected.Items.Length / (double)Math.Max(1, candidates.Count);
        var fitQuality = invalidFit
            ? 0
            : relativeError <= options.MaximumAxisRelativeRootMeanSquareError
                ? 1
                : Math.Clamp(options.MaximumAxisRelativeRootMeanSquareError / relativeError, 0, 1);
        var confidence = selected.Items.Length < 2
            ? 0
            : Math.Clamp(
                selected.Items.Average(static item => item.Confidence) * supportFraction * fitQuality,
                0,
                1);

        return Result(
            selected.Items,
            rejected,
            needsReview,
            reasons,
            fit.Slope,
            fit.Intercept,
            fit.RootMeanSquareErrorPixels,
            confidence);
    }

    private static Chain FindBestChain(TickCandidate[] candidates, bool increasing)
    {
        var scores = new double[candidates.Length];
        var predecessors = new int[candidates.Length];
        Array.Fill(predecessors, -1);
        var bestIndex = 0;
        for (var index = 0; index < candidates.Length; index++)
        {
            scores[index] = candidates[index].Confidence;
            for (var prior = 0; prior < index; prior++)
            {
                var monotonic = increasing
                    ? candidates[index].Value > candidates[prior].Value
                    : candidates[index].Value < candidates[prior].Value;
                if (monotonic && scores[prior] + candidates[index].Confidence > scores[index])
                {
                    scores[index] = scores[prior] + candidates[index].Confidence;
                    predecessors[index] = prior;
                }
            }

            if (scores[index] > scores[bestIndex])
            {
                bestIndex = index;
            }
        }

        var chain = new List<TickCandidate>();
        for (var index = bestIndex; index >= 0; index = predecessors[index])
        {
            chain.Add(candidates[index]);
            if (predecessors[index] < 0)
            {
                break;
            }
        }

        chain.Reverse();
        return new Chain(chain.ToArray(), scores[bestIndex]);
    }

    private static (double Slope, double Intercept, double RootMeanSquareErrorPixels) Fit(
        TickCandidate[] ticks)
    {
        if (ticks.Length < 2)
        {
            return (double.NaN, double.NaN, double.PositiveInfinity);
        }

        var meanPixel = ticks.Average(static tick => tick.PixelPosition);
        var meanValue = ticks.Average(static tick => tick.Value);
        var numerator = ticks.Sum(tick => (tick.PixelPosition - meanPixel) * (tick.Value - meanValue));
        var denominator = ticks.Sum(tick => Math.Pow(tick.PixelPosition - meanPixel, 2));
        if (Math.Abs(denominator) <= double.Epsilon)
        {
            return (double.NaN, double.NaN, double.PositiveInfinity);
        }

        var slope = numerator / denominator;
        var intercept = meanValue - (slope * meanPixel);
        var squaredPixelErrors = ticks.Select(tick =>
        {
            var predictedPixel = (tick.Value - intercept) / slope;
            return Math.Pow(tick.PixelPosition - predictedPixel, 2);
        });
        return (slope, intercept, Math.Sqrt(squaredPixelErrors.Average()));
    }

    private static TickResolutionResult Result(
        IEnumerable<TickCandidate> accepted,
        IEnumerable<RejectedTick> rejected,
        bool needsReview,
        IEnumerable<string> reasons,
        double? slope,
        double? intercept,
        double rootMeanSquareErrorPixels,
        double confidence) =>
        new(
            OcrCollections.Freeze(accepted.Select(static tick =>
                new ResolvedTick(tick.RegionId, tick.PixelPosition, tick.Value, tick.Confidence))),
            OcrCollections.Freeze(rejected),
            needsReview,
            OcrCollections.Freeze(reasons.Distinct(StringComparer.Ordinal)),
            slope,
            intercept,
            rootMeanSquareErrorPixels,
            confidence);

    private sealed record Chain(TickCandidate[] Items, double Score);
}
