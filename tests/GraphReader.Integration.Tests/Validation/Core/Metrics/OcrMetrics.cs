// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests.Validation.Core.Metrics;

public enum TickAxis
{
    X,
    Y
}

public sealed record TickRecognition(
    string TickId,
    TickAxis Axis,
    string ExpectedText,
    string? ActualText);

public sealed record OcrTextRecognition(
    string RegionId,
    string ExpectedText,
    string? ActualText);

public sealed record OcrMetricInput(
    MetricCaseIdentity Identity,
    IReadOnlyList<TickRecognition> Ticks,
    IReadOnlyList<OcrTextRecognition> TextRegions);

public sealed record OcrTextError(
    string RegionId,
    string ExpectedText,
    string? ActualText,
    int EditDistance);

public sealed record OcrMetrics(
    int XTickSupport,
    int XTickExactMatches,
    double XTickNumericExactMatch,
    int YTickSupport,
    int YTickExactMatches,
    double YTickNumericExactMatch,
    int ExpectedCharacterCount,
    int CharacterEditCount,
    double CharacterErrorRate,
    IReadOnlyList<OcrTextError> TextErrors);

public static class OcrMetricsCalculator
{
    public static MetricOutcome<OcrMetrics> Calculate(OcrMetricInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        MetricGuard.Identity(input.Identity);
        ArgumentNullException.ThrowIfNull(input.Ticks);
        ArgumentNullException.ThrowIfNull(input.TextRegions);

        var failures = new List<MetricFailure>();
        var identifiers = new HashSet<string>(StringComparer.Ordinal);
        foreach (var tick in input.Ticks)
        {
            ArgumentNullException.ThrowIfNull(tick);
            ValidateTextObservation(tick.TickId, tick.ExpectedText, identifiers, nameof(input.Ticks));
            if (!string.Equals(tick.ExpectedText, tick.ActualText, StringComparison.Ordinal))
            {
                failures.Add(Mismatch(
                    input.Identity,
                    tick.Axis == TickAxis.X ? "x_tick_numeric_exact_match" : "y_tick_numeric_exact_match",
                    tick.TickId,
                    tick.ExpectedText,
                    tick.ActualText));
            }
        }

        var textErrors = new List<OcrTextError>();
        foreach (var region in input.TextRegions)
        {
            ArgumentNullException.ThrowIfNull(region);
            ValidateTextObservation(region.RegionId, region.ExpectedText, identifiers, nameof(input.TextRegions));
            var editDistance = LevenshteinDistance(region.ExpectedText, region.ActualText ?? string.Empty);
            textErrors.Add(new OcrTextError(
                region.RegionId,
                region.ExpectedText,
                region.ActualText,
                editDistance));

            if (editDistance != 0)
            {
                failures.Add(Mismatch(
                    input.Identity,
                    "ocr_character_error_rate",
                    region.RegionId,
                    region.ExpectedText,
                    region.ActualText));
            }
        }

        var xTicks = input.Ticks.Where(tick => tick.Axis == TickAxis.X).ToArray();
        var yTicks = input.Ticks.Where(tick => tick.Axis == TickAxis.Y).ToArray();
        var xMatches = xTicks.Count(ExactMatch);
        var yMatches = yTicks.Count(ExactMatch);
        var expectedCharacters = textErrors.Sum(error => error.ExpectedText.Length);
        var edits = textErrors.Sum(error => error.EditDistance);
        var score = new OcrMetrics(
            xTicks.Length,
            xMatches,
            MetricMath.Ratio(xMatches, xTicks.Length, whenEmpty: 1),
            yTicks.Length,
            yMatches,
            MetricMath.Ratio(yMatches, yTicks.Length, whenEmpty: 1),
            expectedCharacters,
            edits,
            expectedCharacters == 0 ? (edits == 0 ? 0 : edits) : (double)edits / expectedCharacters,
            textErrors);

        return new MetricOutcome<OcrMetrics>(score, failures);
    }

    public static int LevenshteinDistance(string expected, string actual)
    {
        ArgumentNullException.ThrowIfNull(expected);
        ArgumentNullException.ThrowIfNull(actual);

        if (expected.Length == 0)
        {
            return actual.Length;
        }

        if (actual.Length == 0)
        {
            return expected.Length;
        }

        var previous = Enumerable.Range(0, actual.Length + 1).ToArray();
        var current = new int[actual.Length + 1];

        for (var expectedIndex = 1; expectedIndex <= expected.Length; expectedIndex++)
        {
            current[0] = expectedIndex;
            for (var actualIndex = 1; actualIndex <= actual.Length; actualIndex++)
            {
                var substitution = previous[actualIndex - 1] +
                    (expected[expectedIndex - 1] == actual[actualIndex - 1] ? 0 : 1);
                current[actualIndex] = Math.Min(
                    Math.Min(previous[actualIndex] + 1, current[actualIndex - 1] + 1),
                    substitution);
            }

            (previous, current) = (current, previous);
        }

        return previous[actual.Length];
    }

    private static bool ExactMatch(TickRecognition tick)
        => string.Equals(tick.ExpectedText, tick.ActualText, StringComparison.Ordinal);

    private static void ValidateTextObservation(
        string observationId,
        string expectedText,
        HashSet<string> identifiers,
        string parameterName)
    {
        if (string.IsNullOrWhiteSpace(observationId))
        {
            throw new ArgumentException("An OCR observation ID is required.", parameterName);
        }

        ArgumentNullException.ThrowIfNull(expectedText);
        if (!identifiers.Add(observationId))
        {
            throw new ArgumentException($"OCR observation ID '{observationId}' is duplicated.", parameterName);
        }
    }

    private static MetricFailure Mismatch(
        MetricCaseIdentity identity,
        string metric,
        string observationId,
        string expected,
        string? actual)
        => new(
            identity.Module,
            identity.CaseId,
            metric,
            expected,
            actual ?? "missing",
            $"OCR observation '{observationId}' did not exactly match the expected text.");
}
