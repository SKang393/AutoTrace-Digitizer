// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Ocr.Tests;

internal static class NumericCorpus
{
    public const double AgreedExactMatchThreshold = 0.95d;

    public static IReadOnlyList<NumericCorpusCase> Cases { get; } =
    [
        new("0", 0d),
        new("O", 0d),
        new("10O", 100d),
        new("1", 1d),
        new("l", 1d),
        new("l0", 10d),
        new("5", 5d),
        new("12", 12d),
        new("100", 100d),
        new("-5", -5d),
        new("\u22125", -5d),
        new(".5", 0.5d),
        new("0.5", 0.5d),
        new("12.5", 12.5d),
        new("50%", 50d, IsPercent: true),
        new("25.0%", 25d, IsPercent: true),
        new(" 20 ", 20d),
        new("1,000", 1000d),
        new("-0.25", -0.25d),
        new("80.0", 80d),
        new("99%", 99d, IsPercent: true),
        new("001", 1d),
    ];
}

internal sealed record NumericCorpusCase(string Text, double ExpectedValue, bool IsPercent = false);
