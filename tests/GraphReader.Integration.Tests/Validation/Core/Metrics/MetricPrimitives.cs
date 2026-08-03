// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests.Validation.Core.Metrics;

public readonly record struct MetricPoint(double X, double Y)
{
    public double DistanceTo(MetricPoint other)
    {
        var deltaX = X - other.X;
        var deltaY = Y - other.Y;
        return Math.Sqrt((deltaX * deltaX) + (deltaY * deltaY));
    }
}

public sealed record MetricCaseIdentity(string Module, string CaseId);

public sealed record MetricFailure(
    string Module,
    string CaseId,
    string Metric,
    string Expected,
    string Actual,
    string Detail);

public sealed record MetricOutcome<T>(T Value, IReadOnlyList<MetricFailure> Failures);

internal static class MetricGuard
{
    public static void Identity(MetricCaseIdentity identity)
    {
        ArgumentNullException.ThrowIfNull(identity);

        if (string.IsNullOrWhiteSpace(identity.Module))
        {
            throw new ArgumentException("A module name is required.", nameof(identity));
        }

        if (string.IsNullOrWhiteSpace(identity.CaseId))
        {
            throw new ArgumentException("A case ID is required.", nameof(identity));
        }
    }

    public static void Finite(double value, string parameterName)
    {
        if (!double.IsFinite(value))
        {
            throw new ArgumentOutOfRangeException(parameterName, "The value must be finite.");
        }
    }

    public static void NonNegativeFinite(double value, string parameterName)
    {
        Finite(value, parameterName);
        if (value < 0)
        {
            throw new ArgumentOutOfRangeException(parameterName, "The value must be non-negative.");
        }
    }

    public static void Probability(double value, string parameterName)
    {
        Finite(value, parameterName);
        if (value is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(parameterName, "The value must be between zero and one.");
        }
    }
}

internal static class MetricMath
{
    public static double Ratio(int numerator, int denominator, double whenEmpty = 0)
        => denominator == 0 ? whenEmpty : (double)numerator / denominator;

    public static double F1(double precision, double recall)
        => precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);

    public static string Invariant(double value)
        => value.ToString("G17", System.Globalization.CultureInfo.InvariantCulture);
}
