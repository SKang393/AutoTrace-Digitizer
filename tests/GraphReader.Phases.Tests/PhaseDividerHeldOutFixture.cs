// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;

namespace GraphReader.Phases.Tests;

internal sealed record PhaseDividerHeldOutCase(
    string CaseId,
    double ExpectedX,
    double SessionPitchPixels,
    IReadOnlyList<PhaseDividerSegment> Segments);

internal static class PhaseDividerHeldOutFixture
{
    private const string FileName = "phase-divider-synthetic-heldout-v1.csv";

    public static IReadOnlyList<PhaseDividerHeldOutCase> Cases { get; } = Load();

    private static PhaseDividerHeldOutCase[] Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", FileName);
        HeldOutRow[] rows = File.ReadLines(path)
            .Where(static line => !string.IsNullOrWhiteSpace(line) && !line.StartsWith('#'))
            .Skip(1)
            .Select(Parse)
            .ToArray();

        return rows
            .GroupBy(static row => new { row.CaseId, row.ExpectedX, row.SessionPitchPixels })
            .OrderBy(static group => group.Key.CaseId, StringComparer.Ordinal)
            .Select(group => new PhaseDividerHeldOutCase(
                group.Key.CaseId,
                group.Key.ExpectedX,
                group.Key.SessionPitchPixels,
                group
                    .OrderBy(static row => row.SegmentId, StringComparer.Ordinal)
                    .Select(static row => new PhaseDividerSegment(
                        row.SegmentId,
                        PhaseTestFixture.PanelId,
                        new PhasePoint(row.X, row.Top),
                        new PhasePoint(row.X, row.Bottom),
                        1,
                        PhaseDividerStyle.Dashed,
                        0.96,
                        PhaseSegmentKind.Candidate))
                    .ToArray()))
            .ToArray();
    }

    private static HeldOutRow Parse(string line)
    {
        string[] fields = line.Split(',');
        if (fields.Length != 7)
        {
            throw new InvalidDataException($"Invalid phase divider fixture row: {line}");
        }

        return new HeldOutRow(
            fields[0],
            ParseDouble(fields[1]),
            ParseDouble(fields[2]),
            fields[3],
            ParseDouble(fields[4]),
            ParseDouble(fields[5]),
            ParseDouble(fields[6]));
    }

    private static double ParseDouble(string value) =>
        double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);

    private sealed record HeldOutRow(
        string CaseId,
        double ExpectedX,
        double SessionPitchPixels,
        string SegmentId,
        double X,
        double Top,
        double Bottom);
}
