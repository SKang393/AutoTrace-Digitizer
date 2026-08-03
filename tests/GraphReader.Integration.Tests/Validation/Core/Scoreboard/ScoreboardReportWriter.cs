// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text;

namespace GraphReader.Validation.Scoreboard;

public static class ScoreboardReportWriter
{
    private static readonly UTF8Encoding Utf8WithoutByteOrderMark = new(false);

    public static ScoreboardReportPaths WriteAll(
        string outputDirectory,
        ScoreboardResult result,
        string baseName = "scoreboard") =>
        WriteAll(outputDirectory, ScoreboardReportGenerator.Generate(result), baseName);

    public static ScoreboardReportPaths WriteAll(
        string outputDirectory,
        ScoreboardReport report,
        string baseName = "scoreboard")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);
        ArgumentNullException.ThrowIfNull(report);
        ArgumentException.ThrowIfNullOrWhiteSpace(baseName);
        if (!string.Equals(Path.GetFileName(baseName), baseName, StringComparison.Ordinal) ||
            baseName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        {
            throw new ArgumentException("Report base name must be a single valid file name.", nameof(baseName));
        }

        string fullDirectory = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(fullDirectory);
        string jsonPath = Path.Combine(fullDirectory, baseName + ".json");
        string markdownPath = Path.Combine(fullDirectory, baseName + ".md");
        string htmlPath = Path.Combine(fullDirectory, baseName + ".html");
        File.WriteAllText(jsonPath, report.Json, Utf8WithoutByteOrderMark);
        File.WriteAllText(markdownPath, report.Markdown, Utf8WithoutByteOrderMark);
        File.WriteAllText(htmlPath, report.Html, Utf8WithoutByteOrderMark);
        return new ScoreboardReportPaths(jsonPath, markdownPath, htmlPath);
    }
}
