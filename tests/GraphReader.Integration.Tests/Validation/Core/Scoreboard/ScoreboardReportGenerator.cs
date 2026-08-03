// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace GraphReader.Validation.Scoreboard;

public static class ScoreboardReportGenerator
{
    private static readonly JsonSerializerOptions JsonOptions = CreateJsonOptions();

    public static ScoreboardReport Generate(ScoreboardResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        return new ScoreboardReport(
            NormalizeNewlines(JsonSerializer.Serialize(result, JsonOptions)) + "\n",
            BuildMarkdown(result),
            BuildHtml(result));
    }

    private static string BuildMarkdown(ScoreboardResult result)
    {
        StringBuilder builder = new();
        builder.Append("# Graph Auto Reader validation scoreboard\n\n")
            .Append("- Suite: `").Append(EscapeMarkdown(result.SuiteName)).Append("`\n")
            .Append("- Suite version: `").Append(EscapeMarkdown(result.SuiteVersion)).Append("`\n")
            .Append("- Evidence scope: ").Append(EscapeMarkdown(result.EvidenceScope)).Append('\n')
            .Append("- Gate: **").Append(result.Gate.Passed ? "PASS" : "FAIL").Append("**\n")
            .Append("- Exit code: `").Append(result.Gate.ExitCode).Append("`\n\n")
            .Append("## Metrics\n\n")
            .Append("| Metric | Direction | Unit | Aggregate | Samples | Threshold |\n")
            .Append("|---|---:|---:|---:|---:|---:|\n");

        foreach (MetricSummary metric in result.Metrics)
        {
            RegressionThreshold? threshold = result.Thresholds.FirstOrDefault(candidate =>
                string.Equals(candidate.MetricId, metric.Definition.Id, StringComparison.Ordinal));
            builder.Append("| ").Append(EscapeMarkdown(metric.Definition.Id))
                .Append(" | ").Append(metric.Definition.Direction)
                .Append(" | ").Append(EscapeMarkdown(metric.Definition.Unit))
                .Append(" | ").Append(Format(metric.AggregateValue))
                .Append(" | ").Append(metric.SampleCount)
                .Append(" | ").Append(FormatThreshold(threshold))
                .Append(" |\n");
        }

        builder.Append("\n## Case and module failures\n\n");
        if (result.Gate.Failures.Count == 0)
        {
            builder.Append("None.\n");
        }
        else
        {
            builder.Append("| Module | Case | Criterion | Actual | Message |\n")
                .Append("|---|---|---|---:|---|\n");
            foreach (GateFailure failure in result.Gate.Failures)
            {
                builder.Append("| ").Append(EscapeMarkdown(failure.ModuleId))
                    .Append(" | ").Append(EscapeMarkdown(failure.CaseId))
                    .Append(" | ").Append(EscapeMarkdown(failure.CriterionId))
                    .Append(" | ").Append(Format(failure.Actual))
                    .Append(" | ").Append(EscapeMarkdown(failure.Message))
                    .Append(" |\n");
            }
        }

        builder.Append("\n## Article split validation\n\n")
            .Append(result.ArticleSplits.IsValid
                ? "No article leakage detected.\n"
                : $"Detected {result.ArticleSplits.Leaks.Count} article split leak(s).\n")
            .Append("\n| Dataset | Article | Case | Split |\n")
            .Append("|---|---|---|---|\n");
        foreach (ArticleSplitRecord split in result.ArticleSplitMetadata)
        {
            builder.Append("| ").Append(EscapeMarkdown(split.DatasetId))
                .Append(" | ").Append(EscapeMarkdown(split.ArticleId))
                .Append(" | ").Append(EscapeMarkdown(split.CaseId))
                .Append(" | ").Append(split.Split)
                .Append(" |\n");
        }

        builder
            .Append("\n## Confidence calibration\n\n")
            .Append("- Observations: ").Append(result.ConfidenceCalibration.ObservationCount).Append('\n')
            .Append("- Expected calibration error: ")
            .Append(Format(result.ConfidenceCalibration.ExpectedCalibrationError)).Append('\n')
            .Append("- Brier score: ").Append(Format(result.ConfidenceCalibration.BrierScore)).Append("\n\n")
            .Append("| Bin | Range | Count | Mean confidence | Accuracy | Gap |\n")
            .Append("|---:|---:|---:|---:|---:|---:|\n");
        foreach (CalibrationBin bin in result.ConfidenceCalibration.Bins)
        {
            builder.Append("| ").Append(bin.Index)
                .Append(" | [").Append(Format(bin.LowerBound)).Append(", ")
                .Append(Format(bin.UpperBound)).Append(bin.Index == result.ConfidenceCalibration.Bins.Count - 1 ? "]" : ")")
                .Append(" | ").Append(bin.Count)
                .Append(" | ").Append(Format(bin.MeanConfidence))
                .Append(" | ").Append(Format(bin.Accuracy))
                .Append(" | ").Append(Format(bin.AbsoluteGap))
                .Append(" |\n");
        }

        builder.Append("\n## Timing and memory\n\n")
            .Append("| Module | Case | Cold ms | Warm ms | Peak bytes | Machine |\n")
            .Append("|---|---|---:|---:|---:|---|\n");
        if (result.Performance.Count == 0)
        {
            builder.Append("| unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |\n");
        }
        else
        {
            foreach (PerformanceRecord performance in result.Performance)
            {
                builder.Append("| ").Append(EscapeMarkdown(performance.ModuleId))
                    .Append(" | ").Append(EscapeMarkdown(performance.CaseId))
                    .Append(" | ").Append(Format(performance.ColdRuntimeMilliseconds))
                    .Append(" | ").Append(Format(performance.WarmRuntimeMilliseconds))
                    .Append(" | ").Append(Format(performance.PeakMemoryBytes))
                    .Append(" | ").Append(EscapeMarkdown(performance.Machine ?? "unmeasured"))
                    .Append(" |\n");
            }
        }

        builder.Append("\n## License manifest validation\n\n")
            .Append("| Manifest | Component | Status | Issues |\n")
            .Append("|---|---|---:|---|\n");
        foreach (LicenseManifestValidation manifest in result.LicenseManifests)
        {
            builder.Append("| ").Append(EscapeMarkdown(manifest.ManifestPath))
                .Append(" | ").Append(EscapeMarkdown(manifest.ComponentId))
                .Append(" | ").Append(manifest.IsValid ? "PASS" : "FAIL")
                .Append(" | ").Append(EscapeMarkdown(
                    manifest.Issues.Count == 0 ? "None" : string.Join("; ", manifest.Issues)))
                .Append(" |\n");
        }

        return NormalizeNewlines(builder.ToString());
    }

    private static string BuildHtml(ScoreboardResult result)
    {
        StringBuilder builder = new();
        builder.Append("<!doctype html>\n<html lang=\"en\">\n<head>\n")
            .Append("<meta charset=\"utf-8\">\n")
            .Append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
            .Append("<title>Graph Auto Reader validation scoreboard</title>\n")
            .Append("<style>body{font-family:Segoe UI,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222}")
            .Append("table{border-collapse:collapse;width:100%;margin-bottom:2rem}th,td{border:1px solid #bbb;padding:.4rem;text-align:left}")
            .Append("th{background:#eee}.number{text-align:right}.pass{color:#176b2c}.fail{color:#a51d1d}.muted{color:#666}</style>\n")
            .Append("</head>\n<body>\n<h1>Graph Auto Reader validation scoreboard</h1>\n")
            .Append("<dl><dt>Suite</dt><dd><code>").Append(Html(result.SuiteName)).Append("</code></dd>")
            .Append("<dt>Suite version</dt><dd><code>").Append(Html(result.SuiteVersion)).Append("</code></dd>")
            .Append("<dt>Evidence scope</dt><dd>").Append(Html(result.EvidenceScope)).Append("</dd>")
            .Append("<dt>Gate</dt><dd class=\"").Append(result.Gate.Passed ? "pass\">PASS" : "fail\">FAIL")
            .Append("</dd><dt>Exit code</dt><dd><code>").Append(result.Gate.ExitCode).Append("</code></dd></dl>\n")
            .Append("<h2>Metrics</h2>\n<table><thead><tr><th>Metric</th><th>Direction</th><th>Unit</th>")
            .Append("<th>Aggregate</th><th>Samples</th><th>Threshold</th></tr></thead><tbody>\n");

        foreach (MetricSummary metric in result.Metrics)
        {
            RegressionThreshold? threshold = result.Thresholds.FirstOrDefault(candidate =>
                string.Equals(candidate.MetricId, metric.Definition.Id, StringComparison.Ordinal));
            builder.Append("<tr><td><code>").Append(Html(metric.Definition.Id)).Append("</code></td><td>")
                .Append(Html(metric.Definition.Direction.ToString())).Append("</td><td>")
                .Append(Html(metric.Definition.Unit)).Append("</td><td class=\"number\">")
                .Append(Html(Format(metric.AggregateValue))).Append("</td><td class=\"number\">")
                .Append(metric.SampleCount).Append("</td><td class=\"number\">")
                .Append(Html(FormatThreshold(threshold))).Append("</td></tr>\n");
        }

        builder.Append("</tbody></table>\n<h2>Case and module failures</h2>\n");
        if (result.Gate.Failures.Count == 0)
        {
            builder.Append("<p>None.</p>\n");
        }
        else
        {
            builder.Append("<table><thead><tr><th>Module</th><th>Case</th><th>Criterion</th><th>Actual</th><th>Message</th></tr></thead><tbody>\n");
            foreach (GateFailure failure in result.Gate.Failures)
            {
                builder.Append("<tr><td>").Append(Html(failure.ModuleId)).Append("</td><td>")
                    .Append(Html(failure.CaseId)).Append("</td><td><code>")
                    .Append(Html(failure.CriterionId)).Append("</code></td><td class=\"number\">")
                    .Append(Html(Format(failure.Actual))).Append("</td><td>")
                    .Append(Html(failure.Message)).Append("</td></tr>\n");
            }

            builder.Append("</tbody></table>\n");
        }

        AppendHtmlArticleSplits(builder, result.ArticleSplitMetadata, result.ArticleSplits);
        AppendHtmlCalibration(builder, result.ConfidenceCalibration);
        AppendHtmlPerformance(builder, result.Performance);
        AppendHtmlLicenses(builder, result.LicenseManifests);
        builder.Append("</body>\n</html>\n");
        return NormalizeNewlines(builder.ToString());
    }

    private static void AppendHtmlArticleSplits(
        StringBuilder builder,
        IReadOnlyList<ArticleSplitRecord> metadata,
        ArticleSplitValidation validation)
    {
        builder.Append("<h2>Article split validation</h2>\n<p class=\"")
            .Append(validation.IsValid ? "pass\">No article leakage detected." : "fail\">Article leakage detected.")
            .Append("</p>\n<table><thead><tr><th>Dataset</th><th>Article</th><th>Case</th><th>Split</th></tr></thead><tbody>\n");
        foreach (ArticleSplitRecord split in metadata)
        {
            builder.Append("<tr><td>").Append(Html(split.DatasetId)).Append("</td><td>")
                .Append(Html(split.ArticleId)).Append("</td><td>")
                .Append(Html(split.CaseId)).Append("</td><td>")
                .Append(Html(split.Split.ToString())).Append("</td></tr>\n");
        }

        builder.Append("</tbody></table>\n");
    }

    private static void AppendHtmlCalibration(
        StringBuilder builder,
        ConfidenceCalibrationReport calibration)
    {
        builder.Append("<h2>Confidence calibration</h2>\n<ul><li>Observations: ")
            .Append(calibration.ObservationCount).Append("</li><li>Expected calibration error: ")
            .Append(Html(Format(calibration.ExpectedCalibrationError))).Append("</li><li>Brier score: ")
            .Append(Html(Format(calibration.BrierScore))).Append("</li></ul>\n")
            .Append("<table><thead><tr><th>Bin</th><th>Range</th><th>Count</th><th>Mean confidence</th><th>Accuracy</th><th>Gap</th></tr></thead><tbody>\n");
        foreach (CalibrationBin bin in calibration.Bins)
        {
            builder.Append("<tr><td class=\"number\">").Append(bin.Index).Append("</td><td class=\"number\">")
                .Append(Html($"[{Format(bin.LowerBound)}, {Format(bin.UpperBound)}{(bin.Index == calibration.Bins.Count - 1 ? "]" : ")")}"))
                .Append("</td><td class=\"number\">").Append(bin.Count)
                .Append("</td><td class=\"number\">").Append(Html(Format(bin.MeanConfidence)))
                .Append("</td><td class=\"number\">").Append(Html(Format(bin.Accuracy)))
                .Append("</td><td class=\"number\">").Append(Html(Format(bin.AbsoluteGap)))
                .Append("</td></tr>\n");
        }

        builder.Append("</tbody></table>\n");
    }

    private static void AppendHtmlPerformance(
        StringBuilder builder,
        IReadOnlyList<PerformanceRecord> performance)
    {
        builder.Append("<h2>Timing and memory</h2>\n<table><thead><tr><th>Module</th><th>Case</th><th>Cold ms</th><th>Warm ms</th><th>Peak bytes</th><th>Machine</th></tr></thead><tbody>\n");
        if (performance.Count == 0)
        {
            builder.Append("<tr class=\"muted\"><td colspan=\"6\">unmeasured</td></tr>\n");
        }
        else
        {
            foreach (PerformanceRecord record in performance)
            {
                builder.Append("<tr><td>").Append(Html(record.ModuleId)).Append("</td><td>")
                    .Append(Html(record.CaseId)).Append("</td><td class=\"number\">")
                    .Append(Html(Format(record.ColdRuntimeMilliseconds))).Append("</td><td class=\"number\">")
                    .Append(Html(Format(record.WarmRuntimeMilliseconds))).Append("</td><td class=\"number\">")
                    .Append(Html(Format(record.PeakMemoryBytes))).Append("</td><td>")
                    .Append(Html(record.Machine ?? "unmeasured")).Append("</td></tr>\n");
            }
        }

        builder.Append("</tbody></table>\n");
    }

    private static void AppendHtmlLicenses(
        StringBuilder builder,
        IReadOnlyList<LicenseManifestValidation> manifests)
    {
        builder.Append("<h2>License manifest validation</h2>\n<table><thead><tr><th>Manifest</th><th>Component</th><th>Status</th><th>Issues</th></tr></thead><tbody>\n");
        foreach (LicenseManifestValidation manifest in manifests)
        {
            builder.Append("<tr><td><code>").Append(Html(manifest.ManifestPath)).Append("</code></td><td>")
                .Append(Html(manifest.ComponentId)).Append("</td><td class=\"")
                .Append(manifest.IsValid ? "pass\">PASS" : "fail\">FAIL")
                .Append("</td><td>").Append(Html(
                    manifest.Issues.Count == 0 ? "None" : string.Join("; ", manifest.Issues)))
                .Append("</td></tr>\n");
        }

        builder.Append("</tbody></table>\n");
    }

    private static string FormatThreshold(RegressionThreshold? threshold)
    {
        if (threshold is null)
        {
            return "none";
        }

        if (threshold.Minimum is not null && threshold.Maximum is not null)
        {
            return $"[{Format(threshold.Minimum)}, {Format(threshold.Maximum)}]";
        }

        return threshold.Minimum is not null
            ? $">= {Format(threshold.Minimum)}"
            : $"<= {Format(threshold.Maximum)}";
    }

    private static string Format(double? value) => value is null
        ? "unmeasured"
        : value.Value.ToString("0.######", CultureInfo.InvariantCulture);

    private static string Format(long? value) => value?.ToString(CultureInfo.InvariantCulture) ?? "unmeasured";

    private static string EscapeMarkdown(string value) =>
        value.Replace("|", "\\|", StringComparison.Ordinal)
            .Replace("\r", " ", StringComparison.Ordinal)
            .Replace("\n", " ", StringComparison.Ordinal);

    private static string Html(string value) => WebUtility.HtmlEncode(value);

    private static string NormalizeNewlines(string value) =>
        value.Replace("\r\n", "\n", StringComparison.Ordinal)
            .Replace("\r", "\n", StringComparison.Ordinal);

    private static JsonSerializerOptions CreateJsonOptions()
    {
        JsonSerializerOptions options = new()
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        };
        options.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower));
        return options;
    }
}
