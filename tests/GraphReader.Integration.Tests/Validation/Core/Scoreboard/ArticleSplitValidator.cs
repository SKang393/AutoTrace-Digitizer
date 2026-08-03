// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Validation.Scoreboard;

public static class ArticleSplitValidator
{
    public static ArticleSplitValidation Validate(
        IEnumerable<ArticleSplitRecord> records,
        IEnumerable<string>? scoredCaseIds = null)
    {
        ArgumentNullException.ThrowIfNull(records);
        ArticleSplitRecord[] materialized = records.ToArray();

        ArticleSplitLeak[] leaks = materialized
            .GroupBy(
                record => (record.DatasetId, record.ArticleId),
                StringTupleComparer.Ordinal)
            .Select(group => new ArticleSplitLeak(
                group.Key.DatasetId,
                group.Key.ArticleId,
                group.Select(record => record.Split)
                    .Distinct()
                    .Order()
                    .ToArray(),
                group.Select(record => record.CaseId)
                    .Distinct(StringComparer.Ordinal)
                    .Order(StringComparer.Ordinal)
                    .ToArray()))
            .Where(leak => leak.Splits.Count > 1)
            .OrderBy(leak => leak.DatasetId, StringComparer.Ordinal)
            .ThenBy(leak => leak.ArticleId, StringComparer.Ordinal)
            .ToArray();

        List<ArticleSplitMetadataIssue> issues = [];
        foreach (ArticleSplitRecord record in materialized)
        {
            if (string.IsNullOrWhiteSpace(record.DatasetId) ||
                string.IsNullOrWhiteSpace(record.ArticleId) ||
                string.IsNullOrWhiteSpace(record.CaseId))
            {
                issues.Add(new ArticleSplitMetadataIssue(
                    record.CaseId ?? string.Empty,
                    "Dataset, article, and case identifiers must all be nonempty."));
            }
        }

        foreach (IGrouping<string, ArticleSplitRecord> group in materialized
                     .Where(record => !string.IsNullOrWhiteSpace(record.CaseId))
                     .GroupBy(record => record.CaseId, StringComparer.Ordinal)
                     .Where(group => group.Count() != 1))
        {
            issues.Add(new ArticleSplitMetadataIssue(
                group.Key,
                "Each scored case must map to exactly one article and split record."));
        }

        if (scoredCaseIds is not null)
        {
            HashSet<string> scored = scoredCaseIds
                .Where(caseId => !string.IsNullOrWhiteSpace(caseId))
                .ToHashSet(StringComparer.Ordinal);
            HashSet<string> recorded = materialized
                .Where(record => !string.IsNullOrWhiteSpace(record.CaseId))
                .Select(record => record.CaseId)
                .ToHashSet(StringComparer.Ordinal);
            foreach (string caseId in scored.Except(recorded, StringComparer.Ordinal).Order(StringComparer.Ordinal))
            {
                issues.Add(new ArticleSplitMetadataIssue(
                    caseId,
                    "Scored case is missing article-level split metadata."));
            }

            foreach (string caseId in recorded.Except(scored, StringComparer.Ordinal).Order(StringComparer.Ordinal))
            {
                issues.Add(new ArticleSplitMetadataIssue(
                    caseId,
                    "Article-level split metadata is not linked to a scored case."));
            }
        }

        ArticleSplitMetadataIssue[] orderedIssues = issues
            .OrderBy(issue => issue.CaseId, StringComparer.Ordinal)
            .ThenBy(issue => issue.Message, StringComparer.Ordinal)
            .ToArray();
        return new ArticleSplitValidation(
            leaks.Length == 0 && orderedIssues.Length == 0,
            leaks,
            orderedIssues);
    }

    public static IReadOnlyList<ArticleSplitLeak> FindLeakage(
        IEnumerable<ArticleSplitRecord> records) => Validate(records).Leaks;

    private sealed class StringTupleComparer : IEqualityComparer<(string DatasetId, string ArticleId)>
    {
        public static StringTupleComparer Ordinal { get; } = new();

        public bool Equals(
            (string DatasetId, string ArticleId) x,
            (string DatasetId, string ArticleId) y) =>
            string.Equals(x.DatasetId, y.DatasetId, StringComparison.Ordinal) &&
            string.Equals(x.ArticleId, y.ArticleId, StringComparison.Ordinal);

        public int GetHashCode((string DatasetId, string ArticleId) obj) =>
            HashCode.Combine(
                StringComparer.Ordinal.GetHashCode(obj.DatasetId),
                StringComparer.Ordinal.GetHashCode(obj.ArticleId));
    }
}
