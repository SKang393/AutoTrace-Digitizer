// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Validation.Scoreboard;
using Microsoft.VisualStudio.TestTools.UnitTesting;

#pragma warning disable CA1861 // Small collection expressions are intentional test fixtures.

namespace GraphReader.Integration.Tests.Validation.Tests;

[TestClass]
public sealed class ValidationScoreboardGateTests
{
    [TestMethod]
    public void ValidationArticleSplitDetectsArticleLevelLeakage()
    {
        ArticleSplitValidation result = ArticleSplitValidator.Validate(
        [
            new ArticleSplitRecord("public", "article-1", "case-train", DataSplit.Train),
            new ArticleSplitRecord("public", "article-1", "case-test", DataSplit.Test),
            new ArticleSplitRecord("public", "article-2", "case-validation", DataSplit.Validation),
        ]);

        Assert.IsFalse(result.IsValid);
        Assert.HasCount(1, result.Leaks);
        Assert.AreEqual("article-1", result.Leaks[0].ArticleId);
        CollectionAssert.AreEqual(
            new[] { DataSplit.Train, DataSplit.Test },
            result.Leaks[0].Splits.ToArray());
        CollectionAssert.AreEqual(
            new[] { "case-test", "case-train" },
            result.Leaks[0].CaseIds.ToArray());
    }

    [TestMethod]
    public void ValidationRegressionGateReturnsZeroForPassingThreshold()
    {
        GateOutcome outcome = RegressionGate.Evaluate(
            CreateInput(new CaseMetricRecord("markers", "case-pass", "quality", 0.9, 1)),
            [new RegressionThreshold("quality", Minimum: 0.8, Maximum: null)]);

        Assert.IsTrue(outcome.Passed);
        Assert.AreEqual(0, outcome.ExitCode);
        Assert.IsEmpty(outcome.Failures);
    }

    [TestMethod]
    public void ValidationRegressionGateReturnsNonzeroAndCaseIdentityForFailure()
    {
        GateOutcome outcome = RegressionGate.Evaluate(
            CreateInput(new CaseMetricRecord("markers", "case-fail", "quality", 0.7, 1)),
            [new RegressionThreshold("quality", Minimum: 0.8, Maximum: null)]);

        Assert.IsFalse(outcome.Passed);
        Assert.AreNotEqual(0, outcome.ExitCode);
        Assert.HasCount(1, outcome.Failures);
        Assert.AreEqual("markers", outcome.Failures[0].ModuleId);
        Assert.AreEqual("case-fail", outcome.Failures[0].CaseId);
        Assert.AreEqual("quality", outcome.Failures[0].CriterionId);
    }

    [TestMethod]
    public void ValidationRegressionGateRejectsRequiredMetricWithZeroSupport()
    {
        GateOutcome outcome = RegressionGate.Evaluate(
            CreateInput(new CaseMetricRecord("markers", "case-empty", "quality", 1.0, 0)),
            [new RegressionThreshold("quality", Minimum: 0.8, Maximum: null)]);

        Assert.IsFalse(outcome.Passed);
        GateFailure failure = AssertSingleFailure(outcome);
        Assert.AreEqual("case-empty", failure.CaseId);
        StringAssert.Contains(failure.Message, "zero sample support");
    }

    [TestMethod]
    public void ValidationRegressionGateReportsArticleLeakageWithCaseIdentity()
    {
        ScoreboardInput input = CreateInput(
            new CaseMetricRecord("markers", "case-train", "quality", 0.9, 1)) with
        {
            Metrics =
            [
                new CaseMetricRecord("markers", "case-train", "quality", 0.9, 1),
                new CaseMetricRecord("markers", "case-test", "quality", 0.9, 1),
            ],
            ArticleSplits =
            [
                new ArticleSplitRecord("public", "article-1", "case-train", DataSplit.Train),
                new ArticleSplitRecord("public", "article-1", "case-test", DataSplit.Test),
            ],
        };

        GateOutcome outcome = RegressionGate.Evaluate(
            input,
            [new RegressionThreshold("quality", Minimum: 0.8, Maximum: null)]);

        Assert.IsFalse(outcome.Passed);
        GateFailure failure = AssertSingleFailure(outcome);
        Assert.AreEqual("data_split", failure.ModuleId);
        Assert.AreEqual("case-test,case-train", failure.CaseId);
        Assert.AreEqual("article_split_leakage", failure.CriterionId);
    }

    [TestMethod]
    public void ValidationRegressionGateRejectsMissingScoredCaseSplitMetadata()
    {
        ScoreboardInput input = CreateInput(
            new CaseMetricRecord("markers", "case-scored", "quality", 0.9, 1)) with
        {
            ArticleSplits = Array.Empty<ArticleSplitRecord>(),
        };

        GateOutcome outcome = RegressionGate.Evaluate(
            input,
            [new RegressionThreshold("quality", Minimum: 0.8, Maximum: null)]);

        Assert.IsFalse(outcome.Passed);
        GateFailure failure = AssertSingleFailure(outcome);
        Assert.AreEqual("data_split", failure.ModuleId);
        Assert.AreEqual("case-scored", failure.CaseId);
        Assert.AreEqual("article_split_metadata", failure.CriterionId);
    }

    [TestMethod]
    public void ValidationRegressionGateLinksConfidenceAndPerformanceCasesToSplitMetadata()
    {
        ScoreboardInput input = CreateInput(
            new CaseMetricRecord("markers", "case-scored", "quality", 0.9, 1)) with
        {
            ConfidenceObservations =
            [
                new GraphReader.Validation.Scoreboard.ConfidenceObservation(
                    "markers",
                    "case-confidence",
                    0.9,
                    IsCorrect: true),
            ],
            Performance =
            [
                new PerformanceRecord("validation_harness", "case-performance", 10, 5, 1024),
            ],
        };

        GateOutcome outcome = RegressionGate.Evaluate(
            input,
            [new RegressionThreshold("quality", Minimum: 0.8, Maximum: null)]);

        Assert.IsFalse(outcome.Passed);
        string[] missingCaseIds = outcome.Failures
            .Where(failure => failure.CriterionId == "article_split_metadata")
            .Select(failure => failure.CaseId)
            .Order(StringComparer.Ordinal)
            .ToArray();
        CollectionAssert.AreEqual(
            new[] { "case-confidence", "case-performance", "public" },
            missingCaseIds);
    }

    [TestMethod]
    public void ValidationLicenseManifestFailureBlocksReleaseWithManifestIdentity()
    {
        ScoreboardInput input = CreateInput(
            new CaseMetricRecord("markers", "case-pass", "quality", 0.9, 1)) with
        {
            LicenseManifests =
            [
                new LicenseManifestValidation(
                    "models/manifest/prohibited.json",
                    "prohibited-model",
                    IsValid: false,
                    ["license.spdx 'GPL-3.0-only' is prohibited for distribution."]),
            ],
        };

        GateOutcome outcome = RegressionGate.Evaluate(
            input,
            [new RegressionThreshold("quality", Minimum: 0.8, Maximum: null)]);

        Assert.IsFalse(outcome.Passed);
        GateFailure failure = AssertSingleFailure(outcome);
        Assert.AreEqual("release", failure.ModuleId);
        Assert.AreEqual("models/manifest/prohibited.json", failure.CaseId);
        Assert.AreEqual("license_manifest_validation", failure.CriterionId);
    }

    [TestMethod]
    public void ValidationLicenseManifestValidatorRejectsProhibitedOrUnreviewedLicense()
    {
        string root = CreateTemporaryRepository();
        try
        {
            string manifestPath = Path.Combine(root, "models", "manifest", "bad.json");
            File.WriteAllText(
                manifestPath,
                """
                {
                  "model_id": "bad-model",
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "commercial_use": true,
                  "redistribution": true,
                  "files": [],
                  "license": {
                    "spdx": "GPL-3.0-only",
                    "notice_path": "NOTICE",
                    "reviewed": false
                  }
                }
                """);

            IReadOnlyList<LicenseManifestValidation> results =
                LicenseManifestValidator.ValidateRepository(root);

            Assert.HasCount(1, results);
            Assert.IsFalse(results[0].IsValid);
            Assert.AreEqual("models/manifest/bad.json", results[0].ManifestPath);
            Assert.IsTrue(results[0].Issues.Any(issue => issue.Contains("prohibited", StringComparison.Ordinal)));
            Assert.IsTrue(results[0].Issues.Any(issue => issue.Contains("reviewed", StringComparison.Ordinal)));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static ScoreboardInput CreateInput(CaseMetricRecord metric) =>
        new(
            "public",
            "1",
            [metric],
            [new ArticleSplitRecord("public", "article-1", metric.CaseId, DataSplit.Test)],
            Array.Empty<GraphReader.Validation.Scoreboard.ConfidenceObservation>(),
            Array.Empty<PerformanceRecord>(),
            [
                new LicenseManifestValidation(
                    "models/manifest/valid.json",
                    "valid-model",
                    IsValid: true,
                    Array.Empty<string>()),
            ]);

    private static GateFailure AssertSingleFailure(GateOutcome outcome)
    {
        Assert.HasCount(1, outcome.Failures);
        return outcome.Failures[0];
    }

    private static string CreateTemporaryRepository()
    {
        string root = Directory.CreateDirectory(Path.Combine(
            Path.GetTempPath(),
            $"GraphReader-License-Validation-{Guid.NewGuid():N}")).FullName;
        Directory.CreateDirectory(Path.Combine(root, "models", "manifest"));
        File.WriteAllText(Path.Combine(root, "NOTICE"), "Synthetic test notice.");
        return root;
    }
}
