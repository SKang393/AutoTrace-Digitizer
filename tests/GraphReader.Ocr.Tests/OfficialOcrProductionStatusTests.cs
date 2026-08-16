// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OfficialOcrProductionStatusTests
{
    private const string DetectionSha256 =
        "d4aa24d408cd70b8b9f66cc758e20f397fc31a9c69d8477cf8887fc53bd5fceb";
    private const string RecognitionSha256 =
        "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743";
    private const string LicenseSha256 =
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4";
    private const string NoticeSha256 =
        "8d81f5d0c58547cce471c24f82efe768a9d907d06764f67e90cc680c6d777729";
    private const string StructureConsensusExecutionSourceCommit =
        "7fa6abee5deaf7c17ad19169928290b96a65ce2a";
    private static readonly Dictionary<string, string> ReviewedSourcesChangedAfterConsumedRun =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["src/GraphReader.Ocr/LocalOnnxTextRegionDetector.cs"] =
                "c17cbd77bc646f7646f2f3f60b2120be735201b79c0b32a48318a30464b0aa38",
            ["src/GraphReader.App/Integration/Workflow/ProductionOcrAdapter.cs"] =
                "e57550f89e7eff4656ea6b9d74f0f2f0473da852471c8dcfa9647bb2e9f9e1fd",
        };

    [TestMethod]
    public void FailedOfficialCandidatesRemainUnmanifestedUntilEveryGatePasses()
    {
        string root = FindRepositoryRoot();
        string manifestDirectory = Path.Combine(root, "models", "manifest", "ocr");
        string[] manifests = Directory.GetFiles(manifestDirectory, "*.json", SearchOption.TopDirectoryOnly);

        Assert.AreEqual(
            0,
            manifests.Length,
            "A tracked OCR JSON manifest was added before the official pair completed its direct " +
            "public, sealed, notice, model-store, and packaging gates. Update this intentional " +
            "fail-closed lock only with the complete approval evidence.");

        string audit = File.ReadAllText(Path.Combine(
            manifestDirectory,
            "PP_OCRV5_OFFICIAL_ARCHIVE_AUDIT.md"));
        string promotion = File.ReadAllText(Path.Combine(
            manifestDirectory,
            "OFFICIAL_OCR_PRODUCTION_PROMOTION.md"));
        string licensePath = Path.Combine(
            root,
            "LICENSES",
            "PaddlePaddle-PP-OCRv5-Models-Apache-2.0.txt");
        string noticePath = Path.Combine(
            root,
            "LICENSES",
            "PaddlePaddle-PP-OCRv5-Models-Notice.txt");

        StringAssert.Contains(audit, DetectionSha256);
        StringAssert.Contains(audit, RecognitionSha256);
        StringAssert.Contains(audit, "benchmark ran and failed every");
        StringAssert.Contains(audit, "detection exact count `18/220`");
        StringAssert.Contains(audit, LicenseSha256);
        StringAssert.Contains(audit, NoticeSha256);
        StringAssert.Contains(promotion, "intentional fail-closed state");
        StringAssert.Contains(promotion, "status=fail");
        StringAssert.Contains(promotion, "cannot satisfy the strengthened gate");
        StringAssert.Contains(promotion, "probability_with_1e-5_clamp");
        StringAssert.Contains(promotion, "consumed 500-case attempt");
        StringAssert.Contains(promotion, "production_approval = true");
        StringAssert.Contains(promotion, "It must not emit release artifacts.");
        Assert.AreEqual(LicenseSha256, Sha256(licensePath));
        Assert.AreEqual(NoticeSha256, Sha256(noticePath));
    }

    [TestMethod]
    public void StructureConsensusRunRemainsSourceBoundAndRecordedAsFailedClosed()
    {
        string root = FindRepositoryRoot();
        string protocolPath = Path.Combine(
            root,
            "ml",
            "ocr",
            "official_bakeoff",
            "STRUCTURE_CONSENSUS_GATE_PROTOCOL.json");
        using System.Text.Json.JsonDocument document = System.Text.Json.JsonDocument.Parse(
            File.ReadAllText(protocolPath));
        System.Text.Json.JsonElement protocol = document.RootElement;

        Assert.AreEqual(
            "frozen_before_fixture_generation_and_inference",
            protocol.GetProperty("status").GetString());
        Assert.AreEqual(
            ProductionOcrProfile,
            protocol.GetProperty("profile").GetString());
        Assert.IsFalse(protocol.GetProperty("private_data").GetBoolean());
        Assert.IsFalse(protocol.GetProperty("chandler_used").GetBoolean());
        Assert.AreEqual(
            "graph-structure-consensus-v1",
            protocol.GetProperty("candidate").GetProperty("composition_id").GetString());
        Assert.AreEqual(
            "1fc3b2e72f89cbfb0d8854ec8701368e7ae764cbd5c6fef17b7e497d06ec9f09",
            protocol.GetProperty("prior_exposed_split_forbidden")
                .GetProperty("split_sha256")
                .GetString());
        Assert.AreEqual(
            1,
            protocol.GetProperty("experiment_budget")
                .GetProperty("official_composition_evaluations")
                .GetInt32());
        Assert.AreEqual(
            0,
            protocol.GetProperty("experiment_budget")
                .GetProperty("workflow_changes_after_inference")
                .GetInt32());

        foreach (System.Text.Json.JsonProperty source in
            protocol.GetProperty("reviewed_source_sha256").EnumerateObject())
        {
            string sourcePath = Path.Combine(root, source.Name.Replace('/', Path.DirectorySeparatorChar));
            Assert.IsTrue(File.Exists(sourcePath), $"Preregistered OCR source is missing: {source.Name}");
            string protocolSha256 = source.Value.GetString()!;
            if (ReviewedSourcesChangedAfterConsumedRun.TryGetValue(
                source.Name,
                out string? historicalSha256))
            {
                Assert.AreEqual(
                    historicalSha256,
                    protocolSha256,
                    $"The consumed gate's historical source hash changed: {source.Name}");
            }
            else
            {
                Assert.AreEqual(
                    protocolSha256,
                    Sha256(sourcePath),
                    $"Preregistered OCR source changed without an immutable post-run binding: {source.Name}");
            }
        }

        string readme = File.ReadAllText(Path.Combine(
            root,
            "ml",
            "ocr",
            "official_bakeoff",
            "README.md"));
        StringAssert.Contains(readme, "The authoritative 500-case split was frozen once before inference");
        StringAssert.Contains(readme, StructureConsensusExecutionSourceCommit);
        StringAssert.Contains(
            readme,
            "8685a3dfcb8212f612115c20d0f70437e0738fa1c4d86743cfd0e50bc5a41a8d");
        StringAssert.Contains(readme, "frozen and checksum-bound");
        StringAssert.Contains(readme, "The single authorized official composition execution then failed closed");
        StringAssert.Contains(readme, "BLOCKED: Detector output is not a probability tensor.");
        StringAssert.Contains(readme, "must not be rerun, repaired, or tuned");
        StringAssert.Contains(readme, "probability_with_1e-5_clamp");
        StringAssert.Contains(readme, "disjoint split, a new frozen protocol");
        StringAssert.Contains(readme, "release authorization therefore remain false");

        string evaluationRequirements = File.ReadAllText(Path.Combine(
            root,
            "ml",
            "ocr",
            "official_bakeoff",
            "requirements-structure-consensus.txt"));
        StringAssert.Contains(evaluationRequirements, "-r requirements-conversion.txt");
        StringAssert.Contains(evaluationRequirements, "opencv-python-headless==4.10.0.84");
        StringAssert.Contains(
            evaluationRequirements,
            "afcf28bd1209dd58810d33defb622b325d3cbe49dcd7a43a902982c33e5fad05");
    }

    private const string ProductionOcrProfile =
        "graphreader-ocr-structure-consensus-public-gate-v1";

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "GraphAutoReader.slnx")) &&
                File.Exists(Path.Combine(directory.FullName, "contracts", "model-manifest.schema.json")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the Graph Auto Reader repository root.");
    }

    private static string Sha256(string path) =>
        Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(path)));
}
