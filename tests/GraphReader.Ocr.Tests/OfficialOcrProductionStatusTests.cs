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
        StringAssert.Contains(promotion, "production_approval = true");
        StringAssert.Contains(promotion, "It must not emit release artifacts.");
        Assert.AreEqual(LicenseSha256, Sha256(licensePath));
        Assert.AreEqual(NoticeSha256, Sha256(noticePath));
    }

    [TestMethod]
    public void StructureConsensusCandidateIsPreregisteredAndSourceBoundBeforeInference()
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
            Assert.AreEqual(
                source.Value.GetString(),
                Sha256(sourcePath),
                $"Preregistered OCR source changed before the one-run gate: {source.Name}");
        }

        string readme = File.ReadAllText(Path.Combine(
            root,
            "ml",
            "ocr",
            "official_bakeoff",
            "README.md"));
        StringAssert.Contains(readme, "The evaluator and C# approval gate are now");
        StringAssert.Contains(readme, "frozen and checksum-bound");
        StringAssert.Contains(readme, "No fixtures have been generated");
        StringAssert.Contains(readme, "executed at this checkpoint");
        StringAssert.Contains(readme, "must not be rerun");
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
