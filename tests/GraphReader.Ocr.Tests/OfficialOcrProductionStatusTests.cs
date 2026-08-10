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
