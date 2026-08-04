// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text.Json;

namespace GraphReader.Pdf.Tests;

[TestClass]
public sealed class InternalPortableStagingPolicyTests
{
    private const string ExpectedPolicySha256 =
        "8bea103823fbef6f91f12439f731cb78e98081e74f0f44b9d895e086eeb197a1";

    [TestMethod]
    public void TrackedPolicyPinsExactInternalRunnerAndKeepsAllApprovalsFalse()
    {
        string policyPath = FindRepositoryFile(
            "packaging",
            "pdfium-source",
            "internal-portable-policy.json");
        byte[] bytes = File.ReadAllBytes(policyPath);
        Assert.AreEqual(ExpectedPolicySha256, Convert.ToHexStringLower(SHA256.HashData(bytes)));

        using JsonDocument document = JsonDocument.Parse(bytes);
        JsonElement root = document.RootElement;
        Assert.AreEqual("graphreader-pdfium-internal-dev-portable-v1", root.GetProperty("policyId").GetString());
        Assert.AreEqual("internal-development-portable-only", root.GetProperty("stagingMode").GetString());
        Assert.AreEqual(
            "2870fa9244b0f0f69fb743fab1e08deefcb07b2b",
            root.GetProperty("source").GetProperty("revision").GetString());
        Assert.AreEqual(
            "efd13a38cf3cd8e04d8284a42fff42923267293170424153b1a2a96dbf6fe8ea",
            root.GetProperty("inputs").GetProperty("runner").GetProperty("sha256").GetString());
        Assert.IsFalse(root.GetProperty("reviewApproved").GetBoolean());
        Assert.IsFalse(root.GetProperty("cleanMachineEvidence").GetBoolean());
        Assert.IsFalse(root.GetProperty("releaseApproved").GetBoolean());
    }

    [TestMethod]
    public void TrackedPolicyStagesOnlyRunnerCandidateNoticeAndInternalMetadata()
    {
        string policyPath = FindRepositoryFile(
            "packaging",
            "pdfium-source",
            "internal-portable-policy.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(policyPath));
        JsonElement root = document.RootElement;

        Assert.AreEqual(
            "inventory-only-unreviewed",
            root.GetProperty("inputs").GetProperty("dependencyGraph").GetProperty("reviewStatus").GetString());
        Assert.AreEqual(
            "dependency-mapped-not-approved",
            root.GetProperty("inputs").GetProperty("dependencyReviewPolicy").GetProperty("reviewStatus").GetString());
        Assert.AreEqual(
            "REVIEW STATUS: DEPENDENCY-MAPPED",
            root.GetProperty("inputs").GetProperty("deterministicNotice").GetProperty("firstLine").GetString());
        JsonElement stagedFiles = root.GetProperty("stagedFiles");
        Assert.AreEqual(3, stagedFiles.EnumerateObject().Count());
        Assert.AreEqual("graphreader_pdfium_renderer.exe", stagedFiles.GetProperty("runner").GetString());
        Assert.AreEqual("THIRD-PARTY-NOTICES-PDFIUM.txt", stagedFiles.GetProperty("notice").GetString());
        Assert.AreEqual("pdfium-internal-metadata.json", stagedFiles.GetProperty("metadata").GetString());
    }

    private static string FindRepositoryFile(params string[] relativeSegments)
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            string candidate = relativeSegments.Aggregate(current.FullName, Path.Combine);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            current = current.Parent;
        }

        throw new FileNotFoundException(
            $"Unable to locate repository file {Path.Combine(relativeSegments)} from {AppContext.BaseDirectory}.");
    }
}
