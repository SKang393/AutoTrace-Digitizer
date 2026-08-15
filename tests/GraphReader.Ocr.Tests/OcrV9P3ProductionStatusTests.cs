// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV9P3ProductionStatusTests
{
    [TestMethod]
    public void FailedSelectionRemainsConsumedAndCannotAuthorizePublicOrProduction()
    {
        string root = FindRepositoryRoot();
        string candidateRoot = Path.Combine(root, "ml", "ocr", "cross_model_consensus_v9_p3");
        string resultPath = Path.Combine(candidateRoot, "P3_SELECTION_RESULT.json");
        string publicAuthorizationPath = Path.Combine(candidateRoot, "PUBLIC_GATE_AUTHORIZATION.json");

        Assert.AreEqual(
            "f7935a64de07f1187a4ca854c01a90b4fcd004c533f57c6f464d98c5e59105e2",
            Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(resultPath))).ToLowerInvariant());
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(resultPath));
        JsonElement rootElement = document.RootElement;

        Assert.AreEqual("P3", rootElement.GetProperty("candidate_id").GetString());
        Assert.IsTrue(rootElement.GetProperty("execution_consumed").GetBoolean());
        Assert.AreEqual(1, rootElement.GetProperty("execution_count").GetInt32());
        Assert.IsFalse(rootElement.GetProperty("selection_gate_passed").GetBoolean());
        Assert.IsFalse(rootElement.GetProperty("rerun_or_repair_authorized").GetBoolean());
        Assert.IsFalse(rootElement.GetProperty("public_gate_authorized").GetBoolean());
        Assert.AreEqual(0, rootElement.GetProperty("public_execution_count").GetInt32());
        Assert.IsFalse(rootElement.GetProperty("marker_creation_evaluated").GetBoolean());
        Assert.IsFalse(rootElement.GetProperty("artifact_mask_production_approval").GetBoolean());
        Assert.IsFalse(rootElement.GetProperty("manifest_creation_authorized").GetBoolean());
        Assert.IsFalse(rootElement.GetProperty("model_store_promotion_authorized").GetBoolean());
        Assert.IsFalse(rootElement.GetProperty("private_validation_authorized").GetBoolean());
        Assert.IsFalse(rootElement.GetProperty("production_approval").GetBoolean());
        Assert.IsFalse(rootElement.GetProperty("release_eligible").GetBoolean());
        Assert.IsTrue(rootElement.GetProperty("direct_runtime_evidence_passed").GetBoolean());
        Assert.AreEqual(5, rootElement.GetProperty("model_execution_count").GetInt32());

        JsonElement metrics = rootElement.GetProperty("metrics");
        Assert.AreEqual(192, metrics.GetProperty("scene_count").GetInt32());
        Assert.AreEqual(102, metrics.GetProperty("exact_detection_scene_count").GetInt32());
        Assert.AreEqual(5, metrics.GetProperty("false_positives").GetInt32());
        Assert.AreEqual(91, metrics.GetProperty("false_negatives").GetInt32());
        Assert.AreEqual(5, metrics.GetProperty("prohibited_structure_hits").GetInt32());

        Assert.IsFalse(File.Exists(publicAuthorizationPath));
        Assert.IsFalse(Directory.Exists(Path.Combine(root, "models", "manifest", "ocr", "cross_model_consensus_v9_p3")));
    }

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "GraphAutoReader.slnx")))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not find repository root.");
    }
}
