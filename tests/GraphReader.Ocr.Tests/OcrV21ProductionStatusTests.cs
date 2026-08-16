// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV21ProductionStatusTests
{
    [TestMethod]
    public void FrozenIdentitiesHaveOnlyBoundedTrainingAuthorizationAndRemainUnapproved()
    {
        string root = FindRepositoryRoot();
        string candidateRoot = Path.Combine(root, "ml", "ocr", "relational_scene_proposal_role_v21");
        using JsonDocument document = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(candidateRoot, "PROTOCOL.json")));
        JsonElement protocol = document.RootElement;

        Assert.AreEqual(3, protocol.GetProperty("candidate_budget").GetProperty("candidate_limit").GetInt32());
        Assert.AreEqual(1, protocol.GetProperty("candidate_budget").GetProperty("candidate_number").GetInt32());
        Assert.AreEqual(1536, protocol.GetProperty("candidate_budget").GetProperty("optimizer_steps_maximum").GetInt32());
        Assert.IsFalse(protocol.GetProperty("fixture_identity_frozen").GetBoolean());
        Assert.IsFalse(protocol.GetProperty("training_authorized").GetBoolean());
        Assert.IsFalse(protocol.GetProperty("public_execution_authorized").GetBoolean());
        Assert.IsFalse(protocol.GetProperty("marker_creation_evaluated").GetBoolean());
        Assert.IsFalse(protocol.GetProperty("artifact_mask_production_approval").GetBoolean());
        Assert.IsFalse(protocol.GetProperty("manifest_created").GetBoolean());
        Assert.IsFalse(protocol.GetProperty("model_store_promoted").GetBoolean());
        Assert.IsFalse(protocol.GetProperty("private_validation_authorized").GetBoolean());
        Assert.IsFalse(protocol.GetProperty("production_approval").GetBoolean());
        Assert.IsFalse(protocol.GetProperty("release_eligible").GetBoolean());

        string sealPath = Path.Combine(candidateRoot, "SPLIT_SEAL.json");
        Assert.AreEqual(
            "085c93c73731ca97bc85d4eed52841547e6faab28effa56ca14db90d999b3047",
            Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(sealPath))).ToLowerInvariant());
        using JsonDocument sealDocument = JsonDocument.Parse(File.ReadAllBytes(sealPath));
        JsonElement seal = sealDocument.RootElement;
        Assert.AreEqual(0, seal.GetProperty("optimizer_steps_at_freeze").GetInt32());
        Assert.AreEqual(0, seal.GetProperty("selection_evaluations").GetInt32());
        Assert.AreEqual(0, seal.GetProperty("public_evaluations").GetInt32());
        Assert.IsFalse(seal.GetProperty("training_authorized").GetBoolean());
        Assert.IsFalse(seal.GetProperty("public_execution_authorized").GetBoolean());
        Assert.IsFalse(seal.GetProperty("marker_creation_evaluated").GetBoolean());
        Assert.IsFalse(seal.GetProperty("artifact_mask_production_approval").GetBoolean());
        Assert.IsFalse(seal.GetProperty("production_approval").GetBoolean());
        Assert.IsFalse(seal.GetProperty("release_eligible").GetBoolean());
        Assert.AreEqual(
            "c82c527daa4afdadaa477895cd58a93072d73c6ece12b39318f5b6b5f951563d",
            seal.GetProperty("train").GetProperty("archive_sha256").GetString());
        Assert.AreEqual(
            "9d3831f31cdb097f0ec4a2d174ed8d9653d76472394fb5f42c44a17e99990371",
            seal.GetProperty("selection").GetProperty("archive_sha256").GetString());
        Assert.AreEqual(
            "b4ae7547731949ac6df1f9afe3fd83178b3cf9c55c81dbd017592a71d90ddab8",
            seal.GetProperty("sealed_public").GetProperty("archive_sha256").GetString());
        string authorizationPath = Path.Combine(candidateRoot, "P1_TRAINING_AUTHORIZATION.json");
        using JsonDocument authorizationDocument = JsonDocument.Parse(File.ReadAllBytes(authorizationPath));
        JsonElement authorization = authorizationDocument.RootElement;
        Assert.AreEqual(
            "d9d5ed2eda4f53da54660f47ef1de594b5e628b7",
            authorization.GetProperty("authorized_source_commit").GetString());
        Assert.AreEqual(
            "e3fdbb0208a49b890ae4eebda0bf3db9b52417c31ecdbef5d9521fd327be5fca",
            authorization.GetProperty("candidate_config_sha256").GetString());
        Assert.AreEqual(
            "f6a090b2611d41ddd045939f1c4e918d464297e99b080a9bee696a3ddc26a4a1",
            authorization.GetProperty("runner_source_bundle_sha256").GetString());
        Assert.AreEqual(1, authorization.GetProperty("execution_limit").GetInt32());
        Assert.AreEqual(0, authorization.GetProperty("execution_count").GetInt32());
        Assert.IsTrue(authorization.GetProperty("training_authorized").GetBoolean());
        Assert.IsFalse(authorization.GetProperty("public_execution_authorized").GetBoolean());
        Assert.IsFalse(authorization.GetProperty("private_validation_authorized").GetBoolean());
        Assert.IsFalse(authorization.GetProperty("production_approval").GetBoolean());
        Assert.IsFalse(authorization.GetProperty("release_eligible").GetBoolean());
        using JsonDocument resultDocument = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(candidateRoot, "P1_SELECTION_RESULT.json")));
        JsonElement result = resultDocument.RootElement;
        Assert.IsTrue(result.GetProperty("p1_consumed").GetBoolean());
        Assert.AreEqual(1536, result.GetProperty("optimizer_steps").GetInt32());
        Assert.IsFalse(result.GetProperty("selection_gate_passed").GetBoolean());
        Assert.AreEqual(128, result.GetProperty("scene_count").GetInt32());
        Assert.AreEqual(107, result.GetProperty("exact_scene_count").GetInt32());
        Assert.AreEqual(1004, result.GetProperty("true_positives").GetInt32());
        Assert.AreEqual(1, result.GetProperty("false_positives").GetInt32());
        Assert.AreEqual(20, result.GetProperty("false_negatives").GetInt32());
        Assert.AreEqual(0, result.GetProperty("duplicate_regions").GetInt32());
        Assert.AreEqual(1, result.GetProperty("prohibited_structure_hits").GetInt32());
        Assert.IsFalse(result.GetProperty("onnx_parity_passed").GetBoolean());
        Assert.IsFalse(result.GetProperty("public_archive_opened").GetBoolean());
        Assert.AreEqual(0, result.GetProperty("public_evaluation_count").GetInt32());
        Assert.IsFalse(result.GetProperty("production_approval").GetBoolean());
        Assert.IsFalse(result.GetProperty("release_eligible").GetBoolean());
        using JsonDocument p2ResultDocument = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(candidateRoot, "P2_SELECTION_RESULT.json")));
        JsonElement p2Result = p2ResultDocument.RootElement;
        Assert.IsTrue(p2Result.GetProperty("p2_consumed").GetBoolean());
        Assert.AreEqual(384, p2Result.GetProperty("candidate_optimizer_steps").GetInt32());
        Assert.AreEqual(1920, p2Result.GetProperty("total_optimizer_steps").GetInt32());
        Assert.IsFalse(p2Result.GetProperty("selection_gate_passed").GetBoolean());
        Assert.AreEqual(128, p2Result.GetProperty("scene_count").GetInt32());
        Assert.AreEqual(109, p2Result.GetProperty("exact_scene_count").GetInt32());
        Assert.AreEqual(1006, p2Result.GetProperty("true_positives").GetInt32());
        Assert.AreEqual(1, p2Result.GetProperty("false_positives").GetInt32());
        Assert.AreEqual(18, p2Result.GetProperty("false_negatives").GetInt32());
        Assert.AreEqual(0, p2Result.GetProperty("duplicate_regions").GetInt32());
        Assert.AreEqual(1, p2Result.GetProperty("prohibited_structure_hits").GetInt32());
        Assert.IsFalse(p2Result.GetProperty("onnx_parity_passed").GetBoolean());
        Assert.IsFalse(p2Result.GetProperty("public_archive_opened").GetBoolean());
        Assert.AreEqual(0, p2Result.GetProperty("public_evaluation_count").GetInt32());
        Assert.IsFalse(p2Result.GetProperty("production_approval").GetBoolean());
        Assert.IsFalse(p2Result.GetProperty("release_eligible").GetBoolean());
        using JsonDocument p3ResultDocument = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(candidateRoot, "P3_SELECTION_RESULT.json")));
        JsonElement p3Result = p3ResultDocument.RootElement;
        Assert.IsTrue(p3Result.GetProperty("p3_consumed").GetBoolean());
        Assert.AreEqual(0, p3Result.GetProperty("optimizer_steps").GetInt32());
        Assert.IsFalse(p3Result.GetProperty("selection_gate_passed").GetBoolean());
        Assert.AreEqual(128, p3Result.GetProperty("scene_count").GetInt32());
        Assert.AreEqual(116, p3Result.GetProperty("exact_scene_count").GetInt32());
        Assert.AreEqual(1014, p3Result.GetProperty("true_positives").GetInt32());
        Assert.AreEqual(2, p3Result.GetProperty("false_positives").GetInt32());
        Assert.AreEqual(10, p3Result.GetProperty("false_negatives").GetInt32());
        Assert.AreEqual(0, p3Result.GetProperty("duplicate_regions").GetInt32());
        Assert.AreEqual(2, p3Result.GetProperty("prohibited_structure_hits").GetInt32());
        Assert.IsTrue(p3Result.GetProperty("onnx_parity_passed").GetBoolean());
        Assert.IsFalse(p3Result.GetProperty("public_archive_opened").GetBoolean());
        Assert.AreEqual(0, p3Result.GetProperty("public_evaluation_count").GetInt32());
        Assert.IsFalse(p3Result.GetProperty("production_approval").GetBoolean());
        Assert.IsFalse(p3Result.GetProperty("release_eligible").GetBoolean());
        Assert.IsFalse(File.Exists(Path.Combine(candidateRoot, "PUBLIC_GATE_AUTHORIZATION.json")));
        Assert.IsFalse(Directory.Exists(Path.Combine(root, "models", "manifest", "ocr", "relational_scene_proposal_role_v21")));
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
