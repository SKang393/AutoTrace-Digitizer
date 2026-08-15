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
    public void FrozenIdentitiesRemainUnauthorizedAndUnapproved()
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
        Assert.IsFalse(File.Exists(Path.Combine(candidateRoot, "P1_TRAINING_AUTHORIZATION.json")));
        Assert.IsFalse(File.Exists(Path.Combine(candidateRoot, "P1_SELECTION_RESULT.json")));
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
