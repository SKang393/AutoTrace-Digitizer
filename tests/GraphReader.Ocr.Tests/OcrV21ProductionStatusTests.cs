// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV21ProductionStatusTests
{
    [TestMethod]
    public void PreregisteredCandidateRemainsUnfrozenAndUnapproved()
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

        Assert.IsFalse(File.Exists(Path.Combine(candidateRoot, "SPLIT_SEAL.json")));
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
