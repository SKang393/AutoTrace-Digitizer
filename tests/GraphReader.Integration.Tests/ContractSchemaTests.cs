// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
using Json.Schema;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Integration.Tests;

[TestClass]
public sealed class ContractSchemaTests
{
    private static readonly Lazy<Dictionary<string, JsonSchema>> FrozenSchemas =
        new(LoadFrozenSchemas);

    private const string ValidProject = """
        {
          "schema_version": 1,
          "project_id": "c9ad63cf-64bb-42bc-8e60-4547f6f43e74",
          "app_version": "0.0.1",
          "sources": [],
          "panels": [],
          "audit": { "events": [] }
        }
        """;

    private const string InvalidProject = """
        {
          "schema_version": 2,
          "project_id": "c9ad63cf-64bb-42bc-8e60-4547f6f43e74",
          "app_version": "0.0.1",
          "sources": [],
          "panels": [],
          "audit": { "events": [] }
        }
        """;

    private const string ValidVisionResult = """
        {
          "contract_version": 1,
          "run_id": "99062763-b0ee-4c28-961f-92ba3ed1a3b4",
          "project_id": "c9ad63cf-64bb-42bc-8e60-4547f6f43e74",
          "panel_id": "42f0271f-5ee7-4e89-9b83-42e91a9701c2",
          "stage": "axis",
          "stage_version": "0.1.0",
          "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "coordinate_space": "original_pixels",
          "timing_ms": { "total": 1.25 },
          "confidence": 0.9,
          "warnings": [],
          "payload": {}
        }
        """;

    private const string InvalidVisionResult = """
        {
          "contract_version": 1,
          "run_id": "99062763-b0ee-4c28-961f-92ba3ed1a3b4",
          "project_id": "c9ad63cf-64bb-42bc-8e60-4547f6f43e74",
          "panel_id": "42f0271f-5ee7-4e89-9b83-42e91a9701c2",
          "stage": "axis",
          "stage_version": "0.1.0",
          "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "coordinate_space": "enhanced_pixels",
          "timing_ms": { "total": 1.25 },
          "confidence": 1.1,
          "warnings": [],
          "payload": {}
        }
        """;

    private const string ValidModelManifest = """
        {
          "manifest_version": 1,
          "model_id": "marker-center-default",
          "model_version": "0.1.0",
          "task": "marker_center",
          "source": {
            "name": "Graph Auto Reader synthetic baseline",
            "url": "https://example.invalid/model",
            "revision": "none"
          },
          "license": {
            "spdx": "Apache-2.0",
            "notice_path": "THIRD_PARTY_NOTICES.md",
            "reviewed": true
          },
          "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
          "files": ["marker-center.onnx"],
          "inputs": [],
          "outputs": [],
          "commercial_use": true,
          "redistribution": true,
          "providers": ["cpu"]
        }
        """;

    private const string InvalidModelManifest = """
        {
          "manifest_version": 1,
          "model_id": "marker-center-default",
          "model_version": "0.1.0",
          "task": "marker_center",
          "source": {
            "name": "Graph Auto Reader synthetic baseline",
            "url": "https://example.invalid/model",
            "revision": "none"
          },
          "license": {
            "spdx": "Apache-2.0",
            "notice_path": "THIRD_PARTY_NOTICES.md",
            "reviewed": true
          },
          "sha256": "not-a-checksum",
          "files": [],
          "inputs": [],
          "outputs": [],
          "commercial_use": true,
          "redistribution": true,
          "providers": []
        }
        """;

    [TestMethod]
    public void AllFrozenSchemasParseAsJsonAndJsonSchema()
    {
        string contractsDirectory = Path.Combine(RepositoryRoot.Find(), "contracts");
        string[] schemaFiles = Directory.GetFiles(contractsDirectory, "*.schema.json")
            .Order(StringComparer.Ordinal)
            .ToArray();

        Assert.AreEqual(3, schemaFiles.Length, "Goal 00 freezes exactly three contract schemas.");

        foreach (string schemaFile in schemaFiles)
        {
            string schemaText = File.ReadAllText(schemaFile);
            using JsonDocument document = JsonDocument.Parse(schemaText);
            Assert.IsTrue(document.RootElement.TryGetProperty("$schema", out _), schemaFile);
            Assert.IsTrue(FrozenSchemas.Value.ContainsKey(Path.GetFileName(schemaFile)), schemaFile);
        }
    }

    [TestMethod]
    public void ProjectSchemaAcceptsPositiveAndRejectsNegativeSample() =>
        AssertPositiveAndNegative("project.schema.json", ValidProject, InvalidProject);

    [TestMethod]
    public void VisionResultSchemaAcceptsPositiveAndRejectsNegativeSample() =>
        AssertPositiveAndNegative(
            "vision-result.schema.json",
            ValidVisionResult,
            InvalidVisionResult);

    [TestMethod]
    public void ModelManifestSchemaAcceptsPositiveAndRejectsNegativeSample() =>
        AssertPositiveAndNegative(
            "model-manifest.schema.json",
            ValidModelManifest,
            InvalidModelManifest);

    private static void AssertPositiveAndNegative(
        string schemaFileName,
        string validInstance,
        string invalidInstance)
    {
        JsonSchema schema = FrozenSchemas.Value[schemaFileName];
        using JsonDocument validDocument = JsonDocument.Parse(validInstance);
        using JsonDocument invalidDocument = JsonDocument.Parse(invalidInstance);

        Assert.IsTrue(
            schema.Evaluate(validDocument.RootElement).IsValid,
            $"Positive sample failed {schemaFileName}.");
        Assert.IsFalse(
            schema.Evaluate(invalidDocument.RootElement).IsValid,
            $"Negative sample passed {schemaFileName}.");
    }

    private static Dictionary<string, JsonSchema> LoadFrozenSchemas()
    {
        string contractsDirectory = Path.Combine(RepositoryRoot.Find(), "contracts");
        return Directory.GetFiles(contractsDirectory, "*.schema.json")
            .ToDictionary(
                static path => Path.GetFileName(path),
                static path => JsonSchema.FromText(File.ReadAllText(path)),
                StringComparer.Ordinal);
    }
}
