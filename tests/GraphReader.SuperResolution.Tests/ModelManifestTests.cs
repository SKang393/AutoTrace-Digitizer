// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
using Json.Schema;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.SuperResolution.Tests;

[TestClass]
public sealed class ModelManifestTests
{
    private static readonly string ManifestDirectory =
        RepositoryPaths.FromRoot("models", "manifest", "super-resolution");

    private static readonly IReadOnlyDictionary<string, string> ExpectedModelHashes =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["RealESRGAN_x2plus"] =
                "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
            ["realesr-general-x4v3"] =
                "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
            ["realesr-animevideov3"] =
                "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d"
        };

    private static readonly string[] RequiredMetrics =
    {
        "marker_center_f1",
        "shape_fill_classification_f1",
        "numeric_ocr_exact_match",
        "axis_line_localization_error_pixels",
        "hallucinated_structure_rate",
        "runtime_ms",
        "peak_memory_bytes"
    };

    private static readonly string[] ReferenceProviders = ["cpu", "cuda"];
    private static readonly string[] NcnnProviders = ["vulkan"];
    private static readonly string[] NcnnFiles =
    [
        "models/realesr-animevideov3-x2.param",
        "models/realesr-animevideov3-x2.bin"
    ];

    [TestMethod]
    public void RequiredBenchmarkModelMatrixIsCompleteAndSchemaValid()
    {
        JsonSchema schema = JsonSchema.FromText(
            File.ReadAllText(RepositoryPaths.FromRoot("contracts", "model-manifest.schema.json")));
        JsonDocument[] manifests = LoadManifests();
        try
        {
            CollectionAssert.AreEquivalent(
                ExpectedModelHashes.Keys.ToArray(),
                manifests.Select(GetModelId).ToArray());

            foreach (JsonDocument manifest in manifests)
            {
                EvaluationResults evaluation = schema.Evaluate(manifest.RootElement);
                Assert.IsTrue(evaluation.IsValid, $"{GetModelId(manifest)} failed the frozen schema.");
                Assert.AreEqual(
                    "super_resolution",
                    manifest.RootElement.GetProperty("task").GetString(),
                    GetModelId(manifest));
            }
        }
        finally
        {
            DisposeAll(manifests);
        }
    }

    [TestMethod]
    public void OfficialModelChecksumsAndRedistributionReviewArePinned()
    {
        JsonDocument[] manifests = LoadManifests();
        try
        {
            foreach (JsonDocument manifest in manifests)
            {
                JsonElement root = manifest.RootElement;
                string modelId = GetModelId(manifest);
                Assert.AreEqual(ExpectedModelHashes[modelId], root.GetProperty("sha256").GetString());
                Assert.IsTrue(root.GetProperty("commercial_use").GetBoolean(), modelId);
                Assert.IsTrue(root.GetProperty("redistribution").GetBoolean(), modelId);

                JsonElement license = root.GetProperty("license");
                Assert.AreEqual("BSD-3-Clause", license.GetProperty("spdx").GetString(), modelId);
                Assert.IsTrue(license.GetProperty("reviewed").GetBoolean(), modelId);
                Assert.AreEqual(
                    "LICENSES/Real-ESRGAN-BSD-3-Clause.txt",
                    license.GetProperty("notice_path").GetString(),
                    modelId);
            }
        }
        finally
        {
            DisposeAll(manifests);
        }

        string notice = File.ReadAllText(
            RepositoryPaths.FromRoot("LICENSES", "Real-ESRGAN-BSD-3-Clause.txt"));
        StringAssert.Contains(notice, "Copyright (c) 2021, Xintao Wang");
        StringAssert.Contains(notice, "Redistribution and use in source and binary forms");
        StringAssert.Contains(notice, "Neither the name of the copyright holder");
        StringAssert.Contains(notice, "THIS SOFTWARE IS PROVIDED");
    }

    [TestMethod]
    public void BenchmarkEntriesMakeNoInventedQualityOrTimingClaims()
    {
        JsonDocument[] manifests = LoadManifests();
        try
        {
            foreach (JsonDocument manifest in manifests)
            {
                string modelId = GetModelId(manifest);
                JsonElement benchmark = manifest.RootElement.GetProperty("benchmarks")[0];
                Assert.AreEqual("not_run", benchmark.GetProperty("status").GetString(), modelId);
                Assert.AreEqual("none", benchmark.GetProperty("quality_claims").GetString(), modelId);
                Assert.AreEqual(2, benchmark.GetProperty("configured_output_scale").GetInt32(), modelId);

                string[] metrics = benchmark.GetProperty("metrics_required")
                    .EnumerateArray()
                    .Select(static metric => metric.GetString()!)
                    .ToArray();
                CollectionAssert.AreEquivalent(RequiredMetrics, metrics, modelId);
            }
        }
        finally
        {
            DisposeAll(manifests);
        }
    }

    [TestMethod]
    public void ProviderCompatibilityTruthfullySeparatesReferenceAndNcnnArtifacts()
    {
        JsonDocument[] manifests = LoadManifests();
        try
        {
            Dictionary<string, JsonElement> byId = manifests.ToDictionary(
                GetModelId,
                static document => document.RootElement,
                StringComparer.Ordinal);

            AssertReferenceProvider(byId["RealESRGAN_x2plus"]);
            AssertReferenceProvider(byId["realesr-general-x4v3"]);

            JsonElement ncnn = byId["realesr-animevideov3"];
            CollectionAssert.AreEqual(
                NcnnProviders,
                ReadStringArray(ncnn.GetProperty("providers")));
            string compatibility = ncnn.GetProperty("benchmarks")[0]
                .GetProperty("adapter_compatibility")
                .GetString()!;
            StringAssert.Contains(compatibility, "Official NCNN param/bin payload");

            string[] files = ReadStringArray(ncnn.GetProperty("files"));
            CollectionAssert.AreEquivalent(
                NcnnFiles,
                files);
        }
        finally
        {
            DisposeAll(manifests);
        }
    }

    [TestMethod]
    public void EveryManifestDeclaresExactTwoXOutputAndInverseMapping()
    {
        JsonDocument[] manifests = LoadManifests();
        try
        {
            foreach (JsonDocument manifest in manifests)
            {
                string modelId = GetModelId(manifest);
                JsonElement output = manifest.RootElement.GetProperty("outputs")[0];
                Assert.AreEqual(2, output.GetProperty("configured_output_scale").GetInt32(), modelId);
                Assert.AreEqual("2 * input_width", output.GetProperty("width").GetString(), modelId);
                Assert.AreEqual("2 * input_height", output.GetProperty("height").GetString(), modelId);
                Assert.AreEqual("enhanced_pixels", output.GetProperty("coordinate_space").GetString(), modelId);
                Assert.AreEqual(
                    "original_x = enhanced_x / 2; original_y = enhanced_y / 2",
                    output.GetProperty("inverse_coordinate_mapping").GetString(),
                    modelId);
            }
        }
        finally
        {
            DisposeAll(manifests);
        }
    }

    private static void AssertReferenceProvider(JsonElement manifest)
    {
        string modelId = manifest.GetProperty("model_id").GetString()!;
        CollectionAssert.AreEquivalent(
            ReferenceProviders,
            ReadStringArray(manifest.GetProperty("providers")),
            modelId);
        string compatibility = manifest.GetProperty("benchmarks")[0]
            .GetProperty("adapter_compatibility")
            .GetString()!;
        StringAssert.Contains(compatibility, "not directly consumable by realesrgan-ncnn-vulkan");
    }

    private static string[] ReadStringArray(JsonElement element) =>
        element.EnumerateArray().Select(static value => value.GetString()!).ToArray();

    private static JsonDocument[] LoadManifests() =>
        Directory.GetFiles(ManifestDirectory, "*.json")
            .Order(StringComparer.Ordinal)
            .Select(static path => JsonDocument.Parse(File.ReadAllText(path)))
            .ToArray();

    private static string GetModelId(JsonDocument document) =>
        document.RootElement.GetProperty("model_id").GetString()!;

    private static void DisposeAll(IEnumerable<JsonDocument> documents)
    {
        foreach (JsonDocument document in documents)
        {
            document.Dispose();
        }
    }
}
