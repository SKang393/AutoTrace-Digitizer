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
                "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d",
            ["RealESRGAN_x4plus_anime_6B"] =
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
    private static readonly string[] SecondaryNcnnFiles =
    [
        "models/realesrgan-x4plus-anime.param",
        "models/realesrgan-x4plus-anime.bin"
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
    public void BenchmarkEntriesRecordOnlyDirectEvidenceAndNoInventedQualityClaims()
    {
        JsonDocument[] manifests = LoadManifests();
        try
        {
            foreach (JsonDocument manifest in manifests)
            {
                string modelId = GetModelId(manifest);
                JsonElement benchmark = manifest.RootElement.GetProperty("benchmarks")[0];
                Assert.AreEqual("none", benchmark.GetProperty("quality_claims").GetString(), modelId);
                Assert.AreEqual(2, benchmark.GetProperty("configured_output_scale").GetInt32(), modelId);

                string[] metrics = benchmark.GetProperty("metrics_required")
                    .EnumerateArray()
                    .Select(static metric => metric.GetString()!)
                    .ToArray();
                CollectionAssert.AreEquivalent(RequiredMetrics, metrics, modelId);

                string status = benchmark.GetProperty("status").GetString()!;
                switch (modelId)
                {
                    case "RealESRGAN_x2plus":
                        Assert.AreEqual("blocked_adapter_incompatible", status);
                        Assert.IsFalse(benchmark.GetProperty("production_approval").GetBoolean());
                        Assert.AreEqual(
                            ExpectedModelHashes[modelId],
                            benchmark.GetProperty("artifact_verification")
                                .GetProperty("verified_sha256")
                                .GetString());
                        break;
                    case "realesr-general-x4v3":
                        Assert.AreEqual("not_run", status);
                        break;
                    case "realesr-animevideov3":
                        Assert.AreEqual("partial_runtime_only", status);
                        Assert.IsFalse(benchmark.GetProperty("production_approval").GetBoolean());
                        JsonElement runtime = benchmark.GetProperty("fixed_public_synthetic_runtime");
                        Assert.AreEqual(2, runtime.GetProperty("adapter_success_count").GetInt32());
                        Assert.AreEqual(2, runtime.GetProperty("exact_two_x_dimension_count").GetInt32());
                        Assert.AreEqual("cache_hit", runtime.GetProperty("cache_repeat_status").GetString());
                        Assert.AreEqual("vulkan", runtime.GetProperty("provider").GetString());
                        break;
                    case "RealESRGAN_x4plus_anime_6B":
                        Assert.AreEqual("failed_scientific_fidelity", status);
                        Assert.IsFalse(benchmark.GetProperty("production_approval").GetBoolean());
                        JsonElement secondaryRuntime = benchmark.GetProperty("fixed_public_synthetic_runtime");
                        Assert.AreEqual(2, secondaryRuntime.GetProperty("runtime_success_count").GetInt32());
                        Assert.AreEqual(2, secondaryRuntime.GetProperty("exact_two_x_dimension_count").GetInt32());
                        Assert.AreEqual("vulkan", secondaryRuntime.GetProperty("provider").GetString());
                        JsonElement privateFailure = benchmark.GetProperty("private_chandler_scientific_fidelity");
                        Assert.AreEqual("failed", privateFailure.GetProperty("status").GetString());
                        Assert.IsTrue(privateFailure.GetProperty("source_unchanged").GetBoolean());
                        Assert.AreEqual(
                            "d05e259e69f139d2649aaab8e99f866ccd4092534021612e93650b5048c97e85",
                            privateFailure.GetProperty("output_sha256").GetString());
                        break;
                    default:
                        Assert.Fail($"Unexpected benchmark model '{modelId}'.");
                        break;
                }
            }
        }
        finally
        {
            DisposeAll(manifests);
        }
    }

    [TestMethod]
    public async Task OfficialPyTorchX2PlusCannotBeMisrepresentedAsAnNcnnAdapterModel()
    {
        using var environment = new AdapterTestEnvironment();
        var x2Plus = new EnhancementModel(
            "RealESRGAN_x2plus",
            "v0.2.1",
            ExpectedModelHashes["RealESRGAN_x2plus"],
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "v0.2.1@64ad194ddaf9c4d8c4b0d1b98cac6d89d3ea0d11",
            "BSD-3-Clause",
            "LICENSES/Real-ESRGAN-BSD-3-Clause.txt",
            [new ModelArtifact("RealESRGAN_x2plus.pth", ExpectedModelHashes["RealESRGAN_x2plus"])]);

        EnhancementResult result = await environment.CreateAdapter().EnhanceAsync(
            environment.CreateRequest(model: x2Plus),
            CancellationToken.None);

        Assert.AreEqual(EnhancementStatus.Failed, result.Status);
        Assert.AreEqual(EnhancementFailureCode.InvalidRequest, result.Diagnostic.Code);
        Assert.AreEqual(0, environment.Runner.InvocationCount);
        StringAssert.Contains(result.Diagnostic.Message, "NCNN model");
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

            JsonElement secondary = byId["RealESRGAN_x4plus_anime_6B"];
            CollectionAssert.AreEqual(
                NcnnProviders,
                ReadStringArray(secondary.GetProperty("providers")));
            Assert.AreEqual(
                "realesrgan-x4plus-anime",
                secondary.GetProperty("preprocessing").GetProperty("runtime_model_name").GetString());
            Assert.AreEqual(
                "RealESRGAN_x4plus_anime_6B",
                secondary.GetProperty("preprocessing").GetProperty("upstream_model_identity").GetString());
            Assert.IsFalse(
                secondary.GetProperty("preprocessing").GetProperty("local_adapter_approval").GetBoolean());
            StringAssert.Contains(
                secondary.GetProperty("preprocessing").GetProperty("upstream_identity_mapping_url").GetString(),
                "685d429c81888252bdb10f56c7754baededc3823/docs/anime_model.md#L20-L35");
            CollectionAssert.AreEquivalent(
                SecondaryNcnnFiles,
                ReadStringArray(secondary.GetProperty("files")));
        }
        finally
        {
            DisposeAll(manifests);
        }
    }

    [TestMethod]
    public void AnimeVideoV3IsTheDeveloperDefaultWhileProductionApprovalsRemainBlocked()
    {
        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(ManifestDirectory, ManifestDrivenRealEsrganBackend.DefaultManifestFileName)));
        JsonElement root = manifest.RootElement;
        Assert.AreEqual(ManifestDrivenRealEsrganBackend.DefaultModelId, GetModelId(manifest));
        JsonElement preprocessing = root.GetProperty("preprocessing");
        Assert.AreEqual("realesr-animevideov3", preprocessing.GetProperty("runtime_model_name").GetString());
        Assert.IsTrue(preprocessing.GetProperty("local_adapter_approval").GetBoolean());
        Assert.AreEqual(
            "developer_local_evaluation_only",
            preprocessing.GetProperty("local_adapter_approval_scope").GetString());
        JsonElement localEvidence = preprocessing.GetProperty("local_adapter_evidence");
        Assert.IsTrue(localEvidence.GetProperty("source_preservation_verified").GetBoolean());
        Assert.IsTrue(localEvidence.GetProperty("exact_two_x_dimensions_verified").GetBoolean());
        Assert.AreEqual("none", localEvidence.GetProperty("production_quality_claims").GetString());
        Assert.AreEqual(2, preprocessing.GetProperty("runtime_scale_argument").GetInt32());
        Assert.AreEqual(
            "07e49f7cbb4ede01ae4dd4c399d3a7e5846e3d2085c3128eff881e55cb7b1a0c",
            preprocessing.GetProperty("runtime_executable_sha256").GetString());
        Assert.AreEqual(
            "55aba23cdcd6484fbb06f4155b8ca75adfce7a881f10afd0c49457165e677164",
            preprocessing.GetProperty("runtime_files_sha256").GetProperty("vcomp140.dll").GetString());
        JsonElement runtimeRedistribution = preprocessing.GetProperty("runtime_redistribution");
        Assert.IsTrue(runtimeRedistribution.GetProperty("provenance_reviewed").GetBoolean());
        Assert.AreEqual(
            "redistribution_provenance_only",
            runtimeRedistribution.GetProperty("approval_scope").GetString());
        Assert.IsFalse(runtimeRedistribution.GetProperty("approved").GetBoolean());
        Assert.IsFalse(root.GetProperty("benchmarks")[0].GetProperty("production_approval").GetBoolean());
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
