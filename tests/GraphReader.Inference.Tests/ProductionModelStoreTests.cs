// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text.Json;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Inference.Tests;

[TestClass]
public sealed class ProductionModelStoreTests
{
    [TestMethod]
    public async Task ApprovedStoreResolvesAndExecutesRealCpuModel()
    {
        using var store = TestStore.Create();
        var resolver = store.CreateResolver(new FakeExecutionProviderDiscovery("CPUExecutionProvider"));

        var resolved = await resolver.ResolveAsync(
            TestStore.ModelId,
            TestStore.Version,
            InferenceProvider.Cpu,
            CancellationToken.None);

        Assert.AreEqual(store.ModelPath, resolved.Identity.FilePath);
        Assert.AreEqual(store.ModelSha256, resolved.Identity.Sha256, ignoreCase: true);
        CollectionAssert.AreEqual(new[] { InferenceProvider.Cpu }, resolved.AvailableProviders.ToArray());

        await using var registry = new OnnxSessionRegistry(
            new FakeExecutionProviderDiscovery("CPUExecutionProvider"),
            new WindowsExecutionProviderPolicy(),
            new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance),
            CpuThreadConfiguration.Create(1, new FixedCoreDetector()));
        var first = await registry.GetOrCreateAsync(resolved.Identity, CancellationToken.None);
        var second = await registry.GetOrCreateAsync(resolved.Identity, CancellationToken.None);
        Assert.AreSame(first.Session, second.Session);
        Assert.AreEqual(1, registry.CreatedSessionCount);

        var execution = await Task.Run(async () => await first.Session!.RunAsync(
            new InferenceInput(new float[] { 7, 8, 9 }, new long[] { 1, 3 }),
            CancellationToken.None));
        CollectionAssert.AreEqual(new float[] { 7, 8, 9 }, execution.Output.ToArray());
        Assert.AreEqual(InferenceProvider.Cpu, execution.Provider);
    }

    [TestMethod]
    public async Task DeclaredDirectMlFallsBackToCpuWhenRuntimeDoesNotProvideIt()
    {
        using var store = TestStore.Create(providers: ["cpu", "directml"]);
        var resolver = store.CreateResolver(new FakeExecutionProviderDiscovery("CPUExecutionProvider"));

        var resolved = await resolver.ResolveAsync(
            TestStore.ModelId,
            TestStore.Version,
            requiredProvider: null,
            CancellationToken.None);

        CollectionAssert.AreEqual(new[] { InferenceProvider.Cpu }, resolved.AvailableProviders.ToArray());
        var error = await Assert.ThrowsExactlyAsync<ProductionModelValidationException>(async () =>
            await resolver.ResolveAsync(
                TestStore.ModelId,
                TestStore.Version,
                InferenceProvider.DirectMl,
                CancellationToken.None));
        Assert.AreEqual("MODEL_PROVIDER_UNAVAILABLE", error.Code);
    }

    [TestMethod]
    public async Task MalformedManifestFailsClosed()
    {
        using var store = TestStore.Create();
        File.WriteAllText(store.ManifestPath, "{not-json");

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_MANIFEST_INVALID", error.Code);
    }

    [TestMethod]
    public async Task MissingPayloadFailsClosed()
    {
        using var store = TestStore.Create();
        File.Delete(store.ModelPath);

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_PAYLOAD_MISSING", error.Code);
    }

    [TestMethod]
    public async Task ExtraFileInApprovedLeafFailsClosed()
    {
        using var store = TestStore.Create();
        File.WriteAllText(Path.Combine(store.ModelDirectory, "unlisted.bin"), "not approved");

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_STORE_EXTRA_FILE", error.Code);
    }

    [TestMethod]
    public async Task PayloadHashMismatchFailsClosed()
    {
        using var store = TestStore.Create();
        File.AppendAllText(store.ModelPath, "tampered");

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_PAYLOAD_CHECKSUM_MISMATCH", error.Code);
    }

    [TestMethod]
    public async Task EveryAuxiliaryPayloadRequiresAndPassesItsOwnChecksum()
    {
        using var missingHash = TestStore.Create(includeAuxiliaryPayload: true, includeAuxiliaryHash: false);
        var missingHashError = await ResolveFailureAsync(missingHash);
        Assert.AreEqual("MODEL_PAYLOAD_INVALID", missingHashError.Code);

        using var mismatch = TestStore.Create(includeAuxiliaryPayload: true);
        File.AppendAllText(mismatch.AuxiliaryPath!, "tampered");
        var mismatchError = await ResolveFailureAsync(mismatch);
        Assert.AreEqual("MODEL_PAYLOAD_CHECKSUM_MISMATCH", mismatchError.Code);

        using var valid = TestStore.Create(includeAuxiliaryPayload: true);
        var resolved = await valid.CreateResolver().ResolveAsync(
            TestStore.ModelId,
            TestStore.Version,
            InferenceProvider.Cpu,
            CancellationToken.None);
        Assert.AreEqual(valid.ModelSha256, resolved.Identity.Sha256, ignoreCase: true);
    }

    [TestMethod]
    public async Task MissingProductionApprovalFailsClosed()
    {
        using var store = TestStore.Create(productionApproval: false);

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_NOT_APPROVED", error.Code);
    }

    [TestMethod]
    public async Task UnreviewedOrMissingNoticeFailsClosed()
    {
        using var unreviewed = TestStore.Create(licenseReviewed: false);
        var unreviewedError = await ResolveFailureAsync(unreviewed);
        Assert.AreEqual("MODEL_LICENSE_NOT_REVIEWED", unreviewedError.Code);

        using var missing = TestStore.Create();
        File.Delete(missing.NoticePath);
        var missingError = await ResolveFailureAsync(missing);
        Assert.AreEqual("MODEL_NOTICE_MISSING", missingError.Code);
    }

    [TestMethod]
    public async Task TamperedBenchmarkEvidenceFailsClosed()
    {
        using var store = TestStore.Create();
        File.AppendAllText(store.BenchmarkPath, "tampered");

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_BENCHMARK_CHECKSUM_MISMATCH", error.Code);
    }

    [TestMethod]
    public async Task CpuCompatibilityIsMandatoryEvenForDirectMlModels()
    {
        using var store = TestStore.Create(providers: ["directml"]);

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_CPU_FALLBACK_UNAVAILABLE", error.Code);
    }

    [TestMethod]
    public async Task PreCanceledResolutionDoesNotReadOrReturnModel()
    {
        using var store = TestStore.Create();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsExactlyAsync<OperationCanceledException>(async () =>
            await store.CreateResolver().ResolveAsync(
                TestStore.ModelId,
                TestStore.Version,
                InferenceProvider.Cpu,
                cancellation.Token));
    }

    private static async Task<ProductionModelValidationException> ResolveFailureAsync(TestStore store) =>
        await Assert.ThrowsExactlyAsync<ProductionModelValidationException>(async () =>
            await store.CreateResolver().ResolveAsync(
                TestStore.ModelId,
                TestStore.Version,
                InferenceProvider.Cpu,
                CancellationToken.None));

    private sealed class FixedCoreDetector : IPhysicalCoreDetector
    {
        public int GetPhysicalCoreCount() => 1;
    }

    private sealed class TestStore : IDisposable
    {
        public const string ModelId = "test-production-identity";
        public const string Version = "1.0.0";

        private TestStore(
            string root,
            string modelDirectory,
            string manifestPath,
            string modelPath,
            string modelSha256,
            string noticePath,
            string benchmarkPath,
            string? auxiliaryPath)
        {
            Root = root;
            ModelDirectory = modelDirectory;
            ManifestPath = manifestPath;
            ModelPath = modelPath;
            ModelSha256 = modelSha256;
            NoticePath = noticePath;
            BenchmarkPath = benchmarkPath;
            AuxiliaryPath = auxiliaryPath;
        }

        public string Root { get; }
        public string ModelDirectory { get; }
        public string ManifestPath { get; }
        public string ModelPath { get; }
        public string ModelSha256 { get; }
        public string NoticePath { get; }
        public string BenchmarkPath { get; }
        public string? AuxiliaryPath { get; }

        public static TestStore Create(
            bool productionApproval = true,
            bool licenseReviewed = true,
            string[]? providers = null,
            bool includeAuxiliaryPayload = false,
            bool includeAuxiliaryHash = true)
        {
            var root = Path.Combine(Path.GetTempPath(), "GraphReaderProductionModelStoreTests", Guid.NewGuid().ToString("N"));
            var modelDirectory = Path.Combine(root, ModelId, Version);
            Directory.CreateDirectory(modelDirectory);

            using var generated = TestOnnxModel.CreateIdentity();
            var modelPath = Path.Combine(modelDirectory, "identity.onnx");
            File.Copy(generated.Path, modelPath);
            var modelBytes = File.ReadAllBytes(modelPath);
            var modelSha = Convert.ToHexString(SHA256.HashData(modelBytes));

            var noticePath = Path.Combine(modelDirectory, "NOTICE.txt");
            File.WriteAllText(noticePath, "Apache-2.0 test model notice.");
            var benchmarkPath = Path.Combine(modelDirectory, "benchmark.json");
            File.WriteAllText(benchmarkPath, "{\"status\":\"pass\",\"fixture_count\":1}");
            var benchmarkSha = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(benchmarkPath)));

            string? auxiliaryPath = null;
            var files = new List<string> { "identity.onnx" };
            var payloadHashes = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["identity.onnx"] = modelSha
            };
            if (includeAuxiliaryPayload)
            {
                auxiliaryPath = Path.Combine(modelDirectory, "labels.txt");
                File.WriteAllText(auxiliaryPath, "0\n1\n2\n");
                files.Add("labels.txt");
                if (includeAuxiliaryHash)
                {
                    payloadHashes["labels.txt"] = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(auxiliaryPath)));
                }
            }

            var manifest = new Dictionary<string, object?>
            {
                ["manifest_version"] = 1,
                ["model_id"] = ModelId,
                ["model_version"] = Version,
                ["task"] = "marker_classifier",
                ["source"] = new Dictionary<string, object?>
                {
                    ["name"] = "GraphReader deterministic test identity",
                    ["url"] = "local://tests/identity",
                    ["revision"] = "test-revision-1"
                },
                ["license"] = new Dictionary<string, object?>
                {
                    ["spdx"] = "Apache-2.0",
                    ["notice_path"] = $"{ModelId}/{Version}/NOTICE.txt",
                    ["reviewed"] = licenseReviewed
                },
                ["sha256"] = modelSha,
                ["files"] = files,
                ["inputs"] = new[] { new Dictionary<string, object?> { ["name"] = "x" } },
                ["outputs"] = new[] { new Dictionary<string, object?> { ["name"] = "y" } },
                ["commercial_use"] = true,
                ["redistribution"] = true,
                ["providers"] = providers ?? ["cpu"],
                ["benchmarks"] = new[]
                {
                    new Dictionary<string, object?>
                    {
                        ["profile"] = "production-gate-v1",
                        ["status"] = "pass",
                        ["release_eligible"] = true,
                        ["production_approval"] = productionApproval,
                        ["evidence_path"] = $"{ModelId}/{Version}/benchmark.json",
                        ["evidence_sha256"] = benchmarkSha
                    }
                }
            };
            if (includeAuxiliaryPayload)
            {
                manifest["preprocessing"] = new Dictionary<string, object?>
                {
                    ["model_payload_sha256"] = payloadHashes
                };
            }

            var manifestPath = Path.Combine(modelDirectory, "manifest.json");
            File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest));
            return new TestStore(
                root,
                modelDirectory,
                manifestPath,
                modelPath,
                modelSha,
                noticePath,
                benchmarkPath,
                auxiliaryPath);
        }

        public ProductionModelStore CreateResolver(IExecutionProviderDiscovery? discovery = null) =>
            new(Root, discovery ?? new FakeExecutionProviderDiscovery("CPUExecutionProvider"));

        public void Dispose()
        {
            try
            {
                Directory.Delete(Root, recursive: true);
            }
            catch (IOException)
            {
            }
        }
    }
}
