// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Inference.Tests;

[TestClass]
public sealed class ProductionModelStoreTests
{
    [TestMethod]
    public async Task PackagedDistributionTreeResolvesAndExecutesRealCpuModel()
    {
        using var store = TestStore.Create();
        var resolver = store.CreateResolver(new FakeExecutionProviderDiscovery("CPUExecutionProvider"));

        var resolved = await resolver.ResolveAsync(
            TestStore.ModelId,
            TestStore.Version,
            InferenceProvider.Cpu,
            CancellationToken.None);

        Assert.AreEqual(store.ModelPath, resolved.Identity.FilePath);
        StringAssert.Contains(resolved.ManifestPath, Path.Combine("manifest", TestStore.ModelId));
        StringAssert.Contains(resolved.Identity.FilePath, Path.Combine("runtime", TestStore.ModelId, TestStore.Version));
        StringAssert.Contains(resolved.NoticePath, Path.Combine("notices", TestStore.ModelId, TestStore.Version, "test-model.txt"));
        StringAssert.Contains(resolved.BenchmarkEvidencePath, Path.Combine("evidence", TestStore.ModelId, TestStore.Version, "test-model-benchmark.json"));
        Assert.AreEqual(store.ModelSha256, resolved.Identity.Sha256, ignoreCase: true);
        CollectionAssert.AreEqual(new[] { InferenceProvider.Cpu }, resolved.AvailableProviders.ToArray());

        await using var registry = new OnnxSessionRegistry(
            new FakeExecutionProviderDiscovery("CPUExecutionProvider"),
            new WindowsExecutionProviderPolicy(),
            new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance),
            CpuThreadConfiguration.Create(1, new FixedCoreDetector()));
        var first = await registry.GetOrCreateAsync(resolved, CancellationToken.None);
        var second = await registry.GetOrCreateAsync(resolved, CancellationToken.None);
        Assert.AreSame(first.Session, second.Session);
        Assert.AreEqual(1, registry.CreatedSessionCount);

        var execution = await Task.Run(async () => await first.Session!.RunAsync(
            new InferenceInput(new float[] { 7, 8, 9 }, new long[] { 1, 3 }),
            CancellationToken.None));
        CollectionAssert.AreEqual(new float[] { 7, 8, 9 }, execution.Output.ToArray());
        Assert.AreEqual(InferenceProvider.Cpu, execution.Provider);
    }

    [TestMethod]
    public async Task ResolveAllReturnsOnlyChecksumValidatedCpuApprovedModels()
    {
        using var store = TestStore.Create();
        var resolver = store.CreateResolver(new FakeExecutionProviderDiscovery("CPUExecutionProvider"));

        IReadOnlyList<ResolvedProductionModel> resolved = await resolver.ResolveAllAsync(
            InferenceProvider.Cpu,
            CancellationToken.None);

        Assert.HasCount(1, resolved);
        Assert.AreEqual(TestStore.ModelId, resolved[0].Identity.ModelId);
        Assert.AreEqual(TestStore.Version, resolved[0].Identity.Version);
        Assert.AreEqual("marker_classifier", resolved[0].Task);
        Assert.AreEqual(store.ModelSha256, resolved[0].Identity.Sha256, ignoreCase: true);
        CollectionAssert.AreEqual(new[] { InferenceProvider.Cpu }, resolved[0].AvailableProviders.ToArray());
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
        store.RebindManifestChecksum();

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_MANIFEST_INVALID", error.Code);
    }

    [TestMethod]
    public async Task MalformedPackageIndexReturnsStablePackageCode()
    {
        using var store = TestStore.Create();
        File.WriteAllText(store.IndexPath, "{not-json");

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_PACKAGE_INDEX_INVALID", error.Code);
    }

    [TestMethod]
    [DataRow("identity")]
    [DataRow("path")]
    [DataRow("hash")]
    public async Task DuplicatePackageIndexPropertiesFailBeforeLastValueWinsAccess(string propertyKind)
    {
        using var store = TestStore.Create();
        store.InjectDuplicateIndexProperty(propertyKind);

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_PACKAGE_INDEX_INVALID", error.Code);
        StringAssert.Contains(error.Message, "duplicate JSON property");
    }

    [TestMethod]
    [DataRow("source")]
    [DataRow("license")]
    [DataRow("providers")]
    [DataRow("approval")]
    [DataRow("payload_hash")]
    public async Task DuplicateManifestPropertiesAtReviewedDepthsFailBeforeLastValueWinsAccess(string propertyKind)
    {
        using var store = TestStore.Create(includeAuxiliaryPayload: propertyKind == "payload_hash");
        store.InjectDuplicateManifestProperty(propertyKind);

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_MANIFEST_INVALID", error.Code);
        StringAssert.Contains(error.Message, "duplicate JSON property");
    }

    [TestMethod]
    [DataRow("manifest")]
    [DataRow("payload")]
    [DataRow("notice")]
    [DataRow("benchmark_evidence")]
    public async Task SafeChecksummedInRootResourceRelocationFailsClosed(string resourceKind)
    {
        using var store = TestStore.Create();
        store.RelocateIndexedResource(resourceKind);

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_PACKAGE_INDEX_INVALID", error.Code);
        StringAssert.Contains(error.Message, "must be canonical");
    }

    [TestMethod]
    public async Task CpuOnlyResolvedModelNeverAttemptsDirectMl()
    {
        using var store = TestStore.Create(providers: ["cpu"]);
        var resolver = store.CreateResolver(new FakeExecutionProviderDiscovery("DmlExecutionProvider", "CPUExecutionProvider"));
        var resolved = await resolver.ResolveAsync(
            TestStore.ModelId,
            TestStore.Version,
            requiredProvider: null,
            CancellationToken.None);
        var factory = new RecordingFactory();
        await using var registry = new OnnxSessionRegistry(
            new FakeExecutionProviderDiscovery("DmlExecutionProvider", "CPUExecutionProvider"),
            new WindowsExecutionProviderPolicy(),
            factory,
            CpuThreadConfiguration.Create(1, new FixedCoreDetector()));

        var acquisition = await registry.GetOrCreateAsync(resolved, CancellationToken.None);

        Assert.IsTrue(acquisition.Succeeded);
        CollectionAssert.AreEqual(new[] { InferenceProvider.Cpu }, factory.Attempts.ToArray());
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
    public async Task UnindexedManifestFileFailsClosed()
    {
        using var store = TestStore.Create();
        File.WriteAllText(Path.Combine(Path.GetDirectoryName(store.ManifestPath)!, "unindexed.json"), "{}");

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
    [DataRow("GPL-3.0-only")]
    [DataRow("AGPL-3.0-only")]
    [DataRow("LGPL-3.0-only")]
    [DataRow("SSPL-1.0")]
    [DataRow("BUSL-1.1")]
    [DataRow("CC-BY-NC-4.0")]
    [DataRow("LicenseRef-Unknown")]
    [DataRow("NOASSERTION")]
    public async Task ProhibitedOrUnclearLicensesFailClosed(string spdx)
    {
        using var store = TestStore.Create(licenseSpdx: spdx);

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_LICENSE_PROHIBITED", error.Code);
    }

    [TestMethod]
    public async Task UnrecognizedSpdxExpressionFailsClosed()
    {
        using var store = TestStore.Create(licenseSpdx: "Apache-2.0 OR MIT");

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_LICENSE_UNRECOGNIZED", error.Code);
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
    public async Task LockedPackageIndexReturnsStableIoFailure()
    {
        using var store = TestStore.Create();
        using var locked = new FileStream(store.IndexPath, FileMode.Open, FileAccess.Read, FileShare.None);

        var error = await ResolveFailureAsync(store);

        Assert.AreEqual("MODEL_STORE_IO_ERROR", error.Code);
    }

    [TestMethod]
    public async Task RootSymbolicLinkIsRejectedBeforeIndexOpen()
    {
        using var store = TestStore.Create();
        var link = store.Root + "-link";
        Directory.CreateSymbolicLink(link, store.Root);
        try
        {
            var resolver = new ProductionModelStore(link, new FakeExecutionProviderDiscovery("CPUExecutionProvider"));
            var error = await Assert.ThrowsExactlyAsync<ProductionModelValidationException>(async () =>
                await resolver.ResolveAsync(TestStore.ModelId, TestStore.Version, InferenceProvider.Cpu, CancellationToken.None));
            Assert.AreEqual("MODEL_STORE_REPARSE_POINT", error.Code);
        }
        finally
        {
            Directory.Delete(link);
        }
    }

    [TestMethod]
    public async Task ManifestSymbolicLinkIsRejectedBeforeOpen()
    {
        using var store = TestStore.Create();
        var external = store.Root + "-external-manifest.json";
        File.Move(store.ManifestPath, external);
        File.CreateSymbolicLink(store.ManifestPath, external);
        try
        {
            var error = await ResolveFailureAsync(store);
            Assert.AreEqual("MODEL_STORE_REPARSE_POINT", error.Code);
        }
        finally
        {
            File.Delete(external);
        }
    }

    [TestMethod]
    public async Task NestedPayloadDirectoryJunctionIsRejectedBeforeEnumeration()
    {
        using var store = TestStore.Create();
        var external = store.Root + "-external-runtime";
        Directory.Move(store.ModelDirectory, external);
        CreateJunction(store.ModelDirectory, external);
        try
        {
            var error = await ResolveFailureAsync(store);
            Assert.AreEqual("MODEL_STORE_REPARSE_POINT", error.Code);
        }
        finally
        {
            Directory.Delete(store.ModelDirectory);
            Directory.Delete(external, recursive: true);
        }
    }

    [TestMethod]
    public async Task NoticeSymbolicLinkIsRejectedBeforeOpen()
    {
        using var store = TestStore.Create();
        var external = store.Root + "-external-notice.txt";
        File.Move(store.NoticePath, external);
        File.CreateSymbolicLink(store.NoticePath, external);
        try
        {
            var error = await ResolveFailureAsync(store);
            Assert.AreEqual("MODEL_STORE_REPARSE_POINT", error.Code);
        }
        finally
        {
            File.Delete(external);
        }
    }

    [TestMethod]
    public async Task EvidenceSymbolicLinkIsRejectedBeforeOpen()
    {
        using var store = TestStore.Create();
        var external = store.Root + "-external-evidence.json";
        File.Move(store.BenchmarkPath, external);
        File.CreateSymbolicLink(store.BenchmarkPath, external);
        try
        {
            var error = await ResolveFailureAsync(store);
            Assert.AreEqual("MODEL_STORE_REPARSE_POINT", error.Code);
        }
        finally
        {
            File.Delete(external);
        }
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

    private static void CreateJunction(string junctionPath, string targetPath)
    {
        using var process = Process.Start(new ProcessStartInfo
        {
            FileName = "cmd.exe",
            UseShellExecute = false,
            CreateNoWindow = true,
            ArgumentList = { "/d", "/c", "mklink", "/J", junctionPath, targetPath }
        })!;
        process.WaitForExit();
        Assert.AreEqual(0, process.ExitCode, "Windows mklink /J must create the test junction.");
    }

    private sealed class FixedCoreDetector : IPhysicalCoreDetector
    {
        public int GetPhysicalCoreCount() => 1;
    }

    private sealed class RecordingFactory : IInferenceSessionFactory
    {
        private readonly FakeInferenceSessionFactory _inner = new();

        public List<InferenceProvider> Attempts { get; } = [];

        public ValueTask<IInferenceSession> CreateAsync(
            ModelIdentity model,
            InferenceProvider provider,
            CpuThreadConfiguration cpuConfiguration,
            CancellationToken cancellationToken)
        {
            Attempts.Add(provider);
            return _inner.CreateAsync(model, provider, cpuConfiguration, cancellationToken);
        }
    }

    private sealed class TestStore : IDisposable
    {
        public const string ModelId = "test-production-identity";
        public const string Version = "1.0.0";

        private TestStore(
            string root,
            string modelDirectory,
            string manifestPath,
            string indexPath,
            string modelPath,
            string modelSha256,
            string noticePath,
            string benchmarkPath,
            string? auxiliaryPath)
        {
            Root = root;
            ModelDirectory = modelDirectory;
            ManifestPath = manifestPath;
            IndexPath = indexPath;
            ModelPath = modelPath;
            ModelSha256 = modelSha256;
            NoticePath = noticePath;
            BenchmarkPath = benchmarkPath;
            AuxiliaryPath = auxiliaryPath;
        }

        public string Root { get; }
        public string ModelDirectory { get; }
        public string ManifestPath { get; }
        public string IndexPath { get; }
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
            bool includeAuxiliaryHash = true,
            string licenseSpdx = "Apache-2.0")
        {
            var root = Path.Combine(Path.GetTempPath(), "GraphReaderProductionModelStoreTests", Guid.NewGuid().ToString("N"));
            var modelDirectory = Path.Combine(root, "runtime", ModelId, Version);
            var manifestDirectory = Path.Combine(root, "manifest", ModelId, Version);
            var noticeDirectory = Path.Combine(root, "notices", ModelId, Version);
            var evidenceDirectory = Path.Combine(root, "evidence", ModelId, Version);
            Directory.CreateDirectory(modelDirectory);
            Directory.CreateDirectory(manifestDirectory);
            Directory.CreateDirectory(noticeDirectory);
            Directory.CreateDirectory(evidenceDirectory);

            using var generated = TestOnnxModel.CreateIdentity();
            var modelPath = Path.Combine(modelDirectory, "identity.onnx");
            File.Copy(generated.Path, modelPath);
            var modelBytes = File.ReadAllBytes(modelPath);
            var modelSha = Convert.ToHexString(SHA256.HashData(modelBytes));

            var noticePath = Path.Combine(noticeDirectory, "test-model.txt");
            File.WriteAllText(noticePath, "Apache-2.0 test model notice.");
            var noticeSha = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(noticePath)));
            var benchmarkPath = Path.Combine(evidenceDirectory, "test-model-benchmark.json");
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
                    ["spdx"] = licenseSpdx,
                    ["notice_path"] = "LICENSES/test-model.txt",
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
                        ["evidence_path"] = "artifacts/evidence/test-model-benchmark.json",
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

            var manifestPath = Path.Combine(manifestDirectory, "manifest.json");
            File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest));
            var manifestSha = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(manifestPath)));
            var packagePayloads = files.Select(file => new Dictionary<string, object?>
            {
                ["declared_path"] = file,
                ["path"] = $"runtime/{ModelId}/{Version}/{file}",
                ["sha256"] = file == "identity.onnx"
                    ? modelSha
                    : Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(modelDirectory, file))))
            }).ToArray();
            var packageIndex = new Dictionary<string, object?>
            {
                ["schema_version"] = 1,
                ["models"] = new[]
                {
                    new Dictionary<string, object?>
                    {
                        ["model_id"] = ModelId,
                        ["model_version"] = Version,
                        ["manifest"] = new Dictionary<string, object?>
                        {
                            ["path"] = $"manifest/{ModelId}/{Version}/manifest.json",
                            ["sha256"] = manifestSha
                        },
                        ["payloads"] = packagePayloads,
                        ["notice"] = new Dictionary<string, object?>
                        {
                            ["declared_path"] = "LICENSES/test-model.txt",
                            ["path"] = $"notices/{ModelId}/{Version}/test-model.txt",
                            ["sha256"] = noticeSha
                        },
                        ["benchmark_evidence"] = new Dictionary<string, object?>
                        {
                            ["declared_path"] = "artifacts/evidence/test-model-benchmark.json",
                            ["path"] = $"evidence/{ModelId}/{Version}/test-model-benchmark.json",
                            ["sha256"] = benchmarkSha
                        }
                    }
                }
            };
            var indexPath = Path.Combine(root, "production-model-index.json");
            File.WriteAllText(indexPath, JsonSerializer.Serialize(packageIndex));
            return new TestStore(
                root,
                modelDirectory,
                manifestPath,
                indexPath,
                modelPath,
                modelSha,
                noticePath,
                benchmarkPath,
                auxiliaryPath);
        }

        public ProductionModelStore CreateResolver(IExecutionProviderDiscovery? discovery = null) =>
            new(Root, discovery ?? new FakeExecutionProviderDiscovery("CPUExecutionProvider"));

        public void RebindManifestChecksum()
        {
            var index = JsonNode.Parse(File.ReadAllText(IndexPath))!;
            index["models"]![0]!["manifest"]!["sha256"] =
                Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(ManifestPath)));
            File.WriteAllText(IndexPath, index.ToJsonString());
        }

        public void RelocateIndexedResource(string resourceKind)
        {
            var index = JsonNode.Parse(File.ReadAllText(IndexPath))!;
            var model = index["models"]![0]!;
            var resource = resourceKind == "payload"
                ? model["payloads"]![0]!
                : model[resourceKind]!;
            var currentRelativePath = resource["path"]!.GetValue<string>();
            var currentPath = Path.Combine(Root, currentRelativePath.Replace('/', Path.DirectorySeparatorChar));
            var relocatedRelativePath = $"relocated/{resourceKind}/{Path.GetFileName(currentRelativePath)}";
            var relocatedPath = Path.Combine(Root, relocatedRelativePath.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(relocatedPath)!);
            File.Move(currentPath, relocatedPath);
            resource["path"] = relocatedRelativePath;
            File.WriteAllText(IndexPath, index.ToJsonString());
        }

        public void InjectDuplicateIndexProperty(string propertyKind)
        {
            var json = File.ReadAllText(IndexPath);
            json = propertyKind switch
            {
                "identity" => ReplaceFirst(
                    json,
                    $"{{\"model_id\":\"{ModelId}\"",
                    $"{{\"model_id\":\"duplicate-identity\",\"model_id\":\"{ModelId}\""),
                "path" => ReplaceFirst(
                    json,
                    $"\"manifest\":{{\"path\":\"manifest/{ModelId}/{Version}/manifest.json\"",
                    $"\"manifest\":{{\"path\":\"relocated/manifest.json\",\"path\":\"manifest/{ModelId}/{Version}/manifest.json\""),
                "hash" => ReplaceFirst(
                    json,
                    $"\"manifest\":{{\"path\":\"manifest/{ModelId}/{Version}/manifest.json\",\"sha256\":",
                    $"\"manifest\":{{\"path\":\"manifest/{ModelId}/{Version}/manifest.json\",\"sha256\":\"{new string('0', 64)}\",\"sha256\":"),
                _ => throw new ArgumentOutOfRangeException(nameof(propertyKind), propertyKind, null)
            };
            File.WriteAllText(IndexPath, json);
        }

        public void InjectDuplicateManifestProperty(string propertyKind)
        {
            var json = File.ReadAllText(ManifestPath);
            json = propertyKind switch
            {
                "source" => ReplaceFirst(
                    json,
                    "\"source\":{\"name\":",
                    "\"source\":{\"name\":\"duplicate source\",\"name\":"),
                "license" => ReplaceFirst(
                    json,
                    "\"license\":{\"spdx\":",
                    "\"license\":{\"spdx\":\"GPL-3.0-only\",\"spdx\":"),
                "providers" => ReplaceFirst(
                    json,
                    "\"providers\":",
                    "\"providers\":[\"directml\"],\"providers\":"),
                "approval" => ReplaceFirst(
                    json,
                    "\"production_approval\":",
                    "\"production_approval\":false,\"production_approval\":"),
                "payload_hash" => ReplaceFirst(
                    json,
                    "\"model_payload_sha256\":{\"identity.onnx\":",
                    $"\"model_payload_sha256\":{{\"identity.onnx\":\"{new string('0', 64)}\",\"identity.onnx\":"),
                _ => throw new ArgumentOutOfRangeException(nameof(propertyKind), propertyKind, null)
            };
            File.WriteAllText(ManifestPath, json);
            RebindManifestChecksum();
        }

        private static string ReplaceFirst(string value, string oldValue, string newValue)
        {
            var index = value.IndexOf(oldValue, StringComparison.Ordinal);
            if (index < 0)
            {
                throw new InvalidOperationException($"Test JSON marker was not found: {oldValue}");
            }

            return string.Concat(value.AsSpan(0, index), newValue, value.AsSpan(index + oldValue.Length));
        }

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
