// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Inference.Tests;

/// <summary>
/// Opt-in verification for the ignored production model store assembled from
/// checksum-bound local candidate bytes. Ordinary tests require no model files.
/// </summary>
[TestClass]
public sealed class ProductionMarkerClassifierStoreIntegrationTests
{
    private const string StoreEnvironmentVariable = "GRAPHREADER_PRODUCTION_MODEL_STORE";
    private const string ExpectedSha256 =
        "26f9304f1689053a0b94aa896a1e239f6ade1e5c1920736a3535c1b32f803b8a";

    [TestMethod]
    public async Task ExactClassifierResolvesForCpuAndDirectMl()
    {
        string? storePath = Environment.GetEnvironmentVariable(StoreEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(storePath))
        {
            Assert.Inconclusive(
                $"Set {StoreEnvironmentVariable} to the ignored production model store to run this probe.");
        }

        var discovery = new OrtExecutionProviderDiscovery();
        var store = new ProductionModelStore(Path.GetFullPath(storePath), discovery);
        ResolvedProductionModel cpu = await store.ResolveAsync(
            "graph-marker-classifier",
            "0.1.0",
            InferenceProvider.Cpu,
            CancellationToken.None);
        ResolvedProductionModel directMl = await store.ResolveAsync(
            "graph-marker-classifier",
            "0.1.0",
            InferenceProvider.DirectMl,
            CancellationToken.None);

        Assert.AreEqual(ExpectedSha256, cpu.Identity.Sha256.ToLowerInvariant());
        Assert.AreEqual(cpu.Identity, directMl.Identity);
        Assert.Contains(InferenceProvider.Cpu, cpu.AvailableProviders);
        Assert.Contains(InferenceProvider.DirectMl, cpu.AvailableProviders);
        Assert.IsTrue(File.Exists(cpu.Identity.FilePath));
        Assert.IsTrue(File.Exists(cpu.ManifestPath));
        Assert.IsTrue(File.Exists(cpu.NoticePath));
        Assert.IsTrue(File.Exists(cpu.BenchmarkEvidencePath));
    }
}
