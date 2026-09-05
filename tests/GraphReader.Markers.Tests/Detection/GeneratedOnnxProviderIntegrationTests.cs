// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using GraphReader.Markers.Detection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Detection;

/// <summary>
/// Exercises the real ONNX Runtime CPU and DirectML paths with a generated identity graph
/// conforming to the marker v2 tensor contract. This proves runtime-contract/provider parity,
/// not trained marker-model accuracy or visual detection behavior.
/// </summary>
[TestClass]
public sealed class GeneratedOnnxProviderIntegrationTests
{
    public TestContext TestContext { get; set; } = null!;

    [TestMethod]
    public async Task ActualCpuExecutesMarkerTensorContract()
    {
        using var modelFile = GeneratedMarkerContractOnnx.CreateIdentity(64, 64);
        await using InferenceRuntime runtime = CreateRuntime(new CpuOnlyDiscovery());
        MarkerDetectionResult result = await new MarkerCenterDetector(runtime)
            .DetectAsync(IdentityContractRequest(modelFile), CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.AreEqual(InferenceProvider.Cpu, result.Model.Provider);
        Assert.HasCount(2, result.Markers);
        Assert.IsTrue(result.Markers.All(marker => marker.ReviewState == MarkerReviewState.Unreviewed));
    }

    [TestMethod]
    public async Task ActualCpuAndDirectMlExecuteMarkerTensorContractWithParity()
    {
        var discovery = new OrtExecutionProviderDiscovery();
        IReadOnlyList<string> available = discovery.GetAvailableProviders();
        CollectionAssert.Contains(available.ToArray(), "CPUExecutionProvider");
        if (!available.Contains("DmlExecutionProvider", StringComparer.Ordinal))
        {
            Assert.Inconclusive("DirectML is not installed on this test host.");
        }
        using var modelFile = GeneratedMarkerContractOnnx.CreateIdentity(64, 64);
        await using InferenceRuntime cpuRuntime = CreateRuntime(new CpuOnlyDiscovery());
        await using InferenceRuntime directMlRuntime = CreateRuntime(discovery);
        MarkerDetectionRequest request = IdentityContractRequest(modelFile);

        MarkerDetectionResult cpu = await new MarkerCenterDetector(cpuRuntime)
            .DetectAsync(request, CancellationToken.None);
        MarkerDetectionResult directMl = await new MarkerCenterDetector(directMlRuntime)
            .DetectAsync(request, CancellationToken.None);

        Assert.IsTrue(cpu.Succeeded, cpu.Failure?.TechnicalMessage);
        Assert.IsTrue(directMl.Succeeded, directMl.Failure?.TechnicalMessage);
        Assert.AreEqual(InferenceProvider.Cpu, cpu.Model.Provider);
        if (directMl.Model.Provider == InferenceProvider.Cpu)
        {
            Assert.Inconclusive("DirectML execution was unavailable on this host; the runtime used its CPU fallback.");
        }
        Assert.AreEqual(InferenceProvider.DirectMl, directMl.Model.Provider);
        Assert.HasCount(2, cpu.Markers);
        Assert.HasCount(2, directMl.Markers);
        CollectionAssert.AreEqual(
            cpu.Markers.Select(Signature).ToArray(),
            directMl.Markers.Select(Signature).ToArray());
        Assert.IsTrue(cpu.Markers.All(marker => marker.ReviewState == MarkerReviewState.Unreviewed));
        Assert.IsTrue(directMl.Markers.All(marker => marker.ReviewState == MarkerReviewState.Unreviewed));
        TestContext.WriteLine("available_providers=" + string.Join(",", available));
        TestContext.WriteLine(
            $"cpu_ms={cpu.Timing.TotalMilliseconds:F3}; directml_ms={directMl.Timing.TotalMilliseconds:F3}");
    }

    [TestMethod]
    public async Task ActualCpuExecutesAfterInjectedDirectMlAcquisitionFailure()
    {
        using var modelFile = GeneratedMarkerContractOnnx.CreateIdentity(64, 64);
        await using InferenceRuntime runtime = CreateRuntime(
            new OrtExecutionProviderDiscovery(),
            new RejectDirectMlFactory(new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance)));

        MarkerDetectionResult result = await new MarkerCenterDetector(runtime).DetectAsync(
            IdentityContractRequest(modelFile),
            CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.AreEqual(InferenceProvider.Cpu, result.Model.Provider);
        Assert.HasCount(2, result.Markers);
        Assert.HasCount(2, result.Frames[0].ProviderAttempts);
        Assert.AreEqual(InferenceProvider.DirectMl, result.Frames[0].ProviderAttempts[0].Provider);
        Assert.IsFalse(result.Frames[0].ProviderAttempts[0].Succeeded);
        Assert.AreEqual(InferenceProvider.Cpu, result.Frames[0].ProviderAttempts[1].Provider);
        Assert.IsTrue(result.Frames[0].ProviderAttempts[1].Succeeded);
    }

    private static MarkerDetectionRequest IdentityContractRequest(GeneratedMarkerContractOnnx modelFile)
    {
        MarkerPoint[] centers = [new(14, 18), new(44, 38)];
        var luminance = Enumerable.Repeat(1f, 64 * 64).ToArray();
        var radiusPlane = new float[64 * 64];
        foreach (MarkerPoint center in centers)
        {
            int index = ((int)center.Y * 64) + (int)center.X;
            luminance[index] = 0.05f;
            radiusPlane[index] = 0.75f;
        }

        var frame = new MarkerImageFrame(
            64,
            64,
            1,
            luminance,
            MarkerSourceImage.Original,
            MarkerAffineTransform.Identity,
            new MarkerMask(64, 64, radiusPlane),
            MarkerMask.Empty(64, 64));
        MarkerDetectionOptions options = MarkerDetectionTestSupport.Options() with
        {
            CenterThreshold = 0.9f,
            ArtifactThreshold = 1,
            MaskThreshold = 1,
        };
        var model = new ModelIdentity(
            "generated.marker-contract.identity",
            "1",
            modelFile.Sha256,
            modelFile.Path);
        return MarkerDetectionTestSupport.Request(
            options: options,
            original: frame,
            model: model,
            transformChain: "generated-identity-contract");
    }

    private static InferenceRuntime CreateRuntime(
        IExecutionProviderDiscovery discovery,
        IInferenceSessionFactory? factory = null)
    {
        var registry = new OnnxSessionRegistry(
            discovery,
            new WindowsExecutionProviderPolicy(),
            factory ?? new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance),
            CpuThreadConfiguration.Create(1, new SingleCoreDetector()));
        return new InferenceRuntime(
            registry,
            new BoundedInferenceScheduler(2, 1),
            new NoOpStageCache());
    }

    private static string Signature(MarkerCenter marker) => string.Join(
        "|",
        marker.Center.X.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        marker.Center.Y.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        marker.Radius.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        marker.CenterConfidence.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        marker.ArtifactProbability.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        marker.ReviewState,
        marker.Disagreement);

    private sealed class CpuOnlyDiscovery : IExecutionProviderDiscovery
    {
        public IReadOnlyList<string> GetAvailableProviders() => ["CPUExecutionProvider"];
    }

    private sealed class SingleCoreDetector : IPhysicalCoreDetector
    {
        public int GetPhysicalCoreCount() => 1;
    }

    private sealed class RejectDirectMlFactory(IInferenceSessionFactory inner) : IInferenceSessionFactory
    {
        public ValueTask<IInferenceSession> CreateAsync(
            ModelIdentity model,
            InferenceProvider provider,
            CpuThreadConfiguration cpuConfiguration,
            CancellationToken cancellationToken) =>
            provider == InferenceProvider.DirectMl
                ? ValueTask.FromException<IInferenceSession>(
                    new InvalidOperationException("Injected DirectML acquisition failure."))
                : inner.CreateAsync(model, provider, cpuConfiguration, cancellationToken);
    }

    private sealed class NoOpStageCache : IStageCache
    {
        public ValueTask<byte[]?> TryGetAsync(StageCacheKey key, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<byte[]?>(null);
        }

        public ValueTask PutAsync(
            StageCacheKey key,
            ReadOnlyMemory<byte> value,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.CompletedTask;
        }
    }
}
