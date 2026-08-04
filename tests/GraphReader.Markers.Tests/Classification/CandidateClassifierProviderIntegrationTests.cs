// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Classification;

/// <summary>
/// Opt-in verification for the ignored, locally trained classifier candidate.
/// Ordinary test runs remain independent of generated model weights.
/// </summary>
[TestClass]
public sealed class CandidateClassifierProviderIntegrationTests
{
    private const string CandidateEnvironmentVariable = "GRAPHREADER_MARKER_CLASSIFIER_CANDIDATE";
    private const string CandidateSha256 =
        "59b4af98fe40abd436f01a8c14bf0d12a7c82682ec072c65cef92881aa18b0ef";

    public TestContext TestContext { get; set; } = null!;

    [TestMethod]
    public async Task ExactCandidateExecutesCpuAndDirectMlWithinTolerance()
    {
        string? modelPath = Environment.GetEnvironmentVariable(CandidateEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            Assert.Inconclusive(
                $"Set {CandidateEnvironmentVariable} to the ignored packed classifier ONNX to run this candidate probe.");
        }

        modelPath = Path.GetFullPath(modelPath);
        Assert.IsTrue(File.Exists(modelPath), $"Candidate model does not exist: {modelPath}");
        var discovery = new OrtExecutionProviderDiscovery();
        IReadOnlyList<string> available = discovery.GetAvailableProviders();
        CollectionAssert.Contains(available.ToArray(), "CPUExecutionProvider");
        CollectionAssert.Contains(available.ToArray(), "DmlExecutionProvider");

        ModelIdentity model = new(
            "graph-marker-classifier",
            "0.1.0",
            CandidateSha256,
            modelPath);
        InferenceRequest request = CreateRequest(model);
        await using InferenceRuntime cpuRuntime = CreateRuntime(new CpuOnlyDiscovery());
        await using InferenceRuntime directMlRuntime = CreateRuntime(discovery);

        InferenceResponse cpu = await cpuRuntime.RunAsync(request, CancellationToken.None);
        InferenceResponse directMl = await directMlRuntime.RunAsync(request, CancellationToken.None);

        Assert.IsTrue(cpu.Succeeded, cpu.Error?.TechnicalMessage);
        Assert.IsTrue(directMl.Succeeded, directMl.Error?.TechnicalMessage);
        Assert.AreEqual(InferenceProvider.Cpu, cpu.Execution?.Provider);
        Assert.AreEqual(InferenceProvider.DirectMl, directMl.Execution?.Provider);
        Assert.HasCount(50, cpu.Execution!.Output);
        Assert.HasCount(50, directMl.Execution!.Output);
        double maximumDifference = cpu.Execution.Output
            .Zip(directMl.Execution.Output, static (left, right) => Math.Abs(left - right))
            .Max();
        Assert.IsLessThanOrEqualTo(1e-4, maximumDifference);

        TestContext.WriteLine("available_providers=" + string.Join(',', available));
        TestContext.WriteLine($"cpu_inference_ms={cpu.Execution.Timing.InferenceMilliseconds:F4}");
        TestContext.WriteLine($"directml_inference_ms={directMl.Execution.Timing.InferenceMilliseconds:F4}");
        TestContext.WriteLine($"maximum_absolute_difference={maximumDifference:R}");
    }

    private static InferenceRequest CreateRequest(ModelIdentity model)
    {
        const int batch = 2;
        const int width = 32;
        const int height = 32;
        var values = new float[batch * width * height];
        var random = new Random(20260803);
        for (var index = 0; index < values.Length; index++)
        {
            values[index] = (float)random.NextDouble();
        }

        return new InferenceRequest(
            model,
            new InferenceInput(
                values,
                Array.AsReadOnly<long>([batch, 1, height, width]),
                "marker_patch",
                "classification_heads"),
            new StageCacheMaterial(
                new string('a', 64),
                "candidate-provider-probe",
                "identity",
                "markers",
                "0.1.0",
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["batch"] = batch,
                    ["seed"] = 20260803,
                },
                1),
            TimeSpan.FromSeconds(30));
    }

    private static InferenceRuntime CreateRuntime(IExecutionProviderDiscovery discovery)
    {
        var registry = new OnnxSessionRegistry(
            discovery,
            new WindowsExecutionProviderPolicy(),
            new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance),
            CpuThreadConfiguration.Create(1, new SingleCoreDetector()));
        return new InferenceRuntime(
            registry,
            new BoundedInferenceScheduler(2, 1),
            new NoOpStageCache());
    }

    private sealed class CpuOnlyDiscovery : IExecutionProviderDiscovery
    {
        public IReadOnlyList<string> GetAvailableProviders() => ["CPUExecutionProvider"];
    }

    private sealed class SingleCoreDetector : IPhysicalCoreDetector
    {
        public int GetPhysicalCoreCount() => 1;
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
