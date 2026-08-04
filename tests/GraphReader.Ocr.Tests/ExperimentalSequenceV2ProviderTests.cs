// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

/// <summary>
/// Opt-in provider probe for the quality-failed V2 sequence experiment.
/// Runtime compatibility does not imply model approval.
/// </summary>
[TestClass]
public sealed class ExperimentalSequenceV2ProviderTests
{
    private const string ModelEnvironmentVariable = "GRAPHREADER_OCR_SEQUENCE_V2_EXPERIMENTAL";
    private const string ModelSha256 =
        "e7f31d5065f92be6142cd1c17814364646e35d612d5bf0750ed3e403b3b08e3c";

    public TestContext TestContext { get; set; } = null!;

    [TestMethod]
    public async Task ExactFailedExperimentExecutesCpuAndDirectMlWithinTolerance()
    {
        string? modelPath = Environment.GetEnvironmentVariable(ModelEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            Assert.Inconclusive(
                $"Set {ModelEnvironmentVariable} to the ignored V2 ONNX to run this provider probe.");
        }

        modelPath = Path.GetFullPath(modelPath);
        Assert.IsTrue(File.Exists(modelPath), $"Experimental model does not exist: {modelPath}");
        var discovery = new OrtExecutionProviderDiscovery();
        IReadOnlyList<string> available = discovery.GetAvailableProviders();
        CollectionAssert.Contains(available.ToArray(), "CPUExecutionProvider");
        CollectionAssert.Contains(available.ToArray(), "DmlExecutionProvider");

        ModelIdentity model = new(
            "graph-numeric-sequence-v2-experimental",
            "0.1.0-goal19-failed",
            ModelSha256,
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
        Assert.HasCount(2 * 32 * 14, cpu.Execution!.Output);
        Assert.HasCount(2 * 32 * 14, directMl.Execution!.Output);
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
        const int height = 32;
        const int width = 128;
        var values = new float[batch * height * width];
        var random = new Random(20260804);
        for (var index = 0; index < values.Length; index++)
        {
            values[index] = (float)((random.NextDouble() - 0.5) * 2.0);
        }

        return new InferenceRequest(
            model,
            new InferenceInput(
                values,
                Array.AsReadOnly<long>([batch, 1, height, width]),
                "input",
                "output"),
            new StageCacheMaterial(
                new string('c', 64),
                "goal19-failed-sequence-v2-provider-probe",
                "identity",
                "ocr_recognition",
                "0.1.0",
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["batch"] = batch,
                    ["seed"] = 20260804,
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
