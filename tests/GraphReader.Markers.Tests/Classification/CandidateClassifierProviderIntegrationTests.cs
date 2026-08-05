// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using GraphReader.Markers.Classification;
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
        "26f9304f1689053a0b94aa896a1e239f6ade1e5c1920736a3535c1b32f803b8a";

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
        using (FileStream stream = File.OpenRead(modelPath))
        {
            string actualSha256 = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream))
                .ToLowerInvariant();
            Assert.AreEqual(CandidateSha256, actualSha256);
        }

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
        AssertProbabilityContract(cpu.Execution.Output, batchCount: 2);
        AssertProbabilityContract(directMl.Execution.Output, batchCount: 2);
        double maximumDifference = cpu.Execution.Output
            .Zip(directMl.Execution.Output, static (left, right) => Math.Abs(left - right))
            .Max();
        Assert.IsLessThanOrEqualTo(1e-4, maximumDifference);

        var classificationContract = new MarkerClassifierTensorContract(
            "marker_patch",
            "classification_probabilities",
            32,
            32,
            1,
            12)
        {
            OutputEncoding = MarkerClassifierOutputEncoding.Probabilities,
        };
        var classificationRequest = new MarkerClassificationRequest(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            new string('a', 64),
            model,
            ClassificationTestSupport.Frame(),
            [ClassificationTestSupport.Marker("candidate-probe", 16, 16)],
            new MarkerClassificationOptions(classificationContract));
        var classificationService = new MarkerClassificationService(cpuRuntime);
        MarkerClassificationResult classification = await classificationService.ClassifyAsync(
            classificationRequest,
            CancellationToken.None);
        Assert.IsTrue(classification.Succeeded, classification.Failure?.TechnicalMessage);
        Assert.HasCount(1, classification.Markers);

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
                "classification_probabilities"),
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

    private static void AssertProbabilityContract(IReadOnlyList<float> output, int batchCount)
    {
        const int valuesPerMarker = 25;
        for (var batchIndex = 0; batchIndex < batchCount; batchIndex++)
        {
            int offset = batchIndex * valuesPerMarker;
            double shapeTotal = output.Skip(offset).Take(9).Sum(value => (double)value);
            double fillTotal = output.Skip(offset + 9).Take(3).Sum(value => (double)value);
            Assert.AreEqual(1, shapeTotal, 1e-4);
            Assert.AreEqual(1, fillTotal, 1e-4);
            Assert.IsGreaterThanOrEqualTo(0, output[offset + 12]);
            Assert.IsLessThanOrEqualTo(1, output[offset + 12]);
        }
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
