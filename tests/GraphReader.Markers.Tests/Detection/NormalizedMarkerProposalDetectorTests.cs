// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Inference;
using GraphReader.Markers.Detection;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Security.Cryptography;

namespace GraphReader.Markers.Tests.Detection;

[TestClass]
public sealed class NormalizedMarkerProposalDetectorTests
{
    private static readonly ModelIdentity Model = new(
        "graph-marker-center-normalized-v4-p1",
        "0.1.0-candidate",
        "017fca04fa3817596ce3088d73f51003dd3658bc56ec3130e25c92252e6bf739",
        "marker-center-normalized-training-p1.onnx");

    public TestContext TestContext { get; set; } = null!;

    [TestMethod]
    public async Task CandidateRuntimeBindsExactTensorContractAndProducesOriginalPixelCenter()
    {
        MarkerImageFrame frame = FilledMarkerFrame();
        NormalizedMarkerProposalBatch proposals =
            NormalizedMarkerProposalPreprocessor.Prepare(frame, CancellationToken.None);
        int centerIndex = CoordinateIndex(proposals, 16, 16);
        InferenceRequest? captured = null;
        var runner = new MarkerInferenceRunnerStub((request, _) =>
        {
            captured = request;
            float[] output = Output(proposals.Count);
            Set(output, centerIndex, 0.9f, 0.25f, -0.25f, 4f);
            return ValueTask.FromResult(Success(output));
        });

        MarkerDetectionResult result = await new NormalizedMarkerProposalDetector(runner)
            .DetectAsync(Request(frame), CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Markers);
        Assert.AreEqual(new MarkerPoint(17, 15), result.Markers[0].Center);
        Assert.AreEqual(4, result.Markers[0].Radius, 0);
        Assert.AreEqual(0.9, result.Markers[0].CenterConfidence, 0.000001);
        Assert.AreEqual(InferenceProvider.Cpu, result.Model.Provider);
        Assert.IsNotNull(captured);
        CollectionAssert.AreEqual(new long[] { proposals.Count, 3, 33, 33 }, captured.Input.Shape.ToArray());
        Assert.AreEqual(NormalizedMarkerProposalContract.InputName, captured.Input.InputName);
        Assert.AreEqual(NormalizedMarkerProposalPostprocessContract.OutputName, captured.Input.OutputName);
        CollectionAssert.AreEqual(new[] { InferenceProvider.Cpu }, captured.AllowedProviders!.ToArray());
        Assert.AreEqual(
            proposals.TensorSha256,
            captured.CacheMaterial.Parameters["proposal_tensor_sha256"]);
        Assert.AreEqual(
            NormalizedMarkerProposalContract.PreprocessRevision,
            captured.CacheMaterial.Parameters["preprocess_revision"]);
        Assert.AreEqual(
            NormalizedMarkerProposalPostprocessContract.Revision,
            captured.CacheMaterial.Parameters["postprocess_revision"]);
        Assert.AreEqual(InferenceCacheKeyDeriver.Derive(captured).Value, result.Frames[0].CacheKey);
        Assert.AreEqual(1, result.Frames[0].RawCandidateCount);
        Assert.AreEqual(1, result.Frames[0].AcceptedCandidateCount);
    }

    [TestMethod]
    public async Task FrozenPythonRefinementFixtureMovesToTheSameOnePixelNeighbor()
    {
        const int size = 33;
        float[] ink = new float[size * size];
        ink[(16 * size) + 14] = 1;
        ink[(16 * size) + 20] = 1;
        ink[(13 * size) + 17] = 1;
        MarkerImageFrame frame = FrameFromInk(size, ink);
        NormalizedMarkerProposalBatch proposals =
            NormalizedMarkerProposalPreprocessor.Prepare(frame, CancellationToken.None);
        int centerIndex = CoordinateIndex(proposals, 16, 16);
        var runner = new MarkerInferenceRunnerStub((_, _) =>
        {
            float[] output = Output(proposals.Count);
            Set(output, centerIndex, 0.9f, 0, 0, 3);
            return ValueTask.FromResult(Success(output));
        });

        MarkerDetectionResult result = await new NormalizedMarkerProposalDetector(runner)
            .DetectAsync(Request(frame), CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Markers);
        Assert.AreEqual(new MarkerPoint(17, 16), result.Markers[0].Center);
        Assert.AreEqual(3, result.Markers[0].Radius, 0);
    }

    [TestMethod]
    public async Task RadiusAwareSuppressionKeepsOnlyTheHighestConfidenceDuplicate()
    {
        MarkerImageFrame frame = FilledMarkerFrame();
        NormalizedMarkerProposalBatch proposals =
            NormalizedMarkerProposalPreprocessor.Prepare(frame, CancellationToken.None);
        int leftIndex = CoordinateIndex(proposals, 12, 16);
        int centerIndex = CoordinateIndex(proposals, 16, 16);
        var runner = new MarkerInferenceRunnerStub((_, _) =>
        {
            float[] output = Output(proposals.Count);
            Set(output, leftIndex, 0.91f, 0.75f, 0, 4);
            Set(output, centerIndex, 0.90f, -0.25f, 0, 4);
            return ValueTask.FromResult(Success(output));
        });

        MarkerDetectionResult result = await new NormalizedMarkerProposalDetector(runner)
            .DetectAsync(Request(frame), CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.HasCount(1, result.Markers);
        Assert.AreEqual(new MarkerPoint(15, 16), result.Markers[0].Center);
        Assert.AreEqual(0.91, result.Markers[0].CenterConfidence, 0.000001);
        Assert.AreEqual(2, result.Frames[0].RawCandidateCount);
        Assert.AreEqual(1, result.Frames[0].AcceptedCandidateCount);
    }

    [TestMethod]
    public async Task InvalidOutputAndNonCpuProviderFailClosed()
    {
        MarkerImageFrame frame = FilledMarkerFrame();
        var invalidOutput = new NormalizedMarkerProposalDetector(
            new MarkerInferenceRunnerStub(Success([0f])));

        MarkerDetectionResult shapeFailure = await invalidOutput.DetectAsync(
            Request(frame),
            CancellationToken.None);

        Assert.IsFalse(shapeFailure.Succeeded);
        Assert.AreEqual("MARKER_MODEL_OUTPUT_SHAPE_MISMATCH", shapeFailure.Failure?.Code);
        Assert.HasCount(1, shapeFailure.Frames);
        Assert.AreEqual(InferenceProvider.Cpu, shapeFailure.Frames[0].Provider);
        Assert.AreEqual("MARKER_MODEL_OUTPUT_SHAPE_MISMATCH", shapeFailure.Frames[0].Failure?.Code);

        NormalizedMarkerProposalBatch proposals =
            NormalizedMarkerProposalPreprocessor.Prepare(frame, CancellationToken.None);
        var wrongProvider = new NormalizedMarkerProposalDetector(
            new MarkerInferenceRunnerStub(Success(Output(proposals.Count), InferenceProvider.DirectMl)));

        MarkerDetectionResult providerFailure = await wrongProvider.DetectAsync(
            Request(frame),
            CancellationToken.None);

        Assert.IsFalse(providerFailure.Succeeded);
        Assert.AreEqual("MARKER_PROVIDER_UNAPPROVED", providerFailure.Failure?.Code);
        Assert.HasCount(1, providerFailure.Frames);
        Assert.AreEqual(InferenceProvider.DirectMl, providerFailure.Frames[0].Provider);
    }

    [TestMethod]
    public async Task RuntimeFailurePreservesCacheAndProviderAttemptEvidence()
    {
        MarkerImageFrame frame = FilledMarkerFrame();
        var detector = new NormalizedMarkerProposalDetector(
            new MarkerInferenceRunnerStub(MarkerDetectionTestSupport.Failure("CPU_RUNTIME_FAILED")));

        MarkerDetectionResult result = await detector.DetectAsync(
            Request(frame),
            CancellationToken.None);

        Assert.IsFalse(result.Succeeded);
        Assert.AreEqual("CPU_RUNTIME_FAILED", result.Failure?.Code);
        Assert.HasCount(1, result.Frames);
        Assert.AreNotEqual(string.Empty, result.Frames[0].CacheKey);
        Assert.HasCount(1, result.Frames[0].ProviderAttempts);
        Assert.AreEqual(InferenceProvider.Cpu, result.Frames[0].ProviderAttempts[0].Provider);
        Assert.IsFalse(result.Frames[0].ProviderAttempts[0].Succeeded);
        Assert.AreEqual("CPU_RUNTIME_FAILED", result.Frames[0].Failure?.Code);
    }

    [TestMethod]
    public async Task ContractDriftAndCancellationStopBeforeUsableResult()
    {
        MarkerImageFrame frame = FilledMarkerFrame();
        var runner = new MarkerInferenceRunnerStub(Success([]));
        NormalizedMarkerProposalDetectionRequest drifted = Request(frame) with
        {
            Options = new NormalizedMarkerProposalDetectionOptions { MarkerThreshold = 0.59f },
        };

        MarkerDetectionResult invalid = await new NormalizedMarkerProposalDetector(runner)
            .DetectAsync(drifted, CancellationToken.None);

        Assert.IsFalse(invalid.Succeeded);
        Assert.AreEqual("MARKER_REQUEST_INVALID", invalid.Failure?.Code);
        Assert.IsEmpty(runner.Requests);

        var cancellation = new CancellationToken(canceled: true);
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            new NormalizedMarkerProposalDetector(runner)
                .DetectAsync(Request(frame), cancellation)
                .AsTask());
        Assert.IsEmpty(runner.Requests);
    }

    [TestMethod]
    public async Task ExactCandidateExecutesThroughCpuProposalRuntimeWhenExplicitlyProvided()
    {
        string? modelPath = Environment.GetEnvironmentVariable(
            "GRAPHREADER_MARKER_NORMALIZED_V4_ONNX_PATH");
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
        {
            Assert.Inconclusive(
                "Set GRAPHREADER_MARKER_NORMALIZED_V4_ONNX_PATH to the ignored exact P1 ONNX payload.");
        }

        string fullPath = Path.GetFullPath(modelPath);
        string hash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(fullPath)));
        Assert.AreEqual(Model.Sha256, hash, "The opt-in payload must be the exact sealed P1 ONNX.");
        await using InferenceRuntime runtime = CreateCpuRuntime();
        MarkerImageFrame frame = FilledMarkerFrame();
        NormalizedMarkerProposalDetectionRequest request = Request(frame) with
        {
            Model = Model with { FilePath = fullPath },
        };

        MarkerDetectionResult result = await new NormalizedMarkerProposalDetector(runtime)
            .DetectAsync(request, CancellationToken.None);

        Assert.IsTrue(result.Succeeded, result.Failure?.TechnicalMessage);
        Assert.AreEqual(InferenceProvider.Cpu, result.Model.Provider);
        Assert.HasCount(1, result.Frames);
        Assert.IsTrue(result.Frames[0].RawCandidateCount >= result.Frames[0].AcceptedCandidateCount);
        TestContext.WriteLine(
            $"cpu_ms={result.Timing.TotalMilliseconds:F3}; proposals_cache={result.Frames[0].CacheKey}; accepted={result.Markers.Count}");
    }

    private static NormalizedMarkerProposalDetectionRequest Request(MarkerImageFrame frame) =>
        new(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            new string('b', 64),
            Model,
            frame,
            new NormalizedMarkerProposalDetectionOptions());

    private static MarkerImageFrame FilledMarkerFrame()
    {
        const int size = 33;
        float[] ink = Enumerable.Repeat(0.04f, size * size).ToArray();
        for (var y = 15; y < 18; y++)
        {
            for (var x = 15; x < 18; x++)
            {
                ink[(y * size) + x] = 0.90f;
            }
        }

        return FrameFromInk(size, ink);
    }

    private static MarkerImageFrame FrameFromInk(int size, float[] ink)
    {
        float[] luminance = ink.Select(static value => 1f - value).ToArray();
        return new MarkerImageFrame(
            size,
            size,
            1,
            luminance,
            MarkerSourceImage.Original,
            MarkerAffineTransform.Identity,
            MarkerMask.Empty(size, size),
            MarkerMask.Empty(size, size));
    }

    private static int CoordinateIndex(NormalizedMarkerProposalBatch proposals, int x, int y) =>
        proposals.Coordinates
            .Select((point, index) => (point, index))
            .Single(item => item.point == new MarkerPoint(x, y))
            .index;

    private static float[] Output(int proposalCount)
    {
        var output = new float[proposalCount * 4];
        for (var index = 0; index < proposalCount; index++)
        {
            output[(index * 4) + 3] = 2.5f;
        }

        return output;
    }

    private static void Set(
        float[] output,
        int index,
        float probability,
        float offsetX,
        float offsetY,
        float radius)
    {
        int offset = index * 4;
        output[offset] = probability;
        output[offset + 1] = offsetX;
        output[offset + 2] = offsetY;
        output[offset + 3] = radius;
    }

    private static InferenceResponse Success(
        IReadOnlyList<float> output,
        InferenceProvider provider = InferenceProvider.Cpu) =>
        new(
            true,
            new InferenceExecution(
                output,
                provider,
                new StageTiming(0, 1.25, 0, 1.25, 0, false, false),
                new MemoryDiagnostics(0, 0, 0, 0, 0)),
            null,
            [new ProviderAttempt(provider, true, null)]);

    private static InferenceRuntime CreateCpuRuntime()
    {
        var registry = new OnnxSessionRegistry(
            new CpuOnlyDiscovery(),
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
