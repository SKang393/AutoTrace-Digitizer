// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Runtime.InteropServices;
using GraphReader.App.Integration.Workflow;
using GraphReader.Inference;
using GraphReader.Markers.Detection;
using GraphReader.Ocr;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class ProductionMarkerArtifactMaskAdapterTests
{
    [TestMethod]
    public async Task ApprovedAdapterMapsDenseArtifactHeadAndPreservesSeedMask()
    {
        TestContextData context = CreateContext();
        var output = new float[]
        {
            0.1f, 0.1f, 0.1f, 0.1f,
            1f, 1f, 1f, 1f,
            0f, 1f, 0f, 0f,
        };
        var runner = new Runner(Success(output, InferenceProvider.Cpu));
        var adapter = CreateAdapter(isApproved: true, runner);

        ProductionArtifactMaskEvidence evidence = await adapter.DetectAsync(
            context.Request,
            context.Raster,
            context.Seed,
            CancellationToken.None);

        Assert.AreEqual(1, runner.CallCount);
        Assert.IsNotNull(runner.LastRequest);
        CollectionAssert.AreEqual(
            new long[] { 1, 3, 2, 2 },
            runner.LastRequest.Input.Shape.ToArray());
        Assert.AreEqual("marker_artifact_mask", runner.LastRequest.CacheMaterial.StageName);
        Assert.AreEqual("markers", evidence.Envelope.Stage);
        Assert.AreEqual("cpu", evidence.Envelope.Model?.Provider);
        Assert.AreEqual(context.Request.Image.Sha256, evidence.RasterSha256);
        Assert.AreEqual(1f, evidence.CopyMask().Values.Span[15]);
        Assert.AreEqual(1f, evidence.CopyMask().Values.Span[3]);
        Assert.IsTrue(evidence.CopyMask().Values.ToArray().Any(static value => value > 0));
        CollectionAssert.Contains(
            evidence.Warnings.ToArray(),
            "artifact_mask_scope:marker_center_artifact_head_full_frame_seeded_by_ocr_axis");
    }

    [TestMethod]
    public async Task UnapprovedAdapterFailsBeforeInference()
    {
        TestContextData context = CreateContext();
        var runner = new Runner(Success(new float[12], InferenceProvider.Cpu));
        var adapter = CreateAdapter(isApproved: false, runner);

        ProductionWorkflowStageException exception =
            await Assert.ThrowsAsync<ProductionWorkflowStageException>(() => adapter.DetectAsync(
                context.Request,
                context.Raster,
                context.Seed,
                CancellationToken.None));

        Assert.AreEqual(ProductionWorkflowFailureCodes.DetectionModelsUnavailable, exception.Failure.Code);
        Assert.AreEqual(0, runner.CallCount);
    }

    [TestMethod]
    public async Task CancellationStopsBeforeInference()
    {
        TestContextData context = CreateContext();
        var runner = new Runner(Success(new float[12], InferenceProvider.Cpu));
        var adapter = CreateAdapter(isApproved: true, runner);
        using var cancellation = new CancellationTokenSource();
        await cancellation.CancelAsync();

        await Assert.ThrowsExactlyAsync<OperationCanceledException>(() => adapter.DetectAsync(
            context.Request,
            context.Raster,
            context.Seed,
            cancellation.Token));

        Assert.AreEqual(0, runner.CallCount);
    }

    [TestMethod]
    public async Task InvalidOutputFailsClosedWithoutArtifactEvidence()
    {
        TestContextData context = CreateContext();
        var output = new float[]
        {
            0.1f, 0.1f, 0.1f, 0.1f,
            1f, 1f, 1f, 1f,
            0f, float.NaN, 0f, 0f,
        };
        var adapter = CreateAdapter(
            isApproved: true,
            new Runner(Success(output, InferenceProvider.Cpu)));

        ProductionWorkflowStageException exception =
            await Assert.ThrowsAsync<ProductionWorkflowStageException>(() => adapter.DetectAsync(
                context.Request,
                context.Raster,
                context.Seed,
                CancellationToken.None));

        Assert.AreEqual(ProductionWorkflowFailureCodes.DetectionEvidenceRejected, exception.Failure.Code);
        StringAssert.Contains(exception.Failure.TechnicalMessage, "invalid center, radius, or artifact values");
    }

    [TestMethod]
    public async Task FakeProviderIsRejectedAfterExecution()
    {
        TestContextData context = CreateContext();
        var output = new float[]
        {
            0.1f, 0.1f, 0.1f, 0.1f,
            1f, 1f, 1f, 1f,
            0f, 0f, 0f, 0f,
        };
        var adapter = CreateAdapter(
            isApproved: true,
            new Runner(Success(output, InferenceProvider.Fake)));

        ProductionWorkflowStageException exception =
            await Assert.ThrowsAsync<ProductionWorkflowStageException>(() => adapter.DetectAsync(
                context.Request,
                context.Raster,
                context.Seed,
                CancellationToken.None));

        Assert.AreEqual(ProductionWorkflowFailureCodes.DetectionEvidenceRejected, exception.Failure.Code);
        StringAssert.Contains(exception.Failure.TechnicalMessage, "non-production provider");
    }

    [TestMethod]
    [DoNotParallelize]
    public void ProductionSizeSeedFramesReuseImmutableFullResolutionPlanes()
    {
        const int width = 2048;
        const int height = 2048;
        int pixelCount = checked(width * height);
        byte[] sourceBytes = [9, 8, 7, 6];
        string sha256 = Convert.ToHexStringLower(SHA256.HashData(sourceBytes));
        var raster = new ProductionDecodedRaster(
            width,
            height,
            sha256,
            WorkflowImageVariant.Original,
            MarkerAffineTransform.Identity,
            OcrFrameTransform.Identity,
            width,
            height,
            new byte[pixelCount],
            new float[pixelCount]);
        var ocrMask = new float[pixelCount];
        var artifactMask = new float[pixelCount];
        long beforeSeed = GC.GetAllocatedBytesForCurrentThread();
        var seed = new ProductionDetectionMaskSeed(
            width,
            height,
            sha256,
            WorkflowImageVariant.Original,
            ocrMask,
            artifactMask);
        long seedAllocated = GC.GetAllocatedBytesForCurrentThread() - beforeSeed;

        MarkerImageFrame first = seed.CreateMarkerFrame(raster);
        long before = GC.GetAllocatedBytesForCurrentThread();
        MarkerImageFrame second = seed.CreateMarkerFrame(raster);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.IsTrue(seedAllocated < 1_000_000, $"Seed ownership allocated {seedAllocated} bytes.");
        Assert.IsTrue(allocated < 1_000_000, $"Second full-resolution frame allocated {allocated} bytes.");
        Assert.IsTrue(MemoryMarshal.TryGetArray(first.ChannelsFirstPixels, out ArraySegment<float> firstPixels));
        Assert.IsTrue(MemoryMarshal.TryGetArray(second.ChannelsFirstPixels, out ArraySegment<float> secondPixels));
        Assert.AreSame(firstPixels.Array, secondPixels.Array);
        Assert.IsTrue(MemoryMarshal.TryGetArray(first.OcrMask.Values, out ArraySegment<float> firstOcr));
        Assert.IsTrue(MemoryMarshal.TryGetArray(second.OcrMask.Values, out ArraySegment<float> secondOcr));
        Assert.AreSame(firstOcr.Array, secondOcr.Array);
        Assert.IsTrue(MemoryMarshal.TryGetArray(first.ArtifactMask.Values, out ArraySegment<float> firstArtifact));
        Assert.IsTrue(MemoryMarshal.TryGetArray(second.ArtifactMask.Values, out ArraySegment<float> secondArtifact));
        Assert.AreSame(firstArtifact.Array, secondArtifact.Array);
    }

    private static ProductionMarkerArtifactMaskAdapter CreateAdapter(
        bool isApproved,
        IProductionArtifactMaskInferenceRunner runner) =>
        new(
            new ModelIdentity(
                "test-marker-center",
                "1.0.0",
                new string('a', 64),
                "memory:test-marker-center.onnx"),
            new ProductionArtifactMaskTensorContract(
                "image_and_masks",
                "marker_heads",
                TensorWidth: 2,
                TensorHeight: 2,
                ArtifactChannelIndex: 2,
                StageVersion: "marker-artifact-mask-v1:1.0.0",
                Timeout: TimeSpan.FromSeconds(1)),
            isApproved,
            runner);

    private static TestContextData CreateContext()
    {
        byte[] bytes = [1, 2, 3, 4];
        string sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes));
        Guid panelId = Guid.Parse("70000000-0000-0000-0000-000000000019");
        var image = new WorkflowImageEvidence(
            "memory:artifact-mask.png",
            sha256,
            4,
            4,
            WorkflowImageVariant.Original);
        var imported = new WorkflowImportedPanel(
            panelId,
            Guid.Parse("71000000-0000-0000-0000-000000000019"),
            "artifact-mask.png",
            image);
        var request = new ProductionWorkflowDetectionRequest(
            new WorkflowPreparedPanel(imported, image, enhanced: null),
            image,
            WorkflowImageVariant.Original,
            Guid.Parse("72000000-0000-0000-0000-000000000019"),
            Guid.Parse("73000000-0000-0000-0000-000000000019"),
            bytes);
        var raster = new ProductionDecodedRaster(
            4,
            4,
            sha256,
            WorkflowImageVariant.Original,
            MarkerAffineTransform.Identity,
            OcrFrameTransform.Identity,
            4,
            4,
            new byte[16],
            Enumerable.Repeat(0.5f, 16).ToArray());
        var ocrMask = new float[16];
        ocrMask[0] = 1;
        var artifactMask = new float[16];
        artifactMask[15] = 1;
        var seed = new ProductionDetectionMaskSeed(
            4,
            4,
            sha256,
            WorkflowImageVariant.Original,
            ocrMask,
            artifactMask);
        return new TestContextData(request, raster, seed);
    }

    private static InferenceResponse Success(
        float[] output,
        InferenceProvider provider) =>
        new(
            true,
            new InferenceExecution(
                output,
                provider,
                new StageTiming(0, 1, 0, 1, 0, false, false),
                new MemoryDiagnostics(0, 0, 0, 0, output.Length)),
            null,
            [new ProviderAttempt(provider, true, null)]);

    private sealed record TestContextData(
        ProductionWorkflowDetectionRequest Request,
        ProductionDecodedRaster Raster,
        ProductionDetectionMaskSeed Seed);

    private sealed class Runner(InferenceResponse response) : IProductionArtifactMaskInferenceRunner
    {
        public int CallCount { get; private set; }

        public InferenceRequest? LastRequest { get; private set; }

        public ValueTask<InferenceResponse> RunAsync(
            InferenceRequest request,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            CallCount++;
            LastRequest = request;
            return ValueTask.FromResult(response);
        }
    }
}
