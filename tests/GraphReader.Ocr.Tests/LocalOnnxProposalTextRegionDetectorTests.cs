// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class LocalOnnxProposalTextRegionDetectorTests
{
    private const string ModelEnvironmentVariable = "GRAPHREADER_OCR_COMPONENT_FUSION_V8";
    private const string V11ModelEnvironmentVariable = "GRAPHREADER_OCR_COMPOSITE_ROLE_V11";
    private const string V11ModelSha256 =
        "af13b387140d70946b23ff7349fed82649fde95eb6f6cabe90179b2914a16631";
    private const string ModelSha256 =
        "e0254920b26784a87369aa25cc4ec387c6544db30bda4f9542b7ce9a8712e431";
    private const string ExpectedTensorSha256 =
        "f0b0bd21da7c8db49e17fc7894fbaa4ce239b8ad2f16e8af38b6ca198998ac8b";
    private const string ExpectedGradientTensorSha256 =
        "c8c69a23d54223dca634680e78a222daebd0b20f28af364c2cc440fe983c22da";
    private const string ExpectedV11TensorSha256 =
        "b34d5b428e1ce0f77c6824738d7a083a003fbd8540179045b3e2b39cdc6ddd8a";

    public TestContext TestContext { get; set; } = null!;

    [TestMethod]
    public async Task ProposalDetectorMatchesFrozenGroupingEncodingAndThreshold()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity identity = await CreateModelAsync(directory);
            var factory = new ProposalLogitSessionFactory([20f, -20f, -20f, 20f]);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var detector = new LocalOnnxProposalTextRegionDetector(
                runtime,
                Options(identity));

            IReadOnlyList<OcrDetectedRegion> regions = await detector.DetectAsync(
                FixtureImage(),
                CancellationToken.None);

            Assert.HasCount(1, regions);
            Assert.AreEqual(new OcrRectangle(50, 24, 3, 3), regions[0].Polygon.Bounds);
            Assert.IsGreaterThan(0.999, regions[0].DetectionConfidence);
            Assert.AreEqual(1, regions[0].Evidence?.ComponentCount);
            Assert.AreEqual(
                LocalOnnxProposalTextRegionDetector.PostprocessingAlgorithm,
                regions[0].Evidence?.Reasons.Single());
            Assert.AreEqual(1, factory.Session.RunCount);
            CollectionAssert.AreEqual(
                new long[] { 2, 2, 32, 140 },
                factory.Session.LastInputShape.ToArray());
            string tensorSha256 = Convert.ToHexStringLower(SHA256.HashData(
                MemoryMarshal.AsBytes(factory.Session.LastInputValues.ToArray().AsSpan())));
            Assert.AreEqual(0.10f, factory.Session.LastInputValues[128], 1e-7f);
            Assert.AreEqual(0.15f, factory.Session.LastInputValues[129], 1e-7f);
            Assert.AreEqual(0.75f, factory.Session.LastInputValues[130], 1e-7f);
            Assert.AreEqual(0.125f, factory.Session.LastInputValues[133], 1e-7f);
            Assert.AreEqual(197f / 255f, factory.Session.LastInputValues[134], 1e-7f);
            Assert.AreEqual(ExpectedTensorSha256, tensorSha256);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task ProposalSeamRetainsReviewedRescueBandWithoutChangingNormalAcceptance()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity identity = await CreateModelAsync(directory);
            float rescueLogit = MathF.Log(0.90f / 0.10f);
            var factory = new ProposalLogitSessionFactory(
                [0f, rescueLogit, 20f, -20f]);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var detector = new LocalOnnxProposalTextRegionDetector(runtime, Options(identity));

            IReadOnlyList<OcrDetectedRegion> normal = await detector.DetectAsync(
                FixtureImage(),
                CancellationToken.None);
            IReadOnlyList<OcrDetectedRegion> proposals = await detector.DetectProposalsAsync(
                FixtureImage(),
                CancellationToken.None);

            Assert.IsEmpty(normal);
            Assert.HasCount(1, proposals);
            Assert.AreEqual(0.90, proposals[0].DetectionConfidence, 1e-5);
            Assert.IsGreaterThanOrEqualTo(
                LocalOnnxProposalTextRegionDetector.ProposalConfidenceFloor,
                proposals[0].DetectionConfidence);
            Assert.IsLessThan(0.95, proposals[0].DetectionConfidence);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task CompositeRoleContractAppendsPositionFeaturesAndExposesRoleHint()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity identity = await CreateModelAsync(directory);
            float[] output =
            [
                -20f, 20f, 0f, 0f, 0f, 0f, 0f, 12f, 0f, 0f,
                20f, -20f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            ];
            var factory = new ProposalLogitSessionFactory(output);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var detector = new LocalOnnxProposalTextRegionDetector(
                runtime,
                V11Options(identity));

            IReadOnlyList<OcrDetectedRegion> regions = await detector.DetectProposalsAsync(
                FixtureImage(),
                CancellationToken.None);

            Assert.HasCount(1, regions);
            Assert.AreEqual(OcrTextRole.Participant, regions[0].Context?.ExplicitRoleHint);
            Assert.AreEqual(
                LocalOnnxProposalTextRegionDetector.CompositeRolePostprocessingAlgorithm,
                regions[0].Evidence?.Reasons.Single());
            CollectionAssert.AreEqual(
                new long[] { 2, 2, 32, 144 },
                factory.Session.LastInputShape.ToArray());
            Assert.AreEqual(0.175f, factory.Session.LastInputValues[140], 1e-7f);
            Assert.AreEqual(0.325f, factory.Session.LastInputValues[141], 1e-7f);
            Assert.AreEqual(0.125f, factory.Session.LastInputValues[142], 1e-7f);
            Assert.AreEqual(0.25f, factory.Session.LastInputValues[143], 1e-7f);
            string tensorSha256 = Convert.ToHexStringLower(SHA256.HashData(
                MemoryMarshal.AsBytes(factory.Session.LastInputValues.ToArray().AsSpan())));
            Assert.AreEqual(ExpectedV11TensorSha256, tensorSha256);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task CompositeRoleContractFailsClosedOnWrongShapeOrRoleLogit()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity identity = await CreateModelAsync(directory);
            await using (InferenceRuntime wrongShapeRuntime = CreateRuntime(
                directory,
                new ProposalLogitSessionFactory(new float[4])))
            {
                var detector = new LocalOnnxProposalTextRegionDetector(
                    wrongShapeRuntime,
                    V11Options(identity) with { BypassCache = true });
                InvalidDataException error = await Assert.ThrowsExactlyAsync<InvalidDataException>(async () =>
                    await detector.DetectProposalsAsync(FixtureImage(), CancellationToken.None));
                StringAssert.Contains(error.Message, "20 were required");
            }

            float[] nonFiniteRole = new float[20];
            nonFiniteRole[0] = -20f;
            nonFiniteRole[1] = 20f;
            nonFiniteRole[2] = float.NaN;
            nonFiniteRole[10] = 20f;
            nonFiniteRole[11] = -20f;
            await using (InferenceRuntime nonFiniteRuntime = CreateRuntime(
                directory,
                new ProposalLogitSessionFactory(nonFiniteRole)))
            {
                var detector = new LocalOnnxProposalTextRegionDetector(
                    nonFiniteRuntime,
                    V11Options(identity) with { BypassCache = true });
                InvalidDataException error = await Assert.ThrowsExactlyAsync<InvalidDataException>(async () =>
                    await detector.DetectProposalsAsync(FixtureImage(), CancellationToken.None));
                StringAssert.Contains(error.Message, "non-finite logit");
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task ProposalDetectorMapsAcceptedBoxBackToOriginalPixels()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity identity = await CreateModelAsync(directory);
            var factory = new ProposalLogitSessionFactory([-20f, 20f, 20f, -20f]);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var detector = new LocalOnnxProposalTextRegionDetector(runtime, Options(identity));
            OcrImage source = FixtureImage() with
            {
                OriginalToImage = new OcrFrameTransform(2, 2, 4, 6),
                CanonicalOriginalWidth = 80,
                CanonicalOriginalHeight = 40,
            };

            IReadOnlyList<OcrDetectedRegion> regions = await detector.DetectAsync(
                source,
                CancellationToken.None);

            Assert.HasCount(1, regions);
            Assert.AreEqual(new OcrRectangle(3, 2, 4, 3), regions[0].Polygon.Bounds);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task ProposalDetectorMatchesPythonTensorForGradientAndNonUniformInk()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity identity = await CreateModelAsync(directory);
            var factory = new ProposalLogitSessionFactory([20f, -20f, 20f, -20f]);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var detector = new LocalOnnxProposalTextRegionDetector(runtime, Options(identity));

            _ = await detector.DetectAsync(GradientFixtureImage(), CancellationToken.None);

            CollectionAssert.AreEqual(
                new long[] { 2, 2, 32, 140 },
                factory.Session.LastInputShape.ToArray());
            string tensorSha256 = Convert.ToHexStringLower(SHA256.HashData(
                MemoryMarshal.AsBytes(factory.Session.LastInputValues.ToArray().AsSpan())));
            Assert.AreEqual(ExpectedGradientTensorSha256, tensorSha256);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task ProposalDetectorFailsClosedOnWrongOrNonFiniteOutput()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity identity = await CreateModelAsync(directory);
            await using (InferenceRuntime wrongShapeRuntime = CreateRuntime(
                directory,
                new ProposalLogitSessionFactory([0f])))
            {
                var detector = new LocalOnnxProposalTextRegionDetector(
                    wrongShapeRuntime,
                    Options(identity) with { BypassCache = true });
                InvalidDataException error = await Assert.ThrowsExactlyAsync<InvalidDataException>(async () =>
                    await detector.DetectAsync(FixtureImage(), CancellationToken.None));
                StringAssert.Contains(error.Message, "4 were required");
            }

            await using (InferenceRuntime nonFiniteRuntime = CreateRuntime(
                directory,
                new ProposalLogitSessionFactory([0f, float.NaN, 0f, 1f])))
            {
                var detector = new LocalOnnxProposalTextRegionDetector(
                    nonFiniteRuntime,
                    Options(identity) with { BypassCache = true });
                InvalidDataException error = await Assert.ThrowsExactlyAsync<InvalidDataException>(async () =>
                    await detector.DetectAsync(FixtureImage(), CancellationToken.None));
                StringAssert.Contains(error.Message, "non-finite logit");
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task ProposalDetectorRejectsDriftedFrozenRuntimeOptions()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity identity = await CreateModelAsync(directory);
            await using InferenceRuntime runtime = CreateRuntime(
                directory,
                new ProposalLogitSessionFactory([]));

            Assert.Throws<ArgumentException>(() => new LocalOnnxProposalTextRegionDetector(
                runtime,
                Options(identity) with { ConfidenceThreshold = 0.949f }));
            Assert.Throws<ArgumentException>(() => new LocalOnnxProposalTextRegionDetector(
                runtime,
                Options(identity) with { AllowedProviders = [InferenceProvider.DirectMl] }));
            Assert.Throws<ArgumentException>(() => new LocalOnnxProposalTextRegionDetector(
                runtime,
                Options(identity) with { InputName = "input" }));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task ExactV8PayloadExecutesThroughProposalDetectorOnCpu()
    {
        string? modelPath = Environment.GetEnvironmentVariable(ModelEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            Assert.Inconclusive(
                $"Set {ModelEnvironmentVariable} to the ignored V8 ONNX to run this direct payload probe.");
        }

        modelPath = Path.GetFullPath(modelPath);
        Assert.IsTrue(File.Exists(modelPath), $"OCR V8 model does not exist: {modelPath}");
        Assert.AreEqual(ModelSha256, Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(modelPath))));
        var identity = new ModelIdentity(
            "graph-text-component-fusion-v8",
            "0.0.21-p2",
            ModelSha256,
            modelPath);
        var registry = new OnnxSessionRegistry(
            new CpuOnlyDiscovery(),
            new WindowsExecutionProviderPolicy(),
            new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance),
            CpuThreadConfiguration.Create(1, new SingleCoreDetector()));
        await using var runtime = new InferenceRuntime(
            registry,
            new BoundedInferenceScheduler(2, 1),
            new NoOpStageCache());
        var detector = new LocalOnnxProposalTextRegionDetector(
            runtime,
            Options(identity) with { BypassCache = true });

        IReadOnlyList<OcrDetectedRegion> regions = await detector.DetectAsync(
            FixtureImage(),
            CancellationToken.None);

        TestContext.WriteLine($"accepted_region_count={regions.Count}");
        TestContext.WriteLine("provider=CPUExecutionProvider");
    }

    [TestMethod]
    public async Task ExactV11PayloadExecutesThroughCompositeRoleDetectorOnCpu()
    {
        string? modelPath = Environment.GetEnvironmentVariable(V11ModelEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            Assert.Inconclusive(
                $"Set {V11ModelEnvironmentVariable} to the ignored V11 ONNX to run this direct payload probe.");
        }

        modelPath = Path.GetFullPath(modelPath);
        Assert.IsTrue(File.Exists(modelPath), $"OCR V11 model does not exist: {modelPath}");
        Assert.AreEqual(V11ModelSha256, Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(modelPath))));
        var identity = new ModelIdentity(
            "graph-text-composite-proposal-role-v11-p2",
            "0.0.21-p2",
            V11ModelSha256,
            modelPath);
        var registry = new OnnxSessionRegistry(
            new CpuOnlyDiscovery(),
            new WindowsExecutionProviderPolicy(),
            new OnnxInferenceSessionFactory(NoUiThreadGuard.Instance),
            CpuThreadConfiguration.Create(1, new SingleCoreDetector()));
        await using var runtime = new InferenceRuntime(
            registry,
            new BoundedInferenceScheduler(2, 1),
            new NoOpStageCache());
        var detector = new LocalOnnxProposalTextRegionDetector(
            runtime,
            V11Options(identity) with { BypassCache = true });

        IReadOnlyList<OcrDetectedRegion> regions = await detector.DetectProposalsAsync(
            FixtureImage(),
            CancellationToken.None);

        Assert.IsTrue(regions.All(static region => region.Context?.ExplicitRoleHint is not null));
        TestContext.WriteLine($"accepted_region_count={regions.Count}");
        TestContext.WriteLine("provider=CPUExecutionProvider");
    }

    private static LocalOnnxProposalTextRegionDetectorOptions Options(ModelIdentity identity) =>
        new(identity)
        {
            AllowedProviders = [InferenceProvider.Cpu],
        };

    private static LocalOnnxProposalTextRegionDetectorOptions V11Options(ModelIdentity identity) =>
        new(identity)
        {
            Contract = OcrProposalClassifierContract.CompositeProposalRoleV11,
            GeometryFeatureCount = 16,
            OutputName = "proposal_role_logits",
            StageVersion = "0.0.21-v11-p2",
            AllowedProviders = [InferenceProvider.Cpu],
        };

    private static OcrImage FixtureImage()
    {
        var pixels = Enumerable.Repeat((byte)250, 80 * 40).ToArray();
        Fill(pixels, 80, 10, 10, 3, 6, 20);
        Fill(pixels, 80, 15, 10, 3, 6, 20);
        Fill(pixels, 80, 50, 24, 3, 3, 20);
        return new OcrImage(
            80,
            40,
            80,
            pixels,
            OcrSourceImage.Original,
            OcrFrameTransform.Identity,
            CanonicalOriginalWidth: 80,
            CanonicalOriginalHeight: 40);
    }

    private static OcrImage GradientFixtureImage()
    {
        const int width = 96;
        const int height = 48;
        var pixels = new byte[width * height];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                pixels[(y * width) + x] = checked((byte)(230 + ((x * 3 + y * 5) % 20)));
            }
        }

        for (var y = 8; y < 17; y++)
        {
            for (var x = 7; x < 11; x++)
            {
                pixels[(y * width) + x] = checked((byte)(18 + ((x + y) % 7)));
            }
        }

        for (var y = 9; y < 17; y++)
        {
            for (var x = 14; x < 19; x++)
            {
                pixels[(y * width) + x] = checked((byte)(27 + (((2 * x) + y) % 11)));
            }
        }

        for (var y = 30; y < 36; y++)
        {
            for (var x = 66; x < 70; x++)
            {
                pixels[(y * width) + x] = checked((byte)(35 + ((x + (3 * y)) % 13)));
            }
        }

        return new OcrImage(
            width,
            height,
            width,
            pixels,
            OcrSourceImage.Original,
            OcrFrameTransform.Identity,
            CanonicalOriginalWidth: width,
            CanonicalOriginalHeight: height);
    }

    private static void Fill(
        byte[] pixels,
        int stride,
        int left,
        int top,
        int width,
        int height,
        byte value)
    {
        for (var y = top; y < top + height; y++)
        {
            for (var x = left; x < left + width; x++)
            {
                pixels[(y * stride) + x] = value;
            }
        }
    }

    private static async Task<ModelIdentity> CreateModelAsync(string directory)
    {
        string path = Path.Combine(directory, "component-fusion.onnx");
        await File.WriteAllBytesAsync(path, [8, 2, 5, 6]);
        return new ModelIdentity(
            "graph-text-component-fusion-v8",
            "0.0.21-p2",
            Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(path))),
            path);
    }

    private static InferenceRuntime CreateRuntime(
        string directory,
        IInferenceSessionFactory factory)
    {
        var registry = new OnnxSessionRegistry(
            new FakeExecutionProviderDiscovery("CPUExecutionProvider"),
            new WindowsExecutionProviderPolicy(),
            factory,
            CpuThreadConfiguration.Create(1));
        return new InferenceRuntime(
            registry,
            new BoundedInferenceScheduler(2, 1),
            new ContentAddressedStageCache(Path.Combine(directory, "cache-" + Guid.NewGuid().ToString("N"))));
    }

    private static string CreateDirectory()
    {
        string path = Path.Combine(
            Path.GetTempPath(),
            "GraphReaderProposalOcrTests",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    private sealed class ProposalLogitSessionFactory : IInferenceSessionFactory
    {
        private readonly float[] output;

        public ProposalLogitSessionFactory(float[] output)
        {
            this.output = output;
        }

        public ProposalLogitSession Session { get; private set; } = null!;

        public ValueTask<IInferenceSession> CreateAsync(
            ModelIdentity model,
            InferenceProvider provider,
            CpuThreadConfiguration cpuConfiguration,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Session = new ProposalLogitSession(provider, output);
            return ValueTask.FromResult<IInferenceSession>(Session);
        }
    }

    private sealed class ProposalLogitSession : IInferenceSession
    {
        private readonly float[] output;

        public ProposalLogitSession(InferenceProvider provider, float[] output)
        {
            Provider = provider;
            this.output = output;
        }

        public InferenceProvider Provider { get; }

        public int RunCount { get; private set; }

        public ReadOnlyCollection<float> LastInputValues { get; private set; } =
            Array.AsReadOnly(Array.Empty<float>());

        public IReadOnlyList<long> LastInputShape { get; private set; } = [];

        public ValueTask<InferenceExecution> RunAsync(
            InferenceInput input,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            RunCount++;
            LastInputValues = Array.AsReadOnly(input.Values.ToArray());
            LastInputShape = Array.AsReadOnly(input.Shape.ToArray());
            return ValueTask.FromResult(new InferenceExecution(
                Array.AsReadOnly(output.ToArray()),
                Provider,
                new StageTiming(0, 0.1, 0, 0.1, 0, RunCount == 1, false),
                new MemoryDiagnostics(0, 0, 0, 0, output.Length)));
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
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
