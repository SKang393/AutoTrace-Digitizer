// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class LocalOnnxTextRegionDetectorTests
{
    private static readonly float[] ExpectedBgrTensor = [10f / 255f, 20f / 255f, 30f / 255f];
    private static readonly byte[] BgrDetectorGrayscalePixel = [99];
    private static readonly byte[] MissingBgrGrayscalePixel = [0];

    [TestMethod]
    public async Task DetectorRunsOnBoundCpuPolicyAndMapsProbabilityRegionToOriginalPixels()
    {
        string directory = CreateDirectory();
        string modelPath = Path.Combine(directory, "detector.onnx");
        await File.WriteAllBytesAsync(modelPath, [2, 7, 1, 8]);
        try
        {
            var output = new float[8 * 4];
            foreach (int y in new[] { 1, 2 })
            {
                foreach (int x in new[] { 2, 3, 4 })
                {
                    output[(y * 8) + x] = 0.9f;
                }
            }

            var factory = new ProbabilityMapSessionFactory(output);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var detector = new LocalOnnxTextRegionDetector(
                runtime,
                Options(Identity(modelPath)) with
                {
                    ChannelMeans = [0f, 0.5f, 1f],
                    ChannelScales = [1f, 2f, 3f],
                    AllowedProviders = [InferenceProvider.Cpu],
                });
            var image = new OcrImage(
                8,
                4,
                8,
                Enumerable.Repeat((byte)255, 32).ToArray(),
                OcrSourceImage.Enhanced,
                new OcrFrameTransform(2, 2, 0, 0),
                CanonicalOriginalWidth: 4,
                CanonicalOriginalHeight: 2);

            IReadOnlyList<OcrDetectedRegion> first = await detector.DetectAsync(
                image,
                CancellationToken.None);
            IReadOnlyList<OcrDetectedRegion> cached = await detector.DetectAsync(
                image,
                CancellationToken.None);

            Assert.HasCount(1, first);
            Assert.AreEqual(new OcrRectangle(1, 0.5, 1.5, 1), first[0].Polygon.Bounds);
            Assert.AreEqual(OcrContract.CoordinateSpace, first[0].CoordinateSpace);
            Assert.AreEqual(0.9, first[0].DetectionConfidence, 0.0001);
            Assert.AreEqual(first[0].RegionId, cached[0].RegionId);
            Assert.IsTrue(Guid.TryParse(first[0].RegionId, out _));
            CollectionAssert.AreEqual(
                new[] { InferenceProvider.Cpu },
                factory.CreatedProviders.ToArray());
            Assert.IsNotNull(factory.LastInput);
            CollectionAssert.AreEqual(new long[] { 1, 3, 4, 8 }, factory.LastInput.Shape.ToArray());
            float[] input = factory.LastInput.Values.ToArray();
            Assert.IsTrue(input.Take(32).All(static value => value == 1f));
            Assert.IsTrue(input.Skip(32).Take(32).All(static value => value == 1f));
            Assert.IsTrue(input.Skip(64).Take(32).All(static value => value == 0f));
            Assert.AreEqual(1, factory.RunCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task DetectorRejectsOutputShapeAndProbabilityContractViolations()
    {
        string directory = CreateDirectory();
        string modelPath = Path.Combine(directory, "invalid-output.onnx");
        await File.WriteAllBytesAsync(modelPath, [4, 2, 9]);
        try
        {
            var image = Image();
            await using (InferenceRuntime wrongShapeRuntime = CreateRuntime(
                Path.Combine(directory, "wrong-shape"),
                new ProbabilityMapSessionFactory(new float[31])))
            {
                var detector = new LocalOnnxTextRegionDetector(
                    wrongShapeRuntime,
                    Options(Identity(modelPath)));
                InvalidDataException wrongShape = await Assert.ThrowsExactlyAsync<InvalidDataException>(
                    () => detector.DetectAsync(image, CancellationToken.None).AsTask());
                StringAssert.Contains(wrongShape.Message, "31 values; 32 were required");
            }

            float[] invalidProbability = new float[32];
            invalidProbability[10] = 1.1f;
            await using (InferenceRuntime probabilityRuntime = CreateRuntime(
                Path.Combine(directory, "invalid-probability"),
                new ProbabilityMapSessionFactory(invalidProbability)))
            {
                var detector = new LocalOnnxTextRegionDetector(
                    probabilityRuntime,
                    Options(Identity(modelPath)));
                InvalidDataException invalid = await Assert.ThrowsExactlyAsync<InvalidDataException>(
                    () => detector.DetectAsync(image, CancellationToken.None).AsTask());
                StringAssert.Contains(invalid.Message, "within [0,1]");
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task DetectorPreservesDeclaredBgrChannelOrder()
    {
        string directory = CreateDirectory();
        string modelPath = Path.Combine(directory, "bgr-detector.onnx");
        await File.WriteAllBytesAsync(modelPath, [9, 3, 7]);
        try
        {
            var factory = new ProbabilityMapSessionFactory([0f]);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var detector = new LocalOnnxTextRegionDetector(
                runtime,
                Options(Identity(modelPath)) with
                {
                    InputColorMode = OcrTensorColorMode.Bgr,
                    ChannelMeans = [0f, 0f, 0f],
                    ChannelScales = [1f, 1f, 1f],
                });
            var image = new OcrImage(
                1,
                1,
                1,
                BgrDetectorGrayscalePixel,
                OcrSourceImage.Original,
                OcrFrameTransform.Identity,
                CanonicalOriginalWidth: 1,
                CanonicalOriginalHeight: 1,
                BgrPixels: new OcrBgrBytePixels(3, new byte[] { 10, 20, 30 }));

            _ = await detector.DetectAsync(image, CancellationToken.None);

            Assert.IsNotNull(factory.LastInput);
            CollectionAssert.AreEqual(new long[] { 1, 3, 1, 1 }, factory.LastInput.Shape.ToArray());
            CollectionAssert.AreEqual(ExpectedBgrTensor, factory.LastInput.Values.ToArray());
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task BgrDetectorRejectsMissingColorPlaneBeforeProviderExecution()
    {
        string directory = CreateDirectory();
        string modelPath = Path.Combine(directory, "missing-bgr.onnx");
        await File.WriteAllBytesAsync(modelPath, [7, 1, 3]);
        try
        {
            var factory = new ProbabilityMapSessionFactory([0f]);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var detector = new LocalOnnxTextRegionDetector(
                runtime,
                Options(Identity(modelPath)) with { InputColorMode = OcrTensorColorMode.Bgr });

            await Assert.ThrowsExactlyAsync<ArgumentException>(() =>
                detector.DetectAsync(
                    new OcrImage(
                        1,
                        1,
                        1,
                        MissingBgrGrayscalePixel,
                        OcrSourceImage.Original,
                        OcrFrameTransform.Identity,
                        CanonicalOriginalWidth: 1,
                        CanonicalOriginalHeight: 1),
                    CancellationToken.None).AsTask());

            Assert.AreEqual(0, factory.RunCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task DetectorHonorsCancellationBeforeProviderExecution()
    {
        string directory = CreateDirectory();
        string modelPath = Path.Combine(directory, "cancelled.onnx");
        await File.WriteAllBytesAsync(modelPath, [8, 6, 7, 5]);
        try
        {
            var factory = new ProbabilityMapSessionFactory(new float[32]);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var detector = new LocalOnnxTextRegionDetector(runtime, Options(Identity(modelPath)));

            await Assert.ThrowsExactlyAsync<OperationCanceledException>(() =>
                detector.DetectAsync(Image(), new CancellationToken(canceled: true)).AsTask());

            Assert.AreEqual(0, factory.RunCount);
            Assert.IsEmpty(factory.CreatedProviders);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task DetectorNeverRoundsPastNonDivisibleMaximumSide()
    {
        string directory = CreateDirectory();
        string modelPath = Path.Combine(directory, "bounded.onnx");
        await File.WriteAllBytesAsync(modelPath, [3, 1, 4, 1]);
        try
        {
            const int alignedWidth = 96;
            const int alignedHeight = 64;
            var factory = new ProbabilityMapSessionFactory(new float[alignedWidth * alignedHeight]);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var detector = new LocalOnnxTextRegionDetector(
                runtime,
                Options(Identity(modelPath)) with
                {
                    MaximumSideLength = 100,
                    DimensionMultiple = 32,
                });
            var image = new OcrImage(
                100,
                80,
                100,
                new byte[8_000],
                OcrSourceImage.Original,
                OcrFrameTransform.Identity,
                CanonicalOriginalWidth: 100,
                CanonicalOriginalHeight: 80);

            _ = await detector.DetectAsync(image, CancellationToken.None);

            Assert.IsNotNull(factory.LastInput);
            CollectionAssert.AreEqual(
                new long[] { 1, 3, alignedHeight, alignedWidth },
                factory.LastInput.Shape.ToArray());
            Assert.IsTrue(factory.LastInput.Shape.Skip(2).All(dimension => dimension <= 100));
            Assert.IsTrue(factory.LastInput.Shape.Skip(2).All(dimension => dimension % 32 == 0));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task DetectorRejectsProviderPoliciesWithoutMandatoryCpuFallback()
    {
        string directory = CreateDirectory();
        try
        {
            var model = new ModelIdentity(
                "detector",
                "1",
                new string('a', 64),
                Path.Combine(directory, "unused.onnx"));
            await using InferenceRuntime runtime = CreateRuntime(
                directory,
                new ProbabilityMapSessionFactory(new float[32]));

            Assert.Throws<ArgumentException>(() => new LocalOnnxTextRegionDetector(
                runtime,
                Options(model) with { AllowedProviders = [InferenceProvider.DirectMl] }));
            Assert.Throws<ArgumentException>(() => new LocalOnnxTextRegionDetector(
                runtime,
                Options(model) with { AllowedProviders = [InferenceProvider.Cpu, InferenceProvider.Fake] }));
            Assert.Throws<ArgumentException>(() => new LocalOnnxTextRegionDetector(
                runtime,
                Options(model) with { AllowedProviders = [InferenceProvider.Cpu, (InferenceProvider)99] }));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static LocalOnnxTextRegionDetectorOptions Options(ModelIdentity model) => new(model)
    {
        MaximumSideLength = 32,
        DimensionMultiple = 1,
        InputChannels = 3,
        ProbabilityThreshold = 0.5f,
        BoxConfidenceThreshold = 0.8f,
        UnclipRatio = 0,
        MinimumComponentArea = 2,
        MinimumSideLength = 1,
        MaximumRegions = 20,
        AllowedProviders = [InferenceProvider.Cpu],
    };

    private static OcrImage Image() => new(
        8,
        4,
        8,
        Enumerable.Repeat((byte)255, 32).ToArray(),
        OcrSourceImage.Original,
        OcrFrameTransform.Identity);

    private static ModelIdentity Identity(string path) => new(
        "fixture-ocr-detector",
        "1.0.0",
        Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))),
        path);

    private static InferenceRuntime CreateRuntime(
        string directory,
        ProbabilityMapSessionFactory factory)
    {
        var registry = new OnnxSessionRegistry(
            new FakeExecutionProviderDiscovery("DmlExecutionProvider", "CPUExecutionProvider"),
            new WindowsExecutionProviderPolicy(),
            factory,
            CpuThreadConfiguration.Create(1));
        return new InferenceRuntime(
            registry,
            new BoundedInferenceScheduler(capacity: 2, workerCount: 1),
            new ContentAddressedStageCache(Path.Combine(directory, "cache")));
    }

    private static string CreateDirectory()
    {
        string path = Path.Combine(
            Path.GetTempPath(),
            "GraphReaderOcrDetectorTests",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    private sealed class ProbabilityMapSessionFactory(float[] output) : IInferenceSessionFactory
    {
        private readonly float[] output = (float[])output.Clone();

        public List<InferenceProvider> CreatedProviders { get; } = [];

        public InferenceInput? LastInput { get; private set; }

        public int RunCount { get; private set; }

        public ValueTask<IInferenceSession> CreateAsync(
            ModelIdentity model,
            InferenceProvider provider,
            CpuThreadConfiguration cpuConfiguration,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            CreatedProviders.Add(provider);
            return ValueTask.FromResult<IInferenceSession>(new Session(this, provider));
        }

        private sealed class Session(
            ProbabilityMapSessionFactory owner,
            InferenceProvider provider) : IInferenceSession
        {
            public InferenceProvider Provider { get; } = provider;

            public ValueTask<InferenceExecution> RunAsync(
                InferenceInput input,
                CancellationToken cancellationToken)
            {
                cancellationToken.ThrowIfCancellationRequested();
                owner.LastInput = new InferenceInput(
                    input.Values.ToArray(),
                    input.Shape.ToArray(),
                    input.InputName,
                    input.OutputName);
                owner.RunCount++;
                return ValueTask.FromResult(new InferenceExecution(
                    Array.AsReadOnly((float[])owner.output.Clone()),
                    Provider,
                    new StageTiming(0, 1, 0, 1, 0, owner.RunCount == 1, false),
                    new MemoryDiagnostics(0, 0, 0, 0, owner.output.Length)));
            }

            public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        }
    }
}
