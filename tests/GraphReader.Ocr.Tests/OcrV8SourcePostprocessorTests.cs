// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Security.Cryptography;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class OcrV8SourcePostprocessorTests
{
    [TestMethod]
    public void SourceCropUsesFrozenPaddingAndBindsExactOriginalBytes()
    {
        var pixels = Enumerable.Range(0, 12 * 8).Select(static value => (byte)value).ToArray();
        var image = new OcrImage(
            12,
            8,
            12,
            pixels,
            OcrSourceImage.Original,
            OcrFrameTransform.Identity);
        OcrDetectedRegion region = OcrTestFixtures.Region("source", 9, 3, 2, 2);

        OcrV8SourceCrop crop = OcrV8SourcePostprocessor.ExtractSourceCrop(image, region);

        Assert.AreEqual(11, crop.Width);
        Assert.AreEqual(6, crop.Height);
        CollectionAssert.AreEqual(
            Enumerable.Range(1, 6)
                .SelectMany(y => pixels.Skip((y * 12) + 1).Take(11))
                .ToArray(),
            crop.Pixels.ToArray());
        Assert.AreEqual(
            Convert.ToHexStringLower(SHA256.HashData(crop.Pixels.Span)),
            crop.PixelSha256);
    }

    [TestMethod]
    public void ConservativeSpacingAddsOnlySourceSupportedWhitespace()
    {
        OcrV8SourceCrop separated = CropWithBars((2, 4), (14, 17));
        OcrV8SourceCrop joined = CropWithBars((2, 5), (8, 11));

        Assert.AreEqual(
            "AB CD",
            OcrV8SourcePostprocessor.RestoreConservativeSourceSpaces(separated, "ABCD"));
        Assert.AreEqual(
            "ABCD",
            OcrV8SourcePostprocessor.RestoreConservativeSourceSpaces(joined, "ABCD"));
        Assert.AreEqual(
            "7",
            OcrV8SourcePostprocessor.RestoreConservativeSourceSpaces(separated, "7"));
    }

    [TestMethod]
    public void ConservativeSpacingCountsUnicodeScalarsWithoutSplittingSurrogatePairs()
    {
        OcrV8SourceCrop separated = CropWithBars((2, 4), (14, 17));

        Assert.AreEqual(
            "A 😀B",
            OcrV8SourcePostprocessor.RestoreConservativeSourceSpaces(separated, "A😀B"));
    }

    [TestMethod]
    public async Task ExtendedAliasRouteExecutesExactGroupsAndReconstructsCanonicalText()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity model = await ModelAsync(directory);
            var factory = new AmbiguitySessionFactory([0, 1, 2, 3]);
            await using InferenceRuntime runtime = Runtime(directory, factory);
            var processor = new OcrV8SourcePostprocessor(
                runtime,
                new LocalOnnxAmbiguitySourceGroupOptions(model)
                {
                    AllowedProviders = [InferenceProvider.Cpu],
                });
            OcrV8SourceCrop crop = CropWithBars((2, 4), (11, 13), (20, 22), (29, 31));

            OcrV8AmbiguityResult result = await processor.ResolveAmbiguityAsync(
                crop,
                "! i O l",
                CancellationToken.None);

            Assert.IsTrue(result.ModelExecuted);
            Assert.AreEqual("O o l I", result.Text);
            Assert.AreEqual(4, result.ChangedCharacterCount);
            Assert.AreEqual(64, result.InputTensorSha256.Length);
            CollectionAssert.AreEqual(
                new long[] { 4, 1, 32, 32 },
                factory.Session.LastInputShape.ToArray());
            Assert.IsTrue(factory.Session.LastInputValues.All(static value =>
                float.IsFinite(value) && value is >= 0 and <= 1));
            Assert.AreEqual(64, processor.ConfigurationFingerprint.Length);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task NonAmbiguityTextCannotExecuteSpecialist()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity model = await ModelAsync(directory);
            var factory = new AmbiguitySessionFactory([0]);
            await using InferenceRuntime runtime = Runtime(directory, factory);
            var processor = new OcrV8SourcePostprocessor(
                runtime,
                new LocalOnnxAmbiguitySourceGroupOptions(model));

            OcrV8AmbiguityResult result = await processor.ResolveAmbiguityAsync(
                CropWithBars((2, 4), (14, 17)),
                "AB",
                CancellationToken.None);

            Assert.AreEqual("AB", result.Text);
            Assert.IsFalse(result.ModelExecuted);
            Assert.AreEqual(0, factory.SessionCreationCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task WrongOrNonFiniteAmbiguityOutputFailsClosed()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity model = await ModelAsync(directory);
            OcrV8SourceCrop crop = CropWithBars((2, 4), (14, 17));
            await using (InferenceRuntime wrongRuntime = Runtime(
                directory,
                new RawOutputSessionFactory([0f])))
            {
                var processor = new OcrV8SourcePostprocessor(
                    wrongRuntime,
                    new LocalOnnxAmbiguitySourceGroupOptions(model) { BypassCache = true });
                await Assert.ThrowsExactlyAsync<InvalidDataException>(async () =>
                    await processor.ResolveAmbiguityAsync(crop, "Oi", CancellationToken.None));
            }

            await using (InferenceRuntime nonFiniteRuntime = Runtime(
                directory,
                new RawOutputSessionFactory([float.NaN, 0f, 0f, 0f, 0f, 1f, 0f, 0f])))
            {
                var processor = new OcrV8SourcePostprocessor(
                    nonFiniteRuntime,
                    new LocalOnnxAmbiguitySourceGroupOptions(model) { BypassCache = true });
                await Assert.ThrowsExactlyAsync<InvalidDataException>(async () =>
                    await processor.ResolveAmbiguityAsync(crop, "Oi", CancellationToken.None));
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task OfficialDecoratorAppliesSourceRulesWithoutChangingGeometryOrConfidence()
    {
        string directory = CreateDirectory();
        try
        {
            ModelIdentity model = await ModelAsync(directory);
            var factory = new AmbiguitySessionFactory([0, 1]);
            await using InferenceRuntime runtime = Runtime(directory, factory);
            var postprocessor = new OcrV8SourcePostprocessor(
                runtime,
                new LocalOnnxAmbiguitySourceGroupOptions(model)
                {
                    AllowedProviders = [InferenceProvider.Cpu],
                });
            var official = new StubTextRecognizer((crops, cancellationToken) =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return ValueTask.FromResult<IReadOnlyList<OcrRecognition>>(
                    [new OcrRecognition(
                        crops[0].RegionId,
                        crops[0].SourceImage,
                        [new OcrRecognitionAlternative("!i", 0.91, crops[0].SourceImage)],
                        1.25)]);
            });
            var decorator = new OcrV8OfficialTextRecognizer(official, postprocessor);
            OcrV8SourceCrop sourceCrop = CropWithBars((2, 4), (14, 17));
            var polygon = OcrPolygon.FromRectangle(new OcrRectangle(5, 6, 20, 10));
            var crop = new OcrCrop(
                "official",
                OcrSourceImage.Original,
                320,
                48,
                new float[320 * 48],
                new string('a', 64),
                polygon,
                SourceCrop: sourceCrop);

            OcrRecognition result = (await decorator.RecognizeBatchAsync(
                [crop],
                CancellationToken.None)).Single();

            Assert.AreEqual("O o", result.Alternatives.Single().Text);
            Assert.AreEqual(0.91, result.Alternatives.Single().Confidence, 1e-9);
            Assert.AreEqual(OcrSourceImage.Original, result.Alternatives.Single().SourceImage);
            Assert.IsGreaterThan(1.25, result.InferenceMilliseconds);
            Assert.AreEqual(64, decorator.ConfigurationFingerprint.Length);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static OcrV8SourceCrop CropWithBars(params (int Left, int Right)[] bars)
    {
        const int width = 40;
        const int height = 20;
        var pixels = Enumerable.Repeat((byte)255, width * height).ToArray();
        foreach ((int left, int right) in bars)
        {
            for (var y = 4; y < 16; y++)
            {
                for (var x = left; x < right; x++)
                {
                    pixels[(y * width) + x] = 20;
                }
            }
        }

        return new OcrV8SourceCrop(
            width,
            height,
            pixels,
            Convert.ToHexStringLower(SHA256.HashData(pixels)),
            OcrPolygon.FromRectangle(new OcrRectangle(0, 0, width, height)));
    }

    private static async Task<ModelIdentity> ModelAsync(string directory)
    {
        string path = Path.Combine(directory, "ambiguity.onnx");
        await File.WriteAllBytesAsync(path, [4, 3, 2, 1]);
        return new ModelIdentity(
            "graph-ambiguity-source-group-v3-p2",
            "0.0.21-p2",
            Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(path))),
            path);
    }

    private static InferenceRuntime Runtime(string directory, IInferenceSessionFactory factory) =>
        new(
            new OnnxSessionRegistry(
                new FakeExecutionProviderDiscovery("CPUExecutionProvider"),
                new WindowsExecutionProviderPolicy(),
                factory,
                CpuThreadConfiguration.Create(1)),
            new BoundedInferenceScheduler(2, 1),
            new ContentAddressedStageCache(Path.Combine(directory, "cache-" + Guid.NewGuid().ToString("N"))));

    private static string CreateDirectory()
    {
        string path = Path.Combine(
            Path.GetTempPath(),
            "GraphReaderOcrV8SourceTests",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    private sealed class AmbiguitySessionFactory : IInferenceSessionFactory
    {
        private readonly int[] classes;

        public AmbiguitySessionFactory(int[] classes) => this.classes = classes;

        public AmbiguitySession Session { get; private set; } = null!;

        public int SessionCreationCount { get; private set; }

        public ValueTask<IInferenceSession> CreateAsync(
            ModelIdentity model,
            InferenceProvider provider,
            CpuThreadConfiguration cpuConfiguration,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            SessionCreationCount++;
            Session = new AmbiguitySession(provider, classes);
            return ValueTask.FromResult<IInferenceSession>(Session);
        }
    }

    private sealed class AmbiguitySession : IInferenceSession
    {
        private readonly int[] classes;

        public AmbiguitySession(InferenceProvider provider, int[] classes)
        {
            Provider = provider;
            this.classes = classes;
        }

        public InferenceProvider Provider { get; }

        public ReadOnlyCollection<float> LastInputValues { get; private set; } =
            Array.AsReadOnly(Array.Empty<float>());

        public IReadOnlyList<long> LastInputShape { get; private set; } = [];

        public ValueTask<InferenceExecution> RunAsync(
            InferenceInput input,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            LastInputValues = Array.AsReadOnly(input.Values.ToArray());
            LastInputShape = Array.AsReadOnly(input.Shape.ToArray());
            int count = checked((int)input.Shape[0]);
            Assert.AreEqual(count, classes.Length);
            var output = Enumerable.Repeat(-20f, count * 4).ToArray();
            for (var index = 0; index < count; index++)
            {
                output[(index * 4) + classes[index]] = 20f;
            }

            return ValueTask.FromResult(Execution(output));
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class RawOutputSessionFactory : IInferenceSessionFactory
    {
        private readonly float[] output;

        public RawOutputSessionFactory(float[] output) => this.output = output;

        public ValueTask<IInferenceSession> CreateAsync(
            ModelIdentity model,
            InferenceProvider provider,
            CpuThreadConfiguration cpuConfiguration,
            CancellationToken cancellationToken) =>
            ValueTask.FromResult<IInferenceSession>(new RawOutputSession(provider, output));
    }

    private sealed class RawOutputSession : IInferenceSession
    {
        private readonly float[] output;

        public RawOutputSession(InferenceProvider provider, float[] output)
        {
            Provider = provider;
            this.output = output;
        }

        public InferenceProvider Provider { get; }

        public ValueTask<InferenceExecution> RunAsync(
            InferenceInput input,
            CancellationToken cancellationToken) =>
            ValueTask.FromResult(Execution(output));

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private static InferenceExecution Execution(float[] output) =>
        new(
            Array.AsReadOnly(output),
            InferenceProvider.Cpu,
            new StageTiming(0, 0.1, 0, 0.1, 0, true, false),
            new MemoryDiagnostics(0, 0, 0, 0, output.Length));
}
