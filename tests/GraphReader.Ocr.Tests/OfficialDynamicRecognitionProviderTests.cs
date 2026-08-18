// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using GraphReader.Inference;

namespace GraphReader.Ocr.Tests;

/// <summary>
/// Opt-in exact-payload probe for the reviewed PP-OCRv5 recognition conversion.
/// Runtime compatibility is not model-quality or production-approval evidence.
/// </summary>
[TestClass]
public sealed class OfficialDynamicRecognitionProviderTests
{
    private const string ModelEnvironmentVariable = "GRAPHREADER_OCR_OFFICIAL_DYNAMIC_WIDTH";
    private const string YamlEnvironmentVariable = "GRAPHREADER_OCR_OFFICIAL_DYNAMIC_YAML";
    private const string ModelSha256 =
        "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743";
    private const string YamlSha256 =
        "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067";

    [TestMethod]
    public async Task ExactOfficialPayloadExecutesCpuAtMinimumAndExpandedWidth()
    {
        string? modelPath = Environment.GetEnvironmentVariable(ModelEnvironmentVariable);
        string? yamlPath = Environment.GetEnvironmentVariable(YamlEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(modelPath) || string.IsNullOrWhiteSpace(yamlPath))
        {
            Assert.Inconclusive(
                $"Set {ModelEnvironmentVariable} and {YamlEnvironmentVariable} to run the exact ignored payload probe.");
        }

        modelPath = Path.GetFullPath(modelPath);
        yamlPath = Path.GetFullPath(yamlPath);
        Assert.AreEqual(ModelSha256, HashFile(modelPath));
        Assert.AreEqual(YamlSha256, HashFile(yamlPath));
        string alphabet = OcrV8DirectPublicCorpusTests.ReadOfficialAlphabet(yamlPath);
        var identity = new ModelIdentity(
            "en_PP-OCRv5_mobile_rec",
            "0.0.21-converted",
            ModelSha256,
            modelPath);
        await using InferenceRuntime runtime = CreateRuntime();
        var recognizer = new LocalOnnxTextRecognizer(
            runtime,
            new LocalOnnxTextRecognizerOptions(identity, alphabet)
            {
                InputWidth = 320,
                MaximumInputWidth = 4096,
                DynamicInputWidth = true,
                InputHeight = 48,
                InputChannels = 3,
                InputLayout = OcrTensorLayout.ChannelsFirst,
                InputColorMode = OcrTensorColorMode.Bgr,
                OutputLayout = OcrOutputLayout.BatchTimeClass,
                OutputActivation = OcrRecognitionOutputActivation.Probabilities,
                BlankClassIndex = 0,
                MaximumAlternatives = 1,
                InputName = "x",
                OutputName = "fetch_name_0",
                StageVersion = "0.0.21-ppocrv5-dynamic-width",
                ChannelMeans = [0.5f, 0.5f, 0.5f],
                ChannelScales = [2f, 2f, 2f],
                AllowedProviders = [InferenceProvider.Cpu],
                BypassCache = true,
            });

        IReadOnlyList<OcrRecognition> minimum = await recognizer.RecognizeBatchAsync(
            [NeutralCrop("minimum", 320)],
            CancellationToken.None);
        IReadOnlyList<OcrRecognition> expanded = await recognizer.RecognizeBatchAsync(
            [NeutralCrop("expanded", 321)],
            CancellationToken.None);

        Assert.HasCount(1, minimum);
        Assert.HasCount(1, expanded);
        Assert.IsNull(minimum[0].Failure);
        Assert.IsNull(expanded[0].Failure);
    }

    private static OcrCrop NeutralCrop(string regionId, int width)
    {
        float[] grayscale = Enumerable.Repeat(0.5f, checked(width * 48)).ToArray();
        float[] bgr = Enumerable.Repeat(0.5f, checked(width * 48 * 3)).ToArray();
        string hash = Convert.ToHexStringLower(SHA256.HashData(
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(grayscale.AsSpan())));
        return new OcrCrop(
            regionId,
            OcrSourceImage.Original,
            width,
            48,
            grayscale,
            hash,
            OcrPolygon.FromRectangle(new OcrRectangle(0, 0, width, 48)),
            new OcrBgrFloatPixels(width * 3, bgr));
    }

    private static string HashFile(string path)
    {
        Assert.IsTrue(File.Exists(path), $"Reviewed OCR payload is missing: {path}");
        return Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    }

    private static InferenceRuntime CreateRuntime()
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
        public ValueTask<byte[]?> TryGetAsync(StageCacheKey key, CancellationToken cancellationToken) =>
            ValueTask.FromResult<byte[]?>(null);

        public ValueTask PutAsync(
            StageCacheKey key,
            ReadOnlyMemory<byte> value,
            CancellationToken cancellationToken) => ValueTask.CompletedTask;
    }
}
