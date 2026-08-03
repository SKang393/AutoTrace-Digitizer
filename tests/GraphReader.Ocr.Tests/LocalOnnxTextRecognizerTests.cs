// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using GraphReader.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class LocalOnnxTextRecognizerTests
{
    [TestMethod]
    public async Task LocalAdapterBatchesThroughInferenceRuntimeAndDecodesCtcOutput()
    {
        string directory = CreateDirectory();
        string modelPath = Path.Combine(directory, "fixture.onnx");
        await File.WriteAllBytesAsync(modelPath, [1, 3, 3, 7]);
        try
        {
            var identity = new ModelIdentity(
                "fixture-ocr",
                "1",
                Convert.ToHexString(SHA256.HashData(await File.ReadAllBytesAsync(modelPath))).ToLowerInvariant(),
                modelPath);
            var factory = new FakeInferenceSessionFactory(scale: 1);
            await using InferenceRuntime runtime = CreateRuntime(directory, factory);
            var recognizer = new LocalOnnxTextRecognizer(
                runtime,
                new LocalOnnxTextRecognizerOptions(identity, "01")
                {
                    InputWidth = 3,
                    InputHeight = 1,
                    BlankClassIndex = 0,
                    NormalizeMean = 0,
                    NormalizeScale = 1,
                });
            OcrCrop[] crops =
            [
                Crop("first", [0f, 8f, 1f]),
                Crop("second", [0f, 1f, 8f]),
            ];

            IReadOnlyList<OcrRecognition> results = await recognizer.RecognizeBatchAsync(
                crops,
                CancellationToken.None);

            Assert.HasCount(2, results);
            Assert.AreEqual("0", results[0].Alternatives[0].Text);
            Assert.AreEqual("1", results[1].Alternatives[0].Text);
            Assert.IsTrue(results.All(static result => result.Failure is null));
            Assert.AreEqual(1, factory.Sessions.Single().RunCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task MissingLocalModelReturnsStructuredFailureForEveryCrop()
    {
        string directory = CreateDirectory();
        try
        {
            var identity = new ModelIdentity(
                "missing-ocr",
                "1",
                new string('a', 64),
                Path.Combine(directory, "missing.onnx"));
            await using InferenceRuntime runtime = CreateRuntime(directory, new FakeInferenceSessionFactory());
            var recognizer = new LocalOnnxTextRecognizer(
                runtime,
                new LocalOnnxTextRecognizerOptions(identity, "01")
                {
                    InputWidth = 3,
                    InputHeight = 1,
                    NormalizeMean = 0,
                    NormalizeScale = 1,
                });

            IReadOnlyList<OcrRecognition> results = await recognizer.RecognizeBatchAsync(
                [Crop("first", [0f, 8f, 1f]), Crop("second", [0f, 1f, 8f])],
                CancellationToken.None);

            Assert.HasCount(2, results);
            Assert.IsTrue(results.All(static result => result.Failure?.Code == "MODEL_NOT_FOUND"));
            Assert.IsTrue(results.All(static result => result.Failure?.Recoverable == true));
            Assert.IsTrue(results.All(static result => result.Alternatives.Count == 0));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task AdapterRejectsOutputElementCountThatCannotRepresentBatchTimeClasses()
    {
        string directory = CreateDirectory();
        string modelPath = Path.Combine(directory, "bad-shape.onnx");
        await File.WriteAllBytesAsync(modelPath, [2, 4, 6, 8]);
        try
        {
            var identity = new ModelIdentity(
                "bad-shape-ocr",
                "1",
                Convert.ToHexString(SHA256.HashData(await File.ReadAllBytesAsync(modelPath))).ToLowerInvariant(),
                modelPath);
            await using InferenceRuntime runtime = CreateRuntime(directory, new FakeInferenceSessionFactory(scale: 1));
            var recognizer = new LocalOnnxTextRecognizer(
                runtime,
                new LocalOnnxTextRecognizerOptions(identity, "01")
                {
                    InputWidth = 4,
                    InputHeight = 1,
                    BlankClassIndex = 0,
                    NormalizeMean = 0,
                    NormalizeScale = 1,
                });

            IReadOnlyList<OcrRecognition> results = await recognizer.RecognizeBatchAsync(
                [Crop("invalid-shape", [0f, 8f, 1f, 0f], width: 4)],
                CancellationToken.None);

            Assert.HasCount(1, results);
            Assert.AreEqual("OCR_MODEL_OUTPUT_INVALID", results[0].Failure?.Code);
            Assert.IsEmpty(results[0].Alternatives);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task AdapterRejectsDeclaredTimeStepShapeMismatch()
    {
        string directory = CreateDirectory();
        string modelPath = Path.Combine(directory, "time-shape.onnx");
        await File.WriteAllBytesAsync(modelPath, [1, 2, 3, 4]);
        try
        {
            var identity = new ModelIdentity(
                "time-shape-ocr",
                "1",
                Convert.ToHexString(SHA256.HashData(await File.ReadAllBytesAsync(modelPath))).ToLowerInvariant(),
                modelPath);
            await using InferenceRuntime runtime = CreateRuntime(directory, new FakeInferenceSessionFactory(scale: 1));
            var recognizer = new LocalOnnxTextRecognizer(
                runtime,
                new LocalOnnxTextRecognizerOptions(identity, "01")
                {
                    InputWidth = 6,
                    InputHeight = 1,
                    ExpectedTimeSteps = 3,
                    NormalizeMean = 0,
                    NormalizeScale = 1,
                });

            IReadOnlyList<OcrRecognition> results = await recognizer.RecognizeBatchAsync(
                [Crop("wrong-time", [0f, 8f, 1f, 0f, 1f, 8f], width: 6)],
                CancellationToken.None);

            Assert.HasCount(1, results);
            Assert.AreEqual("OCR_MODEL_OUTPUT_SHAPE_MISMATCH", results[0].Failure?.Code);
            Assert.IsEmpty(results[0].Alternatives);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task AdapterRejectsInvalidChannelAndTensorLayoutDeclarations()
    {
        string directory = CreateDirectory();
        try
        {
            var identity = new ModelIdentity(
                "layout-ocr",
                "1",
                new string('a', 64),
                Path.Combine(directory, "unused.onnx"));
            await using InferenceRuntime runtime = CreateRuntime(directory, new FakeInferenceSessionFactory());

            Assert.Throws<ArgumentException>(() => new LocalOnnxTextRecognizer(
                runtime,
                new LocalOnnxTextRecognizerOptions(identity, "01") { InputChannels = 0 }));
            Assert.Throws<ArgumentException>(() => new LocalOnnxTextRecognizer(
                runtime,
                new LocalOnnxTextRecognizerOptions(identity, "01")
                {
                    InputLayout = (OcrTensorLayout)999,
                }));
            Assert.Throws<ArgumentException>(() => new LocalOnnxTextRecognizer(
                runtime,
                new LocalOnnxTextRecognizerOptions(identity, "01")
                {
                    OutputLayout = (OcrOutputLayout)999,
                }));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task DeclaredTimeBatchClassLayoutIsReorderedPerCropBeforeDecoding()
    {
        string directory = CreateDirectory();
        string modelPath = Path.Combine(directory, "time-major.onnx");
        await File.WriteAllBytesAsync(modelPath, [9, 8, 7, 6]);
        try
        {
            var identity = new ModelIdentity(
                "time-major-ocr",
                "1",
                Convert.ToHexString(SHA256.HashData(await File.ReadAllBytesAsync(modelPath))).ToLowerInvariant(),
                modelPath);
            await using InferenceRuntime runtime = CreateRuntime(directory, new FakeInferenceSessionFactory(scale: 1));
            var recognizer = new LocalOnnxTextRecognizer(
                runtime,
                new LocalOnnxTextRecognizerOptions(identity, "01")
                {
                    InputWidth = 6,
                    InputHeight = 1,
                    OutputLayout = OcrOutputLayout.TimeBatchClass,
                    ExpectedTimeSteps = 2,
                    NormalizeMean = 0,
                    NormalizeScale = 1,
                });

            IReadOnlyList<OcrRecognition> results = await recognizer.RecognizeBatchAsync(
                [
                    Crop("zero", [0f, 8f, 1f, 0f, 1f, 8f], width: 6),
                    Crop("one", [8f, 0f, 0f, 8f, 0f, 0f], width: 6),
                ],
                CancellationToken.None);

            Assert.HasCount(2, results);
            Assert.AreEqual("0", results[0].Alternatives[0].Text);
            Assert.AreEqual("1", results[1].Alternatives[0].Text);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static OcrCrop Crop(string id, float[] pixels, int width = 3) =>
        new(
            id,
            OcrSourceImage.Original,
            width,
            1,
            pixels,
            Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(pixels.AsSpan())))
                .ToLowerInvariant(),
            OcrPolygon.FromRectangle(new OcrRectangle(1, 1, width, 1)));

    private static InferenceRuntime CreateRuntime(
        string directory,
        FakeInferenceSessionFactory factory)
    {
        var registry = new OnnxSessionRegistry(
            new FakeExecutionProviderDiscovery("CPUExecutionProvider"),
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
        string path = Path.Combine(Path.GetTempPath(), "GraphReaderOcrTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }
}
