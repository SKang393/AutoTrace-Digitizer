// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text;
using GraphReader.Inference;

namespace GraphReader.Ocr;

public enum OcrTensorLayout
{
    ChannelsFirst,
    ChannelsLast,
}

public enum OcrOutputLayout
{
    BatchTimeClass,
    TimeBatchClass,
}

public sealed record LocalOnnxTextRecognizerOptions(
    ModelIdentity Model,
    string Alphabet)
{
    public int InputWidth { get; init; } = 128;

    public int InputHeight { get; init; } = 32;

    public int InputChannels { get; init; } = 1;

    public OcrTensorLayout InputLayout { get; init; } = OcrTensorLayout.ChannelsFirst;

    public OcrOutputLayout OutputLayout { get; init; } = OcrOutputLayout.BatchTimeClass;

    public int? ExpectedTimeSteps { get; init; }

    public int BlankClassIndex { get; init; }

    public int MaximumAlternatives { get; init; } = 3;

    public string InputName { get; init; } = "input";

    public string OutputName { get; init; } = "output";

    public string StageVersion { get; init; } = "0.1.0";

    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

    public float NormalizeMean { get; init; } = 0.5f;

    public float NormalizeScale { get; init; } = 2f;
}

public sealed record CtcDecodedAlternative(string Text, double Confidence);

public static class CtcRecognitionDecoder
{
    public static IReadOnlyList<CtcDecodedAlternative> Decode(
        ReadOnlySpan<float> logits,
        int timeSteps,
        string alphabet,
        int blankClassIndex = 0,
        int maximumAlternatives = 3)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(alphabet);
        var classCount = alphabet.Length + 1;
        if (timeSteps <= 0 || logits.Length != checked(timeSteps * classCount) ||
            blankClassIndex < 0 || blankClassIndex >= classCount || maximumAlternatives <= 0)
        {
            throw new ArgumentException("CTC logits or decoder options are invalid.", nameof(logits));
        }

        var bestClasses = new int[timeSteps];
        var secondClasses = new int[timeSteps];
        var bestProbabilities = new double[timeSteps];
        var uncertainty = new double[timeSteps];
        for (var time = 0; time < timeSteps; time++)
        {
            var row = logits.Slice(time * classCount, classCount);
            var maximum = row.ToArray().Max();
            var denominator = 0d;
            for (var classIndex = 0; classIndex < classCount; classIndex++)
            {
                denominator += Math.Exp(row[classIndex] - maximum);
            }

            var best = -1;
            var second = -1;
            var bestProbability = double.NegativeInfinity;
            var secondProbability = double.NegativeInfinity;
            for (var classIndex = 0; classIndex < classCount; classIndex++)
            {
                var probability = Math.Exp(row[classIndex] - maximum) / denominator;
                if (probability > bestProbability)
                {
                    second = best;
                    secondProbability = bestProbability;
                    best = classIndex;
                    bestProbability = probability;
                }
                else if (probability > secondProbability)
                {
                    second = classIndex;
                    secondProbability = probability;
                }
            }

            bestClasses[time] = best;
            secondClasses[time] = second;
            bestProbabilities[time] = bestProbability;
            uncertainty[time] = secondProbability / Math.Max(bestProbability, double.Epsilon);
        }

        var greedy = DecodeClasses(bestClasses, bestProbabilities, alphabet, blankClassIndex);
        var candidates = new List<CtcDecodedAlternative>
        {
            greedy,
        };
        foreach (var time in Enumerable.Range(0, timeSteps).OrderByDescending(index => uncertainty[index]))
        {
            if (candidates.Count >= maximumAlternatives)
            {
                break;
            }

            var modified = (int[])bestClasses.Clone();
            modified[time] = secondClasses[time];
            var probabilities = (double[])bestProbabilities.Clone();
            probabilities[time] *= uncertainty[time];
            var candidate = DecodeClasses(modified, probabilities, alphabet, blankClassIndex);
            if (candidate.Text.Length > 0 && candidates.All(item => !string.Equals(item.Text, candidate.Text, StringComparison.Ordinal)))
            {
                candidates.Add(candidate with
                {
                    Confidence = Math.Min(candidate.Confidence, greedy.Confidence * 0.99),
                });
            }
        }

        return OcrCollections.Freeze(candidates.Where(static candidate => candidate.Text.Length > 0));
    }

    private static CtcDecodedAlternative DecodeClasses(
        int[] classes,
        double[] probabilities,
        string alphabet,
        int blankClassIndex)
    {
        var builder = new StringBuilder(classes.Length);
        var acceptedProbabilities = new List<double>();
        var prior = -1;
        for (var index = 0; index < classes.Length; index++)
        {
            var classIndex = classes[index];
            if (classIndex != blankClassIndex && classIndex != prior)
            {
                var alphabetIndex = classIndex < blankClassIndex ? classIndex : classIndex - 1;
                builder.Append(alphabet[alphabetIndex]);
                acceptedProbabilities.Add(probabilities[index]);
            }

            prior = classIndex;
        }

        var confidence = acceptedProbabilities.Count == 0 ? 0 : acceptedProbabilities.Average();
        return new CtcDecodedAlternative(builder.ToString(), Math.Clamp(confidence, 0, 1));
    }
}

/// <summary>
/// Local-only batched CTC recognition over the shared ONNX inference runtime.
/// The adapter contains no model weights and does not own the supplied runtime.
/// </summary>
public sealed class LocalOnnxTextRecognizer : ITextRecognizer
{
    private readonly InferenceRuntime _runtime;
    private readonly LocalOnnxTextRecognizerOptions _options;
    private readonly string _configurationFingerprint;

    public LocalOnnxTextRecognizer(
        InferenceRuntime runtime,
        LocalOnnxTextRecognizerOptions options)
    {
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.Model.Validate();
        if (string.IsNullOrEmpty(_options.Alphabet) ||
            _options.Alphabet.Distinct().Count() != _options.Alphabet.Length ||
            _options.InputWidth <= 0 || _options.InputHeight <= 0 || _options.InputChannels <= 0 ||
            !Enum.IsDefined(_options.InputLayout) || !Enum.IsDefined(_options.OutputLayout) ||
            _options.ExpectedTimeSteps is <= 0 ||
            _options.BlankClassIndex < 0 || _options.BlankClassIndex > _options.Alphabet.Length ||
            _options.MaximumAlternatives <= 0 || _options.Timeout <= TimeSpan.Zero ||
            !float.IsFinite(_options.NormalizeMean) || !float.IsFinite(_options.NormalizeScale) ||
            _options.NormalizeScale == 0)
        {
            throw new ArgumentException("Local ONNX OCR recognizer options are invalid.", nameof(options));
        }

        _configurationFingerprint = CreateConfigurationFingerprint(_options);
    }

    public string ModelId => _options.Model.ModelId;

    public string ModelVersion => _options.Model.Version;

    public string ModelSha256 => _options.Model.Sha256;

    public string ConfigurationFingerprint => _configurationFingerprint;

    public async ValueTask<IReadOnlyList<OcrRecognition>> RecognizeBatchAsync(
        IReadOnlyList<OcrCrop> crops,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(crops);
        cancellationToken.ThrowIfCancellationRequested();
        if (crops.Count == 0)
        {
            return Array.Empty<OcrRecognition>();
        }

        if (crops.Any(crop =>
                crop.Width != _options.InputWidth || crop.Height != _options.InputHeight ||
                crop.Pixels.Length != checked(crop.Width * crop.Height)))
        {
            throw new ArgumentException("Every OCR crop must match the configured model input dimensions.", nameof(crops));
        }

        var pixelsPerCrop = checked(_options.InputWidth * _options.InputHeight);
        var valuesPerCrop = checked(pixelsPerCrop * _options.InputChannels);
        var inputValues = new float[checked(crops.Count * valuesPerCrop)];
        for (var cropIndex = 0; cropIndex < crops.Count; cropIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var source = crops[cropIndex].Pixels.Span;
            for (var valueIndex = 0; valueIndex < source.Length; valueIndex++)
            {
                var normalized = (source[valueIndex] - _options.NormalizeMean) * _options.NormalizeScale;
                for (var channel = 0; channel < _options.InputChannels; channel++)
                {
                    var destinationIndex = _options.InputLayout == OcrTensorLayout.ChannelsFirst
                        ? (cropIndex * valuesPerCrop) + (channel * pixelsPerCrop) + valueIndex
                        : (cropIndex * valuesPerCrop) + (valueIndex * _options.InputChannels) + channel;
                    inputValues[destinationIndex] = normalized;
                }
            }
        }

        IReadOnlyList<long> inputShape = _options.InputLayout == OcrTensorLayout.ChannelsFirst
            ? [crops.Count, _options.InputChannels, _options.InputHeight, _options.InputWidth]
            : [crops.Count, _options.InputHeight, _options.InputWidth, _options.InputChannels];

        var inputHash = HashStrings(crops.Select(static crop => crop.CropSha256));
        var request = new InferenceRequest(
            _options.Model,
            new InferenceInput(
                inputValues,
                inputShape,
                _options.InputName,
                _options.OutputName),
            new StageCacheMaterial(
                inputHash,
                string.Join(',', crops.Select(static crop => crop.RegionId)),
                string.Join(',', crops.Select(static crop => crop.SourceImage.ToString())),
                "ocr_recognition",
                _options.StageVersion,
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["alphabet_sha256"] = HashStrings([_options.Alphabet]),
                    ["blank_class_index"] = _options.BlankClassIndex,
                    ["input_height"] = _options.InputHeight,
                    ["input_width"] = _options.InputWidth,
                    ["input_channels"] = _options.InputChannels,
                    ["input_layout"] = _options.InputLayout.ToString(),
                    ["output_layout"] = _options.OutputLayout.ToString(),
                    ["expected_time_steps"] = _options.ExpectedTimeSteps,
                    ["normalization_mean"] = _options.NormalizeMean,
                    ["normalization_scale"] = _options.NormalizeScale,
                },
                OcrContract.Version),
            _options.Timeout);

        var response = await _runtime.RunAsync(request, cancellationToken).ConfigureAwait(false);
        if (!response.Succeeded || response.Execution is null)
        {
            var failure = ToOcrFailure(response.Error);
            return OcrCollections.Freeze(crops.Select(crop => new OcrRecognition(
                crop.RegionId,
                crop.SourceImage,
                Array.Empty<OcrRecognitionAlternative>(),
                0,
                failure)));
        }

        var classCount = _options.Alphabet.Length + 1;
        var denominator = checked(crops.Count * classCount);
        if (response.Execution.Output.Count == 0 || response.Execution.Output.Count % denominator != 0)
        {
            var failure = new OcrFailure(
                "OCR_MODEL_OUTPUT_INVALID",
                "error",
                "Errors.OCR_MODEL_OUTPUT_INVALID",
                "OCR model output is not a flattened [batch,time,class] tensor for the configured alphabet.",
                false,
                "select_compatible_model");
            return OcrCollections.Freeze(crops.Select(crop => new OcrRecognition(
                crop.RegionId,
                crop.SourceImage,
                Array.Empty<OcrRecognitionAlternative>(),
                response.Execution.Timing.InferenceMilliseconds / crops.Count,
                failure)));
        }

        var timeSteps = response.Execution.Output.Count / denominator;
        if (_options.ExpectedTimeSteps.HasValue && timeSteps != _options.ExpectedTimeSteps.Value)
        {
            var failure = new OcrFailure(
                "OCR_MODEL_OUTPUT_SHAPE_MISMATCH",
                "error",
                "Errors.OCR_MODEL_OUTPUT_SHAPE_MISMATCH",
                $"OCR model returned {timeSteps} time steps; {_options.ExpectedTimeSteps.Value} were declared.",
                false,
                "select_compatible_model");
            return OcrCollections.Freeze(crops.Select(crop => new OcrRecognition(
                crop.RegionId,
                crop.SourceImage,
                Array.Empty<OcrRecognitionAlternative>(),
                response.Execution.Timing.InferenceMilliseconds / crops.Count,
                failure)));
        }

        var valuesPerResult = checked(timeSteps * classCount);
        var output = response.Execution.Output.ToArray();
        var results = new OcrRecognition[crops.Count];
        for (var cropIndex = 0; cropIndex < crops.Count; cropIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var cropOutput = GetCropOutput(
                output,
                cropIndex,
                crops.Count,
                timeSteps,
                classCount,
                _options.OutputLayout);
            var decoded = CtcRecognitionDecoder.Decode(
                cropOutput,
                timeSteps,
                _options.Alphabet,
                _options.BlankClassIndex,
                _options.MaximumAlternatives);
            results[cropIndex] = new OcrRecognition(
                crops[cropIndex].RegionId,
                crops[cropIndex].SourceImage,
                OcrCollections.Freeze(decoded.Select(alternative => new OcrRecognitionAlternative(
                    alternative.Text,
                    alternative.Confidence,
                    crops[cropIndex].SourceImage))),
                response.Execution.Timing.InferenceMilliseconds / crops.Count);
        }

        return Array.AsReadOnly(results);
    }

    private static string CreateConfigurationFingerprint(LocalOnnxTextRecognizerOptions options) =>
        HashStrings([
            options.Alphabet,
            options.InputWidth.ToString(System.Globalization.CultureInfo.InvariantCulture),
            options.InputHeight.ToString(System.Globalization.CultureInfo.InvariantCulture),
            options.InputChannels.ToString(System.Globalization.CultureInfo.InvariantCulture),
            options.InputLayout.ToString(),
            options.OutputLayout.ToString(),
            options.ExpectedTimeSteps?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "dynamic",
            options.BlankClassIndex.ToString(System.Globalization.CultureInfo.InvariantCulture),
            options.InputName,
            options.OutputName,
            options.StageVersion,
            options.NormalizeMean.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            options.NormalizeScale.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        ]);

    private static ReadOnlySpan<float> GetCropOutput(
        float[] output,
        int cropIndex,
        int batchSize,
        int timeSteps,
        int classCount,
        OcrOutputLayout layout)
    {
        var valuesPerResult = checked(timeSteps * classCount);
        if (layout == OcrOutputLayout.BatchTimeClass)
        {
            return output.AsSpan(cropIndex * valuesPerResult, valuesPerResult);
        }

        var reordered = new float[valuesPerResult];
        for (var time = 0; time < timeSteps; time++)
        {
            var sourceOffset = checked(((time * batchSize) + cropIndex) * classCount);
            output.AsSpan(sourceOffset, classCount).CopyTo(reordered.AsSpan(time * classCount, classCount));
        }

        return reordered;
    }

    private static string HashStrings(IEnumerable<string> values)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var value in values)
        {
            hash.AppendData(Encoding.UTF8.GetBytes(value));
            hash.AppendData([0]);
        }

        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }

    private static OcrFailure ToOcrFailure(InferenceError? error) =>
        error is null
            ? new OcrFailure(
                "OCR_INFERENCE_FAILED",
                "error",
                "Errors.OCR_INFERENCE_FAILED",
                "The local inference runtime returned no OCR result.",
                true,
                "retry")
            : new OcrFailure(
                error.Code,
                error.Severity,
                error.UserMessageKey,
                error.TechnicalMessage,
                error.Recoverable,
                error.SuggestedAction);
}
