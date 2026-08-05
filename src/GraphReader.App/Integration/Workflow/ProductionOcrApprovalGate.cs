// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using System.Text.Json;
using System.IO;
using GraphReader.Inference;

namespace GraphReader.App.Integration.Workflow;

internal sealed record ProductionOcrGateEvidence(
    string EvaluatorSourceSha256,
    string SealedSplitSha256,
    string PredictionsSha256,
    string RuntimeResultsSha256,
    double ValidationExactMatch,
    double ValidationCer,
    double ValidationRoleAccuracy,
    double SealedTestExactMatch,
    double SealedTestCer,
    double SealedTestRoleAccuracy,
    double OnnxMaxAbsError,
    double DetectionExactRate,
    int MarkerCreationCount);

internal static class ProductionOcrApprovalGate
{
    internal const string FrozenEvaluatorSourceSha256 =
        "cc354ec53e4d0ecc5eab7dcf6243e5538e39f043057a949fd3d6ce84a83d50ee";
    private const int MaximumResourceBytes = 8 * 1024 * 1024;
    private static readonly HashSet<string> AllowedRoles = new(StringComparer.Ordinal)
    {
        "x_tick", "y_tick", "phase_header", "annotation", "participant", "other",
    };
    private static readonly HashSet<string> RequiredFamilies = new(StringComparer.Ordinal)
    {
        "integer", "decimal", "negative", "percentage", "ambiguity",
    };

    internal static ProductionOcrGateEvidence Validate(
        ResolvedProductionModel detectionModel,
        ResolvedProductionModel recognitionModel)
    {
        ArgumentNullException.ThrowIfNull(detectionModel);
        ArgumentNullException.ThrowIfNull(recognitionModel);
        VerifyChecksum(
            detectionModel.BenchmarkEvidencePath,
            detectionModel.BenchmarkEvidenceSha256,
            "OCR detection benchmark evidence");
        VerifyChecksum(
            recognitionModel.BenchmarkEvidencePath,
            recognitionModel.BenchmarkEvidenceSha256,
            "OCR recognition benchmark evidence");
        if (!string.Equals(
                detectionModel.BenchmarkEvidenceSha256,
                recognitionModel.BenchmarkEvidenceSha256,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                "OCR detection and recognition must bind the same checksum-exact public gate report.");
        }

        string alphabet = ReadRecognitionAlphabet(recognitionModel.ManifestPath);
        using JsonDocument document = JsonDocument.Parse(
            File.ReadAllBytes(detectionModel.BenchmarkEvidencePath),
            new JsonDocumentOptions { MaxDepth = 64 });
        JsonElement root = document.RootElement;
        RejectDuplicatePropertyNames(root, "OCR public gate evidence");
        RequireString(root, "schema", "graphreader.ocr-production-gate.v1", "OCR public gate evidence");
        RequireString(root, "profile", ProductionOcrAdapter.ApprovalBenchmarkProfile, "OCR public gate evidence");
        RequireString(root, "status", "pass", "OCR public gate evidence");
        RequireString(root, "scope", "public_synthetic_sealed", "OCR public gate evidence");
        RequireBoolean(root, "release_eligible", true, "OCR public gate evidence");
        RequireBoolean(root, "production_approval", true, "OCR public gate evidence");
        RequireBoolean(root, "private_data", false, "OCR public gate evidence");
        RequireBoolean(root, "chandler_used", false, "OCR public gate evidence");
        RequireBoolean(root, "marker_creation_evaluated", true, "OCR public gate evidence");
        RequireString(root, "provider", "cpu", "OCR public gate evidence");
        RequireString(root, "coordinate_space", "original_pixels", "OCR public gate evidence");
        RequireMatchingSha256(
            root,
            "detection_model_sha256",
            detectionModel.Identity.Sha256,
            "OCR public gate evidence");
        RequireMatchingSha256(
            root,
            "recognition_model_sha256",
            recognitionModel.Identity.Sha256,
            "OCR public gate evidence");

        JsonElement resources = RequiredObject(root, "reviewed_resources", "OCR public gate evidence");
        EmbeddedResource evaluator = ReadEmbeddedResource(
            resources,
            "evaluator_source",
            "text/x-python");
        EmbeddedResource split = ReadEmbeddedResource(resources, "sealed_split", "application/json");
        EmbeddedResource predictions = ReadEmbeddedResource(resources, "predictions", "application/json");
        EmbeddedResource runtime = ReadEmbeddedResource(resources, "runtime_results", "application/json");
        if (!string.Equals(
                evaluator.Sha256,
                FrozenEvaluatorSourceSha256,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                "OCR public gate evidence does not contain the frozen reviewed evaluator source.");
        }

        Dictionary<string, SplitCase> cases = ReadSplit(
            split.Bytes,
            evaluator.Sha256,
            alphabet);
        Dictionary<string, Prediction> predicted = ReadPredictions(
            predictions.Bytes,
            split.Sha256,
            detectionModel.Identity.Sha256,
            recognitionModel.Identity.Sha256,
            alphabet);
        if (!cases.Keys.ToHashSet(StringComparer.Ordinal)
                .SetEquals(predicted.Keys))
        {
            throw new InvalidDataException(
                "OCR prediction record IDs do not match the frozen sealed split.");
        }

        PartitionMetrics validation = EvaluatePartition(cases, predicted, "validation");
        PartitionMetrics sealedTest = EvaluatePartition(cases, predicted, "sealed_test");
        int detectionExact = cases.Count(pair =>
            predicted[pair.Key].DetectedRegionCount == pair.Value.ExpectedRegionCount &&
            predicted[pair.Key].FalseRegionCount == 0);
        double detectionExactRate = detectionExact / (double)cases.Count;
        int markerCreationCount = predicted.Values.Sum(static value => value.MarkerCreationCount);
        double onnxMaxAbsError = ReadRuntimeResults(
            runtime.Bytes,
            evaluator.Sha256,
            split.Sha256,
            predictions.Sha256,
            detectionModel.Identity.Sha256,
            recognitionModel.Identity.Sha256);
        var evidence = new ProductionOcrGateEvidence(
            evaluator.Sha256,
            split.Sha256,
            predictions.Sha256,
            runtime.Sha256,
            validation.ExactMatch,
            validation.Cer,
            validation.RoleAccuracy,
            sealedTest.ExactMatch,
            sealedTest.Cer,
            sealedTest.RoleAccuracy,
            onnxMaxAbsError,
            detectionExactRate,
            markerCreationCount);
        ValidateThresholds(evidence);
        MatchReportedMetrics(root, evidence);
        RequireApprovalBenchmark(detectionModel, evidence);
        RequireApprovalBenchmark(recognitionModel, evidence);
        return evidence;
    }

    private static Dictionary<string, SplitCase> ReadSplit(
        byte[] bytes,
        string evaluatorSha256,
        string alphabet)
    {
        using JsonDocument document = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 32 });
        JsonElement root = document.RootElement;
        RejectDuplicatePropertyNames(root, "OCR sealed split");
        RequireString(root, "schema", "graphreader.ocr-sealed-split.v1", "OCR sealed split");
        RequireString(root, "profile", ProductionOcrAdapter.ApprovalBenchmarkProfile, "OCR sealed split");
        RequireString(root, "scope", "public_synthetic", "OCR sealed split");
        RequireBoolean(root, "sealed", true, "OCR sealed split");
        RequireBoolean(root, "selection_locked_before_inference", true, "OCR sealed split");
        RequireBoolean(root, "private_data", false, "OCR sealed split");
        RequireBoolean(root, "chandler_used", false, "OCR sealed split");
        RequireMatchingSha256(root, "evaluator_source_sha256", evaluatorSha256, "OCR sealed split");

        var result = new Dictionary<string, SplitCase>(StringComparer.Ordinal);
        var families = new HashSet<string>(StringComparer.Ordinal);
        int validationText = 0;
        int sealedText = 0;
        int exclusions = 0;
        foreach (JsonElement item in RequiredArray(root, "cases", "OCR sealed split").EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("Every OCR sealed case must be an object.");
            }

            string caseId = RequiredString(item, "case_id", "OCR sealed case");
            string partition = RequiredString(item, "partition", "OCR sealed case");
            string kind = RequiredString(item, "kind", "OCR sealed case");
            string family = RequiredString(item, "family", "OCR sealed case");
            string truth = RequiredStringAllowEmpty(item, "truth_text", "OCR sealed case");
            string role = RequiredString(item, "truth_role", "OCR sealed case");
            int expectedRegions = RequiredNonNegativeInt32(
                item,
                "expected_region_count",
                "OCR sealed case");
            RequireSha256(item, "source_sha256", "OCR sealed case");
            if (partition is not ("validation" or "sealed_test") ||
                kind is not ("text" or "exclusion") ||
                !AllowedRoles.Contains(role) ||
                truth.Length > 64 ||
                truth.Any(character => !alphabet.Contains(character)) ||
                !result.TryAdd(caseId, new SplitCase(
                    partition,
                    kind,
                    family,
                    truth,
                    role,
                    expectedRegions)))
            {
                throw new InvalidDataException("OCR sealed cases contain invalid or duplicate values.");
            }

            if (kind == "text")
            {
                if (truth.Length == 0 || expectedRegions != 1 ||
                    !TruthMatchesFamily(truth, family))
                {
                    throw new InvalidDataException(
                        "OCR text cases require nonempty truth and exactly one expected region.");
                }

                families.Add(family);
                if (partition == "validation")
                {
                    validationText++;
                }
                else
                {
                    sealedText++;
                }
            }
            else
            {
                if (truth.Length != 0 || role != "other" || expectedRegions != 0)
                {
                    throw new InvalidDataException(
                        "OCR exclusion cases require empty truth, role other, and zero expected regions.");
                }

                exclusions++;
            }
        }

        if (validationText < 100 || sealedText < 100 || exclusions < 20 ||
            !RequiredFamilies.IsSubsetOf(families))
        {
            throw new InvalidDataException(
                "OCR sealed split requires 100 text cases per partition, 20 exclusions, and all numeric ambiguity families.");
        }

        return result;
    }

    private static bool TruthMatchesFamily(string truth, string family) => family switch
    {
        "integer" => truth.All(char.IsAsciiDigit),
        "decimal" => truth.Count(static character => character == '.') == 1 &&
            truth.Where(static character => character != '.').All(char.IsAsciiDigit),
        "negative" => truth.Length > 1 && truth[0] == '-' &&
            truth[1..].All(char.IsAsciiDigit),
        "percentage" => truth.Length > 1 && truth[^1] == '%' &&
            truth[..^1].All(char.IsAsciiDigit),
        "ambiguity" => truth.Any(static character => character is 'O' or 'o' or 'l' or 'I'),
        _ => false,
    };

    private static Dictionary<string, Prediction> ReadPredictions(
        byte[] bytes,
        string splitSha256,
        string detectionModelSha256,
        string recognitionModelSha256,
        string alphabet)
    {
        using JsonDocument document = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 32 });
        JsonElement root = document.RootElement;
        RejectDuplicatePropertyNames(root, "OCR predictions");
        RequireString(root, "schema", "graphreader.ocr-predictions.v1", "OCR predictions");
        RequireString(root, "profile", ProductionOcrAdapter.ApprovalBenchmarkProfile, "OCR predictions");
        RequireString(root, "provider", "cpu", "OCR predictions");
        RequireMatchingSha256(root, "sealed_split_sha256", splitSha256, "OCR predictions");
        RequireMatchingSha256(root, "detection_model_sha256", detectionModelSha256, "OCR predictions");
        RequireMatchingSha256(root, "recognition_model_sha256", recognitionModelSha256, "OCR predictions");
        var result = new Dictionary<string, Prediction>(StringComparer.Ordinal);
        foreach (JsonElement item in RequiredArray(root, "records", "OCR predictions").EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("Every OCR prediction must be an object.");
            }

            string caseId = RequiredString(item, "case_id", "OCR prediction");
            string text = RequiredStringAllowEmpty(item, "predicted_text", "OCR prediction");
            string role = RequiredString(item, "predicted_role", "OCR prediction");
            if (text.Length > 64 ||
                text.Any(character => !alphabet.Contains(character)) ||
                !AllowedRoles.Contains(role) ||
                !result.TryAdd(caseId, new Prediction(
                    text,
                    role,
                    RequiredNonNegativeInt32(item, "detected_region_count", "OCR prediction"),
                    RequiredNonNegativeInt32(item, "false_region_count", "OCR prediction"),
                    RequiredNonNegativeInt32(item, "marker_creation_count", "OCR prediction"))))
            {
                throw new InvalidDataException("OCR predictions contain invalid or duplicate values.");
            }
        }

        return result;
    }

    private static double ReadRuntimeResults(
        byte[] bytes,
        string evaluatorSha256,
        string splitSha256,
        string predictionsSha256,
        string detectionModelSha256,
        string recognitionModelSha256)
    {
        using JsonDocument document = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 32 });
        JsonElement root = document.RootElement;
        RejectDuplicatePropertyNames(root, "OCR runtime results");
        RequireString(root, "schema", "graphreader.ocr-runtime-results.v1", "OCR runtime results");
        RequireString(root, "profile", ProductionOcrAdapter.ApprovalBenchmarkProfile, "OCR runtime results");
        RequireString(root, "provider", "cpu", "OCR runtime results");
        RequireBoolean(root, "detection_executed", true, "OCR runtime results");
        RequireBoolean(root, "recognition_executed", true, "OCR runtime results");
        RequireMatchingSha256(root, "evaluator_source_sha256", evaluatorSha256, "OCR runtime results");
        RequireMatchingSha256(root, "sealed_split_sha256", splitSha256, "OCR runtime results");
        RequireMatchingSha256(root, "predictions_sha256", predictionsSha256, "OCR runtime results");
        RequireMatchingSha256(root, "detection_model_sha256", detectionModelSha256, "OCR runtime results");
        RequireMatchingSha256(root, "recognition_model_sha256", recognitionModelSha256, "OCR runtime results");
        double detectionMaximum = MaximumAbsoluteError(
            RequiredArray(root, "detection_parity", "OCR runtime results"),
            "OCR detection parity");
        double recognitionMaximum = MaximumAbsoluteError(
            RequiredArray(root, "recognition_parity", "OCR runtime results"),
            "OCR recognition parity");
        return Math.Max(detectionMaximum, recognitionMaximum);
    }

    private static double MaximumAbsoluteError(JsonElement values, string label)
    {
        JsonElement[] pairs = values.EnumerateArray().ToArray();
        if (pairs.Length < 16)
        {
            throw new InvalidDataException($"{label} requires at least 16 direct value pairs.");
        }

        double maximum = 0;
        foreach (JsonElement pair in pairs)
        {
            double reference = RequiredFiniteDouble(pair, "reference", label);
            double onnx = RequiredFiniteDouble(pair, "onnx", label);
            maximum = Math.Max(maximum, Math.Abs(reference - onnx));
        }

        return maximum;
    }

    private static PartitionMetrics EvaluatePartition(
        IReadOnlyDictionary<string, SplitCase> cases,
        IReadOnlyDictionary<string, Prediction> predictions,
        string partition)
    {
        KeyValuePair<string, SplitCase>[] selected = cases
            .Where(pair => pair.Value.Partition == partition && pair.Value.Kind == "text")
            .ToArray();
        long truthCharacters = selected.Sum(pair => pair.Value.TruthText.Length);
        if (selected.Length == 0 || truthCharacters == 0)
        {
            throw new InvalidDataException($"OCR partition '{partition}' has no evaluable text.");
        }

        int exact = selected.Count(pair =>
            string.Equals(pair.Value.TruthText, predictions[pair.Key].Text, StringComparison.Ordinal));
        int roles = selected.Count(pair =>
            string.Equals(pair.Value.TruthRole, predictions[pair.Key].Role, StringComparison.Ordinal));
        long edits = selected.Sum(pair =>
            EditDistance(pair.Value.TruthText, predictions[pair.Key].Text));
        return new PartitionMetrics(
            exact / (double)selected.Length,
            edits / (double)truthCharacters,
            roles / (double)selected.Length);
    }

    private static int EditDistance(string expected, string actual)
    {
        var previous = Enumerable.Range(0, actual.Length + 1).ToArray();
        var current = new int[actual.Length + 1];
        for (int expectedIndex = 1; expectedIndex <= expected.Length; expectedIndex++)
        {
            current[0] = expectedIndex;
            for (int actualIndex = 1; actualIndex <= actual.Length; actualIndex++)
            {
                current[actualIndex] = Math.Min(
                    Math.Min(current[actualIndex - 1] + 1, previous[actualIndex] + 1),
                    previous[actualIndex - 1] +
                    (expected[expectedIndex - 1] == actual[actualIndex - 1] ? 0 : 1));
            }

            (previous, current) = (current, previous);
        }

        return previous[actual.Length];
    }

    private static void ValidateThresholds(ProductionOcrGateEvidence evidence)
    {
        if (evidence.ValidationExactMatch < 0.90 || evidence.ValidationCer > 0.05 ||
            evidence.ValidationRoleAccuracy < 0.90 || evidence.SealedTestExactMatch < 0.90 ||
            evidence.SealedTestCer > 0.05 || evidence.SealedTestRoleAccuracy < 0.90 ||
            evidence.OnnxMaxAbsError > 1e-4 || evidence.DetectionExactRate != 1 ||
            evidence.MarkerCreationCount != 0)
        {
            throw new InvalidDataException(
                "OCR direct resources do not meet the fixed production approval thresholds.");
        }
    }

    private static void MatchReportedMetrics(JsonElement root, ProductionOcrGateEvidence evidence)
    {
        RequireExactDouble(root, "validation_exact_match", evidence.ValidationExactMatch, "OCR public gate evidence");
        RequireExactDouble(root, "validation_cer", evidence.ValidationCer, "OCR public gate evidence");
        RequireExactDouble(root, "validation_role_accuracy", evidence.ValidationRoleAccuracy, "OCR public gate evidence");
        RequireExactDouble(root, "sealed_test_exact_match", evidence.SealedTestExactMatch, "OCR public gate evidence");
        RequireExactDouble(root, "sealed_test_cer", evidence.SealedTestCer, "OCR public gate evidence");
        RequireExactDouble(root, "sealed_test_role_accuracy", evidence.SealedTestRoleAccuracy, "OCR public gate evidence");
        RequireExactDouble(root, "onnx_max_abs_error", evidence.OnnxMaxAbsError, "OCR public gate evidence");
        RequireExactDouble(root, "detection_exact_rate", evidence.DetectionExactRate, "OCR public gate evidence");
        if (RequiredNonNegativeInt32(root, "marker_creation_count", "OCR public gate evidence") !=
            evidence.MarkerCreationCount)
        {
            throw new InvalidDataException(
                "OCR public gate marker creation count does not match direct predictions.");
        }

        RequireMatchingSha256(root, "evaluator_source_sha256", evidence.EvaluatorSourceSha256, "OCR public gate evidence");
        RequireMatchingSha256(root, "sealed_split_sha256", evidence.SealedSplitSha256, "OCR public gate evidence");
        RequireMatchingSha256(root, "predictions_sha256", evidence.PredictionsSha256, "OCR public gate evidence");
        RequireMatchingSha256(root, "runtime_results_sha256", evidence.RuntimeResultsSha256, "OCR public gate evidence");
    }

    private static void RequireApprovalBenchmark(
        ResolvedProductionModel model,
        ProductionOcrGateEvidence evidence)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(model.ManifestPath));
        JsonElement[] approvals = RequiredArray(
                document.RootElement,
                "benchmarks",
                $"OCR model '{model.Identity.ModelId}'")
            .EnumerateArray()
            .Where(static benchmark => benchmark.ValueKind == JsonValueKind.Object &&
                benchmark.TryGetProperty("production_approval", out JsonElement approval) &&
                approval.ValueKind == JsonValueKind.True)
            .ToArray();
        if (approvals.Length != 1)
        {
            throw new InvalidDataException(
                $"OCR model '{model.Identity.ModelId}' must contain one production approval benchmark.");
        }

        JsonElement selected = approvals[0];
        string label = $"OCR model '{model.Identity.ModelId}' benchmark";
        RequireString(selected, "profile", ProductionOcrAdapter.ApprovalBenchmarkProfile, label);
        RequireString(selected, "status", "pass", label);
        RequireBoolean(selected, "release_eligible", true, label);
        RequireMatchingSha256(selected, "evidence_sha256", model.BenchmarkEvidenceSha256, label);
        RequireMatchingSha256(selected, "evaluator_source_sha256", evidence.EvaluatorSourceSha256, label);
        RequireMatchingSha256(selected, "sealed_split_sha256", evidence.SealedSplitSha256, label);
        RequireMatchingSha256(selected, "predictions_sha256", evidence.PredictionsSha256, label);
        RequireMatchingSha256(selected, "runtime_results_sha256", evidence.RuntimeResultsSha256, label);
        RequireExactDouble(selected, "sealed_test_exact_match", evidence.SealedTestExactMatch, label);
        RequireExactDouble(selected, "sealed_test_cer", evidence.SealedTestCer, label);
        RequireExactDouble(selected, "onnx_max_abs_error", evidence.OnnxMaxAbsError, label);
    }

    private static string ReadRecognitionAlphabet(string manifestPath)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement outputs = RequiredArray(document.RootElement, "outputs", "OCR recognition manifest");
        if (outputs.GetArrayLength() != 1 || outputs[0].ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("OCR recognition manifest requires one output.");
        }

        return RequiredString(outputs[0], "alphabet", "OCR recognition output");
    }

    private static EmbeddedResource ReadEmbeddedResource(
        JsonElement resources,
        string propertyName,
        string mediaType)
    {
        JsonElement resource = RequiredObject(resources, propertyName, "OCR reviewed resources");
        RequireString(resource, "media_type", mediaType, $"OCR resource '{propertyName}'");
        RequireString(resource, "encoding", "base64", $"OCR resource '{propertyName}'");
        string sha256 = RequireSha256(resource, "sha256", $"OCR resource '{propertyName}'");
        string content = RequiredString(resource, "content_base64", $"OCR resource '{propertyName}'");
        if (content.Length > ((MaximumResourceBytes + 2) / 3 * 4) + 4)
        {
            throw new InvalidDataException($"OCR resource '{propertyName}' exceeds the size limit.");
        }

        byte[] bytes;
        try
        {
            bytes = Convert.FromBase64String(content);
        }
        catch (FormatException exception)
        {
            throw new InvalidDataException($"OCR resource '{propertyName}' is not valid base64.", exception);
        }

        if (bytes.Length == 0 || bytes.Length > MaximumResourceBytes ||
            !string.Equals(
                Convert.ToHexStringLower(SHA256.HashData(bytes)),
                sha256,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"OCR resource '{propertyName}' failed checksum validation.");
        }

        return new EmbeddedResource(sha256.ToLowerInvariant(), bytes);
    }

    private static JsonElement RequiredObject(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be an object.");
        }

        return value;
    }

    private static JsonElement RequiredArray(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be an array.");
        }

        return value;
    }

    private static string RequiredString(JsonElement parent, string propertyName, string label)
    {
        string value = RequiredStringAllowEmpty(parent, propertyName, label);
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be nonempty.");
        }

        return value;
    }

    private static string RequiredStringAllowEmpty(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.String)
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be a string.");
        }

        return value.GetString()!;
    }

    private static void RequireString(
        JsonElement parent,
        string propertyName,
        string expected,
        string label)
    {
        string actual = RequiredString(parent, propertyName, label);
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"{label} field '{propertyName}' must be '{expected}', found '{actual}'.");
        }
    }

    private static void RequireBoolean(
        JsonElement parent,
        string propertyName,
        bool expected,
        string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind is not (JsonValueKind.True or JsonValueKind.False) ||
            value.GetBoolean() != expected)
        {
            throw new InvalidDataException(
                $"{label} field '{propertyName}' must be {expected.ToString().ToLowerInvariant()}.");
        }
    }

    private static int RequiredNonNegativeInt32(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Number ||
            !value.TryGetInt32(out int result) || result < 0)
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be a nonnegative int32.");
        }

        return result;
    }

    private static double RequiredFiniteDouble(JsonElement parent, string propertyName, string label)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Number ||
            !value.TryGetDouble(out double result) || !double.IsFinite(result))
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be finite.");
        }

        return result;
    }

    private static void RequireExactDouble(
        JsonElement parent,
        string propertyName,
        double expected,
        string label)
    {
        double actual = RequiredFiniteDouble(parent, propertyName, label);
        if (Math.Abs(actual - expected) > 1e-12)
        {
            throw new InvalidDataException(
                $"{label} field '{propertyName}' does not match metrics derived from direct resources.");
        }
    }

    private static string RequireSha256(JsonElement parent, string propertyName, string label)
    {
        string value = RequiredString(parent, propertyName, label);
        if (value.Length != 64 || !value.All(Uri.IsHexDigit))
        {
            throw new InvalidDataException($"{label} field '{propertyName}' must be SHA-256.");
        }

        return value;
    }

    private static void RequireMatchingSha256(
        JsonElement parent,
        string propertyName,
        string expected,
        string label)
    {
        string actual = RequireSha256(parent, propertyName, label);
        if (!string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"{label} field '{propertyName}' does not match its checksum-bound resource.");
        }
    }

    private static void RejectDuplicatePropertyNames(JsonElement element, string label)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (JsonProperty property in element.EnumerateObject())
            {
                if (!names.Add(property.Name))
                {
                    throw new InvalidDataException(
                        $"{label} contains duplicate property '{property.Name}'.");
                }

                RejectDuplicatePropertyNames(property.Value, label);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                RejectDuplicatePropertyNames(item, label);
            }
        }
    }

    private static void VerifyChecksum(string path, string expectedSha256, string label)
    {
        if (!File.Exists(path) || !string.Equals(
                Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))),
                expectedSha256,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"The checksum-resolved {label} is missing or changed.");
        }
    }

    private sealed record EmbeddedResource(string Sha256, byte[] Bytes);

    private sealed record SplitCase(
        string Partition,
        string Kind,
        string Family,
        string TruthText,
        string TruthRole,
        int ExpectedRegionCount);

    private sealed record Prediction(
        string Text,
        string Role,
        int DetectedRegionCount,
        int FalseRegionCount,
        int MarkerCreationCount);

    private sealed record PartitionMetrics(double ExactMatch, double Cer, double RoleAccuracy);
}
