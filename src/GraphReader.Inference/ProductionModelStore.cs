// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Security.Cryptography;
using System.Text.Json;

namespace GraphReader.Inference;

/// <summary>
/// Resolves checksum-bound, benchmark-approved ONNX models from an offline store.
/// The store layout is <c>{root}/{model-id}/{version}/manifest.json</c>.
/// </summary>
public sealed class ProductionModelStore
{
    private static readonly HashSet<string> AllowedRootProperties = new(StringComparer.Ordinal)
    {
        "manifest_version", "model_id", "model_version", "task", "source", "license", "sha256",
        "files", "inputs", "outputs", "preprocessing", "postprocessing", "commercial_use",
        "redistribution", "providers", "benchmarks"
    };

    private static readonly HashSet<string> AllowedTasks = new(StringComparer.Ordinal)
    {
        "super_resolution", "ocr_detection", "ocr_recognition", "marker_center",
        "marker_classifier", "panelization"
    };

    private static readonly HashSet<string> AllowedProviders = new(StringComparer.Ordinal)
    {
        "cpu", "directml", "winml", "cuda", "openvino", "vulkan"
    };

    private readonly string _root;
    private readonly IExecutionProviderDiscovery _providerDiscovery;

    public ProductionModelStore(string root, IExecutionProviderDiscovery? providerDiscovery = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        _root = Path.GetFullPath(root);
        _providerDiscovery = providerDiscovery ?? new OrtExecutionProviderDiscovery();
    }

    public async ValueTask<ResolvedProductionModel> ResolveAsync(
        string modelId,
        string version,
        InferenceProvider? requiredProvider,
        CancellationToken cancellationToken)
    {
        ValidatePathComponent(modelId, nameof(modelId));
        ValidatePathComponent(version, nameof(version));
        if (requiredProvider is InferenceProvider.Fake)
        {
            throw new ArgumentOutOfRangeException(nameof(requiredProvider), "Fake is not a production execution provider.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        var modelDirectory = ResolveUnderRoot(Path.Combine(modelId, version));
        var manifestPath = Path.Combine(modelDirectory, "manifest.json");
        if (!File.Exists(manifestPath))
        {
            throw Failure("MODEL_MANIFEST_NOT_FOUND", $"Production model manifest was not found: {manifestPath}");
        }

        JsonDocument document;
        try
        {
            await using var stream = OpenRead(manifestPath);
            document = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        catch (JsonException exception)
        {
            throw Failure("MODEL_MANIFEST_INVALID", $"Production model manifest is malformed: {exception.Message}", exception);
        }

        using (document)
        {
            var root = document.RootElement;
            ValidateManifestShape(root, modelId, version);
            ValidateSource(root.GetProperty("source"));
            var noticePath = ValidateLicense(root.GetProperty("license"));

            if (!root.GetProperty("commercial_use").GetBoolean() || !root.GetProperty("redistribution").GetBoolean())
            {
                throw Failure("MODEL_REDISTRIBUTION_NOT_APPROVED", "Production models require commercial-use and redistribution approval.");
            }

            var providers = ValidateProviders(root.GetProperty("providers"), requiredProvider);
            var approvalEvidence = ValidateApproval(root.GetProperty("benchmarks"));
            var payloads = ReadPayloads(root, modelDirectory);
            if (payloads.Count(path => string.Equals(Path.GetExtension(path), ".onnx", StringComparison.OrdinalIgnoreCase)) != 1)
            {
                throw Failure("MODEL_PAYLOAD_INVALID", "A production ONNX manifest must identify exactly one .onnx payload.");
            }

            var expectedHashes = ReadPayloadHashes(root, payloads, modelDirectory);
            foreach (var payload in payloads)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!File.Exists(payload))
                {
                    throw Failure("MODEL_PAYLOAD_MISSING", $"Manifest payload does not exist: {payload}");
                }

                RejectReparsePoint(payload, "MODEL_PAYLOAD_INVALID");
                await VerifyHashAsync(payload, expectedHashes[payload], "MODEL_PAYLOAD_CHECKSUM_MISMATCH", cancellationToken)
                    .ConfigureAwait(false);
            }

            var resolvedNotice = ResolveUnderRoot(noticePath);
            if (!File.Exists(resolvedNotice) || new FileInfo(resolvedNotice).Length == 0)
            {
                throw Failure("MODEL_NOTICE_MISSING", $"Reviewed model notice is missing or empty: {resolvedNotice}");
            }

            RejectReparsePoint(resolvedNotice, "MODEL_NOTICE_INVALID");
            var evidencePath = ResolveUnderRoot(approvalEvidence.Path);
            if (!File.Exists(evidencePath))
            {
                throw Failure("MODEL_BENCHMARK_EVIDENCE_MISSING", $"Approval evidence does not exist: {evidencePath}");
            }

            RejectReparsePoint(evidencePath, "MODEL_BENCHMARK_INVALID");
            await VerifyHashAsync(
                evidencePath,
                approvalEvidence.Sha256,
                "MODEL_BENCHMARK_CHECKSUM_MISMATCH",
                cancellationToken).ConfigureAwait(false);

            ValidateNoExtraFiles(modelDirectory, manifestPath, payloads, resolvedNotice, evidencePath);

            var onnxPath = payloads.Single(path =>
                string.Equals(Path.GetExtension(path), ".onnx", StringComparison.OrdinalIgnoreCase));
            var identity = new ModelIdentity(
                modelId,
                version,
                expectedHashes[onnxPath].ToUpperInvariant(),
                onnxPath);
            identity.Validate();
            return new ResolvedProductionModel(
                identity,
                root.GetProperty("task").GetString()!,
                new ReadOnlyCollection<InferenceProvider>(providers.ToArray()),
                manifestPath,
                resolvedNotice,
                evidencePath);
        }
    }

    private static void ValidateManifestShape(JsonElement root, string expectedModelId, string expectedVersion)
    {
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw Failure("MODEL_MANIFEST_INVALID", "The production model manifest root must be an object.");
        }

        foreach (var property in root.EnumerateObject())
        {
            if (!AllowedRootProperties.Contains(property.Name))
            {
                throw Failure("MODEL_MANIFEST_INVALID", $"Unsupported manifest property: {property.Name}");
            }
        }

        var required = new[]
        {
            "manifest_version", "model_id", "model_version", "task", "source", "license", "sha256",
            "files", "inputs", "outputs", "commercial_use", "redistribution", "providers", "benchmarks"
        };
        foreach (var name in required)
        {
            if (!root.TryGetProperty(name, out _))
            {
                throw Failure("MODEL_MANIFEST_INVALID", $"Required manifest property is missing: {name}");
            }
        }

        if (root.GetProperty("manifest_version").ValueKind != JsonValueKind.Number ||
            !root.GetProperty("manifest_version").TryGetInt32(out var manifestVersion) || manifestVersion != 1)
        {
            throw Failure("MODEL_MANIFEST_INVALID", "Only model manifest version 1 is supported.");
        }

        RequireExactString(root, "model_id", expectedModelId);
        RequireExactString(root, "model_version", expectedVersion);
        var task = RequireNonEmptyString(root, "task");
        if (!AllowedTasks.Contains(task))
        {
            throw Failure("MODEL_MANIFEST_INVALID", $"Unsupported model task: {task}");
        }

        RequireSha256(root, "sha256");
        RequireNonEmptyArray(root, "inputs");
        RequireNonEmptyArray(root, "outputs");
        RequireBoolean(root, "commercial_use");
        RequireBoolean(root, "redistribution");
    }

    private static void ValidateSource(JsonElement source)
    {
        RequireObjectWithExactProperties(source, "source", "name", "url", "revision");
        RequireNonEmptyString(source, "name");
        RequireNonEmptyString(source, "url");
        RequireNonEmptyString(source, "revision");
    }

    private static string ValidateLicense(JsonElement license)
    {
        RequireObjectWithExactProperties(license, "license", "spdx", "notice_path", "reviewed");
        RequireNonEmptyString(license, "spdx");
        var notice = RequireNonEmptyString(license, "notice_path");
        RequireBoolean(license, "reviewed");
        if (!license.GetProperty("reviewed").GetBoolean())
        {
            throw Failure("MODEL_LICENSE_NOT_REVIEWED", "The model license and notice must be reviewed before production use.");
        }

        return notice;
    }

    private ReadOnlyCollection<InferenceProvider> ValidateProviders(JsonElement element, InferenceProvider? requiredProvider)
    {
        RequireNonEmptyArray(element, "providers");
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in element.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(item.GetString()))
            {
                throw Failure("MODEL_PROVIDER_INVALID", "Every provider must be a non-empty string.");
            }

            var provider = item.GetString()!;
            if (!AllowedProviders.Contains(provider) || !names.Add(provider))
            {
                throw Failure("MODEL_PROVIDER_INVALID", $"Unsupported or duplicate provider: {provider}");
            }
        }

        if (!names.Contains("cpu"))
        {
            throw Failure("MODEL_CPU_FALLBACK_UNAVAILABLE", "Every production ONNX model must declare CPU compatibility.");
        }

        var resolved = new List<InferenceProvider> { InferenceProvider.Cpu };
        if (names.Contains("directml"))
        {
            IReadOnlyList<string> available;
            try
            {
                available = _providerDiscovery.GetAvailableProviders();
            }
            catch (Exception exception)
            {
                if (requiredProvider == InferenceProvider.DirectMl)
                {
                    throw Failure("MODEL_PROVIDER_UNAVAILABLE", "DirectML discovery failed.", exception);
                }

                available = Array.Empty<string>();
            }

            if (available.Contains("DmlExecutionProvider", StringComparer.OrdinalIgnoreCase))
            {
                resolved.Insert(0, InferenceProvider.DirectMl);
            }
            else if (requiredProvider == InferenceProvider.DirectMl)
            {
                throw Failure("MODEL_PROVIDER_UNAVAILABLE", "The manifest declares DirectML, but this runtime cannot provide it.");
            }
        }
        else if (requiredProvider == InferenceProvider.DirectMl)
        {
            throw Failure("MODEL_PROVIDER_UNAVAILABLE", "The model manifest does not declare DirectML compatibility.");
        }

        return resolved.AsReadOnly();
    }

    private static ApprovalEvidence ValidateApproval(JsonElement benchmarks)
    {
        RequireNonEmptyArray(benchmarks, "benchmarks");
        foreach (var benchmark in benchmarks.EnumerateArray())
        {
            if (benchmark.ValueKind != JsonValueKind.Object ||
                !benchmark.TryGetProperty("production_approval", out var approval) ||
                approval.ValueKind != JsonValueKind.True)
            {
                continue;
            }

            if (!benchmark.TryGetProperty("release_eligible", out var eligible) || eligible.ValueKind != JsonValueKind.True ||
                !benchmark.TryGetProperty("status", out var status) || status.ValueKind != JsonValueKind.String ||
                !string.Equals(status.GetString(), "pass", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var profile = RequireNonEmptyString(benchmark, "profile");
            var evidencePath = RequireNonEmptyString(benchmark, "evidence_path");
            var evidenceSha = RequireSha256(benchmark, "evidence_sha256");
            return new ApprovalEvidence(profile, evidencePath, evidenceSha);
        }

        throw Failure(
            "MODEL_NOT_APPROVED",
            "No benchmark entry has status pass, release_eligible true, production_approval true, and checksum-bound evidence.");
    }

    private static List<string> ReadPayloads(JsonElement root, string modelDirectory)
    {
        var files = root.GetProperty("files");
        RequireNonEmptyArray(files, "files");
        var paths = new List<string>();
        var unique = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in files.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(item.GetString()))
            {
                throw Failure("MODEL_PAYLOAD_INVALID", "Every manifest file must be a non-empty relative path.");
            }

            var path = ResolveUnder(modelDirectory, item.GetString()!, "MODEL_PAYLOAD_INVALID");
            if (!unique.Add(path))
            {
                throw Failure("MODEL_PAYLOAD_INVALID", $"Duplicate manifest payload: {item.GetString()}");
            }

            paths.Add(path);
        }

        return paths;
    }

    private static Dictionary<string, string> ReadPayloadHashes(
        JsonElement root,
        List<string> payloads,
        string modelDirectory)
    {
        var hashes = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (root.TryGetProperty("preprocessing", out var preprocessing) &&
            preprocessing.ValueKind == JsonValueKind.Object &&
            preprocessing.TryGetProperty("model_payload_sha256", out var map))
        {
            if (map.ValueKind != JsonValueKind.Object)
            {
                throw Failure("MODEL_MANIFEST_INVALID", "preprocessing.model_payload_sha256 must be an object.");
            }

            foreach (var item in map.EnumerateObject())
            {
                var path = ResolveUnder(modelDirectory, item.Name, "MODEL_PAYLOAD_INVALID");
                if (!payloads.Contains(path, StringComparer.OrdinalIgnoreCase) || hashes.ContainsKey(path) ||
                    item.Value.ValueKind != JsonValueKind.String || !IsSha256(item.Value.GetString()))
                {
                    throw Failure("MODEL_PAYLOAD_INVALID", $"Invalid or unlisted payload checksum entry: {item.Name}");
                }

                hashes.Add(path, item.Value.GetString()!);
            }
        }

        if (hashes.Count == 0 && payloads.Count == 1)
        {
            hashes.Add(payloads[0], root.GetProperty("sha256").GetString()!);
        }

        if (hashes.Count != payloads.Count)
        {
            throw Failure("MODEL_PAYLOAD_INVALID", "Every payload must have exactly one SHA-256 checksum.");
        }

        return hashes;
    }

    private static void ValidateNoExtraFiles(
        string modelDirectory,
        string manifestPath,
        IReadOnlyList<string> payloads,
        string noticePath,
        string evidencePath)
    {
        var allowed = new HashSet<string>(payloads.Select(Path.GetFullPath), StringComparer.OrdinalIgnoreCase)
        {
            Path.GetFullPath(manifestPath)
        };
        if (IsUnder(modelDirectory, noticePath))
        {
            allowed.Add(Path.GetFullPath(noticePath));
        }

        if (IsUnder(modelDirectory, evidencePath))
        {
            allowed.Add(Path.GetFullPath(evidencePath));
        }

        foreach (var file in Directory.EnumerateFiles(modelDirectory, "*", SearchOption.AllDirectories))
        {
            if (!allowed.Contains(Path.GetFullPath(file)))
            {
                throw Failure("MODEL_STORE_EXTRA_FILE", $"Unlisted file exists in the production model directory: {file}");
            }
        }
    }

    private string ResolveUnderRoot(string relativePath) => ResolveUnder(_root, relativePath, "MODEL_PATH_INVALID");

    private static string ResolveUnder(string root, string relativePath, string code)
    {
        if (Path.IsPathRooted(relativePath))
        {
            throw Failure(code, $"Absolute paths are prohibited in production model metadata: {relativePath}");
        }

        var fullRoot = Path.GetFullPath(root);
        var fullPath = Path.GetFullPath(Path.Combine(fullRoot, relativePath));
        if (!IsUnder(fullRoot, fullPath))
        {
            throw Failure(code, $"Production model path escapes its approved root: {relativePath}");
        }

        return fullPath;
    }

    private static bool IsUnder(string root, string path)
    {
        var relative = Path.GetRelativePath(Path.GetFullPath(root), Path.GetFullPath(path));
        return !Path.IsPathRooted(relative) && relative != ".." &&
            !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal);
    }

    private static async Task VerifyHashAsync(
        string path,
        string expected,
        string code,
        CancellationToken cancellationToken)
    {
        await using var stream = OpenRead(path);
        var actual = Convert.ToHexString(await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false));
        if (!string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
        {
            throw Failure(code, $"SHA-256 mismatch for '{path}'. Expected {expected}, found {actual}.");
        }
    }

    private static FileStream OpenRead(string path) => new(
        path,
        FileMode.Open,
        FileAccess.Read,
        FileShare.Read,
        64 * 1024,
        FileOptions.Asynchronous | FileOptions.SequentialScan);

    private static void RejectReparsePoint(string path, string code)
    {
        if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
        {
            throw Failure(code, $"Reparse points are prohibited in the production model store: {path}");
        }
    }

    private static void ValidatePathComponent(string value, string parameterName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value, parameterName);
        if (value is "." or ".." || value.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 ||
            value.Contains(Path.DirectorySeparatorChar) || value.Contains(Path.AltDirectorySeparatorChar))
        {
            throw new ArgumentException("Model identifiers and versions must be single safe path components.", parameterName);
        }
    }

    private static void RequireObjectWithExactProperties(JsonElement element, string name, params string[] expected)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw Failure("MODEL_MANIFEST_INVALID", $"{name} must be an object.");
        }

        var actual = element.EnumerateObject().Select(property => property.Name).ToHashSet(StringComparer.Ordinal);
        if (!actual.SetEquals(expected))
        {
            throw Failure("MODEL_MANIFEST_INVALID", $"{name} must contain exactly: {string.Join(", ", expected)}.");
        }
    }

    private static void RequireExactString(JsonElement parent, string name, string expected)
    {
        var actual = RequireNonEmptyString(parent, name);
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw Failure("MODEL_MANIFEST_IDENTITY_MISMATCH", $"Manifest {name} '{actual}' does not match requested '{expected}'.");
        }
    }

    private static string RequireNonEmptyString(JsonElement parent, string name)
    {
        if (!parent.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.String ||
            string.IsNullOrWhiteSpace(value.GetString()))
        {
            throw Failure("MODEL_MANIFEST_INVALID", $"{name} must be a non-empty string.");
        }

        return value.GetString()!;
    }

    private static string RequireSha256(JsonElement parent, string name)
    {
        var value = RequireNonEmptyString(parent, name);
        if (!IsSha256(value))
        {
            throw Failure("MODEL_MANIFEST_INVALID", $"{name} must contain exactly 64 hexadecimal characters.");
        }

        return value;
    }

    private static bool IsSha256(string? value) =>
        value is { Length: 64 } && value.All(Uri.IsHexDigit);

    private static void RequireNonEmptyArray(JsonElement parent, string name)
    {
        JsonElement value;
        if (parent.ValueKind == JsonValueKind.Array)
        {
            value = parent;
        }
        else if (!parent.TryGetProperty(name, out value))
        {
            throw Failure("MODEL_MANIFEST_INVALID", $"{name} is required.");
        }

        if (value.ValueKind != JsonValueKind.Array || value.GetArrayLength() == 0)
        {
            throw Failure("MODEL_MANIFEST_INVALID", $"{name} must be a non-empty array.");
        }
    }

    private static void RequireBoolean(JsonElement parent, string name)
    {
        if (!parent.TryGetProperty(name, out var value) ||
            value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        {
            throw Failure("MODEL_MANIFEST_INVALID", $"{name} must be a boolean.");
        }
    }

    private static ProductionModelValidationException Failure(string code, string message, Exception? inner = null) =>
        new(code, message, inner);

    private sealed record ApprovalEvidence(string Profile, string Path, string Sha256);
}

public sealed record ResolvedProductionModel(
    ModelIdentity Identity,
    string Task,
    IReadOnlyList<InferenceProvider> AvailableProviders,
    string ManifestPath,
    string NoticePath,
    string BenchmarkEvidencePath);

public sealed class ProductionModelValidationException : Exception
{
    public ProductionModelValidationException(string code, string message, Exception? innerException = null)
        : base(message, innerException)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(code);
        Code = code;
    }

    public string Code { get; }
}
