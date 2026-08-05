// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Security.Cryptography;
using System.Security;
using System.Text.Json;

namespace GraphReader.Inference;

/// <summary>
/// Resolves checksum-bound, benchmark-approved ONNX models from an offline store.
/// The store root is the packaged <c>models</c> directory. A checksum-bound
/// <c>production-model-index.json</c> maps manifests, runtime payloads, reviewed
/// notices, and benchmark evidence to distribution-relative paths.
/// </summary>
public sealed class ProductionModelStore
{
    private const string PackageIndexFileName = "production-model-index.json";
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

    private static readonly HashSet<string> ReviewedLicenseAllowlist = new(StringComparer.Ordinal)
    {
        "Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib",
        "BSL-1.0", "Unlicense", "CC0-1.0"
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

        try
        {
            return await ResolveCoreAsync(modelId, version, requiredProvider, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (ProductionModelValidationException)
        {
            throw;
        }
        catch (UnauthorizedAccessException exception)
        {
            throw Failure("MODEL_STORE_ACCESS_DENIED", "Access to the production model store was denied.", exception);
        }
        catch (SecurityException exception)
        {
            throw Failure("MODEL_STORE_ACCESS_DENIED", "Access to the production model store was denied.", exception);
        }
        catch (IOException exception)
        {
            throw Failure("MODEL_STORE_IO_ERROR", "The production model store could not be read safely.", exception);
        }
    }

    public async ValueTask<IReadOnlyList<ResolvedProductionModel>> ResolveAllAsync(
        InferenceProvider? requiredProvider,
        CancellationToken cancellationToken)
    {
        if (requiredProvider is InferenceProvider.Fake)
        {
            throw new ArgumentOutOfRangeException(
                nameof(requiredProvider),
                "Fake is not a production execution provider.");
        }

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            RejectReparseAncestors(_root, "MODEL_STORE_REPARSE_POINT");
            string packageIndexPath = ResolveUnderRoot(PackageIndexFileName);
            RejectReparseAncestors(packageIndexPath, "MODEL_STORE_REPARSE_POINT");
            if (!File.Exists(packageIndexPath))
            {
                throw Failure(
                    "MODEL_PACKAGE_INDEX_NOT_FOUND",
                    $"Production model package index was not found: {packageIndexPath}");
            }

            PackageIndex packageIndex = await ReadPackageIndexAsync(
                    packageIndexPath,
                    cancellationToken)
                .ConfigureAwait(false);
            ValidateCanonicalPackagePaths(packageIndex);
            ValidatePackageTree(packageIndexPath, packageIndex);

            var resolved = new List<ResolvedProductionModel>(packageIndex.Models.Count);
            foreach (PackageModel model in packageIndex.Models.OrderBy(
                         static item => item.ModelId,
                         StringComparer.Ordinal))
            {
                cancellationToken.ThrowIfCancellationRequested();
                resolved.Add(await ResolveCoreAsync(
                        model.ModelId,
                        model.Version,
                        requiredProvider,
                        cancellationToken)
                    .ConfigureAwait(false));
            }

            return new ReadOnlyCollection<ResolvedProductionModel>(resolved);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (ProductionModelValidationException)
        {
            throw;
        }
        catch (UnauthorizedAccessException exception)
        {
            throw Failure("MODEL_STORE_ACCESS_DENIED", "Access to the production model store was denied.", exception);
        }
        catch (SecurityException exception)
        {
            throw Failure("MODEL_STORE_ACCESS_DENIED", "Access to the production model store was denied.", exception);
        }
        catch (IOException exception)
        {
            throw Failure("MODEL_STORE_IO_ERROR", "The production model store could not be read safely.", exception);
        }
    }

    private async ValueTask<ResolvedProductionModel> ResolveCoreAsync(
        string modelId,
        string version,
        InferenceProvider? requiredProvider,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        RejectReparseAncestors(_root, "MODEL_STORE_REPARSE_POINT");
        var packageIndexPath = ResolveUnderRoot(PackageIndexFileName);
        RejectReparseAncestors(packageIndexPath, "MODEL_STORE_REPARSE_POINT");
        if (!File.Exists(packageIndexPath))
        {
            throw Failure("MODEL_PACKAGE_INDEX_NOT_FOUND", $"Production model package index was not found: {packageIndexPath}");
        }

        var packageIndex = await ReadPackageIndexAsync(packageIndexPath, cancellationToken).ConfigureAwait(false);
        ValidateCanonicalPackagePaths(packageIndex);
        var packageModel = packageIndex.Models.SingleOrDefault(model =>
            string.Equals(model.ModelId, modelId, StringComparison.Ordinal) &&
            string.Equals(model.Version, version, StringComparison.Ordinal));
        if (packageModel is null)
        {
            throw Failure("MODEL_MANIFEST_NOT_FOUND", $"The package index does not contain model '{modelId}' version '{version}'.");
        }

        ValidatePackageTree(packageIndexPath, packageIndex);
        var manifestPath = ResolveUnderRoot(packageModel.Manifest.Path);
        if (!File.Exists(manifestPath))
        {
            throw Failure("MODEL_MANIFEST_NOT_FOUND", $"Production model manifest was not found: {manifestPath}");
        }

        await VerifyHashAsync(
            manifestPath,
            packageModel.Manifest.Sha256,
            "MODEL_PACKAGE_MANIFEST_CHECKSUM_MISMATCH",
            cancellationToken).ConfigureAwait(false);

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
            RejectDuplicatePropertyNames(root, "MODEL_MANIFEST_INVALID", "Production model manifest");
            ValidateManifestShape(root, modelId, version);
            ValidateSource(root.GetProperty("source"));
            var noticePath = ValidateLicense(root.GetProperty("license"));

            if (!root.GetProperty("commercial_use").GetBoolean() || !root.GetProperty("redistribution").GetBoolean())
            {
                throw Failure("MODEL_REDISTRIBUTION_NOT_APPROVED", "Production models require commercial-use and redistribution approval.");
            }

            var providers = ValidateProviders(root.GetProperty("providers"), requiredProvider);
            var approvalEvidence = ValidateApproval(root.GetProperty("benchmarks"));
            var expectedHashes = ReadDeclaredPayloadHashes(root);
            if (expectedHashes.Keys.Count(path => string.Equals(Path.GetExtension(path), ".onnx", StringComparison.OrdinalIgnoreCase)) != 1)
            {
                throw Failure("MODEL_PAYLOAD_INVALID", "A production ONNX manifest must identify exactly one .onnx payload.");
            }

            var indexedPayloads = packageModel.Payloads.ToDictionary(
                payload => payload.DeclaredPath,
                StringComparer.OrdinalIgnoreCase);
            if (!expectedHashes.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase)
                    .SetEquals(indexedPayloads.Keys))
            {
                throw Failure("MODEL_PACKAGE_INDEX_INVALID", "Package payload mappings must exactly match manifest files.");
            }

            var resolvedPayloads = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var expected in expectedHashes)
            {
                var packaged = indexedPayloads[expected.Key];
                if (!string.Equals(packaged.Sha256, expected.Value, StringComparison.OrdinalIgnoreCase))
                {
                    throw Failure("MODEL_PACKAGE_INDEX_INVALID", $"Package checksum for '{expected.Key}' differs from its manifest checksum.");
                }

                var payloadPath = ResolveUnderRoot(packaged.Path);
                if (!File.Exists(payloadPath))
                {
                    throw Failure("MODEL_PAYLOAD_MISSING", $"Packaged model payload does not exist: {payloadPath}");
                }

                await VerifyHashAsync(payloadPath, expected.Value, "MODEL_PAYLOAD_CHECKSUM_MISMATCH", cancellationToken)
                    .ConfigureAwait(false);
                resolvedPayloads.Add(expected.Key, payloadPath);
            }

            if (!string.Equals(packageModel.Notice.DeclaredPath, noticePath, StringComparison.Ordinal))
            {
                throw Failure("MODEL_PACKAGE_INDEX_INVALID", "Package notice mapping does not match the manifest notice path.");
            }

            var resolvedNotice = ResolveUnderRoot(packageModel.Notice.Path);
            if (!File.Exists(resolvedNotice) || new FileInfo(resolvedNotice).Length == 0)
            {
                throw Failure("MODEL_NOTICE_MISSING", $"Reviewed model notice is missing or empty: {resolvedNotice}");
            }

            await VerifyHashAsync(
                resolvedNotice,
                packageModel.Notice.Sha256,
                "MODEL_NOTICE_CHECKSUM_MISMATCH",
                cancellationToken).ConfigureAwait(false);
            if (!string.Equals(packageModel.BenchmarkEvidence.DeclaredPath, approvalEvidence.Path, StringComparison.Ordinal) ||
                !string.Equals(packageModel.BenchmarkEvidence.Sha256, approvalEvidence.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                throw Failure("MODEL_PACKAGE_INDEX_INVALID", "Package benchmark evidence mapping does not match the approved benchmark.");
            }

            var evidencePath = ResolveUnderRoot(packageModel.BenchmarkEvidence.Path);
            if (!File.Exists(evidencePath))
            {
                throw Failure("MODEL_BENCHMARK_EVIDENCE_MISSING", $"Approval evidence does not exist: {evidencePath}");
            }

            await VerifyHashAsync(
                evidencePath,
                approvalEvidence.Sha256,
                "MODEL_BENCHMARK_CHECKSUM_MISMATCH",
                cancellationToken).ConfigureAwait(false);

            var onnxDeclaredPath = expectedHashes.Keys.Single(path =>
                string.Equals(Path.GetExtension(path), ".onnx", StringComparison.OrdinalIgnoreCase));
            var onnxPath = resolvedPayloads[onnxDeclaredPath];
            var identity = new ModelIdentity(
                modelId,
                version,
                expectedHashes[onnxDeclaredPath].ToUpperInvariant(),
                onnxPath);
            identity.Validate();
            return new ResolvedProductionModel(
                identity,
                root.GetProperty("task").GetString()!,
                new ReadOnlyCollection<InferenceProvider>(providers.ToArray()),
                manifestPath,
                packageModel.Manifest.Sha256,
                resolvedNotice,
                packageModel.Notice.Sha256,
                evidencePath,
                packageModel.BenchmarkEvidence.Sha256);
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
        var spdx = RequireNonEmptyString(license, "spdx");
        var notice = RequireNonEmptyString(license, "notice_path");
        RequireBoolean(license, "reviewed");
        if (!license.GetProperty("reviewed").GetBoolean())
        {
            throw Failure("MODEL_LICENSE_NOT_REVIEWED", "The model license and notice must be reviewed before production use.");
        }

        if (IsProhibitedLicense(spdx))
        {
            throw Failure("MODEL_LICENSE_PROHIBITED", $"Model license '{spdx}' is prohibited by the Apache distribution policy.");
        }

        if (!ReviewedLicenseAllowlist.Contains(spdx))
        {
            throw Failure("MODEL_LICENSE_UNRECOGNIZED", $"Model license '{spdx}' is not on the reviewed production allowlist.");
        }

        return notice;
    }

    private static bool IsProhibitedLicense(string spdx)
    {
        var normalized = spdx.ToUpperInvariant();
        return normalized.Contains("GPL", StringComparison.Ordinal) ||
            normalized.Contains("AGPL", StringComparison.Ordinal) ||
            normalized.Contains("LGPL", StringComparison.Ordinal) ||
            normalized.Contains("SSPL", StringComparison.Ordinal) ||
            normalized.Contains("BUSL", StringComparison.Ordinal) ||
            normalized.Contains("-NC", StringComparison.Ordinal) ||
            normalized.Contains("NONCOMMERCIAL", StringComparison.Ordinal) ||
            normalized.Contains("NOASSERTION", StringComparison.Ordinal) ||
            normalized.Contains("UNKNOWN", StringComparison.Ordinal) ||
            normalized.Contains("UNCLEAR", StringComparison.Ordinal) ||
            normalized.Contains("PROPRIETARY", StringComparison.Ordinal) ||
            normalized.StartsWith("LICENSEREF-", StringComparison.Ordinal);
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

    private static Dictionary<string, string> ReadDeclaredPayloadHashes(JsonElement root)
    {
        var files = root.GetProperty("files");
        RequireNonEmptyArray(files, "files");
        var declared = new List<string>();
        foreach (var item in files.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(item.GetString()))
            {
                throw Failure("MODEL_PAYLOAD_INVALID", "Every manifest file must be a non-empty relative path.");
            }

            var path = NormalizeDeclaredPath(item.GetString()!, "MODEL_PAYLOAD_INVALID");
            if (declared.Contains(path, StringComparer.OrdinalIgnoreCase))
            {
                throw Failure("MODEL_PAYLOAD_INVALID", $"Duplicate manifest payload: {path}");
            }

            declared.Add(path);
        }

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
                var path = NormalizeDeclaredPath(item.Name, "MODEL_PAYLOAD_INVALID");
                if (!declared.Contains(path, StringComparer.OrdinalIgnoreCase) || hashes.ContainsKey(path) ||
                    item.Value.ValueKind != JsonValueKind.String || !IsSha256(item.Value.GetString()))
                {
                    throw Failure("MODEL_PAYLOAD_INVALID", $"Invalid or unlisted payload checksum entry: {item.Name}");
                }

                hashes.Add(path, item.Value.GetString()!);
            }
        }

        if (hashes.Count == 0 && declared.Count == 1)
        {
            hashes.Add(declared[0], root.GetProperty("sha256").GetString()!);
        }

        if (hashes.Count != declared.Count)
        {
            throw Failure("MODEL_PAYLOAD_INVALID", "Every payload must have exactly one SHA-256 checksum.");
        }

        return hashes;
    }

    private static async Task<PackageIndex> ReadPackageIndexAsync(string indexPath, CancellationToken cancellationToken)
    {
        try
        {
            return await ReadPackageIndexCoreAsync(indexPath, cancellationToken).ConfigureAwait(false);
        }
        catch (ProductionModelValidationException exception) when (exception.Code == "MODEL_MANIFEST_INVALID")
        {
            throw Failure("MODEL_PACKAGE_INDEX_INVALID", "Production model package index metadata is invalid.", exception);
        }
    }

    private static async Task<PackageIndex> ReadPackageIndexCoreAsync(string indexPath, CancellationToken cancellationToken)
    {
        JsonDocument document;
        try
        {
            await using var stream = OpenRead(indexPath);
            document = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        catch (JsonException exception)
        {
            throw Failure("MODEL_PACKAGE_INDEX_INVALID", $"Production model package index is malformed: {exception.Message}", exception);
        }

        using (document)
        {
            var root = document.RootElement;
            RejectDuplicatePropertyNames(root, "MODEL_PACKAGE_INDEX_INVALID", "Production model package index");
            RequireObjectWithExactProperties(root, "package index", "schema_version", "models");
            if (!root.GetProperty("schema_version").TryGetInt32(out var version) || version != 1)
            {
                throw Failure("MODEL_PACKAGE_INDEX_INVALID", "Only production model package index version 1 is supported.");
            }

            RequireNonEmptyArray(root, "models");
            var models = new List<PackageModel>();
            var identities = new HashSet<string>(StringComparer.Ordinal);
            foreach (var model in root.GetProperty("models").EnumerateArray())
            {
                RequireObjectWithExactProperties(
                    model,
                    "package model",
                    "model_id",
                    "model_version",
                    "manifest",
                    "payloads",
                    "notice",
                    "benchmark_evidence");
                var modelId = RequireNonEmptyString(model, "model_id");
                var modelVersion = RequireNonEmptyString(model, "model_version");
                if (!identities.Add(modelId + "\0" + modelVersion))
                {
                    throw Failure("MODEL_PACKAGE_INDEX_INVALID", $"Duplicate packaged model identity: {modelId} {modelVersion}");
                }

                var manifest = ReadPackageResource(model.GetProperty("manifest"), "manifest", hasDeclaredPath: false);
                RequireNonEmptyArray(model, "payloads");
                var payloads = model.GetProperty("payloads").EnumerateArray()
                    .Select(payload => ReadPackageResource(payload, "payload", hasDeclaredPath: true))
                    .ToArray();
                if (payloads.Select(payload => payload.DeclaredPath).Distinct(StringComparer.OrdinalIgnoreCase).Count() != payloads.Length)
                {
                    throw Failure("MODEL_PACKAGE_INDEX_INVALID", $"Duplicate packaged payload declaration for model: {modelId}");
                }

                models.Add(new PackageModel(
                    modelId,
                    modelVersion,
                    manifest,
                    Array.AsReadOnly(payloads),
                    ReadPackageResource(model.GetProperty("notice"), "notice", hasDeclaredPath: true),
                    ReadPackageResource(model.GetProperty("benchmark_evidence"), "benchmark evidence", hasDeclaredPath: true)));
            }

            return new PackageIndex(models.AsReadOnly());
        }
    }

    private static PackageResource ReadPackageResource(JsonElement element, string name, bool hasDeclaredPath)
    {
        RequireObjectWithExactProperties(
            element,
            name,
            hasDeclaredPath ? ["declared_path", "path", "sha256"] : ["path", "sha256"]);
        var declaredPath = hasDeclaredPath
            ? NormalizeDeclaredPath(RequireNonEmptyString(element, "declared_path"), "MODEL_PACKAGE_INDEX_INVALID")
            : string.Empty;
        var packagePath = NormalizeDeclaredPath(RequireNonEmptyString(element, "path"), "MODEL_PACKAGE_INDEX_INVALID");
        var sha256 = RequireSha256(element, "sha256");
        return new PackageResource(declaredPath, packagePath, sha256);
    }

    private void ValidatePackageTree(string indexPath, PackageIndex index)
    {
        var allowed = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { Path.GetFullPath(indexPath) };
        foreach (var model in index.Models)
        {
            foreach (var resource in model.Resources)
            {
                var path = ResolveUnderRoot(resource.Path);
                RejectReparseAncestors(path, "MODEL_STORE_REPARSE_POINT");
                allowed.Add(path);
            }
        }

        var pending = new Queue<DirectoryInfo>();
        pending.Enqueue(new DirectoryInfo(_root));
        while (pending.TryDequeue(out var directory))
        {
            RejectReparseAncestors(directory.FullName, "MODEL_STORE_REPARSE_POINT");
            foreach (var entry in directory.EnumerateFileSystemInfos())
            {
                if ((entry.Attributes & FileAttributes.ReparsePoint) != 0)
                {
                    throw Failure("MODEL_STORE_REPARSE_POINT", $"Reparse points are prohibited in the production model store: {entry.FullName}");
                }

                if (entry is DirectoryInfo child)
                {
                    pending.Enqueue(child);
                }
                else if (!allowed.Contains(Path.GetFullPath(entry.FullName)))
                {
                    throw Failure("MODEL_STORE_EXTRA_FILE", $"Unlisted file exists in the production model directory: {entry.FullName}");
                }
            }
        }
    }

    private static void ValidateCanonicalPackagePaths(PackageIndex index)
    {
        foreach (var model in index.Models)
        {
            var manifestPath = $"manifest/{model.ModelId}/{model.Version}/manifest.json";
            RequireCanonicalPackagePath(model.Manifest.Path, manifestPath, "manifest");

            foreach (var payload in model.Payloads)
            {
                var payloadPath = $"runtime/{model.ModelId}/{model.Version}/{payload.DeclaredPath}";
                RequireCanonicalPackagePath(payload.Path, payloadPath, "payload");
            }

            var noticeLeaf = model.Notice.DeclaredPath.Split('/')[^1];
            var noticePath = $"notices/{model.ModelId}/{model.Version}/{noticeLeaf}";
            RequireCanonicalPackagePath(model.Notice.Path, noticePath, "notice");

            var evidenceLeaf = model.BenchmarkEvidence.DeclaredPath.Split('/')[^1];
            var evidencePath = $"evidence/{model.ModelId}/{model.Version}/{evidenceLeaf}";
            RequireCanonicalPackagePath(model.BenchmarkEvidence.Path, evidencePath, "benchmark evidence");
        }
    }

    private static void RequireCanonicalPackagePath(string actual, string expected, string resourceKind)
    {
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw Failure(
                "MODEL_PACKAGE_INDEX_INVALID",
                $"Production model {resourceKind} path must be canonical. Expected '{expected}', found '{actual}'.");
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

    private static void RejectReparseAncestors(string path, string code)
    {
        FileSystemInfo? current = Directory.Exists(path)
            ? new DirectoryInfo(Path.GetFullPath(path))
            : new FileInfo(Path.GetFullPath(path));
        while (current is not null)
        {
            if (current.Exists && (current.Attributes & FileAttributes.ReparsePoint) != 0)
            {
                throw Failure(code, $"Reparse points are prohibited in the production model store: {current.FullName}");
            }

            current = current switch
            {
                FileInfo file => file.Directory,
                DirectoryInfo directory => directory.Parent,
                _ => null
            };
        }
    }

    private static string NormalizeDeclaredPath(string path, string code)
    {
        if (string.IsNullOrWhiteSpace(path) || path != path.Trim() || Path.IsPathRooted(path))
        {
            throw Failure(code, $"Model package paths must be non-empty relative paths: {path}");
        }

        var normalized = path.Replace('\\', '/');
        var segments = normalized.Split('/');
        if (segments.Any(segment =>
                string.IsNullOrWhiteSpace(segment) || segment is "." or ".." ||
                segment.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0))
        {
            throw Failure(code, $"Model package path is not a safe relative path: {path}");
        }

        return normalized;
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

        var actual = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject())
        {
            if (!actual.Add(property.Name))
            {
                throw Failure("MODEL_MANIFEST_INVALID", $"{name} contains duplicate JSON property '{property.Name}'.");
            }
        }

        if (!actual.SetEquals(expected))
        {
            throw Failure("MODEL_MANIFEST_INVALID", $"{name} must contain exactly: {string.Join(", ", expected)}.");
        }
    }

    private static void RejectDuplicatePropertyNames(JsonElement element, string code, string description)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in element.EnumerateObject())
            {
                if (!names.Add(property.Name))
                {
                    throw Failure(code, $"{description} contains duplicate JSON property '{property.Name}'.");
                }

                RejectDuplicatePropertyNames(property.Value, code, description);
            }

            return;
        }

        if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in element.EnumerateArray())
            {
                RejectDuplicatePropertyNames(item, code, description);
            }
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

    private sealed record PackageIndex(ReadOnlyCollection<PackageModel> Models);

    private sealed record PackageModel(
        string ModelId,
        string Version,
        PackageResource Manifest,
        ReadOnlyCollection<PackageResource> Payloads,
        PackageResource Notice,
        PackageResource BenchmarkEvidence)
    {
        public IEnumerable<PackageResource> Resources =>
            new[] { Manifest, Notice, BenchmarkEvidence }.Concat(Payloads);
    }

    private sealed record PackageResource(string DeclaredPath, string Path, string Sha256);
}

public sealed class ResolvedProductionModel
{
    internal ResolvedProductionModel(
        ModelIdentity identity,
        string task,
        IReadOnlyList<InferenceProvider> availableProviders,
        string manifestPath,
        string manifestSha256,
        string noticePath,
        string noticeSha256,
        string benchmarkEvidencePath,
        string benchmarkEvidenceSha256)
    {
        Identity = identity;
        Task = task;
        AvailableProviders = availableProviders;
        ManifestPath = manifestPath;
        ManifestSha256 = manifestSha256;
        NoticePath = noticePath;
        NoticeSha256 = noticeSha256;
        BenchmarkEvidencePath = benchmarkEvidencePath;
        BenchmarkEvidenceSha256 = benchmarkEvidenceSha256;
    }

    public ModelIdentity Identity { get; }

    public string Task { get; }

    public IReadOnlyList<InferenceProvider> AvailableProviders { get; }

    public string ManifestPath { get; }

    public string ManifestSha256 { get; }

    public string NoticePath { get; }

    public string NoticeSha256 { get; }

    public string BenchmarkEvidencePath { get; }

    public string BenchmarkEvidenceSha256 { get; }
}

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
