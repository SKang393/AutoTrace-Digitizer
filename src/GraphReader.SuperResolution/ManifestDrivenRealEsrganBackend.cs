// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace GraphReader.SuperResolution;

public enum RealEsrganBackendPurpose
{
    Distribution,
    LocalEvaluation
}

public enum RealEsrganBackendAvailability
{
    Available,
    AvailableForLocalEvaluationOnly,
    ManifestMissing,
    ManifestInvalid,
    NoticeMissing,
    RuntimeMissing,
    RuntimeChecksumMismatch,
    ModelMissing,
    ModelChecksumMismatch,
    ModelIncompatible,
    RedistributionBlocked
}

public sealed record RealEsrganBackendOptions(
    string ManifestPath,
    string ExecutablePath,
    string ModelsDirectory,
    string CacheDirectory,
    string RepositoryRoot,
    RealEsrganBackendPurpose Purpose = RealEsrganBackendPurpose.Distribution,
    TimeSpan? DefaultTimeout = null,
    int MaxDiagnosticCharacters = 32_768);

public sealed record RealEsrganAvailabilityDiagnostic(
    [property: JsonPropertyName("code")] string Code,
    [property: JsonPropertyName("severity")] string Severity,
    [property: JsonPropertyName("user_message_key")] string UserMessageKey,
    [property: JsonPropertyName("technical_message")] string TechnicalMessage,
    [property: JsonPropertyName("recoverable")] bool Recoverable,
    [property: JsonPropertyName("suggested_action")] string SuggestedAction);

public sealed record RealEsrganBackendResolution(
    RealEsrganBackendAvailability Availability,
    EnhancementModel? Model,
    RealEsrganConfiguration? Configuration,
    IEnhancementService? Service,
    RealEsrganAvailabilityDiagnostic? Diagnostic,
    bool ReleaseEligible)
{
    public bool IsAvailable =>
        Availability is RealEsrganBackendAvailability.Available or
            RealEsrganBackendAvailability.AvailableForLocalEvaluationOnly;
}

public static class ManifestDrivenRealEsrganBackend
{
    public const string DefaultModelId = "realesr-animevideov3";
    public const string DefaultManifestFileName = "realesr-animevideov3-ncnn-x2.json";

    public static Task<RealEsrganBackendResolution> ResolveFromRuntimeRootAsync(
        string manifestPath,
        string runtimeRoot,
        string cacheRoot,
        RealEsrganBackendPurpose purpose,
        CancellationToken cancellationToken,
        IProcessRunner? processRunner = null,
        IOutputImageInspector? imageInspector = null)
    {
        string repositoryRoot;
        string fullRuntimeRoot;
        try
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);
            ArgumentException.ThrowIfNullOrWhiteSpace(runtimeRoot);
            ArgumentException.ThrowIfNullOrWhiteSpace(cacheRoot);
            repositoryRoot = FindRepositoryRoot(manifestPath);
            fullRuntimeRoot = Path.GetFullPath(runtimeRoot);
        }
        catch (Exception exception) when (exception is ArgumentException or NotSupportedException)
        {
            return Task.FromResult(Unavailable(
                RealEsrganBackendAvailability.ManifestInvalid,
                "ENHANCEMENT_CONFIGURATION_INVALID",
                "Errors.EnhancementConfigurationInvalid",
                exception.Message,
                "select_manual_mode"));
        }

        return ResolveAsync(
            new RealEsrganBackendOptions(
                manifestPath,
                Path.Combine(fullRuntimeRoot, "realesrgan-ncnn-vulkan.exe"),
                Path.Combine(fullRuntimeRoot, "models"),
                cacheRoot,
                repositoryRoot,
                purpose),
            cancellationToken,
            processRunner,
            imageInspector);
    }

    public static async Task<RealEsrganBackendResolution> ResolveAsync(
        RealEsrganBackendOptions options,
        CancellationToken cancellationToken,
        IProcessRunner? processRunner = null,
        IOutputImageInspector? imageInspector = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        string manifestPath;
        string executablePath;
        string modelsDirectory;
        string cacheDirectory;
        string repositoryRoot;
        try
        {
            manifestPath = Path.GetFullPath(options.ManifestPath);
            executablePath = Path.GetFullPath(options.ExecutablePath);
            modelsDirectory = Path.GetFullPath(options.ModelsDirectory);
            cacheDirectory = Path.GetFullPath(options.CacheDirectory);
            repositoryRoot = Path.GetFullPath(options.RepositoryRoot);
        }
        catch (Exception exception) when (exception is ArgumentException or NotSupportedException)
        {
            return Unavailable(
                RealEsrganBackendAvailability.ManifestInvalid,
                "ENHANCEMENT_CONFIGURATION_INVALID",
                "Errors.EnhancementConfigurationInvalid",
                exception.Message,
                "select_manual_mode");
        }

        if (!File.Exists(manifestPath))
        {
            return Unavailable(
                RealEsrganBackendAvailability.ManifestMissing,
                "MODEL_MANIFEST_NOT_FOUND",
                "Errors.ModelManifestNotFound",
                $"The Real-ESRGAN manifest was not found at '{manifestPath}'.",
                "install_model");
        }

        ParsedManifest manifest;
        try
        {
            manifest = await ParseManifestAsync(manifestPath, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception exception) when (
            exception is JsonException or InvalidDataException or ArgumentException or IOException or UnauthorizedAccessException)
        {
            return Unavailable(
                RealEsrganBackendAvailability.ManifestInvalid,
                "MODEL_MANIFEST_INVALID",
                "Errors.ModelManifestInvalid",
                $"The Real-ESRGAN manifest is invalid: {exception.Message}",
                "install_model");
        }

        string noticePath;
        try
        {
            noticePath = ResolveUnderRoot(repositoryRoot, manifest.NoticePath);
        }
        catch (ArgumentException exception)
        {
            return Unavailable(
                RealEsrganBackendAvailability.ManifestInvalid,
                "MODEL_MANIFEST_INVALID",
                "Errors.ModelManifestInvalid",
                exception.Message,
                "install_model");
        }

        if (!File.Exists(noticePath))
        {
            return Unavailable(
                RealEsrganBackendAvailability.NoticeMissing,
                "MODEL_NOTICE_NOT_FOUND",
                "Errors.ModelNoticeNotFound",
                $"The required model notice was not found at '{noticePath}'.",
                "install_model");
        }

        if (!manifest.LocalAdapterApproved)
        {
            return Unavailable(
                RealEsrganBackendAvailability.ModelIncompatible,
                "MODEL_RUNTIME_INCOMPATIBLE",
                "Errors.EnhancementModelIncompatible",
                manifest.LocalAdapterBlocker,
                "select_manual_mode");
        }

        if (!File.Exists(executablePath))
        {
            return Unavailable(
                RealEsrganBackendAvailability.RuntimeMissing,
                "RUNTIME_NOT_FOUND",
                "Errors.EnhancementRuntimeUnavailable",
                $"The configured Real-ESRGAN runtime was not found at '{executablePath}'.",
                "select_manual_mode");
        }

        FileHashProbe runtimeHash = await ProbeHashAsync(executablePath, cancellationToken).ConfigureAwait(false);
        if (!runtimeHash.Success)
        {
            return Unavailable(
                RealEsrganBackendAvailability.RuntimeMissing,
                "RUNTIME_NOT_FOUND",
                "Errors.EnhancementRuntimeUnavailable",
                $"The configured Real-ESRGAN runtime could not be read: {runtimeHash.Error}",
                "select_manual_mode");
        }

        string runtimeSha256 = runtimeHash.Sha256!;
        if (!string.Equals(runtimeSha256, manifest.RuntimeSha256, StringComparison.Ordinal))
        {
            return Unavailable(
                RealEsrganBackendAvailability.RuntimeChecksumMismatch,
                "RUNTIME_CHECKSUM_MISMATCH",
                "Errors.EnhancementRuntimeChecksumMismatch",
                $"The configured Real-ESRGAN runtime checksum '{runtimeSha256}' does not match the manifest checksum '{manifest.RuntimeSha256}'.",
                "install_model");
        }

        string runtimeDirectory = Path.GetDirectoryName(executablePath)!;
        foreach (ModelArtifact artifact in manifest.RuntimeArtifacts)
        {
            string artifactPath = EnhancementPaths.ResolveArtifact(runtimeDirectory, artifact.RelativePath);
            if (!File.Exists(artifactPath))
            {
                return Unavailable(
                    RealEsrganBackendAvailability.RuntimeMissing,
                    "RUNTIME_NOT_FOUND",
                    "Errors.EnhancementRuntimeUnavailable",
                    $"The required Real-ESRGAN runtime artifact '{artifact.RelativePath}' was not found in '{runtimeDirectory}'.",
                    "select_manual_mode");
            }

            FileHashProbe artifactHash = await ProbeHashAsync(artifactPath, cancellationToken).ConfigureAwait(false);
            if (!artifactHash.Success)
            {
                return Unavailable(
                    RealEsrganBackendAvailability.RuntimeMissing,
                    "RUNTIME_NOT_FOUND",
                    "Errors.EnhancementRuntimeUnavailable",
                    $"The Real-ESRGAN runtime artifact '{artifact.RelativePath}' could not be read: {artifactHash.Error}",
                    "select_manual_mode");
            }

            string actualSha256 = artifactHash.Sha256!;
            if (!string.Equals(actualSha256, artifact.Sha256, StringComparison.Ordinal))
            {
                return Unavailable(
                    RealEsrganBackendAvailability.RuntimeChecksumMismatch,
                    "RUNTIME_CHECKSUM_MISMATCH",
                    "Errors.EnhancementRuntimeChecksumMismatch",
                    $"The Real-ESRGAN runtime artifact '{artifact.RelativePath}' checksum does not match its manifest.",
                    "install_model");
            }
        }

        foreach (ModelArtifact artifact in manifest.Artifacts)
        {
            string artifactPath = EnhancementPaths.ResolveArtifact(modelsDirectory, artifact.RelativePath);
            if (!File.Exists(artifactPath))
            {
                return Unavailable(
                    RealEsrganBackendAvailability.ModelMissing,
                    "MODEL_NOT_FOUND",
                    "Errors.ModelNotFound",
                    $"The required Real-ESRGAN model artifact '{artifact.RelativePath}' was not found in '{modelsDirectory}'.",
                    "install_model");
            }

            FileHashProbe artifactHash = await ProbeHashAsync(artifactPath, cancellationToken).ConfigureAwait(false);
            if (!artifactHash.Success)
            {
                return Unavailable(
                    RealEsrganBackendAvailability.ModelMissing,
                    "MODEL_NOT_FOUND",
                    "Errors.ModelNotFound",
                    $"The Real-ESRGAN model artifact '{artifact.RelativePath}' could not be read: {artifactHash.Error}",
                    "install_model");
            }

            string actualSha256 = artifactHash.Sha256!;
            if (!string.Equals(actualSha256, artifact.Sha256, StringComparison.Ordinal))
            {
                return Unavailable(
                    RealEsrganBackendAvailability.ModelChecksumMismatch,
                    "MODEL_CHECKSUM_MISMATCH",
                    "Errors.ModelChecksumMismatch",
                    $"The Real-ESRGAN model artifact '{artifact.RelativePath}' checksum does not match its manifest.",
                    "install_model");
            }
        }

        var model = new EnhancementModel(
            manifest.ModelId,
            manifest.ModelVersion,
            manifest.PackageSha256,
            manifest.Source,
            manifest.Revision,
            manifest.LicenseSpdx,
            manifest.NoticePath,
            manifest.Artifacts,
            EnhancementProvider.Vulkan,
            manifest.RuntimeModelName);
        var configuration = new RealEsrganConfiguration(
            executablePath,
            modelsDirectory,
            cacheDirectory,
            manifest.RuntimeSha256,
            options.DefaultTimeout,
            options.MaxDiagnosticCharacters);

        bool releaseEligible = manifest.ModelRedistributionApproved &&
                               manifest.RuntimeRedistributionApproved &&
                               manifest.ProductionApproved;
        if (!releaseEligible && options.Purpose == RealEsrganBackendPurpose.Distribution)
        {
            return new RealEsrganBackendResolution(
                RealEsrganBackendAvailability.RedistributionBlocked,
                model,
                configuration,
                null,
                new RealEsrganAvailabilityDiagnostic(
                    "ENHANCEMENT_REDISTRIBUTION_BLOCKED",
                    "error",
                    "Errors.EnhancementRedistributionBlocked",
                    manifest.ReleaseBlocker,
                    Recoverable: true,
                    "select_manual_mode"),
                ReleaseEligible: false);
        }

        IEnhancementService service = new RealEsrganAdapter(configuration, processRunner, imageInspector);
        if (!releaseEligible)
        {
            return new RealEsrganBackendResolution(
                RealEsrganBackendAvailability.AvailableForLocalEvaluationOnly,
                model,
                configuration,
                service,
                new RealEsrganAvailabilityDiagnostic(
                    "ENHANCEMENT_LOCAL_EVALUATION_ONLY",
                    "warning",
                    "Warnings.EnhancementLocalEvaluationOnly",
                    manifest.ReleaseBlocker,
                    Recoverable: true,
                    "select_manual_mode"),
                ReleaseEligible: false);
        }

        return new RealEsrganBackendResolution(
            RealEsrganBackendAvailability.Available,
            model,
            configuration,
            service,
            null,
            ReleaseEligible: true);
    }

    private static async Task<ParsedManifest> ParseManifestAsync(
        string manifestPath,
        CancellationToken cancellationToken)
    {
        await using FileStream stream = File.Open(
            manifestPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read);
        using JsonDocument document = await JsonDocument.ParseAsync(
            stream,
            cancellationToken: cancellationToken).ConfigureAwait(false);
        JsonElement root = document.RootElement;
        if (RequiredInt32(root, "manifest_version") != 1 ||
            !string.Equals(RequiredString(root, "task"), "super_resolution", StringComparison.Ordinal))
        {
            throw new InvalidDataException("Only version 1 super-resolution manifests are supported.");
        }

        string modelId = RequiredString(root, "model_id");
        string modelVersion = RequiredString(root, "model_version");
        string packageSha256 = EnhancementHashing.NormalizeSha256(RequiredString(root, "sha256"));
        JsonElement source = RequiredObject(root, "source");
        string sourceUrl = RequiredString(source, "url");
        string revision = RequiredString(source, "revision");
        JsonElement license = RequiredObject(root, "license");
        string licenseSpdx = RequiredString(license, "spdx");
        string noticePath = RequiredString(license, "notice_path");
        if (!RequiredBoolean(license, "reviewed"))
        {
            throw new InvalidDataException("The model license review is incomplete.");
        }

        if (!RequiredBoolean(root, "commercial_use"))
        {
            throw new InvalidDataException("The model is not approved for commercial use.");
        }

        bool modelRedistributionApproved = RequiredBoolean(root, "redistribution");
        string[] providers = RequiredStringArray(root, "providers");
        if (!providers.Contains("vulkan", StringComparer.Ordinal) || providers.Length != 1)
        {
            throw new InvalidDataException("The NCNN adapter requires an exact Vulkan provider declaration.");
        }

        JsonElement preprocessing = RequiredObject(root, "preprocessing");
        if (!string.Equals(
                RequiredString(preprocessing, "runtime"),
                "realesrgan-ncnn-vulkan",
                StringComparison.Ordinal) ||
            RequiredInt32(preprocessing, "runtime_scale_argument") != EnhancementDefaults.Scale)
        {
            throw new InvalidDataException("The manifest must declare the Real-ESRGAN NCNN runtime at output scale 2.");
        }

        string runtimeModelName = RequiredString(preprocessing, "runtime_model_name");
        bool localAdapterApproved = RequiredBoolean(preprocessing, "local_adapter_approval");
        string localAdapterBlocker = localAdapterApproved
            ? string.Empty
            : RequiredString(preprocessing, "local_adapter_blocker");
        string runtimeSha256 = EnhancementHashing.NormalizeSha256(
            RequiredString(preprocessing, "runtime_executable_sha256"));
        JsonElement runtimeHashes = RequiredObject(preprocessing, "runtime_files_sha256");
        var runtimeArtifacts = new List<ModelArtifact>();
        foreach (JsonProperty property in runtimeHashes.EnumerateObject())
        {
            if (!string.Equals(Path.GetFileName(property.Name), property.Name, StringComparison.Ordinal) ||
                property.Value.ValueKind != JsonValueKind.String)
            {
                throw new InvalidDataException("Runtime artifact entries must be file names with SHA-256 string values.");
            }

            runtimeArtifacts.Add(new ModelArtifact(
                property.Name,
                EnhancementHashing.NormalizeSha256(property.Value.GetString()!)));
        }

        ModelArtifact[] executableArtifacts = runtimeArtifacts.Where(static artifact =>
                string.Equals(
                    artifact.RelativePath,
                    "realesrgan-ncnn-vulkan.exe",
                    StringComparison.OrdinalIgnoreCase))
            .ToArray();
        if (executableArtifacts.Length != 1 ||
            !string.Equals(executableArtifacts[0].Sha256, runtimeSha256, StringComparison.Ordinal))
        {
            throw new InvalidDataException("The runtime checksum inventory must contain the configured executable checksum.");
        }

        JsonElement runtimeRedistribution = RequiredObject(preprocessing, "runtime_redistribution");
        bool runtimeRedistributionApproved = RequiredBoolean(runtimeRedistribution, "approved");
        string runtimeBlocker = RequiredString(runtimeRedistribution, "blocker");
        JsonElement artifactHashes = RequiredObject(preprocessing, "model_payload_sha256");
        string[] files = RequiredStringArray(root, "files");
        if (files.Length != 2)
        {
            throw new InvalidDataException("The NCNN model manifest must declare one parameter file and one weight file.");
        }

        var artifacts = new List<ModelArtifact>(files.Length);
        foreach (string file in files)
        {
            string normalized = file.Replace('\\', '/');
            if (!normalized.StartsWith("models/", StringComparison.Ordinal) ||
                !string.Equals(Path.GetFileName(normalized), normalized["models/".Length..], StringComparison.Ordinal))
            {
                throw new InvalidDataException($"Model artifact path '{file}' is outside the manifest model root.");
            }

            if (!artifactHashes.TryGetProperty(file, out JsonElement hashElement) ||
                hashElement.ValueKind != JsonValueKind.String)
            {
                throw new InvalidDataException($"Model artifact '{file}' has no checksum entry.");
            }

            artifacts.Add(new ModelArtifact(
                normalized["models/".Length..],
                EnhancementHashing.NormalizeSha256(hashElement.GetString()!)));
        }

        JsonElement outputs = RequiredArray(root, "outputs");
        if (outputs.GetArrayLength() == 0)
        {
            throw new InvalidDataException("The NCNN model manifest must declare an output contract.");
        }

        JsonElement output = outputs[0];
        if (RequiredInt32(output, "configured_output_scale") != EnhancementDefaults.Scale ||
            !string.Equals(RequiredString(output, "coordinate_space"), "enhanced_pixels", StringComparison.Ordinal))
        {
            throw new InvalidDataException("The manifest output must be exact scale-2 enhanced pixels.");
        }

        JsonElement benchmarks = RequiredArray(root, "benchmarks");
        if (benchmarks.GetArrayLength() == 0)
        {
            throw new InvalidDataException("A direct benchmark status is required.");
        }

        bool productionApproved = RequiredBoolean(benchmarks[0], "production_approval");
        string releaseBlocker = runtimeBlocker;
        if (!productionApproved)
        {
            releaseBlocker = $"{releaseBlocker} Production benchmark approval is false.";
        }

        if (!modelRedistributionApproved)
        {
            releaseBlocker = $"{releaseBlocker} Model redistribution approval is false.";
        }

        return new ParsedManifest(
            modelId,
            modelVersion,
            packageSha256,
            sourceUrl,
            revision,
            licenseSpdx,
            noticePath,
            runtimeModelName,
            localAdapterApproved,
            localAdapterBlocker,
            runtimeSha256,
            new ReadOnlyCollection<ModelArtifact>(runtimeArtifacts),
            new ReadOnlyCollection<ModelArtifact>(artifacts),
            modelRedistributionApproved,
            runtimeRedistributionApproved,
            productionApproved,
            releaseBlocker);
    }

    private static RealEsrganBackendResolution Unavailable(
        RealEsrganBackendAvailability availability,
        string code,
        string userMessageKey,
        string technicalMessage,
        string suggestedAction) =>
        new(
            availability,
            null,
            null,
            null,
            new RealEsrganAvailabilityDiagnostic(
                code,
                "warning",
                userMessageKey,
                technicalMessage,
                Recoverable: true,
                suggestedAction),
            ReleaseEligible: false);

    private static string ResolveUnderRoot(string root, string relativePath)
    {
        if (Path.IsPathRooted(relativePath))
        {
            throw new ArgumentException("Notice paths must be repository-relative.", nameof(relativePath));
        }

        string candidate = Path.GetFullPath(Path.Combine(root, relativePath));
        string relative = Path.GetRelativePath(root, candidate);
        if (relative.Equals("..", StringComparison.Ordinal) ||
            relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal) ||
            Path.IsPathRooted(relative))
        {
            throw new ArgumentException("Notice paths may not escape the repository root.", nameof(relativePath));
        }

        return candidate;
    }

    private static async Task<FileHashProbe> ProbeHashAsync(
        string path,
        CancellationToken cancellationToken)
    {
        try
        {
            string sha256 = await EnhancementHashing.ComputeFileSha256Async(
                path,
                cancellationToken).ConfigureAwait(false);
            return new FileHashProbe(true, sha256, null);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return new FileHashProbe(false, null, exception.Message);
        }
    }

    private static string FindRepositoryRoot(string manifestPath)
    {
        DirectoryInfo? directory = new FileInfo(Path.GetFullPath(manifestPath)).Directory;
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "LICENSES")) &&
                Directory.Exists(Path.Combine(directory.FullName, "models")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new ArgumentException(
            "The manifest path is not inside a Graph Auto Reader distribution root containing LICENSES and models.",
            nameof(manifestPath));
    }

    private static JsonElement RequiredObject(JsonElement parent, string name)
    {
        JsonElement value = RequiredProperty(parent, name);
        return value.ValueKind == JsonValueKind.Object
            ? value
            : throw new InvalidDataException($"Manifest property '{name}' must be an object.");
    }

    private static JsonElement RequiredArray(JsonElement parent, string name)
    {
        JsonElement value = RequiredProperty(parent, name);
        return value.ValueKind == JsonValueKind.Array
            ? value
            : throw new InvalidDataException($"Manifest property '{name}' must be an array.");
    }

    private static string RequiredString(JsonElement parent, string name)
    {
        JsonElement value = RequiredProperty(parent, name);
        string? text = value.ValueKind == JsonValueKind.String ? value.GetString() : null;
        return !string.IsNullOrWhiteSpace(text)
            ? text
            : throw new InvalidDataException($"Manifest property '{name}' must be a nonempty string.");
    }

    private static int RequiredInt32(JsonElement parent, string name)
    {
        JsonElement value = RequiredProperty(parent, name);
        return value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int result)
            ? result
            : throw new InvalidDataException($"Manifest property '{name}' must be an integer.");
    }

    private static bool RequiredBoolean(JsonElement parent, string name)
    {
        JsonElement value = RequiredProperty(parent, name);
        return value.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? value.GetBoolean()
            : throw new InvalidDataException($"Manifest property '{name}' must be a boolean.");
    }

    private static string[] RequiredStringArray(JsonElement parent, string name)
    {
        JsonElement value = RequiredArray(parent, name);
        string[] values = value.EnumerateArray()
            .Select(static item => item.ValueKind == JsonValueKind.String ? item.GetString() : null)
            .Where(static item => !string.IsNullOrWhiteSpace(item))
            .Select(static item => item!)
            .ToArray();
        if (values.Length != value.GetArrayLength() || values.Length == 0)
        {
            throw new InvalidDataException($"Manifest property '{name}' must contain only nonempty strings.");
        }

        return values;
    }

    private static JsonElement RequiredProperty(JsonElement parent, string name) =>
        parent.TryGetProperty(name, out JsonElement value)
            ? value
            : throw new InvalidDataException($"Manifest property '{name}' is required.");

    private sealed record ParsedManifest(
        string ModelId,
        string ModelVersion,
        string PackageSha256,
        string Source,
        string Revision,
        string LicenseSpdx,
        string NoticePath,
        string RuntimeModelName,
        bool LocalAdapterApproved,
        string LocalAdapterBlocker,
        string RuntimeSha256,
        IReadOnlyList<ModelArtifact> RuntimeArtifacts,
        IReadOnlyList<ModelArtifact> Artifacts,
        bool ModelRedistributionApproved,
        bool RuntimeRedistributionApproved,
        bool ProductionApproved,
        string ReleaseBlocker);

    private sealed record FileHashProbe(bool Success, string? Sha256, string? Error);
}
