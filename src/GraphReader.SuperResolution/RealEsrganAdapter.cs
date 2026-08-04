// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;

namespace GraphReader.SuperResolution;

public sealed class RealEsrganAdapter : IEnhancementService
{
    public const int ContractVersion = 1;
    public const string StageVersion = "0.1.0";

    private static readonly IReadOnlyList<string> EvidenceWarnings = Array.AsReadOnly(
        new[]
        {
            "Enhanced output is derivative evidence. Map detections to original pixels and review material disagreement with the original."
        });

    private readonly RealEsrganConfiguration _configuration;
    private readonly IProcessRunner _processRunner;
    private readonly IOutputImageInspector _imageInspector;
    private readonly EnhancementCache _cache;
    private static readonly KeyedGate ProcessKeyedGate = new();

    public RealEsrganAdapter(
        RealEsrganConfiguration configuration,
        IProcessRunner? processRunner = null,
        IOutputImageInspector? imageInspector = null)
    {
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        _processRunner = processRunner ?? new LocalProcessRunner();
        _imageInspector = imageInspector ?? new PngOutputImageInspector();
        _cache = new EnhancementCache(configuration.CacheDirectory, _imageInspector);
    }

    public async Task<EnhancementResult> EnhanceAsync(
        EnhancementRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        var total = Stopwatch.StartNew();
        string? workDirectory = null;
        try
        {
            EnhancementOptions options = request.Options ?? new EnhancementOptions();
            EnhancementResult? validationFailure = ValidateRequest(request, options);
            if (validationFailure is not null)
            {
                return validationFailure;
            }

            cancellationToken.ThrowIfCancellationRequested();
            string inputPath = Path.GetFullPath(request.InputPath);
            string outputPath = Path.GetFullPath(request.OutputPath);
            string executablePath = Path.GetFullPath(_configuration.ExecutablePath);
            string modelRoot = Path.GetFullPath(_configuration.ModelsDirectory);

            if (!File.Exists(inputPath))
            {
                return Failure(
                    EnhancementFailureCode.SourceMissing,
                    "The source image is unavailable.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement);
            }

            if (File.Exists(outputPath))
            {
                return Failure(
                    EnhancementFailureCode.OutputAlreadyExists,
                    "The requested derivative output already exists and will not be overwritten.",
                    options);
            }

            if (!File.Exists(executablePath))
            {
                return Failure(
                    EnhancementFailureCode.RuntimeMissing,
                    "The configured Real-ESRGAN runtime is unavailable.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement);
            }

            if (!Directory.Exists(modelRoot))
            {
                return Failure(
                    EnhancementFailureCode.ModelMissing,
                    "The configured Real-ESRGAN model directory is unavailable.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement);
            }

            var preprocess = Stopwatch.StartNew();
            string runtimeSha256 = await EnhancementHashing.ComputeFileSha256Async(
                executablePath,
                cancellationToken).ConfigureAwait(false);
            if (_configuration.ExpectedExecutableSha256 is not null &&
                !string.Equals(
                    runtimeSha256,
                    EnhancementHashing.NormalizeSha256(_configuration.ExpectedExecutableSha256),
                    StringComparison.Ordinal))
            {
                return Failure(
                    EnhancementFailureCode.RuntimeChecksumMismatch,
                    "The Real-ESRGAN executable checksum does not match configuration.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement);
            }

            string? modelFailure = await VerifyModelAsync(
                request.Model,
                modelRoot,
                cancellationToken).ConfigureAwait(false);
            if (modelFailure is not null)
            {
                return Failure(
                    modelFailure.Contains("checksum", StringComparison.OrdinalIgnoreCase)
                        ? EnhancementFailureCode.ModelChecksumMismatch
                        : EnhancementFailureCode.ModelMissing,
                    modelFailure,
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement);
            }

            string sourceSha256 = await EnhancementHashing.ComputeFileSha256Async(
                inputPath,
                cancellationToken).ConfigureAwait(false);
            string cacheKey = CreateCacheKey(
                sourceSha256,
                runtimeSha256,
                request,
                options);
            PixelDimensions enhancedDimensions = new(
                checked(request.SourceDimensions.Width * options.Scale),
                checked(request.SourceDimensions.Height * options.Scale));
            preprocess.Stop();

            string processGateKey = EnhancementHashing.ComputeCacheKey(
                Path.GetFullPath(_configuration.CacheDirectory).ToUpperInvariant(),
                cacheKey);
            await using KeyedGate.Lease lease = await ProcessKeyedGate.AcquireAsync(
                processGateKey,
                cancellationToken).ConfigureAwait(false);
            await using EnhancementCache.EntryLease entryLease = await _cache.AcquireEntryLeaseAsync(
                cacheKey,
                cancellationToken).ConfigureAwait(false);

            if (File.Exists(outputPath))
            {
                return Failure(
                    EnhancementFailureCode.OutputAlreadyExists,
                    "The requested derivative output appeared while enhancement was waiting and will not be overwritten.",
                    options);
            }

            workDirectory = _cache.CreateWorkDirectory();
            EnhancementCacheMetadata expectedCache = CreateCacheMetadata(
                cacheKey,
                sourceSha256,
                string.Empty,
                runtimeSha256,
                request,
                options,
                enhancedDimensions);
            string cachedOutputPath = Path.Combine(workDirectory, "cached-enhanced.png");
            CacheRestoreResult restored = await _cache.TryRestoreAsync(
                expectedCache,
                cachedOutputPath,
                cancellationToken).ConfigureAwait(false);
            if (restored.Hit)
            {
                string cachedSourceHash = await EnhancementHashing.ComputeFileSha256Async(
                    inputPath,
                    cancellationToken).ConfigureAwait(false);
                if (!string.Equals(sourceSha256, cachedSourceHash, StringComparison.Ordinal))
                {
                    return Failure(
                        EnhancementFailureCode.SourceChanged,
                        "The source image changed while restoring its cached derivative, so the derivative was discarded.",
                        options,
                        EnhancementStatus.ContinuedWithoutEnhancement);
                }

                string cachedRuntimeHash = await EnhancementHashing.ComputeFileSha256Async(
                    executablePath,
                    cancellationToken).ConfigureAwait(false);
                if (!string.Equals(runtimeSha256, cachedRuntimeHash, StringComparison.Ordinal))
                {
                    return Failure(
                        EnhancementFailureCode.RuntimeChecksumMismatch,
                        "The configured runtime changed while restoring a cached derivative, so the derivative was discarded.",
                        options,
                        EnhancementStatus.ContinuedWithoutEnhancement);
                }

                if (await VerifyModelAsync(request.Model, modelRoot, cancellationToken).ConfigureAwait(false) is not null)
                {
                    return Failure(
                        EnhancementFailureCode.ModelChecksumMismatch,
                        "The configured model changed while restoring a cached derivative, so the derivative was discarded.",
                        options,
                        EnhancementStatus.ContinuedWithoutEnhancement);
                }

                try
                {
                    await EnhancementCache.PromoteCopyAsync(
                        cachedOutputPath,
                        outputPath,
                        cancellationToken).ConfigureAwait(false);
                }
                catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
                {
                    return Failure(
                        EnhancementFailureCode.CacheFailure,
                        $"Verified cached enhancement could not be committed atomically: {exception.Message}",
                        options,
                        EnhancementStatus.ContinuedWithoutEnhancement);
                }

                total.Stop();
                EnhancementEnvelope envelope = CreateEnvelope(
                    request,
                    options,
                    sourceSha256,
                    restored.OutputSha256!,
                    runtimeSha256,
                    cacheKey,
                    cacheHit: true,
                    enhancedDimensions,
                    new EnhancementTiming(preprocess.Elapsed.TotalMilliseconds, 0, 0, total.Elapsed.TotalMilliseconds));
                return Success(EnhancementStatus.CacheHit, outputPath, envelope);
            }

            string processInputPath = Path.Combine(workDirectory, "source-input" + Path.GetExtension(inputPath));
            string processOutputPath = Path.Combine(workDirectory, "enhanced.png");
            await EnhancementCache.CopyFileAsync(
                inputPath,
                processInputPath,
                cancellationToken).ConfigureAwait(false);
            string stagedSourceHash = await EnhancementHashing.ComputeFileSha256Async(
                processInputPath,
                cancellationToken).ConfigureAwait(false);
            if (!string.Equals(sourceSha256, stagedSourceHash, StringComparison.Ordinal))
            {
                return Failure(
                    EnhancementFailureCode.SourceChanged,
                    "The source image changed while creating the private process input, so enhancement was not started.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement);
            }

            ProcessInvocation invocation = CreateInvocation(
                executablePath,
                modelRoot,
                processInputPath,
                processOutputPath,
                request.Model.RuntimeModelName,
                options);

            ProcessExecutionResult execution = await _processRunner.RunAsync(
                invocation,
                cancellationToken).ConfigureAwait(false);
            if (execution.Completion == ProcessCompletion.Cancelled)
            {
                return Failure(
                    EnhancementFailureCode.ProcessCancelled,
                    "Real-ESRGAN enhancement was cancelled.",
                    options,
                    EnhancementStatus.Cancelled,
                    execution);
            }

            if (execution.Completion == ProcessCompletion.TimedOut)
            {
                return Failure(
                    EnhancementFailureCode.ProcessTimedOut,
                    "Real-ESRGAN enhancement exceeded its configured timeout.",
                    options,
                    EnhancementStatus.TimedOut,
                    execution);
            }

            if (execution.Completion == ProcessCompletion.StartFailed)
            {
                return Failure(
                    EnhancementFailureCode.ProcessStartFailed,
                    "The Real-ESRGAN runtime could not be started.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement,
                    execution);
            }

            if (execution.ExitCode != 0)
            {
                return Failure(
                    EnhancementFailureCode.ProcessFailed,
                    "The Real-ESRGAN runtime returned a nonzero exit code.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement,
                    execution);
            }

            var postprocess = Stopwatch.StartNew();
            if (!File.Exists(processOutputPath))
            {
                return Failure(
                    EnhancementFailureCode.OutputMissing,
                    "The Real-ESRGAN runtime completed without producing an output image.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement,
                    execution);
            }

            PixelDimensions actualDimensions;
            try
            {
                actualDimensions = _imageInspector.ReadDimensions(processOutputPath);
            }
            catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException)
            {
                return Failure(
                    EnhancementFailureCode.OutputCorrupt,
                    $"The Real-ESRGAN output could not be verified: {exception.Message}",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement,
                    execution);
            }

            if (actualDimensions != enhancedDimensions)
            {
                return Failure(
                    EnhancementFailureCode.DimensionMismatch,
                    $"Expected {enhancedDimensions.Width}x{enhancedDimensions.Height} pixels but received {actualDimensions.Width}x{actualDimensions.Height}.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement,
                    execution);
            }

            string outputSha256 = await EnhancementHashing.ComputeFileSha256Async(
                processOutputPath,
                cancellationToken).ConfigureAwait(false);
            string finalSourceHash = await EnhancementHashing.ComputeFileSha256Async(
                inputPath,
                cancellationToken).ConfigureAwait(false);
            if (!string.Equals(sourceSha256, finalSourceHash, StringComparison.Ordinal))
            {
                return Failure(
                    EnhancementFailureCode.SourceChanged,
                    "The source image changed during enhancement, so the derivative was discarded.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement,
                    execution);
            }

            string finalRuntimeHash = await EnhancementHashing.ComputeFileSha256Async(
                executablePath,
                cancellationToken).ConfigureAwait(false);
            if (!string.Equals(runtimeSha256, finalRuntimeHash, StringComparison.Ordinal))
            {
                return Failure(
                    EnhancementFailureCode.RuntimeChecksumMismatch,
                    "The configured runtime changed during enhancement, so the derivative was discarded.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement,
                    execution);
            }

            if (await VerifyModelAsync(request.Model, modelRoot, cancellationToken).ConfigureAwait(false) is not null)
            {
                return Failure(
                    EnhancementFailureCode.ModelChecksumMismatch,
                    "The configured model changed during enhancement, so the derivative was discarded.",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement,
                    execution);
            }

            EnhancementCacheMetadata metadata = CreateCacheMetadata(
                cacheKey,
                sourceSha256,
                outputSha256,
                runtimeSha256,
                request,
                options,
                enhancedDimensions);
            try
            {
                await _cache.StoreAsync(processOutputPath, metadata, cancellationToken).ConfigureAwait(false);
                await EnhancementCache.PromoteCopyAsync(
                    processOutputPath,
                    outputPath,
                    cancellationToken).ConfigureAwait(false);
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                return Failure(
                    EnhancementFailureCode.CacheFailure,
                    $"Verified enhancement could not be committed atomically: {exception.Message}",
                    options,
                    EnhancementStatus.ContinuedWithoutEnhancement,
                    execution);
            }

            postprocess.Stop();
            total.Stop();
            EnhancementEnvelope completedEnvelope = CreateEnvelope(
                request,
                options,
                sourceSha256,
                outputSha256,
                runtimeSha256,
                cacheKey,
                cacheHit: false,
                enhancedDimensions,
                new EnhancementTiming(
                    preprocess.Elapsed.TotalMilliseconds,
                    execution.Duration.TotalMilliseconds,
                    postprocess.Elapsed.TotalMilliseconds,
                    total.Elapsed.TotalMilliseconds));
            return Success(EnhancementStatus.Succeeded, outputPath, completedEnvelope);
        }
        catch (OperationCanceledException)
        {
            return Failure(
                EnhancementFailureCode.ProcessCancelled,
                "Real-ESRGAN enhancement was cancelled.",
                request.Options ?? new EnhancementOptions(),
                EnhancementStatus.Cancelled);
        }
        catch (Exception exception) when (exception is ArgumentException or IOException or UnauthorizedAccessException or OverflowException)
        {
            return Failure(
                EnhancementFailureCode.InvalidRequest,
                exception.Message,
                request.Options ?? new EnhancementOptions());
        }
        finally
        {
            if (workDirectory is not null)
            {
                _cache.DeleteWorkDirectory(workDirectory);
            }
        }
    }

    public ProcessInvocation CreateInvocation(
        string executablePath,
        string modelsDirectory,
        string inputPath,
        string outputPath,
        string modelId,
        EnhancementOptions options)
    {
        TimeSpan timeout = options.Timeout ?? _configuration.DefaultTimeout ?? EnhancementDefaults.Timeout;
        string[] arguments =
        [
            "-i", inputPath,
            "-o", outputPath,
            "-n", modelId,
            "-s", options.Scale.ToString(CultureInfo.InvariantCulture),
            "-t", options.TileSize.ToString(CultureInfo.InvariantCulture),
            "-m", modelsDirectory,
            "-g", options.GpuIndex.ToString(CultureInfo.InvariantCulture),
            "-f", "png"
        ];
        return new ProcessInvocation(
            executablePath,
            Path.GetDirectoryName(executablePath)!,
            Array.AsReadOnly(arguments),
            timeout,
            _configuration.MaxDiagnosticCharacters);
    }

    private EnhancementResult? ValidateRequest(EnhancementRequest request, EnhancementOptions options)
    {
        if (request.Model is null)
        {
            return Failure(EnhancementFailureCode.InvalidRequest, "A model configuration is required.", options);
        }

        if (request.ProjectId == Guid.Empty || request.PanelId == Guid.Empty)
        {
            return Failure(EnhancementFailureCode.InvalidRequest, "Project and panel IDs are required.", options);
        }

        if (string.IsNullOrWhiteSpace(request.InputPath) || string.IsNullOrWhiteSpace(request.OutputPath))
        {
            return Failure(EnhancementFailureCode.InvalidRequest, "Input and output paths are required.", options);
        }

        if (EnhancementPaths.SamePath(request.InputPath, request.OutputPath))
        {
            return Failure(EnhancementFailureCode.InvalidRequest, "Enhancement output may not overwrite the original image.", options);
        }

        if (!request.SourceDimensions.IsPositive)
        {
            return Failure(EnhancementFailureCode.InvalidRequest, "Positive source dimensions are required.", options);
        }

        if (options.Scale != EnhancementDefaults.Scale)
        {
            return Failure(EnhancementFailureCode.InvalidRequest, "This adapter supports only reversible 2x enhancement.", options);
        }

        if (options.TileSize != 0 && options.TileSize < 32)
        {
            return Failure(EnhancementFailureCode.InvalidRequest, "Tile size must be zero for automatic selection or at least 32 pixels.", options);
        }

        if (options.GpuIndex < 0)
        {
            return Failure(EnhancementFailureCode.InvalidRequest, "GPU index must be zero or greater.", options);
        }

        TimeSpan timeout = options.Timeout ?? _configuration.DefaultTimeout ?? EnhancementDefaults.Timeout;
        if (timeout <= TimeSpan.Zero)
        {
            return Failure(EnhancementFailureCode.InvalidRequest, "Enhancement timeout must be positive.", options);
        }

        if (_configuration.MaxDiagnosticCharacters is < 256 or > 1_048_576)
        {
            return Failure(EnhancementFailureCode.InvalidRequest, "Diagnostic capture limit is outside the supported range.", options);
        }

        if (options.RequestCpuFallback)
        {
            return Failure(
                EnhancementFailureCode.CpuFallbackUnsupported,
                "realesrgan-ncnn-vulkan has no supported CPU execution provider. Continue with the original image.",
                options,
                EnhancementStatus.ContinuedWithoutEnhancement);
        }

        if (request.Model.Provider != EnhancementProvider.Vulkan ||
            !IsSafeModelId(request.Model.ModelId) ||
            !IsSafeModelId(request.Model.RuntimeModelName) ||
            !HasRuntimeBoundArtifacts(request.Model, options.Scale))
        {
            return Failure(
                EnhancementFailureCode.InvalidRequest,
                "A checksum-verified Vulkan NCNN model with exact runtime-bound scale-2 parameter and weight artifacts is required.",
                options);
        }

        if (!Path.IsPathFullyQualified(_configuration.ExecutablePath) ||
            !Path.IsPathFullyQualified(_configuration.ModelsDirectory) ||
            !Path.IsPathFullyQualified(_configuration.CacheDirectory))
        {
            return Failure(
                EnhancementFailureCode.InvalidRequest,
                "Runtime, model, and cache configuration paths must be fully qualified.",
                options);
        }

        if (string.IsNullOrWhiteSpace(request.Model.Version) ||
            string.IsNullOrWhiteSpace(request.Model.Source) ||
            string.IsNullOrWhiteSpace(request.Model.Revision) ||
            string.IsNullOrWhiteSpace(request.Model.LicenseSpdx) ||
            string.IsNullOrWhiteSpace(request.Model.NoticePath))
        {
            return Failure(
                EnhancementFailureCode.InvalidRequest,
                "Complete model version, source, revision, license, and notice provenance is required.",
                options);
        }

        return null;
    }

    private static bool IsSafeModelId(string modelId) =>
        !string.IsNullOrWhiteSpace(modelId) &&
        modelId.Length <= 128 &&
        char.IsAsciiLetterOrDigit(modelId[0]) &&
        char.IsAsciiLetterOrDigit(modelId[^1]) &&
        modelId.All(static character => char.IsAsciiLetterOrDigit(character) || character is '-' or '_');

    private static bool HasRuntimeBoundArtifacts(EnhancementModel model, int scale)
    {
        if (model.Artifacts.Count != 2)
        {
            return false;
        }

        string[] artifactNames = model.Artifacts
            .Select(static artifact => artifact.RelativePath)
            .ToArray();
        if (artifactNames.Any(static path =>
                string.IsNullOrWhiteSpace(path) ||
                Path.IsPathRooted(path) ||
                !string.Equals(Path.GetFileName(path), path, StringComparison.Ordinal)))
        {
            return false;
        }

        string scaledPrefix = $"{model.RuntimeModelName}-x{scale}";
        return MatchesRuntimePair(artifactNames, scaledPrefix) ||
               MatchesRuntimePair(artifactNames, model.RuntimeModelName);
    }

    private static bool MatchesRuntimePair(IEnumerable<string> artifactNames, string prefix)
    {
        var expected = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            $"{prefix}.param",
            $"{prefix}.bin"
        };
        return artifactNames.All(expected.Remove) && expected.Count == 0;
    }

    private static async Task<string?> VerifyModelAsync(
        EnhancementModel model,
        string modelRoot,
        CancellationToken cancellationToken)
    {
        _ = EnhancementHashing.NormalizeSha256(model.Sha256);
        foreach (ModelArtifact artifact in model.Artifacts)
        {
            string path = EnhancementPaths.ResolveArtifact(modelRoot, artifact.RelativePath);
            if (!File.Exists(path))
            {
                return $"Required model artifact '{artifact.RelativePath}' is missing.";
            }

            string expectedHash = EnhancementHashing.NormalizeSha256(artifact.Sha256);
            string actualHash = await EnhancementHashing.ComputeFileSha256Async(path, cancellationToken).ConfigureAwait(false);
            if (!string.Equals(expectedHash, actualHash, StringComparison.Ordinal))
            {
                return $"Model artifact '{artifact.RelativePath}' checksum does not match configuration.";
            }
        }

        return null;
    }

    private static string CreateCacheKey(
        string sourceSha256,
        string runtimeSha256,
        EnhancementRequest request,
        EnhancementOptions options) =>
        EnhancementHashing.ComputeCacheKey(
            ContractVersion.ToString(CultureInfo.InvariantCulture),
            StageVersion,
            sourceSha256,
            runtimeSha256,
            request.Model.ModelId,
            request.Model.RuntimeModelName,
            request.Model.Version,
            EnhancementHashing.NormalizeSha256(request.Model.Sha256),
            EnhancementHashing.ComputeModelSha256(
                request.Model.Artifacts.Select(static artifact =>
                    (artifact.RelativePath, EnhancementHashing.NormalizeSha256(artifact.Sha256)))),
            EnhancementProvider.Vulkan.ToString().ToLowerInvariant(),
            request.SourceDimensions.Width.ToString(CultureInfo.InvariantCulture),
            request.SourceDimensions.Height.ToString(CultureInfo.InvariantCulture),
            options.Scale.ToString(CultureInfo.InvariantCulture),
            options.TileSize.ToString(CultureInfo.InvariantCulture),
            options.GpuIndex.ToString(CultureInfo.InvariantCulture));

    private static EnhancementCacheMetadata CreateCacheMetadata(
        string cacheKey,
        string sourceSha256,
        string outputSha256,
        string runtimeSha256,
        EnhancementRequest request,
        EnhancementOptions options,
        PixelDimensions enhancedDimensions) =>
        new(
            ContractVersion,
            StageVersion,
            cacheKey,
            sourceSha256,
            outputSha256,
            EnhancementHashing.NormalizeSha256(request.Model.Sha256),
            runtimeSha256,
            enhancedDimensions.Width,
            enhancedDimensions.Height,
            options.Scale,
            options.TileSize,
            options.GpuIndex,
            "vulkan");

    private static EnhancementEnvelope CreateEnvelope(
        EnhancementRequest request,
        EnhancementOptions options,
        string sourceSha256,
        string outputSha256,
        string runtimeSha256,
        string cacheKey,
        bool cacheHit,
        PixelDimensions enhancedDimensions,
        EnhancementTiming timing) =>
        new(
            ContractVersion,
            Guid.NewGuid(),
            request.ProjectId,
            request.PanelId,
            "enhancement",
            StageVersion,
            sourceSha256,
            "original_pixels",
            new EnhancementModelProvenance(
                request.Model.ModelId,
                request.Model.Version,
                EnhancementHashing.NormalizeSha256(request.Model.Sha256),
                "vulkan"),
            timing,
            1,
            EvidenceWarnings,
            new EnhancementPayload(
                outputSha256,
                request.SourceDimensions,
                enhancedDimensions,
                runtimeSha256,
                cacheKey,
                cacheHit,
                EnhancementTransform.CreateScale2(),
                new EnhancementModelAudit(
                    request.Model.Source,
                    request.Model.Revision,
                    request.Model.LicenseSpdx,
                    request.Model.NoticePath,
                    EnhancementHashing.ComputeModelSha256(
                        request.Model.Artifacts.Select(static artifact =>
                            (artifact.RelativePath, EnhancementHashing.NormalizeSha256(artifact.Sha256))))),
                options.TileSize,
                options.GpuIndex));

    private static EnhancementResult Success(
        EnhancementStatus status,
        string outputPath,
        EnhancementEnvelope envelope) =>
        new(
            status,
            outputPath,
            new EnhancementDiagnostic(EnhancementFailureCode.None, "Enhancement completed successfully."),
            envelope,
            MayContinueUnenhanced: true);

    private static EnhancementResult Failure(
        EnhancementFailureCode code,
        string message,
        EnhancementOptions options,
        EnhancementStatus status = EnhancementStatus.Failed,
        ProcessExecutionResult? execution = null)
    {
        bool mayContinue = options.ContinueWithoutEnhancement;
        EnhancementStatus effectiveStatus = status == EnhancementStatus.ContinuedWithoutEnhancement && !mayContinue
            ? EnhancementStatus.Failed
            : status;
        string standardError = execution is null
            ? string.Empty
            : string.IsNullOrEmpty(execution.StandardError)
                ? execution.StartError ?? string.Empty
                : execution.StandardError;
        if (!string.IsNullOrWhiteSpace(execution?.TerminationError))
        {
            standardError = string.IsNullOrEmpty(standardError)
                ? execution.TerminationError
                : $"{standardError}{Environment.NewLine}{execution.TerminationError}";
        }
        return new EnhancementResult(
            effectiveStatus,
            null,
            new EnhancementDiagnostic(
                code,
                message,
                execution?.ExitCode,
                execution?.StandardOutput ?? string.Empty,
                standardError),
            null,
            mayContinue);
    }

    private sealed class KeyedGate
    {
        private readonly ConcurrentDictionary<string, Entry> _entries = new(StringComparer.Ordinal);

        public async ValueTask<Lease> AcquireAsync(string key, CancellationToken cancellationToken)
        {
            while (true)
            {
                Entry entry = _entries.GetOrAdd(key, static _ => new Entry());
                lock (entry.Sync)
                {
                    if (entry.Retired)
                    {
                        continue;
                    }

                    entry.ReferenceCount++;
                }

                try
                {
                    await entry.Semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
                    return new Lease(this, key, entry);
                }
                catch
                {
                    ReleaseReference(key, entry, releaseSemaphore: false);
                    throw;
                }
            }
        }

        private void Release(string key, Entry entry) =>
            ReleaseReference(key, entry, releaseSemaphore: true);

        private void ReleaseReference(string key, Entry entry, bool releaseSemaphore)
        {
            if (releaseSemaphore)
            {
                entry.Semaphore.Release();
            }

            bool dispose = false;
            lock (entry.Sync)
            {
                entry.ReferenceCount--;
                if (entry.ReferenceCount == 0)
                {
                    entry.Retired = true;
                    dispose = _entries.TryRemove(new KeyValuePair<string, Entry>(key, entry));
                }
            }

            if (dispose)
            {
                entry.Semaphore.Dispose();
            }
        }

        internal sealed class Entry
        {
            public object Sync { get; } = new();
            public SemaphoreSlim Semaphore { get; } = new(1, 1);
            public int ReferenceCount { get; set; }
            public bool Retired { get; set; }
        }

        public sealed class Lease : IAsyncDisposable
        {
            private readonly KeyedGate _owner;
            private readonly string _key;
            private Entry? _entry;

            internal Lease(KeyedGate owner, string key, Entry entry)
            {
                _owner = owner;
                _key = key;
                _entry = entry;
            }

            public ValueTask DisposeAsync()
            {
                Entry? entry = Interlocked.Exchange(ref _entry, null);
                if (entry is not null)
                {
                    _owner.Release(_key, entry);
                }

                return ValueTask.CompletedTask;
            }
        }
    }
}
