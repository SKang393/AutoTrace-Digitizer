// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Text.Json.Serialization;

namespace GraphReader.SuperResolution;

public enum EnhancementStatus
{
    Succeeded,
    CacheHit,
    ContinuedWithoutEnhancement,
    Failed,
    Cancelled,
    TimedOut
}

public enum EnhancementFailureCode
{
    None,
    InvalidRequest,
    SourceMissing,
    SourceChanged,
    OutputAlreadyExists,
    RuntimeMissing,
    RuntimeChecksumMismatch,
    ModelMissing,
    ModelChecksumMismatch,
    CpuFallbackUnsupported,
    ProcessStartFailed,
    ProcessFailed,
    ProcessTimedOut,
    ProcessCancelled,
    OutputMissing,
    OutputCorrupt,
    DimensionMismatch,
    CacheFailure
}

public enum EnhancementProvider
{
    Vulkan
}

public readonly record struct PixelDimensions(
    [property: JsonPropertyName("width")] int Width,
    [property: JsonPropertyName("height")] int Height)
{
    public bool IsPositive => Width > 0 && Height > 0;
}

public readonly record struct EnhancementPoint(double X, double Y);

public sealed record EnhancementOptions(
    int Scale = EnhancementDefaults.Scale,
    int TileSize = 0,
    int GpuIndex = 0,
    TimeSpan? Timeout = null,
    bool ContinueWithoutEnhancement = true,
    bool RequestCpuFallback = false);

public static class EnhancementDefaults
{
    public const bool EnhanceOnImport = true;
    public const int Scale = 2;
    public static readonly TimeSpan Timeout = TimeSpan.FromMinutes(2);
}

public sealed record ModelArtifact(string RelativePath, string Sha256);

public sealed class EnhancementModel
{
    private readonly ReadOnlyCollection<ModelArtifact> _artifacts;

    public EnhancementModel(
        string modelId,
        string version,
        string sha256,
        string source,
        string revision,
        string licenseSpdx,
        string noticePath,
        IEnumerable<ModelArtifact> artifacts,
        EnhancementProvider provider = EnhancementProvider.Vulkan)
    {
        ModelId = modelId;
        Version = version;
        Sha256 = sha256;
        Source = source;
        Revision = revision;
        LicenseSpdx = licenseSpdx;
        NoticePath = noticePath;
        Provider = provider;
        _artifacts = Array.AsReadOnly(artifacts?.ToArray() ?? throw new ArgumentNullException(nameof(artifacts)));
    }

    public string ModelId { get; }
    public string Version { get; }
    public string Sha256 { get; }
    public string Source { get; }
    public string Revision { get; }
    public string LicenseSpdx { get; }
    public string NoticePath { get; }
    public EnhancementProvider Provider { get; }
    public IReadOnlyList<ModelArtifact> Artifacts => _artifacts;
}

public sealed record RealEsrganConfiguration(
    string ExecutablePath,
    string ModelsDirectory,
    string CacheDirectory,
    string? ExpectedExecutableSha256 = null,
    TimeSpan? DefaultTimeout = null,
    int MaxDiagnosticCharacters = 32_768);

public sealed record EnhancementRequest(
    Guid ProjectId,
    Guid PanelId,
    string InputPath,
    string OutputPath,
    PixelDimensions SourceDimensions,
    EnhancementModel Model,
    EnhancementOptions? Options = null);

public sealed record EnhancementDiagnostic(
    EnhancementFailureCode Code,
    string Message,
    int? ExitCode = null,
    string StandardOutput = "",
    string StandardError = "");

public sealed record EnhancementModelProvenance(
    [property: JsonPropertyName("model_id")] string ModelId,
    [property: JsonPropertyName("version")] string Version,
    [property: JsonPropertyName("sha256")] string Sha256,
    [property: JsonPropertyName("provider")] string Provider);

public sealed record EnhancementModelAudit(
    [property: JsonPropertyName("source")] string Source,
    [property: JsonPropertyName("revision")] string Revision,
    [property: JsonPropertyName("license_spdx")] string LicenseSpdx,
    [property: JsonPropertyName("notice_path")] string NoticePath,
    [property: JsonPropertyName("verified_artifact_set_sha256")] string VerifiedArtifactSetSha256);

public sealed record EnhancementTiming(
    [property: JsonPropertyName("preprocess")] double Preprocess,
    [property: JsonPropertyName("inference")] double Inference,
    [property: JsonPropertyName("postprocess")] double Postprocess,
    [property: JsonPropertyName("total")] double Total);

public sealed record EnhancementTransform(
    [property: JsonPropertyName("transform_id")] Guid TransformId,
    [property: JsonPropertyName("kind")] string Kind,
    [property: JsonPropertyName("source_space")] string SourceSpace,
    [property: JsonPropertyName("target_space")] string TargetSpace,
    [property: JsonPropertyName("matrix_3x3")] IReadOnlyList<double> Matrix3X3,
    [property: JsonPropertyName("inverse_matrix_3x3")] IReadOnlyList<double> InverseMatrix3X3,
    [property: JsonPropertyName("parameters")] IReadOnlyDictionary<string, double> Parameters,
    [property: JsonPropertyName("lossy")] bool Lossy)
{
    private static readonly IReadOnlyList<double> ForwardScale2 =
        Array.AsReadOnly<double>([2d, 0d, 0d, 0d, 2d, 0d, 0d, 0d, 1d]);
    private static readonly IReadOnlyList<double> InverseScale2 =
        Array.AsReadOnly<double>([0.5d, 0d, 0d, 0d, 0.5d, 0d, 0d, 0d, 1d]);
    private static readonly IReadOnlyDictionary<string, double> Scale2Parameters =
        new ReadOnlyDictionary<string, double>(new Dictionary<string, double>(StringComparer.Ordinal)
        {
            ["scale"] = EnhancementDefaults.Scale
        });

    [JsonIgnore]
    public int Scale => checked((int)Parameters["scale"]);

    public static EnhancementTransform CreateScale2() =>
        new(
            Guid.NewGuid(),
            "scale",
            "original_pixels",
            "enhanced_pixels",
            ForwardScale2,
            InverseScale2,
            Scale2Parameters,
            Lossy: false);

    public EnhancementPoint ToEnhanced(EnhancementPoint point) =>
        Apply(Matrix3X3, point);

    public EnhancementPoint ToOriginal(EnhancementPoint point) =>
        Apply(InverseMatrix3X3, point);

    private static EnhancementPoint Apply(IReadOnlyList<double> matrix, EnhancementPoint point)
    {
        double denominator = (matrix[6] * point.X) + (matrix[7] * point.Y) + matrix[8];
        if (Math.Abs(denominator) < double.Epsilon)
        {
            throw new InvalidOperationException("Enhancement transform maps the point to infinity.");
        }

        return new EnhancementPoint(
            ((matrix[0] * point.X) + (matrix[1] * point.Y) + matrix[2]) / denominator,
            ((matrix[3] * point.X) + (matrix[4] * point.Y) + matrix[5]) / denominator);
    }
}

public sealed record EnhancementPayload(
    [property: JsonPropertyName("output_sha256")] string OutputSha256,
    [property: JsonPropertyName("original_dimensions")] PixelDimensions OriginalDimensions,
    [property: JsonPropertyName("enhanced_dimensions")] PixelDimensions EnhancedDimensions,
    [property: JsonPropertyName("runtime_sha256")] string RuntimeSha256,
    [property: JsonPropertyName("cache_key")] string CacheKey,
    [property: JsonPropertyName("cache_hit")] bool CacheHit,
    [property: JsonPropertyName("transform")] EnhancementTransform Transform,
    [property: JsonPropertyName("model_provenance")] EnhancementModelAudit ModelProvenance,
    [property: JsonPropertyName("tile_size")] int TileSize,
    [property: JsonPropertyName("gpu_index")] int GpuIndex);

public sealed record EnhancementEnvelope(
    [property: JsonPropertyName("contract_version")] int ContractVersion,
    [property: JsonPropertyName("run_id")] Guid RunId,
    [property: JsonPropertyName("project_id")] Guid ProjectId,
    [property: JsonPropertyName("panel_id")] Guid PanelId,
    [property: JsonPropertyName("stage")] string Stage,
    [property: JsonPropertyName("stage_version")] string StageVersion,
    [property: JsonPropertyName("input_sha256")] string InputSha256,
    [property: JsonPropertyName("coordinate_space")] string CoordinateSpace,
    [property: JsonPropertyName("model")] EnhancementModelProvenance Model,
    [property: JsonPropertyName("timing_ms")] EnhancementTiming TimingMs,
    [property: JsonPropertyName("confidence")] double Confidence,
    [property: JsonPropertyName("warnings")] IReadOnlyList<string> Warnings,
    [property: JsonPropertyName("payload")] EnhancementPayload Payload);

public sealed record EnhancementResult(
    EnhancementStatus Status,
    string? OutputPath,
    EnhancementDiagnostic Diagnostic,
    EnhancementEnvelope? Envelope,
    bool MayContinueUnenhanced)
{
    public bool IsSuccess => Status is EnhancementStatus.Succeeded or EnhancementStatus.CacheHit;
}

public interface IEnhancementService
{
    Task<EnhancementResult> EnhanceAsync(EnhancementRequest request, CancellationToken cancellationToken);
}
