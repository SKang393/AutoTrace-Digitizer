// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;

namespace GraphReader.Pdf;

public static class PdfPageRenderFailureCodes
{
    public const string InvalidRequest = "PDF_RENDER_INVALID_REQUEST";
    public const string BackendProvenanceRejected = "PDF_RENDER_BACKEND_PROVENANCE_REJECTED";
    public const string BackendCapabilityMissing = "PDF_RENDER_BACKEND_CAPABILITY_MISSING";
    public const string BackendFailure = "PDF_RENDER_BACKEND_FAILURE";
    public const string BackendOutputInvalid = "PDF_RENDER_BACKEND_OUTPUT_INVALID";
    public const string Cancelled = "PDF_RENDER_CANCELLED";
}

public enum PdfPageRenderStatus
{
    Succeeded,
    CacheHit,
    Failed,
    Cancelled
}

public enum PdfPageRenderCacheDisposition
{
    Miss,
    MemoryHit,
    Coalesced
}

public sealed class ImmutableByteBuffer
{
    private readonly byte[] _bytes;

    public ImmutableByteBuffer(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        _bytes = (byte[])bytes.Clone();
    }

    public ImmutableByteBuffer(ReadOnlySpan<byte> bytes) =>
        _bytes = bytes.ToArray();

    public int Length => _bytes.Length;

    public byte[] ToArray() => (byte[])_bytes.Clone();

    internal ReadOnlyMemory<byte> Memory => _bytes;
}

public sealed class PdfPageRenderRequest
{
    public PdfPageRenderRequest(
        byte[] pdfBytes,
        int pageNumber,
        int dpi,
        int contractVersion = 1)
        : this(new ImmutableByteBuffer(pdfBytes), pageNumber, dpi, contractVersion)
    {
    }

    public PdfPageRenderRequest(
        ImmutableByteBuffer pdfBytes,
        int pageNumber,
        int dpi,
        int contractVersion = 1)
    {
        PdfBytes = pdfBytes ?? throw new ArgumentNullException(nameof(pdfBytes));
        PageNumber = pageNumber;
        Dpi = dpi;
        ContractVersion = contractVersion;
    }

    public ImmutableByteBuffer PdfBytes { get; }

    public int PageNumber { get; }

    public int Dpi { get; }

    public int ContractVersion { get; }
}

public sealed record PdfiumBackendProvenance(
    string RendererId,
    string RendererVersion,
    string BinarySha256,
    string Source,
    string SourceRevision,
    string LicenseSpdx,
    string NoticePath,
    bool ReviewApproved,
    bool RedistributionApproved,
    bool IsBundled);

public sealed record PdfiumBackendPolicyDecision(bool Approved, string TechnicalMessage)
{
    public static PdfiumBackendPolicyDecision Allow() => new(true, string.Empty);

    public static PdfiumBackendPolicyDecision Reject(string technicalMessage) =>
        new(false, technicalMessage);
}

public interface IPdfiumBackendProvenancePolicy
{
    string PolicyId { get; }

    PdfiumBackendPolicyDecision Evaluate(PdfiumBackendProvenance provenance);
}

public sealed class PdfiumCompatibleLicensePolicy : IPdfiumBackendProvenancePolicy
{
    public static PdfiumCompatibleLicensePolicy Default { get; } = new();

    public string PolicyId => "pdfium-compatible-license-v1";

    public PdfiumBackendPolicyDecision Evaluate(PdfiumBackendProvenance provenance) =>
        provenance.IsBundled
            ? PdfiumBackendPolicyDecision.Reject(
                "The compatible-license policy does not independently approve bundled native binaries.")
            : PdfiumBackendPolicyDecision.Allow();
}

public sealed record PdfiumBackendApproval(
    string RendererId,
    string RendererVersion,
    string BinarySha256,
    string Source,
    string SourceRevision,
    string LicenseSpdx,
    string NoticePath,
    bool BundlingApproved);

public sealed class ExactPdfiumBackendApprovalPolicy : IPdfiumBackendProvenancePolicy
{
    private readonly PdfiumBackendApproval _approval;

    public ExactPdfiumBackendApprovalPolicy(PdfiumBackendApproval approval) =>
        _approval = approval ?? throw new ArgumentNullException(nameof(approval));

    public string PolicyId => "pdfium-exact-independent-approval-v1";

    public PdfiumBackendPolicyDecision Evaluate(PdfiumBackendProvenance provenance)
    {
        if (!Matches(provenance.RendererId, _approval.RendererId) ||
            !Matches(provenance.RendererVersion, _approval.RendererVersion) ||
            !Matches(provenance.BinarySha256, _approval.BinarySha256) ||
            !Matches(provenance.Source, _approval.Source) ||
            !Matches(provenance.SourceRevision, _approval.SourceRevision) ||
            !Matches(provenance.LicenseSpdx, _approval.LicenseSpdx) ||
            !Matches(provenance.NoticePath, _approval.NoticePath))
        {
            return PdfiumBackendPolicyDecision.Reject(
                "The backend provenance does not match the independently approved renderer artifact.");
        }

        return provenance.IsBundled && !_approval.BundlingApproved
            ? PdfiumBackendPolicyDecision.Reject(
                "The independently approved renderer artifact is not approved for bundling.")
            : PdfiumBackendPolicyDecision.Allow();
    }

    private static bool Matches(string left, string right) =>
        string.Equals(left, right, StringComparison.OrdinalIgnoreCase);
}

public sealed record PdfiumBackendCapabilities(
    bool SupportsPageRendering,
    bool SupportsCancellation,
    bool SupportsPngEncoding,
    bool IsLocalOnly,
    int MinimumDpi,
    int MaximumDpi);

public sealed record PdfiumBackendFailure(
    string Code,
    string TechnicalMessage,
    bool Recoverable,
    string SuggestedAction);

public sealed class PdfiumBackendRenderRequest
{
    public PdfiumBackendRenderRequest(
        ImmutableByteBuffer pdfBytes,
        string pdfSha256,
        int pageNumber,
        int dpi)
    {
        PdfBytes = pdfBytes ?? throw new ArgumentNullException(nameof(pdfBytes));
        PdfSha256 = pdfSha256;
        PageNumber = pageNumber;
        Dpi = dpi;
    }

    public ImmutableByteBuffer PdfBytes { get; }

    public string PdfSha256 { get; }

    public int PageNumber { get; }

    public int Dpi { get; }
}

public sealed record PdfRenderedPage(
    ImmutableByteBuffer PngBytes,
    int Width,
    int Height,
    string MediaType = "image/png");

public sealed class PdfiumBackendRenderResult
{
    private PdfiumBackendRenderResult(PdfRenderedPage? page, PdfiumBackendFailure? failure)
    {
        Page = page;
        Failure = failure;
    }

    public PdfRenderedPage? Page { get; }

    public PdfiumBackendFailure? Failure { get; }

    public bool Succeeded => Page is not null && Failure is null;

    public static PdfiumBackendRenderResult Success(
        byte[] pngBytes,
        int width,
        int height) =>
        new(new PdfRenderedPage(new ImmutableByteBuffer(pngBytes), width, height), null);

    public static PdfiumBackendRenderResult Success(PdfRenderedPage page)
    {
        ArgumentNullException.ThrowIfNull(page);
        return new PdfiumBackendRenderResult(ClonePage(page), null);
    }

    public static PdfiumBackendRenderResult Failed(PdfiumBackendFailure failure)
    {
        ArgumentNullException.ThrowIfNull(failure);
        return new PdfiumBackendRenderResult(null, failure);
    }

    private static PdfRenderedPage ClonePage(PdfRenderedPage page) =>
        new(new ImmutableByteBuffer(page.PngBytes.ToArray()), page.Width, page.Height, page.MediaType);
}

public interface IPdfiumPageRendererBackend
{
    PdfiumBackendProvenance Provenance { get; }

    PdfiumBackendCapabilities Capabilities { get; }

    Task<PdfiumBackendRenderResult> RenderPageAsync(
        PdfiumBackendRenderRequest request,
        CancellationToken cancellationToken);
}

public sealed record PdfPageRenderMetadata(
    string CacheKey,
    PdfPageRenderCacheDisposition CacheDisposition,
    string PdfSha256,
    int PageNumber,
    int Dpi,
    string RendererId,
    string RendererVersion,
    string RendererSha256,
    int ContractVersion,
    double RenderMilliseconds)
{
    public bool CacheHit => CacheDisposition is not PdfPageRenderCacheDisposition.Miss;
}

public sealed record PdfPageRenderResult(
    PdfPageRenderStatus Status,
    PdfRenderedPage? Page,
    PdfPageRenderMetadata? Metadata,
    PdfFailure? Failure)
{
    public bool Succeeded =>
        Status is PdfPageRenderStatus.Succeeded or PdfPageRenderStatus.CacheHit &&
        Page is not null &&
        Failure is null;
}

public sealed record PdfPageRenderCacheOptions(
    int MaximumEntries = 32,
    long MaximumEncodedBytes = 256L * 1024L * 1024L);

public sealed record PdfPageRenderSafetyLimits(
    int MaximumWidth = 32_768,
    int MaximumHeight = 32_768,
    long MaximumPixelCount = 268_435_456,
    long MaximumEncodedBytes = 256L * 1024L * 1024L,
    long MaximumDecodedBytes = 512L * 1024L * 1024L,
    int MaximumChunkBytes = 64 * 1024 * 1024,
    int MaximumChunkCount = 100_000);

public sealed class PdfPageRenderCacheKey : IEquatable<PdfPageRenderCacheKey>
{
    private PdfPageRenderCacheKey(
        string digest,
        string pdfSha256,
        int pageNumber,
        int dpi,
        string rendererId,
        string rendererVersion,
        string rendererSha256,
        int contractVersion)
    {
        Digest = digest;
        PdfSha256 = pdfSha256;
        PageNumber = pageNumber;
        Dpi = dpi;
        RendererId = rendererId;
        RendererVersion = rendererVersion;
        RendererSha256 = rendererSha256;
        ContractVersion = contractVersion;
    }

    public string Digest { get; }

    public string PdfSha256 { get; }

    public int PageNumber { get; }

    public int Dpi { get; }

    public string RendererId { get; }

    public string RendererVersion { get; }

    public string RendererSha256 { get; }

    public int ContractVersion { get; }

    public static PdfPageRenderCacheKey Create(
        ImmutableByteBuffer pdfBytes,
        int pageNumber,
        int dpi,
        PdfiumBackendProvenance provenance,
        int contractVersion,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pdfBytes);
        ArgumentNullException.ThrowIfNull(provenance);

        string pdfSha256 = ComputeSha256(pdfBytes.Memory, cancellationToken);
        string rendererSha256 = provenance.BinarySha256.ToLowerInvariant();

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Append(hash, "graph-reader-pdf-page-render-cache-v1");
        Append(hash, pdfSha256);
        Append(hash, pageNumber.ToString(CultureInfo.InvariantCulture));
        Append(hash, dpi.ToString(CultureInfo.InvariantCulture));
        Append(hash, provenance.RendererId);
        Append(hash, provenance.RendererVersion);
        Append(hash, rendererSha256);
        Append(hash, contractVersion.ToString(CultureInfo.InvariantCulture));

        return new PdfPageRenderCacheKey(
            Convert.ToHexStringLower(hash.GetHashAndReset()),
            pdfSha256,
            pageNumber,
            dpi,
            provenance.RendererId,
            provenance.RendererVersion,
            rendererSha256,
            contractVersion);
    }

    public bool Equals(PdfPageRenderCacheKey? other) =>
        other is not null && string.Equals(Digest, other.Digest, StringComparison.Ordinal);

    public override bool Equals(object? obj) =>
        obj is PdfPageRenderCacheKey other && Equals(other);

    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(Digest);

    public override string ToString() => Digest;

    private static string ComputeSha256(
        ReadOnlyMemory<byte> bytes,
        CancellationToken cancellationToken)
    {
        const int chunkSize = 64 * 1024;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        for (var offset = 0; offset < bytes.Length; offset += chunkSize)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSize, bytes.Length - offset);
            hash.AppendData(bytes.Span.Slice(offset, count));
        }

        cancellationToken.ThrowIfCancellationRequested();
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    private static void Append(IncrementalHash hash, string value)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(value);
        Span<byte> length = stackalloc byte[sizeof(int)];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(length, encoded.Length);
        hash.AppendData(length);
        hash.AppendData(encoded);
    }
}

public interface IPdfPageRenderingService
{
    Task<PdfPageRenderResult> RenderAsync(
        PdfPageRenderRequest request,
        CancellationToken cancellationToken);
}

public sealed class PdfiumPageRendererAdapter : IPdfPageRenderingService
{
    private static readonly HashSet<string> CompatibleRendererLicenses = new(StringComparer.OrdinalIgnoreCase)
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "ISC",
        "MIT",
        "Zlib",
    };

    private readonly IPdfiumPageRendererBackend _backend;
    private readonly IPdfiumBackendProvenancePolicy _provenancePolicy;
    private readonly PdfPageRenderSafetyLimits _safetyLimits;
    private readonly BoundedPageRenderCache _cache;
    private readonly ConcurrentDictionary<PdfPageRenderCacheKey, InFlightRender> _inFlight = new();

    public PdfiumPageRendererAdapter(
        IPdfiumPageRendererBackend backend,
        PdfPageRenderCacheOptions? cacheOptions = null)
        : this(
            backend,
            PdfiumCompatibleLicensePolicy.Default,
            cacheOptions,
            safetyLimits: null)
    {
    }

    public PdfiumPageRendererAdapter(
        IPdfiumPageRendererBackend backend,
        IPdfiumBackendProvenancePolicy provenancePolicy,
        PdfPageRenderCacheOptions? cacheOptions = null,
        PdfPageRenderSafetyLimits? safetyLimits = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _provenancePolicy = provenancePolicy ?? throw new ArgumentNullException(nameof(provenancePolicy));
        _safetyLimits = safetyLimits ?? new PdfPageRenderSafetyLimits();
        ValidateSafetyLimits(_safetyLimits);
        _cache = new BoundedPageRenderCache(cacheOptions ?? new PdfPageRenderCacheOptions());
    }

    public int CachedEntryCount => _cache.Count;

    public long CachedEncodedBytes => _cache.EncodedBytes;

    public async Task<PdfPageRenderResult> RenderAsync(
        PdfPageRenderRequest request,
        CancellationToken cancellationToken)
    {
        if (cancellationToken.IsCancellationRequested)
        {
            return Cancelled("Page rendering was cancelled before validation.");
        }

        PdfFailure? requestFailure = ValidateRequest(request);
        if (requestFailure is not null)
        {
            return Failed(requestFailure);
        }

        PdfiumBackendProvenance? provenance;
        PdfiumBackendCapabilities? capabilities;
        try
        {
            provenance = _backend.Provenance;
            capabilities = _backend.Capabilities;
        }
        catch (Exception exception)
        {
            return Failed(Error(
                PdfPageRenderFailureCodes.BackendProvenanceRejected,
                "Errors.PdfRenderBackendNotReviewed",
                $"The injected PDFium backend could not report its identity: " +
                $"{exception.GetType().Name}: {exception.Message}",
                recoverable: false,
                "install_reviewed_renderer"));
        }

        PdfFailure? provenanceFailure = ValidateProvenance(provenance, _provenancePolicy);
        if (provenanceFailure is not null)
        {
            return Failed(provenanceFailure);
        }

        PdfFailure? capabilityFailure = ValidateCapabilities(capabilities, request.Dpi);
        if (capabilityFailure is not null)
        {
            return Failed(capabilityFailure);
        }

        PdfPageRenderCacheKey key;
        try
        {
            key = PdfPageRenderCacheKey.Create(
                request.PdfBytes,
                request.PageNumber,
                request.Dpi,
                provenance!,
                request.ContractVersion,
                cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return Cancelled("Page rendering was cancelled while hashing the PDF.");
        }

        if (_cache.TryGet(key, out CachedPage? cached))
        {
            return CreateSuccess(cached!, PdfPageRenderCacheDisposition.MemoryHit);
        }

        InFlightRender inFlight = _inFlight.GetOrAdd(key, static _ => new InFlightRender());
        bool isLeader = inFlight.Attach();
        Task<PdfPageRenderResult> sharedTask = inFlight.GetOrStart(
            token => RenderAndCacheAsync(request, key, provenance!, token),
            () => _inFlight.TryRemove(new KeyValuePair<PdfPageRenderCacheKey, InFlightRender>(key, inFlight)));

        try
        {
            PdfPageRenderResult result = await sharedTask.WaitAsync(cancellationToken).ConfigureAwait(false);
            return !isLeader && result.Succeeded
                ? AsCoalesced(result)
                : CloneResult(result);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return Cancelled("Page rendering was cancelled by the caller.");
        }
        catch (Exception exception)
        {
            return Failed(Error(
                PdfPageRenderFailureCodes.BackendFailure,
                "Errors.PdfRenderBackendFailed",
                $"The page-render operation failed internally with {exception.GetType().Name}: " +
                exception.Message,
                recoverable: true,
                "retry"));
        }
        finally
        {
            if (inFlight.Detach() && !sharedTask.IsCompleted)
            {
                _inFlight.TryRemove(
                    new KeyValuePair<PdfPageRenderCacheKey, InFlightRender>(key, inFlight));
                inFlight.Cancel();
            }
        }
    }

    private async Task<PdfPageRenderResult> RenderAndCacheAsync(
        PdfPageRenderRequest request,
        PdfPageRenderCacheKey key,
        PdfiumBackendProvenance provenance,
        CancellationToken cancellationToken)
    {
        if (_cache.TryGet(key, out CachedPage? cached))
        {
            return CreateSuccess(cached!, PdfPageRenderCacheDisposition.MemoryHit);
        }

        var stopwatch = Stopwatch.StartNew();
        PdfiumBackendRenderResult backendResult;
        try
        {
            var backendRequest = new PdfiumBackendRenderRequest(
                new ImmutableByteBuffer(request.PdfBytes.ToArray()),
                key.PdfSha256,
                request.PageNumber,
                request.Dpi);
            backendResult = await _backend.RenderPageAsync(backendRequest, cancellationToken)
                .ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return Cancelled("The PDFium backend acknowledged cancellation.");
        }
        catch (Exception exception)
        {
            return Failed(Error(
                PdfPageRenderFailureCodes.BackendFailure,
                "Errors.PdfRenderBackendFailed",
                $"The PDFium backend threw {exception.GetType().Name}: {exception.Message}",
                recoverable: true,
                "retry"));
        }
        finally
        {
            stopwatch.Stop();
        }

        if (backendResult is null)
        {
            return Failed(InvalidBackendOutput("The PDFium backend returned a null result."));
        }

        if (!backendResult.Succeeded)
        {
            PdfiumBackendFailure? backendFailure = backendResult.Failure;
            if (backendFailure is null)
            {
                return Failed(InvalidBackendOutput(
                    "The PDFium backend returned neither a rendered page nor a failure."));
            }

            return Failed(Error(
                PdfPageRenderFailureCodes.BackendFailure,
                "Errors.PdfRenderBackendFailed",
                $"Backend code '{backendFailure.Code}': {backendFailure.TechnicalMessage}",
                backendFailure.Recoverable,
                backendFailure.SuggestedAction));
        }

        PdfRenderedPage page = backendResult.Page!;
        PdfFailure? outputFailure;
        try
        {
            outputFailure = ValidateOutput(page, _safetyLimits, cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return Cancelled("Page rendering was cancelled while validating PNG output.");
        }
        if (outputFailure is not null)
        {
            return Failed(outputFailure);
        }

        var metadata = new PdfPageRenderMetadata(
            key.Digest,
            PdfPageRenderCacheDisposition.Miss,
            key.PdfSha256,
            key.PageNumber,
            key.Dpi,
            provenance.RendererId,
            provenance.RendererVersion,
            key.RendererSha256,
            key.ContractVersion,
            stopwatch.Elapsed.TotalMilliseconds);
        var completed = new CachedPage(ClonePage(page), metadata);
        _cache.Store(key, completed);
        return CreateSuccess(completed, PdfPageRenderCacheDisposition.Miss);
    }

    private static PdfFailure? ValidateRequest(PdfPageRenderRequest? request)
    {
        if (request is null)
        {
            return InvalidRequest("A page-render request is required.");
        }

        if (request.PdfBytes.Length == 0)
        {
            return InvalidRequest("PDF content must not be empty.");
        }

        if (request.PageNumber < 1)
        {
            return InvalidRequest("Page numbers are one-based and must be at least 1.");
        }

        if (request.Dpi < 1)
        {
            return InvalidRequest("Render DPI must be positive.");
        }

        if (request.ContractVersion < 1)
        {
            return InvalidRequest("The render contract version must be positive.");
        }

        return null;
    }

    private static PdfFailure? ValidateProvenance(
        PdfiumBackendProvenance? provenance,
        IPdfiumBackendProvenancePolicy provenancePolicy)
    {
        if (provenance is null)
        {
            return InvalidProvenance("The injected PDFium backend did not provide provenance.");
        }

        if (string.IsNullOrWhiteSpace(provenance.RendererId) ||
            string.IsNullOrWhiteSpace(provenance.RendererVersion) ||
            string.IsNullOrWhiteSpace(provenance.Source) ||
            string.IsNullOrWhiteSpace(provenance.SourceRevision) ||
            string.IsNullOrWhiteSpace(provenance.LicenseSpdx) ||
            string.IsNullOrWhiteSpace(provenance.NoticePath))
        {
            return InvalidProvenance(
                "Renderer identity, version, source revision, license, and notice path are required.");
        }

        if (!IsSha256(provenance.BinarySha256))
        {
            return InvalidProvenance("The PDFium backend binary SHA-256 is missing or invalid.");
        }

        string normalizedLicense = provenance.LicenseSpdx.Trim();
        if (!CompatibleRendererLicenses.Contains(normalizedLicense))
        {
            return InvalidProvenance(
                $"Renderer license '{normalizedLicense}' is not an explicitly compatible permissive license. " +
                "GPL, AGPL, SSPL, BUSL, non-commercial, compound, and unclear licenses are rejected.");
        }

        if (IsMissingNoticePath(provenance.NoticePath))
        {
            return InvalidProvenance("The PDFium backend requires a concrete third-party notice path.");
        }

        if (!provenance.ReviewApproved)
        {
            return InvalidProvenance(
                "The backend provenance declaration is not marked reviewed; independent policy approval is also required.");
        }

        if (provenance.IsBundled && !provenance.RedistributionApproved)
        {
            return InvalidProvenance(
                "A bundled PDFium binary must declare redistribution approval and pass independent policy approval.");
        }

        PdfiumBackendPolicyDecision decision;
        try
        {
            decision = provenancePolicy.Evaluate(provenance);
        }
        catch (Exception exception)
        {
            return InvalidProvenance(
                $"The independent backend approval policy threw {exception.GetType().Name}: " +
                exception.Message);
        }

        if (decision is null || !decision.Approved)
        {
            return InvalidProvenance(
                string.IsNullOrWhiteSpace(decision?.TechnicalMessage)
                    ? "The independent backend approval policy rejected the renderer."
                    : decision.TechnicalMessage);
        }

        return null;
    }

    private static bool IsMissingNoticePath(string value)
    {
        string normalized = value.Trim();
        return normalized.Length == 0 ||
            normalized.Equals("none", StringComparison.OrdinalIgnoreCase) ||
            normalized.Equals("n/a", StringComparison.OrdinalIgnoreCase) ||
            normalized.Equals("unknown", StringComparison.OrdinalIgnoreCase);
    }

    private static PdfFailure? ValidateCapabilities(
        PdfiumBackendCapabilities? capabilities,
        int requestedDpi)
    {
        if (capabilities is null)
        {
            return MissingCapability("The injected PDFium backend did not declare capabilities.");
        }

        if (!capabilities.SupportsPageRendering ||
            !capabilities.SupportsCancellation ||
            !capabilities.SupportsPngEncoding ||
            !capabilities.IsLocalOnly)
        {
            return MissingCapability(
                "The PDFium backend must support local cancellable page rendering to PNG.");
        }

        if (capabilities.MinimumDpi < 1 || capabilities.MaximumDpi < capabilities.MinimumDpi)
        {
            return MissingCapability("The PDFium backend declared an invalid DPI range.");
        }

        if (requestedDpi < capabilities.MinimumDpi || requestedDpi > capabilities.MaximumDpi)
        {
            return MissingCapability(
                $"Requested DPI {requestedDpi} is outside the backend range " +
                $"{capabilities.MinimumDpi}..{capabilities.MaximumDpi}.");
        }

        return null;
    }

    private static PdfFailure? ValidateOutput(
        PdfRenderedPage page,
        PdfPageRenderSafetyLimits limits,
        CancellationToken cancellationToken)
    {
        if (page.PngBytes is null || page.PngBytes.Length == 0)
        {
            return InvalidBackendOutput("The PDFium backend returned empty image bytes.");
        }

        if (page.Width < 1 || page.Height < 1)
        {
            return InvalidBackendOutput("The PDFium backend returned non-positive page dimensions.");
        }

        if (!string.Equals(page.MediaType, "image/png", StringComparison.OrdinalIgnoreCase))
        {
            return InvalidBackendOutput("The PDFium backend must return image/png output.");
        }

        if (page.PngBytes.Length > limits.MaximumEncodedBytes)
        {
            return InvalidBackendOutput(
                $"The encoded PNG exceeds the {limits.MaximumEncodedBytes}-byte safety limit.");
        }

        return ValidatePngStructure(
            page.PngBytes.Memory,
            page.Width,
            page.Height,
            limits,
            cancellationToken);
    }

    private static PdfFailure? ValidatePngStructure(
        ReadOnlyMemory<byte> encoded,
        int reportedWidth,
        int reportedHeight,
        PdfPageRenderSafetyLimits limits,
        CancellationToken cancellationToken)
    {
        ReadOnlySpan<byte> signature = [137, 80, 78, 71, 13, 10, 26, 10];
        if (encoded.Length < signature.Length || !encoded.Span[..signature.Length].SequenceEqual(signature))
        {
            return InvalidBackendOutput("The renderer output does not have a valid PNG signature.");
        }

        var idatSegments = new List<PngDataSegment>();
        var offset = signature.Length;
        var chunkCount = 0;
        var seenHeader = false;
        var seenPalette = false;
        var seenTransparency = false;
        var seenImageData = false;
        var imageDataEnded = false;
        var seenEnd = false;
        var seenChromaticity = false;
        var seenGamma = false;
        var seenSrgb = false;
        var seenSignificantBits = false;
        var seenPhysicalDimensions = false;
        var colorType = -1;
        var bitDepth = 0;
        var paletteEntries = 0;
        long rowStride = 0;
        long expectedDecodedBytes = 0;

        while (offset < encoded.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            chunkCount++;
            if (chunkCount > limits.MaximumChunkCount)
            {
                return InvalidBackendOutput(
                    $"The PNG contains more than {limits.MaximumChunkCount} chunks.");
            }

            if (encoded.Length - offset < 12)
            {
                return InvalidBackendOutput("The PNG ends inside a chunk header or checksum.");
            }

            ReadOnlySpan<byte> remaining = encoded.Span[offset..];
            uint unsignedLength = System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(remaining[..4]);
            if (unsignedLength > int.MaxValue || unsignedLength > limits.MaximumChunkBytes)
            {
                return InvalidBackendOutput(
                    $"A PNG chunk exceeds the {limits.MaximumChunkBytes}-byte chunk safety limit.");
            }

            int chunkLength = (int)unsignedLength;
            if (chunkLength > encoded.Length - offset - 12)
            {
                return InvalidBackendOutput("A PNG chunk declares data beyond the encoded buffer.");
            }

            int typeOffset = offset + 4;
            int dataOffset = typeOffset + 4;
            int crcOffset = dataOffset + chunkLength;
            ReadOnlySpan<byte> typeBytes = encoded.Span.Slice(typeOffset, 4);
            if (!IsValidPngChunkType(typeBytes))
            {
                return InvalidBackendOutput("The PNG contains an invalid or reserved chunk type.");
            }

            uint storedCrc = System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(
                encoded.Span.Slice(crcOffset, 4));
            uint computedCrc = ComputePngCrc(
                encoded.Span.Slice(typeOffset, chunkLength + 4),
                cancellationToken);
            if (storedCrc != computedCrc)
            {
                return InvalidBackendOutput(
                    $"PNG chunk {Encoding.ASCII.GetString(typeBytes)} has an invalid CRC.");
            }

            uint chunkType = System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(typeBytes);
            ReadOnlySpan<byte> chunkData = encoded.Span.Slice(dataOffset, chunkLength);
            if (chunkCount == 1 && chunkType != PngChunkTypes.Header)
            {
                return InvalidBackendOutput("IHDR must be the first PNG chunk.");
            }

            switch (chunkType)
            {
                case PngChunkTypes.Header:
                    if (seenHeader || chunkCount != 1 || chunkLength != 13)
                    {
                        return InvalidBackendOutput("The PNG must contain one 13-byte IHDR first.");
                    }

                    uint unsignedWidth = System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(
                        chunkData[..4]);
                    uint unsignedHeight = System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(
                        chunkData.Slice(4, 4));
                    if (unsignedWidth == 0 || unsignedHeight == 0 ||
                        unsignedWidth > limits.MaximumWidth || unsignedHeight > limits.MaximumHeight)
                    {
                        return InvalidBackendOutput("PNG dimensions are zero or exceed configured safety limits.");
                    }

                    int width = checked((int)unsignedWidth);
                    int height = checked((int)unsignedHeight);
                    if (width != reportedWidth || height != reportedHeight)
                    {
                        return InvalidBackendOutput(
                            $"PNG dimensions {width}x{height} do not match the reported " +
                            $"{reportedWidth}x{reportedHeight} dimensions.");
                    }

                    long pixelCount = checked((long)width * height);
                    if (pixelCount > limits.MaximumPixelCount)
                    {
                        return InvalidBackendOutput(
                            $"The PNG exceeds the {limits.MaximumPixelCount}-pixel safety limit.");
                    }

                    bitDepth = chunkData[8];
                    colorType = chunkData[9];
                    if (!TryGetPngChannelCount(bitDepth, colorType, out int channelCount))
                    {
                        return InvalidBackendOutput("The PNG uses an unsupported color type or bit depth.");
                    }

                    if (chunkData[10] != 0 || chunkData[11] != 0 || chunkData[12] != 0)
                    {
                        return InvalidBackendOutput(
                            "Only standard compression/filtering and non-interlaced PNG output are supported.");
                    }

                    long rowBits = checked((long)width * channelCount * bitDepth);
                    long rowBytes = checked((rowBits + 7L) / 8L);
                    rowStride = checked(rowBytes + 1L);
                    expectedDecodedBytes = checked(rowStride * height);
                    if (expectedDecodedBytes > limits.MaximumDecodedBytes)
                    {
                        return InvalidBackendOutput(
                            $"The decoded PNG exceeds the {limits.MaximumDecodedBytes}-byte safety limit.");
                    }

                    seenHeader = true;
                    break;

                case PngChunkTypes.Palette:
                    if (!seenHeader || seenPalette || seenImageData || colorType is 0 or 4 ||
                        chunkLength is < 3 or > 768 || chunkLength % 3 != 0)
                    {
                        return InvalidBackendOutput("The PNG contains an invalid or misplaced PLTE chunk.");
                    }

                    paletteEntries = chunkLength / 3;
                    if (colorType == 3 && paletteEntries > 1 << bitDepth)
                    {
                        return InvalidBackendOutput("The indexed PNG palette exceeds its bit-depth capacity.");
                    }

                    seenPalette = true;
                    break;

                case PngChunkTypes.ImageData:
                    if (!seenHeader || seenEnd || imageDataEnded)
                    {
                        return InvalidBackendOutput("PNG IDAT chunks must be consecutive and precede IEND.");
                    }

                    if (colorType == 3 && !seenPalette)
                    {
                        return InvalidBackendOutput("An indexed PNG requires PLTE before IDAT.");
                    }

                    seenImageData = true;
                    idatSegments.Add(new PngDataSegment(dataOffset, chunkLength));
                    break;

                case PngChunkTypes.End:
                    imageDataEnded = seenImageData;
                    if (!seenHeader || !seenImageData || seenEnd || chunkLength != 0)
                    {
                        return InvalidBackendOutput("The PNG contains an invalid or misplaced IEND chunk.");
                    }

                    seenEnd = true;
                    if (crcOffset + 4 != encoded.Length)
                    {
                        return InvalidBackendOutput("The PNG contains trailing data after IEND.");
                    }

                    break;

                case PngChunkTypes.Transparency:
                    imageDataEnded = seenImageData;
                    if (seenTransparency || seenImageData ||
                        !ValidateTransparencyChunk(
                            colorType,
                            bitDepth,
                            seenPalette,
                            paletteEntries,
                            chunkData))
                    {
                        return InvalidBackendOutput("The PNG contains an invalid or misplaced tRNS chunk.");
                    }

                    seenTransparency = true;
                    break;

                case PngChunkTypes.Chromaticity:
                    imageDataEnded = seenImageData;
                    if (seenChromaticity || seenPalette || seenImageData || chunkLength != 32)
                    {
                        return InvalidBackendOutput("The PNG contains an invalid or misplaced cHRM chunk.");
                    }

                    seenChromaticity = true;
                    break;

                case PngChunkTypes.Gamma:
                    imageDataEnded = seenImageData;
                    if (seenGamma || seenPalette || seenImageData || chunkLength != 4 ||
                        System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(chunkData) == 0)
                    {
                        return InvalidBackendOutput("The PNG contains an invalid or misplaced gAMA chunk.");
                    }

                    seenGamma = true;
                    break;

                case PngChunkTypes.StandardRgb:
                    imageDataEnded = seenImageData;
                    if (seenSrgb || seenPalette || seenImageData || chunkLength != 1 || chunkData[0] > 3)
                    {
                        return InvalidBackendOutput("The PNG contains an invalid or misplaced sRGB chunk.");
                    }

                    seenSrgb = true;
                    break;

                case PngChunkTypes.SignificantBits:
                    imageDataEnded = seenImageData;
                    if (seenSignificantBits || seenPalette || seenImageData ||
                        !ValidateSignificantBits(colorType, bitDepth, chunkData))
                    {
                        return InvalidBackendOutput("The PNG contains an invalid or misplaced sBIT chunk.");
                    }

                    seenSignificantBits = true;
                    break;

                case PngChunkTypes.PhysicalDimensions:
                    imageDataEnded = seenImageData;
                    if (seenPhysicalDimensions || seenImageData || chunkLength != 9 || chunkData[8] > 1)
                    {
                        return InvalidBackendOutput("The PNG contains an invalid or misplaced pHYs chunk.");
                    }

                    seenPhysicalDimensions = true;
                    break;

                default:
                    return InvalidBackendOutput(
                        $"PNG chunk {Encoding.ASCII.GetString(typeBytes)} is outside the safe supported structure.");
            }

            offset = crcOffset + 4;
            if (seenEnd)
            {
                break;
            }

            if (seenImageData && chunkType != PngChunkTypes.ImageData)
            {
                imageDataEnded = true;
            }
        }

        long totalImageDataBytes = idatSegments.Aggregate(
            0L,
            static (total, segment) => checked(total + segment.Length));
        if (!seenHeader || !seenImageData || !seenEnd || totalImageDataBytes == 0)
        {
            return InvalidBackendOutput("The PNG is missing required IHDR, IDAT, or IEND content.");
        }

        return ValidatePngImageData(
            encoded,
            idatSegments,
            rowStride,
            expectedDecodedBytes,
            cancellationToken);
    }

    private static PdfFailure? ValidatePngImageData(
        ReadOnlyMemory<byte> encoded,
        IReadOnlyList<PngDataSegment> segments,
        long rowStride,
        long expectedDecodedBytes,
        CancellationToken cancellationToken)
    {
        long compressedLength = segments.Aggregate(
            0L,
            static (total, segment) => checked(total + segment.Length));
        if (compressedLength < 6)
        {
            return InvalidBackendOutput("The PNG IDAT zlib stream is too short.");
        }

        byte compressionMethod = GetPngImageDataByte(encoded, segments, 0);
        byte compressionFlags = GetPngImageDataByte(encoded, segments, 1);
        int zlibHeader = (compressionMethod << 8) | compressionFlags;
        if ((compressionMethod & 0x0F) != 8 || (compressionMethod >> 4) > 7 ||
            zlibHeader % 31 != 0 || (compressionFlags & 0x20) != 0)
        {
            return InvalidBackendOutput("The PNG IDAT data has an invalid or unsupported zlib header.");
        }

        uint expectedAdler32 =
            (uint)GetPngImageDataByte(encoded, segments, compressedLength - 4) << 24 |
            (uint)GetPngImageDataByte(encoded, segments, compressedLength - 3) << 16 |
            (uint)GetPngImageDataByte(encoded, segments, compressedLength - 2) << 8 |
            GetPngImageDataByte(encoded, segments, compressedLength - 1);

        try
        {
            using var compressed = new SegmentedPngDataStream(encoded, segments);
            using var decompressor = new ZLibStream(compressed, CompressionMode.Decompress, leaveOpen: true);
            byte[] buffer = new byte[16 * 1024];
            long decodedPosition = 0;
            ulong adlerS1 = 1;
            ulong adlerS2 = 0;
            int read;
            while ((read = decompressor.Read(buffer, 0, buffer.Length)) != 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (decodedPosition + read > expectedDecodedBytes)
                {
                    return InvalidBackendOutput("The PNG expands beyond its declared scanline dimensions.");
                }

                for (var index = 0; index < read; index++)
                {
                    if ((decodedPosition + index) % rowStride == 0 && buffer[index] > 4)
                    {
                        return InvalidBackendOutput("The PNG contains an invalid scanline filter byte.");
                    }

                    adlerS1 += buffer[index];
                    adlerS2 += adlerS1;
                }

                adlerS1 %= 65_521;
                adlerS2 %= 65_521;
                decodedPosition += read;
            }

            if (decodedPosition != expectedDecodedBytes)
            {
                return InvalidBackendOutput(
                    $"The PNG expands to {decodedPosition} bytes instead of the expected " +
                    $"{expectedDecodedBytes} bytes.");
            }

            uint computedAdler32 = (uint)(adlerS2 << 16 | adlerS1);
            if (computedAdler32 != expectedAdler32)
            {
                return InvalidBackendOutput("The PNG IDAT zlib checksum does not match decoded pixels.");
            }

            if (compressed.Position != compressed.Length)
            {
                return InvalidBackendOutput("The PNG IDAT stream contains trailing compressed data.");
            }

            return null;
        }
        catch (InvalidDataException exception)
        {
            return InvalidBackendOutput($"The PNG IDAT stream is invalid: {exception.Message}");
        }
        catch (IOException exception)
        {
            return InvalidBackendOutput($"The PNG IDAT stream could not be read: {exception.Message}");
        }
    }

    private static byte GetPngImageDataByte(
        ReadOnlyMemory<byte> encoded,
        IReadOnlyList<PngDataSegment> segments,
        long logicalOffset)
    {
        long remaining = logicalOffset;
        foreach (PngDataSegment segment in segments)
        {
            if (remaining < segment.Length)
            {
                return encoded.Span[checked(segment.Offset + (int)remaining)];
            }

            remaining -= segment.Length;
        }

        throw new InvalidDataException("The PNG IDAT logical offset is outside the segment set.");
    }

    private static bool IsValidPngChunkType(ReadOnlySpan<byte> type)
    {
        if (type.Length != 4 || (type[2] & 0x20) != 0)
        {
            return false;
        }

        foreach (byte value in type)
        {
            if (value is not (>= (byte)'A' and <= (byte)'Z') and
                not (>= (byte)'a' and <= (byte)'z'))
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryGetPngChannelCount(int bitDepth, int colorType, out int channelCount)
    {
        channelCount = colorType switch
        {
            0 => 1,
            2 => 3,
            3 => 1,
            4 => 2,
            6 => 4,
            _ => 0,
        };

        return colorType switch
        {
            0 => bitDepth is 1 or 2 or 4 or 8 or 16,
            2 => bitDepth is 8 or 16,
            3 => bitDepth is 1 or 2 or 4 or 8,
            4 => bitDepth is 8 or 16,
            6 => bitDepth is 8 or 16,
            _ => false,
        };
    }

    private static bool ValidateTransparencyChunk(
        int colorType,
        int bitDepth,
        bool seenPalette,
        int paletteEntries,
        ReadOnlySpan<byte> data) =>
        colorType switch
        {
            0 => data.Length == 2 && IsValidTransparentSample(data, bitDepth),
            2 => data.Length == 6 &&
                IsValidTransparentSample(data[..2], bitDepth) &&
                IsValidTransparentSample(data.Slice(2, 2), bitDepth) &&
                IsValidTransparentSample(data.Slice(4, 2), bitDepth),
            3 => seenPalette && data.Length > 0 && data.Length <= paletteEntries,
            _ => false,
        };

    private static bool IsValidTransparentSample(ReadOnlySpan<byte> data, int bitDepth)
    {
        ushort value = System.Buffers.Binary.BinaryPrimitives.ReadUInt16BigEndian(data);
        return bitDepth == 16 || value < 1 << bitDepth;
    }

    private static bool ValidateSignificantBits(
        int colorType,
        int bitDepth,
        ReadOnlySpan<byte> data)
    {
        int expectedLength = colorType switch
        {
            0 => 1,
            2 => 3,
            3 => 3,
            4 => 2,
            6 => 4,
            _ => 0,
        };
        int maximum = colorType == 3 ? 8 : bitDepth;
        if (data.Length != expectedLength)
        {
            return false;
        }

        foreach (byte value in data)
        {
            if (value == 0 || value > maximum)
            {
                return false;
            }
        }

        return true;
    }

    private static uint ComputePngCrc(
        ReadOnlySpan<byte> data,
        CancellationToken cancellationToken)
    {
        uint crc = uint.MaxValue;
        for (var index = 0; index < data.Length; index++)
        {
            if ((index & 0xFFFF) == 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }

            crc = PngCrcTable.Values[(crc ^ data[index]) & 0xFF] ^ (crc >> 8);
        }

        return crc ^ uint.MaxValue;
    }

    private static void ValidateSafetyLimits(PdfPageRenderSafetyLimits limits)
    {
        if (limits.MaximumWidth < 1 || limits.MaximumHeight < 1 || limits.MaximumPixelCount < 1 ||
            limits.MaximumEncodedBytes < 1 || limits.MaximumEncodedBytes > int.MaxValue ||
            limits.MaximumDecodedBytes < 1 || limits.MaximumChunkBytes < 1 ||
            limits.MaximumChunkCount < 3)
        {
            throw new ArgumentOutOfRangeException(
                nameof(limits),
                "PNG safety limits must be positive, bounded, and allow required chunks.");
        }
    }

    private static bool IsSha256(string? value) =>
        value is { Length: 64 } && value.All(Uri.IsHexDigit);

    private static PdfPageRenderResult CreateSuccess(
        CachedPage cached,
        PdfPageRenderCacheDisposition disposition)
    {
        PdfPageRenderMetadata metadata = cached.Metadata with { CacheDisposition = disposition };
        return new PdfPageRenderResult(
            disposition is PdfPageRenderCacheDisposition.Miss
                ? PdfPageRenderStatus.Succeeded
                : PdfPageRenderStatus.CacheHit,
            ClonePage(cached.Page),
            metadata,
            null);
    }

    private static PdfPageRenderResult AsCoalesced(PdfPageRenderResult result)
    {
        if (result.Page is null || result.Metadata is null)
        {
            return CloneResult(result);
        }

        return new PdfPageRenderResult(
            PdfPageRenderStatus.CacheHit,
            ClonePage(result.Page),
            result.Metadata with { CacheDisposition = PdfPageRenderCacheDisposition.Coalesced },
            null);
    }

    private static PdfPageRenderResult CloneResult(PdfPageRenderResult result) =>
        result with { Page = result.Page is null ? null : ClonePage(result.Page) };

    private static PdfRenderedPage ClonePage(PdfRenderedPage page) =>
        new(new ImmutableByteBuffer(page.PngBytes.ToArray()), page.Width, page.Height, page.MediaType);

    private static PdfPageRenderResult Failed(PdfFailure failure) =>
        new(PdfPageRenderStatus.Failed, null, null, failure);

    private static PdfPageRenderResult Cancelled(string technicalMessage) =>
        new(
            PdfPageRenderStatus.Cancelled,
            null,
            null,
            new PdfFailure(
                PdfPageRenderFailureCodes.Cancelled,
                PdfFailureSeverity.Warning,
                "Errors.PdfRenderCancelled",
                technicalMessage,
                Recoverable: true,
                "retry"));

    private static PdfFailure InvalidRequest(string technicalMessage) =>
        Error(
            PdfPageRenderFailureCodes.InvalidRequest,
            "Errors.PdfRenderInvalidRequest",
            technicalMessage,
            recoverable: true,
            "correct_input");

    private static PdfFailure InvalidProvenance(string technicalMessage) =>
        Error(
            PdfPageRenderFailureCodes.BackendProvenanceRejected,
            "Errors.PdfRenderBackendNotReviewed",
            technicalMessage,
            recoverable: false,
            "install_reviewed_renderer");

    private static PdfFailure MissingCapability(string technicalMessage) =>
        Error(
            PdfPageRenderFailureCodes.BackendCapabilityMissing,
            "Errors.PdfRenderBackendUnsupported",
            technicalMessage,
            recoverable: false,
            "install_supported_renderer");

    private static PdfFailure InvalidBackendOutput(string technicalMessage) =>
        Error(
            PdfPageRenderFailureCodes.BackendOutputInvalid,
            "Errors.PdfRenderOutputInvalid",
            technicalMessage,
            recoverable: true,
            "retry");

    private static PdfFailure Error(
        string code,
        string userMessageKey,
        string technicalMessage,
        bool recoverable,
        string suggestedAction) =>
        new(
            code,
            PdfFailureSeverity.Error,
            userMessageKey,
            technicalMessage,
            recoverable,
            suggestedAction);

    private sealed record CachedPage(PdfRenderedPage Page, PdfPageRenderMetadata Metadata);

    private readonly record struct PngDataSegment(int Offset, int Length);

    private static class PngChunkTypes
    {
        public const uint Header = 0x49484452;
        public const uint Palette = 0x504C5445;
        public const uint ImageData = 0x49444154;
        public const uint End = 0x49454E44;
        public const uint Transparency = 0x74524E53;
        public const uint Chromaticity = 0x6348524D;
        public const uint Gamma = 0x67414D41;
        public const uint StandardRgb = 0x73524742;
        public const uint SignificantBits = 0x73424954;
        public const uint PhysicalDimensions = 0x70485973;
    }

    private static class PngCrcTable
    {
        public static readonly uint[] Values = Create();

        private static uint[] Create()
        {
            var values = new uint[256];
            for (uint index = 0; index < values.Length; index++)
            {
                uint value = index;
                for (var bit = 0; bit < 8; bit++)
                {
                    value = (value & 1) == 0
                        ? value >> 1
                        : 0xEDB88320U ^ (value >> 1);
                }

                values[index] = value;
            }

            return values;
        }
    }

    private sealed class SegmentedPngDataStream : Stream
    {
        private readonly ReadOnlyMemory<byte> _source;
        private readonly IReadOnlyList<PngDataSegment> _segments;
        private readonly long _length;
        private int _segmentIndex;
        private int _segmentOffset;
        private long _position;

        public SegmentedPngDataStream(
            ReadOnlyMemory<byte> source,
            IReadOnlyList<PngDataSegment> segments)
        {
            _source = source;
            _segments = segments;
            _length = segments.Aggregate(
                0L,
                static (total, segment) => checked(total + segment.Length));
        }

        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => _length;

        public override long Position
        {
            get => _position;
            set => throw new NotSupportedException();
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            ArgumentNullException.ThrowIfNull(buffer);
            return Read(buffer.AsSpan(offset, count));
        }

        public override int Read(Span<byte> buffer)
        {
            var written = 0;
            while (!buffer.IsEmpty && _segmentIndex < _segments.Count)
            {
                PngDataSegment segment = _segments[_segmentIndex];
                int available = segment.Length - _segmentOffset;
                if (available == 0)
                {
                    _segmentIndex++;
                    _segmentOffset = 0;
                    continue;
                }

                int count = Math.Min(available, buffer.Length);
                _source.Span.Slice(segment.Offset + _segmentOffset, count).CopyTo(buffer);
                buffer = buffer[count..];
                written += count;
                _segmentOffset += count;
                _position += count;
            }

            return written;
        }

        public override void Flush()
        {
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private sealed class BoundedPageRenderCache
    {
        private readonly object _sync = new();
        private readonly int _maximumEntries;
        private readonly long _maximumEncodedBytes;
        private readonly Dictionary<PdfPageRenderCacheKey, CacheNode> _entries = new();
        private readonly LinkedList<PdfPageRenderCacheKey> _leastRecentlyUsed = new();
        private long _encodedBytes;

        public BoundedPageRenderCache(PdfPageRenderCacheOptions options)
        {
            if (options.MaximumEntries < 1)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(options),
                    "The page-render cache must allow at least one entry.");
            }

            if (options.MaximumEncodedBytes < 1)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(options),
                    "The page-render cache byte limit must be positive.");
            }

            _maximumEntries = options.MaximumEntries;
            _maximumEncodedBytes = options.MaximumEncodedBytes;
        }

        public int Count
        {
            get
            {
                lock (_sync)
                {
                    return _entries.Count;
                }
            }
        }

        public long EncodedBytes
        {
            get
            {
                lock (_sync)
                {
                    return _encodedBytes;
                }
            }
        }

        public bool TryGet(PdfPageRenderCacheKey key, out CachedPage? page)
        {
            lock (_sync)
            {
                if (!_entries.TryGetValue(key, out CacheNode? node))
                {
                    page = null;
                    return false;
                }

                _leastRecentlyUsed.Remove(node.RecencyNode);
                _leastRecentlyUsed.AddLast(node.RecencyNode);
                page = CloneCachedPage(node.Page);
                return true;
            }
        }

        public void Store(PdfPageRenderCacheKey key, CachedPage page)
        {
            long byteCount = page.Page.PngBytes.Length;
            if (byteCount > _maximumEncodedBytes)
            {
                return;
            }

            lock (_sync)
            {
                if (_entries.TryGetValue(key, out CacheNode? existing))
                {
                    _encodedBytes -= existing.Page.Page.PngBytes.Length;
                    _leastRecentlyUsed.Remove(existing.RecencyNode);
                    _entries.Remove(key);
                }

                while (_entries.Count >= _maximumEntries ||
                       _encodedBytes + byteCount > _maximumEncodedBytes)
                {
                    LinkedListNode<PdfPageRenderCacheKey>? oldest = _leastRecentlyUsed.First;
                    if (oldest is null || !_entries.Remove(oldest.Value, out CacheNode? removed))
                    {
                        break;
                    }

                    _leastRecentlyUsed.RemoveFirst();
                    _encodedBytes -= removed.Page.Page.PngBytes.Length;
                }

                CachedPage stored = CloneCachedPage(page);
                LinkedListNode<PdfPageRenderCacheKey> recencyNode = _leastRecentlyUsed.AddLast(key);
                _entries.Add(key, new CacheNode(stored, recencyNode));
                _encodedBytes += byteCount;
            }
        }

        private static CachedPage CloneCachedPage(CachedPage page) =>
            new(ClonePage(page.Page), page.Metadata);

        private sealed record CacheNode(
            CachedPage Page,
            LinkedListNode<PdfPageRenderCacheKey> RecencyNode);
    }

    private sealed class InFlightRender : IDisposable
    {
        private readonly object _sync = new();
        private readonly CancellationTokenSource _cancellation = new();
        private Task<PdfPageRenderResult>? _task;
        private int _attachedCount;
        private int _hasLeader;

        public bool Attach()
        {
            Interlocked.Increment(ref _attachedCount);
            return Interlocked.Exchange(ref _hasLeader, 1) == 0;
        }

        public Task<PdfPageRenderResult> GetOrStart(
            Func<CancellationToken, Task<PdfPageRenderResult>> factory,
            Action completed)
        {
            lock (_sync)
            {
                if (_task is not null)
                {
                    return _task;
                }

                _task = factory(_cancellation.Token);
                _ = _task.ContinueWith(
                    static (_, state) =>
                    {
                        var completionState = ((Action Completed, InFlightRender Owner))state!;
                        completionState.Completed();
                        completionState.Owner.Dispose();
                    },
                    (completed, this),
                    CancellationToken.None,
                    TaskContinuationOptions.ExecuteSynchronously,
                    TaskScheduler.Default);
                return _task;
            }
        }

        public bool Detach() => Interlocked.Decrement(ref _attachedCount) == 0;

        public void Cancel() => _cancellation.Cancel();

        public void Dispose() => _cancellation.Dispose();
    }
}
