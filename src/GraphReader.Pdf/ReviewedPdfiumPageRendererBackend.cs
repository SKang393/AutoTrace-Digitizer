// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Buffers;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace GraphReader.Pdf;

/// <summary>
/// Loads an independently reviewed PDFium runner approval and refuses to use
/// any binary, build manifest, source lock, or notice whose bytes have changed.
/// </summary>
public sealed class ReviewedPdfiumPageRendererBackend : IPdfiumPageRendererBackend
{
    public const string PinnedSource = "https://pdfium.googlesource.com/pdfium";
    public const string PinnedSourceRevision = "2870fa9244b0f0f69fb743fab1e08deefcb07b2b";
    public const string RequiredProfileId = "graphreader-pdfium-minimal-win-x64";

    private static readonly PdfiumBackendCapabilities BackendCapabilities = new(
        SupportsPageRendering: true,
        SupportsCancellation: true,
        SupportsPngEncoding: true,
        IsLocalOnly: true,
        MinimumDpi: 36,
        MaximumDpi: 1_200);

    private readonly ReviewedPdfiumApproval _approval;
    private readonly IPdfiumRunnerProcess _runner;

    private ReviewedPdfiumPageRendererBackend(
        ReviewedPdfiumApproval approval,
        IPdfiumRunnerProcess runner)
    {
        _approval = approval;
        _runner = runner;
        Provenance = new PdfiumBackendProvenance(
            approval.RendererId,
            approval.RendererVersion,
            approval.BinarySha256,
            approval.Source,
            approval.SourceRevision,
            approval.LicenseSpdx,
            approval.NoticePath,
            ReviewApproved: true,
            RedistributionApproved: true,
            IsBundled: true);
    }

    public PdfiumBackendProvenance Provenance { get; }

    public PdfiumBackendCapabilities Capabilities => BackendCapabilities;

    public static ReviewedPdfiumPageRendererBackend Load(string approvalPath)
    {
        ReviewedPdfiumApproval approval = ReviewedPdfiumApprovalLoader.Load(approvalPath);
        return new ReviewedPdfiumPageRendererBackend(approval, new LocalPdfiumRunnerProcess());
    }

    internal static ReviewedPdfiumPageRendererBackend Load(
        string approvalPath,
        IPdfiumRunnerProcess runner)
    {
        ArgumentNullException.ThrowIfNull(runner);
        ReviewedPdfiumApproval approval = ReviewedPdfiumApprovalLoader.Load(approvalPath);
        return new ReviewedPdfiumPageRendererBackend(approval, runner);
    }

    public IPdfPageRenderingService CreateRenderingService(
        PdfPageRenderCacheOptions? cacheOptions = null,
        PdfPageRenderSafetyLimits? safetyLimits = null)
    {
        var independentApproval = new PdfiumBackendApproval(
            _approval.RendererId,
            _approval.RendererVersion,
            _approval.BinarySha256,
            _approval.Source,
            _approval.SourceRevision,
            _approval.LicenseSpdx,
            _approval.NoticePath,
            BundlingApproved: true);
        return new PdfiumPageRendererAdapter(
            this,
            new ExactPdfiumBackendApprovalPolicy(independentApproval),
            cacheOptions,
            safetyLimits);
    }

    public async Task<PdfiumBackendRenderResult> RenderPageAsync(
        PdfiumBackendRenderRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();

        try
        {
            ReviewedPdfiumApprovalLoader.VerifyRetainedInputs(_approval);
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            return PdfiumBackendRenderResult.Failed(new PdfiumBackendFailure(
                "PDFIUM_APPROVAL_INPUT_CHANGED",
                exception.Message,
                Recoverable: false,
                "restore_reviewed_pdfium_artifacts"));
        }

        try
        {
            PdfiumRawPage rawPage = await _runner.RenderAsync(
                _approval.BinaryPath,
                request.PdfBytes,
                request.PageNumber,
                request.Dpi,
                cancellationToken).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();

            byte[] png = PdfiumPngEncoder.Encode(rawPage, cancellationToken);
            return PdfiumBackendRenderResult.Success(png, rawPage.Width, rawPage.Height);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            return PdfiumBackendRenderResult.Failed(new PdfiumBackendFailure(
                "PDFIUM_RUNNER_FAILED",
                exception.Message,
                Recoverable: true,
                "inspect_pdfium_diagnostics"));
        }
    }
}

internal sealed record ReviewedPdfiumApproval(
    string ApprovalPath,
    string RendererId,
    string RendererVersion,
    string BinaryPath,
    string BinarySha256,
    string Source,
    string SourceRevision,
    string SourceLockPath,
    string SourceLockSha256,
    string BuildManifestPath,
    string BuildManifestSha256,
    string LicenseSpdx,
    string NoticePath,
    string NoticeSha256);

internal static class ReviewedPdfiumApprovalLoader
{
    private static readonly string[] ApprovalFields =
    [
        "schemaVersion", "rendererId", "rendererVersion", "binaryPath", "binarySha256",
        "source", "sourceRevision", "sourceLockPath", "sourceLockSha256",
        "buildManifestPath", "buildManifestSha256", "licenseSpdx", "noticePath",
        "noticeSha256", "reviewApproved", "redistributionApproved", "bundlingApproved",
    ];

    private static readonly string[] BuildManifestFields =
    [
        "schemaVersion", "profileId", "generatedUtc", "reviewStatus", "source",
        "sourceRevision", "sourceLockSha256", "argsGnSha256", "overlayBuildSha256",
        "overlayRootTargetSha256", "overlaySourceSha256", "compatibilityPatchSha256",
        "targetDependenciesSha256", "peImportsSha256", "binarySha256", "features", "warning",
    ];

    private static readonly string[] SourceLockFields =
    [
        "schemaVersion", "profileId", "compatibilityPatchSha256", "sources", "target", "toolchain",
    ];

    public static ReviewedPdfiumApproval Load(string approvalPath)
    {
        if (string.IsNullOrWhiteSpace(approvalPath))
        {
            throw new ArgumentException("A reviewed PDFium approval path is required.", nameof(approvalPath));
        }

        string fullApprovalPath = Path.GetFullPath(approvalPath);
        if (!File.Exists(fullApprovalPath))
        {
            throw new FileNotFoundException("The reviewed PDFium approval file is missing.", fullApprovalPath);
        }

        EnsureNoReparsePoints(fullApprovalPath, "PDFium approval");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(fullApprovalPath, Encoding.UTF8));
        JsonElement root = document.RootElement;
        RequireExactObject(root, "reviewed approval", ApprovalFields);
        RequireInteger(root, "schemaVersion", 1);
        RequireBoolean(root, "reviewApproved", true);
        RequireBoolean(root, "redistributionApproved", true);
        RequireBoolean(root, "bundlingApproved", true);

        string baseDirectory = Path.GetDirectoryName(fullApprovalPath)!;
        var approval = new ReviewedPdfiumApproval(
            fullApprovalPath,
            RequireText(root, "rendererId"),
            RequireText(root, "rendererVersion"),
            ResolvePath(baseDirectory, RequireText(root, "binaryPath")),
            RequireSha256(root, "binarySha256"),
            RequireText(root, "source"),
            RequireText(root, "sourceRevision"),
            ResolvePath(baseDirectory, RequireText(root, "sourceLockPath")),
            RequireSha256(root, "sourceLockSha256"),
            ResolvePath(baseDirectory, RequireText(root, "buildManifestPath")),
            RequireSha256(root, "buildManifestSha256"),
            RequireText(root, "licenseSpdx"),
            ResolvePath(baseDirectory, RequireText(root, "noticePath")),
            RequireSha256(root, "noticeSha256"));

        if (!string.Equals(approval.Source, ReviewedPdfiumPageRendererBackend.PinnedSource, StringComparison.Ordinal) ||
            !string.Equals(approval.SourceRevision, ReviewedPdfiumPageRendererBackend.PinnedSourceRevision, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(approval.RendererVersion, ReviewedPdfiumPageRendererBackend.PinnedSourceRevision, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"The approval must identify PDFium revision {ReviewedPdfiumPageRendererBackend.PinnedSourceRevision} from {ReviewedPdfiumPageRendererBackend.PinnedSource}.");
        }

        if (!string.Equals(approval.LicenseSpdx, "BSD-3-Clause", StringComparison.Ordinal))
        {
            throw new InvalidDataException("The reviewed PDFium approval must declare BSD-3-Clause.");
        }

        VerifyRetainedInputs(approval);
        ValidateSourceLock(approval.SourceLockPath);
        ValidateBuildManifest(approval);
        ValidateReviewedNotice(approval.NoticePath);
        return approval;
    }

    public static void VerifyRetainedInputs(ReviewedPdfiumApproval approval)
    {
        VerifyFile(approval.BinaryPath, approval.BinarySha256, "PDFium runner binary");
        VerifyFile(approval.SourceLockPath, approval.SourceLockSha256, "PDFium source lock");
        VerifyFile(approval.BuildManifestPath, approval.BuildManifestSha256, "PDFium build manifest");
        VerifyFile(approval.NoticePath, approval.NoticeSha256, "PDFium reviewed notice");
    }

    private static void ValidateBuildManifest(ReviewedPdfiumApproval approval)
    {
        EnsureNoReparsePoints(approval.BuildManifestPath, "PDFium build manifest");
        using JsonDocument document = JsonDocument.Parse(
            File.ReadAllText(approval.BuildManifestPath, Encoding.UTF8));
        JsonElement root = document.RootElement;
        RequireExactObject(root, "build manifest", BuildManifestFields);
        RequireInteger(root, "schemaVersion", 1);
        RequireText(root, "generatedUtc");
        RequireText(root, "reviewStatus");
        if (!string.Equals(RequireText(root, "source"), ReviewedPdfiumPageRendererBackend.PinnedSource, StringComparison.Ordinal))
        {
            throw new InvalidDataException("The PDFium build manifest does not identify the pinned official source.");
        }

        foreach (string digestField in new[]
                 {
                     "argsGnSha256", "overlayBuildSha256", "overlayRootTargetSha256",
                     "overlaySourceSha256", "compatibilityPatchSha256",
                     "targetDependenciesSha256", "peImportsSha256",
                 })
        {
            RequireSha256(root, digestField);
        }

        RequireText(root, "warning");
        if (!string.Equals(RequireText(root, "profileId"), ReviewedPdfiumPageRendererBackend.RequiredProfileId, StringComparison.Ordinal) ||
            !string.Equals(RequireText(root, "sourceRevision"), ReviewedPdfiumPageRendererBackend.PinnedSourceRevision, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(RequireSha256(root, "binarySha256"), approval.BinarySha256, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(RequireSha256(root, "sourceLockSha256"), approval.SourceLockSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("The PDFium build manifest does not match the approved pinned artifact.");
        }

        JsonElement features = RequireObjectProperty(root, "features", "build manifest");
        RequireExactObject(features, "build manifest features", ["v8", "xfa", "skia", "icuDataFile"]);
        RequireBoolean(features, "v8", false);
        RequireBoolean(features, "xfa", false);
        RequireBoolean(features, "skia", false);
        RequireBoolean(features, "icuDataFile", false);
    }

    private static void ValidateSourceLock(string sourceLockPath)
    {
        EnsureNoReparsePoints(sourceLockPath, "PDFium source lock");
        using JsonDocument document = JsonDocument.Parse(
            File.ReadAllText(sourceLockPath, Encoding.UTF8));
        JsonElement root = document.RootElement;
        RequireExactObject(root, "source lock", SourceLockFields);
        RequireInteger(root, "schemaVersion", 1);
        RequireSha256(root, "compatibilityPatchSha256");
        if (!string.Equals(
                RequireText(root, "profileId"),
                ReviewedPdfiumPageRendererBackend.RequiredProfileId,
                StringComparison.Ordinal))
        {
            throw new InvalidDataException("The PDFium source lock has an unexpected profile ID.");
        }

        JsonElement sources = RequireObjectProperty(root, "sources", "source lock");
        RequireExactObject(sources, "source lock sources", ["pdfium", "depotTools"]);
        JsonElement pdfium = RequireObjectProperty(sources, "pdfium", "source lock sources");
        RequireObjectProperty(sources, "depotTools", "source lock sources");
        RequireExactObject(
            pdfium,
            "source lock sources/pdfium",
            ["repository", "revision", "rootBuildGnBlob", "renderDeviceHeaderBlob", "license"]);
        if (!string.Equals(
                RequireText(pdfium, "repository"),
                ReviewedPdfiumPageRendererBackend.PinnedSource,
                StringComparison.Ordinal) ||
            !string.Equals(
                RequireText(pdfium, "revision"),
                ReviewedPdfiumPageRendererBackend.PinnedSourceRevision,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("The PDFium source lock does not contain the pinned official source.");
        }
        RequireHexDigest(pdfium, "rootBuildGnBlob", 40, "Git blob");
        RequireHexDigest(pdfium, "renderDeviceHeaderBlob", 40, "Git blob");
        if (!string.Equals(RequireText(pdfium, "license"), "BSD-3-Clause", StringComparison.Ordinal))
        {
            throw new InvalidDataException("The PDFium source lock must declare BSD-3-Clause for PDFium.");
        }

        JsonElement target = RequireObjectProperty(root, "target", "source lock");
        RequireExactObject(
            target,
            "source lock target",
            [
                "os", "cpu", "configuration", "binaryName", "maxParallelCompileJobs",
                "v8", "xfa", "skia", "fontations", "partitionAlloc", "icuDataFile",
            ]);
        if (!string.Equals(RequireText(target, "os"), "win", StringComparison.Ordinal) ||
            !string.Equals(RequireText(target, "cpu"), "x64", StringComparison.Ordinal))
        {
            throw new InvalidDataException("The PDFium source lock must target Windows x64.");
        }
        if (!string.Equals(RequireText(target, "configuration"), "Release", StringComparison.Ordinal) ||
            !string.Equals(RequireText(target, "binaryName"), "graphreader_pdfium_renderer.exe", StringComparison.Ordinal))
        {
            throw new InvalidDataException("The PDFium source lock has an unexpected binary target configuration.");
        }

        RequireInteger(target, "maxParallelCompileJobs", 4);

        RequireBoolean(target, "v8", false);
        RequireBoolean(target, "xfa", false);
        RequireBoolean(target, "skia", false);
        RequireBoolean(target, "fontations", false);
        RequireBoolean(target, "partitionAlloc", false);
        RequireBoolean(target, "icuDataFile", false);
        RequireObjectProperty(root, "toolchain", "source lock");
    }

    private static void ValidateReviewedNotice(string noticePath)
    {
        EnsureNoReparsePoints(noticePath, "PDFium reviewed notice");
        using var reader = new StreamReader(noticePath, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        string? firstLine = reader.ReadLine();
        if (!string.Equals(firstLine?.Trim(), "REVIEW STATUS: COMPLETE", StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                "The PDFium notice is not independently marked REVIEW STATUS: COMPLETE.");
        }
    }

    private static void VerifyFile(string path, string expectedSha256, string label)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"The approved {label} is missing.", path);
        }

        EnsureNoReparsePoints(path, label);
        string actual = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
        if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"The approved {label} SHA-256 is {actual}, expected {expectedSha256}.");
        }
    }

    private static string ResolvePath(string baseDirectory, string configuredPath)
    {
        if (Path.IsPathRooted(configuredPath))
        {
            throw new InvalidDataException("PDFium approval resource paths must be relative to the approval directory.");
        }

        string fullBaseDirectory = Path.GetFullPath(baseDirectory);
        string fullPath = Path.GetFullPath(Path.Combine(fullBaseDirectory, configuredPath));
        string containedPrefix = Path.TrimEndingDirectorySeparator(fullBaseDirectory) + Path.DirectorySeparatorChar;
        if (!fullPath.StartsWith(containedPrefix, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("PDFium approval resource paths must remain under the approval directory.");
        }

        return fullPath;
    }

    private static void EnsureNoReparsePoints(string path, string label)
    {
        string current = Path.GetFullPath(path);
        while (true)
        {
            FileAttributes attributes = File.GetAttributes(current);
            if ((attributes & FileAttributes.ReparsePoint) != 0)
            {
                throw new InvalidDataException($"The approved {label} path contains a reparse point: {current}");
            }

            string? parent = Path.GetDirectoryName(current);
            if (string.IsNullOrEmpty(parent) || string.Equals(parent, current, StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            current = parent;
        }
    }

    private static string RequireText(JsonElement root, string propertyName)
    {
        if (!root.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.String ||
            string.IsNullOrWhiteSpace(value.GetString()))
        {
            throw new InvalidDataException($"PDFium approval field '{propertyName}' is required.");
        }

        return value.GetString()!.Trim();
    }

    private static JsonElement RequireObjectProperty(
        JsonElement root,
        string propertyName,
        string parentContext)
    {
        if (!root.TryGetProperty(propertyName, out JsonElement value) || value.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException(
                $"PDFium {parentContext} field '{propertyName}' must be a JSON object.");
        }

        return value;
    }

    private static void RequireExactObject(
        JsonElement value,
        string context,
        IReadOnlyCollection<string> expectedProperties)
    {
        if (value.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"PDFium {context} must be a JSON object.");
        }

        var expected = new HashSet<string>(expectedProperties, StringComparer.Ordinal);
        var observed = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonProperty property in value.EnumerateObject())
        {
            if (!observed.Add(property.Name))
            {
                throw new InvalidDataException(
                    $"PDFium {context} contains duplicate field '{property.Name}'.");
            }

            if (!expected.Contains(property.Name))
            {
                throw new InvalidDataException(
                    $"PDFium {context} contains unexpected field '{property.Name}'.");
            }
        }

        foreach (string propertyName in expected)
        {
            if (!observed.Contains(propertyName))
            {
                throw new InvalidDataException(
                    $"PDFium {context} is missing field '{propertyName}'.");
            }
        }
    }

    private static string RequireSha256(JsonElement root, string propertyName)
    {
        return RequireHexDigest(root, propertyName, 64, "SHA-256");
    }

    private static string RequireHexDigest(
        JsonElement root,
        string propertyName,
        int expectedLength,
        string digestName)
    {
        string value = RequireText(root, propertyName).ToLowerInvariant();
        if (value.Length != expectedLength || value.Any(static character => !Uri.IsHexDigit(character)))
        {
            throw new InvalidDataException(
                $"PDFium field '{propertyName}' must be a {digestName} digest.");
        }

        return value;
    }

    private static void RequireInteger(JsonElement root, string propertyName, int expected)
    {
        if (!root.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Number ||
            !value.TryGetInt32(out int actual) ||
            actual != expected)
        {
            throw new InvalidDataException($"PDFium field '{propertyName}' must equal {expected}.");
        }
    }

    private static void RequireBoolean(JsonElement root, string propertyName, bool expected)
    {
        if (!root.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind is not (JsonValueKind.True or JsonValueKind.False) ||
            value.GetBoolean() != expected)
        {
            throw new InvalidDataException($"PDFium field '{propertyName}' must equal {expected.ToString(CultureInfo.InvariantCulture).ToLowerInvariant()}.");
        }
    }
}

internal sealed record PdfiumRawPage(ImmutableByteBuffer BgraBytes, int Width, int Height, int Stride);

internal interface IPdfiumRunnerProcess
{
    Task<PdfiumRawPage> RenderAsync(
        string binaryPath,
        ImmutableByteBuffer pdfBytes,
        int pageNumber,
        int dpi,
        CancellationToken cancellationToken);
}

internal sealed class LocalPdfiumRunnerProcess : IPdfiumRunnerProcess
{
    private static readonly byte[] RawMagic = "GRPDF01\0"u8.ToArray();
    private const long MaximumRawBytes = PdfiumPngEncoder.MaximumPixelBytes;
    private static readonly TimeSpan ProcessExitTimeout = TimeSpan.FromSeconds(2);
    private static readonly TimeSpan CleanupRetryDelay = TimeSpan.FromMilliseconds(50);
    private const int CleanupAttempts = 6;

    public async Task<PdfiumRawPage> RenderAsync(
        string binaryPath,
        ImmutableByteBuffer pdfBytes,
        int pageNumber,
        int dpi,
        CancellationToken cancellationToken)
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "GraphReader", "Pdfium", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        string rawPath = Path.Combine(tempRoot, "page.raw");

        try
        {
            using var process = new Process
            {
                StartInfo = CreateStartInfo(binaryPath, rawPath, pageNumber, dpi),
                EnableRaisingEvents = true,
            };
            if (!process.Start())
            {
                throw new IOException("The reviewed PDFium runner did not start.");
            }

            Task<string> standardOutput = process.StandardOutput.ReadToEndAsync(cancellationToken);
            Task<string> standardError = process.StandardError.ReadToEndAsync(cancellationToken);
            try
            {
                await process.StandardInput.BaseStream.WriteAsync(pdfBytes.Memory, cancellationToken).ConfigureAwait(false);
                await process.StandardInput.BaseStream.FlushAsync(cancellationToken).ConfigureAwait(false);
                process.StandardInput.Close();
                await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                TryKill(process);
                await WaitForExitAfterKillAsync(process).ConfigureAwait(false);
                throw;
            }
            catch
            {
                TryKill(process);
                await WaitForExitAfterKillAsync(process).ConfigureAwait(false);
                throw;
            }

            string stdout = await standardOutput.ConfigureAwait(false);
            string stderr = await standardError.ConfigureAwait(false);
            if (process.ExitCode != 0)
            {
                throw new InvalidDataException(
                    $"The reviewed PDFium runner exited with code {process.ExitCode}. stderr: {Limit(stderr)} stdout: {Limit(stdout)}");
            }

            return await ReadRawPageAsync(rawPath, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            await DeleteDirectoryWithRetriesAsync(tempRoot).ConfigureAwait(false);
        }
    }

    private static ProcessStartInfo CreateStartInfo(
        string binaryPath,
        string rawPath,
        int pageNumber,
        int dpi)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = binaryPath,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(binaryPath)!,
        };
        startInfo.ArgumentList.Add("--output");
        startInfo.ArgumentList.Add(rawPath);
        startInfo.ArgumentList.Add("--page");
        startInfo.ArgumentList.Add((pageNumber - 1).ToString(CultureInfo.InvariantCulture));
        startInfo.ArgumentList.Add("--dpi");
        startInfo.ArgumentList.Add(dpi.ToString(CultureInfo.InvariantCulture));
        return startInfo;
    }

    private static async Task<PdfiumRawPage> ReadRawPageAsync(
        string rawPath,
        CancellationToken cancellationToken)
    {
        if (!File.Exists(rawPath))
        {
            throw new InvalidDataException("The reviewed PDFium runner did not produce its raw page output.");
        }

        await using FileStream stream = new(
            rawPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 64 * 1024,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        if (stream.Length < 28 || stream.Length > MaximumRawBytes + 28)
        {
            throw new InvalidDataException("The reviewed PDFium runner produced an invalid raw page size.");
        }

        byte[] header = new byte[28];
        await stream.ReadExactlyAsync(header, cancellationToken).ConfigureAwait(false);
        if (!header.AsSpan(0, 8).SequenceEqual(RawMagic))
        {
            throw new InvalidDataException("The reviewed PDFium runner output has an invalid header.");
        }

        int width = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(8, 4));
        int height = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(12, 4));
        int stride = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(16, 4));
        long payloadLength = BinaryPrimitives.ReadInt64LittleEndian(header.AsSpan(20, 8));
        long expectedLength;
        try
        {
            expectedLength = checked((long)stride * height);
        }
        catch (OverflowException exception)
        {
            throw new InvalidDataException("The reviewed PDFium runner output dimensions overflow.", exception);
        }

        if (width < 1 || height < 1 || width > 32_768 || height > 32_768 ||
            stride < checked(width * 4) || payloadLength != expectedLength ||
            payloadLength > MaximumRawBytes || stream.Length != payloadLength + header.Length)
        {
            throw new InvalidDataException("The reviewed PDFium runner output dimensions are inconsistent.");
        }

        byte[] pixels = GC.AllocateUninitializedArray<byte>(checked((int)payloadLength));
        await stream.ReadExactlyAsync(pixels, cancellationToken).ConfigureAwait(false);
        return new PdfiumRawPage(new ImmutableByteBuffer(pixels), width, height, stride);
    }

    private static void TryKill(Process process)
    {
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }
        }
        catch (InvalidOperationException)
        {
        }
    }

    private static async Task WaitForExitAfterKillAsync(Process process)
    {
        try
        {
            if (process.HasExited)
            {
                return;
            }

            using var timeout = new CancellationTokenSource(ProcessExitTimeout);
            await process.WaitForExitAsync(timeout.Token).ConfigureAwait(false);
        }
        catch (InvalidOperationException)
        {
        }
        catch (OperationCanceledException)
        {
        }
    }

    internal static async Task DeleteDirectoryWithRetriesAsync(string directoryPath)
    {
        for (var attempt = 1; attempt <= CleanupAttempts; attempt++)
        {
            try
            {
                if (Directory.Exists(directoryPath))
                {
                    Directory.Delete(directoryPath, recursive: true);
                }

                return;
            }
            catch (IOException) when (attempt < CleanupAttempts)
            {
            }
            catch (UnauthorizedAccessException) when (attempt < CleanupAttempts)
            {
            }
            catch (IOException)
            {
                return;
            }
            catch (UnauthorizedAccessException)
            {
                return;
            }

            await Task.Delay(CleanupRetryDelay, CancellationToken.None).ConfigureAwait(false);
        }
    }

    private static string Limit(string value) =>
        value.Length <= 2_048 ? value.Trim() : value[..2_048].Trim() + "...";
}

internal static class PdfiumPngEncoder
{
    private static readonly byte[] Signature = [137, 80, 78, 71, 13, 10, 26, 10];
    internal const long MaximumPixelBytes = 160L * 1024L * 1024L;
    private const int IdatChunkSize = 64 * 1024;

    public static byte[] Encode(PdfiumRawPage page, CancellationToken cancellationToken) =>
        Encode(page, scanlineCompleted: null, cancellationToken);

    internal static byte[] Encode(
        PdfiumRawPage page,
        Action<int>? scanlineCompleted,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(page);
        ValidateLayout(page.BgraBytes.Length, page.Width, page.Height, page.Stride);
        cancellationToken.ThrowIfCancellationRequested();
        ReadOnlySpan<byte> bgra = page.BgraBytes.Memory.Span;
        int rowLength = checked(1 + page.Width * 4);

        using var png = new MemoryStream();
        png.Write(Signature);
        Span<byte> header = stackalloc byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(header[..4], checked((uint)page.Width));
        BinaryPrimitives.WriteUInt32BigEndian(header.Slice(4, 4), checked((uint)page.Height));
        header[8] = 8;
        header[9] = 6;
        WriteChunk(png, "IHDR"u8, header);

        byte[] rowBuffer = ArrayPool<byte>.Shared.Rent(rowLength);
        try
        {
            using var idat = new PngIdatChunkStream(png, cancellationToken);
            using (var zlib = new ZLibStream(idat, CompressionLevel.SmallestSize, leaveOpen: true))
            {
                Span<byte> row = rowBuffer.AsSpan(0, rowLength);
                for (var y = 0; y < page.Height; y++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    row[0] = 0;
                    int sourceOffset = checked(y * page.Stride);
                    for (var x = 0; x < page.Width; x++)
                    {
                        int sourcePixel = sourceOffset + x * 4;
                        int targetPixel = 1 + x * 4;
                        row[targetPixel] = bgra[sourcePixel + 2];
                        row[targetPixel + 1] = bgra[sourcePixel + 1];
                        row[targetPixel + 2] = bgra[sourcePixel];
                        row[targetPixel + 3] = bgra[sourcePixel + 3];
                    }

                    zlib.Write(row);
                    scanlineCompleted?.Invoke(y);
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rowBuffer, clearArray: false);
        }

        cancellationToken.ThrowIfCancellationRequested();
        WriteChunk(png, "IEND"u8, []);
        return png.ToArray();
    }

    internal static void ValidateLayout(int pixelBufferLength, int width, int height, int stride)
    {
        long payloadLength;
        try
        {
            payloadLength = checked((long)stride * height);
        }
        catch (OverflowException exception)
        {
            throw new InvalidDataException("The PDFium page dimensions overflow.", exception);
        }

        if (payloadLength > MaximumPixelBytes)
        {
            throw new InvalidDataException(
                $"The PDFium page exceeds the {MaximumPixelBytes / (1024 * 1024)} MiB managed pixel limit.");
        }

        if (width < 1 || height < 1 || width > 32_768 || height > 32_768 ||
            stride < checked(width * 4) || payloadLength != pixelBufferLength)
        {
            throw new InvalidDataException("The PDFium page pixel layout is inconsistent.");
        }
    }

    private static void WriteChunk(Stream output, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(length, checked((uint)data.Length));
        output.Write(length);
        output.Write(type);
        output.Write(data);

        uint crcValue = UpdateCrc32(uint.MaxValue, type);
        crcValue = UpdateCrc32(crcValue, data);
        Span<byte> crc = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(crc, ~crcValue);
        output.Write(crc);
    }

    private static uint UpdateCrc32(uint crc, ReadOnlySpan<byte> data)
    {
        foreach (byte value in data)
        {
            crc ^= value;
            for (var bit = 0; bit < 8; bit++)
            {
                uint mask = unchecked((uint)-(int)(crc & 1));
                crc = (crc >> 1) ^ (0xedb88320u & mask);
            }
        }

        return crc;
    }

    private sealed class PngIdatChunkStream : Stream
    {
        private readonly Stream _output;
        private readonly CancellationToken _cancellationToken;
        private readonly byte[] _buffer = new byte[IdatChunkSize];
        private int _buffered;
        private bool _disposed;

        public PngIdatChunkStream(Stream output, CancellationToken cancellationToken)
        {
            _output = output;
            _cancellationToken = cancellationToken;
        }

        public override bool CanRead => false;

        public override bool CanSeek => false;

        public override bool CanWrite => !_disposed;

        public override long Length => throw new NotSupportedException();

        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override void Flush()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            FlushChunk();
        }

        public override void Write(byte[] buffer, int offset, int count) =>
            Write(buffer.AsSpan(offset, count));

        public override void Write(ReadOnlySpan<byte> buffer)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            while (!buffer.IsEmpty)
            {
                _cancellationToken.ThrowIfCancellationRequested();
                int copied = Math.Min(_buffer.Length - _buffered, buffer.Length);
                buffer[..copied].CopyTo(_buffer.AsSpan(_buffered));
                _buffered += copied;
                buffer = buffer[copied..];
                if (_buffered == _buffer.Length)
                {
                    FlushChunk();
                }
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposed && disposing)
            {
                FlushChunk();
            }

            _disposed = true;
            base.Dispose(disposing);
        }

        private void FlushChunk()
        {
            _cancellationToken.ThrowIfCancellationRequested();
            if (_buffered == 0)
            {
                return;
            }

            WriteChunk(_output, "IDAT"u8, _buffer.AsSpan(0, _buffered));
            _buffered = 0;
        }

        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();
    }
}
