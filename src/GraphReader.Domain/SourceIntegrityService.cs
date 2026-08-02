// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;

using System.Diagnostics.CodeAnalysis;

namespace GraphReader.Domain;

public sealed record SourceIntegrityResult(
    SourceId SourceId,
    string Path,
    string ExpectedSha256,
    string ActualSha256,
    bool Matches);

public sealed class SourceIntegrityService
{
    [SuppressMessage("Performance", "CA1822:Mark members as static", Justification = "The instance service is an injectable integration seam.")]
    public async Task<DomainResult<SourceReference>> CreateReferenceAsync(
        string path,
        SourceKind kind,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string fullPath = Path.GetFullPath(path);

        DomainResult<string> hashResult = await ComputeSha256Async(fullPath, cancellationToken).ConfigureAwait(false);
        if (!hashResult.IsSuccess || hashResult.Value is null)
        {
            return DomainResult<SourceReference>.Failure(hashResult.Errors);
        }

        return DomainResult<SourceReference>.Success(new SourceReference(
            SourceId.New(),
            kind,
            Path.GetFileName(fullPath),
            fullPath,
            hashResult.Value,
            ArticleMetadata: null));
    }

    [SuppressMessage("Performance", "CA1822:Mark members as static", Justification = "The instance service is an injectable integration seam.")]
    public async Task<DomainResult<SourceIntegrityResult>> VerifyAsync(
        SourceReference source,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (string.IsNullOrWhiteSpace(source.LocalPath))
        {
            return DomainResult<SourceIntegrityResult>.Failure(new DomainError(
                "SOURCE_PATH_UNAVAILABLE",
                DomainErrorSeverity.Warning,
                "Errors.SourcePathUnavailable",
                $"Source '{source.SourceId.Value}' has no local path to verify.",
                Recoverable: true,
                "locate_source"));
        }

        string fullPath = Path.GetFullPath(source.LocalPath);
        DomainResult<string> hashResult = await ComputeSha256Async(fullPath, cancellationToken).ConfigureAwait(false);
        if (!hashResult.IsSuccess || hashResult.Value is null)
        {
            return DomainResult<SourceIntegrityResult>.Failure(hashResult.Errors);
        }

        bool matches = string.Equals(source.Sha256, hashResult.Value, StringComparison.OrdinalIgnoreCase);
        if (!matches)
        {
            return DomainResult<SourceIntegrityResult>.Failure(new DomainError(
                "SOURCE_HASH_MISMATCH",
                DomainErrorSeverity.Error,
                "Errors.SourceHashMismatch",
                $"Source '{fullPath}' has SHA-256 '{hashResult.Value}', expected '{source.Sha256}'.",
                Recoverable: true,
                "locate_original_source"));
        }

        return DomainResult<SourceIntegrityResult>.Success(new SourceIntegrityResult(
            source.SourceId,
            fullPath,
            source.Sha256.ToLowerInvariant(),
            hashResult.Value,
            Matches: true));
    }

    private static async Task<DomainResult<string>> ComputeSha256Async(
        string fullPath,
        CancellationToken cancellationToken)
    {
        try
        {
            await using var stream = new FileStream(
                fullPath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 128 * 1024,
                options: FileOptions.Asynchronous | FileOptions.SequentialScan);
            byte[] hash = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
            return DomainResult<string>.Success(Convert.ToHexStringLower(hash));
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return DomainResult<string>.Failure(DomainErrors.IoFailure(
                "SOURCE_READ_FAILED",
                "Errors.SourceReadFailed",
                $"Source file '{fullPath}' could not be hashed: {exception.Message}",
                "locate_source"));
        }
    }
}
