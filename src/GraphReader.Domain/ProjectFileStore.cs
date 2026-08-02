// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text;

namespace GraphReader.Domain;

public enum AtomicSaveStage
{
    TemporaryFileCreated,
    TemporaryFileFlushed,
    BeforeCommit
}

public interface IAtomicSaveInterceptor
{
    ValueTask OnStageAsync(
        AtomicSaveStage stage,
        string targetPath,
        string temporaryPath,
        CancellationToken cancellationToken);
}

public sealed class NoOpAtomicSaveInterceptor : IAtomicSaveInterceptor
{
    public static NoOpAtomicSaveInterceptor Instance { get; } = new();

    private NoOpAtomicSaveInterceptor()
    {
    }

    public ValueTask OnStageAsync(
        AtomicSaveStage stage,
        string targetPath,
        string temporaryPath,
        CancellationToken cancellationToken) => ValueTask.CompletedTask;
}

public sealed record ProjectSaveReceipt(
    string Path,
    long BytesWritten,
    DateTimeOffset CompletedUtc);

public sealed class ProjectFileStore
{
    private const long MaximumProjectBytes = 64L * 1024 * 1024;
    private static readonly UTF8Encoding Utf8WithoutBom = new(false, true);
    private readonly IAtomicSaveInterceptor _interceptor;
    private readonly ProjectJsonSerializer _serializer;

    public ProjectFileStore(
        ProjectJsonSerializer? serializer = null,
        IAtomicSaveInterceptor? interceptor = null)
    {
        _serializer = serializer ?? new ProjectJsonSerializer();
        _interceptor = interceptor ?? NoOpAtomicSaveInterceptor.Instance;
    }

    public Task<DomainResult<ProjectSaveReceipt>> SaveAsync(
        ProjectDocument project,
        string path,
        CancellationToken cancellationToken = default) =>
        SaveCoreAsync(project, path, overwrite: true, cancellationToken);

    public Task<DomainResult<ProjectSaveReceipt>> SaveNewAsync(
        ProjectDocument project,
        string path,
        CancellationToken cancellationToken = default) =>
        SaveCoreAsync(project, path, overwrite: false, cancellationToken);

    public async Task<DomainResult<ProjectDocument>> LoadAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string fullPath = Path.GetFullPath(path);

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            var file = new FileInfo(fullPath);
            if (!file.Exists)
            {
                return DomainResult<ProjectDocument>.Failure(DomainErrors.IoFailure(
                    "PROJECT_NOT_FOUND",
                    "Errors.ProjectNotFound",
                    $"Project file '{fullPath}' does not exist.",
                    "select_project"));
            }

            if (file.Length > MaximumProjectBytes)
            {
                return DomainResult<ProjectDocument>.Failure(DomainErrors.CorruptProject(
                    $"Project file '{fullPath}' exceeds the {MaximumProjectBytes}-byte safety limit."));
            }

            await using var stream = new FileStream(
                fullPath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 64 * 1024,
                options: FileOptions.Asynchronous | FileOptions.SequentialScan);
            using var reader = new StreamReader(
                stream,
                Utf8WithoutBom,
                detectEncodingFromByteOrderMarks: true,
                bufferSize: 64 * 1024,
                leaveOpen: false);
            string json = await reader.ReadToEndAsync(cancellationToken).ConfigureAwait(false);
            return _serializer.Deserialize(json);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (DecoderFallbackException exception)
        {
            return DomainResult<ProjectDocument>.Failure(DomainErrors.CorruptProject(
                $"Project file '{fullPath}' is not valid UTF-8: {exception.Message}"));
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return DomainResult<ProjectDocument>.Failure(DomainErrors.IoFailure(
                "PROJECT_READ_FAILED",
                "Errors.ProjectReadFailed",
                $"Project file '{fullPath}' could not be read: {exception.Message}",
                "retry"));
        }
    }

    private async Task<DomainResult<ProjectSaveReceipt>> SaveCoreAsync(
        ProjectDocument project,
        string path,
        bool overwrite,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(project);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        cancellationToken.ThrowIfCancellationRequested();

        DomainResult<string> serialized = _serializer.Serialize(project);
        if (!serialized.IsSuccess || serialized.Value is null)
        {
            return DomainResult<ProjectSaveReceipt>.Failure(serialized.Errors);
        }

        string targetPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(targetPath);
        if (string.IsNullOrWhiteSpace(directory))
        {
            return DomainResult<ProjectSaveReceipt>.Failure(DomainErrors.IoFailure(
                "PROJECT_SAVE_PATH_INVALID",
                "Errors.ProjectSavePathInvalid",
                $"Project path '{targetPath}' does not have a parent directory.",
                "select_project_path"));
        }

        string temporaryPath = Path.Combine(
            directory,
            $".{Path.GetFileName(targetPath)}.{Guid.NewGuid():N}.tmp");
        byte[] bytes = Utf8WithoutBom.GetBytes(serialized.Value);

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            Directory.CreateDirectory(directory);
            if (!overwrite && File.Exists(targetPath))
            {
                return DomainResult<ProjectSaveReceipt>.Failure(DomainErrors.IoFailure(
                    "PROJECT_TARGET_EXISTS",
                    "Errors.ProjectTargetExists",
                    $"Project file '{targetPath}' already exists.",
                    "select_new_project_path"));
            }

            await using (var stream = new FileStream(
                temporaryPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 64 * 1024,
                options: FileOptions.Asynchronous | FileOptions.WriteThrough))
            {
                await _interceptor.OnStageAsync(
                    AtomicSaveStage.TemporaryFileCreated,
                    targetPath,
                    temporaryPath,
                    cancellationToken).ConfigureAwait(false);
                await stream.WriteAsync(bytes, cancellationToken).ConfigureAwait(false);
                await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
                stream.Flush(flushToDisk: true);
            }

            await _interceptor.OnStageAsync(
                AtomicSaveStage.TemporaryFileFlushed,
                targetPath,
                temporaryPath,
                cancellationToken).ConfigureAwait(false);
            await _interceptor.OnStageAsync(
                AtomicSaveStage.BeforeCommit,
                targetPath,
                temporaryPath,
                cancellationToken).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();

            CommitTemporaryFile(temporaryPath, targetPath, overwrite);
            return DomainResult<ProjectSaveReceipt>.Success(new ProjectSaveReceipt(
                targetPath,
                bytes.LongLength,
                DateTimeOffset.UtcNow));
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return DomainResult<ProjectSaveReceipt>.Failure(DomainErrors.IoFailure(
                "PROJECT_SAVE_FAILED",
                "Errors.ProjectSaveFailed",
                $"Project file '{targetPath}' could not be saved atomically: {exception.Message}",
                "retry"));
        }
        finally
        {
            try
            {
                if (File.Exists(temporaryPath))
                {
                    File.Delete(temporaryPath);
                }
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                // A stranded adjacent temporary file is safe and can be removed during later maintenance.
            }
        }
    }

    private static void CommitTemporaryFile(string temporaryPath, string targetPath, bool overwrite)
    {
        if (!File.Exists(targetPath))
        {
            File.Move(temporaryPath, targetPath);
            return;
        }

        if (!overwrite)
        {
            throw new IOException($"Project file '{targetPath}' was created before the recovery copy could be committed.");
        }

        File.Replace(temporaryPath, targetPath, destinationBackupFileName: null, ignoreMetadataErrors: true);
    }
}
