// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;

namespace GraphReader.SuperResolution;

internal sealed record EnhancementCacheMetadata(
    int ContractVersion,
    string StageVersion,
    string CacheKey,
    string SourceSha256,
    string OutputSha256,
    string ModelSha256,
    string RuntimeSha256,
    int Width,
    int Height,
    int Scale,
    int TileSize,
    int GpuIndex,
    string Provider);

internal sealed record CacheRestoreResult(bool Hit, string? OutputSha256 = null);

internal sealed class EnhancementCache
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private readonly string _cacheRoot;
    private readonly IOutputImageInspector _imageInspector;

    public EnhancementCache(string cacheRoot, IOutputImageInspector imageInspector)
    {
        _cacheRoot = Path.GetFullPath(cacheRoot);
        _imageInspector = imageInspector;
    }

    public string CreateWorkDirectory()
    {
        string workRoot = Path.Combine(_cacheRoot, "work");
        Directory.CreateDirectory(workRoot);
        string workDirectory = Path.Combine(workRoot, Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(workDirectory);
        return workDirectory;
    }

    public async ValueTask<EntryLease> AcquireEntryLeaseAsync(
        string cacheKey,
        CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheKey);
        string lockDirectory = Path.Combine(_cacheRoot, "locks");
        Directory.CreateDirectory(lockDirectory);
        string lockPath = Path.Combine(lockDirectory, $"{cacheKey}.lock");
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                var stream = new FileStream(
                    lockPath,
                    FileMode.OpenOrCreate,
                    FileAccess.ReadWrite,
                    FileShare.None,
                    bufferSize: 1,
                    FileOptions.Asynchronous);
                return new EntryLease(stream);
            }
            catch (IOException exception) when (IsSharingViolation(exception))
            {
                await Task.Delay(TimeSpan.FromMilliseconds(25), cancellationToken).ConfigureAwait(false);
            }
        }
    }

    public async Task<CacheRestoreResult> TryRestoreAsync(
        EnhancementCacheMetadata expected,
        string destinationPath,
        CancellationToken cancellationToken)
    {
        string entry = EntryDirectory(expected.CacheKey);
        string imagePath = Path.Combine(entry, "enhanced.png");
        string metadataPath = Path.Combine(entry, "metadata.json");
        if (!File.Exists(imagePath) || !File.Exists(metadataPath))
        {
            return new CacheRestoreResult(false);
        }

        try
        {
            await using FileStream metadataStream = File.OpenRead(metadataPath);
            EnhancementCacheMetadata? actual = await JsonSerializer.DeserializeAsync<EnhancementCacheMetadata>(
                metadataStream,
                JsonOptions,
                cancellationToken).ConfigureAwait(false);
            if (actual is null || !SameIdentity(actual, expected))
            {
                return new CacheRestoreResult(false);
            }

            string outputHash = await EnhancementHashing.ComputeFileSha256Async(imagePath, cancellationToken).ConfigureAwait(false);
            if (!string.Equals(outputHash, actual.OutputSha256, StringComparison.OrdinalIgnoreCase) ||
                _imageInspector.ReadDimensions(imagePath) != new PixelDimensions(expected.Width, expected.Height))
            {
                return new CacheRestoreResult(false);
            }

            await PromoteCopyAsync(imagePath, destinationPath, cancellationToken).ConfigureAwait(false);
            string stagedHash = await EnhancementHashing.ComputeFileSha256Async(
                destinationPath,
                cancellationToken).ConfigureAwait(false);
            if (!string.Equals(stagedHash, outputHash, StringComparison.OrdinalIgnoreCase) ||
                _imageInspector.ReadDimensions(destinationPath) != new PixelDimensions(expected.Width, expected.Height))
            {
                File.Delete(destinationPath);
                return new CacheRestoreResult(false);
            }

            return new CacheRestoreResult(true, stagedHash);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or JsonException or InvalidDataException)
        {
            return new CacheRestoreResult(false);
        }
    }

    public async Task StoreAsync(
        string sourceImagePath,
        EnhancementCacheMetadata metadata,
        CancellationToken cancellationToken)
    {
        string entry = EntryDirectory(metadata.CacheKey);
        if (Directory.Exists(entry))
        {
            Directory.Delete(entry, recursive: true);
        }

        string parent = Path.GetDirectoryName(entry)!;
        Directory.CreateDirectory(parent);
        string temporary = Path.Combine(parent, $".{metadata.CacheKey}.{Guid.NewGuid():N}.tmp");
        Directory.CreateDirectory(temporary);
        try
        {
            string imagePath = Path.Combine(temporary, "enhanced.png");
            await CopyFileAsync(sourceImagePath, imagePath, cancellationToken).ConfigureAwait(false);
            string metadataPath = Path.Combine(temporary, "metadata.json");
            await using (FileStream metadataStream = new(
                metadataPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                16 * 1024,
                FileOptions.Asynchronous))
            {
                await JsonSerializer.SerializeAsync(
                    metadataStream,
                    metadata,
                    JsonOptions,
                    cancellationToken).ConfigureAwait(false);
                await metadataStream.FlushAsync(cancellationToken).ConfigureAwait(false);
            }

            Directory.Move(temporary, entry);
        }
        finally
        {
            if (Directory.Exists(temporary))
            {
                Directory.Delete(temporary, recursive: true);
            }
        }
    }

    public static async Task PromoteCopyAsync(
        string sourcePath,
        string destinationPath,
        CancellationToken cancellationToken)
    {
        string destination = Path.GetFullPath(destinationPath);
        string? parent = Path.GetDirectoryName(destination);
        if (parent is null)
        {
            throw new IOException("Output path has no parent directory.");
        }

        Directory.CreateDirectory(parent);
        string temporary = Path.Combine(parent, $".{Path.GetFileName(destination)}.{Guid.NewGuid():N}.tmp");
        try
        {
            await CopyFileAsync(sourcePath, temporary, cancellationToken).ConfigureAwait(false);
            File.Move(temporary, destination, overwrite: false);
        }
        finally
        {
            if (File.Exists(temporary))
            {
                File.Delete(temporary);
            }
        }
    }

    public static async Task CopyFileAsync(
        string sourcePath,
        string destinationPath,
        CancellationToken cancellationToken)
    {
        await using FileStream source = new(
            sourcePath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            128 * 1024,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        await using FileStream destination = new(
            destinationPath,
            FileMode.CreateNew,
            FileAccess.Write,
            FileShare.None,
            128 * 1024,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        await source.CopyToAsync(destination, cancellationToken).ConfigureAwait(false);
        await destination.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    public void DeleteWorkDirectory(string workDirectory)
    {
        string workRoot = Path.GetFullPath(Path.Combine(_cacheRoot, "work"));
        string candidate = Path.GetFullPath(workDirectory);
        string relative = Path.GetRelativePath(workRoot, candidate);
        if (!relative.Equals(".", StringComparison.Ordinal) &&
            !relative.Equals("..", StringComparison.Ordinal) &&
            !relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal) &&
            !Path.IsPathRooted(relative) &&
            Directory.Exists(candidate))
        {
            Directory.Delete(candidate, recursive: true);
        }
    }

    private string EntryDirectory(string cacheKey) =>
        Path.Combine(_cacheRoot, "entries", cacheKey[..2], cacheKey);

    private static bool IsSharingViolation(IOException exception) =>
        exception.HResult is unchecked((int)0x80070020) or unchecked((int)0x80070021);

    private static bool SameIdentity(
        EnhancementCacheMetadata actual,
        EnhancementCacheMetadata expected) =>
        actual.ContractVersion == expected.ContractVersion &&
        string.Equals(actual.StageVersion, expected.StageVersion, StringComparison.Ordinal) &&
        string.Equals(actual.CacheKey, expected.CacheKey, StringComparison.Ordinal) &&
        string.Equals(actual.SourceSha256, expected.SourceSha256, StringComparison.OrdinalIgnoreCase) &&
        string.Equals(actual.ModelSha256, expected.ModelSha256, StringComparison.OrdinalIgnoreCase) &&
        string.Equals(actual.RuntimeSha256, expected.RuntimeSha256, StringComparison.OrdinalIgnoreCase) &&
        actual.Width == expected.Width &&
        actual.Height == expected.Height &&
        actual.Scale == expected.Scale &&
        actual.TileSize == expected.TileSize &&
        actual.GpuIndex == expected.GpuIndex &&
        string.Equals(actual.Provider, expected.Provider, StringComparison.Ordinal);

    internal sealed class EntryLease : IAsyncDisposable
    {
        private FileStream? _stream;

        public EntryLease(FileStream stream)
        {
            _stream = stream;
        }

        public async ValueTask DisposeAsync()
        {
            FileStream? stream = Interlocked.Exchange(ref _stream, null);
            if (stream is not null)
            {
                await stream.DisposeAsync().ConfigureAwait(false);
            }
        }
    }
}
