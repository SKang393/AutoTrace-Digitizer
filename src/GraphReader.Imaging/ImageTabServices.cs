// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.IO;

namespace GraphReader.Imaging;

public sealed record ImageTab(Guid TabId, string Title, ImportedImage Image);

public sealed record ImageTabOpenResult(ImageTab? Tab, ImageImportError? Error)
{
    public bool IsSuccess => Tab is not null && Error is null;
}

public interface IImageTabService
{
    IReadOnlyList<ImageTab> Tabs { get; }

    Task<ImageTabOpenResult> OpenAsync(string path, CancellationToken cancellationToken);

    bool Close(Guid tabId);
}

public sealed class ImageTabService : IImageTabService, IDisposable
{
    private readonly IImageImportService importService;
    private readonly List<ImageTab> tabs = new();
    private readonly SemaphoreSlim gate = new(1, 1);

    public ImageTabService(IImageImportService importService)
    {
        this.importService = importService ?? throw new ArgumentNullException(nameof(importService));
    }

    public IReadOnlyList<ImageTab> Tabs
    {
        get
        {
            gate.Wait();
            try
            {
                return new ReadOnlyCollection<ImageTab>(tabs.ToArray());
            }
            finally
            {
                gate.Release();
            }
        }
    }

    public async Task<ImageTabOpenResult> OpenAsync(string path, CancellationToken cancellationToken)
    {
        ImageImportResult result = await importService.ImportAsync(path, cancellationToken).ConfigureAwait(false);
        if (result.Image is null)
        {
            return new ImageTabOpenResult(null, result.Error);
        }

        var tab = new ImageTab(Guid.NewGuid(), Path.GetFileName(path), result.Image);
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            tabs.Add(tab);
        }
        finally
        {
            gate.Release();
        }

        return new ImageTabOpenResult(tab, null);
    }

    public bool Close(Guid tabId)
    {
        gate.Wait();
        try
        {
            int index = tabs.FindIndex(tab => tab.TabId == tabId);
            if (index < 0)
            {
                return false;
            }

            tabs.RemoveAt(index);
            return true;
        }
        finally
        {
            gate.Release();
        }
    }

    public void Dispose() => gate.Dispose();
}

public sealed class FakeImageImportService : IImageImportService
{
    private readonly IReadOnlyDictionary<string, ImageImportResult> configuredResults;

    public FakeImageImportService(IReadOnlyDictionary<string, ImageImportResult> configuredResults)
    {
        this.configuredResults = configuredResults ?? throw new ArgumentNullException(nameof(configuredResults));
    }

    public List<string> RequestedPaths { get; } = new();

    public Task<ImageImportResult> ImportAsync(string path, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        RequestedPaths.Add(path);
        if (configuredResults.TryGetValue(path, out ImageImportResult? result))
        {
            return Task.FromResult(result);
        }

        return Task.FromResult(ImageImportResult.Failure(
            path,
            0,
            new ImageImportError(
                ImageImportErrorCode.FileNotFound,
                ImageErrorSeverity.Error,
                "Errors.ImageNotFound",
                "The fake importer has no configured result for this path.",
                Recoverable: true,
                ImageSuggestedAction.Retry,
                path)));
    }

    public async Task<BatchImportResult> ImportBatchAsync(
        IEnumerable<string> paths,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(paths);
        string[] orderedPaths = paths.ToArray();
        var firstIndexByHash = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var results = new List<ImageImportResult>(orderedPaths.Length);
        for (int index = 0; index < orderedPaths.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string path = orderedPaths[index];
            ImageImportResult result = await ImportAsync(path, cancellationToken).ConfigureAwait(false);
            if (result.Image is { } image)
            {
                image = image with { InputIndex = index, DuplicateOfInputIndex = null };
                if (firstIndexByHash.TryGetValue(image.Sha256, out int originalIndex))
                {
                    image = image with { DuplicateOfInputIndex = originalIndex };
                }
                else
                {
                    firstIndexByHash.Add(image.Sha256, index);
                }

                result = ImageImportResult.Success(image);
            }
            else if (result.Error is { } error)
            {
                result = ImageImportResult.Failure(path, index, error with { SourcePath = path });
            }

            results.Add(result);
        }

        return new BatchImportResult(results);
    }
}
