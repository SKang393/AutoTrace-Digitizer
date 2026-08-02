// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace GraphReader.Imaging;

public sealed record DerivedCacheKeyInput(
    string InputSha256,
    string PanelCrop,
    TransformChain TransformChain,
    string StageName,
    string StageVersion,
    string ModelSha256,
    IReadOnlyDictionary<string, string> Parameters,
    string ContractVersion);

public readonly record struct DerivedCacheKey
{
    public DerivedCacheKey(string sha256)
    {
        if (sha256.Length != 64 || sha256.Any(static character => !Uri.IsHexDigit(character)))
        {
            throw new ArgumentException("A cache key must be a 64-character SHA-256 value.", nameof(sha256));
        }

        Sha256 = sha256.ToLowerInvariant();
    }

    public string Sha256 { get; }

    public static DerivedCacheKey Create(DerivedCacheKeyInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var canonical = new StringBuilder();
        Append(canonical, input.InputSha256);
        Append(canonical, input.PanelCrop);
        foreach (ImageTransform transform in input.TransformChain.Transforms)
        {
            Append(canonical, transform.Kind.ToString());
            Append(canonical, transform.SourceSpace.ToString());
            Append(canonical, transform.TargetSpace.ToString());
            foreach (double value in transform.Matrix.ToValues())
            {
                Append(canonical, value.ToString("R", CultureInfo.InvariantCulture));
            }

            foreach (KeyValuePair<string, double> parameter in transform.Parameters.OrderBy(static pair => pair.Key, StringComparer.Ordinal))
            {
                Append(canonical, parameter.Key);
                Append(canonical, parameter.Value.ToString("R", CultureInfo.InvariantCulture));
            }
        }

        Append(canonical, input.StageName);
        Append(canonical, input.StageVersion);
        Append(canonical, input.ModelSha256);
        foreach (KeyValuePair<string, string> parameter in input.Parameters.OrderBy(static pair => pair.Key, StringComparer.Ordinal))
        {
            Append(canonical, parameter.Key);
            Append(canonical, parameter.Value);
        }

        Append(canonical, input.ContractVersion);
        return new DerivedCacheKey(Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(canonical.ToString()))));
    }

    private static void Append(StringBuilder builder, string value) =>
        builder.Append(value.Length.ToString(CultureInfo.InvariantCulture)).Append(':').Append(value).Append('|');
}

public sealed class ContentAddressedDerivedCache
{
    private readonly string rootDirectory;

    public ContentAddressedDerivedCache(string rootDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(rootDirectory);
        this.rootDirectory = Path.GetFullPath(rootDirectory);
    }

    public async Task PutAsync(DerivedCacheKey key, ReadOnlyMemory<byte> content, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string targetPath = GetPath(key);
        string directory = Path.GetDirectoryName(targetPath)!;
        Directory.CreateDirectory(directory);
        string temporaryPath = targetPath + "." + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture) + ".tmp";

        try
        {
            await using (var stream = new FileStream(
                temporaryPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 81920,
                FileOptions.Asynchronous | FileOptions.WriteThrough))
            {
                await stream.WriteAsync(content, cancellationToken).ConfigureAwait(false);
                await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
                stream.Flush(flushToDisk: true);
            }

            File.Move(temporaryPath, targetPath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporaryPath))
            {
                File.Delete(temporaryPath);
            }
        }
    }

    public async Task<byte[]?> TryReadAsync(DerivedCacheKey key, CancellationToken cancellationToken)
    {
        string path = GetPath(key);
        try
        {
            return await File.ReadAllBytesAsync(path, cancellationToken).ConfigureAwait(false);
        }
        catch (FileNotFoundException)
        {
            return null;
        }
        catch (DirectoryNotFoundException)
        {
            return null;
        }
    }

    public bool Contains(DerivedCacheKey key) => File.Exists(GetPath(key));

    private string GetPath(DerivedCacheKey key) =>
        Path.Combine(rootDirectory, key.Sha256[..2], key.Sha256 + ".bin");
}
