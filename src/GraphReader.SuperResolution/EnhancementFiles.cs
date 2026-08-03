// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Buffers.Binary;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;

namespace GraphReader.SuperResolution;

public static class EnhancementHashing
{
    public static async Task<string> ComputeFileSha256Async(
        string path,
        CancellationToken cancellationToken)
    {
        await using FileStream stream = new(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            128 * 1024,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        byte[] hash = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexStringLower(hash);
    }

    public static string ComputeCacheKey(params string[] components)
    {
        ArgumentNullException.ThrowIfNull(components);
        var canonical = new StringBuilder();
        foreach (string component in components)
        {
            ArgumentNullException.ThrowIfNull(component);
            canonical.Append(component.Length).Append(':').Append(component).Append(';');
        }

        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(canonical.ToString())));
    }

    public static string ComputeModelSha256(IEnumerable<(string RelativePath, string Sha256)> artifacts)
    {
        ArgumentNullException.ThrowIfNull(artifacts);
        var canonical = new StringBuilder();
        foreach ((string relativePath, string sha256) in artifacts.OrderBy(
                     static item => item.RelativePath,
                     StringComparer.Ordinal))
        {
            string normalizedPath = relativePath.Replace('\\', '/');
            string normalizedHash = NormalizeSha256(sha256);
            canonical
                .Append(normalizedPath.Length)
                .Append(':')
                .Append(normalizedPath)
                .Append(':')
                .Append(normalizedHash)
                .Append(';');
        }

        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(canonical.ToString())));
    }

    public static string NormalizeSha256(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        if (value.Length != 64 || value.Any(static character => !Uri.IsHexDigit(character)))
        {
            throw new ArgumentException("SHA-256 values must contain exactly 64 hexadecimal characters.", nameof(value));
        }

        return value.ToLowerInvariant();
    }
}

public interface IOutputImageInspector
{
    PixelDimensions ReadDimensions(string path);
}

public sealed class PngOutputImageInspector : IOutputImageInspector
{
    private static readonly byte[] Signature = [137, 80, 78, 71, 13, 10, 26, 10];
    private const int MaximumChunkBytes = 256 * 1024 * 1024;
    private const long MaximumDecodedBytes = 512L * 1024 * 1024;

    public PixelDimensions ReadDimensions(string path)
    {
        Span<byte> signature = stackalloc byte[8];
        using FileStream stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        stream.ReadExactly(signature);
        if (!signature.SequenceEqual(Signature))
        {
            throw new InvalidDataException("Real-ESRGAN output does not have a valid PNG signature.");
        }

        PixelDimensions dimensions = default;
        bool sawHeader = false;
        bool sawImageData = false;
        bool sawEnd = false;
        int channels = 0;
        using var compressedImageData = new MemoryStream();
        Span<byte> chunkHeader = stackalloc byte[8];
        Span<byte> expectedCrcBytes = stackalloc byte[4];
        Span<byte> headerData = stackalloc byte[13];
        var buffer = new byte[64 * 1024];
        while (!sawEnd)
        {
            headerData.Clear();
            stream.ReadExactly(chunkHeader);
            int length = BinaryPrimitives.ReadInt32BigEndian(chunkHeader[..4]);
            if (length < 0 || length > MaximumChunkBytes)
            {
                throw new InvalidDataException("Real-ESRGAN output contains an invalid PNG chunk length.");
            }

            ReadOnlySpan<byte> chunkType = chunkHeader[4..8];
            uint crc = UpdateCrc(uint.MaxValue, chunkType);
            int remaining = length;
            int dataOffset = 0;
            while (remaining > 0)
            {
                int count = Math.Min(remaining, buffer.Length);
                stream.ReadExactly(buffer.AsSpan(0, count));
                crc = UpdateCrc(crc, buffer.AsSpan(0, count));
                if (chunkType.SequenceEqual("IHDR"u8) && dataOffset < headerData.Length)
                {
                    int retained = Math.Min(count, headerData.Length - dataOffset);
                    buffer.AsSpan(0, retained).CopyTo(headerData[dataOffset..]);
                }

                if (chunkType.SequenceEqual("IDAT"u8))
                {
                    if (compressedImageData.Length + count > MaximumChunkBytes)
                    {
                        throw new InvalidDataException("Real-ESRGAN output contains too much compressed PNG image data.");
                    }

                    compressedImageData.Write(buffer, 0, count);
                }

                dataOffset += count;
                remaining -= count;
            }

            stream.ReadExactly(expectedCrcBytes);
            uint expectedCrc = BinaryPrimitives.ReadUInt32BigEndian(expectedCrcBytes);
            if (~crc != expectedCrc)
            {
                throw new InvalidDataException("Real-ESRGAN output contains a PNG chunk with an invalid checksum.");
            }

            if (!sawHeader)
            {
                if (!chunkType.SequenceEqual("IHDR"u8) || length != 13)
                {
                    throw new InvalidDataException("Real-ESRGAN output does not begin with a valid PNG IHDR chunk.");
                }

                int width = BinaryPrimitives.ReadInt32BigEndian(headerData[..4]);
                int height = BinaryPrimitives.ReadInt32BigEndian(headerData.Slice(4, 4));
                if (width <= 0 || height <= 0)
                {
                    throw new InvalidDataException("Real-ESRGAN output has invalid pixel dimensions.");
                }

                dimensions = new PixelDimensions(width, height);
                channels = ValidateHeaderEncoding(headerData);
                sawHeader = true;
            }
            else if (chunkType.SequenceEqual("IDAT"u8))
            {
                sawImageData |= length > 0;
            }
            else if (chunkType.SequenceEqual("IEND"u8))
            {
                if (length != 0)
                {
                    throw new InvalidDataException("Real-ESRGAN output has an invalid PNG IEND chunk.");
                }

                sawEnd = true;
            }
        }

        if (!sawImageData || stream.Position != stream.Length)
        {
            throw new InvalidDataException("Real-ESRGAN output is incomplete or has trailing PNG data.");
        }

        ValidateDecodedImageData(compressedImageData, dimensions, channels);

        return dimensions;
    }

    private static int ValidateHeaderEncoding(ReadOnlySpan<byte> headerData)
    {
        int bitDepth = headerData[8];
        int colorType = headerData[9];
        int compressionMethod = headerData[10];
        int filterMethod = headerData[11];
        int interlaceMethod = headerData[12];
        if (bitDepth != 8 || compressionMethod != 0 || filterMethod != 0 || interlaceMethod != 0)
        {
            throw new InvalidDataException(
                "Real-ESRGAN output must be a non-interlaced 8-bit PNG using standard compression and filtering.");
        }

        return colorType switch
        {
            0 => 1,
            2 => 3,
            4 => 2,
            6 => 4,
            _ => throw new InvalidDataException("Real-ESRGAN output uses an unsupported PNG color type.")
        };
    }

    private static void ValidateDecodedImageData(
        MemoryStream compressedImageData,
        PixelDimensions dimensions,
        int channels)
    {
        long pixelBytesPerRow = checked((long)dimensions.Width * channels);
        long expectedBytes = checked((pixelBytesPerRow + 1) * dimensions.Height);
        if (expectedBytes > MaximumDecodedBytes)
        {
            throw new InvalidDataException("Real-ESRGAN output exceeds the supported decoded PNG size.");
        }

        compressedImageData.Position = 0;
        using var decoder = new ZLibStream(compressedImageData, CompressionMode.Decompress, leaveOpen: true);
        var buffer = new byte[64 * 1024];
        try
        {
            for (int index = 0; index < dimensions.Height; index++)
            {
                int filter = decoder.ReadByte();
                if (filter < 0)
                {
                    throw new EndOfStreamException();
                }

                if (filter > 4)
                {
                    throw new InvalidDataException("Real-ESRGAN output contains an invalid PNG scanline filter.");
                }

                long remaining = pixelBytesPerRow;
                while (remaining > 0)
                {
                    int count = decoder.Read(buffer, 0, (int)Math.Min(remaining, buffer.Length));
                    if (count == 0)
                    {
                        throw new EndOfStreamException();
                    }

                    remaining -= count;
                }
            }

            if (decoder.ReadByte() != -1)
            {
                throw new InvalidDataException("Real-ESRGAN output contains excess decoded PNG image data.");
            }
        }
        catch (EndOfStreamException exception)
        {
            throw new InvalidDataException("Real-ESRGAN output contains incomplete PNG image data.", exception);
        }
    }

    private static uint UpdateCrc(uint crc, ReadOnlySpan<byte> bytes)
    {
        foreach (byte value in bytes)
        {
            crc ^= value;
            for (int bit = 0; bit < 8; bit++)
            {
                crc = (crc & 1) == 0
                    ? crc >> 1
                    : (crc >> 1) ^ 0xedb88320u;
            }
        }

        return crc;
    }
}

internal static class EnhancementPaths
{
    public static string ResolveArtifact(string modelRoot, string relativePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(relativePath);
        if (Path.IsPathRooted(relativePath))
        {
            throw new ArgumentException("Model artifact paths must be relative to the configured model directory.", nameof(relativePath));
        }

        string root = Path.GetFullPath(modelRoot);
        string candidate = Path.GetFullPath(Path.Combine(root, relativePath));
        string relative = Path.GetRelativePath(root, candidate);
        if (relative.Equals("..", StringComparison.Ordinal) ||
            relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal) ||
            Path.IsPathRooted(relative))
        {
            throw new ArgumentException("Model artifact paths may not escape the configured model directory.", nameof(relativePath));
        }

        return candidate;
    }

    public static bool SamePath(string first, string second) =>
        string.Equals(
            Path.GetFullPath(first).TrimEnd(Path.DirectorySeparatorChar),
            Path.GetFullPath(second).TrimEnd(Path.DirectorySeparatorChar),
            StringComparison.OrdinalIgnoreCase);
}
