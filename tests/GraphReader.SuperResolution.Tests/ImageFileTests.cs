// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Buffers.Binary;
using System.IO.Compression;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.SuperResolution.Tests;

[TestClass]
public sealed class ImageFileTests
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    [TestMethod]
    public void PngInspectorReadsExactBigEndianDimensions()
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.png");
        try
        {
            File.WriteAllBytes(path, CreatePngHeader(62, 38));

            PixelDimensions dimensions = new PngOutputImageInspector().ReadDimensions(path);

            Assert.AreEqual(new PixelDimensions(62, 38), dimensions);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [TestMethod]
    public void PngInspectorRejectsCorruptSignatureAndInvalidDimensions()
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.png");
        try
        {
            File.WriteAllBytes(path, new byte[24]);
            Assert.ThrowsExactly<InvalidDataException>(() =>
                new PngOutputImageInspector().ReadDimensions(path));

            File.WriteAllBytes(path, CreatePngHeader(0, 38));
            Assert.ThrowsExactly<InvalidDataException>(() =>
                new PngOutputImageInspector().ReadDimensions(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [TestMethod]
    public void PngInspectorRejectsBadChunkCrcAndTruncation()
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.png");
        try
        {
            byte[] badCrc = CreatePngHeader(62, 38);
            badCrc[^1] ^= 0xff;
            File.WriteAllBytes(path, badCrc);
            Assert.ThrowsExactly<InvalidDataException>(() =>
                new PngOutputImageInspector().ReadDimensions(path));

            byte[] truncated = CreatePngHeader(62, 38)[..^3];
            File.WriteAllBytes(path, truncated);
            Assert.ThrowsExactly<EndOfStreamException>(() =>
                new PngOutputImageInspector().ReadDimensions(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [TestMethod]
    public void PngInspectorRejectsEmptyAndUndecodableImageDataWithValidChunkCrcs()
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.png");
        try
        {
            File.WriteAllBytes(path, CreatePng(62, 38, []));
            Assert.ThrowsExactly<InvalidDataException>(() =>
                new PngOutputImageInspector().ReadDimensions(path));

            File.WriteAllBytes(path, CreatePng(62, 38, [1, 2, 3, 4]));
            Assert.ThrowsExactly<InvalidDataException>(() =>
                new PngOutputImageInspector().ReadDimensions(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [TestMethod]
    public async Task HashingIsDeterministicAndCacheComponentsAreOrderSensitive()
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.bin");
        try
        {
            await File.WriteAllTextAsync(path, "abc");
            string hash = await EnhancementHashing.ComputeFileSha256Async(path, CancellationToken.None);
            Assert.AreEqual(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                hash);

            string first = EnhancementHashing.ComputeCacheKey("source", "runtime", "model", "tile=0");
            string same = EnhancementHashing.ComputeCacheKey("source", "runtime", "model", "tile=0");
            string changed = EnhancementHashing.ComputeCacheKey("source", "runtime", "model", "tile=64");
            string reordered = EnhancementHashing.ComputeCacheKey("runtime", "source", "model", "tile=0");
            Assert.AreEqual(first, same);
            Assert.AreNotEqual(first, changed);
            Assert.AreNotEqual(first, reordered);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static byte[] CreatePngHeader(int width, int height)
    {
        int encodedWidth = Math.Max(1, width);
        int encodedHeight = Math.Max(1, height);
        var pixels = new byte[checked((encodedWidth * 3 + 1) * encodedHeight)];
        using var compressed = new MemoryStream();
        using (var encoder = new ZLibStream(compressed, CompressionLevel.SmallestSize, leaveOpen: true))
        {
            encoder.Write(pixels);
        }

        return CreatePng(width, height, compressed.ToArray());
    }

    private static byte[] CreatePng(int width, int height, ReadOnlySpan<byte> imageData)
    {
        var headerData = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(headerData.AsSpan(0, 4), width);
        BinaryPrimitives.WriteInt32BigEndian(headerData.AsSpan(4, 4), height);
        headerData[8] = 8;
        headerData[9] = 2;

        using var stream = new MemoryStream();
        stream.Write(PngSignature);
        WriteChunk(stream, "IHDR"u8, headerData);
        WriteChunk(stream, "IDAT"u8, imageData);
        WriteChunk(stream, "IEND"u8, []);
        return stream.ToArray();
    }

    private static void WriteChunk(Stream stream, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(length, data.Length);
        stream.Write(length);
        stream.Write(type);
        stream.Write(data);

        uint crc = UpdateCrc(uint.MaxValue, type);
        crc = UpdateCrc(crc, data);
        Span<byte> checksum = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(checksum, ~crc);
        stream.Write(checksum);
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
