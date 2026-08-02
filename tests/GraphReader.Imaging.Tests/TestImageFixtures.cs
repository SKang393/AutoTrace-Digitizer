// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Imazen.WebP;

namespace GraphReader.Imaging.Tests;

internal static class TestImageFixtures
{
    public const int Width = 7;
    public const int Height = 5;

    public static string CreateDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "GraphReader.Imaging.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    public static string Write(string directory, ImageFileFormat format, string? fileName = null)
    {
        string extension = format switch
        {
            ImageFileFormat.Png => ".png",
            ImageFileFormat.Jpeg => ".jpg",
            ImageFileFormat.Tiff => ".tif",
            ImageFileFormat.Bmp => ".bmp",
            ImageFileFormat.WebP => ".webp",
            _ => throw new ArgumentOutOfRangeException(nameof(format))
        };
        string path = Path.Combine(directory, fileName ?? (format + extension));
        File.WriteAllBytes(path, CreateBytes(format));
        return path;
    }

    public static byte[] CreateBytes(ImageFileFormat format)
    {
        byte[] bgra = CreateBgraPixels();
        if (format == ImageFileFormat.WebP)
        {
            return WebPEncoder.Encode(
                bgra,
                Width,
                Height,
                Width * 4,
                WebPPixelFormat.Bgra,
                quality: 100);
        }

        BitmapSource bitmap = BitmapSource.Create(
            Width,
            Height,
            120,
            120,
            PixelFormats.Bgra32,
            palette: null,
            bgra,
            Width * 4);
        BitmapEncoder encoder = format switch
        {
            ImageFileFormat.Png => new PngBitmapEncoder(),
            ImageFileFormat.Jpeg => new JpegBitmapEncoder { QualityLevel = 100 },
            ImageFileFormat.Tiff => new TiffBitmapEncoder(),
            ImageFileFormat.Bmp => new BmpBitmapEncoder(),
            _ => throw new ArgumentOutOfRangeException(nameof(format))
        };
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        using var stream = new MemoryStream();
        encoder.Save(stream);
        return stream.ToArray();
    }

    public static ImportedImage FakeImportedImage(string path, byte seed)
    {
        byte[] bytes = [seed, 2, 3, 4];
        return new ImportedImage(
            path,
            new string(seed < 10 ? 'a' : 'b', 64),
            new ImageMetadata(Width, Height, ImageFileFormat.Png, "image/png", 1, 8, 96, 96, bytes.Length),
            new ImmutableImageBytes(bytes));
    }

    private static byte[] CreateBgraPixels()
    {
        var pixels = new byte[Width * Height * 4];
        for (int pixel = 0; pixel < Width * Height; pixel++)
        {
            pixels[(pixel * 4) + 0] = checked((byte)(pixel * 3));
            pixels[(pixel * 4) + 1] = checked((byte)(255 - (pixel * 2)));
            pixels[(pixel * 4) + 2] = checked((byte)(pixel * 5));
            pixels[(pixel * 4) + 3] = 255;
        }

        return pixels;
    }
}
