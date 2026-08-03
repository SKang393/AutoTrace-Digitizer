// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Security.Cryptography;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace GraphReader.App.Tests;

internal sealed record RenderedScreenshot(int Width, int Height, byte[] PngBytes, string Sha256);

internal static class WpfScreenshotHarness
{
    public static RenderedScreenshot Render(Func<FrameworkElement> elementFactory, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(elementFactory);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);

        return StaTestHost.Run(
            () =>
            {
                var element = elementFactory();
                ArgumentNullException.ThrowIfNull(element);
                element.Measure(new Size(width, height));
                element.Arrange(new Rect(0, 0, width, height));
                element.UpdateLayout();

                var bitmap = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
                bitmap.Render(element);
                bitmap.Freeze();

                var encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(bitmap));
                using var output = new MemoryStream();
                encoder.Save(output);
                var png = output.ToArray();
                return new RenderedScreenshot(
                    bitmap.PixelWidth,
                    bitmap.PixelHeight,
                    png,
                    Convert.ToHexStringLower(SHA256.HashData(png)));
            });
    }
}
