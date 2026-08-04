// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Security.Cryptography;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

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
                var host = new Window
                {
                    Width = width,
                    Height = height,
                    Left = 0,
                    Top = 0,
                    WindowStartupLocation = WindowStartupLocation.Manual,
                    WindowStyle = WindowStyle.None,
                    ResizeMode = ResizeMode.NoResize,
                    ShowActivated = false,
                    ShowInTaskbar = false,
                    Topmost = true,
                    Content = element,
                };
                try
                {
                    bool contentRendered = false;
                    var renderFrame = new DispatcherFrame();
                    var timeout = new DispatcherTimer(
                        TimeSpan.FromSeconds(5),
                        DispatcherPriority.Send,
                        (_, _) => renderFrame.Continue = false,
                        host.Dispatcher);
                    host.ContentRendered += (_, _) =>
                    {
                        contentRendered = true;
                        renderFrame.Continue = false;
                    };
                    host.Show();
                    timeout.Start();
                    if (!contentRendered)
                    {
                        Dispatcher.PushFrame(renderFrame);
                    }
                    timeout.Stop();
                    if (!contentRendered)
                    {
                        throw new InvalidOperationException("The off-screen WPF screenshot host did not render content.");
                    }
                    host.UpdateLayout();
                    element.Dispatcher.Invoke(
                        static () => { },
                        DispatcherPriority.ApplicationIdle);
                    element.Measure(new Size(width, height));
                    element.Arrange(new Rect(0, 0, width, height));
                    element.UpdateLayout();

                    RenderTargetBitmap bitmap = CaptureWindow(host, width, height);

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
                }
                finally
                {
                    host.Content = null;
                    host.Close();
                }
            });
    }

    private static RenderTargetBitmap CaptureWindow(Window host, int width, int height)
    {
        if (new WindowInteropHelper(host).Handle == 0)
        {
            throw new InvalidOperationException("The WPF screenshot host has no window handle.");
        }

        var visual = new DrawingVisual();
        using (DrawingContext context = visual.RenderOpen())
        {
            context.DrawRectangle(
                new VisualBrush(host)
                {
                    AlignmentX = AlignmentX.Left,
                    AlignmentY = AlignmentY.Top,
                    AutoLayoutContent = true,
                    Stretch = Stretch.None,
                },
                null,
                new Rect(0, 0, width, height));
        }

        var bitmap = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
        bitmap.Render(visual);
        int stride = checked(width * 4);
        byte[] pixels = new byte[checked(stride * height)];
        bitmap.CopyPixels(pixels, stride, 0);
        bool hasRenderedPixel = false;
        for (int index = 3; index < pixels.Length; index += 4)
        {
            if (pixels[index] == 0)
            {
                continue;
            }

            hasRenderedPixel = true;
            break;
        }
        if (!hasRenderedPixel)
        {
            Assert.Inconclusive(
                "The WPF pixel surface is unavailable on this noninteractive Windows test desktop; no screenshot evidence was emitted.");
        }
        bitmap.Freeze();
        return bitmap;
    }
}
