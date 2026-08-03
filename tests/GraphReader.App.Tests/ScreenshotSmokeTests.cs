// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Windows;
using GraphReader.App.Controls;
using GraphReader.App.Services;
using GraphReader.App.ViewModels;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class ScreenshotSmokeTests
{
    private const string LightThemeSource =
        "/GraphReader.App;component/Themes/LightTheme.xaml";
    private const string DarkThemeSource =
        "/GraphReader.App;component/Themes/DarkTheme.xaml";

    [TestMethod]
    public void FakeGraphWorkspaceRendersDeterministicLightTheme()
    {
        AssertDeterministicRender(LightThemeSource, "session02-workspace-light.png");
    }

    [TestMethod]
    public void FakeGraphWorkspaceRendersDeterministicDarkTheme()
    {
        AssertDeterministicRender(DarkThemeSource, "session02-workspace-dark.png");
    }

    [TestMethod]
    public void PhaseOverlayToggleChangesRenderedPixels()
    {
        var visible = WpfScreenshotHarness.Render(
            () => CreatePhaseCanvas(true),
            520,
            280);
        var hidden = WpfScreenshotHarness.Render(
            () => CreatePhaseCanvas(false),
            520,
            280);

        Assert.AreNotEqual(visible.Sha256, hidden.Sha256);
        CollectionAssert.AreNotEqual(visible.PngBytes, hidden.PngBytes);
    }

    private static GraphCanvasControl CreatePhaseCanvas(bool showPhaseOverlay)
    {
        var tab = new FakeWorkspaceService().CreateWorkspace()[0];
        var canvas = new GraphCanvasControl
        {
            ImageSource = tab.ImageSource,
            PhaseOverlayContent = tab.PhaseOverlayContent,
            PhaseOverlayVisible = showPhaseOverlay,
            ShowCrosshair = false,
        };
        AddResources(canvas, LightThemeSource);
        return canvas;
    }

    private static void AssertDeterministicRender(string themeSource, string evidenceFileName)
    {
        var first = WpfScreenshotHarness.Render(
            () => CreateFakeWorkspace(themeSource, true),
            1180,
            720);
        var second = WpfScreenshotHarness.Render(
            () => CreateFakeWorkspace(themeSource, true),
            1180,
            720);

        Assert.AreEqual(1180, first.Width);
        Assert.AreEqual(720, first.Height);
        Assert.IsGreaterThan(2_000, first.PngBytes.Length);
        Assert.AreEqual(first.Sha256, second.Sha256);
        CollectionAssert.AreEqual(first.PngBytes, second.PngBytes);

        var evidenceDirectory = Path.Combine(
            RepositoryTestPaths.Root,
            ".codex",
            "full-build",
            "evidence");
        Directory.CreateDirectory(evidenceDirectory);
        File.WriteAllBytes(
            Path.Combine(evidenceDirectory, evidenceFileName),
            first.PngBytes);
    }

    private static FrameworkElement CreateFakeWorkspace(string themeSource, bool showPhaseOverlay)
    {
        var viewModel = new MainWindowViewModel
        {
            IsPhaseOverlayVisible = showPhaseOverlay,
        };
        var window = new MainWindow
        {
            DataContext = viewModel,
        };
        AddResources(window, themeSource);

        return (FrameworkElement)window.Content;
    }

    private static void AddResources(FrameworkElement element, string themeSource)
    {
        string[] resourceSources =
        [
            "/GraphReader.App;component/Themes/DesignTokens.xaml",
            themeSource,
            "/GraphReader.App;component/Themes/Controls.xaml",
            "/GraphReader.App;component/Localization/Resources.en-US.xaml",
        ];
        foreach (var source in resourceSources)
        {
            element.Resources.MergedDictionaries.Add(
                new ResourceDictionary
                {
                    Source = new Uri(source, UriKind.RelativeOrAbsolute),
                });
        }

    }
}
