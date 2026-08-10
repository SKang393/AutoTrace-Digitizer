// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Xml.Linq;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class CalmPrecisionDesignTests
{
    private const string PresentationNamespace = "http://schemas.microsoft.com/winfx/2006/xaml/presentation";
    private const string XamlNamespace = "http://schemas.microsoft.com/winfx/2006/xaml";
    private const string AutomationNamespace = "clr-namespace:System.Windows.Automation;assembly=PresentationCore";
    private static readonly int[] ExpectedIconSizes = [16, 20, 24, 32, 40, 48, 64, 128, 256];

    [TestMethod]
    public void TypographyAndLightPaletteMatchApprovedSemanticTokens()
    {
        XNamespace xaml = XamlNamespace;
        XDocument tokens = LoadAppXaml("Themes", "DesignTokens.xaml");
        var tokenValues = tokens.Descendants()
            .Where(element => element.Attribute(xaml + "Key") is not null)
            .ToDictionary(
                element => element.Attribute(xaml + "Key")!.Value,
                element => element.Value.Trim(),
                StringComparer.Ordinal);

        Assert.AreEqual("../Assets/Fonts/#Noto Sans", tokenValues["App.FontFamily.Default"]);
        Assert.AreEqual("12", tokenValues["App.FontSize.Caption"]);
        Assert.AreEqual("16", tokenValues["App.LineHeight.Caption"]);
        Assert.AreEqual("14", tokenValues["App.FontSize.Body"]);
        Assert.AreEqual("20", tokenValues["App.LineHeight.Body"]);
        Assert.AreEqual("18", tokenValues["App.FontSize.Subtitle"]);
        Assert.AreEqual("24", tokenValues["App.LineHeight.Subtitle"]);
        Assert.AreEqual("24", tokenValues["App.FontSize.Title"]);
        Assert.AreEqual("32", tokenValues["App.LineHeight.Title"]);

        XDocument light = LoadAppXaml("Themes", "LightTheme.xaml");
        var colors = light.Descendants(XName.Get("SolidColorBrush", PresentationNamespace))
            .ToDictionary(
                element => element.Attribute(xaml + "Key")!.Value,
                element => element.Attribute("Color")!.Value,
                StringComparer.Ordinal);
        Assert.AreEqual("#FFF5F7FA", colors["App.Brush.Canvas"]);
        Assert.AreEqual("#FFFFFFFF", colors["App.Brush.Surface"]);
        Assert.AreEqual("#FF1B1D22", colors["App.Brush.Text"]);
        Assert.AreEqual("#FF5F6673", colors["App.Brush.TextMuted"]);
        Assert.AreEqual("#FFD9DEE7", colors["App.Brush.Border"]);
        Assert.AreEqual("#FF3659E3", colors["App.Brush.Primary"]);
        Assert.AreEqual("#FF087E6B", colors["App.Brush.Success"]);
        Assert.AreEqual("#FFA15C00", colors["App.Brush.Warning"]);
        Assert.AreEqual("#FFB42318", colors["App.Brush.Danger"]);
    }

    [TestMethod]
    public void ShellExposesCalmPrecisionStatesModesDropAndExportEvidence()
    {
        XDocument shell = LoadAppXaml("MainWindow.xaml");
        XNamespace presentation = PresentationNamespace;
        XNamespace automation = AutomationNamespace;
        XElement root = shell.Root!;

        Assert.AreEqual("True", root.Attribute("AllowDrop")?.Value);
        Assert.AreEqual("OnPreviewDragOver", root.Attribute("PreviewDragOver")?.Value);
        Assert.AreEqual("OnFilesDropped", root.Attribute("Drop")?.Value);

        string[] stateKeys = ["State.Empty", "State.Analyzing", "State.Reviewing", "State.ExportPreview"];
        var styles = shell.Descendants(presentation + "Style")
            .Select(element => element.Attribute(XName.Get("Key", XamlNamespace))?.Value)
            .Where(static value => value is not null)
            .ToHashSet(StringComparer.Ordinal);
        foreach (string key in stateKeys)
        {
            Assert.Contains(key, styles);
        }

        string[] modeIds = ["Manual.Mode.Select", "Manual.Mode.Calibrate", "Manual.Mode.Points", "Manual.Mode.Phases"];
        var automationIds = shell.Descendants()
            .Select(element => element.Attribute(automation + "AutomationProperties.AutomationId")?.Value)
            .Where(static value => value is not null)
            .ToHashSet(StringComparer.Ordinal);
        foreach (string id in modeIds)
        {
            Assert.Contains(id, automationIds);
        }

        string source = File.ReadAllText(Path.Combine(AppDirectory, "MainWindow.xaml"));
        StringAssert.Contains(source, "ItemsSource=\"{Binding RecentProjects}\"");
        StringAssert.Contains(source, "ExportSummary.ProvenanceSummary");
        StringAssert.Contains(source, "ExportSummary.OutputDirectoryDisplay");
        StringAssert.Contains(source, "ExportSummary.OutputFileNames");
        StringAssert.Contains(source, "OpenExportPreviewCommand");
        StringAssert.Contains(source, "ConfirmExportCommand");
        StringAssert.Contains(source, "Series.Toggle");
        StringAssert.Contains(source, "Inspector.Toggle");
    }

    [TestMethod]
    public void ApplicationIconContainsEveryApprovedResolution()
    {
        string projectPath = Path.Combine(AppDirectory, "GraphReader.App.csproj");
        XDocument project = XDocument.Load(projectPath);
        Assert.AreEqual(
            "Resources\\Identity\\GraphAutoReader.ico",
            project.Descendants("ApplicationIcon").Single().Value.Trim());

        string iconPath = Path.Combine(AppDirectory, "Resources", "Identity", "GraphAutoReader.ico");
        using var reader = new BinaryReader(File.OpenRead(iconPath));
        Assert.AreEqual((ushort)0, reader.ReadUInt16());
        Assert.AreEqual((ushort)1, reader.ReadUInt16());
        int count = reader.ReadUInt16();
        var sizes = new List<int>(count);
        for (int index = 0; index < count; index++)
        {
            byte width = reader.ReadByte();
            byte height = reader.ReadByte();
            sizes.Add(width == 0 ? 256 : width);
            Assert.AreEqual(width, height);
            _ = reader.ReadBytes(14);
        }

        CollectionAssert.AreEqual(
            ExpectedIconSizes,
            sizes);
    }

    [TestMethod]
    public void DecorativePopupAnimationsAreDisabled()
    {
        string shell = File.ReadAllText(Path.Combine(AppDirectory, "MainWindow.xaml"));
        string controls = File.ReadAllText(Path.Combine(AppDirectory, "Themes", "Controls.xaml"));
        Assert.IsFalse(shell.Contains("PopupAnimation=\"Fade\"", StringComparison.Ordinal));
        Assert.IsFalse(controls.Contains("PopupAnimation=\"Fade\"", StringComparison.Ordinal));
        Assert.IsFalse(shell.Contains("Storyboard", StringComparison.Ordinal));
        Assert.IsFalse(controls.Contains("Storyboard", StringComparison.Ordinal));
    }

    [TestMethod]
    public void SystemThemeUsesWindowsBrushesForHighContrastRoles()
    {
        string systemTheme = File.ReadAllText(Path.Combine(AppDirectory, "Themes", "SystemTheme.xaml"));
        string[] requiredSystemColors =
        [
            "SystemColors.WindowColorKey",
            "SystemColors.WindowTextColorKey",
            "SystemColors.HighlightColorKey",
            "SystemColors.HighlightTextColorKey",
            "SystemColors.GrayTextColorKey",
        ];
        foreach (string systemColor in requiredSystemColors)
        {
            StringAssert.Contains(systemTheme, systemColor);
        }
    }

    private static string AppDirectory => Path.Combine(
        RepositoryTestPaths.Root,
        "src",
        "GraphReader.App");

    private static XDocument LoadAppXaml(params string[] segments) =>
        XDocument.Load(Path.Combine([AppDirectory, .. segments]));
}
