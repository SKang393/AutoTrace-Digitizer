// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Windows;
using System.Windows.Automation.Peers;
using System.Xml.Linq;
using GraphReader.App.Appearance;
using GraphReader.App.Controls;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class ThemeAccessibilityTests
{
    private const string PresentationNamespace = "http://schemas.microsoft.com/winfx/2006/xaml/presentation";
    private const string XamlNamespace = "http://schemas.microsoft.com/winfx/2006/xaml";
    private const string AutomationNamespace = "clr-namespace:System.Windows.Automation;assembly=PresentationCore";

    [TestMethod]
    public void HighContrastUsesSystemPaletteAndReducedMotionUsesZeroDuration()
    {
        StaTestHost.Run(
            () =>
            {
                Application.ResourceAssembly ??= typeof(ThemeService).Assembly;
                var resources = new ResourceDictionary();
                using var systemTheme = new FakeSystemThemeProvider(
                    ApplicationTheme.Dark,
                    isHighContrast: true,
                    animationsEnabled: false);
                using var themeService = new ThemeService(resources, systemTheme);

                AssertThemeSource(resources, "SystemTheme.xaml");
                Assert.AreEqual(TimeSpan.Zero, ReadMotionDuration(resources));

                systemTheme.SetAccessibility(isHighContrast: false, animationsEnabled: true);

                AssertThemeSource(resources, "DarkTheme.xaml");
                Assert.AreEqual(TimeSpan.FromMilliseconds(140), ReadMotionDuration(resources));
            });
    }

    [TestMethod]
    public void EmptySurfaceAndWorkspaceAreMutuallyExclusiveLayoutSiblings()
    {
        XDocument shell = LoadAppXaml("MainWindow.xaml");
        XNamespace xaml = XamlNamespace;
        XNamespace presentation = PresentationNamespace;
        XElement workspace = shell.Descendants()
            .Single(element => element.Attribute(xaml + "Name")?.Value == "WorkspaceSurface");
        XElement empty = shell.Descendants()
            .Single(element => element.Attribute(xaml + "Name")?.Value == "EmptySurface");

        Assert.AreSame(workspace.Parent, empty.Parent);
        Assert.AreEqual("{StaticResource State.Workspace}", workspace.Attribute("Style")?.Value);
        Assert.AreEqual("{StaticResource State.Empty}", empty.Attribute("Style")?.Value);

        XElement workspaceStyle = shell.Descendants(presentation + "Style")
            .Single(element => element.Attribute(xaml + "Key")?.Value == "State.Workspace");
        XElement emptyTrigger = workspaceStyle.Descendants(presentation + "DataTrigger")
            .Single(element => element.Attribute("Value")?.Value.Contains("WorkspaceSurfaceState.Empty", StringComparison.Ordinal) == true);
        Assert.IsTrue(emptyTrigger.Descendants(presentation + "Setter")
            .Any(element => element.Attribute("Property")?.Value == "Visibility" &&
                element.Attribute("Value")?.Value == "Collapsed"));
    }

    [TestMethod]
    public void ReviewIssuesAndDataRowsExposeNativeVisibleSelection()
    {
        XDocument shell = LoadAppXaml("MainWindow.xaml");
        XDocument controls = LoadAppXaml("Themes", "Controls.xaml");
        XNamespace presentation = PresentationNamespace;
        XNamespace xaml = XamlNamespace;

        XElement review = FindList(shell, "{Binding ReviewIssues}");
        XElement data = FindList(shell, "{Binding DataPreviewRows}");
        Assert.AreEqual("{Binding SelectedReviewIssue, Mode=TwoWay}", review.Attribute("SelectedItem")?.Value);
        Assert.AreEqual("{Binding SelectedDataPreviewRow, Mode=TwoWay}", data.Attribute("SelectedItem")?.Value);
        Assert.AreEqual("Stretch", review.Attribute("HorizontalContentAlignment")?.Value);
        Assert.AreEqual("Disabled", review.Attribute("ScrollViewer.HorizontalScrollBarVisibility")?.Value);
        Assert.AreEqual("Stretch", data.Attribute("HorizontalContentAlignment")?.Value);
        Assert.AreEqual("Disabled", data.Attribute("ScrollViewer.HorizontalScrollBarVisibility")?.Value);

        XElement selectionStyle = controls.Descendants(presentation + "Style")
            .Single(element => element.Attribute(xaml + "Key")?.Value == "App.Style.ListBoxItem");
        XElement selectedTrigger = selectionStyle.Descendants(presentation + "Trigger")
            .Single(element => element.Attribute("Property")?.Value == "IsSelected");
        string selectionMarkup = selectedTrigger.ToString(SaveOptions.DisableFormatting);
        StringAssert.Contains(selectionMarkup, "App.Brush.Selection");
        StringAssert.Contains(selectionMarkup, "App.Brush.SelectionText");
        StringAssert.Contains(selectionMarkup, "App.Brush.Primary");
    }

    [TestMethod]
    public void SelectionTransitionUsesApprovedDurationToken()
    {
        XDocument tokens = LoadAppXaml("Themes", "DesignTokens.xaml");
        XNamespace xaml = XamlNamespace;
        XElement duration = tokens.Descendants()
            .Single(element => element.Attribute(xaml + "Key")?.Value == "App.Motion.Duration.Short");
        TimeSpan parsed = TimeSpan.Parse(duration.Value, System.Globalization.CultureInfo.InvariantCulture);

        Assert.IsGreaterThanOrEqualTo(TimeSpan.FromMilliseconds(100), parsed);
        Assert.IsLessThanOrEqualTo(TimeSpan.FromMilliseconds(180), parsed);

        string controls = File.ReadAllText(Path.Combine(AppDirectory, "Themes", "Controls.xaml"));
        StringAssert.Contains(controls, "Tag\" Value=\"{DynamicResource App.Motion.Enabled}\"");
        StringAssert.Contains(controls, "Duration=\"0:0:0.14\"");
    }

    [TestMethod]
    public void GraphCanvasExposesSemanticPaneAutomationType()
    {
        StaTestHost.Run(
            () =>
            {
                Application.ResourceAssembly ??= typeof(GraphCanvasControl).Assembly;
                var control = new GraphCanvasControl();
                AutomationPeer? peer = UIElementAutomationPeer.CreatePeerForElement(control);

                Assert.IsNotNull(peer);
                Assert.AreEqual(AutomationControlType.Pane, peer.GetAutomationControlType());
                Assert.AreEqual(nameof(GraphCanvasControl), peer.GetClassName());
            });
    }

    [TestMethod]
    public void InteractiveContainersExposeReadableAutomationNames()
    {
        XDocument shell = LoadAppXaml("MainWindow.xaml");
        XNamespace presentation = PresentationNamespace;
        XNamespace xaml = XamlNamespace;
        XNamespace automation = AutomationNamespace;

        XElement tabs = shell.Descendants(presentation + "TabControl")
            .Single(element => element.Attribute(xaml + "Name")?.Value == "GraphTabStrip");
        XElement tabNameSetter = tabs.Descendants(presentation + "Setter")
            .Single(element =>
                element.Attribute("Property")?.Value == "automation:AutomationProperties.Name");
        Assert.AreEqual("{Binding DisplayName}", tabNameSetter.Attribute("Value")?.Value);

        XElement probes = shell.Descendants(presentation + "ItemsControl")
            .Single(element => element.Attribute("ItemsSource")?.Value == "{Binding ProbeRelationChoices}");
        Assert.AreEqual("False", probes.Attribute("Focusable")?.Value);
        Assert.AreEqual("False", probes.Attribute("IsTabStop")?.Value);
        Assert.AreEqual(
            "Manual.ApplicableProbes",
            probes.Attribute(automation + "AutomationProperties.AutomationId")?.Value);
        Assert.AreEqual(
            "{DynamicResource Manual.ApplicableProbes}",
            probes.Attribute(automation + "AutomationProperties.Name")?.Value);
        XElement emptyProbeTrigger = probes.Descendants(presentation + "DataTrigger")
            .Single(element => element.Attribute("Value")?.Value == "0");
        Assert.AreEqual(
            "{Binding ProbeRelationChoices.Count}",
            emptyProbeTrigger.Attribute("Binding")?.Value);
        Assert.IsTrue(emptyProbeTrigger.Descendants(presentation + "Setter")
            .Any(element =>
                element.Attribute("Property")?.Value == "Visibility" &&
                element.Attribute("Value")?.Value == "Collapsed"));

        XElement manualTools = shell.Descendants(presentation + "ScrollViewer")
            .Single(element => element.Descendants(presentation + "TextBlock")
                .Any(text => text.Attribute("Text")?.Value == "{DynamicResource Manual.Title}"));
        Assert.AreEqual(
            "Inspector.ManualTools",
            manualTools.Attribute(automation + "AutomationProperties.AutomationId")?.Value);
        Assert.AreEqual(
            "{DynamicResource Manual.Title}",
            manualTools.Attribute(automation + "AutomationProperties.Name")?.Value);
    }

    private static XElement FindList(XDocument shell, string itemsSource) =>
        shell.Descendants(XName.Get("ListBox", PresentationNamespace))
            .Single(element => element.Attribute("ItemsSource")?.Value == itemsSource);

    private static TimeSpan ReadMotionDuration(ResourceDictionary resources)
    {
        var duration = (Duration)resources["App.Motion.Duration.Short"];
        Assert.IsTrue(duration.HasTimeSpan);
        return duration.TimeSpan;
    }

    private static void AssertThemeSource(ResourceDictionary resources, string expectedFile)
    {
        Assert.HasCount(1, resources.MergedDictionaries);
        Assert.IsTrue(
            (resources.MergedDictionaries[0].Source?.OriginalString ?? string.Empty)
                .EndsWith(expectedFile, StringComparison.OrdinalIgnoreCase));
    }

    private static string AppDirectory => Path.Combine(
        RepositoryTestPaths.Root,
        "src",
        "GraphReader.App");

    private static XDocument LoadAppXaml(params string[] segments) =>
        XDocument.Load(Path.Combine([AppDirectory, .. segments]));

    private sealed class FakeSystemThemeProvider(
        ApplicationTheme theme,
        bool isHighContrast,
        bool animationsEnabled) : ISystemThemeProvider
    {
        public ApplicationTheme EffectiveTheme { get; } = theme;

        public bool IsHighContrast { get; private set; } = isHighContrast;

        public bool AreClientAreaAnimationsEnabled { get; private set; } = animationsEnabled;

        public event EventHandler? ThemeChanged;

        public void SetAccessibility(bool isHighContrast, bool animationsEnabled)
        {
            IsHighContrast = isHighContrast;
            AreClientAreaAnimationsEnabled = animationsEnabled;
            ThemeChanged?.Invoke(this, EventArgs.Empty);
        }

        public void Dispose()
        {
        }
    }
}
