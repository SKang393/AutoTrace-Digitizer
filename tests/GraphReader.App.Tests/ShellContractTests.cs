// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Xml.Linq;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class ShellContractTests
{
    private const string PresentationNamespace = "http://schemas.microsoft.com/winfx/2006/xaml/presentation";
    private const string XamlNamespace = "http://schemas.microsoft.com/winfx/2006/xaml";
    private const string AutomationNamespace = "clr-namespace:System.Windows.Automation;assembly=PresentationCore";

    private static readonly Dictionary<string, (string AutomationId, string Command)> WorkflowControls =
        new Dictionary<string, (string, string)>(StringComparer.Ordinal)
        {
            ["WorkflowImportButton"] = ("Workflow.Import", "{Binding ImportCommand}"),
            ["WorkflowEnhanceButton"] = ("Workflow.Enhance", "{Binding EnhanceCommand}"),
            ["WorkflowAutoDetectButton"] = ("Workflow.AutoDetect", "{Binding AutoDetectCommand}"),
            ["WorkflowReviewButton"] = ("Workflow.Review", "{Binding ReviewCommand}"),
            ["WorkflowExportButton"] = ("Workflow.Export", "{Binding ExportCommand}"),
        };

    [TestMethod]
    public void CoreWorkflowIsVisibleLocalizedAndAccessibleWithoutMenus()
    {
        var document = LoadMainWindow();
        XNamespace presentation = PresentationNamespace;
        XNamespace xaml = XamlNamespace;
        XNamespace automation = AutomationNamespace;

        Assert.IsFalse(document.Descendants(presentation + "Menu").Any());
        var tabIndices = new HashSet<string>(StringComparer.Ordinal);
        foreach (var expected in WorkflowControls)
        {
            var button = FindNamed(document, expected.Key);
            Assert.AreEqual(presentation + "Button", button.Name);
            Assert.AreEqual(expected.Value.AutomationId, button.Attribute(automation + "AutomationProperties.AutomationId")?.Value);
            Assert.AreEqual($"{{DynamicResource {expected.Value.AutomationId}}}", button.Attribute(automation + "AutomationProperties.Name")?.Value);
            Assert.AreEqual(expected.Value.Command, button.Attribute("Command")?.Value);
            Assert.IsTrue(tabIndices.Add(button.Attribute("TabIndex")?.Value ?? string.Empty));
            Assert.IsTrue(button.Descendants(presentation + "AccessText").Any());
        }

        Assert.HasCount(WorkflowControls.Count, tabIndices);
        Assert.IsNotNull(document.Root?.Attribute(xaml + "Class"));
    }

    [TestMethod]
    public void KeyboardBindingsCoverWorkflowOverlayAndCancellation()
    {
        var document = LoadMainWindow();
        XNamespace presentation = PresentationNamespace;
        var bindings = document.Descendants(presentation + "KeyBinding")
            .Select(binding => new
            {
                Key = binding.Attribute("Key")?.Value ?? string.Empty,
                Modifiers = binding.Attribute("Modifiers")?.Value ?? string.Empty,
                Command = binding.Attribute("Command")?.Value ?? string.Empty,
            })
            .ToArray();

        var expected = new[]
        {
            ("I", "Control", "{Binding ImportCommand}"),
            ("E", "Control", "{Binding EnhanceCommand}"),
            ("F5", string.Empty, "{Binding AutoDetectCommand}"),
            ("R", "Control", "{Binding ReviewCommand}"),
            ("E", "Control+Shift", "{Binding ExportCommand}"),
            ("P", "Control", "{Binding TogglePhaseOverlayCommand}"),
            ("Escape", string.Empty, "{Binding CancelCommand}"),
        };

        Assert.HasCount(expected.Length, bindings);
        foreach (var item in expected)
        {
            Assert.IsTrue(
                bindings.Any(binding =>
                    binding.Key == item.Item1 &&
                    binding.Modifiers == item.Item2 &&
                    binding.Command == item.Item3),
                $"Missing keyboard binding {item.Item2}+{item.Item1} -> {item.Item3}.");
        }
    }

    [TestMethod]
    public void MagnifierAndSeriesRemainInRightInspector()
    {
        var document = LoadMainWindow();
        XNamespace automation = AutomationNamespace;

        var inspector = FindNamed(document, "InspectorPane");
        var magnifier = FindNamed(document, "MagnifierInspector");
        var series = FindNamed(document, "SeriesList");
        var canvas = FindNamed(document, "GraphCanvasHost");

        Assert.IsTrue(magnifier.AncestorsAndSelf().Contains(inspector));
        Assert.IsTrue(series.AncestorsAndSelf().Contains(inspector));
        Assert.IsFalse(canvas.AncestorsAndSelf().Contains(inspector));
        Assert.AreEqual("Inspector.Magnifier", magnifier.Attribute(automation + "AutomationProperties.AutomationId")?.Value);
        Assert.AreEqual("Inspector.Series", series.Attribute(automation + "AutomationProperties.AutomationId")?.Value);
        Assert.AreEqual("Canvas.GraphHost", canvas.Attribute(automation + "AutomationProperties.AutomationId")?.Value);
        Assert.AreEqual(
            "{Binding Magnifier.NearestDetectionName}",
            magnifier.Attribute("NearestDetectionName")?.Value,
            "The nearest-detection label must come from fake detection data so the empty placeholder remains reachable.");
    }

    [TestMethod]
    public void ThemeSelectorIsVisibleLocalizedAccessibleAndBound()
    {
        var document = LoadMainWindow();
        XNamespace presentation = PresentationNamespace;
        XNamespace automation = AutomationNamespace;
        var selector = FindNamed(document, "ThemeSelector");

        Assert.AreEqual(presentation + "ComboBox", selector.Name);
        Assert.AreEqual("5", selector.Attribute("TabIndex")?.Value);
        Assert.AreEqual("{Binding AppearanceMode, Mode=TwoWay}", selector.Attribute("SelectedValue")?.Value);
        Assert.AreEqual("Theme.Select", selector.Attribute(automation + "AutomationProperties.AutomationId")?.Value);
        Assert.AreEqual(
            "{DynamicResource Theme.Select.AutomationName}",
            selector.Attribute(automation + "AutomationProperties.Name")?.Value);

        var items = selector.Elements(presentation + "ComboBoxItem").ToArray();
        Assert.HasCount(3, items);
        Assert.IsTrue(items.All(item =>
            (item.Attribute("Content")?.Value ?? string.Empty)
                .StartsWith("{DynamicResource Theme.", StringComparison.Ordinal)));
    }

    [TestMethod]
    public void VisibleShellLiteralsUseLocalizationResources()
    {
        var document = LoadMainWindow();
        XNamespace automation = AutomationNamespace;
        var visibleAttributes = new HashSet<XName>
        {
            "Title",
            "Text",
            "Content",
            "Header",
            "ToolTip",
            automation + "AutomationProperties.Name",
        };

        var literalAttributes = (document.Root?.DescendantsAndSelf() ?? Enumerable.Empty<XElement>())
            .Attributes()
            .Where(attribute => visibleAttributes.Contains(attribute.Name))
            .Where(attribute => !string.IsNullOrWhiteSpace(attribute.Value))
            .Where(attribute => !attribute.Value.StartsWith("{DynamicResource ", StringComparison.Ordinal))
            .Where(attribute => !attribute.Value.StartsWith("{Binding ", StringComparison.Ordinal))
            .ToArray();

        Assert.IsEmpty(
            literalAttributes,
            string.Join(", ", literalAttributes.Select(attribute => $"{attribute.Parent?.Name.LocalName}.{attribute.Name.LocalName}='{attribute.Value}'")));
    }

    private static XDocument LoadMainWindow()
    {
        return XDocument.Load(Path.Combine(RepositoryTestPaths.Root, "src", "GraphReader.App", "MainWindow.xaml"), LoadOptions.SetLineInfo);
    }

    private static XElement FindNamed(XDocument document, string name)
    {
        XNamespace xaml = XamlNamespace;
        return document.Descendants()
            .Single(element => string.Equals(element.Attribute(xaml + "Name")?.Value, name, StringComparison.Ordinal));
    }
}
