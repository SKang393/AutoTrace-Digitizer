// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Windows;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class ArchitectureAcceptanceTests
{
    private static readonly string[] ForbiddenTerms =
    {
        "WebView",
        "WebView2",
        "Electron",
        "Blazor",
        "Microsoft.AspNetCore",
        "HttpListener",
        "Kestrel",
        "AngleSharp",
        "CefSharp",
    };

    [TestMethod]
    public void ApplicationAssemblyIsWindowsNativeWpf()
    {
        var assembly = typeof(MainWindow).Assembly;

        Assert.AreEqual("GraphReader.App", assembly.GetName().Name);
        Assert.IsTrue(typeof(Window).IsAssignableFrom(typeof(MainWindow)));
        Assert.IsFalse(
            assembly.GetReferencedAssemblies().Any(
                reference => ForbiddenTerms.Any(
                    term => reference.Name?.Contains(term, StringComparison.OrdinalIgnoreCase) is true)));
    }

    [TestMethod]
    public void ProjectContainsNoForbiddenWebOrInferenceDependency()
    {
        var projectRoot = Path.Combine(RepositoryTestPaths.Root, "src", "GraphReader.App");
        var inspectedFiles = Directory.EnumerateFiles(projectRoot, "*", SearchOption.AllDirectories)
            .Where(path =>
                string.Equals(Path.GetExtension(path), ".cs", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(Path.GetExtension(path), ".xaml", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(Path.GetExtension(path), ".csproj", StringComparison.OrdinalIgnoreCase))
            .Where(path => !path.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase))
            .Where(path => !path.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase))
            .ToArray();

        Assert.IsNotEmpty(inspectedFiles);
        foreach (var path in inspectedFiles)
        {
            var content = File.ReadAllText(path);
            foreach (var forbiddenTerm in ForbiddenTerms)
            {
                Assert.IsFalse(
                    content.Contains(forbiddenTerm, StringComparison.OrdinalIgnoreCase),
                    $"{Path.GetRelativePath(RepositoryTestPaths.Root, path)} contains forbidden term '{forbiddenTerm}'.");
            }
        }

        var projectText = File.ReadAllText(Path.Combine(projectRoot, "GraphReader.App.csproj"));
        StringAssert.Contains(projectText, "<UseWPF>true</UseWPF>");
        StringAssert.Contains(projectText, "<TargetFramework>net10.0-windows</TargetFramework>");
        Assert.IsFalse(projectText.Contains("OpenCv", StringComparison.OrdinalIgnoreCase));
        Assert.IsFalse(projectText.Contains("Onnx", StringComparison.OrdinalIgnoreCase));
        Assert.IsFalse(projectText.Contains("PdfPig", StringComparison.OrdinalIgnoreCase));
        Assert.IsFalse(projectText.Contains("RealEsrgan", StringComparison.OrdinalIgnoreCase));
    }

    [TestMethod]
    public void ApplicationAssemblyReferencesApprovedManualAndProductionIntegrationModules()
    {
        string[] requiredManualAssemblyPrefixes =
        {
            "GraphReader.Imaging",
            "GraphReader.Axis",
            "GraphReader.Phases",
            "GraphReader.Export",
            "GraphReader.SuperResolution",
            "GraphReader.Pdf",
        };

        var referencedAssemblies = typeof(MainWindow).Assembly.GetReferencedAssemblies();
        foreach (string prefix in requiredManualAssemblyPrefixes)
        {
            Assert.IsTrue(
                referencedAssemblies.Any(reference => reference.Name?.StartsWith(prefix, StringComparison.Ordinal) is true),
                $"The application composition must reference real integration module {prefix}.");
        }
    }
}
