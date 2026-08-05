// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Reflection;
using System.Reflection.Emit;
using System.Windows;
using GraphReader.App.Integration;
using GraphReader.App.Integration.Workflow;

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
    public void OrdinaryApplicationBuildIsBoundToManualPreviewAndCannotSelectRecordedFake()
    {
        Assembly assembly = typeof(MainWindow).Assembly;
        AssemblyMetadataAttribute[] runtimeModes = assembly
            .GetCustomAttributes<AssemblyMetadataAttribute>()
            .Where(attribute => string.Equals(
                attribute.Key,
                RuntimeModeSelector.BuildMetadataKey,
                StringComparison.Ordinal))
            .ToArray();

        Assert.HasCount(1, runtimeModes);
        Assert.AreEqual(nameof(WorkflowRuntimeEnvironment.ManualPreview), runtimeModes[0].Value);
        Assert.AreEqual(
            WorkflowRuntimeEnvironment.ManualPreview,
            RuntimeModeSelector.SelectBuildDefault(assembly));
        Assert.AreEqual(
            WorkflowRuntimeEnvironment.ManualPreview,
            RuntimeModeSelector.Select(nameof(WorkflowRuntimeEnvironment.RecordedFake), assembly));
    }

    [TestMethod]
    public void ProductionBuildMetadataCannotBeDowngradedByRuntimeConfiguration()
    {
        AssemblyBuilder productionAssembly = AssemblyBuilder.DefineDynamicAssembly(
            new AssemblyName("GraphReader.App.ProductionModeTest"),
            AssemblyBuilderAccess.Run);
        ConstructorInfo metadataConstructor = typeof(AssemblyMetadataAttribute)
            .GetConstructor([typeof(string), typeof(string)])!;
        productionAssembly.SetCustomAttribute(new CustomAttributeBuilder(
            metadataConstructor,
            [RuntimeModeSelector.BuildMetadataKey, nameof(WorkflowRuntimeEnvironment.Production)]));

        Assert.AreEqual(
            WorkflowRuntimeEnvironment.Production,
            RuntimeModeSelector.SelectBuildDefault(productionAssembly));
        Assert.AreEqual(
            WorkflowRuntimeEnvironment.Production,
            RuntimeModeSelector.Select(nameof(WorkflowRuntimeEnvironment.ManualPreview), productionAssembly));
        Assert.AreEqual(
            WorkflowRuntimeEnvironment.Production,
            RuntimeModeSelector.Select(nameof(WorkflowRuntimeEnvironment.RecordedFake), productionAssembly));
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

    [TestMethod]
    public void DistributionBuildsProbeTheirCompiledRuntimeModes()
    {
        string packagingRoot = Path.Combine(RepositoryTestPaths.Root, "packaging");
        string releaseBuild = File.ReadAllText(Path.Combine(packagingRoot, "Build-Windows.ps1"));
        string developmentBuild = File.ReadAllText(Path.Combine(packagingRoot, "Build-DevPortable.ps1"));

        StringAssert.Contains(releaseBuild, "-p:GraphReaderRuntimeMode=Production");
        StringAssert.Contains(releaseBuild, "--production-runtime-smoke");
        StringAssert.Contains(releaseBuild, "productionRuntimeSmokeExitCode");
        StringAssert.Contains(developmentBuild, "--production-runtime-smoke");
        StringAssert.Contains(developmentBuild, "productionRuntimeSmokeProcess.ExitCode -ne 2");
        Assert.IsFalse(
            developmentBuild.Contains(
                "-p:GraphReaderRuntimeMode=Production",
                StringComparison.Ordinal));
    }
}
