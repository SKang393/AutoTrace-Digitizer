// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Xml.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Integration.Tests;

[TestClass]
public sealed class RepositoryFoundationTests
{
    private static readonly string[] ExpectedCSharpHeader =
    [
        "// SPDX-License-Identifier: Apache-2.0",
        "// Copyright 2026 Sungwoo Kang",
    ];

    private static readonly string[] ModuleNames =
    [
        "Domain",
        "Imaging",
        "Pdf",
        "SuperResolution",
        "Axis",
        "Ocr",
        "Markers",
        "Legends",
        "Phases",
        "Export",
        "Inference",
    ];

    private static readonly string[] ForbiddenProjectTokens =
    [
        "Microsoft.NET.Sdk.Web",
        "Microsoft.AspNetCore",
        "WebView",
        "Blazor",
        "Electron",
        "Ultralytics",
        "AGPL",
        "GPL",
        "SSPL",
        "BUSL",
    ];

    private static readonly string[] RequiredSourceProjectNames =
    [
        "GraphReader.App",
        .. ModuleNames.Select(static moduleName => $"GraphReader.{moduleName}"),
    ];

    private static readonly string[] RequiredTestProjectNames =
    [
        "GraphReader.App.Tests",
        "GraphReader.Integration.Tests",
        .. ModuleNames.Select(static moduleName => $"GraphReader.{moduleName}.Tests"),
    ];

    [TestMethod]
    public void RootBuildConfigurationIsDeterministicStrictAndVersioned()
    {
        string root = RepositoryRoot.Find();
        XDocument buildProperties = XDocument.Load(Path.Combine(root, "Directory.Build.props"));
        XDocument packageProperties = XDocument.Load(Path.Combine(root, "Directory.Packages.props"));

        Assert.AreEqual("enable", PropertyValue(buildProperties, "Nullable"));
        Assert.AreEqual("true", PropertyValue(buildProperties, "Deterministic"));
        Assert.AreEqual("true", PropertyValue(buildProperties, "TreatWarningsAsErrors"));
        Assert.AreEqual("true", PropertyValue(buildProperties, "EnableNETAnalyzers"));
        string version = PropertyValue(buildProperties, "Version") ?? string.Empty;
        StringAssert.Matches(version, new System.Text.RegularExpressions.Regex(@"^\d{1,2}\.\d{1,2}\.\d{1,2}$"));
        Assert.AreEqual($"{version}.0", PropertyValue(buildProperties, "AssemblyVersion"));
        Assert.AreEqual($"{version}.0", PropertyValue(buildProperties, "FileVersion"));
        Assert.AreEqual(version, PropertyValue(buildProperties, "InformationalVersion"));
        Assert.AreEqual("true", PropertyValue(packageProperties, "ManagePackageVersionsCentrally"));
    }

    [TestMethod]
    public void SourceAndMatchingTestProjectsExist()
    {
        string root = RepositoryRoot.Find();

        foreach (string moduleName in ModuleNames)
        {
            string sourceProject = Path.Combine(
                root,
                "src",
                $"GraphReader.{moduleName}",
                $"GraphReader.{moduleName}.csproj");
            string testProject = Path.Combine(
                root,
                "tests",
                $"GraphReader.{moduleName}.Tests",
                $"GraphReader.{moduleName}.Tests.csproj");

            Assert.IsTrue(File.Exists(sourceProject), $"Missing source project: {sourceProject}");
            Assert.IsTrue(File.Exists(testProject), $"Missing test project: {testProject}");
        }

        Assert.IsTrue(File.Exists(Path.Combine(
            root,
            "tests",
            "GraphReader.Integration.Tests",
            "GraphReader.Integration.Tests.csproj")));
    }

    [TestMethod]
    public void ApplicationProjectIsWindowsNativeWpf()
    {
        string projectPath = Path.Combine(
            RepositoryRoot.Find(),
            "src",
            "GraphReader.App",
            "GraphReader.App.csproj");
        XDocument project = XDocument.Load(projectPath);
        string sdk = project.Root?.Attribute("Sdk")?.Value ?? string.Empty;

        Assert.AreEqual("Microsoft.NET.Sdk", sdk);
        Assert.AreEqual("net10.0-windows", PropertyValue(project, "TargetFramework"));
        Assert.AreEqual("WinExe", PropertyValue(project, "OutputType"));
        Assert.AreEqual("true", PropertyValue(project, "UseWPF"));
        Assert.IsFalse(project.ToString().Contains("WebView", StringComparison.OrdinalIgnoreCase));
        Assert.IsFalse(project.ToString().Contains("Blazor", StringComparison.OrdinalIgnoreCase));
        Assert.IsFalse(project.ToString().Contains("Microsoft.NET.Sdk.Web", StringComparison.OrdinalIgnoreCase));
    }

    [TestMethod]
    public void ProjectDefinitionsContainNoForbiddenPlatformOrLicenseDependencies()
    {
        string root = RepositoryRoot.Find();
        string[] projectFiles = Directory.GetFiles(Path.Combine(root, "src"), "*.csproj", SearchOption.AllDirectories)
            .Concat(Directory.GetFiles(Path.Combine(root, "tests"), "*.csproj", SearchOption.AllDirectories))
            .OrderBy(static path => path, StringComparer.Ordinal)
            .ToArray();
        string[] projectDefinitionFiles = projectFiles
            .Append(Path.Combine(root, "Directory.Build.props"))
            .Append(Path.Combine(root, "Directory.Packages.props"))
            .ToArray();

        AssertRequiredProjectMembership(
            projectFiles,
            RequiredSourceProjectNames,
            "production");
        AssertRequiredProjectMembership(
            projectFiles,
            RequiredTestProjectNames,
            "test");

        foreach (string projectFile in projectDefinitionFiles)
        {
            string projectText = File.ReadAllText(projectFile);
            foreach (string forbiddenToken in ForbiddenProjectTokens)
            {
                Assert.IsFalse(
                    projectText.Contains(forbiddenToken, StringComparison.OrdinalIgnoreCase),
                    $"{projectFile} contains forbidden project token '{forbiddenToken}'.");
            }
        }
    }

    [TestMethod]
    public void TestProjectsUseCentralPackageVersions()
    {
        string testsDirectory = Path.Combine(RepositoryRoot.Find(), "tests");
        string[] projectFiles = Directory.GetFiles(testsDirectory, "*.csproj", SearchOption.AllDirectories);

        AssertRequiredProjectMembership(
            projectFiles,
            RequiredTestProjectNames,
            "test");

        foreach (string projectFile in projectFiles)
        {
            XDocument project = XDocument.Load(projectFile);
            foreach (XElement packageReference in project.Descendants()
                         .Where(static element => element.Name.LocalName == "PackageReference"))
            {
                Assert.IsNull(
                    packageReference.Attribute("Version"),
                    $"Use Directory.Packages.props instead of a Version attribute in {projectFile}.");
                Assert.IsFalse(
                    packageReference.Elements().Any(static element => element.Name.LocalName == "Version"),
                    $"Use Directory.Packages.props instead of a Version element in {projectFile}.");
            }
        }
    }

    [TestMethod]
    public void OriginalCSharpFilesHaveApacheSpdxHeaders()
    {
        string root = RepositoryRoot.Find();
        string[] csharpFiles = Directory.GetFiles(Path.Combine(root, "src"), "*.cs", SearchOption.AllDirectories)
            .Concat(Directory.GetFiles(Path.Combine(root, "tests"), "*.cs", SearchOption.AllDirectories))
            .Where(static path => !HasPathSegment(path, "bin") && !HasPathSegment(path, "obj"))
            .OrderBy(static path => path, StringComparer.Ordinal)
            .ToArray();

        Assert.IsTrue(csharpFiles.Length > 0);

        foreach (string csharpFile in csharpFiles)
        {
            string[] firstLines = File.ReadLines(csharpFile).Take(2).ToArray();
            CollectionAssert.AreEqual(
                ExpectedCSharpHeader,
                firstLines,
                csharpFile);
        }
    }

    [TestMethod]
    public void PublicLicenseAndNoticeFilesIdentifyApacheLicense()
    {
        string root = RepositoryRoot.Find();
        string license = File.ReadAllText(Path.Combine(root, "LICENSE"));
        string notice = File.ReadAllText(Path.Combine(root, "NOTICE"));
        string thirdPartyNotices = File.ReadAllText(Path.Combine(root, "THIRD_PARTY_NOTICES.md"));

        StringAssert.Contains(license, "Apache License");
        StringAssert.Contains(license, "Version 2.0");
        StringAssert.Contains(notice, "Copyright 2026 Sungwoo Kang");
        StringAssert.Contains(thirdPartyNotices, "Third-Party");
    }

    private static string? PropertyValue(XDocument project, string propertyName) =>
        project.Descendants()
            .FirstOrDefault(element => element.Name.LocalName == propertyName)
            ?.Value;

    private static bool HasPathSegment(string path, string segment) =>
        path.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            .Contains(segment, StringComparer.OrdinalIgnoreCase);

    private static void AssertRequiredProjectMembership(
        IEnumerable<string> projectFiles,
        IEnumerable<string> requiredProjectNames,
        string category)
    {
        HashSet<string> projectNames = projectFiles
            .Select(Path.GetFileNameWithoutExtension)
            .Where(static projectName => projectName is not null)
            .Select(static projectName => projectName!)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        foreach (string requiredProjectName in requiredProjectNames)
        {
            Assert.IsTrue(
                projectNames.Contains(requiredProjectName),
                $"Missing required {category} project: {requiredProjectName}");
        }
    }
}
