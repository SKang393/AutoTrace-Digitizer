// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
using System.Xml.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Integration.Tests;

[TestClass]
public sealed class PackagingContractTests
{
    [TestMethod]
    public void ArtifactDefinitionsRequireInstallerAndPortableFromCommonPublish()
    {
        string root = RepositoryRoot.Find();
        string packagingDirectory = Path.Combine(root, "packaging");
        string artifactDefinition = Path.Combine(packagingDirectory, "artifacts.json");

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(artifactDefinition));
        JsonElement definition = document.RootElement;

        Assert.AreEqual(2, definition.GetProperty("schemaVersion").GetInt32());
        Assert.AreEqual(
            "Directory.Build.props#Project/PropertyGroup/Version",
            definition.GetProperty("versionSource").GetString());
        string rid = definition.GetProperty("rid").GetString() ?? string.Empty;
        Assert.AreEqual("win-x64", rid);
        Assert.IsFalse(string.IsNullOrWhiteSpace(definition.GetProperty("commonPublish").GetString()));

        string version = ReadProjectVersion(root);
        AssertArtifactDefinition(
            packagingDirectory,
            definition.GetProperty("installer"),
            version,
            rid,
            "installer/installer.json",
            "GraphAutoReader-{version}-{rid}-setup.exe",
            $"GraphAutoReader-{version}-win-x64-setup.exe");
        AssertArtifactDefinition(
            packagingDirectory,
            definition.GetProperty("portable"),
            version,
            rid,
            "portable/portable.json",
            "GraphAutoReader-{version}-{rid}-portable.zip",
            $"GraphAutoReader-{version}-win-x64-portable.zip");

        AssertSkeletonDirectory(Path.Combine(packagingDirectory, "common"));
        AssertSkeletonDirectory(Path.Combine(packagingDirectory, "installer"));
        AssertSkeletonDirectory(Path.Combine(packagingDirectory, "portable"));
    }

    private static void AssertArtifactDefinition(
        string packagingDirectory,
        JsonElement artifact,
        string version,
        string rid,
        string expectedDefinitionPath,
        string expectedTemplate,
        string expectedFileName)
    {
        Assert.AreEqual(
            expectedDefinitionPath,
            artifact.GetProperty("definition").GetString());
        string fileNameTemplate = artifact.GetProperty("fileNameTemplate").GetString() ?? string.Empty;
        Assert.AreEqual(expectedTemplate, fileNameTemplate);

        string renderedFileName = fileNameTemplate
            .Replace("{version}", version, StringComparison.Ordinal)
            .Replace("{rid}", rid, StringComparison.Ordinal);
        Assert.AreEqual(expectedFileName, renderedFileName);

        string definitionPath = Path.Combine(
            packagingDirectory,
            expectedDefinitionPath.Replace('/', Path.DirectorySeparatorChar));
        Assert.IsTrue(File.Exists(definitionPath), $"Missing artifact definition: {definitionPath}");

        using JsonDocument artifactDocument = JsonDocument.Parse(File.ReadAllText(definitionPath));
        Assert.AreEqual(2, artifactDocument.RootElement.GetProperty("schemaVersion").GetInt32());
        Assert.IsTrue(artifactDocument.RootElement.GetProperty("commonPublishOnly").GetBoolean());
    }

    private static string ReadProjectVersion(string root)
    {
        XDocument buildProperties = XDocument.Load(Path.Combine(root, "Directory.Build.props"));
        string? version = buildProperties.Descendants()
            .FirstOrDefault(static element => element.Name.LocalName == "Version")
            ?.Value;

        Assert.IsFalse(string.IsNullOrWhiteSpace(version), "Directory.Build.props must define Version.");
        return version!;
    }

    private static void AssertSkeletonDirectory(string path)
    {
        Assert.IsTrue(Directory.Exists(path), $"Missing packaging directory: {path}");
        Assert.IsTrue(
            Directory.EnumerateFileSystemEntries(path).Any(),
            $"Packaging skeleton has no definition files: {path}");
    }
}
