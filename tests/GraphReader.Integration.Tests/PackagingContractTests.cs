// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
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

        Assert.AreEqual("0.0.1", definition.GetProperty("version").GetString());
        Assert.AreEqual("win-x64", definition.GetProperty("rid").GetString());
        Assert.IsFalse(string.IsNullOrWhiteSpace(definition.GetProperty("commonPublish").GetString()));
        Assert.AreEqual(
            "GraphAutoReader-0.0.1-win-x64-setup.exe",
            definition.GetProperty("installer").GetProperty("fileName").GetString());
        Assert.AreEqual(
            "GraphAutoReader-0.0.1-win-x64-portable.zip",
            definition.GetProperty("portable").GetProperty("fileName").GetString());

        AssertSkeletonDirectory(Path.Combine(packagingDirectory, "common"));
        AssertSkeletonDirectory(Path.Combine(packagingDirectory, "installer"));
        AssertSkeletonDirectory(Path.Combine(packagingDirectory, "portable"));
    }

    private static void AssertSkeletonDirectory(string path)
    {
        Assert.IsTrue(Directory.Exists(path), $"Missing packaging directory: {path}");
        Assert.IsTrue(
            Directory.EnumerateFileSystemEntries(path).Any(),
            $"Packaging skeleton has no definition files: {path}");
    }
}
