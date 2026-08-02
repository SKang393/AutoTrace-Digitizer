// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Domain;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Domain.Tests;

[TestClass]
public sealed class ApplicationPathsTests
{
    private string _temporaryRoot = null!;

    [TestInitialize]
    public void CreateTemporaryRoot()
    {
        _temporaryRoot = Path.Combine(
            Path.GetTempPath(),
            "GraphReader.Tests",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_temporaryRoot);
    }

    [TestCleanup]
    public void DeleteTemporaryRoot()
    {
        if (Directory.Exists(_temporaryRoot))
        {
            Directory.Delete(_temporaryRoot, recursive: true);
        }
    }

    [TestMethod]
    public void CreateWithoutPortableSentinelUsesInstalledRoots()
    {
        string executableDirectory = Path.Combine(_temporaryRoot, "Application");
        string localApplicationData = Path.Combine(_temporaryRoot, "LocalAppData");
        Directory.CreateDirectory(executableDirectory);

        ApplicationPaths paths = ApplicationPaths.Create(executableDirectory, localApplicationData);
        string applicationRoot = Path.Combine(localApplicationData, "GraphAutoReader");

        Assert.AreEqual(DistributionMode.Installed, paths.Mode);
        Assert.AreEqual(Path.Combine(applicationRoot, "Settings"), paths.SettingsRoot);
        Assert.AreEqual(Path.Combine(applicationRoot, "Cache"), paths.CacheRoot);
        Assert.AreEqual(Path.Combine(applicationRoot, "Logs"), paths.LogsRoot);
        Assert.AreEqual(Path.Combine(applicationRoot, "Autosave"), paths.AutosaveRoot);
        Assert.AreEqual(Path.Combine(applicationRoot, "Recovery"), paths.RecoveryRoot);
        Assert.AreEqual(Path.Combine(executableDirectory, "models"), paths.ModelRoot);
        Assert.IsFalse(Directory.Exists(applicationRoot), "Resolving paths must not create storage.");
    }

    [TestMethod]
    public void CreateWithPortableSentinelUsesPortableDataRoots()
    {
        string executableDirectory = Path.Combine(_temporaryRoot, "Portable Application");
        Directory.CreateDirectory(executableDirectory);
        File.WriteAllText(Path.Combine(executableDirectory, "portable.mode"), string.Empty);

        ApplicationPaths paths = ApplicationPaths.Create(
            executableDirectory,
            Path.Combine(_temporaryRoot, "UnusedLocalAppData"));
        string dataRoot = Path.Combine(executableDirectory, "Data");

        Assert.AreEqual(DistributionMode.Portable, paths.Mode);
        Assert.AreEqual(Path.Combine(dataRoot, "Settings"), paths.SettingsRoot);
        Assert.AreEqual(Path.Combine(dataRoot, "Cache"), paths.CacheRoot);
        Assert.AreEqual(Path.Combine(dataRoot, "Logs"), paths.LogsRoot);
        Assert.AreEqual(Path.Combine(dataRoot, "Autosave"), paths.AutosaveRoot);
        Assert.AreEqual(Path.Combine(dataRoot, "Recovery"), paths.RecoveryRoot);
        Assert.AreEqual(Path.Combine(executableDirectory, "models"), paths.ModelRoot);
        Assert.IsFalse(Directory.Exists(dataRoot), "Resolving paths must not create storage.");
    }
}
