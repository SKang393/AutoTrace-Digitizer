// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class StartupArgumentsTests
{
    [TestMethod]
    public void ParseAcceptsPortableSmokeAndAbsoluteImagePath()
    {
        string imagePath = Path.GetFullPath(Path.Combine("images", "Chandler graph.png"));

        StartupArguments result = StartupArguments.Parse(
            ["--portable-smoke", "--open-image", imagePath]);

        Assert.IsTrue(result.PortableSmoke);
        Assert.AreEqual(imagePath, result.OpenImagePath);
    }

    [TestMethod]
    public void ParseRejectsMissingImagePath()
    {
        Assert.ThrowsExactly<ArgumentException>(
            () => StartupArguments.Parse(["--open-image"]));
    }

    [TestMethod]
    public void ParseRejectsDuplicateImageArguments()
    {
        string imagePath = Path.GetFullPath("graph.png");

        Assert.ThrowsExactly<ArgumentException>(
            () => StartupArguments.Parse(
                ["--open-image", imagePath, "--open-image", imagePath]));
    }

    [TestMethod]
    public void ParseRejectsRelativeImagePath()
    {
        Assert.ThrowsExactly<ArgumentException>(
            () => StartupArguments.Parse(["--open-image", "graph.png"]));
    }
}
