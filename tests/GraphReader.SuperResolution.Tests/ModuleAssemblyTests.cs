// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.SuperResolution.Tests;

[TestClass]
public sealed class ModuleAssemblyTests
{
    [TestMethod]
    public void SuperResolutionAssemblyLoads() =>
        Assert.AreEqual(
            "GraphReader.SuperResolution",
            Assembly.Load("GraphReader.SuperResolution").GetName().Name);
}
