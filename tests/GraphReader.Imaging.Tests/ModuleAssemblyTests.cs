// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Imaging.Tests;

[TestClass]
public sealed class ModuleAssemblyTests
{
    [TestMethod]
    public void ImagingAssemblyLoads() =>
        Assert.AreEqual("GraphReader.Imaging", Assembly.Load("GraphReader.Imaging").GetName().Name);
}
