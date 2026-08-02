// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests;

[TestClass]
public sealed class ModuleAssemblyTests
{
    [TestMethod]
    public void MarkersAssemblyLoads() =>
        Assert.AreEqual("GraphReader.Markers", Assembly.Load("GraphReader.Markers").GetName().Name);
}
