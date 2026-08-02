// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Axis.Tests;

[TestClass]
public sealed class ModuleAssemblyTests
{
    [TestMethod]
    public void AxisAssemblyLoads() =>
        Assert.AreEqual("GraphReader.Axis", Assembly.Load("GraphReader.Axis").GetName().Name);
}
