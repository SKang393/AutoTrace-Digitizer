// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Export.Tests;

[TestClass]
public sealed class ModuleAssemblyTests
{
    [TestMethod]
    public void ExportAssemblyLoads() =>
        Assert.AreEqual("GraphReader.Export", Assembly.Load("GraphReader.Export").GetName().Name);
}
