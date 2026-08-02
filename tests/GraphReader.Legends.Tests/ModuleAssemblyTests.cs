// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Legends.Tests;

[TestClass]
public sealed class ModuleAssemblyTests
{
    [TestMethod]
    public void LegendsAssemblyLoads() =>
        Assert.AreEqual("GraphReader.Legends", Assembly.Load("GraphReader.Legends").GetName().Name);
}
