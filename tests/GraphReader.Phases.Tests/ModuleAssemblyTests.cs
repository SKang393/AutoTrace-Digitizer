// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Phases.Tests;

[TestClass]
public sealed class ModuleAssemblyTests
{
    [TestMethod]
    public void PhasesAssemblyLoads() =>
        Assert.AreEqual("GraphReader.Phases", Assembly.Load("GraphReader.Phases").GetName().Name);
}
