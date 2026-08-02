// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Inference.Tests;

[TestClass]
public sealed class ModuleAssemblyTests
{
    [TestMethod]
    public void InferenceAssemblyLoads() =>
        Assert.AreEqual("GraphReader.Inference", Assembly.Load("GraphReader.Inference").GetName().Name);
}
