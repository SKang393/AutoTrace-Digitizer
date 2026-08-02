// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Ocr.Tests;

[TestClass]
public sealed class ModuleAssemblyTests
{
    [TestMethod]
    public void OcrAssemblyLoads() =>
        Assert.AreEqual("GraphReader.Ocr", Assembly.Load("GraphReader.Ocr").GetName().Name);
}
