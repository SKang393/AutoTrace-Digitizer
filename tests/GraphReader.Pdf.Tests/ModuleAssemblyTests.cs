// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Pdf.Tests;

[TestClass]
public sealed class ModuleAssemblyTests
{
    [TestMethod]
    public void PdfAssemblyLoads() =>
        Assert.AreEqual("GraphReader.Pdf", Assembly.Load("GraphReader.Pdf").GetName().Name);
}
