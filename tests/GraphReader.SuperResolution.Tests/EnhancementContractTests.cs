// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.SuperResolution.Tests;

[TestClass]
public sealed class EnhancementContractTests
{
    private static readonly double[] ExpectedForwardMatrix =
        [2d, 0d, 0d, 0d, 2d, 0d, 0d, 0d, 1d];

    private static readonly double[] ExpectedInverseMatrix =
        [0.5d, 0d, 0d, 0d, 0.5d, 0d, 0d, 0d, 1d];

    private static readonly EnhancementProvider[] ExpectedProviders =
        [EnhancementProvider.Vulkan];

    [TestMethod]
    public void DefaultsSelectVisibleReversibleTwoXEnhancement()
    {
        var options = new EnhancementOptions();
        Assert.AreEqual(2, options.Scale);
        Assert.IsTrue(options.ContinueWithoutEnhancement);
        Assert.IsFalse(options.RequestCpuFallback);
    }

    [TestMethod]
    public void ScaleTransformRoundTripsOriginalAndEnhancedCoordinates()
    {
        EnhancementTransform transform = EnhancementTransform.CreateScale2();
        var original = new EnhancementPoint(17.25, 31.5);

        EnhancementPoint enhanced = transform.ToEnhanced(original);
        EnhancementPoint roundTrip = transform.ToOriginal(enhanced);

        Assert.AreEqual("original_pixels", transform.SourceSpace);
        Assert.AreEqual("enhanced_pixels", transform.TargetSpace);
        Assert.AreNotEqual(Guid.Empty, transform.TransformId);
        Assert.AreEqual("scale", transform.Kind);
        Assert.AreEqual(2, transform.Scale);
        Assert.AreEqual(2d, transform.Parameters["scale"]);
        Assert.IsFalse(transform.Lossy);
        Assert.AreEqual(new EnhancementPoint(34.5, 63), enhanced);
        Assert.AreEqual(original, roundTrip);
        CollectionAssert.AreEqual(
            ExpectedForwardMatrix,
            transform.Matrix3X3.ToArray());
        CollectionAssert.AreEqual(
            ExpectedInverseMatrix,
            transform.InverseMatrix3X3.ToArray());
    }

    [TestMethod]
    public void NcnnAdapterDoesNotAdvertiseCpuAsAnExecutionProvider()
    {
        CollectionAssert.AreEqual(
            ExpectedProviders,
            Enum.GetValues<EnhancementProvider>());
    }
}
