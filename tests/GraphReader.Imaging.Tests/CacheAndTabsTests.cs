// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Imaging.Tests;

[TestClass]
public sealed class CacheAndTabsTests
{
    private static readonly int[] ExpectedBatchIndexes = [0, 1, 2];

    [TestMethod]
    public async Task DerivedCacheKeyIsDeterministicAndInvalidatesOnAnyInputChange()
    {
        string directory = TestImageFixtures.CreateDirectory();
        try
        {
            ImportedImage image = TestImageFixtures.FakeImportedImage("fake.png", 1);
            TransformChain chain = DerivedImageHandles.Scale(image, 2, 2).TransformChain;
            DerivedCacheKeyInput input = CreateKeyInput(chain, "2");
            DerivedCacheKey first = DerivedCacheKey.Create(input);
            DerivedCacheKey reordered = DerivedCacheKey.Create(input with
            {
                Parameters = new Dictionary<string, string> { ["quality"] = "high", ["scale"] = "2" }
            });
            DerivedCacheKey invalidated = DerivedCacheKey.Create(CreateKeyInput(chain, "3"));
            var cache = new ContentAddressedDerivedCache(directory);

            await cache.PutAsync(first, new byte[] { 9, 8, 7 }, CancellationToken.None);

            Assert.AreEqual(first, reordered);
            Assert.AreNotEqual(first, invalidated);
            CollectionAssert.AreEqual(new byte[] { 9, 8, 7 }, await cache.TryReadAsync(first, CancellationToken.None));
            Assert.IsNull(await cache.TryReadAsync(invalidated, CancellationToken.None));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [TestMethod]
    public async Task ImageTabServiceOpensMultipleTabsUsingUiFake()
    {
        const string firstPath = "first.png";
        const string secondPath = "second.png";
        var configured = new Dictionary<string, ImageImportResult>
        {
            [firstPath] = ImageImportResult.Success(TestImageFixtures.FakeImportedImage(firstPath, 1)),
            [secondPath] = ImageImportResult.Success(TestImageFixtures.FakeImportedImage(secondPath, 20))
        };
        var fake = new FakeImageImportService(configured);
        using var tabs = new ImageTabService(fake);

        ImageTabOpenResult first = await tabs.OpenAsync(firstPath, CancellationToken.None);
        ImageTabOpenResult second = await tabs.OpenAsync(secondPath, CancellationToken.None);

        Assert.IsTrue(first.IsSuccess);
        Assert.IsTrue(second.IsSuccess);
        Assert.HasCount(2, tabs.Tabs);
        CollectionAssert.AreEqual(new[] { firstPath, secondPath }, fake.RequestedPaths);
        Assert.IsTrue(tabs.Close(first.Tab!.TabId));
        Assert.HasCount(1, tabs.Tabs);
    }

    [TestMethod]
    public async Task FakeBatchMatchesOrderingFailuresIndexesAndHashDuplicates()
    {
        const string firstPath = "first.png";
        const string missingPath = "missing.png";
        const string duplicatePath = "duplicate.png";
        var configured = new Dictionary<string, ImageImportResult>
        {
            [firstPath] = ImageImportResult.Success(TestImageFixtures.FakeImportedImage(firstPath, 1)),
            [duplicatePath] = ImageImportResult.Success(TestImageFixtures.FakeImportedImage(duplicatePath, 2))
        };
        var fake = new FakeImageImportService(configured);

        BatchImportResult result = await fake.ImportBatchAsync(
            new[] { firstPath, missingPath, duplicatePath },
            CancellationToken.None);

        CollectionAssert.AreEqual(new[] { firstPath, missingPath, duplicatePath }, result.Items.Select(static item => item.SourcePath).ToArray());
        CollectionAssert.AreEqual(ExpectedBatchIndexes, result.Items.Select(static item => item.InputIndex).ToArray());
        Assert.AreEqual(0, result.Items[0].Image?.InputIndex);
        Assert.AreEqual(2, result.Items[2].Image?.InputIndex);
        Assert.AreEqual(0, result.Items[2].Image?.DuplicateOfInputIndex);
        Assert.AreEqual(ImageImportErrorCode.FileNotFound, result.Items[1].Error?.Code);
        Assert.AreEqual(ImageErrorSeverity.Error, result.Items[1].Error?.Severity);
        Assert.AreEqual(2, result.SuccessfulCount);
        Assert.AreEqual(1, result.DuplicateCount);
    }

    private static DerivedCacheKeyInput CreateKeyInput(TransformChain chain, string scale) =>
        new(
            new string('a', 64),
            "10,20,300,200",
            chain,
            "display",
            "1",
            new string('b', 64),
            new Dictionary<string, string> { ["scale"] = scale, ["quality"] = "high" },
            "1");
}
