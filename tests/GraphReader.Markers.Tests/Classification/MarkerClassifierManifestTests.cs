// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
using GraphReader.Markers.Classification;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Classification;

[TestClass]
public sealed class MarkerClassifierManifestTests
{
    private const string ExpectedModelSha256 =
        "59b4af98fe40abd436f01a8c14bf0d12a7c82682ec072c65cef92881aa18b0ef";
    private static readonly string[] ExpectedInputShape = ["N", "1", "32", "32"];
    private static readonly string[] ExpectedOutputShape = ["N", "25"];

    [TestMethod]
    public void CandidateManifestMatchesPackedRuntimeContractAndRemainsFailClosed()
    {
        string repositoryRoot = FindRepositoryRoot();
        string manifestPath = Path.Combine(
            repositoryRoot,
            "models",
            "manifest",
            "markers",
            "graph-marker-classifier-0.1.0.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;

        Assert.AreEqual("graph-marker-classifier", root.GetProperty("model_id").GetString());
        Assert.AreEqual("0.1.0", root.GetProperty("model_version").GetString());
        Assert.AreEqual("marker_classifier", root.GetProperty("task").GetString());
        Assert.AreEqual(ExpectedModelSha256, root.GetProperty("sha256").GetString());
        Assert.IsTrue(root.GetProperty("commercial_use").GetBoolean());
        Assert.IsTrue(root.GetProperty("redistribution").GetBoolean());
        Assert.IsTrue(root.GetProperty("license").GetProperty("reviewed").GetBoolean());

        JsonElement input = root.GetProperty("inputs")[0];
        Assert.AreEqual("marker_patch", input.GetProperty("name").GetString());
        CollectionAssert.AreEqual(
            ExpectedInputShape,
            input.GetProperty("shape").EnumerateArray().Select(JsonValue).ToArray());
        JsonElement output = root.GetProperty("outputs")[0];
        Assert.AreEqual("classification_heads", output.GetProperty("name").GetString());
        CollectionAssert.AreEqual(
            ExpectedOutputShape,
            output.GetProperty("shape").EnumerateArray().Select(JsonValue).ToArray());

        JsonElement postprocessing = root.GetProperty("postprocessing");
        Assert.AreEqual(
            MarkerClassificationContract.ShapeOutputOrder,
            string.Join(',', postprocessing.GetProperty("shape_order").EnumerateArray().Select(Value)));
        Assert.AreEqual(
            MarkerClassificationContract.FillOutputOrder,
            string.Join(',', postprocessing.GetProperty("fill_order").EnumerateArray().Select(Value)));
        Assert.IsTrue(postprocessing.GetProperty("shape_and_fill_separate").GetBoolean());

        JsonElement[] benchmarks = root.GetProperty("benchmarks").EnumerateArray().ToArray();
        JsonElement validation = benchmarks.Single(item =>
            item.GetProperty("profile").GetString() == "validation-selection");
        Assert.AreEqual("fail", validation.GetProperty("status").GetString());
        Assert.IsFalse(validation.GetProperty("release_eligible").GetBoolean());
        Assert.IsLessThan(
            validation.GetProperty("local_shape_gate").GetDouble(),
            validation.GetProperty("shape_macro_f1").GetDouble());
        Assert.IsTrue(benchmarks.All(item => !item.GetProperty("release_eligible").GetBoolean()));

        string noticePath = Path.Combine(
            repositoryRoot,
            root.GetProperty("license").GetProperty("notice_path").GetString()!
                .Replace('/', Path.DirectorySeparatorChar));
        Assert.IsTrue(File.Exists(noticePath), $"Missing model notice: {noticePath}");
        Assert.IsFalse(
            Directory.EnumerateFiles(Path.GetDirectoryName(manifestPath)!, "*.onnx").Any(),
            "Generated model weights must not be committed beside manifests.");
    }

    private static string JsonValue(JsonElement value) =>
        value.ValueKind == JsonValueKind.String
            ? value.GetString()!
            : value.GetRawText();

    private static string Value(JsonElement value) => value.GetString()!;

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "contracts", "model-manifest.schema.json")) &&
                Directory.Exists(Path.Combine(current.FullName, "models", "manifest", "markers")))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the Graph Auto Reader repository root.");
    }
}
