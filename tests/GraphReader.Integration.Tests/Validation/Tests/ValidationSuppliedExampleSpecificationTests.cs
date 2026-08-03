// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

#pragma warning disable CA1861 // Small collection expressions are intentional test fixtures.

namespace GraphReader.Integration.Tests.Validation.Tests;

[TestClass]
public sealed class ValidationSuppliedExampleSpecificationTests
{
    [TestMethod]
    public void ValidationSuppliedExampleRemainsLocalOnlyAndScientificallyUnspecified()
    {
        string specificationPath = Path.Combine(
            RepositoryRoot.Find(),
            "tests",
            "GraphReader.Integration.Tests",
            "Validation",
            "Specifications",
            "supplied-example.expected.json");

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(specificationPath));
        JsonElement root = document.RootElement;
        JsonElement policy = root.GetProperty("fixture_policy");
        Assert.IsFalse(policy.GetProperty("image_included").GetBoolean());
        Assert.IsFalse(policy.GetProperty("redistributable").GetBoolean());
        Assert.IsTrue(policy.GetProperty("private_case_required").GetBoolean());

        string[] excludedArtifacts = root.GetProperty("structure")
            .GetProperty("non_marker_artifacts")
            .EnumerateArray()
            .Select(item => item.GetString()!)
            .ToArray();
        CollectionAssert.AreEquivalent(
            new[]
            {
                "x_axis",
                "y_axis",
                "tick_marks",
                "phase_dividers",
                "top_condition_brackets",
                "axis_title",
                "annotation_text",
                "participant_text",
                "annotation_arrows",
                "line_intersections",
            },
            excludedArtifacts);

        JsonElement markers = root.GetProperty("markers");
        Assert.AreEqual(2, markers.GetProperty("open_circle_probe_count").GetInt32());
        CollectionAssert.AreEqual(
            new[] { 21, 22 },
            markers.GetProperty("open_circle_probe_approximate_sessions")
                .EnumerateArray()
                .Select(item => item.GetInt32())
                .ToArray());

        string[] unspecified = root.GetProperty("intentionally_unspecified")
            .EnumerateArray()
            .Select(item => item.GetString()!)
            .ToArray();
        CollectionAssert.AreEqual(
            new[]
            {
                "exact_observation_count",
                "marker_pixel_coordinates",
                "marker_graph_y_values",
                "series_scientific_labels",
            },
            unspecified);
    }
}
