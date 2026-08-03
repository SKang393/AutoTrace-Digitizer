// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Ocr;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Legends.Tests;

[TestClass]
public sealed class AnnotationArrowResolverTests
{
    [TestMethod]
    public void ArrowCalloutAssociatesTextWithTargetAndMarksBothArtifacts()
    {
        var request = CreateArrowRequest();

        var (callouts, artifacts) = new AnnotationArrowResolver().Resolve(request, CancellationToken.None);

        Assert.HasCount(1, callouts);
        Assert.AreEqual("annotation", callouts[0].TextRegionId);
        Assert.AreEqual(LegendTestFixtures.FilledMarkerId, callouts[0].TargetMarkerId);
        Assert.AreEqual("stroke", callouts[0].StrokeId);
        Assert.AreEqual("arrowhead", callouts[0].ArrowheadId);
        Assert.HasCount(2, artifacts);
        CollectionAssert.AreEquivalent(
            new[] { LegendArtifactKind.ArrowShaft, LegendArtifactKind.Arrowhead },
            artifacts.Select(static artifact => artifact.Kind).ToArray());
    }

    [TestMethod]
    public void ArrowheadArtifactCannotRemainATriangleDataMarker()
    {
        var request = CreateArrowRequest();

        var (_, artifacts) = new AnnotationArrowResolver().Resolve(request, CancellationToken.None);

        var arrowhead = artifacts.Single(static artifact => artifact.Kind == LegendArtifactKind.Arrowhead);
        StringAssert.EndsWith(arrowhead.ArtifactId, ":arrowhead");
        Assert.IsTrue(arrowhead.Bounds.Contains(new LegendPoint(207, 150)));
    }

    [TestMethod]
    public void NearbyNonAnnotationTextDoesNotBecomeCallout()
    {
        var request = CreateArrowRequest(OcrTextRole.Other);

        var (callouts, artifacts) = new AnnotationArrowResolver().Resolve(request, CancellationToken.None);

        Assert.IsEmpty(callouts);
        Assert.IsEmpty(artifacts);
    }

    [TestMethod]
    public void DisconnectedTriangleDoesNotBecomeArrowhead()
    {
        var request = LegendTestFixtures.Request(
            textRegions: [LegendTestFixtures.Text("annotation", 100, 95, "Change", OcrTextRole.Annotation)],
            strokes: [new LegendStrokeCandidate("stroke", new LegendPoint(170, 110), new LegendPoint(210, 150), 1.5, 0.95)],
            triangles:
            [
                new LegendTriangleCandidate(
                    "far-triangle",
                    [new LegendPoint(300, 260), new LegendPoint(294, 255), new LegendPoint(294, 265)],
                    0.96),
            ]);

        var (callouts, artifacts) = new AnnotationArrowResolver().Resolve(request, CancellationToken.None);

        Assert.IsEmpty(callouts);
        Assert.IsEmpty(artifacts);
    }

    [TestMethod]
    public void ResolveHonorsPreCanceledToken()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        Assert.ThrowsExactly<OperationCanceledException>(() =>
            new AnnotationArrowResolver().Resolve(CreateArrowRequest(), cancellation.Token));
    }

    private static LegendReasoningRequest CreateArrowRequest(OcrTextRole role = OcrTextRole.Annotation) =>
        LegendTestFixtures.Request(
            textRegions: [LegendTestFixtures.Text("annotation", 100, 100, "Change condition", role)],
            strokes:
            [
                new LegendStrokeCandidate(
                    "stroke",
                    new LegendPoint(172, 110),
                    new LegendPoint(207, 150),
                    1.5,
                    0.95),
            ],
            triangles:
            [
                new LegendTriangleCandidate(
                    "arrowhead",
                    [new LegendPoint(212, 150), new LegendPoint(204, 145), new LegendPoint(204, 155)],
                    0.96),
            ]);
}
