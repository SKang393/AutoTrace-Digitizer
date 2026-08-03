// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Markers.Classification;
using GraphReader.Markers.Detection;
using GraphReader.Markers.Grouping;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Markers.Tests.Grouping;

[TestClass]
public sealed class MarkerGroupingContractTests
{
    [TestMethod]
    public void GroupingOptionsUseAllRequiredEvidenceAndNormalizedWeights()
    {
        var options = new MarkerGroupingOptions();
        double total = options.ShapeFillWeight + options.EmbeddingWeight +
            options.ConnectionWeight + options.SessionOrderWeight +
            options.PhaseContinuityWeight + options.LegendWeight;

        Assert.AreEqual(1, total, 1e-12);
        Assert.IsGreaterThan(0, options.ShapeFillWeight);
        Assert.IsGreaterThan(0, options.EmbeddingWeight);
        Assert.IsGreaterThan(0, options.ConnectionWeight);
        Assert.IsGreaterThan(0, options.SessionOrderWeight);
        Assert.IsGreaterThan(0, options.PhaseContinuityWeight);
        Assert.IsGreaterThan(0, options.LegendWeight);
    }

    [TestMethod]
    public void TextEvidenceSeparatesLegendParticipantAnnotationAndExplicitConfirmation()
    {
        CollectionAssert.AreEqual(
            new[]
            {
                nameof(MarkerTextEvidenceSource.Legend),
                nameof(MarkerTextEvidenceSource.Participant),
                nameof(MarkerTextEvidenceSource.Annotation),
                nameof(MarkerTextEvidenceSource.UserConfirmed),
            },
            Enum.GetNames<MarkerTextEvidenceSource>());
    }

    [TestMethod]
    public void SeriesStateAndEditCommandsDefensivelyCopyEveryCallerOwnedCollection()
    {
        var seriesMarkerIds = new List<string> { "m1", "m2" };
        var probes = new List<string> { "probe" };
        var series = new MarkerSeries(
            "series",
            "●",
            MarkerShape.Circle,
            MarkerFill.Filled,
            "Filled circle",
            MarkerSeriesRole.Intervention,
            seriesMarkerIds,
            0.9,
            applicableProbeSeriesIds: probes);
        var markers = new List<MarkerGroupingEvidence>
        {
            MarkerGroupingTestSupport.Evidence("m1", 1),
            MarkerGroupingTestSupport.Evidence("m2", 2),
        };
        var connections = new List<MarkerConnection>
        {
            new("m1", "m2", 0.9, MarkerConnectionStyle.Solid),
        };
        var seriesList = new List<MarkerSeries> { series };
        var audit = new List<MarkerGroupingAuditEvent>();
        var state = new MarkerGroupingState(markers, connections, seriesList, audit);
        var commandMarkerIds = new List<string> { "m1" };
        var command = new MarkerGroupingEditCommand(
            MarkerGroupingCommandKind.ReassignMarkers,
            commandMarkerIds,
            sourceSeriesId: "series",
            targetSeriesId: "target");

        seriesMarkerIds.Clear();
        probes.Clear();
        markers.Clear();
        connections.Clear();
        seriesList.Clear();
        audit.Add(new MarkerGroupingAuditEvent(
            "late",
            DateTimeOffset.UnixEpoch,
            MarkerGroupingCommandKind.MergeSeries,
            Array.Empty<string>(),
            Array.Empty<string>(),
            "late"));
        commandMarkerIds.Clear();

        Assert.HasCount(2, series.MarkerIds);
        Assert.HasCount(1, series.ApplicableProbeSeriesIds);
        Assert.HasCount(2, state.Markers);
        Assert.HasCount(1, state.Connections);
        Assert.HasCount(1, state.Series);
        Assert.IsEmpty(state.AuditEvents);
        Assert.HasCount(1, command.MarkerIds);
        Assert.AreEqual(2, series.PointCount);
        Assert.AreEqual(1, state.SeriesCount);
        Assert.AreEqual(2, state.UniqueMarkerCount);
    }

    [TestMethod]
    public void GroupingRequestDefensivelyCopiesMarkersConnectionsAndTextEvidence()
    {
        var markers = new List<MarkerGroupingEvidence>
        {
            MarkerGroupingTestSupport.Evidence("m1", 1),
        };
        var connections = new List<MarkerConnection>();
        var legends = new List<MarkerLegendEvidence>
        {
            new(
                MarkerShape.Circle,
                MarkerFill.Filled,
                "Treatment",
                MarkerTextEvidenceSource.Legend,
                0.9),
        };
        var request = new MarkerGroupingRequest(
            "project",
            "panel",
            markers,
            connections,
            legends);

        markers.Clear();
        connections.Add(new MarkerConnection("m1", "m2", 1, MarkerConnectionStyle.Solid));
        legends.Clear();

        Assert.HasCount(1, request.Markers);
        Assert.IsEmpty(request.Connections);
        Assert.HasCount(1, request.LegendEvidence);
    }

    [TestMethod]
    public void ConnectionRequestAuditAndResultDefensivelyCopyNestedBuffersAndCollections()
    {
        MarkerImageFrame frame = MarkerGroupingTestSupport.Frame();
        float[] pixels = frame.ChannelsFirstPixels.ToArray();
        float[] ocrMask = frame.OcrMask.Values.ToArray();
        float[] artifactMask = frame.ArtifactMask.Values.ToArray();
        frame = frame with
        {
            ChannelsFirstPixels = pixels,
            OcrMask = frame.OcrMask with { Values = ocrMask },
            ArtifactMask = frame.ArtifactMask with { Values = artifactMask },
        };
        var request = new MarkerConnectionRequest(
            frame,
            [MarkerGroupingTestSupport.Evidence("m1", 1)]);
        var affectedSeries = new List<string> { "series" };
        var affectedMarkers = new List<string> { "m1" };
        var auditEvent = new MarkerGroupingAuditEvent(
            "event",
            DateTimeOffset.UnixEpoch,
            MarkerGroupingCommandKind.ReassignMarkers,
            affectedSeries,
            affectedMarkers,
            "review");
        var warnings = new List<string> { "warning" };
        var result = new MarkerGroupingResult(
            null,
            new MarkerGroupingTiming(0, 0, 0),
            0,
            warnings,
            null);

        pixels[0] = 0;
        ocrMask[0] = 1;
        artifactMask[0] = 1;
        affectedSeries.Clear();
        affectedMarkers.Clear();
        warnings.Clear();

        Assert.AreEqual(1, request.Image.ChannelsFirstPixels.Span[0]);
        Assert.AreEqual(0, request.Image.OcrMask.Values.Span[0]);
        Assert.AreEqual(0, request.Image.ArtifactMask.Values.Span[0]);
        Assert.HasCount(1, auditEvent.AffectedSeriesIds);
        Assert.HasCount(1, auditEvent.AffectedMarkerIds);
        Assert.HasCount(1, result.Warnings);
    }
}

internal static class MarkerGroupingTestSupport
{
    internal const int FrameSize = 64;

    internal static MarkerGroupingEvidence Evidence(
        string id,
        int observationIndex,
        MarkerShape shape = MarkerShape.Circle,
        MarkerFill fill = MarkerFill.Filled,
        float[]? embedding = null,
        double x = double.NaN,
        double y = 20,
        string? phase = "phase-a")
    {
        double centerX = double.IsNaN(x) ? observationIndex * 8 : x;
        MarkerSymbolDescriptor descriptor = MarkerSymbolMap.Describe(shape, fill);
        var center = new MarkerCenter(
            id,
            new MarkerPoint(centerX, y),
            3,
            0.01,
            0.95,
            MarkerSourceImage.Original);
        var classified = new ClassifiedMarker(
            center,
            shape,
            fill,
            descriptor.Symbol,
            descriptor.AccessibleName,
            0.01,
            0.95,
            0.95,
            embedding ?? [1f, 0f, 0f, 0f]);
        return new MarkerGroupingEvidence(classified, observationIndex, observationIndex, phase);
    }

    internal static MarkerSeries Series(
        string id,
        MarkerShape shape,
        MarkerFill fill,
        MarkerSeriesRole role,
        IEnumerable<string> markerIds,
        string? displayName = null,
        string? sharedBaselineSeriesId = null,
        IEnumerable<string>? applicableProbeSeriesIds = null)
    {
        MarkerSymbolDescriptor descriptor = MarkerSymbolMap.Describe(shape, fill);
        return new MarkerSeries(
            id,
            descriptor.Symbol,
            shape,
            fill,
            displayName ?? descriptor.AccessibleName,
            role,
            markerIds,
            0.9,
            sharedBaselineSeriesId: sharedBaselineSeriesId,
            applicableProbeSeriesIds: applicableProbeSeriesIds);
    }

    internal static MarkerImageFrame Frame(
        IEnumerable<MarkerPoint>? darkPixels = null,
        IEnumerable<MarkerPoint>? ocrMask = null,
        IEnumerable<MarkerPoint>? artifactMask = null)
    {
        float[] pixels = Enumerable.Repeat(1f, FrameSize * FrameSize).ToArray();
        float[] ocr = new float[FrameSize * FrameSize];
        float[] artifacts = new float[FrameSize * FrameSize];
        if (darkPixels is not null)
        {
            foreach (MarkerPoint point in darkPixels)
            {
                int x = (int)Math.Round(point.X);
                int y = (int)Math.Round(point.Y);
                if (x >= 0 && x < FrameSize && y >= 0 && y < FrameSize)
                {
                    pixels[(y * FrameSize) + x] = 0;
                }
            }
        }

        ApplyMask(ocr, ocrMask);
        ApplyMask(artifacts, artifactMask);

        return new MarkerImageFrame(
            FrameSize,
            FrameSize,
            1,
            pixels,
            MarkerSourceImage.Original,
            MarkerAffineTransform.Identity,
            new MarkerMask(FrameSize, FrameSize, ocr),
            new MarkerMask(FrameSize, FrameSize, artifacts));
    }

    internal static IEnumerable<MarkerPoint> Line(MarkerPoint start, MarkerPoint end)
    {
        var steps = (int)Math.Max(Math.Abs(end.X - start.X), Math.Abs(end.Y - start.Y));
        for (var step = 0; step <= steps; step++)
        {
            var fraction = (double)step / steps;
            yield return new MarkerPoint(
                start.X + ((end.X - start.X) * fraction),
                start.Y + ((end.Y - start.Y) * fraction));
        }
    }

    private static void ApplyMask(float[] values, IEnumerable<MarkerPoint>? points)
    {
        if (points is null)
        {
            return;
        }

        foreach (MarkerPoint point in points)
        {
            int x = (int)Math.Round(point.X);
            int y = (int)Math.Round(point.Y);
            if (x >= 0 && x < FrameSize && y >= 0 && y < FrameSize)
            {
                values[(y * FrameSize) + x] = 1;
            }
        }
    }
}
