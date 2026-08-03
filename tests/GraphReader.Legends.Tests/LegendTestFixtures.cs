// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Markers.Classification;
using GraphReader.Ocr;

namespace GraphReader.Legends.Tests;

internal static class LegendTestFixtures
{
    public const string ProjectId = "00000000-0000-0000-0000-000000000001";
    public const string PanelId = "00000000-0000-0000-0000-000000000002";
    public const string FilledSeriesId = "00000000-0000-0000-0000-000000000003";
    public const string OpenSeriesId = "00000000-0000-0000-0000-000000000004";
    public const string FilledMarkerId = "00000000-0000-0000-0000-000000000005";
    public const string OpenMarkerId = "00000000-0000-0000-0000-000000000006";
    public const string PeerPanelId = "00000000-0000-0000-0000-000000000007";
    public const string InputSha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    public static readonly LegendRectangle PanelBounds = new(0, 0, 500, 300);

    public static readonly LegendRectangle PlotBounds = new(50, 30, 320, 230);

    public static LegendReasoningRequest Request(
        IEnumerable<LegendTextRegion>? textRegions = null,
        IEnumerable<LegendGlyphCandidate>? glyphs = null,
        IEnumerable<LegendSeriesCandidate>? series = null,
        IEnumerable<LegendPlotMarker>? markers = null,
        IEnumerable<LegendStrokeCandidate>? strokes = null,
        IEnumerable<LegendTriangleCandidate>? triangles = null,
        IEnumerable<LegendPeerPanelEvidence>? peers = null,
        LegendReasoningOptions? options = null,
        int contractVersion = LegendReasoningContract.Version) =>
        new(
            ProjectId,
            PanelId,
            InputSha256,
            PanelBounds,
            PlotBounds,
            textRegions ?? [],
            glyphs ?? [],
            series ?? DefaultSeries(),
            markers ?? DefaultMarkers(),
            strokes,
            triangles,
            peers,
            options,
            contractVersion);

    public static LegendTextRegion Text(
        string id,
        double x,
        double y,
        string text,
        OcrTextRole role = OcrTextRole.LegendText,
        double confidence = 0.96) =>
        new(id, new LegendRectangle(x, y, 70, 14), text, role, confidence);

    public static LegendGlyphCandidate Glyph(
        string id,
        double x,
        double y,
        MarkerShape shape,
        MarkerFill fill,
        float[] embedding,
        double confidence = 0.97) =>
        new(id, new LegendRectangle(x, y, 10, 10), shape, fill, embedding, confidence);

    public static LegendSeriesCandidate Series(
        string id,
        string markerId,
        MarkerShape shape,
        MarkerFill fill,
        string symbol,
        string accessibleName,
        float[] embedding,
        string? currentName = null,
        bool confirmed = false) =>
        new(id, shape, fill, symbol, accessibleName, [markerId], embedding, currentName, confirmed);

    public static IReadOnlyList<LegendSeriesCandidate> DefaultSeries() =>
    [
        new LegendSeriesCandidate(
            FilledSeriesId,
            MarkerShape.Circle,
            MarkerFill.Filled,
            "●",
            "filled circle",
            [FilledMarkerId],
            [1f, 0f, 0f]),
        new LegendSeriesCandidate(
            OpenSeriesId,
            MarkerShape.Circle,
            MarkerFill.Open,
            "○",
            "open circle",
            [OpenMarkerId],
            [0f, 1f, 0f]),
    ];

    public static IReadOnlyList<LegendPlotMarker> DefaultMarkers() =>
    [
        new(FilledMarkerId, FilledSeriesId, new LegendPoint(220, 150), MarkerShape.Circle, MarkerFill.Filled),
        new(OpenMarkerId, OpenSeriesId, new LegendPoint(250, 180), MarkerShape.Circle, MarkerFill.Open),
    ];

    public static LegendEntry PeerEntry(
        string panelId = PeerPanelId,
        string text = "Generalization",
        string? normalizedSeriesId = OpenSeriesId) =>
        new(
            "peer-entry",
            "peer-glyph",
            "peer-text",
            text,
            MarkerShape.Circle,
            MarkerFill.Open,
            0.95,
            panelId,
            LegendEvidenceSource.DetectedLegend,
            new LegendSemanticEvidence(LegendSemanticHint.Generalization, "generalization", 0.95),
            normalizedSeriesId);
}
