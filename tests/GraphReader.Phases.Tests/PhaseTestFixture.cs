// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Phases.Tests;

internal static class PhaseTestFixture
{
    public const string ProjectId = "10000000-0000-0000-0000-000000000001";
    public const string PanelId = "20000000-0000-0000-0000-000000000001";
    public const string PeerPanelId = "20000000-0000-0000-0000-000000000002";
    public const string BaselineSeriesId = "30000000-0000-0000-0000-000000000001";
    public const string InterventionOneId = "30000000-0000-0000-0000-000000000002";
    public const string InterventionTwoId = "30000000-0000-0000-0000-000000000003";
    public const string MaintenanceSeriesId = "30000000-0000-0000-0000-000000000004";
    public const string GeneralizationSeriesId = "30000000-0000-0000-0000-000000000005";
    public const string InputSha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    public static PhaseRectangle PlotBounds { get; } = new(20, 20, 480, 240);

    public static PhaseReasoningRequest Request(
        IEnumerable<PhaseDividerSegment>? segments = null,
        IEnumerable<PhaseHeadingEvidence>? headings = null,
        IEnumerable<PhasePointEvidence>? points = null,
        IEnumerable<PhaseSeriesEvidence>? series = null,
        IEnumerable<PhasePanelEvidence>? alignedPanels = null,
        PhaseManualOverrides? overrides = null,
        PhaseReasoningOptions? options = null,
        int contractVersion = PhaseReasoningContract.Version,
        PhaseRectangle? plotBounds = null) =>
        new(
            ProjectId,
            PanelId,
            InputSha256,
            plotBounds ?? PlotBounds,
            segments ?? Array.Empty<PhaseDividerSegment>(),
            headings ?? Array.Empty<PhaseHeadingEvidence>(),
            points ?? Array.Empty<PhasePointEvidence>(),
            series ?? Array.Empty<PhaseSeriesEvidence>(),
            alignedPanels ?? Array.Empty<PhasePanelEvidence>(),
            overrides,
            options,
            contractVersion);

    public static PhaseDividerSegment Segment(
        string id,
        double x,
        double top = 24,
        double bottom = 256,
        PhaseDividerStyle style = PhaseDividerStyle.Solid,
        PhaseSegmentKind kind = PhaseSegmentKind.Candidate,
        string panelId = PanelId,
        double confidence = 0.96,
        double horizontalDrift = 0) =>
        new(
            id,
            panelId,
            new PhasePoint(x, top),
            new PhasePoint(x + horizontalDrift, bottom),
            1,
            style,
            confidence,
            kind);

    public static PhaseHeadingEvidence Heading(
        string id,
        string text,
        double centerX,
        string panelId = PanelId,
        double confidence = 0.96,
        bool rejected = false) =>
        new(
            id,
            panelId,
            new PhaseRectangle(centerX - 35, 1, 70, 16),
            text,
            confidence,
            rejected);

    public static PhasePointEvidence Point(
        string id,
        string seriesId,
        double x,
        double y = 120,
        string panelId = PanelId) =>
        new(id, seriesId, panelId, new PhasePoint(x, y));

    public static PhaseSeriesEvidence Series(
        string seriesId,
        PhaseNormalizedType role,
        IEnumerable<string> pointIds,
        IEnumerable<string>? applicableInterventions = null) =>
        new(seriesId, role, pointIds, applicableInterventions);

    public static async Task<PhaseReasoningResult> ResolveAsync(
        PhaseReasoningRequest request,
        CancellationToken cancellationToken = default) =>
        await new PhaseReasoningService().ResolveAsync(request, cancellationToken);
}
