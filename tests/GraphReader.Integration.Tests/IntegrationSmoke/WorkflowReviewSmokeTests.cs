// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.App.Integration.Workflow;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Integration.Tests.IntegrationSmoke;

[TestClass]
public sealed class WorkflowReviewSmokeTests
{
    private static readonly Guid RunId = Guid.Parse("a0000000-0000-0000-0000-000000000018");
    private static readonly Guid PanelId = Guid.Parse("a1000000-0000-0000-0000-000000000001");
    private static readonly Guid SourceId = Guid.Parse("a2000000-0000-0000-0000-000000000001");
    private static readonly string[] ExpectedCorrectionIds = ["move-1", "series-1", "phase-1"];

    [TestMethod]
    public void EnhancementOffUsesOriginalEvidenceWhileEnhancementOnBuildsConsensus()
    {
        WorkflowDetectionBatch original = Batch(
            WorkflowImageVariant.Original,
            Candidate("point-original", "stable-1", 40, 120, WorkflowImageVariant.Original));
        WorkflowDetectionBatch enhanced = Batch(
            WorkflowImageVariant.Enhanced,
            Candidate("point-enhanced", "stable-1-enhanced", 41, 121, WorkflowImageVariant.Enhanced));

        IReadOnlyList<WorkflowPoint> enhancementOff = OriginalEnhancedConsensus.Merge(original, enhanced: null);
        IReadOnlyList<WorkflowPoint> enhancementOn = OriginalEnhancedConsensus.Merge(original, enhanced);

        Assert.AreEqual(1, enhancementOff.Count);
        Assert.AreEqual(WorkflowImageVariant.Original, enhancementOff[0].SourceImage);
        Assert.AreEqual(40d, enhancementOff[0].OriginalPixelX, 0d);
        Assert.AreEqual(1, enhancementOn.Count);
        Assert.AreEqual(WorkflowImageVariant.Consensus, enhancementOn[0].SourceImage);
        Assert.AreEqual(40.5d, enhancementOn[0].OriginalPixelX, 0d);
        Assert.AreEqual(120.5d, enhancementOn[0].OriginalPixelY, 0d);
        Assert.AreEqual("original_pixels", enhanced.CoordinateSpace);
    }

    [TestMethod]
    public void EnhancedOnlyEvidenceIsNotSilentlyAccepted()
    {
        WorkflowDetectionBatch original = Batch(
            WorkflowImageVariant.Original,
            Candidate("point-original", "stable-1", 10, 10, WorkflowImageVariant.Original));
        WorkflowDetectionBatch enhanced = Batch(
            WorkflowImageVariant.Enhanced,
            Candidate("point-enhanced", "stable-new", 100, 100, WorkflowImageVariant.Enhanced));

        IReadOnlyList<WorkflowPoint> points = OriginalEnhancedConsensus.Merge(original, enhanced);

        Assert.AreEqual(2, points.Count);
        WorkflowPoint enhancedOnly = points.Single(static point => point.SourceImage == WorkflowImageVariant.Enhanced);
        Assert.AreEqual(WorkflowReviewStatus.Unreviewed, enhancedOnly.ReviewStatus);
        Assert.IsTrue(enhancedOnly.Warnings.Contains("ENHANCED_ONLY_DETECTION_REQUIRES_REVIEW", StringComparer.Ordinal));
        Assert.AreEqual(0.45d, enhancedOnly.Confidence, 0.000001d);
    }

    [TestMethod]
    public void DetectionBatchRejectsDuplicatePointIdsBeforeReviewComposition()
    {
        var duplicatePointId = new WorkflowDetectionCandidate(
            "duplicate-point",
            "stable-2",
            20,
            20,
            0.9,
            WorkflowImageVariant.Original);

        Assert.ThrowsExactly<ArgumentException>(() => Batch(
            WorkflowImageVariant.Original,
            Candidate("duplicate-point", "stable-1", 10, 10, WorkflowImageVariant.Original),
            duplicatePointId));
    }

    [TestMethod]
    public void DetectionBatchRetainsFrozenVisionEnvelopeAndReversibleTransformProvenance()
    {
        WorkflowDetectionBatch enhanced = Batch(
            WorkflowImageVariant.Enhanced,
            Candidate("point-enhanced", "stable-enhanced", 20, 20, WorkflowImageVariant.Enhanced));

        Assert.AreEqual(1, enhanced.Envelope.ContractVersion);
        Assert.AreEqual(RunId, enhanced.Envelope.RunId);
        Assert.AreEqual(IntegrationSmokeIds.Project.Value, enhanced.Envelope.ProjectId);
        Assert.AreEqual(PanelId, enhanced.Envelope.PanelId);
        Assert.AreEqual("markers", enhanced.Envelope.Stage);
        Assert.AreEqual("recorded-1", enhanced.Envelope.StageVersion);
        Assert.AreEqual(new string('b', 64), enhanced.Envelope.InputSha256);
        Assert.AreEqual("original_pixels", enhanced.Envelope.CoordinateSpace);
        Assert.AreEqual("recorded-marker-fake", enhanced.Envelope.Model!.ModelId);
        Assert.AreEqual("cpu", enhanced.Envelope.Model.Provider);
        Assert.AreEqual(0d, enhanced.Envelope.Timing.TotalMilliseconds, 0d);
        Assert.AreEqual(0.9d, enhanced.Envelope.Confidence, 0d);
        WorkflowTransformProvenance transform = enhanced.Envelope.Transforms.Single();
        Assert.AreEqual("enhanced_pixels", transform.InputCoordinateSpace);
        Assert.AreEqual("original_pixels", transform.OutputCoordinateSpace);
        Assert.IsNotNull(transform.OutputToInputMatrix);
        Assert.IsFalse(transform.Lossy);
    }

    [TestMethod]
    public void ConsensusMaximizesMatchesBeforeEmittingReviewWarnings()
    {
        WorkflowDetectionBatch original = Batch(
            WorkflowImageVariant.Original,
            Candidate("original-left", "original-left", 0, 0, WorkflowImageVariant.Original),
            Candidate("original-right", "original-right", 3.5, 0, WorkflowImageVariant.Original));
        WorkflowDetectionBatch enhanced = Batch(
            WorkflowImageVariant.Enhanced,
            Candidate("enhanced-left", "enhanced-left", -2.5, 0, WorkflowImageVariant.Enhanced),
            Candidate("enhanced-middle", "enhanced-middle", 2, 0, WorkflowImageVariant.Enhanced));

        IReadOnlyList<WorkflowPoint> points = OriginalEnhancedConsensus.Merge(
            original,
            enhanced,
            new WorkflowConsensusOptions(AgreementDistancePixels: 3, MaximumMatchDistancePixels: 3));

        Assert.HasCount(2, points);
        Assert.IsTrue(points.All(static point => point.SourceImage == WorkflowImageVariant.Consensus));
        Assert.IsFalse(points.SelectMany(static point => point.Warnings).Any(static warning =>
            warning is "ORIGINAL_ONLY_DETECTION_REQUIRES_REVIEW" or "ENHANCED_ONLY_DETECTION_REQUIRES_REVIEW"));
    }

    [TestMethod]
    public void ConsensusMinimizesTotalDisplacementAfterMaximizingMatches()
    {
        WorkflowDetectionBatch original = Batch(
            WorkflowImageVariant.Original,
            Candidate("original-left", "original-left", 0, 0, WorkflowImageVariant.Original),
            Candidate("original-right", "original-right", 3, 0, WorkflowImageVariant.Original));
        WorkflowDetectionBatch enhanced = Batch(
            WorkflowImageVariant.Enhanced,
            Candidate("enhanced-near-right", "a-key", 5, 0, WorkflowImageVariant.Enhanced),
            Candidate("enhanced-near-left", "z-key", 1, 0, WorkflowImageVariant.Enhanced));

        IReadOnlyList<WorkflowPoint> points = OriginalEnhancedConsensus.Merge(
            original,
            enhanced,
            new WorkflowConsensusOptions(AgreementDistancePixels: 5, MaximumMatchDistancePixels: 5));

        Assert.HasCount(2, points);
        Assert.AreEqual(0.5d, points.Single(point => point.PointId == "original-left").OriginalPixelX, 0d);
        Assert.AreEqual(4d, points.Single(point => point.PointId == "original-right").OriginalPixelX, 0d);
    }

    [TestMethod]
    public void CorrectionDetectionKeyTakesPriorityOverRecycledPointId()
    {
        WorkflowPreparedPanel prepared = PreparedPanel(enhancementEnabled: false);
        WorkflowPoint recycledId = OriginalEnhancedConsensus.Merge(
            Batch(WorkflowImageVariant.Original, Candidate("point-recycled", "stable-other", 10, 10, WorkflowImageVariant.Original)),
            enhanced: null).Single();
        WorkflowPoint target = OriginalEnhancedConsensus.Merge(
            Batch(WorkflowImageVariant.Original, Candidate("new-target-id", "stable-target", 20, 20, WorkflowImageVariant.Original)),
            enhanced: null).Single();
        var state = new WorkflowReviewState(
            IntegrationSmokeIds.Project.Value,
            [new WorkflowReviewPanel(prepared, [recycledId, target])]);

        WorkflowReviewState corrected = ManualCorrectionOverlay.Apply(
            state,
            new MoveWorkflowPointCorrection(
                "move-by-key",
                PanelId,
                "point-recycled",
                "stable-target",
                44,
                55));

        Assert.AreEqual(10d, corrected.Panels.Single().Points.Single(point => point.PointId == "point-recycled").OriginalPixelX, 0d);
        WorkflowPoint moved = corrected.Panels.Single().Points.Single(point => point.DetectionKey == "stable-target");
        Assert.AreEqual(44d, moved.OriginalPixelX, 0d);
        Assert.AreEqual(55d, moved.OriginalPixelY, 0d);
    }

    [TestMethod]
    public void MarkerAndPhaseCorrectionsReapplyByStableDetectionKeyAfterRerun()
    {
        WorkflowPreparedPanel prepared = PreparedPanel(enhancementEnabled: true);
        WorkflowReviewPanel initialPanel = new(
            prepared,
            OriginalEnhancedConsensus.Merge(
                Batch(WorkflowImageVariant.Original, Candidate("point-1", "stable-1", 40, 120, WorkflowImageVariant.Original)),
                Batch(WorkflowImageVariant.Enhanced, Candidate("point-1e", "stable-1e", 40, 120, WorkflowImageVariant.Enhanced))));
        WorkflowReviewState state = new(IntegrationSmokeIds.Project.Value, [initialPanel]);
        WorkflowPoint initialPoint = state.Panels.Single().Points.Single();

        state = ManualCorrectionOverlay.Apply(
            state,
            new MoveWorkflowPointCorrection("move-1", PanelId, initialPoint.PointId, initialPoint.DetectionKey, 43, 117));
        state = ManualCorrectionOverlay.Apply(
            state,
            new ReassignWorkflowPointCorrection("series-1", PanelId, initialPoint.PointId, initialPoint.DetectionKey, "series-corrected"));
        state = ManualCorrectionOverlay.Apply(
            state,
            new AssignWorkflowPointPhaseCorrection("phase-1", PanelId, initialPoint.PointId, initialPoint.DetectionKey, "phase-corrected"));

        WorkflowReviewPanel rerunAutomation = new(
            prepared,
            OriginalEnhancedConsensus.Merge(
                Batch(WorkflowImageVariant.Original, Candidate("rerun-point", "stable-1", 41, 119, WorkflowImageVariant.Original)),
                enhanced: null));
        WorkflowReviewPanel reapplied = ManualCorrectionOverlay.Reapply(
            rerunAutomation,
            state.Panels.Single(),
            state.CorrectionJournal);

        WorkflowPoint corrected = reapplied.Points.Single();
        Assert.AreEqual(initialPoint.PointId, corrected.PointId);
        Assert.AreEqual(43d, corrected.OriginalPixelX, 0d);
        Assert.AreEqual(117d, corrected.OriginalPixelY, 0d);
        Assert.IsNull(corrected.GraphX);
        Assert.IsNull(corrected.GraphY);
        Assert.AreEqual("series-corrected", corrected.SeriesId);
        Assert.AreEqual("phase-corrected", corrected.PhaseId);
        Assert.AreEqual(WorkflowReviewStatus.Corrected, corrected.ReviewStatus);
        CollectionAssert.AreEquivalent(
            ExpectedCorrectionIds,
            corrected.CorrectionIds.ToArray());
    }

    [TestMethod]
    public void UserAddedPointSurvivesAutomationRerunWithoutClaimingDetectionEvidence()
    {
        WorkflowPreparedPanel prepared = PreparedPanel(enhancementEnabled: false);
        WorkflowPoint automated = OriginalEnhancedConsensus.Merge(
            Batch(WorkflowImageVariant.Original, Candidate("point-1", "stable-1", 40, 120, WorkflowImageVariant.Original)),
            enhanced: null).Single();
        WorkflowReviewState state = new(
            IntegrationSmokeIds.Project.Value,
            [new WorkflowReviewPanel(prepared, [automated])]);
        var manualPoint = new WorkflowPoint(
            "manual-1",
            null,
            75,
            90,
            1,
            WorkflowImageVariant.Original,
            WorkflowReviewStatus.Corrected,
            "○",
            "circle",
            "open",
            "series-manual",
            "phase-manual",
            6,
            60,
            "manual",
            null,
            isManual: true);
        state = ManualCorrectionOverlay.Apply(
            state,
            new AddWorkflowPointCorrection("add-1", PanelId, manualPoint));

        WorkflowReviewPanel rerun = new(
            prepared,
            OriginalEnhancedConsensus.Merge(
                Batch(WorkflowImageVariant.Original, Candidate("point-1-rerun", "stable-1", 41, 119, WorkflowImageVariant.Original)),
                enhanced: null));
        WorkflowReviewPanel reapplied = ManualCorrectionOverlay.Reapply(
            rerun,
            state.Panels.Single(),
            state.CorrectionJournal);

        Assert.AreEqual(2, reapplied.Points.Count);
        WorkflowPoint preservedManual = reapplied.Points.Single(static point => point.IsManual);
        Assert.IsNull(preservedManual.DetectionKey);
        Assert.AreEqual("manual-1", preservedManual.PointId);
        Assert.IsTrue(preservedManual.CorrectionIds.Contains("add-1", StringComparer.Ordinal));
    }

    private static WorkflowPreparedPanel PreparedPanel(bool enhancementEnabled)
    {
        var original = new WorkflowImageEvidence("synthetic.bmp", new string('a', 64), 120, 200, WorkflowImageVariant.Original);
        var imported = new WorkflowImportedPanel(PanelId, SourceId, "Synthetic panel", original);
        WorkflowImageEvidence? enhanced = enhancementEnabled
            ? new WorkflowImageEvidence("synthetic.enhanced.png", new string('b', 64), 240, 400, WorkflowImageVariant.Enhanced)
            : null;
        return new WorkflowPreparedPanel(imported, original, enhanced);
    }

    private static WorkflowDetectionCandidate Candidate(
        string pointId,
        string detectionKey,
        double x,
        double y,
        WorkflowImageVariant variant) =>
        new(
            pointId,
            detectionKey,
            x,
            y,
            0.9,
            variant,
            "●",
            "circle",
            "filled",
            "series-auto",
            "phase-auto",
            3,
            42,
            "markers",
            "1");

    private static WorkflowDetectionBatch Batch(
        WorkflowImageVariant variant,
        params WorkflowDetectionCandidate[] candidates)
    {
        string inputSha256 = variant == WorkflowImageVariant.Original
            ? new string('a', 64)
            : new string('b', 64);
        WorkflowTransformProvenance[] transforms = variant == WorkflowImageVariant.Enhanced
            ?
            [
                new WorkflowTransformProvenance(
                    "recorded-enhanced-x2",
                    "enhanced_pixels",
                    "original_pixels",
                    [0.5, 0, 0, 0, 0.5, 0, 0, 0, 1],
                    [2, 0, 0, 0, 2, 0, 0, 0, 1],
                    lossy: false),
            ]
            : [];
        var envelope = new WorkflowVisionEnvelope(
            contractVersion: 1,
            RunId,
            IntegrationSmokeIds.Project.Value,
            PanelId,
            stage: "markers",
            stageVersion: "recorded-1",
            inputSha256,
            new WorkflowVisionModel("recorded-marker-fake", "1", new string('c', 64), "cpu"),
            new WorkflowVisionTiming(0, 0, 0, 0),
            confidence: candidates.Length == 0 ? 0 : candidates.Average(static candidate => candidate.Confidence),
            transforms: transforms);
        return new WorkflowDetectionBatch(envelope, variant, candidates);
    }
}
