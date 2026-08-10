// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Security.Cryptography;
using GraphReader.App.Integration.Workflow;
using GraphReader.Axis;
using GraphReader.Inference;
using GraphReader.Legends;
using GraphReader.Markers.Classification;
using GraphReader.Markers.Detection;
using GraphReader.Markers.Grouping;
using GraphReader.Ocr;
using GraphReader.Phases;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.App.Tests;

[TestClass]
public sealed class ProductionAutomaticDetectionAdapterTests
{
    private static readonly string[] ExpectedStages =
        ["axis", "ocr", "ocr", "markers", "markers", "markers", "legends", "phases"];

    [TestMethod]
    public async Task ApprovedCompositePersistsExactProjectionAndReturnsRealCandidates()
    {
        Guid runId = Guid.Parse("10000000-0000-0000-0000-000000000019");
        Guid projectId = Guid.Parse("20000000-0000-0000-0000-000000000019");
        Guid sourceId = Guid.Parse("30000000-0000-0000-0000-000000000019");
        Guid panelId = Guid.Parse("40000000-0000-0000-0000-000000000019");
        byte[] bytes = [1, 9, 2, 1];
        string sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes));
        var image = new WorkflowImageEvidence(
            "memory:production-composite.png",
            sha256,
            100,
            100,
            WorkflowImageVariant.Original);
        var imported = new WorkflowImportedPanel(
            panelId,
            sourceId,
            "production-composite.png",
            image);
        var prepared = new WorkflowPreparedPanel(imported, image, enhanced: null);
        var request = new ProductionWorkflowDetectionRequest(
            prepared,
            image,
            WorkflowImageVariant.Original,
            runId,
            projectId,
            bytes);
        var store = new ProductionWorkflowPanelStore();
        store.Register(new ProductionPanelEvidence(
            imported,
            WorkflowSourceKind.Image,
            bytes));
        var adapter = new ProductionAutomaticDetectionAdapter(
            store,
            new RasterDecoder(),
            new AxisAdapter(),
            new OcrAdapter(),
            new MaskComposer(),
            new CenterAdapter(),
            new ClassificationAdapter(),
            new LegendAdapter(),
            new PhaseAdapter(),
            new EmptyConnectionBuilder());

        WorkflowDetectionBatch batch = await adapter.DetectAsync(
            request,
            CancellationToken.None);

        Assert.IsTrue(adapter.IsApproved);
        Assert.AreEqual(WorkflowImageVariant.Original, batch.SourceImage);
        Assert.HasCount(2, batch.Candidates);
        Assert.IsTrue(batch.Candidates.All(static point => point.GraphX.HasValue));
        Assert.IsTrue(batch.Candidates.All(static point => point.GraphY.HasValue));
        Assert.IsTrue(batch.Candidates.All(static point => Guid.TryParse(point.PointId, out _)));
        Assert.IsTrue(batch.Candidates.All(static point => Guid.TryParse(point.SeriesId, out _)));
        Assert.IsTrue(batch.Candidates.All(static point => Guid.TryParse(point.PhaseId, out _)));
        Assert.AreEqual("classifier-v1", batch.Envelope.Model?.Version);

        ProductionPanelExportEvidence evidence = store.Get(panelId).ExportEvidence!;
        Assert.IsNotNull(evidence.ProjectionEvidence);
        Assert.HasCount(2, evidence.Points);
        Assert.HasCount(2, evidence.ProjectionEvidence.Points);
        Assert.HasCount(2, evidence.ProjectionEvidence.Markers);
        Assert.AreEqual("Chandler", evidence.Participant);
        Assert.AreEqual(8, evidence.Provenance.Count);
        CollectionAssert.AreEqual(
            ExpectedStages,
            evidence.Provenance.Select(static item => item.Stage).ToArray());
    }

    [TestMethod]
    public async Task CompositeRejectsEnhancedEntryWithoutWritingProjection()
    {
        byte[] bytes = [2, 1];
        string sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes));
        Guid panelId = Guid.Parse("41000000-0000-0000-0000-000000000019");
        var original = new WorkflowImageEvidence(
            "memory:original.png",
            sha256,
            100,
            100,
            WorkflowImageVariant.Original);
        var enhanced = new WorkflowImageEvidence(
            "memory:enhanced.png",
            sha256,
            100,
            100,
            WorkflowImageVariant.Enhanced);
        var imported = new WorkflowImportedPanel(
            panelId,
            Guid.Parse("31000000-0000-0000-0000-000000000019"),
            "enhanced.png",
            original);
        var transform = new WorkflowTransformProvenance(
            "identity-enhancement",
            "original_pixels",
            "enhanced_pixels",
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            lossy: false);
        var store = new ProductionWorkflowPanelStore();
        store.Register(new ProductionPanelEvidence(
            imported,
            WorkflowSourceKind.Image,
            bytes,
            enhanced: enhanced,
            enhancedBytes: bytes,
            enhancementTransforms: [transform]));
        var adapter = new ProductionAutomaticDetectionAdapter(
            store,
            new RasterDecoder(),
            new AxisAdapter(),
            new OcrAdapter(),
            new MaskComposer(),
            new CenterAdapter(),
            new ClassificationAdapter(),
            new LegendAdapter(),
            new PhaseAdapter(),
            new EmptyConnectionBuilder());
        var request = new ProductionWorkflowDetectionRequest(
            new WorkflowPreparedPanel(imported, original, enhanced),
            enhanced,
            WorkflowImageVariant.Enhanced,
            Guid.NewGuid(),
            Guid.NewGuid(),
            bytes,
            [transform]);

        ProductionWorkflowStageException exception = await Assert.ThrowsAsync<ProductionWorkflowStageException>(
            () => adapter.DetectAsync(request, CancellationToken.None));

        Assert.AreEqual(ProductionWorkflowFailureCodes.DetectionEvidenceRejected, exception.Failure.Code);
        Assert.IsNull(store.Get(panelId).ExportEvidence);
    }

    private static WorkflowVisionEnvelope Envelope(
        ProductionWorkflowDetectionRequest request,
        string stage,
        string version,
        string modelId,
        char checksum,
        bool deterministic = false) => new(
            1,
            request.RunId,
            request.ProjectId,
            request.Panel.ImportedPanel.PanelId,
            stage,
            version,
            request.Image.Sha256,
            deterministic
                ? null
                : new WorkflowVisionModel(modelId, version, new string(checksum, 64), "cpu"),
            new WorkflowVisionTiming(1, 1, 1, 3),
            0.95,
            transforms: request.Transforms);

    private sealed class RasterDecoder : IProductionRasterFrameDecoder
    {
        public ProductionDecodedRaster Decode(
            ProductionWorkflowDetectionRequest request,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return new ProductionDecodedRaster(
                100,
                100,
                request.Image.Sha256,
                request.ImageVariant,
                MarkerAffineTransform.Identity,
                OcrFrameTransform.Identity,
                100,
                100,
                new byte[10_000],
                new float[10_000]);
        }
    }

    private sealed class AxisAdapter : IProductionAxisGeometryAdapter
    {
        public string AdapterId => "test-axis";

        public bool IsApproved => true;

        public Task<ProductionAxisGeometryEvidence> DetectAsync(
            ProductionWorkflowDetectionRequest request,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var xAxis = new AxisLineFit(
                new GeometryLineSegment(new PixelPoint(10, 90), new PixelPoint(90, 90)),
                0.98,
                0,
                1,
                ["x"]);
            var yAxis = new AxisLineFit(
                new GeometryLineSegment(new PixelPoint(10, 90), new PixelPoint(10, 10)),
                0.98,
                0,
                1,
                ["y"]);
            var geometry = new AxisGeometryResult(
                "original_pixels",
                new PlotPolygon(
                    new PixelPoint(10, 90),
                    new PixelPoint(90, 90),
                    new PixelPoint(90, 10),
                    new PixelPoint(10, 10)),
                xAxis,
                yAxis,
                [
                    Tick("x-1", TickAxis.XAxis, 20, 90),
                    Tick("x-2", TickAxis.XAxis, 80, 90),
                ],
                [
                    new PhaseDividerGeometry(
                        Guid.Parse("50000000-0000-0000-0000-000000000019").ToString("D"),
                        new GeometryLineSegment(new PixelPoint(50, 10), new PixelPoint(50, 90)),
                        DividerStyle.Solid,
                        0.95,
                        1,
                        1,
                        ["divider"]),
                ],
                [],
                0.98,
                new AxisGeometryUncertainty(0, 0, 1, false, []),
                new AxisGeometryDiagnostics(5, 5, 0, 1, 4, 2, 1, 0, TimeSpan.Zero, []));
            return Task.FromResult(new ProductionAxisGeometryEvidence(
                Envelope(request, "axis", "axis-v1", "axis-runtime", 'a'),
                geometry));
        }

        private static AxisTickGeometry Tick(string id, TickAxis axis, double x, double y) => new(
            id,
            axis,
            new PixelPoint(x, y),
            new GeometryLineSegment(new PixelPoint(x, y - 1), new PixelPoint(x, y + 1)),
            0.95,
            [id]);
    }

    private sealed class OcrAdapter : IProductionOcrAdapter
    {
        public string AdapterId => "test-ocr";

        public bool IsApproved => true;

        public Task<ProductionOcrEvidence> RecognizeAsync(
            ProductionWorkflowDetectionRequest request,
            ProductionDecodedRaster originalRaster,
            OcrRectangle plotBounds,
            OcrDetectorImage detectorImage,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Assert.AreEqual(100, detectorImage.Image.Width);
            Assert.AreEqual(100, detectorImage.Image.Height);
            Assert.AreEqual(byte.MaxValue, detectorImage.Image.Pixels.Span[(50 * 100) + 10]);
            Assert.AreEqual(byte.MaxValue, detectorImage.Image.Pixels.Span[(50 * 100) + 50]);
            Assert.AreEqual(byte.MaxValue, detectorImage.Image.Pixels.Span[(90 * 100) + 20]);
            Assert.AreEqual(0, detectorImage.Image.Pixels.Span[(30 * 100) + 30]);
            Assert.AreEqual(
                detectorImage.PixelSha256,
                Convert.ToHexStringLower(SHA256.HashData(detectorImage.Image.Pixels.Span)));
            Assert.IsTrue(originalRaster.CreateOcrImage().Pixels.Span.ToArray()
                .All(static pixel => pixel == 0));
            OcrRegion[] regions =
            [
                Region("x1", 18, 92, "1", OcrTextRole.XTick),
                Region("x2", 78, 92, "2", OcrTextRole.XTick),
                Region("y0", 1, 78, "0", OcrTextRole.YTick),
                Region("y100", 1, 18, "100", OcrTextRole.YTick),
                Region("participant", 82, 84, "Chandler", OcrTextRole.Participant),
            ];
            var result = new OcrResult(
                OcrContract.Version,
                request.RunId.ToString("D"),
                request.ProjectId.ToString("D"),
                request.Panel.ImportedPanel.PanelId.ToString("D"),
                OcrContract.Stage,
                "ocr-v1",
                request.Image.Sha256,
                OcrContract.CoordinateSpace,
                regions,
                regions.Select(region => new OcrMask(
                    region.RegionId,
                    region.Polygon,
                    region.Confidence)).ToArray(),
                new OcrTiming(1, 1, 1, 3),
                0.95,
                [],
                new OcrCacheDiagnostics(false, "test", regions.Length, 1),
                null,
                []);
            return Task.FromResult(new ProductionOcrEvidence(
                result,
                [
                    new ProductionOcrModelEvidence(
                        "ocr_detection",
                        Envelope(request, "ocr", "ocr-v1", "ocr-detection", 'b')),
                    new ProductionOcrModelEvidence(
                        "ocr_recognition",
                        Envelope(request, "ocr", "ocr-v1", "ocr-recognition", 'c')),
                ]));
        }

        private static OcrRegion Region(
            string id,
            double x,
            double y,
            string text,
            OcrTextRole role) => new(
                id,
                OcrPolygon.FromRectangle(new OcrRectangle(x, y, 4, 4)),
                text,
                [new OcrRecognitionAlternative(text, 0.98, OcrSourceImage.Original)],
                role,
                0.98,
                OcrSourceImage.Original,
                OcrReviewStatus.Unreviewed);
    }

    private sealed class MaskComposer : IProductionDetectionMaskComposer
    {
        public string AdapterId => "test-masks";

        public bool IsApproved => true;

        public Task<ProductionDetectionMaskEvidence> ComposeAsync(
            ProductionWorkflowDetectionRequest request,
            ProductionDecodedRaster raster,
            ProductionAxisGeometryEvidence axisEvidence,
            IReadOnlyList<ProductionOcrModelEvidence> ocrModelEvidence,
            OcrResult ocrResult,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(new ProductionDetectionMaskEvidence(
                100,
                100,
                request.Image.Sha256,
                request.ImageVariant,
                new[] { axisEvidence.Envelope }.Concat(ocrModelEvidence.Select(static item => item.Envelope)),
                Envelope(request, "markers", "artifact-mask-v1", "test-artifact-mask", 'f'),
                new float[10_000],
                new float[10_000],
                []));
        }
    }

    private sealed class CenterAdapter : IProductionMarkerCenterAdapter
    {
        public string AdapterId => "test-centers";

        public bool IsApproved => true;

        public ModelIdentity Model { get; } = new(
            "test-center",
            "center-v1",
            new string('d', 64),
            "memory:center.onnx");

        public Task<ProductionMarkerCenterEvidence> DetectAsync(
            ProductionWorkflowDetectionRequest request,
            MarkerImageFrame originalImage,
            MarkerPolygon plotPolygon,
            MarkerImageFrame? enhancedImage,
            IReadOnlyList<WorkflowTransformProvenance>? enhancedTransforms,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            MarkerCenter[] markers =
            [
                new("raw-1", new MarkerPoint(20, 70), 3, 0.01, 0.98, MarkerSourceImage.Original),
                new("raw-2", new MarkerPoint(80, 30), 3, 0.01, 0.97, MarkerSourceImage.Original),
            ];
            return Task.FromResult(new ProductionMarkerCenterEvidence(
                Envelope(request, "markers", "center-v1", "test-center", 'd'),
                markers,
                []));
        }
    }

    private sealed class ClassificationAdapter : IProductionMarkerClassificationAdapter
    {
        public string AdapterId => "test-classifier";

        public bool IsApproved => true;

        public ModelIdentity Model { get; } = new(
            "test-classifier",
            "classifier-v1",
            new string('e', 64),
            "memory:classifier.onnx");

        public Task<ProductionMarkerClassificationEvidence> ClassifyAsync(
            ProductionWorkflowDetectionRequest request,
            MarkerImageFrame image,
            IReadOnlyList<MarkerCenter> markers,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ClassifiedMarker[] classified =
            [
                new(
                    markers[0],
                    MarkerShape.Circle,
                    MarkerFill.Filled,
                    "●",
                    "filled circle",
                    0.01,
                    0.98,
                    0.98,
                    Enumerable.Repeat(0.1f, 12)),
                new(
                    markers[1],
                    MarkerShape.Square,
                    MarkerFill.Open,
                    "□",
                    "open square",
                    0.01,
                    0.97,
                    0.97,
                    Enumerable.Repeat(0.2f, 12)),
            ];
            return Task.FromResult(new ProductionMarkerClassificationEvidence(
                Envelope(request, "markers", "classifier-v1", "test-classifier", 'e'),
                classified));
        }
    }

    private sealed class LegendAdapter : IProductionLegendReasoningAdapter
    {
        public string AdapterId => "test-legends";

        public bool IsApproved => true;

        public Task<ProductionLegendReasoningEvidence> ResolveAsync(
            ProductionWorkflowDetectionRequest request,
            LegendReasoningRequest legendRequest,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            LegendSeriesResolution[] series = legendRequest.Series.Select(item =>
                new LegendSeriesResolution(
                    item.SeriesId,
                    item.CurrentName ?? item.AccessibleName,
                    item.Symbol,
                    item.AccessibleName,
                    LegendEvidenceSource.SymbolFallback,
                    EntryId: null,
                    SourcePanelId: null,
                    0.9,
                    new LegendSemanticEvidence(LegendSemanticHint.Unknown, string.Empty, 0.9),
                    UserConfirmedPreserved: false)).ToArray();
            var payload = new LegendReasoningPayload(
                regions: [],
                series,
                callouts: [],
                artifacts: [],
                [new LegendParticipantMetadata(
                    "participant",
                    "Chandler",
                    new LegendRectangle(82, 84, 12, 4),
                    0.98)],
                excludedArtifactMarkerIds: []);
            return Task.FromResult(new ProductionLegendReasoningEvidence(
                Envelope(request, "legends", "legend-v1", "legend-deterministic", 'f', deterministic: true),
                payload));
        }
    }

    private sealed class PhaseAdapter : IProductionPhaseReasoningAdapter
    {
        private static readonly string DividerId =
            Guid.Parse("60000000-0000-0000-0000-000000000019").ToString("D");
        private static readonly string BaselineId =
            Guid.Parse("70000000-0000-0000-0000-000000000019").ToString("D");
        private static readonly string InterventionId =
            Guid.Parse("80000000-0000-0000-0000-000000000019").ToString("D");

        public string AdapterId => "test-phases";

        public bool IsApproved => true;

        public Task<ProductionPhaseReasoningEvidence> ResolveAsync(
            ProductionWorkflowDetectionRequest request,
            PhaseReasoningRequest phaseRequest,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var divider = new PhaseDivider(
                DividerId,
                50,
                PhaseDividerStyle.Solid,
                [],
                [request.Panel.ImportedPanel.PanelId.ToString("D")],
                0.95,
                PhaseEvidenceSource.ProfilePrior);
            PhaseRegion[] phases =
            [
                new(
                    BaselineId,
                    1,
                    "a",
                    PhaseNormalizedType.Baseline,
                    "Baseline",
                    10,
                    50,
                    null,
                    DividerId,
                    0.95,
                    PhaseEvidenceSource.ProfilePrior),
                new(
                    InterventionId,
                    2,
                    "b",
                    PhaseNormalizedType.Intervention,
                    "Intervention",
                    50,
                    90,
                    DividerId,
                    null,
                    0.95,
                    PhaseEvidenceSource.ProfilePrior),
            ];
            PhasePointAssignment[] assignments = phaseRequest.Points.Select(point =>
                new PhasePointAssignment(
                    point.PointId,
                    point.Center.X < 50 ? BaselineId : InterventionId,
                    point.Center.X)).ToArray();
            var payload = new PhaseReasoningPayload(
                [divider],
                phases,
                assignments,
                [],
                new PhaseManualOverrides());
            return Task.FromResult(new ProductionPhaseReasoningEvidence(
                Envelope(request, "phases", "phase-v1", "phase-deterministic", '1', deterministic: true),
                payload));
        }
    }

    private sealed class EmptyConnectionBuilder : IMarkerConnectionGraphBuilder
    {
        public ValueTask<IReadOnlyList<MarkerConnection>> BuildAsync(
            MarkerConnectionRequest request,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<MarkerConnection>>([]);
        }
    }
}
