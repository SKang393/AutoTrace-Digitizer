// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
using GraphReader.Domain;

namespace GraphReader.Domain.Tests;

internal static class TestProjectFactory
{
    public static readonly DateTimeOffset CreatedUtc = new(2026, 1, 2, 3, 4, 5, TimeSpan.Zero);
    public static readonly ProjectId ProjectId = new(Guid.Parse("11111111-1111-1111-1111-111111111111"));
    public static readonly SourceId SourceId = new(Guid.Parse("22222222-2222-2222-2222-222222222222"));
    public static readonly PanelId PanelId = new(Guid.Parse("33333333-3333-3333-3333-333333333333"));
    public static readonly SeriesId SeriesId = new(Guid.Parse("77777777-7777-7777-7777-777777777777"));
    public static readonly PointId PointId = new(Guid.Parse("88888888-8888-8888-8888-888888888888"));

    public static ProjectDocument Create(bool withCalibration = true, DateTimeOffset? modifiedUtc = null)
    {
        var transform = new TransformRecord(
            new TransformId(Guid.Parse("44444444-4444-4444-4444-444444444444")),
            TransformKind.Crop,
            CoordinateSpace.OriginalPixels,
            CoordinateSpace.PanelPixels,
            new double[] { 1, 0, -10, 0, 1, -20, 0, 0, 1 },
            new double[] { 1, 0, 10, 0, 1, 20, 0, 0, 1 },
            ParseElement("{\"z\":2,\"a\":1}"),
            Lossy: false);
        OcrRegionId ocrRegionId = new(Guid.Parse("55555555-5555-5555-5555-555555555555"));
        CalibrationRecord? calibration = withCalibration
            ? new CalibrationRecord(
                new CalibrationId(Guid.Parse("66666666-6666-6666-6666-666666666666")),
                CalibrationStatus.Valid,
                new CalibrationAnchor[]
                {
                    new(
                        CalibrationAnchorKind.Session1Y0,
                        new PixelPoint(20, 180),
                        new GraphPoint(1, 0),
                        0.99,
                        EvidenceRegionId: null),
                    new(
                        CalibrationAnchorKind.Session1Ymax,
                        new PixelPoint(20, 20),
                        new GraphPoint(1, 100),
                        0.97,
                        ocrRegionId),
                    new(
                        CalibrationAnchorKind.SessionmaxY0,
                        new PixelPoint(300, 180),
                        new GraphPoint(24, 0),
                        0.96,
                        EvidenceRegionId: null)
                },
                new SessionLatticeRecord(20, 12.173913, 1, 24, 0.95, "ocr_ticks"),
                UserConfirmed: true,
                Confidence: 0.96,
                Reasons: Array.Empty<string>())
            : null;
        var ocr = new OcrEvidence(
            ocrRegionId,
            new PixelPoint[]
            {
                new(10, 10),
                new(20, 10),
                new(20, 20),
                new(10, 20)
            },
            "100",
            new OcrAlternative[] { new("100", 0.94), new("10O", 0.03) },
            OcrRole.YTick,
            0.94,
            SourceImageKind.Consensus,
            ReviewStatus.Accepted);
        MarkerId markerId = new(Guid.Parse("99999999-9999-9999-9999-999999999999"));
        PhaseId phaseId = new(Guid.Parse("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"));
        var marker = new MarkerRecord(
            markerId,
            new PixelPoint(40.5, 120.25),
            4.2,
            MarkerShape.Circle,
            MarkerFill.Open,
            "○",
            0.01,
            0.98,
            0.94,
            0.96,
            Embedding: null,
            CandidateSeriesId: SeriesId,
            SourceImageKind.Consensus,
            ReviewStatus.Accepted);
        var point = new PointRecord(
            PointId,
            markerId,
            SeriesId,
            phaseId,
            new PixelPoint(40.5, 120.25),
            GraphX: 2,
            GraphY: 37.25,
            ObservationIndex: 2,
            PrintedXValue: null,
            EstimatedXValue: 2,
            PointXSource.Estimated,
            XConfidence: 0.81,
            YConfidence: 0.91,
            PointConfidence: 0.89,
            SourceStage: "markers",
            ModelVersion: null,
            ReviewStatus.Accepted,
            ModificationHistory: Array.Empty<PointModification>());
        var series = new SeriesRecord(
            SeriesId,
            "○",
            MarkerShape.Circle,
            MarkerFill.Open,
            "Open circle",
            SemanticRole.Generalization,
            LegendText: null,
            new PointId[] { PointId },
            Confidence: 0.88,
            SharedBaselineSeriesId: null,
            ApplicableProbeSeriesIds: Array.Empty<SeriesId>(),
            UserConfirmedName: false);
        var phase = new PhaseRecord(
            phaseId,
            Order: 1,
            Code: "a",
            PhaseNormalizedType.Baseline,
            LabelText: "Baseline",
            ScreenXMin: 20,
            ScreenXMax: 160,
            BoundaryLeftId: null,
            BoundaryRightId: null,
            Confidence: 0.9,
            PhaseSource.Ocr,
            UserConfirmed: false);
        var source = new SourceReference(
            SourceId,
            SourceKind.Image,
            "fixture.png",
            @"C:\Research\fixture.png",
            new string('a', 64),
            ParseElement("{\"year\":2026,\"author\":\"Example\"}"));
        var panel = new PanelRecord(
            PanelId,
            SourceId,
            PageNumber: null,
            DisplayName: "Panel A",
            Participant: null,
            new CropRectangle(10, 20, 320, 200),
            new TransformRecord[] { transform },
            Enhancement: ParseElement("{\"scale\":2,\"model_id\":null}"),
            calibration,
            new OcrEvidence[] { ocr },
            new MarkerRecord[] { marker },
            new SeriesRecord[] { series },
            new PointRecord[] { point },
            new PhaseRecord[] { phase },
            new ExportSettingsRecord("printed_session", true, new SeriesId[] { SeriesId }),
            Validation: null);
        DateTimeOffset modified = (modifiedUtc ?? CreatedUtc).ToUniversalTime();
        return new ProjectDocument(
            ProjectDocument.CurrentSchemaVersion,
            ProjectId,
            "0.0.2",
            CreatedUtc,
            modified,
            ProjectSettings.Default,
            new SourceReference[] { source },
            new PanelRecord[] { panel },
            AuditTrail.Empty);
    }

    public static JsonElement ParseElement(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        return document.RootElement.Clone();
    }
}

internal sealed class TemporaryDirectory : IDisposable
{
    public TemporaryDirectory()
    {
        Path = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            "GraphReader.Domain.Tests",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path);
    }

    public string Path { get; }

    public void Dispose()
    {
        if (Directory.Exists(Path))
        {
            Directory.Delete(Path, recursive: true);
        }
    }
}
