// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using GraphReader.Domain;

namespace GraphReader.Integration.Tests.IntegrationSmoke;

internal sealed class IntegrationSmokeTestEnvironment : IDisposable
{
    private const string DirectoryPrefix = "GraphReader.IntegrationSmoke.Tests-";

    public IntegrationSmokeTestEnvironment()
    {
        Root = Path.Combine(Path.GetTempPath(), DirectoryPrefix + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Root);
    }

    public string Root { get; }

    public string PathFor(params string[] parts) =>
        parts.Aggregate(Root, Path.Combine);

    public string WriteBmp(string fileName, byte blue, byte green, byte red)
    {
        string path = PathFor(fileName);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, CreateBmp(blue, green, red));
        return path;
    }

    public void Dispose()
    {
        string fullRoot = Path.GetFullPath(Root);
        string expectedParent = Path.GetFullPath(Path.GetTempPath());
        if (Directory.Exists(fullRoot) &&
            fullRoot.StartsWith(expectedParent, StringComparison.OrdinalIgnoreCase) &&
            Path.GetFileName(fullRoot).StartsWith(DirectoryPrefix, StringComparison.Ordinal))
        {
            Directory.Delete(fullRoot, recursive: true);
        }
    }

    private static byte[] CreateBmp(byte blue, byte green, byte red)
    {
        const int width = 2;
        const int height = 2;
        const int rowStride = 8;
        const int pixelOffset = 54;
        const int fileSize = pixelOffset + (rowStride * height);
        var bytes = new byte[fileSize];

        bytes[0] = (byte)'B';
        bytes[1] = (byte)'M';
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(2, 4), fileSize);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(10, 4), pixelOffset);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(14, 4), 40);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(18, 4), width);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(22, 4), height);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(26, 2), 1);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(28, 2), 24);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(34, 4), rowStride * height);

        for (int row = 0; row < height; row++)
        {
            int rowOffset = pixelOffset + (row * rowStride);
            for (int column = 0; column < width; column++)
            {
                int pixel = rowOffset + (column * 3);
                bytes[pixel] = blue;
                bytes[pixel + 1] = green;
                bytes[pixel + 2] = red;
            }
        }

        return bytes;
    }
}

internal static class IntegrationSmokeIds
{
    public static readonly ProjectId Project = new(Guid.Parse("10000000-0000-0000-0000-000000000001"));
    public static readonly SourceId Source = new(Guid.Parse("20000000-0000-0000-0000-000000000001"));
    public static readonly PanelId Panel = new(Guid.Parse("30000000-0000-0000-0000-000000000001"));
    public static readonly CalibrationId Calibration = new(Guid.Parse("40000000-0000-0000-0000-000000000001"));
    public static readonly SeriesId BaselineSeries = new(Guid.Parse("50000000-0000-0000-0000-000000000001"));
    public static readonly SeriesId InterventionSeries = new(Guid.Parse("50000000-0000-0000-0000-000000000002"));
    public static readonly SeriesId GeneralizationSeries = new(Guid.Parse("50000000-0000-0000-0000-000000000003"));
    public static readonly PhaseId BaselinePhase = new(Guid.Parse("60000000-0000-0000-0000-000000000001"));
    public static readonly PhaseId InterventionPhase = new(Guid.Parse("60000000-0000-0000-0000-000000000002"));
    public static readonly PhaseId GeneralizationPhase = new(Guid.Parse("60000000-0000-0000-0000-000000000003"));
    public static readonly MarkerId BaselineMarker = new(Guid.Parse("70000000-0000-0000-0000-000000000001"));
    public static readonly MarkerId InterventionMarker = new(Guid.Parse("70000000-0000-0000-0000-000000000002"));
    public static readonly MarkerId GeneralizationMarker = new(Guid.Parse("70000000-0000-0000-0000-000000000003"));
    public static readonly PointId BaselinePoint = new(Guid.Parse("80000000-0000-0000-0000-000000000001"));
    public static readonly PointId InterventionPoint = new(Guid.Parse("80000000-0000-0000-0000-000000000002"));
    public static readonly PointId GeneralizationPoint = new(Guid.Parse("80000000-0000-0000-0000-000000000003"));
    public static readonly DateTimeOffset CreatedUtc = new(2026, 1, 2, 3, 4, 5, TimeSpan.Zero);
}

internal static class IntegrationSmokeProjectFactory
{
    public static ProjectDocument Create(
        string sourcePath,
        string sourceSha256,
        bool userConfirmedCalibration = false,
        DateTimeOffset? modifiedUtc = null)
    {
        DateTimeOffset modified = (modifiedUtc ?? IntegrationSmokeIds.CreatedUtc).ToUniversalTime();
        PointModification correction = new(
            new AuditEventId(Guid.Parse("90000000-0000-0000-0000-000000000001")),
            modified,
            new PixelPoint(39, 121),
            new GraphPoint(3, 42),
            "user_corrected_marker");

        MarkerRecord[] markers =
        [
            Marker(IntegrationSmokeIds.BaselineMarker, IntegrationSmokeIds.BaselineSeries, 20, 150, MarkerFill.Filled, "●"),
            Marker(IntegrationSmokeIds.InterventionMarker, IntegrationSmokeIds.InterventionSeries, 40, 120, MarkerFill.Filled, "●", ReviewStatus.Corrected),
            Marker(IntegrationSmokeIds.GeneralizationMarker, IntegrationSmokeIds.GeneralizationSeries, 60, 100, MarkerFill.Open, "○"),
        ];
        PointRecord[] points =
        [
            Point(IntegrationSmokeIds.BaselinePoint, IntegrationSmokeIds.BaselineMarker, IntegrationSmokeIds.BaselineSeries, IntegrationSmokeIds.BaselinePhase, 20, 150, 1, 10, ReviewStatus.Accepted),
            Point(IntegrationSmokeIds.InterventionPoint, IntegrationSmokeIds.InterventionMarker, IntegrationSmokeIds.InterventionSeries, IntegrationSmokeIds.InterventionPhase, 40, 120, 3, 42, ReviewStatus.Corrected, [correction]),
            Point(IntegrationSmokeIds.GeneralizationPoint, IntegrationSmokeIds.GeneralizationMarker, IntegrationSmokeIds.GeneralizationSeries, IntegrationSmokeIds.GeneralizationPhase, 60, 100, 5, 55, ReviewStatus.Accepted),
        ];
        SeriesRecord[] series =
        [
            new(
                IntegrationSmokeIds.BaselineSeries,
                "●",
                MarkerShape.Circle,
                MarkerFill.Filled,
                "Shared baseline",
                SemanticRole.Baseline,
                null,
                [IntegrationSmokeIds.BaselinePoint],
                0.96,
                null,
                [],
                false),
            new(
                IntegrationSmokeIds.InterventionSeries,
                "●",
                MarkerShape.Circle,
                MarkerFill.Filled,
                "Intervention",
                SemanticRole.Intervention,
                "Intervention",
                [IntegrationSmokeIds.InterventionPoint],
                0.94,
                IntegrationSmokeIds.BaselineSeries,
                [IntegrationSmokeIds.GeneralizationSeries],
                true),
            new(
                IntegrationSmokeIds.GeneralizationSeries,
                "○",
                MarkerShape.Circle,
                MarkerFill.Open,
                "Generalization probe",
                SemanticRole.Generalization,
                "Generalization",
                [IntegrationSmokeIds.GeneralizationPoint],
                0.92,
                null,
                [],
                true),
        ];
        PhaseRecord[] phases =
        [
            new(IntegrationSmokeIds.BaselinePhase, 1, "a", PhaseNormalizedType.Baseline, "Baseline", 10, 29.9, null, IntegrationSmokeIds.InterventionPhase, 0.98, PhaseSource.ProfilePrior, false),
            new(IntegrationSmokeIds.InterventionPhase, 2, "b", PhaseNormalizedType.Intervention, "Intervention", 30, 49.9, IntegrationSmokeIds.BaselinePhase, IntegrationSmokeIds.GeneralizationPhase, 1, PhaseSource.Manual, true),
            new(IntegrationSmokeIds.GeneralizationPhase, 3, "g1", PhaseNormalizedType.Generalization, "Generalization", 50, 70, IntegrationSmokeIds.InterventionPhase, null, 1, PhaseSource.Manual, true),
        ];
        var calibration = new CalibrationRecord(
            IntegrationSmokeIds.Calibration,
            CalibrationStatus.Valid,
            [
                new CalibrationAnchor(CalibrationAnchorKind.Session1Y0, new PixelPoint(20, 180), new GraphPoint(1, 0), 1, null),
                new CalibrationAnchor(CalibrationAnchorKind.Session1Ymax, new PixelPoint(20, 20), new GraphPoint(1, 100), 1, null),
                new CalibrationAnchor(CalibrationAnchorKind.SessionmaxY0, new PixelPoint(100, 180), new GraphPoint(5, 0), 1, null),
            ],
            new SessionLatticeRecord(20, 20, 1, 5, 1, userConfirmedCalibration ? "manual" : "ocr_ticks"),
            userConfirmedCalibration,
            1,
            []);
        var panel = new PanelRecord(
            IntegrationSmokeIds.Panel,
            IntegrationSmokeIds.Source,
            null,
            "Panel 1",
            "Synthetic participant",
            new CropRectangle(0, 0, 120, 200),
            [],
            Json("{\"enabled\":true,\"scale\":2,\"source\":\"recorded_fake\"}"),
            calibration,
            [],
            markers,
            series,
            points,
            phases,
            new ExportSettingsRecord("printed_session", true, [IntegrationSmokeIds.InterventionSeries]),
            null);

        return new ProjectDocument(
            ProjectDocument.CurrentSchemaVersion,
            IntegrationSmokeIds.Project,
            "0.0.19",
            IntegrationSmokeIds.CreatedUtc,
            modified,
            ProjectSettings.Default,
            [new SourceReference(IntegrationSmokeIds.Source, SourceKind.Image, Path.GetFileName(sourcePath), sourcePath, sourceSha256, null)],
            [panel],
            AuditTrail.Empty);
    }

    public static string Sha256(byte[] bytes) =>
        Convert.ToHexStringLower(SHA256.HashData(bytes));

    private static MarkerRecord Marker(
        MarkerId id,
        SeriesId series,
        double x,
        double y,
        MarkerFill fill,
        string symbol,
        ReviewStatus status = ReviewStatus.Accepted) =>
        new(id, new PixelPoint(x, y), 4, MarkerShape.Circle, fill, symbol, 0, 0.99, 0.98, 0.98, null, series, SourceImageKind.Consensus, status);

    private static PointRecord Point(
        PointId id,
        MarkerId marker,
        SeriesId series,
        PhaseId phase,
        double pixelX,
        double pixelY,
        double graphX,
        double graphY,
        ReviewStatus status,
        IReadOnlyList<PointModification>? modifications = null) =>
        new(
            id,
            marker,
            series,
            phase,
            new PixelPoint(pixelX, pixelY),
            graphX,
            graphY,
            (int)graphX,
            graphX,
            null,
            PointXSource.Printed,
            0.99,
            0.98,
            0.97,
            "recorded_marker_fake",
            "recorded-1",
            status,
            modifications ?? []);

    private static JsonElement Json(string value)
    {
        using JsonDocument document = JsonDocument.Parse(value);
        return document.RootElement.Clone();
    }
}
