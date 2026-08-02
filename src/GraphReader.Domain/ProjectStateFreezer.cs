// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.ObjectModel;
using System.Text.Json;

namespace GraphReader.Domain;

internal static class ProjectStateFreezer
{
    public static ProjectDocument Freeze(ProjectDocument project)
    {
        ArgumentNullException.ThrowIfNull(project);
        return project with
        {
            Settings = project.Settings with { },
            Sources = FreezeList(project.Sources.Select(FreezeSource)),
            Panels = FreezeList(project.Panels.Select(FreezePanel)),
            Audit = FreezeAudit(project.Audit)
        };
    }

    private static SourceReference FreezeSource(SourceReference source) => source with
    {
        ArticleMetadata = Clone(source.ArticleMetadata)
    };

    private static PanelRecord FreezePanel(PanelRecord panel) => panel with
    {
        Crop = panel.Crop with { },
        Transforms = FreezeList(panel.Transforms.Select(FreezeTransform)),
        Enhancement = Clone(panel.Enhancement),
        Calibration = panel.Calibration is null ? null : FreezeCalibration(panel.Calibration),
        OcrRegions = FreezeList(panel.OcrRegions.Select(FreezeOcr)),
        Markers = FreezeList(panel.Markers.Select(FreezeMarker)),
        Series = FreezeList(panel.Series.Select(FreezeSeries)),
        Points = FreezeList(panel.Points.Select(FreezePoint)),
        Phases = FreezeList(panel.Phases.Select(phase => phase with { })),
        ExportSettings = panel.ExportSettings is null
            ? null
            : panel.ExportSettings with
            {
                SelectedSeriesIds = FreezeList(panel.ExportSettings.SelectedSeriesIds)
            },
        Validation = Clone(panel.Validation)
    };

    private static TransformRecord FreezeTransform(TransformRecord transform) => transform with
    {
        Matrix3x3 = FreezeList(transform.Matrix3x3),
        InverseMatrix3x3 = transform.InverseMatrix3x3 is null
            ? null
            : FreezeList(transform.InverseMatrix3x3),
        Parameters = Clone(transform.Parameters)
    };

    private static CalibrationRecord FreezeCalibration(CalibrationRecord calibration) => calibration with
    {
        Anchors = FreezeList(calibration.Anchors.Select(anchor => anchor with
        {
            Screen = anchor.Screen with { },
            Graph = anchor.Graph with { }
        })),
        SessionLattice = calibration.SessionLattice is null ? null : calibration.SessionLattice with { },
        Reasons = FreezeList(calibration.Reasons)
    };

    private static OcrEvidence FreezeOcr(OcrEvidence evidence) => evidence with
    {
        Polygon = FreezeList(evidence.Polygon.Select(point => point with { })),
        Alternatives = FreezeList(evidence.Alternatives.Select(alternative => alternative with { }))
    };

    private static MarkerRecord FreezeMarker(MarkerRecord marker) => marker with
    {
        Center = marker.Center with { },
        Embedding = marker.Embedding is null ? null : FreezeList(marker.Embedding)
    };

    private static SeriesRecord FreezeSeries(SeriesRecord series) => series with
    {
        PointIds = FreezeList(series.PointIds),
        ApplicableProbeSeriesIds = FreezeList(series.ApplicableProbeSeriesIds)
    };

    private static PointRecord FreezePoint(PointRecord point) => point with
    {
        OriginalPixel = point.OriginalPixel with { },
        ModificationHistory = FreezeList(point.ModificationHistory.Select(modification => modification with
        {
            PreviousPixel = modification.PreviousPixel is null ? null : modification.PreviousPixel with { },
            PreviousGraph = modification.PreviousGraph is null ? null : modification.PreviousGraph with { }
        }))
    };

    private static AuditTrail FreezeAudit(AuditTrail audit) => audit with
    {
        Events = FreezeList(audit.Events.Select(auditEvent => auditEvent with
        {
            Details = Clone(auditEvent.Details)
        }))
    };

    private static JsonElement Clone(JsonElement value) =>
        value.ValueKind == JsonValueKind.Undefined ? value : value.Clone();

    private static JsonElement? Clone(JsonElement? value) =>
        value is null ? null : Clone(value.Value);

    private static ReadOnlyCollection<T> FreezeList<T>(IEnumerable<T> source) =>
        Array.AsReadOnly(source.ToArray());
}
