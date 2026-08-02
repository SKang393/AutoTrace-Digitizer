// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
using System.Text.RegularExpressions;

using System.Diagnostics.CodeAnalysis;

namespace GraphReader.Domain;

public sealed class ProjectValidator
{
    private const string VersionPattern = "^[0-9]{1,2}\\.[0-9]{1,2}\\.[0-9]{1,2}$";

    [SuppressMessage("Performance", "CA1822:Mark members as static", Justification = "The validator is an injectable persistence policy.")]
    public IReadOnlyList<DomainError> Validate(ProjectDocument project)
    {
        ArgumentNullException.ThrowIfNull(project);
        var errors = new List<DomainError>();

        if (project.SchemaVersion != ProjectDocument.CurrentSchemaVersion)
        {
            errors.Add(DomainErrors.InvalidProject(
                $"Schema version {project.SchemaVersion} is not supported by version {ProjectDocument.CurrentSchemaVersion}."));
        }

        ValidateId(project.ProjectId.Value, "project_id", errors);
        if (string.IsNullOrWhiteSpace(project.AppVersion) ||
            !Regex.IsMatch(
                project.AppVersion,
                VersionPattern,
                RegexOptions.CultureInvariant,
                TimeSpan.FromSeconds(1)))
        {
            errors.Add(DomainErrors.InvalidProject($"App version '{project.AppVersion}' is not a valid x.y.z version."));
        }

        if (project.CreatedUtc.Offset != TimeSpan.Zero || project.ModifiedUtc.Offset != TimeSpan.Zero)
        {
            errors.Add(DomainErrors.InvalidProject("Project timestamps must be expressed in UTC."));
        }

        if (project.ModifiedUtc < project.CreatedUtc)
        {
            errors.Add(DomainErrors.InvalidProject("Project modified_utc precedes created_utc."));
        }

        if (project.Settings is null)
        {
            errors.Add(DomainErrors.InvalidProject("Project settings are missing."));
        }
        else if (project.Settings.DefaultEnhancementScale is not (1 or 2 or 4))
        {
            errors.Add(DomainErrors.InvalidProject("Default enhancement scale must be 1, 2, or 4."));
        }

        if (project.Settings is not null && string.IsNullOrWhiteSpace(project.Settings.Locale))
        {
            errors.Add(DomainErrors.InvalidProject("Project locale is empty."));
        }

        ValidateSources(project.Sources, errors);
        ValidatePanels(project.Panels, project.Sources, errors);
        ValidateAudit(project.Audit, errors);
        return errors.AsReadOnly();
    }

    private static void ValidateSources(
        IReadOnlyList<SourceReference>? sources,
        List<DomainError> errors)
    {
        if (sources is null)
        {
            errors.Add(DomainErrors.InvalidProject("Project sources are missing."));
            return;
        }

        var ids = new HashSet<SourceId>();
        foreach (SourceReference source in sources)
        {
            ValidateId(source.SourceId.Value, "source_id", errors);
            if (!ids.Add(source.SourceId))
            {
                errors.Add(DomainErrors.InvalidProject($"Duplicate source_id '{source.SourceId.Value}'."));
            }

            if (string.IsNullOrWhiteSpace(source.DisplayName))
            {
                errors.Add(DomainErrors.InvalidProject($"Source '{source.SourceId.Value}' has no display name."));
            }

            if (!IsSha256(source.Sha256))
            {
                errors.Add(DomainErrors.InvalidProject($"Source '{source.SourceId.Value}' has an invalid SHA-256 value."));
            }

            if (source.ArticleMetadata is { } metadata && metadata.ValueKind != JsonValueKind.Object)
            {
                errors.Add(DomainErrors.InvalidProject($"Source '{source.SourceId.Value}' article metadata is not an object."));
            }
        }
    }

    private static void ValidatePanels(
        IReadOnlyList<PanelRecord>? panels,
        IReadOnlyList<SourceReference>? sources,
        List<DomainError> errors)
    {
        if (panels is null)
        {
            errors.Add(DomainErrors.InvalidProject("Project panels are missing."));
            return;
        }

        var sourceIds = sources is null
            ? new HashSet<SourceId>()
            : sources.Select(source => source.SourceId).ToHashSet();
        var panelIds = new HashSet<PanelId>();

        foreach (PanelRecord panel in panels)
        {
            ValidateId(panel.PanelId.Value, "panel_id", errors);
            if (!panelIds.Add(panel.PanelId))
            {
                errors.Add(DomainErrors.InvalidProject($"Duplicate panel_id '{panel.PanelId.Value}'."));
            }

            if (!sourceIds.Contains(panel.SourceId))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Panel '{panel.PanelId.Value}' references missing source '{panel.SourceId.Value}'."));
            }

            if (panel.PageNumber is < 1)
            {
                errors.Add(DomainErrors.InvalidProject($"Panel '{panel.PanelId.Value}' has an invalid page number."));
            }

            if (!IsFinite(panel.Crop.X) || !IsFinite(panel.Crop.Y) ||
                !IsPositiveFinite(panel.Crop.Width) || !IsPositiveFinite(panel.Crop.Height))
            {
                errors.Add(DomainErrors.InvalidProject($"Panel '{panel.PanelId.Value}' has an invalid crop rectangle."));
            }

            ValidateTransforms(panel, errors);
            ValidateCalibration(panel, errors);
            ValidateOcr(panel, errors);
            ValidateMarkersSeriesPointsAndPhases(panel, errors);
        }
    }

    private static void ValidateTransforms(PanelRecord panel, List<DomainError> errors)
    {
        var ids = new HashSet<TransformId>();
        foreach (TransformRecord transform in panel.Transforms)
        {
            ValidateId(transform.TransformId.Value, "transform_id", errors);
            if (!ids.Add(transform.TransformId))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Panel '{panel.PanelId.Value}' contains a duplicate transform ID."));
            }

            if (transform.Matrix3x3.Count != 9 || transform.Matrix3x3.Any(value => !IsFinite(value)))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Transform '{transform.TransformId.Value}' must contain a finite 3x3 matrix."));
            }

            if (transform.InverseMatrix3x3 is { } inverse &&
                (inverse.Count != 9 || inverse.Any(value => !IsFinite(value))))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Transform '{transform.TransformId.Value}' has an invalid inverse matrix."));
            }

            if (!transform.Lossy && transform.InverseMatrix3x3 is null)
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Transform '{transform.TransformId.Value}' is not lossy but has no inverse."));
            }

            if (transform.Parameters.ValueKind != JsonValueKind.Object)
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Transform '{transform.TransformId.Value}' parameters are not an object."));
            }
        }
    }

    private static void ValidateCalibration(PanelRecord panel, List<DomainError> errors)
    {
        if (panel.Calibration is not { } calibration)
        {
            return;
        }

        ValidateId(calibration.CalibrationId.Value, "calibration_id", errors);
        ValidateConfidence(calibration.Confidence, "calibration confidence", errors);
        foreach (CalibrationAnchor anchor in calibration.Anchors)
        {
            ValidateConfidence(anchor.Confidence, "calibration anchor confidence", errors);
            if (!IsFinite(anchor.Screen.X) || !IsFinite(anchor.Screen.Y) ||
                !IsFinite(anchor.Graph.X) || !IsFinite(anchor.Graph.Y))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Calibration '{calibration.CalibrationId.Value}' contains a non-finite anchor."));
            }

            if (anchor.EvidenceRegionId is { } evidenceRegionId &&
                panel.OcrRegions.All(evidence => evidence.RegionId != evidenceRegionId))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Calibration '{calibration.CalibrationId.Value}' references missing OCR evidence '{evidenceRegionId.Value}'."));
            }
        }

        if (calibration.SessionLattice is { } lattice &&
            (!IsFinite(lattice.Session1PixelX) || !IsPositiveFinite(lattice.PitchPixels)))
        {
            errors.Add(DomainErrors.InvalidProject(
                $"Calibration '{calibration.CalibrationId.Value}' has an invalid session lattice."));
        }
    }

    private static void ValidateOcr(PanelRecord panel, List<DomainError> errors)
    {
        var ids = new HashSet<OcrRegionId>();
        foreach (OcrEvidence evidence in panel.OcrRegions)
        {
            ValidateId(evidence.RegionId.Value, "region_id", errors);
            if (!ids.Add(evidence.RegionId))
            {
                errors.Add(DomainErrors.InvalidProject($"Panel '{panel.PanelId.Value}' has duplicate OCR evidence."));
            }

            ValidateConfidence(evidence.Confidence, "OCR confidence", errors);
            foreach (OcrAlternative alternative in evidence.Alternatives)
            {
                ValidateConfidence(alternative.Confidence, "OCR alternative confidence", errors);
            }
        }
    }

    private static void ValidateMarkersSeriesPointsAndPhases(
        PanelRecord panel,
        List<DomainError> errors)
    {
        var markerIds = new HashSet<MarkerId>();
        foreach (MarkerRecord marker in panel.Markers)
        {
            ValidateId(marker.MarkerId.Value, "marker_id", errors);
            if (!markerIds.Add(marker.MarkerId))
            {
                errors.Add(DomainErrors.InvalidProject($"Panel '{panel.PanelId.Value}' has duplicate markers."));
            }

            ValidateConfidence(marker.ArtifactProbability, "artifact probability", errors);
            ValidateConfidence(marker.CenterConfidence, "marker center confidence", errors);
            ValidateConfidence(marker.ShapeConfidence, "marker shape confidence", errors);
            ValidateConfidence(marker.FillConfidence, "marker fill confidence", errors);
        }

        var seriesIds = new HashSet<SeriesId>();
        foreach (SeriesRecord series in panel.Series)
        {
            ValidateId(series.SeriesId.Value, "series_id", errors);
            if (!seriesIds.Add(series.SeriesId))
            {
                errors.Add(DomainErrors.InvalidProject($"Panel '{panel.PanelId.Value}' has duplicate series."));
            }

            ValidateConfidence(series.Confidence, "series confidence", errors);
        }

        foreach (MarkerRecord marker in panel.Markers)
        {
            if (marker.CandidateSeriesId is { } candidateSeriesId && !seriesIds.Contains(candidateSeriesId))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Marker '{marker.MarkerId.Value}' references missing candidate series '{candidateSeriesId.Value}'."));
            }
        }

        foreach (SeriesRecord series in panel.Series)
        {
            if (series.SharedBaselineSeriesId is { } baselineSeriesId && !seriesIds.Contains(baselineSeriesId))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Series '{series.SeriesId.Value}' references missing shared baseline series '{baselineSeriesId.Value}'."));
            }

            if (series.ApplicableProbeSeriesIds.Any(probeSeriesId => !seriesIds.Contains(probeSeriesId)))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Series '{series.SeriesId.Value}' references a missing applicable probe series."));
            }
        }

        var phaseIds = new HashSet<PhaseId>();
        foreach (PhaseRecord phase in panel.Phases)
        {
            ValidateId(phase.PhaseId.Value, "phase_id", errors);
            if (!phaseIds.Add(phase.PhaseId))
            {
                errors.Add(DomainErrors.InvalidProject($"Panel '{panel.PanelId.Value}' has duplicate phases."));
            }

            ValidateConfidence(phase.Confidence, "phase confidence", errors);
        }

        foreach (PhaseRecord phase in panel.Phases)
        {
            if (phase.BoundaryLeftId is { } leftId && !phaseIds.Contains(leftId))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Phase '{phase.PhaseId.Value}' references missing left boundary phase '{leftId.Value}'."));
            }

            if (phase.BoundaryRightId is { } rightId && !phaseIds.Contains(rightId))
            {
                errors.Add(DomainErrors.InvalidProject(
                    $"Phase '{phase.PhaseId.Value}' references missing right boundary phase '{rightId.Value}'."));
            }
        }

        var pointIds = new HashSet<PointId>();
        foreach (PointRecord point in panel.Points)
        {
            ValidateId(point.PointId.Value, "point_id", errors);
            if (!pointIds.Add(point.PointId))
            {
                errors.Add(DomainErrors.InvalidProject($"Panel '{panel.PanelId.Value}' has duplicate points."));
            }

            if (point.MarkerId is { } markerId && !markerIds.Contains(markerId))
            {
                errors.Add(DomainErrors.InvalidProject($"Point '{point.PointId.Value}' references a missing marker."));
            }

            if (point.SeriesId is { } seriesId && !seriesIds.Contains(seriesId))
            {
                errors.Add(DomainErrors.InvalidProject($"Point '{point.PointId.Value}' references a missing series."));
            }

            if (point.PhaseId is { } phaseId && !phaseIds.Contains(phaseId))
            {
                errors.Add(DomainErrors.InvalidProject($"Point '{point.PointId.Value}' references a missing phase."));
            }

            if (point.ObservationIndex < 1)
            {
                errors.Add(DomainErrors.InvalidProject($"Point '{point.PointId.Value}' has an invalid observation index."));
            }

            ValidateConfidence(point.XConfidence, "point x confidence", errors);
            ValidateConfidence(point.YConfidence, "point y confidence", errors);
            ValidateConfidence(point.PointConfidence, "point confidence", errors);
        }

        foreach (SeriesRecord series in panel.Series)
        {
            if (series.PointIds.Any(pointId => !pointIds.Contains(pointId)))
            {
                errors.Add(DomainErrors.InvalidProject($"Series '{series.SeriesId.Value}' references a missing point."));
            }
        }
    }

    private static void ValidateAudit(AuditTrail? audit, List<DomainError> errors)
    {
        if (audit is null)
        {
            errors.Add(DomainErrors.InvalidProject("Project audit is missing."));
            return;
        }

        var eventIds = new HashSet<AuditEventId>();
        foreach (AuditEvent auditEvent in audit.Events)
        {
            ValidateId(auditEvent.EventId.Value, "event_id", errors);
            if (!eventIds.Add(auditEvent.EventId))
            {
                errors.Add(DomainErrors.InvalidProject("Project audit contains a duplicate event ID."));
            }

            if (auditEvent.OccurredUtc.Offset != TimeSpan.Zero)
            {
                errors.Add(DomainErrors.InvalidProject("Audit event timestamps must be expressed in UTC."));
            }
        }

        if (audit.LastAutosaveUtc is { } lastAutosave && lastAutosave.Offset != TimeSpan.Zero)
        {
            errors.Add(DomainErrors.InvalidProject("last_autosave_utc must be expressed in UTC."));
        }
    }

    private static void ValidateId(Guid id, string name, List<DomainError> errors)
    {
        if (id == Guid.Empty)
        {
            errors.Add(DomainErrors.InvalidProject($"{name} cannot be an empty UUID."));
        }
    }

    private static void ValidateConfidence(double value, string name, List<DomainError> errors)
    {
        if (!IsFinite(value) || value is < 0 or > 1)
        {
            errors.Add(DomainErrors.InvalidProject($"{name} must be between 0 and 1."));
        }
    }

    private static bool IsSha256(string value)
    {
        if (value is null || value.Length != 64)
        {
            return false;
        }

        foreach (char character in value)
        {
            if (!char.IsAsciiHexDigit(character))
            {
                return false;
            }
        }

        return true;
    }

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    private static bool IsPositiveFinite(double value) => value > 0 && IsFinite(value);
}
