// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Diagnostics.CodeAnalysis;

namespace GraphReader.Domain;

public enum SnapshotTrigger
{
    CalibrationChanged,
    DetectionAccepted,
    PointEdited,
    PhaseEdited,
    ExportSettingsChanged,
    Timer
}

public sealed record AutosaveEligibility(bool IsEligible, string Reason);

public sealed record AutosaveSchedule(
    AutosaveEligibility Eligibility,
    TimeSpan Interval,
    DateTimeOffset? DueUtc);

public sealed class FiveMinuteAutosaveScheduler
{
    public static TimeSpan DefaultInterval { get; } = TimeSpan.FromMinutes(5);

    [SuppressMessage("Performance", "CA1822:Mark members as static", Justification = "The scheduler remains replaceable at the application boundary.")]
    public AutosaveEligibility Evaluate(ProjectDocument project)
    {
        ArgumentNullException.ThrowIfNull(project);
        bool hasCalibration = project.Panels.Any(panel => panel.Calibration is not null);
        return hasCalibration
            ? new AutosaveEligibility(true, "calibration_exists")
            : new AutosaveEligibility(false, "calibration_required");
    }

    public AutosaveSchedule CreateSchedule(
        ProjectDocument project,
        DateTimeOffset eligibilityObservedUtc)
    {
        ArgumentNullException.ThrowIfNull(project);
        DateTimeOffset observedUtc = eligibilityObservedUtc.ToUniversalTime();
        AutosaveEligibility eligibility = Evaluate(project);
        if (!eligibility.IsEligible)
        {
            return new AutosaveSchedule(eligibility, DefaultInterval, DueUtc: null);
        }

        DateTimeOffset dueUtc = project.Audit.LastAutosaveUtc is { } lastAutosave
            ? lastAutosave.ToUniversalTime() + DefaultInterval
            : observedUtc + DefaultInterval;
        return new AutosaveSchedule(eligibility, DefaultInterval, dueUtc);
    }

    [SuppressMessage("Performance", "CA1822:Mark members as static", Justification = "The scheduler remains replaceable at the application boundary.")]
    public bool IsDue(AutosaveSchedule schedule, DateTimeOffset nowUtc)
    {
        ArgumentNullException.ThrowIfNull(schedule);
        return schedule.Eligibility.IsEligible &&
               schedule.DueUtc is { } dueUtc &&
               nowUtc.ToUniversalTime() >= dueUtc;
    }
}

public sealed record ProjectSnapshotReceipt(
    SnapshotTrigger Trigger,
    string SnapshotPath,
    ProjectDocument Snapshot,
    ProjectSaveReceipt SaveReceipt);

public sealed class ProjectSnapshotService
{
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> SnapshotLocks =
        new(StringComparer.OrdinalIgnoreCase);
    private readonly string _autosaveRoot;
    private readonly FiveMinuteAutosaveScheduler _scheduler;
    private readonly ProjectFileStore _store;

    public ProjectSnapshotService(
        string autosaveRoot,
        ProjectFileStore? store = null,
        FiveMinuteAutosaveScheduler? scheduler = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(autosaveRoot);
        _autosaveRoot = Path.GetFullPath(autosaveRoot);
        _store = store ?? new ProjectFileStore();
        _scheduler = scheduler ?? new FiveMinuteAutosaveScheduler();
    }

    public Task<DomainResult<ProjectSnapshotReceipt>> SaveEventSnapshotAsync(
        ProjectDocument project,
        SnapshotTrigger trigger,
        DateTimeOffset occurredUtc,
        PanelId? panelId = null,
        string? entityId = null,
        CancellationToken cancellationToken = default)
    {
        if (trigger == SnapshotTrigger.Timer)
        {
            throw new ArgumentException("Use SaveTimerSnapshotAsync for timer autosaves.", nameof(trigger));
        }

        return SaveSnapshotAsync(
            project,
            trigger,
            occurredUtc,
            panelId,
            entityId,
            requireTimerDue: false,
            schedule: null,
            cancellationToken);
    }

    public Task<DomainResult<ProjectSnapshotReceipt>> SaveTimerSnapshotAsync(
        ProjectDocument project,
        AutosaveSchedule schedule,
        DateTimeOffset occurredUtc,
        CancellationToken cancellationToken = default) =>
        SaveSnapshotAsync(
            project,
            SnapshotTrigger.Timer,
            occurredUtc,
            panelId: null,
            entityId: null,
            requireTimerDue: true,
            schedule,
            cancellationToken);

    public string GetSnapshotPath(ProjectId projectId) =>
        Path.Combine(_autosaveRoot, $"{projectId.Value:N}.autosave.garproj");

    private async Task<DomainResult<ProjectSnapshotReceipt>> SaveSnapshotAsync(
        ProjectDocument project,
        SnapshotTrigger trigger,
        DateTimeOffset occurredUtc,
        PanelId? panelId,
        string? entityId,
        bool requireTimerDue,
        AutosaveSchedule? schedule,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(project);
        cancellationToken.ThrowIfCancellationRequested();
        ProjectDocument frozenProject = ProjectStateFreezer.Freeze(project);
        DateTimeOffset utc = occurredUtc.ToUniversalTime();
        AutosaveEligibility eligibility = _scheduler.Evaluate(frozenProject);
        if (!eligibility.IsEligible)
        {
            return DomainResult<ProjectSnapshotReceipt>.Failure(new DomainError(
                "AUTOSAVE_NOT_ELIGIBLE",
                DomainErrorSeverity.Warning,
                "Errors.AutosaveNotEligible",
                "Autosave is disabled until at least one panel has calibration.",
                Recoverable: true,
                "calibrate_panel"));
        }

        if (requireTimerDue && (schedule is null || !_scheduler.IsDue(schedule, utc)))
        {
            return DomainResult<ProjectSnapshotReceipt>.Failure(new DomainError(
                "AUTOSAVE_NOT_DUE",
                DomainErrorSeverity.Warning,
                "Errors.AutosaveNotDue",
                "The five-minute autosave interval has not elapsed.",
                Recoverable: true,
                "wait"));
        }

        DomainEventKind eventKind = trigger switch
        {
            SnapshotTrigger.CalibrationChanged => DomainEventKind.CalibrationChanged,
            SnapshotTrigger.DetectionAccepted => DomainEventKind.DetectionAccepted,
            SnapshotTrigger.PointEdited => DomainEventKind.PointEdited,
            SnapshotTrigger.PhaseEdited => DomainEventKind.PhaseEdited,
            SnapshotTrigger.ExportSettingsChanged => DomainEventKind.ExportSettingsChanged,
            SnapshotTrigger.Timer => DomainEventKind.TimerAutosave,
            _ => throw new ArgumentOutOfRangeException(nameof(trigger), trigger, "Unknown snapshot trigger.")
        };

        var auditEvent = new AuditEvent(
            AuditEventId.New(),
            utc,
            eventKind,
            panelId,
            entityId,
            Note: null,
            Details: null);
        string snapshotPath = GetSnapshotPath(frozenProject.ProjectId);
        SemaphoreSlim snapshotLock = SnapshotLocks.GetOrAdd(snapshotPath, static _ => new SemaphoreSlim(1, 1));
        await snapshotLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ProjectDocument? existingSnapshot = null;
            if (File.Exists(snapshotPath))
            {
                DomainResult<ProjectDocument> existing = await _store.LoadAsync(
                    snapshotPath,
                    cancellationToken).ConfigureAwait(false);
                if (!existing.IsSuccess || existing.Value is null)
                {
                    return DomainResult<ProjectSnapshotReceipt>.Failure(existing.Errors);
                }

                if (existing.Value.ProjectId != frozenProject.ProjectId)
                {
                    return DomainResult<ProjectSnapshotReceipt>.Failure(DomainErrors.CorruptProject(
                        $"Existing autosave '{snapshotPath}' belongs to a different project."));
                }

                existingSnapshot = existing.Value;
            }

            DateTimeOffset? latestAutosaveUtc = Latest(
                frozenProject.Audit.LastAutosaveUtc,
                existingSnapshot?.Audit.LastAutosaveUtc);
            if (requireTimerDue &&
                latestAutosaveUtc is { } lastAutosaveUtc &&
                utc < lastAutosaveUtc + FiveMinuteAutosaveScheduler.DefaultInterval)
            {
                return AutosaveNotDue();
            }

            IReadOnlyList<AuditEvent> mergedEvents = MergeAuditEvents(
                existingSnapshot?.Audit.Events,
                frozenProject.Audit.Events,
                auditEvent);
            DateTimeOffset snapshotModifiedUtc = Latest(
                utc,
                Latest(frozenProject.ModifiedUtc, existingSnapshot?.ModifiedUtc) ?? utc) ?? utc;
            var updatedAudit = new AuditTrail(
                mergedEvents,
                Latest(utc, latestAutosaveUtc) ?? utc);
            ProjectDocument snapshot = ProjectStateFreezer.Freeze(frozenProject with
            {
                ModifiedUtc = snapshotModifiedUtc,
                Audit = updatedAudit
            });

            DomainResult<ProjectSaveReceipt> saved = await _store.SaveAsync(
                snapshot,
                snapshotPath,
                cancellationToken).ConfigureAwait(false);
            if (!saved.IsSuccess || saved.Value is null)
            {
                return DomainResult<ProjectSnapshotReceipt>.Failure(saved.Errors);
            }

            return DomainResult<ProjectSnapshotReceipt>.Success(new ProjectSnapshotReceipt(
                trigger,
                snapshotPath,
                snapshot,
                saved.Value));
        }
        finally
        {
            snapshotLock.Release();
        }
    }

    private static DomainResult<ProjectSnapshotReceipt> AutosaveNotDue() =>
        DomainResult<ProjectSnapshotReceipt>.Failure(new DomainError(
            "AUTOSAVE_NOT_DUE",
            DomainErrorSeverity.Warning,
            "Errors.AutosaveNotDue",
            "The five-minute autosave interval has not elapsed since the most recent snapshot.",
            Recoverable: true,
            "wait"));

    private static ReadOnlyCollection<AuditEvent> MergeAuditEvents(
        IReadOnlyList<AuditEvent>? existing,
        IReadOnlyList<AuditEvent> incoming,
        AuditEvent appended)
    {
        var byId = new Dictionary<AuditEventId, AuditEvent>();
        if (existing is not null)
        {
            foreach (AuditEvent auditEvent in existing)
            {
                byId[auditEvent.EventId] = auditEvent;
            }
        }

        foreach (AuditEvent auditEvent in incoming)
        {
            byId[auditEvent.EventId] = auditEvent;
        }

        byId[appended.EventId] = appended;
        return Array.AsReadOnly(byId.Values
            .OrderBy(auditEvent => auditEvent.OccurredUtc)
            .ThenBy(auditEvent => auditEvent.EventId.Value)
            .ToArray());
    }

    private static DateTimeOffset? Latest(DateTimeOffset? left, DateTimeOffset? right)
    {
        if (left is null)
        {
            return right;
        }

        if (right is null)
        {
            return left;
        }

        return left >= right ? left : right;
    }
}
