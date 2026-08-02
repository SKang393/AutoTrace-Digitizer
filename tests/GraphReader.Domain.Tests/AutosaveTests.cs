// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using GraphReader.Domain;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Domain.Tests;

[TestClass]
public sealed class AutosaveTests
{
    [TestMethod]
    public void SchedulerIsIneligibleWithoutCalibrationAndUsesFiveMinutesAfterEligibility()
    {
        var scheduler = new FiveMinuteAutosaveScheduler();
        DateTimeOffset observedUtc = TestProjectFactory.CreatedUtc.AddMinutes(1);
        ProjectDocument withoutCalibration = TestProjectFactory.Create(withCalibration: false);
        ProjectDocument withCalibration = TestProjectFactory.Create(withCalibration: true);

        AutosaveSchedule blocked = scheduler.CreateSchedule(withoutCalibration, observedUtc);
        AutosaveSchedule scheduled = scheduler.CreateSchedule(withCalibration, observedUtc);

        Assert.IsFalse(blocked.Eligibility.IsEligible);
        Assert.IsNull(blocked.DueUtc);
        Assert.AreEqual(TimeSpan.FromMinutes(5), scheduled.Interval);
        Assert.AreEqual(observedUtc.AddMinutes(5), scheduled.DueUtc);
        Assert.IsFalse(scheduler.IsDue(scheduled, observedUtc.AddMinutes(4).AddSeconds(59)));
        Assert.IsTrue(scheduler.IsDue(scheduled, observedUtc.AddMinutes(5)));
    }

    [TestMethod]
    public async Task TimerAutosaveNeverWritesBeforeCalibrationOrBeforeDueTime()
    {
        using var directory = new TemporaryDirectory();
        var scheduler = new FiveMinuteAutosaveScheduler();
        var service = new ProjectSnapshotService(directory.Path, scheduler: scheduler);
        DateTimeOffset observedUtc = TestProjectFactory.CreatedUtc.AddMinutes(1);
        ProjectDocument withoutCalibration = TestProjectFactory.Create(withCalibration: false);
        AutosaveSchedule blocked = scheduler.CreateSchedule(withoutCalibration, observedUtc);

        DomainResult<ProjectSnapshotReceipt> noCalibration = await service.SaveTimerSnapshotAsync(
            withoutCalibration,
            blocked,
            observedUtc.AddMinutes(10));

        Assert.IsFalse(noCalibration.IsSuccess);
        Assert.AreEqual("AUTOSAVE_NOT_ELIGIBLE", noCalibration.Errors[0].Code);
        Assert.IsFalse(File.Exists(service.GetSnapshotPath(withoutCalibration.ProjectId)));

        ProjectDocument calibrated = TestProjectFactory.Create(withCalibration: true);
        AutosaveSchedule schedule = scheduler.CreateSchedule(calibrated, observedUtc);
        DomainResult<ProjectSnapshotReceipt> tooEarly = await service.SaveTimerSnapshotAsync(
            calibrated,
            schedule,
            observedUtc.AddMinutes(4));
        DomainResult<ProjectSnapshotReceipt> due = await service.SaveTimerSnapshotAsync(
            calibrated,
            schedule,
            observedUtc.AddMinutes(5));

        Assert.IsFalse(tooEarly.IsSuccess);
        Assert.AreEqual("AUTOSAVE_NOT_DUE", tooEarly.Errors[0].Code);
        Assert.IsTrue(due.IsSuccess, FormatErrors(due.Errors));
        Assert.AreEqual(DomainEventKind.TimerAutosave, due.Value!.Snapshot.Audit.Events[^1].Kind);
    }

    [TestMethod]
    public async Task EveryMaterialEditTriggerWritesAnImmediateSnapshot()
    {
        using var directory = new TemporaryDirectory();
        var service = new ProjectSnapshotService(directory.Path);
        ProjectDocument current = TestProjectFactory.Create();
        SnapshotTrigger[] triggers =
        {
            SnapshotTrigger.CalibrationChanged,
            SnapshotTrigger.DetectionAccepted,
            SnapshotTrigger.PointEdited,
            SnapshotTrigger.PhaseEdited,
            SnapshotTrigger.ExportSettingsChanged
        };

        for (int index = 0; index < triggers.Length; index++)
        {
            DateTimeOffset eventUtc = TestProjectFactory.CreatedUtc.AddMinutes(index + 1);
            DomainResult<ProjectSnapshotReceipt> saved = await service.SaveEventSnapshotAsync(
                current,
                triggers[index],
                eventUtc,
                TestProjectFactory.PanelId,
                $"entity-{index}");
            Assert.IsTrue(saved.IsSuccess, FormatErrors(saved.Errors));
            current = saved.Value!.Snapshot;
        }

        Assert.AreEqual(triggers.Length, current.Audit.Events.Count);
        CollectionAssert.AreEqual(
            new[]
            {
                DomainEventKind.CalibrationChanged,
                DomainEventKind.DetectionAccepted,
                DomainEventKind.PointEdited,
                DomainEventKind.PhaseEdited,
                DomainEventKind.ExportSettingsChanged
            },
            current.Audit.Events.Select(auditEvent => auditEvent.Kind).ToArray());
        string snapshotPath = service.GetSnapshotPath(current.ProjectId);
        Assert.IsTrue(File.Exists(snapshotPath));
        DomainResult<ProjectDocument> reloaded = await new ProjectFileStore().LoadAsync(snapshotPath);
        Assert.IsTrue(reloaded.IsSuccess, FormatErrors(reloaded.Errors));
        Assert.AreEqual(triggers.Length, reloaded.Value!.Audit.Events.Count);
        Assert.AreEqual(TestProjectFactory.CreatedUtc.AddMinutes(5), reloaded.Value.Audit.LastAutosaveUtc);
    }

    private static string FormatErrors(IReadOnlyList<DomainError> errors) =>
        string.Join(Environment.NewLine, errors.Select(error => $"{error.Code}: {error.TechnicalMessage}"));
}
