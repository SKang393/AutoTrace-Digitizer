// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json;
using System.Text.Json.Nodes;
using GraphReader.Domain;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Domain.Tests;

[TestClass]
public sealed class DomainRepairRegressionTests
{
    [TestMethod]
    public async Task InvalidUtf8IsCorruptAndDoesNotStopRecoveryDiscovery()
    {
        using var directory = new TemporaryDirectory();
        string autosaveRoot = Path.Combine(directory.Path, "Autosave");
        ProjectDocument project = TestProjectFactory.Create();
        var snapshots = new ProjectSnapshotService(autosaveRoot);
        string validPath = snapshots.GetSnapshotPath(project.ProjectId);
        DomainResult<ProjectSaveReceipt> saved = await new ProjectFileStore().SaveAsync(project, validPath);
        Assert.IsTrue(saved.IsSuccess, FormatErrors(saved.Errors));
        string invalidPath = Path.Combine(autosaveRoot, "invalid-utf8.autosave.garproj");
        await File.WriteAllBytesAsync(invalidPath, new byte[] { 0x7b, 0x22, 0xc3, 0x28, 0x22, 0x7d });

        DomainResult<ProjectDocument> invalid = await new ProjectFileStore().LoadAsync(invalidPath);
        DomainResult<RecoveryDiscoveryReport> discovery = await new ProjectRecoveryService().DiscoverAsync(
            autosaveRoot,
            project,
            originalProjectPath: null);

        Assert.IsFalse(invalid.IsSuccess);
        Assert.AreEqual("PROJECT_CORRUPT", invalid.Errors[0].Code);
        Assert.IsTrue(discovery.IsSuccess, FormatErrors(discovery.Errors));
        Assert.AreEqual(1, discovery.Value!.Candidates.Count);
        Assert.AreEqual(1, discovery.Value.RejectedCandidates.Count);
        Assert.AreEqual("PROJECT_CORRUPT", discovery.Value.RejectedCandidates[0].Errors[0].Code);
    }

    [TestMethod]
    public async Task RestartRecoveryDiscoversAllAutosavesWhenPrimaryIsActuallyCorrupt()
    {
        using var directory = new TemporaryDirectory();
        string autosaveRoot = Path.Combine(directory.Path, "Autosave");
        string primaryPath = Path.Combine(directory.Path, "primary.garproj");
        const string corruptPrimary = "{\"schema_version\":1,\"project_id\":";
        await File.WriteAllTextAsync(primaryPath, corruptPrimary);
        ProjectDocument first = TestProjectFactory.Create(modifiedUtc: TestProjectFactory.CreatedUtc.AddMinutes(2));
        ProjectDocument second = first with
        {
            ProjectId = new ProjectId(Guid.Parse("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")),
            ModifiedUtc = first.ModifiedUtc.AddMinutes(1)
        };
        var snapshotPaths = new ProjectSnapshotService(autosaveRoot);
        var store = new ProjectFileStore();
        Assert.IsTrue((await store.SaveAsync(first, snapshotPaths.GetSnapshotPath(first.ProjectId))).IsSuccess);
        Assert.IsTrue((await store.SaveAsync(second, snapshotPaths.GetSnapshotPath(second.ProjectId))).IsSuccess);
        var recovery = new ProjectRecoveryService(store);

        DomainResult<RecoveryDiscoveryReport> result = await recovery.DiscoverForProjectPathAsync(
            autosaveRoot,
            primaryPath);

        Assert.IsTrue(result.IsSuccess, FormatErrors(result.Errors));
        Assert.AreEqual(1, result.Value!.PrimaryErrors.Count);
        Assert.AreEqual("PROJECT_CORRUPT", result.Value.PrimaryErrors[0].Code);
        Assert.AreEqual(2, result.Value.Candidates.Count);
        Assert.IsTrue(result.Value.Candidates.All(candidate => !candidate.ProjectIdentityVerified));
        Assert.IsTrue(result.Value.Candidates.All(candidate => candidate.Recommendation == RecoveryRecommendation.Inspect));
        Assert.IsTrue(result.Value.Candidates.All(candidate => candidate.IsNewerThanOriginal is null));

        string recoveredPath = Path.Combine(directory.Path, "inspected-recovery.garproj");
        DomainResult<ProjectSaveReceipt> recovered = await recovery.RecoverToNewFileAsync(
            result.Value.Candidates[0].AutosavePath,
            recoveredPath);
        Assert.IsTrue(recovered.IsSuccess, FormatErrors(recovered.Errors));
        Assert.AreEqual(corruptPrimary, await File.ReadAllTextAsync(primaryPath));
    }

    [TestMethod]
    public void SchemaPreflightDefaultsOptionalSettingsAndModelOnlyCollections()
    {
        JsonObject minimal = CreateMinimalSchemaProject();

        DomainResult<ProjectDocument> loaded = new ProjectJsonSerializer().Deserialize(minimal.ToJsonString());

        Assert.IsTrue(loaded.IsSuccess, FormatErrors(loaded.Errors));
        Assert.AreEqual(ProjectSettings.Default, loaded.Value!.Settings);
        Assert.AreEqual(DateTimeOffset.UnixEpoch, loaded.Value.CreatedUtc);
        Assert.AreEqual(0, loaded.Value.Panels[0].Markers.Count);
        Assert.AreEqual(0, loaded.Value.Panels[0].OcrRegions.Count);
    }

    [TestMethod]
    public void SchemaPreflightRejectsMissingRequiredAndUnknownClosedProperties()
    {
        string[] missingPaths =
        {
            "schema_version",
            "project_id",
            "app_version",
            "sources",
            "panels",
            "audit",
            "source.source_id",
            "source.kind",
            "source.sha256",
            "source.display_name",
            "panel.panel_id",
            "panel.source_id",
            "panel.crop",
            "panel.transforms",
            "panel.series",
            "panel.points",
            "panel.phases",
            "crop.x",
            "crop.y",
            "crop.width",
            "crop.height",
            "audit.events"
        };

        foreach (string missingPath in missingPaths)
        {
            JsonObject candidate = (JsonObject)CreateMinimalSchemaProject().DeepClone();
            RemovePath(candidate, missingPath);
            DomainResult<ProjectDocument> result = new ProjectJsonSerializer().Deserialize(candidate.ToJsonString());
            Assert.IsFalse(result.IsSuccess, $"Missing '{missingPath}' was accepted.");
            Assert.IsTrue(result.Errors.Any(error => error.Code == "PROJECT_CORRUPT"));
        }

        JsonObject unknownRoot = CreateMinimalSchemaProject();
        unknownRoot["unexpected"] = true;
        AssertSchemaCorrupt(unknownRoot, "unknown root property");

        JsonObject unknownSettings = CreateMinimalSchemaProject();
        unknownSettings["settings"] = new JsonObject { ["unexpected"] = true };
        AssertSchemaCorrupt(unknownSettings, "unknown settings property");

        JsonObject unknownCrop = CreateMinimalSchemaProject();
        GetCrop(unknownCrop)["unexpected"] = true;
        AssertSchemaCorrupt(unknownCrop, "unknown crop property");
    }

    [TestMethod]
    public async Task ConcurrentMaterialSnapshotsMergeEveryAuditEvent()
    {
        using var directory = new TemporaryDirectory();
        var service = new ProjectSnapshotService(directory.Path);
        ProjectDocument project = TestProjectFactory.Create();
        Task<DomainResult<ProjectSnapshotReceipt>>[] saves = Enumerable.Range(0, 20)
            .Select(index => service.SaveEventSnapshotAsync(
                project,
                SnapshotTrigger.PointEdited,
                TestProjectFactory.CreatedUtc.AddSeconds(index + 1),
                TestProjectFactory.PanelId,
                $"point-{index}"))
            .ToArray();

        DomainResult<ProjectSnapshotReceipt>[] results = await Task.WhenAll(saves);

        Assert.IsTrue(results.All(result => result.IsSuccess), FormatErrors(results.SelectMany(result => result.Errors).ToArray()));
        DomainResult<ProjectDocument> loaded = await new ProjectFileStore().LoadAsync(
            service.GetSnapshotPath(project.ProjectId));
        Assert.IsTrue(loaded.IsSuccess, FormatErrors(loaded.Errors));
        Assert.AreEqual(20, loaded.Value!.Audit.Events.Count);
        Assert.AreEqual(20, loaded.Value.Audit.Events.Select(auditEvent => auditEvent.EntityId).Distinct().Count());
    }

    [TestMethod]
    public async Task StaleTimerScheduleCannotSaveWithinFiveMinutesOfEventSnapshot()
    {
        using var directory = new TemporaryDirectory();
        var scheduler = new FiveMinuteAutosaveScheduler();
        var service = new ProjectSnapshotService(directory.Path, scheduler: scheduler);
        ProjectDocument staleProject = TestProjectFactory.Create();
        DateTimeOffset observed = TestProjectFactory.CreatedUtc;
        AutosaveSchedule staleSchedule = scheduler.CreateSchedule(staleProject, observed);
        DomainResult<ProjectSnapshotReceipt> eventSave = await service.SaveEventSnapshotAsync(
            staleProject,
            SnapshotTrigger.PointEdited,
            observed.AddMinutes(4));
        Assert.IsTrue(eventSave.IsSuccess, FormatErrors(eventSave.Errors));

        DomainResult<ProjectSnapshotReceipt> timerSave = await service.SaveTimerSnapshotAsync(
            staleProject,
            staleSchedule,
            observed.AddMinutes(5));

        Assert.IsFalse(timerSave.IsSuccess);
        Assert.AreEqual("AUTOSAVE_NOT_DUE", timerSave.Errors[0].Code);
        DomainResult<ProjectDocument> loaded = await new ProjectFileStore().LoadAsync(
            service.GetSnapshotPath(staleProject.ProjectId));
        Assert.AreEqual(1, loaded.Value!.Audit.Events.Count);
    }

    [TestMethod]
    public async Task OutOfOrderEventTimesKeepSnapshotTimestampsMonotonic()
    {
        using var directory = new TemporaryDirectory();
        var service = new ProjectSnapshotService(directory.Path);
        ProjectDocument project = TestProjectFactory.Create();
        DateTimeOffset later = TestProjectFactory.CreatedUtc.AddMinutes(10);
        DomainResult<ProjectSnapshotReceipt> first = await service.SaveEventSnapshotAsync(
            project,
            SnapshotTrigger.PointEdited,
            later,
            entityId: "later");
        Assert.IsTrue(first.IsSuccess, FormatErrors(first.Errors));

        DomainResult<ProjectSnapshotReceipt> second = await service.SaveEventSnapshotAsync(
            project,
            SnapshotTrigger.PhaseEdited,
            TestProjectFactory.CreatedUtc.AddMinutes(5),
            entityId: "earlier");

        Assert.IsTrue(second.IsSuccess, FormatErrors(second.Errors));
        Assert.AreEqual(later, second.Value!.Snapshot.ModifiedUtc);
        Assert.AreEqual(later, second.Value.Snapshot.Audit.LastAutosaveUtc);
        Assert.AreEqual(2, second.Value.Snapshot.Audit.Events.Count);
    }

    [TestMethod]
    public async Task SaveChecksCancellationBeforeSerialization()
    {
        using var directory = new TemporaryDirectory();
        ProjectDocument invalid = TestProjectFactory.Create() with { SchemaVersion = 99 };
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsExactlyAsync<OperationCanceledException>(() =>
            new ProjectFileStore().SaveAsync(
                invalid,
                Path.Combine(directory.Path, "cancelled.garproj"),
                cancellation.Token));
        Assert.AreEqual(0, Directory.GetFiles(directory.Path).Length);
    }

    [TestMethod]
    public async Task ExistingTargetIsReplacedWithCompleteNewProject()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "replace.garproj");
        var store = new ProjectFileStore();
        ProjectDocument original = TestProjectFactory.Create();
        ProjectDocument replacement = original with { ModifiedUtc = original.ModifiedUtc.AddMinutes(1) };
        Assert.IsTrue((await store.SaveAsync(original, path)).IsSuccess);

        DomainResult<ProjectSaveReceipt> overwritten = await store.SaveAsync(replacement, path);
        DomainResult<ProjectDocument> loaded = await store.LoadAsync(path);

        Assert.IsTrue(overwritten.IsSuccess, FormatErrors(overwritten.Errors));
        Assert.IsTrue(loaded.IsSuccess, FormatErrors(loaded.Errors));
        Assert.AreEqual(replacement.ModifiedUtc, loaded.Value!.ModifiedUtc);
        Assert.AreEqual(0, Directory.GetFiles(directory.Path, "*.tmp").Length);
    }

    [TestMethod]
    public async Task SnapshotAndLoadDeepFreezeCallerCollectionsAndJsonElements()
    {
        using var directory = new TemporaryDirectory();
        ProjectDocument baseline = TestProjectFactory.Create();
        var sources = baseline.Sources.ToList();
        var panels = baseline.Panels.ToList();
        using var metadataDocument = JsonDocument.Parse("{\"nested\":{\"value\":7}}");
        sources[0] = sources[0] with { ArticleMetadata = metadataDocument.RootElement };
        ProjectDocument callerOwned = baseline with { Sources = sources, Panels = panels };
        var service = new ProjectSnapshotService(directory.Path);

        DomainResult<ProjectSnapshotReceipt> saved = await service.SaveEventSnapshotAsync(
            callerOwned,
            SnapshotTrigger.ExportSettingsChanged,
            TestProjectFactory.CreatedUtc.AddMinutes(1));
        Assert.IsTrue(saved.IsSuccess, FormatErrors(saved.Errors));
        sources.Clear();
        panels.Clear();
        metadataDocument.Dispose();

        ProjectDocument snapshot = saved.Value!.Snapshot;
        Assert.AreEqual(1, snapshot.Sources.Count);
        Assert.AreEqual(1, snapshot.Panels.Count);
        Assert.AreEqual(7, snapshot.Sources[0].ArticleMetadata!.Value.GetProperty("nested").GetProperty("value").GetInt32());
        Assert.ThrowsExactly<NotSupportedException>(() => ((IList<SourceReference>)snapshot.Sources).Clear());

        DomainResult<ProjectDocument> loaded = await new ProjectFileStore().LoadAsync(saved.Value.SnapshotPath);
        Assert.IsTrue(loaded.IsSuccess, FormatErrors(loaded.Errors));
        Assert.ThrowsExactly<NotSupportedException>(() => ((IList<PanelRecord>)loaded.Value!.Panels).Clear());
    }

    [TestMethod]
    public void ValidatorRejectsAllDanglingDomainReferences()
    {
        ProjectDocument baseline = TestProjectFactory.Create();
        PanelRecord panel = baseline.Panels[0];
        var missingOcr = new OcrRegionId(Guid.Parse("cccccccc-cccc-cccc-cccc-cccccccccccc"));
        var missingSeries = new SeriesId(Guid.Parse("dddddddd-dddd-dddd-dddd-dddddddddddd"));
        var missingPhase = new PhaseId(Guid.Parse("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"));
        CalibrationRecord calibration = panel.Calibration!;
        ProjectDocument[] invalidProjects =
        {
            baseline with
            {
                Panels = new[]
                {
                    panel with
                    {
                        Calibration = calibration with
                        {
                            Anchors = calibration.Anchors
                                .Select((anchor, index) => index == 0 ? anchor with { EvidenceRegionId = missingOcr } : anchor)
                                .ToArray()
                        }
                    }
                }
            },
            baseline with
            {
                Panels = new[]
                {
                    panel with
                    {
                        Markers = panel.Markers.Select(marker => marker with { CandidateSeriesId = missingSeries }).ToArray()
                    }
                }
            },
            baseline with
            {
                Panels = new[]
                {
                    panel with
                    {
                        Series = panel.Series.Select(series => series with { SharedBaselineSeriesId = missingSeries }).ToArray()
                    }
                }
            },
            baseline with
            {
                Panels = new[]
                {
                    panel with
                    {
                        Series = panel.Series.Select(series => series with
                        {
                            ApplicableProbeSeriesIds = new[] { missingSeries }
                        }).ToArray()
                    }
                }
            },
            baseline with
            {
                Panels = new[]
                {
                    panel with
                    {
                        Phases = panel.Phases.Select(phase => phase with
                        {
                            BoundaryLeftId = missingPhase,
                            BoundaryRightId = missingPhase
                        }).ToArray()
                    }
                }
            }
        };

        foreach (ProjectDocument invalidProject in invalidProjects)
        {
            DomainResult<string> result = new ProjectJsonSerializer().Serialize(invalidProject);
            Assert.IsFalse(result.IsSuccess);
            Assert.IsTrue(result.Errors.Any(error => error.Code == "PROJECT_INVALID"));
        }
    }

    private static JsonObject CreateMinimalSchemaProject() =>
        (JsonObject)JsonNode.Parse(
            """
            {
              "schema_version": 1,
              "project_id": "11111111-1111-1111-1111-111111111111",
              "app_version": "0.0.2",
              "sources": [
                {
                  "source_id": "22222222-2222-2222-2222-222222222222",
                  "kind": "image",
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "display_name": "fixture.png"
                }
              ],
              "panels": [
                {
                  "panel_id": "33333333-3333-3333-3333-333333333333",
                  "source_id": "22222222-2222-2222-2222-222222222222",
                  "crop": { "x": 0, "y": 0, "width": 100, "height": 80 },
                  "transforms": [],
                  "series": [],
                  "points": [],
                  "phases": []
                }
              ],
              "audit": { "events": [] }
            }
            """)!;

    private static void RemovePath(JsonObject project, string path)
    {
        string[] segments = path.Split('.');
        JsonObject target = segments.Length == 1
            ? project
            : segments[0] switch
            {
                "source" => (JsonObject)((JsonArray)project["sources"]!)[0]!,
                "panel" => (JsonObject)((JsonArray)project["panels"]!)[0]!,
                "crop" => GetCrop(project),
                "audit" => (JsonObject)project["audit"]!,
                _ => project
            };
        target.Remove(segments[^1]);
    }

    private static JsonObject GetCrop(JsonObject project) =>
        (JsonObject)((JsonObject)((JsonArray)project["panels"]!)[0]!)["crop"]!;

    private static void AssertSchemaCorrupt(JsonObject project, string scenario)
    {
        DomainResult<ProjectDocument> result = new ProjectJsonSerializer().Deserialize(project.ToJsonString());
        Assert.IsFalse(result.IsSuccess, $"Accepted {scenario}.");
        Assert.IsTrue(result.Errors.Any(error => error.Code == "PROJECT_CORRUPT"));
    }

    private static string FormatErrors(IReadOnlyList<DomainError> errors) =>
        string.Join(Environment.NewLine, errors.Select(error => $"{error.Code}: {error.TechnicalMessage}"));
}
