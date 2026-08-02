// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json.Nodes;
using GraphReader.Domain;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.Domain.Tests;

[TestClass]
public sealed class ProjectPersistenceTests
{
    [TestMethod]
    public void ProjectRoundTripPreservesIdsAndNullableEvidence()
    {
        var serializer = new ProjectJsonSerializer();
        ProjectDocument expected = TestProjectFactory.Create();

        DomainResult<string> serialized = serializer.Serialize(expected);
        Assert.IsTrue(serialized.IsSuccess, FormatErrors(serialized.Errors));
        Assert.IsNotNull(serialized.Value);
        DomainResult<ProjectDocument> loaded = serializer.Deserialize(serialized.Value);

        Assert.IsTrue(loaded.IsSuccess, FormatErrors(loaded.Errors));
        Assert.IsNotNull(loaded.Value);
        ProjectDocument actual = loaded.Value;
        Assert.AreEqual(expected.ProjectId, actual.ProjectId);
        Assert.AreEqual(expected.Sources[0].SourceId, actual.Sources[0].SourceId);
        Assert.AreEqual(expected.Panels[0].PanelId, actual.Panels[0].PanelId);
        Assert.AreEqual(expected.Panels[0].Points[0].PointId, actual.Panels[0].Points[0].PointId);
        Assert.IsNull(actual.Panels[0].Participant);
        Assert.IsNull(actual.Panels[0].Points[0].PrintedXValue);
        Assert.IsNull(actual.Panels[0].Points[0].ModelVersion);
        Assert.IsNull(actual.Panels[0].Markers[0].Embedding);
        Assert.IsNull(actual.Panels[0].Series[0].LegendText);

        DomainResult<string> reserialized = serializer.Serialize(actual);
        Assert.IsTrue(reserialized.IsSuccess, FormatErrors(reserialized.Errors));
        Assert.AreEqual(serialized.Value, reserialized.Value);
    }

    [TestMethod]
    public void SerializationIsCanonicalAndDeterministic()
    {
        var serializer = new ProjectJsonSerializer();
        ProjectDocument project = TestProjectFactory.Create();

        string first = serializer.Serialize(project).Value!;
        string second = serializer.Serialize(project).Value!;

        Assert.AreEqual(first, second);
        Assert.IsTrue(first.EndsWith('\n'));
        Assert.IsFalse(first.Contains('\r'));
        Assert.IsTrue(
            first.IndexOf("\"author\"", StringComparison.Ordinal) <
            first.IndexOf("\"year\"", StringComparison.Ordinal),
            "Nested object keys must be canonicalized ordinally.");
        Assert.IsTrue(first.Contains("\"project_id\": \"11111111-1111-1111-1111-111111111111\"", StringComparison.Ordinal));
    }

    [TestMethod]
    public async Task InterruptedAtomicSavePreservesLastValidProject()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "project.garproj");
        ProjectDocument original = TestProjectFactory.Create();
        var normalStore = new ProjectFileStore();
        DomainResult<ProjectSaveReceipt> initialSave = await normalStore.SaveAsync(original, path);
        Assert.IsTrue(initialSave.IsSuccess, FormatErrors(initialSave.Errors));
        string originalBytes = await File.ReadAllTextAsync(path);

        ProjectDocument replacement = original with { ModifiedUtc = original.ModifiedUtc.AddMinutes(1) };
        var interruptedStore = new ProjectFileStore(
            interceptor: new InterruptBeforeCommitInterceptor());
        DomainResult<ProjectSaveReceipt> interrupted = await interruptedStore.SaveAsync(replacement, path);

        Assert.IsFalse(interrupted.IsSuccess);
        Assert.AreEqual("PROJECT_SAVE_FAILED", interrupted.Errors[0].Code);
        Assert.AreEqual(originalBytes, await File.ReadAllTextAsync(path));
        DomainResult<ProjectDocument> reloaded = await normalStore.LoadAsync(path);
        Assert.IsTrue(reloaded.IsSuccess, FormatErrors(reloaded.Errors));
        Assert.AreEqual(original.ModifiedUtc, reloaded.Value!.ModifiedUtc);
        Assert.AreEqual(0, Directory.GetFiles(directory.Path, "*.tmp").Length);
    }

    [TestMethod]
    public void CorruptAndFutureProjectsReturnStructuredErrors()
    {
        var serializer = new ProjectJsonSerializer();

        DomainResult<ProjectDocument> corrupt = serializer.Deserialize("{\"schema_version\":1,\"project_id\":");
        DomainResult<ProjectDocument> empty = serializer.Deserialize(string.Empty);
        DomainResult<ProjectDocument> missingFields = serializer.Deserialize("{\"schema_version\":1}");
        DomainResult<ProjectDocument> future = serializer.Deserialize("{\"schema_version\":2}");

        Assert.IsFalse(corrupt.IsSuccess);
        Assert.AreEqual("PROJECT_CORRUPT", corrupt.Errors[0].Code);
        Assert.IsFalse(empty.IsSuccess);
        Assert.AreEqual("PROJECT_CORRUPT", empty.Errors[0].Code);
        Assert.IsFalse(missingFields.IsSuccess);
        Assert.IsTrue(missingFields.Errors.All(error => error.Code is "PROJECT_CORRUPT" or "PROJECT_INVALID"));
        Assert.IsFalse(future.IsSuccess);
        Assert.AreEqual("PROJECT_VERSION_UNSUPPORTED", future.Errors[0].Code);
    }

    [TestMethod]
    public void MigrationDispatcherRunsRegisteredVersionStep()
    {
        var baselineSerializer = new ProjectJsonSerializer();
        string versionOne = baselineSerializer.Serialize(TestProjectFactory.Create()).Value!;
        string versionZero = versionOne.Replace(
            "\"schema_version\": 1",
            "\"schema_version\": 0",
            StringComparison.Ordinal);
        var migration = new VersionZeroToOneMigration();
        var serializer = new ProjectJsonSerializer(new ProjectMigrationDispatcher(new[] { migration }));

        DomainResult<ProjectDocument> loaded = serializer.Deserialize(versionZero);

        Assert.IsTrue(loaded.IsSuccess, FormatErrors(loaded.Errors));
        Assert.AreEqual(1, migration.CallCount);
        Assert.AreEqual(ProjectDocument.CurrentSchemaVersion, loaded.Value!.SchemaVersion);
    }

    private static string FormatErrors(IReadOnlyList<DomainError> errors) =>
        string.Join(Environment.NewLine, errors.Select(error => $"{error.Code}: {error.TechnicalMessage}"));

    private sealed class InterruptBeforeCommitInterceptor : IAtomicSaveInterceptor
    {
        public ValueTask OnStageAsync(
            AtomicSaveStage stage,
            string targetPath,
            string temporaryPath,
            CancellationToken cancellationToken)
        {
            if (stage == AtomicSaveStage.BeforeCommit)
            {
                throw new IOException("Injected interruption before atomic commit.");
            }

            return ValueTask.CompletedTask;
        }
    }

    private sealed class VersionZeroToOneMigration : IProjectMigration
    {
        public int CallCount { get; private set; }

        public int FromVersion => 0;

        public int ToVersion => 1;

        public DomainResult<JsonObject> Migrate(JsonObject document)
        {
            CallCount++;
            var migrated = (JsonObject)document.DeepClone();
            migrated["schema_version"] = 1;
            return DomainResult<JsonObject>.Success(migrated);
        }
    }
}
