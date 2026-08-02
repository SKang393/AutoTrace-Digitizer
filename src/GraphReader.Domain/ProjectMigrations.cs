// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text.Json.Nodes;

namespace GraphReader.Domain;

public interface IProjectMigration
{
    int FromVersion { get; }

    int ToVersion { get; }

    DomainResult<JsonObject> Migrate(JsonObject document);
}

public sealed class ProjectMigrationDispatcher
{
    private const int MaximumMigrationSteps = 32;
    private readonly Dictionary<int, IProjectMigration> _migrations;

    public ProjectMigrationDispatcher(IEnumerable<IProjectMigration>? migrations = null)
    {
        IProjectMigration[] materialized = migrations?.ToArray() ?? Array.Empty<IProjectMigration>();
        if (materialized.Any(migration => migration.ToVersion != migration.FromVersion + 1))
        {
            throw new ArgumentException("Project migrations must advance exactly one schema version.", nameof(migrations));
        }

        try
        {
            _migrations = materialized.ToDictionary(migration => migration.FromVersion);
        }
        catch (ArgumentException exception)
        {
            throw new ArgumentException("Only one migration may be registered for each source version.", nameof(migrations), exception);
        }
    }

    public DomainResult<JsonObject> Dispatch(JsonObject source)
    {
        ArgumentNullException.ThrowIfNull(source);

        JsonObject working = (JsonObject)source.DeepClone();
        if (!TryReadSchemaVersion(working, out int version))
        {
            return DomainResult<JsonObject>.Failure(DomainErrors.CorruptProject(
                "The project does not contain an integer schema_version."));
        }

        if (version > ProjectDocument.CurrentSchemaVersion)
        {
            return DomainResult<JsonObject>.Failure(new DomainError(
                "PROJECT_VERSION_UNSUPPORTED",
                DomainErrorSeverity.Error,
                "Errors.ProjectVersionUnsupported",
                $"Schema version {version} is newer than supported version {ProjectDocument.CurrentSchemaVersion}.",
                Recoverable: false,
                "update_application"));
        }

        int steps = 0;
        while (version < ProjectDocument.CurrentSchemaVersion)
        {
            if (steps++ >= MaximumMigrationSteps)
            {
                return DomainResult<JsonObject>.Failure(DomainErrors.CorruptProject(
                    "The migration chain exceeded the safe step limit."));
            }

            if (!_migrations.TryGetValue(version, out IProjectMigration? migration))
            {
                return DomainResult<JsonObject>.Failure(new DomainError(
                    "PROJECT_MIGRATION_MISSING",
                    DomainErrorSeverity.Error,
                    "Errors.ProjectMigrationMissing",
                    $"No migration is registered from schema version {version}.",
                    Recoverable: false,
                    "update_application"));
            }

            DomainResult<JsonObject> result;
            try
            {
                result = migration.Migrate(working);
            }
            catch (Exception exception) when (exception is not OutOfMemoryException)
            {
                return DomainResult<JsonObject>.Failure(DomainErrors.CorruptProject(
                    $"Migration from schema version {version} failed: {exception.Message}"));
            }

            if (!result.IsSuccess || result.Value is null)
            {
                return DomainResult<JsonObject>.Failure(result.Errors);
            }

            working = (JsonObject)result.Value.DeepClone();
            if (!TryReadSchemaVersion(working, out int migratedVersion) || migratedVersion != migration.ToVersion)
            {
                return DomainResult<JsonObject>.Failure(DomainErrors.CorruptProject(
                    $"Migration from schema version {version} did not produce schema version {migration.ToVersion}."));
            }

            version = migratedVersion;
        }

        return DomainResult<JsonObject>.Success(working);
    }

    private static bool TryReadSchemaVersion(JsonObject document, out int version)
    {
        version = default;
        return document["schema_version"] is JsonValue value && value.TryGetValue(out version);
    }
}
