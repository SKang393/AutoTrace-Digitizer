// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Domain;

public enum RecoveryRecommendation
{
    RestoreRecommended,
    KeepCurrent,
    Inspect
}

public sealed record RecoveryCandidate(
    string AutosavePath,
    ProjectId ProjectId,
    DateTimeOffset AutosaveModifiedUtc,
    DateTimeOffset FileWrittenUtc,
    string? OriginalProjectPath,
    DateTimeOffset? OriginalModifiedUtc,
    bool? IsNewerThanOriginal,
    bool ProjectIdentityVerified,
    RecoveryRecommendation Recommendation,
    string DisplayName);

public sealed record RejectedRecoveryCandidate(
    string AutosavePath,
    IReadOnlyList<DomainError> Errors);

public sealed record RecoveryDiscoveryReport(
    IReadOnlyList<RecoveryCandidate> Candidates,
    IReadOnlyList<RejectedRecoveryCandidate> RejectedCandidates,
    IReadOnlyList<DomainError> PrimaryErrors);

public sealed class ProjectRecoveryService
{
    private readonly ProjectFileStore _store;

    public ProjectRecoveryService(ProjectFileStore? store = null)
    {
        _store = store ?? new ProjectFileStore();
    }

    public async Task<DomainResult<RecoveryDiscoveryReport>> DiscoverAsync(
        string autosaveRoot,
        ProjectDocument original,
        string? originalProjectPath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(autosaveRoot);
        ArgumentNullException.ThrowIfNull(original);
        return await DiscoverCoreAsync(
            autosaveRoot,
            original.ProjectId,
            original.ModifiedUtc,
            originalProjectPath,
            Array.Empty<DomainError>(),
            cancellationToken).ConfigureAwait(false);
    }

    public async Task<DomainResult<RecoveryDiscoveryReport>> DiscoverForProjectPathAsync(
        string autosaveRoot,
        string originalProjectPath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(autosaveRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(originalProjectPath);
        string fullOriginalPath = Path.GetFullPath(originalProjectPath);
        DomainResult<ProjectDocument> primary = await _store.LoadAsync(
            fullOriginalPath,
            cancellationToken).ConfigureAwait(false);

        return primary.IsSuccess && primary.Value is not null
            ? await DiscoverCoreAsync(
                autosaveRoot,
                primary.Value.ProjectId,
                primary.Value.ModifiedUtc,
                fullOriginalPath,
                Array.Empty<DomainError>(),
                cancellationToken).ConfigureAwait(false)
            : await DiscoverCoreAsync(
                autosaveRoot,
                originalProjectId: null,
                originalModifiedUtc: null,
                fullOriginalPath,
                primary.Errors,
                cancellationToken).ConfigureAwait(false);
    }

    public async Task<DomainResult<ProjectSaveReceipt>> RecoverToNewFileAsync(
        string autosavePath,
        string recoveryDestinationPath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(autosavePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(recoveryDestinationPath);
        string sourcePath = Path.GetFullPath(autosavePath);
        string destinationPath = Path.GetFullPath(recoveryDestinationPath);

        if (string.Equals(sourcePath, destinationPath, StringComparison.OrdinalIgnoreCase))
        {
            return DomainResult<ProjectSaveReceipt>.Failure(DomainErrors.IoFailure(
                "RECOVERY_DESTINATION_INVALID",
                "Errors.RecoveryDestinationInvalid",
                "Recovery must write to a new path so the autosave remains unchanged.",
                "select_new_project_path"));
        }

        DomainResult<ProjectDocument> loaded = await _store.LoadAsync(sourcePath, cancellationToken).ConfigureAwait(false);
        if (!loaded.IsSuccess || loaded.Value is null)
        {
            return DomainResult<ProjectSaveReceipt>.Failure(loaded.Errors);
        }

        return await _store.SaveNewAsync(
            loaded.Value,
            destinationPath,
            cancellationToken).ConfigureAwait(false);
    }

    private async Task<DomainResult<RecoveryDiscoveryReport>> DiscoverCoreAsync(
        string autosaveRoot,
        ProjectId? originalProjectId,
        DateTimeOffset? originalModifiedUtc,
        string? originalProjectPath,
        IReadOnlyList<DomainError> primaryErrors,
        CancellationToken cancellationToken)
    {
        string fullAutosaveRoot = Path.GetFullPath(autosaveRoot);

        if (!Directory.Exists(fullAutosaveRoot))
        {
            return DomainResult<RecoveryDiscoveryReport>.Success(new RecoveryDiscoveryReport(
                Array.Empty<RecoveryCandidate>(),
                Array.Empty<RejectedRecoveryCandidate>(),
                primaryErrors.ToArray()));
        }

        string[] paths;
        try
        {
            paths = Directory
                .EnumerateFiles(fullAutosaveRoot, "*.autosave.garproj", SearchOption.TopDirectoryOnly)
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            return DomainResult<RecoveryDiscoveryReport>.Failure(DomainErrors.IoFailure(
                "RECOVERY_DISCOVERY_FAILED",
                "Errors.RecoveryDiscoveryFailed",
                $"Recovery directory '{fullAutosaveRoot}' could not be enumerated: {exception.Message}",
                "retry"));
        }

        var candidates = new List<RecoveryCandidate>();
        var rejected = new List<RejectedRecoveryCandidate>();
        foreach (string path in paths)
        {
            cancellationToken.ThrowIfCancellationRequested();
            DomainResult<ProjectDocument> loaded = await _store.LoadAsync(path, cancellationToken).ConfigureAwait(false);
            if (!loaded.IsSuccess || loaded.Value is null)
            {
                rejected.Add(new RejectedRecoveryCandidate(path, loaded.Errors));
                continue;
            }

            ProjectDocument autosave = loaded.Value;
            if (originalProjectId is { } expectedProjectId && autosave.ProjectId != expectedProjectId)
            {
                continue;
            }

            bool identityVerified = originalProjectId is not null;
            bool? newer = originalModifiedUtc is { } originalModified
                ? autosave.ModifiedUtc > originalModified
                : null;
            string displayName = autosave.Sources.Count > 0
                ? autosave.Sources[0].DisplayName
                : autosave.ProjectId.Value.ToString("D");
            candidates.Add(new RecoveryCandidate(
                path,
                autosave.ProjectId,
                autosave.ModifiedUtc,
                new DateTimeOffset(File.GetLastWriteTimeUtc(path)),
                string.IsNullOrWhiteSpace(originalProjectPath) ? null : Path.GetFullPath(originalProjectPath),
                originalModifiedUtc,
                newer,
                identityVerified,
                newer switch
                {
                    true => RecoveryRecommendation.RestoreRecommended,
                    false => RecoveryRecommendation.KeepCurrent,
                    null => RecoveryRecommendation.Inspect
                },
                displayName));
        }

        RecoveryCandidate[] orderedCandidates = candidates
            .OrderByDescending(candidate => candidate.AutosaveModifiedUtc)
            .ThenBy(candidate => candidate.AutosavePath, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        return DomainResult<RecoveryDiscoveryReport>.Success(new RecoveryDiscoveryReport(
            orderedCandidates,
            rejected.ToArray(),
            primaryErrors.ToArray()));
    }
}
