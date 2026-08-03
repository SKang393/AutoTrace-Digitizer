// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests.Validation.Core.Private;

internal sealed record PrivateEvaluationRequest(
    bool ExplicitOptIn,
    string? ExternalDirectory);

internal enum PrivateEvaluationUnavailableReason
{
    ExplicitOptInRequired,
    ContinuousIntegrationDetected,
    ExternalDirectoryRequired,
    ExternalDirectoryNotFound,
    ExternalDirectoryMustBeOutsideRepository,
    ExternalDirectoryCouldNotBeResolved,
}

internal sealed record PrivateEvaluationReason(
    PrivateEvaluationUnavailableReason Code,
    string Message);

internal sealed record PrivateEvaluationAvailability(
    bool IsAvailable,
    string? ExternalDirectory,
    IReadOnlyList<PrivateEvaluationReason> Reasons);

/// <summary>
/// Resolves a private evaluation directory without enumerating, reading, or copying its data.
/// </summary>
internal sealed class PrivateEvaluationAdapter
{
    private static readonly string[] ContinuousIntegrationVariables =
    [
        "CI",
        "TF_BUILD",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "BUILD_BUILDID",
        "JENKINS_URL",
        "TEAMCITY_VERSION",
    ];

    private static readonly string[] DisabledEnvironmentValues =
    [
        "0",
        "false",
        "no",
        "off",
    ];

    private readonly string repositoryRoot;
    private readonly Func<string, string?> readEnvironmentVariable;

    public PrivateEvaluationAdapter(
        string repositoryRoot,
        Func<string, string?>? readEnvironmentVariable = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repositoryRoot);

        this.repositoryRoot = ResolveExistingDirectory(repositoryRoot);
        this.readEnvironmentVariable =
            readEnvironmentVariable ?? Environment.GetEnvironmentVariable;
    }

    public PrivateEvaluationAvailability CheckAvailability(PrivateEvaluationRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        List<PrivateEvaluationReason> reasons = [];
        if (!request.ExplicitOptIn)
        {
            reasons.Add(new(
                PrivateEvaluationUnavailableReason.ExplicitOptInRequired,
                "Private evaluation requires an explicit opt-in for each invocation."));
        }

        if (ContinuousIntegrationVariables.Any(IsEnabledEnvironmentVariable))
        {
            reasons.Add(new(
                PrivateEvaluationUnavailableReason.ContinuousIntegrationDetected,
                "Private evaluation is disabled in continuous integration environments."));
        }

        string? externalDirectory = ResolveRequestedDirectory(request.ExternalDirectory, reasons);
        if (externalDirectory is not null && IsSameAsOrInside(externalDirectory, repositoryRoot))
        {
            externalDirectory = null;
            reasons.Add(new(
                PrivateEvaluationUnavailableReason.ExternalDirectoryMustBeOutsideRepository,
                "Private evaluation data must be stored outside the repository."));
        }

        return reasons.Count == 0
            ? new(true, externalDirectory, Array.Empty<PrivateEvaluationReason>())
            : new(false, null, reasons.AsReadOnly());
    }

    private string? ResolveRequestedDirectory(
        string? configuredDirectory,
        List<PrivateEvaluationReason> reasons)
    {
        if (string.IsNullOrWhiteSpace(configuredDirectory))
        {
            reasons.Add(new(
                PrivateEvaluationUnavailableReason.ExternalDirectoryRequired,
                "An external private evaluation directory is required."));
            return null;
        }

        string fullPath;
        try
        {
            fullPath = Path.GetFullPath(configuredDirectory);
        }
        catch (Exception exception) when (
            exception is ArgumentException or NotSupportedException or PathTooLongException)
        {
            reasons.Add(new(
                PrivateEvaluationUnavailableReason.ExternalDirectoryCouldNotBeResolved,
                "The configured private evaluation directory could not be resolved."));
            return null;
        }

        if (!Directory.Exists(fullPath))
        {
            reasons.Add(new(
                PrivateEvaluationUnavailableReason.ExternalDirectoryNotFound,
                "The configured private evaluation directory does not exist."));
            return null;
        }

        if (IsSameAsOrInside(fullPath, repositoryRoot))
        {
            reasons.Add(new(
                PrivateEvaluationUnavailableReason.ExternalDirectoryMustBeOutsideRepository,
                "Private evaluation data must be stored outside the repository."));
            return null;
        }

        try
        {
            return ResolveExistingDirectory(fullPath);
        }
        catch (Exception exception) when (
            exception is IOException or UnauthorizedAccessException or ArgumentException)
        {
            reasons.Add(new(
                PrivateEvaluationUnavailableReason.ExternalDirectoryCouldNotBeResolved,
                "The configured private evaluation directory could not be resolved safely."));
            return null;
        }
    }

    private bool IsEnabledEnvironmentVariable(string variableName)
    {
        string? value = readEnvironmentVariable(variableName);
        if (string.IsNullOrWhiteSpace(value))
        {
            return false;
        }

        return !DisabledEnvironmentValues.Contains(value.Trim(), StringComparer.OrdinalIgnoreCase);
    }

    private static string ResolveExistingDirectory(string path)
    {
        string fullPath = Path.TrimEndingDirectorySeparator(Path.GetFullPath(path));
        DirectoryInfo directory = new(fullPath);
        FileSystemInfo? finalTarget = directory.ResolveLinkTarget(returnFinalTarget: true);
        return Path.TrimEndingDirectorySeparator(finalTarget?.FullName ?? fullPath);
    }

    private static bool IsSameAsOrInside(string candidatePath, string parentPath)
    {
        string relative = Path.GetRelativePath(parentPath, candidatePath);
        return relative == "." ||
               (!Path.IsPathRooted(relative) &&
                !relative.Equals("..", StringComparison.Ordinal) &&
                !relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal) &&
                !relative.StartsWith($"..{Path.AltDirectorySeparatorChar}", StringComparison.Ordinal));
    }
}
