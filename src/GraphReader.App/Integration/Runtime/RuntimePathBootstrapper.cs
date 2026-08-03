// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;
using System.Security;
using GraphReader.Domain;

namespace GraphReader.App.Integration.Runtime;

/// <summary>
/// Resolves installed or portable application paths and prepares mutable roots.
/// </summary>
public sealed class RuntimePathBootstrapper
{
    private const string InstalledFailureCode = "APPLICATION_DATA_NOT_WRITABLE";
    private const string InstalledMessageKey = "Errors.ApplicationDataNotWritable";
    private const string InstalledSuggestedAction = "check_local_app_data_permissions";
    private const string ResolutionFailureCode = "APPLICATION_PATHS_UNAVAILABLE";
    private const string ResolutionMessageKey = "Errors.ApplicationPathsUnavailable";
    private const string ResolutionSuggestedAction = "verify_application_installation";
    private const string PortableFailureCode = "PORTABLE_DATA_NOT_WRITABLE";
    private const string PortableMessageKey = "Errors.PortableDataNotWritable";
    private const string PortableSuggestedAction = "move_portable_installation";

    private readonly IRuntimePathEnvironment _environment;
    private readonly IRuntimeDirectoryAccess _directoryAccess;

    public RuntimePathBootstrapper(
        IRuntimePathEnvironment environment,
        IRuntimeDirectoryAccess directoryAccess)
    {
        _environment = environment ?? throw new ArgumentNullException(nameof(environment));
        _directoryAccess = directoryAccess ?? throw new ArgumentNullException(nameof(directoryAccess));
    }

    public static RuntimePathBootstrapper CreateDefault() =>
        new(SystemRuntimePathEnvironment.Instance, FileRuntimeDirectoryAccess.Instance);

    public DomainResult<IApplicationPaths> Initialize()
    {
        ApplicationPaths? paths = null;
        string? currentDirectory = null;

        try
        {
            string executableDirectory = _environment.ExecutableDirectory;
            string localApplicationDataRoot = _environment.LocalApplicationDataRoot;
            ArgumentException.ThrowIfNullOrWhiteSpace(executableDirectory);
            ArgumentException.ThrowIfNullOrWhiteSpace(localApplicationDataRoot);

            paths = ApplicationPaths.Create(
                executableDirectory,
                localApplicationDataRoot);

            foreach (string directory in GetMutableDirectories(paths))
            {
                currentDirectory = directory;
                _directoryAccess.EnsureWritable(directory);
            }

            return DomainResult<IApplicationPaths>.Success(paths);
        }
        catch (UnauthorizedAccessException exception)
        {
            return CreateFailure(paths?.Mode, currentDirectory, exception);
        }
        catch (IOException exception)
        {
            return CreateFailure(paths?.Mode, currentDirectory, exception);
        }
        catch (SecurityException exception)
        {
            return CreateFailure(paths?.Mode, currentDirectory, exception);
        }
        catch (ArgumentException exception)
        {
            return CreateFailure(paths?.Mode, currentDirectory, exception);
        }
        catch (NotSupportedException exception)
        {
            return CreateFailure(paths?.Mode, currentDirectory, exception);
        }
    }

    private static IEnumerable<string> GetMutableDirectories(ApplicationPaths paths)
    {
        yield return paths.SettingsRoot;
        yield return paths.CacheRoot;
        yield return paths.LogsRoot;
        yield return paths.AutosaveRoot;
        yield return paths.RecoveryRoot;
    }

    private static DomainResult<IApplicationPaths> CreateFailure(
        DistributionMode? mode,
        string? directory,
        Exception exception)
    {
        if (mode is null)
        {
            var resolutionError = new DomainError(
                ResolutionFailureCode,
                DomainErrorSeverity.Error,
                ResolutionMessageKey,
                $"Runtime path resolution failed: {exception.Message}",
                Recoverable: true,
                ResolutionSuggestedAction);

            return DomainResult<IApplicationPaths>.Failure(resolutionError);
        }

        bool isPortable = mode == DistributionMode.Portable;
        string target = string.IsNullOrWhiteSpace(directory)
            ? "the application data location"
            : $"'{directory}'";

        var error = new DomainError(
            isPortable ? PortableFailureCode : InstalledFailureCode,
            DomainErrorSeverity.Error,
            isPortable ? PortableMessageKey : InstalledMessageKey,
            $"Runtime path initialization failed for {target}: {exception.Message}",
            Recoverable: true,
            isPortable ? PortableSuggestedAction : InstalledSuggestedAction);

        return DomainResult<IApplicationPaths>.Failure(error);
    }
}
