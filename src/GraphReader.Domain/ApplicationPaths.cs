// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Domain;

public enum DistributionMode
{
    Installed,
    Portable
}

public interface IApplicationPaths
{
    DistributionMode Mode { get; }

    string SettingsRoot { get; }

    string CacheRoot { get; }

    string LogsRoot { get; }

    string AutosaveRoot { get; }

    string RecoveryRoot { get; }

    string ModelRoot { get; }
}

public sealed class ApplicationPaths : IApplicationPaths
{
    private const string ApplicationDataDirectoryName = "GraphAutoReader";
    private const string PortableDataDirectoryName = "Data";
    private const string PortableSentinelFileName = "portable.mode";

    private ApplicationPaths(
        DistributionMode mode,
        string mutableRoot,
        string modelRoot)
    {
        Mode = mode;
        SettingsRoot = Path.Combine(mutableRoot, "Settings");
        CacheRoot = Path.Combine(mutableRoot, "Cache");
        LogsRoot = Path.Combine(mutableRoot, "Logs");
        AutosaveRoot = Path.Combine(mutableRoot, "Autosave");
        RecoveryRoot = Path.Combine(mutableRoot, "Recovery");
        ModelRoot = modelRoot;
    }

    public DistributionMode Mode { get; }

    public string SettingsRoot { get; }

    public string CacheRoot { get; }

    public string LogsRoot { get; }

    public string AutosaveRoot { get; }

    public string RecoveryRoot { get; }

    public string ModelRoot { get; }

    public static ApplicationPaths Create(
        string executableDirectory,
        string? localApplicationDataRoot = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(executableDirectory);

        var executableRoot = Path.GetFullPath(executableDirectory);
        var modelRoot = Path.Combine(executableRoot, "models");
        var portableSentinel = Path.Combine(executableRoot, PortableSentinelFileName);

        if (File.Exists(portableSentinel))
        {
            var portableRoot = Path.Combine(executableRoot, PortableDataDirectoryName);
            return new ApplicationPaths(DistributionMode.Portable, portableRoot, modelRoot);
        }

        var installedDataRoot = string.IsNullOrWhiteSpace(localApplicationDataRoot)
            ? Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData)
            : Path.GetFullPath(localApplicationDataRoot);

        var installedRoot = Path.Combine(installedDataRoot, ApplicationDataDirectoryName);
        return new ApplicationPaths(DistributionMode.Installed, installedRoot, modelRoot);
    }
}
