// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.App.Integration.Runtime;

/// <summary>
/// Supplies the process locations used to resolve application data paths.
/// </summary>
public interface IRuntimePathEnvironment
{
    string ExecutableDirectory { get; }

    string LocalApplicationDataRoot { get; }
}

/// <summary>
/// Resolves runtime locations through .NET process and known-folder APIs only.
/// </summary>
public sealed class SystemRuntimePathEnvironment : IRuntimePathEnvironment
{
    private SystemRuntimePathEnvironment()
    {
    }

    public static SystemRuntimePathEnvironment Instance { get; } = new();

    public string ExecutableDirectory => AppContext.BaseDirectory;

    public string LocalApplicationDataRoot =>
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
}
