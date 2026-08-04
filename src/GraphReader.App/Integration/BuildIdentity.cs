// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Reflection;

namespace GraphReader.App.Integration;

public sealed record BuildIdentity(string Version, string ShortCommit)
{
    public static BuildIdentity Current()
    {
        Assembly assembly = typeof(BuildIdentity).Assembly;
        Version version = assembly.GetName().Version ?? new Version(0, 0, 0);
        string displayVersion = $"{version.Major}.{version.Minor}.{Math.Max(version.Build, 0)}";
        string? informational = assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?
            .InformationalVersion;
        string commit = informational?.Split('+', 2).ElementAtOrDefault(1) ?? "unknown";
        return new BuildIdentity(displayVersion, commit.Length > 8 ? commit[..8] : commit);
    }
}
