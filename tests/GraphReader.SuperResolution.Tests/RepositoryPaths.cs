// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.SuperResolution.Tests;

internal static class RepositoryPaths
{
    public static string Root { get; } = FindRoot();

    public static string FromRoot(params string[] segments) =>
        Path.GetFullPath(Path.Combine(new[] { Root }.Concat(segments).ToArray()));

    private static string FindRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "GraphAutoReader.slnx")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the Graph Auto Reader repository root.");
    }
}
