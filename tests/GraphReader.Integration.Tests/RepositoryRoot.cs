// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

namespace GraphReader.Integration.Tests;

internal static class RepositoryRoot
{
    private const string OverrideEnvironmentVariable = "GRAPHREADER_REPOSITORY_ROOT";

    public static string Find()
    {
        string? configuredRoot = Environment.GetEnvironmentVariable(OverrideEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(configuredRoot))
        {
            string fullConfiguredRoot = Path.GetFullPath(configuredRoot);
            if (IsRepositoryRoot(fullConfiguredRoot))
            {
                return fullConfiguredRoot;
            }

            throw new DirectoryNotFoundException(
                $"{OverrideEnvironmentVariable} does not identify the Graph Auto Reader repository: " +
                fullConfiguredRoot);
        }

        foreach (string startPath in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            DirectoryInfo? directory = new(Path.GetFullPath(startPath));
            while (directory is not null)
            {
                if (IsRepositoryRoot(directory.FullName))
                {
                    return directory.FullName;
                }

                directory = directory.Parent;
            }
        }

        throw new DirectoryNotFoundException(
            $"Could not locate the Graph Auto Reader repository from {AppContext.BaseDirectory}.");
    }

    private static bool IsRepositoryRoot(string path) =>
        File.Exists(Path.Combine(path, "GraphAutoReader.slnx")) &&
        File.Exists(Path.Combine(path, "contracts", "project.schema.json")) &&
        Directory.Exists(Path.Combine(path, "tests"));
}
