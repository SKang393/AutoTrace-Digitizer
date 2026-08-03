// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.IO;

namespace GraphReader.App.Integration.Runtime;

/// <summary>
/// Initializes a mutable runtime directory and verifies that it accepts writes.
/// </summary>
public interface IRuntimeDirectoryAccess
{
    void EnsureWritable(string directoryPath);
}

/// <summary>
/// Uses a short-lived local file to verify directory write access.
/// </summary>
public sealed class FileRuntimeDirectoryAccess : IRuntimeDirectoryAccess
{
    public static FileRuntimeDirectoryAccess Instance { get; } = new();

    private FileRuntimeDirectoryAccess()
    {
    }

    public void EnsureWritable(string directoryPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directoryPath);

        Directory.CreateDirectory(directoryPath);

        string probePath = Path.Combine(
            directoryPath,
            $".graph-auto-reader-write-probe-{Guid.NewGuid():N}.tmp");

        using var probe = new FileStream(
            probePath,
            FileMode.CreateNew,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 1,
            FileOptions.DeleteOnClose);

        probe.WriteByte(0);
        probe.Flush(flushToDisk: true);
    }
}
