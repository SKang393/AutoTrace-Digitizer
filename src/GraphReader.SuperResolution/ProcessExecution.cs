// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using System.Text;

namespace GraphReader.SuperResolution;

public enum ProcessCompletion
{
    Completed,
    Cancelled,
    TimedOut,
    StartFailed
}

public sealed record ProcessInvocation(
    string FileName,
    string WorkingDirectory,
    IReadOnlyList<string> Arguments,
    TimeSpan Timeout,
    int MaxDiagnosticCharacters);

public sealed record ProcessExecutionResult(
    ProcessCompletion Completion,
    int? ExitCode,
    string StandardOutput,
    string StandardError,
    TimeSpan Duration,
    string? StartError = null,
    string? TerminationError = null);

public interface IProcessRunner
{
    Task<ProcessExecutionResult> RunAsync(ProcessInvocation invocation, CancellationToken cancellationToken);
}

public sealed class LocalProcessRunner : IProcessRunner
{
    public async Task<ProcessExecutionResult> RunAsync(
        ProcessInvocation invocation,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(invocation);
        cancellationToken.ThrowIfCancellationRequested();

        var startInfo = new ProcessStartInfo
        {
            FileName = invocation.FileName,
            WorkingDirectory = invocation.WorkingDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        foreach (string argument in invocation.Arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using var process = new Process { StartInfo = startInfo };
        var stopwatch = Stopwatch.StartNew();
        try
        {
            if (!process.Start())
            {
                return new ProcessExecutionResult(
                    ProcessCompletion.StartFailed,
                    null,
                    string.Empty,
                    string.Empty,
                    stopwatch.Elapsed,
                    "The configured process did not start.");
            }
        }
        catch (Exception exception) when (exception is InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            return new ProcessExecutionResult(
                ProcessCompletion.StartFailed,
                null,
                string.Empty,
                string.Empty,
                stopwatch.Elapsed,
                exception.Message);
        }

        using var stopDiagnostics = new CancellationTokenSource();
        Task<string> stdoutTask = ReadBoundedAsync(
            process.StandardOutput,
            invocation.MaxDiagnosticCharacters,
            stopDiagnostics.Token);
        Task<string> stderrTask = ReadBoundedAsync(
            process.StandardError,
            invocation.MaxDiagnosticCharacters,
            stopDiagnostics.Token);

        using var timeout = new CancellationTokenSource(invocation.Timeout);
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken,
            timeout.Token);

        ProcessCompletion completion = ProcessCompletion.Completed;
        string? terminationError = null;
        try
        {
            await process.WaitForExitAsync(linked.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            completion = cancellationToken.IsCancellationRequested
                ? ProcessCompletion.Cancelled
                : ProcessCompletion.TimedOut;
            terminationError = TryKillProcessTree(process);
            if (!process.HasExited)
            {
                using var reapTimeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                try
                {
                    await process.WaitForExitAsync(reapTimeout.Token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    terminationError ??= "The cancelled process could not be reaped within five seconds.";
                    stopDiagnostics.Cancel();
                }
            }
        }

        string stdout = await stdoutTask.ConfigureAwait(false);
        string stderr = await stderrTask.ConfigureAwait(false);
        stopwatch.Stop();
        return new ProcessExecutionResult(
            completion,
            process.HasExited ? process.ExitCode : null,
            stdout,
            stderr,
            stopwatch.Elapsed,
            TerminationError: terminationError);
    }

    private static async Task<string> ReadBoundedAsync(
        StreamReader reader,
        int maximumCharacters,
        CancellationToken cancellationToken)
    {
        var output = new StringBuilder(Math.Min(maximumCharacters, 4096));
        var buffer = new char[4096];
        bool truncated = false;
        while (true)
        {
            int count;
            try
            {
                count = await reader.ReadAsync(buffer.AsMemory(), cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            if (count == 0)
            {
                break;
            }

            int remaining = maximumCharacters - output.Length;
            if (remaining > 0)
            {
                output.Append(buffer, 0, Math.Min(remaining, count));
            }

            truncated |= count > remaining;
        }

        if (truncated && maximumCharacters >= 16)
        {
            const string marker = "\n[truncated]";
            int retained = Math.Max(0, maximumCharacters - marker.Length);
            if (output.Length > retained)
            {
                output.Length = retained;
            }

            output.Append(marker);
        }

        return output.ToString();
    }

    private static string? TryKillProcessTree(Process process)
    {
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }

            return null;
        }
        catch (InvalidOperationException)
        {
            // A process exiting during cancellation is already in the desired state.
            return null;
        }
        catch (Exception exception) when (exception is System.ComponentModel.Win32Exception or NotSupportedException)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill();
                }

                return $"Process-tree termination was unavailable; direct termination was used: {exception.Message}";
            }
            catch (Exception fallbackException) when (
                fallbackException is InvalidOperationException or System.ComponentModel.Win32Exception or NotSupportedException)
            {
                return $"Process termination failed: {fallbackException.Message}";
            }
        }
    }
}
