// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace GraphReader.SuperResolution.Tests;

[TestClass]
public sealed class LocalProcessRunnerTests
{
    private const int DiagnosticLimit = 512;

    [TestMethod]
    public async Task ArgumentListPreservesSpacesUnicodeAndMetacharactersWithoutShellParsing()
    {
        using var probe = new PowerShellProcessProbe();
        string[] expected =
        [
            "path with spaces\\graph.png",
            "한글 participant",
            "& whoami ^ (literal); [series]",
            "quote \" remains data"
        ];
        ProcessInvocation invocation = probe.CreateInvocation(
            ["echo", .. expected],
            TimeSpan.FromSeconds(10),
            4096);

        ProcessExecutionResult result = await new LocalProcessRunner().RunAsync(
            invocation,
            CancellationToken.None);

        Assert.AreEqual(ProcessCompletion.Completed, result.Completion);
        Assert.AreEqual(0, result.ExitCode);
        string[] actual = result.StandardOutput
            .Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries)
            .Select(static encoded => Encoding.UTF8.GetString(Convert.FromBase64String(encoded)))
            .ToArray();
        CollectionAssert.AreEqual(expected, actual);
    }

    [TestMethod]
    public async Task RedirectedOutputIsBoundedIndependentlyForStdoutAndStderr()
    {
        using var probe = new PowerShellProcessProbe();
        ProcessInvocation invocation = probe.CreateInvocation(
            ["emit"],
            TimeSpan.FromSeconds(10),
            DiagnosticLimit);

        ProcessExecutionResult result = await new LocalProcessRunner().RunAsync(
            invocation,
            CancellationToken.None);

        Assert.AreEqual(ProcessCompletion.Completed, result.Completion);
        Assert.AreEqual(DiagnosticLimit, result.StandardOutput.Length);
        Assert.AreEqual(DiagnosticLimit, result.StandardError.Length);
        StringAssert.EndsWith(result.StandardOutput, "[truncated]");
        StringAssert.EndsWith(result.StandardError, "[truncated]");
    }

    [TestMethod]
    public async Task TimeoutAndCallerCancellationRemainDistinct()
    {
        using var probe = new PowerShellProcessProbe();
        ProcessInvocation timeoutInvocation = probe.CreateInvocation(
            ["sleep"],
            TimeSpan.FromMilliseconds(200),
            4096);
        ProcessExecutionResult timedOut = await new LocalProcessRunner().RunAsync(
            timeoutInvocation,
            CancellationToken.None);
        Assert.AreEqual(ProcessCompletion.TimedOut, timedOut.Completion);
        Assert.IsTrue(timedOut.Duration < TimeSpan.FromSeconds(5));
        Assert.IsNotNull(timedOut.ExitCode, "The timed-out helper must be reaped before the runner returns.");
        AssertReportedProcessExited(timedOut.StandardOutput);

        ProcessInvocation cancelInvocation = probe.CreateInvocation(
            ["sleep"],
            TimeSpan.FromSeconds(10),
            4096);
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        ProcessExecutionResult cancelled = await new LocalProcessRunner().RunAsync(
            cancelInvocation,
            cancellation.Token);
        Assert.AreEqual(ProcessCompletion.Cancelled, cancelled.Completion);
        Assert.IsTrue(cancelled.Duration < TimeSpan.FromSeconds(5));
        Assert.IsNotNull(cancelled.ExitCode, "The cancelled helper must be reaped before the runner returns.");
        AssertReportedProcessExited(cancelled.StandardOutput);
    }

    [TestMethod]
    public async Task MissingExecutableReturnsStartFailureInsteadOfThrowing()
    {
        string missing = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "missing.exe");
        var invocation = new ProcessInvocation(
            missing,
            Path.GetTempPath(),
            [],
            TimeSpan.FromSeconds(1),
            4096);

        ProcessExecutionResult result = await new LocalProcessRunner().RunAsync(
            invocation,
            CancellationToken.None);

        Assert.AreEqual(ProcessCompletion.StartFailed, result.Completion);
        Assert.IsNull(result.ExitCode);
        Assert.IsFalse(string.IsNullOrWhiteSpace(result.StartError));
    }

    private static void AssertReportedProcessExited(string standardOutput)
    {
        if (!int.TryParse(
                standardOutput.Trim(),
                System.Globalization.NumberStyles.None,
                System.Globalization.CultureInfo.InvariantCulture,
                out int processId))
        {
            return;
        }

        try
        {
            using System.Diagnostics.Process process = System.Diagnostics.Process.GetProcessById(processId);
            Assert.IsTrue(process.HasExited, $"Process {processId} remained alive after cancellation or timeout.");
        }
        catch (ArgumentException)
        {
            // The process ID is no longer registered, which proves the helper exited.
        }
    }
}

internal sealed class PowerShellProcessProbe : IDisposable
{
    private const string Script = """
        param(
            [Parameter(Mandatory = $true)][string]$Mode,
            [Parameter(ValueFromRemainingArguments = $true)][string[]]$Values
        )
        if ($Mode -eq 'echo') {
            foreach ($value in $Values) {
                [Console]::WriteLine([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($value)))
            }
            exit 0
        }
        if ($Mode -eq 'emit') {
            [Console]::Out.Write(('O' * 8192))
            [Console]::Error.Write(('E' * 8192))
            exit 0
        }
        if ($Mode -eq 'sleep') {
            [Console]::Out.WriteLine($PID)
            [Console]::Out.Flush()
            Start-Sleep -Seconds 30
            exit 0
        }
        exit 3
        """;

    public PowerShellProcessProbe()
    {
        Root = Path.Combine(
            Path.GetTempPath(),
            "GraphReader process probe",
            Guid.NewGuid().ToString("N"),
            "spaces & unicode 한글");
        Directory.CreateDirectory(Root);
        ScriptPath = Path.Combine(Root, "process probe.ps1");
        File.WriteAllText(ScriptPath, Script, new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
        ExecutablePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.System),
            "WindowsPowerShell",
            "v1.0",
            "powershell.exe");
        if (!File.Exists(ExecutablePath))
        {
            throw new FileNotFoundException("Windows PowerShell is required for process-runner tests.", ExecutablePath);
        }
    }

    public string Root { get; }
    public string ScriptPath { get; }
    public string ExecutablePath { get; }

    public ProcessInvocation CreateInvocation(
        IReadOnlyList<string> probeArguments,
        TimeSpan timeout,
        int maximumDiagnosticCharacters)
    {
        string[] arguments =
        [
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            ScriptPath,
            .. probeArguments
        ];
        return new ProcessInvocation(
            ExecutablePath,
            Root,
            arguments,
            timeout,
            maximumDiagnosticCharacters);
    }

    public void Dispose()
    {
        if (Directory.Exists(Root))
        {
            Directory.Delete(Root, recursive: true);
        }
    }
}
